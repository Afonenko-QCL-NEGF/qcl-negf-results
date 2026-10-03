"""Finite logical-byte accounting for one private export spool.

This is a write guard, not a filesystem quota. Filesystem metadata, allocation
rounding and concurrent writers require an OS quota or dedicated filesystem.
Free-space checks are observations, not reservations against other processes.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
from typing import Any

from qcl_negf_contracts.messages import ContractError

DEFAULT_BYTE_BUDGET = 64 * 1024**3
DEFAULT_RESERVE_BYTES = 64 * 1024**2


def validate_options(byte_budget: int, reserve_bytes: int) -> None:
    if type(byte_budget) is not int or byte_budget <= 0:
        raise ValueError("intermediate export byte_budget must be a positive integer")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("export reserve_bytes must be a nonnegative integer")


class ExportBudget:
    """Count every owned file, including superseded captures and derivations."""

    def __init__(self, directory: Path, *, byte_budget: int = DEFAULT_BYTE_BUDGET,
                 reserve_bytes: int = DEFAULT_RESERVE_BYTES):
        validate_options(byte_budget, reserve_bytes)
        self.directory = Path(directory).resolve()
        self.byte_budget, self.reserve_bytes = byte_budget, reserve_bytes
        self.sizes: dict[Path, int] = {}
        self.used = 0

    def check(self, additional: int = 0) -> int:
        if self.used + additional > self.byte_budget:
            raise ContractError(f"intermediate export budget exhausted: {self.used} + {additional} "
                                f"> {self.byte_budget} bytes", "budget_exceeded")
        free = shutil.disk_usage(self.directory).free
        if free < additional + self.reserve_bytes:
            raise ContractError(f"insufficient export space: {free} free bytes, "
                                f"{additional} requested plus {self.reserve_bytes} reserve", "storage_failure")
        return free

    def _record(self, path: Path, size: int) -> None:
        self.used += size - self.sizes.get(path, 0)
        self.sizes[path] = size

    def check_actual(self) -> int:
        """Reconcile the complete spool before publication, including any strays."""
        self.sizes = {path.resolve(): path.stat().st_size
                      for path in self.directory.rglob("*") if path.is_file()}
        self.used = sum(self.sizes.values())
        return self.check()

    def open(self, path: Path, mode: str = "wb", *, defer_errors: bool = False) -> Any:
        path = Path(path).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("export write is outside its owned spool")
        if path.exists():
            self._record(path, path.stat().st_size)
        self.check()
        stream = path.open(mode, buffering=0)
        self._record(path, path.stat().st_size)
        return _BudgetFile(stream, path, self, defer_errors=defer_errors)

    def replace(self, source: Path, destination: Path) -> None:
        os.replace(source, destination)
        self.used -= self.sizes.pop(Path(source).resolve(), 0)
        self._record(Path(destination).resolve(), destination.stat().st_size)
        self.check()


class _BudgetFile(io.RawIOBase):
    """Use one guard for streaming XZ, Parquet and HDF5 file-object writes."""

    def __init__(self, stream: Any, path: Path, budget: ExportBudget, *, defer_errors: bool = False):
        self.stream, self.path, self.budget = stream, path, budget
        self.defer_errors, self.failure = defer_errors, None

    @property
    def name(self) -> str:
        return str(self.path)

    def write(self, value: Any) -> int:
        value = memoryview(value).cast("B")
        old = self.budget.sizes[self.path]
        position = self.stream.tell()
        if self.failure is None:
            try:
                self.budget.check(max(old, position + len(value)) - old)
                count = 0
                while count < len(value):
                    written = self.stream.write(value[count:])
                    if not written:
                        raise OSError("export file write made no progress")
                    count += written
                    self.budget._record(self.path, max(old, position + count))
            except (ContractError, OSError) as error:
                if not self.defer_errors:
                    raise
                self.failure = error
        if self.failure is not None:
            # HDF5 callbacks must not leave a pending Python exception in C.
            # Discard all remaining output; report refusal after HDF5 closes.
            self.stream.seek(position + len(value))
            return len(value)
        return count

    def truncate(self, size: int | None = None) -> int:
        size = self.tell() if size is None else size
        old = self.budget.sizes[self.path]
        if self.failure is None:
            try:
                self.budget.check(max(0, size - old))
                result = self.stream.truncate(size)
            except (ContractError, OSError) as error:
                if not self.defer_errors:
                    raise
                self.failure = error
        if self.failure is not None:
            return size
        self.budget._record(self.path, size)
        return result

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)

    def readinto(self, buffer: Any) -> int:
        return self.stream.readinto(buffer)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.stream.seek(offset, whence)

    def tell(self) -> int:
        return self.stream.tell()

    def flush(self) -> None:
        if not self.closed:
            self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()

    def readable(self) -> bool:
        return self.stream.readable()

    def writable(self) -> bool:
        return self.stream.writable()

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        if not self.closed:
            super().close()
            self.stream.close()

    def __exit__(self, *args: Any) -> None:
        self.close()
        if self.failure is not None:
            raise self.failure
