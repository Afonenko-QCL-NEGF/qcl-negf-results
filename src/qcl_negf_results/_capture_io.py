"""Small nofollow/same-open primitives for detached closed-byte observers."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

from qcl_negf_contracts.artifacts import relative_path
from qcl_negf_contracts.messages import ContractError

from .commits import decode_json


def fail(code, message):
    raise ContractError(message, code)


def integer(value, minimum=0, maximum=2**63 - 1):
    return type(value) is int and minimum <= value <= maximum


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class ObserverIO:
    """One observer's delivered reads/planned writes, not a physical disk quota."""

    def __init__(self, maximum):
        self.maximum = maximum
        self.read_bytes = self.write_bytes = self.forecast_bytes = self.reserved = 0

    @property
    def used(self):
        return self.read_bytes + self.write_bytes + self.forecast_bytes

    @property
    def remaining(self):
        return self.maximum - self.used - self.reserved

    def require(self, count):
        if count < 0 or count > self.remaining:
            raise ValueError("observer admitted aggregate I/O credit exhausted")

    def account(self, count):
        self.require(count)
        self.read_bytes += count

    def write(self, count):
        self.require(count)
        self.write_bytes += count  # Planned payload, charged before any write.

    def reserve(self, count):
        self.require(count)
        self.reserved += count

    def settle(self, reserved, forecast):
        self.reserved -= reserved
        self.require(forecast)
        self.forecast_bytes += forecast


def stamp(stream, *, include_change=False):
    value = os.fstat(stream.fileno())
    result = value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    # An unlink changes ctime while the already opened copied bytes stay valid.
    # A retained native reader additionally guards changes between operations.
    return result + (value.st_ctime_ns,) if include_change else result


@contextmanager
def regular(path, *, size=None):
    """Open every absolute ancestor relative to an owned nofollow directory FD."""
    path = Path(os.path.abspath(path))
    owned = []
    local_failure = False
    try:
        try:
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            owned.append(directory)
            for part in path.parts[1:-1]:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory,
                )
                owned.append(child)
                owned.remove(directory)
                os.close(directory)
                directory = child
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=directory,
            )
            owned.append(descriptor)
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode) or (
                size is not None and value.st_size != size
            ):
                fail(
                    "corrupt_result",
                    "closed source is not a regular file of its declared size",
                )
            manager = os.fdopen(descriptor, "rb", buffering=0)
            owned.remove(descriptor)  # fdopen owns it after successful transfer.
        except OSError as error:
            raise ContractError(
                "selected closed source cannot be opened safely", "corrupt_result"
            ) from error

        try:
            stream = manager.__enter__()
        except BaseException as error:
            try:
                manager.close()
            except BaseException:
                pass
            if isinstance(error, OSError):
                raise ContractError(
                    "selected closed source cannot be opened safely", "corrupt_result"
                ) from error
            raise

        try:
            yield stream
        except BaseException as error:
            try:
                manager.__exit__(*sys.exc_info())
            except BaseException:
                pass
            if isinstance(error, OSError):
                raise ContractError(
                    "selected closed source cannot be opened safely", "corrupt_result"
                ) from error
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except OSError as error:
                raise ContractError(
                    "selected closed source cannot be opened safely", "corrupt_result"
                ) from error
    except BaseException:
        local_failure = True
        raise
    finally:
        cleanup_error = None
        for fd in reversed(owned):
            try:
                os.close(fd)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if not local_failure and cleanup_error is not None:
            if isinstance(cleanup_error, OSError):
                raise ContractError(
                    "selected closed source cannot be opened safely", "corrupt_result"
                ) from cleanup_error
            raise cleanup_error


def small_json(path, maximum, *, budget=None):
    with regular(path) as stream:
        before = stamp(stream)
        if before[2] > maximum:
            fail("budget_exceeded", "small source metadata exceeds its bound")
        if budget:
            budget.require(before[2])
        raw = stream.read(before[2])
        if budget:
            budget.account(len(raw))
        if len(raw) != before[2] or stamp(stream) != before:
            fail("corrupt_result", "opened metadata changed during bounded capture")
    return decode_json(raw, "selected metadata"), raw


def beneath(root, name):
    relative_path(name)
    return Path(root).joinpath(*name.split("/"))


def copy_closed(source, destination, *, size, sha256, budget=None, before_copy=None):
    """Hash the actual copied inode. Unlink after open is safe; rewrite is refused."""
    if not integer(size) or destination.exists():
        fail("corrupt_result", "invalid selected copy declaration")
    with regular(source, size=size) as stream:
        before = stamp(stream)
        if budget:
            budget.require(size)
        if before_copy:
            before_copy(stream)
        with destination.open("xb") as output:
            found = hashlib.sha256()
            remaining = size
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    fail("corrupt_result", "selected source was truncated")
                if budget:
                    budget.account(len(block))
                found.update(block)
                output.write(block)
                remaining -= len(block)
            if stamp(stream) != before or found.hexdigest() != sha256:
                fail(
                    "corrupt_result",
                    "selected source copy differs from pinned closed bytes",
                )
            output.flush()
            os.fsync(output.fileno())
