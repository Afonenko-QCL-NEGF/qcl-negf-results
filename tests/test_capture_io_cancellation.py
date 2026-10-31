import os

import pytest

from qcl_negf_results import _capture_io as capture


class Cancelled(BaseException):
    pass


def test_primary_read_survives_stream_and_directory_close_errors(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    primary = Cancelled('cancel')
    fdopen, close = os.fdopen, os.close
    closed = []

    def opening(fd, *args, **kwargs):
        raw = fdopen(fd, *args, **kwargs)

        class Proxy:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                raw.close()
                raise OSError('stream close')

            def read(self, *args):
                raise primary

        return Proxy()

    def closing(fd):
        close(fd)
        closed.append(fd)
        if len(closed) > len(path.parts) - 2:
            raise OSError('directory close')

    monkeypatch.setattr(os, 'fdopen', opening)
    monkeypatch.setattr(os, 'close', closing)
    with pytest.raises(Cancelled) as caught:
        with capture.regular(path) as stream:
            stream.read()
    assert caught.value is primary


def test_body_oserror_is_contract_error_with_original_cause_after_close(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    primary = OSError('read')
    fdopen = os.fdopen

    def opening(fd, *args, **kwargs):
        raw = fdopen(fd, *args, **kwargs)

        class Proxy:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                raw.close()
                raise OSError('close')

        return Proxy()

    monkeypatch.setattr(os, 'fdopen', opening)
    with pytest.raises(capture.ContractError) as caught:
        with capture.regular(path):
            raise primary
    assert caught.value.__cause__ is primary
    assert caught.value.code == "corrupt_result"


def test_successful_body_reports_stream_close_error(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    fdopen = os.fdopen

    def opening(fd, *args, **kwargs):
        raw = fdopen(fd, *args, **kwargs)

        class Proxy:
            def __enter__(self):
                return raw

            def __exit__(self, *args):
                raw.close()
                raise OSError('close')

        return Proxy()

    monkeypatch.setattr(os, 'fdopen', opening)
    with pytest.raises(capture.ContractError) as caught:
        with capture.regular(path) as stream:
            assert stream.read() == b'data'
    assert isinstance(caught.value.__cause__, OSError)


def test_entry_failure_closes_transferred_descriptor(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    primary = Cancelled('enter')
    fdopen = os.fdopen
    descriptors = []

    def opening(fd, *args, **kwargs):
        raw = fdopen(fd, *args, **kwargs)
        descriptors.append(fd)

        class Proxy:
            def __enter__(self):
                raise primary

            def close(self):
                raw.close()
                raise OSError('close')

        return Proxy()

    monkeypatch.setattr(os, 'fdopen', opening)
    with pytest.raises(Cancelled) as caught:
        with capture.regular(path):
            pytest.fail('entry should fail')
    assert caught.value is primary
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_ancestor_close_failure_closes_new_child(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    original_open, original_close = os.open, os.close
    opened = []
    fail_once = True

    def opening(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def closing(fd):
        nonlocal fail_once
        original_close(fd)
        if fail_once:
            fail_once = False
            raise OSError('ancestor close')

    monkeypatch.setattr(os, 'open', opening)
    monkeypatch.setattr(os, 'close', closing)
    with pytest.raises(capture.ContractError):
        with capture.regular(path):
            pytest.fail('opening should fail')
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_cleanup_attempts_every_owned_fd_after_open_failure(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    original_close = os.close
    original_fstat = os.fstat
    closed = []

    def fail_fstat(fd):
        raise Cancelled('fstat')

    def closing(fd):
        original_close(fd)
        closed.append(fd)
        raise OSError('close')

    monkeypatch.setattr(os, 'fstat', fail_fstat)
    monkeypatch.setattr(os, 'close', closing)
    # Only enable failing closes once the file is open.
    monkeypatch.setattr(os, 'close', original_close)
    original_open = os.open
    def opening(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        if args[0] == path.name:
            monkeypatch.setattr(os, 'close', closing)
        return fd
    monkeypatch.setattr(os, 'open', opening)
    with pytest.raises(Cancelled):
        with capture.regular(path):
            pass
    assert len(closed) == 2
    for fd in closed:
        with pytest.raises(OSError):
            original_fstat(fd)


def test_repeated_regular_reads_and_rejections_do_not_leak(tmp_path):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    link = tmp_path / 'link'
    link.symlink_to(path)
    before = len(os.listdir('/proc/self/fd'))
    for _ in range(50):
        with capture.regular(path, size=4) as stream:
            assert stream.read() == b'data'
        with pytest.raises(capture.ContractError):
            with capture.regular(path, size=5):
                pass
        with pytest.raises(capture.ContractError):
            with capture.regular(link):
                pass
    assert len(os.listdir('/proc/self/fd')) == before


def test_fdopen_failure_closes_descriptor(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    transferred = []
    def opening(fd, *args, **kwargs):
        transferred.append(fd)
        raise OSError('fdopen')
    monkeypatch.setattr(os, 'fdopen', opening)
    with pytest.raises(capture.ContractError):
        with capture.regular(path):
            pass
    with pytest.raises(OSError):
        os.fstat(transferred[0])


def test_successful_body_reports_directory_close_error(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    original_close = os.close
    # Fail only during final directory cleanup, after the body ran.
    body_done = False
    def closing_after_body(fd):
        original_close(fd)
        if body_done:
            raise OSError('directory close')
    monkeypatch.setattr(os, 'close', closing_after_body)
    with pytest.raises(capture.ContractError) as caught:
        with capture.regular(path) as stream:
            assert stream.read() == b'data'
            body_done = True
    assert isinstance(caught.value.__cause__, OSError)


def test_entry_oserror_is_contract_error_with_original_cause(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    primary = OSError('enter')
    fdopen = os.fdopen

    def opening(fd, *args, **kwargs):
        raw = fdopen(fd, *args, **kwargs)

        class Proxy:
            def __enter__(self):
                raise primary

            def close(self):
                raw.close()
                raise OSError('close')

        return Proxy()

    monkeypatch.setattr(os, 'fdopen', opening)
    with pytest.raises(capture.ContractError) as caught:
        with capture.regular(path):
            pass
    assert caught.value.code == 'corrupt_result'
    assert caught.value.__cause__ is primary


def test_caller_exception_does_not_hide_successful_body_close_error(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    close = os.close
    secondary = OSError('directory close')
    body_done = False

    def closing(fd):
        close(fd)
        if body_done:
            raise secondary

    monkeypatch.setattr(os, 'close', closing)
    try:
        raise LookupError('unrelated caller exception')
    except LookupError:
        with pytest.raises(capture.ContractError) as caught:
            with capture.regular(path) as stream:
                assert stream.read() == b'data'
                body_done = True
        assert caught.value.code == 'corrupt_result'
        assert caught.value.__cause__ is secondary


def test_caller_exception_does_not_replace_own_primary_during_cleanup(tmp_path, monkeypatch):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    close = os.close
    primary = Cancelled('own primary')
    body_started = False

    def closing(fd):
        close(fd)
        if body_started:
            raise OSError('directory close')

    monkeypatch.setattr(os, 'close', closing)
    try:
        raise LookupError('unrelated caller exception')
    except LookupError:
        with pytest.raises(Cancelled) as caught:
            with capture.regular(path):
                body_started = True
                raise primary
        assert caught.value is primary


def test_successful_regular_preserves_caller_exception(tmp_path):
    path = tmp_path / 'data'
    path.write_bytes(b'data')
    ambient = LookupError('caller exception')
    with pytest.raises(LookupError) as caught:
        try:
            raise ambient
        except LookupError:
            with capture.regular(path) as stream:
                assert stream.read() == b'data'
            raise
    assert caught.value is ambient
