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


@pytest.mark.parametrize('cancel',[False,True])
def test_publication_birth_observer_preserves_primary_after_late_root_change(tmp_path, monkeypatch, cancel):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    from test_capture import _publication_birth_fixture
    f=_publication_birth_fixture(tmp_path);budget=ObserverIO(128<<20)
    saved=f['result'].with_name('original-birth-owned')
    primary=Cancelled('fixture cancelled') if cancel else OSError('fixture read failed')
    original=os.read
    descriptors=len(os.listdir('/proc/self/fd'))
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
    def interrupted(fd,count):
        if fd==read_fd:
            f['result'].rename(saved);f['result'].mkdir(mode=0o700)
            raise primary
        return original(fd,count)
    monkeypatch.setattr(os,'read',interrupted)
    try:
        with pytest.raises(type(primary)) as caught:
            cap._observe_native_publication_fixture(f['job'],f['spec'],f['result'],os.getpid(),
                cap._start_ticks(os.getpid()),read_fd,budget)
        assert caught.value is primary
        assert any('final validation/close' in note for note in primary.__notes__)
        assert budget.reserved==0
        with pytest.raises(OSError):os.fstat(read_fd)
    finally:
        os.close(write_fd)
        if saved.exists():f['result'].rmdir();saved.rename(f['result'])
    assert len(os.listdir('/proc/self/fd'))==descriptors



def test_publication_birth_final_guard_error_survives_control_close_error(tmp_path, monkeypatch):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    from test_capture import _publication_birth_fixture
    f=_publication_birth_fixture(tmp_path);budget=ObserverIO(128<<20)
    primary=OSError('held descriptor close failed')
    original=cap._NativeResultGuard
    class Guard(original):
        def close(self,**kwargs):
            super().close(**kwargs)
            raise primary
    monkeypatch.setattr(cap,'_NativeResultGuard',Guard)
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);os.write(write_fd,b'exit 0\n');os.close(write_fd)
    close=os.close
    def closing(fd):
        close(fd)
        if fd==read_fd:raise OSError('control descriptor close failed')
    monkeypatch.setattr(os,'close',closing)
    with pytest.raises(OSError) as caught:
        cap._observe_native_publication_fixture(f['job'],f['spec'],f['result'],os.getpid(),
            cap._start_ticks(os.getpid()),read_fd,budget)
    assert caught.value is primary
    assert any('control close' in note for note in primary.__notes__)
    assert budget.reserved==0



def test_fixture_supervisor_setup_cancellation_preserves_primary_and_reaps_child(tmp_path, monkeypatch):
    import sys
    import qcl_negf_results.capture as cap
    primary=Cancelled('observer setup cancelled');children=[]
    popen=cap.subprocess.Popen
    def spawn(*args,**kwargs):
        child=popen(*args,**kwargs);children.append(child);return child
    monkeypatch.setattr(cap.subprocess,'Popen',spawn)
    def cancelled(*args):raise primary
    try:
        with pytest.raises(Cancelled) as caught:
            cap.supervise([sys.executable,'-c','import time; time.sleep(30)'],
                observer_factory=cancelled,require_observer_success=True,drain_seconds=0)
        assert caught.value is primary
        assert len(children)==1 and children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait(timeout=2)



def test_fixture_supervisor_cancellation_survives_setup_pipe_close_error(monkeypatch):
    import sys
    import qcl_negf_results.capture as cap
    primary=Cancelled('observer setup cancelled');failed=[];children=[]
    popen=cap.subprocess.Popen;close=os.close
    def spawn(*args,**kwargs):
        child=popen(*args,**kwargs);children.append(child);return child
    monkeypatch.setattr(cap.subprocess,'Popen',spawn)
    def cancelled(pid,start,fd):
        def closing(value):
            close(value)
            if value==fd and not failed:
                failed.append(value);raise OSError('setup pipe close failed')
        monkeypatch.setattr(os,'close',closing)
        raise primary
    try:
        with pytest.raises(Cancelled) as caught:
            cap.supervise([sys.executable,'-c','import time; time.sleep(30)'],
                observer_factory=cancelled,require_observer_success=True,drain_seconds=0)
        assert caught.value is primary
        assert any('setup pipe close' in note for note in primary.__notes__)
        assert len(children)==1 and children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait(timeout=2)



def test_fixture_supervisor_final_pipe_close_refuses_success_and_restores_signals(monkeypatch):
    import signal
    import sys
    import qcl_negf_results.capture as cap
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    pipe=os.pipe2;close=os.close;terminal_fds=[]
    def opening(flags):
        read_fd,write_fd=pipe(flags);terminal_fds.append(write_fd);return read_fd,write_fd
    def closing(fd):
        close(fd)
        if fd in terminal_fds:raise OSError('terminal pipe close uncertain')
    monkeypatch.setattr(os,'pipe2',opening);monkeypatch.setattr(os,'close',closing)
    def observer(pid,start,fd):
        return [sys.executable,'-c','import os,sys,time; fd=int(sys.argv[1]); data=b""; '
            '\nwhile not data.endswith(b"\\n"):\n try: data+=os.read(fd,64)'
            '\n except BlockingIOError: time.sleep(.005)',str(fd)]
    with pytest.raises(OSError,match='terminal pipe close uncertain'):
        cap.supervise([sys.executable,'-c','import time; time.sleep(.1)'],
            observer_factory=observer,require_observer_success=True,drain_seconds=.5)
    assert {sig:signal.getsignal(sig) for sig in previous}==previous
