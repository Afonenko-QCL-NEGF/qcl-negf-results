"""Tiny ready/TERM supervisor stand-ins; run only inside the finite test gate."""

import json
from pathlib import Path
import resource
import sys
import time

from qcl_negf_results.capture import supervise

resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
root, mode = Path(sys.argv[1]), sys.argv[2]
solver = r'''
import os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
signal.signal(signal.SIGTERM, lambda *_: sys.exit(23))
(root / "solver-ready").write_text(str(os.getpid()))
while not (root / "observer-ready").exists():
    time.sleep(.01)
if sys.argv[2] == "normal":
    sys.exit(7)
time.sleep(30)
'''
observer = r'''
import os, sys, time
from pathlib import Path
(Path(sys.argv[1]) / "observer-ready").write_text(str(os.getpid()))
time.sleep(30)
'''
start = time.monotonic()
code = supervise(
    [sys.executable, "-c", solver, str(root), mode],
    observer_factory=lambda *_: [sys.executable, "-c", observer, str(root)],
    drain_seconds=.15,
)
observer_pid = int((root / "observer-ready").read_text())
print(json.dumps({
    "code": code, "elapsed": time.monotonic() - start,
    "observer_reaped": not Path(f"/proc/{observer_pid}").exists(),
}))
