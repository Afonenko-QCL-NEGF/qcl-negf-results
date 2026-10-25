"""Command line access to immutable scientific result snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qcl_negf_contracts.messages import ContractError

from .export_budget import DEFAULT_BYTE_BUDGET, DEFAULT_RESERVE_BYTES


def main() -> None:
    # Dispatch before importing any capture observer runtime. This entrypoint is
    # internal launcher plumbing, never a browser-chosen executable/path.
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='capture-run':
        capture=argparse.ArgumentParser(prog='qcl-negf-results capture-run')
        capture.add_argument('--spec',type=Path,required=True)
        capture.add_argument('--solver',required=True)
        capture.add_argument('arguments',nargs=argparse.REMAINDER)
        args=capture.parse_args(sys.argv[2:])
        arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
        from .capture import capture_run
        raise SystemExit(capture_run(args.spec,args.solver,arguments))
    from .export import export_snapshot, preview
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "export"):
        command = commands.add_parser(name)
        command.add_argument("results", type=Path)
        if name == "export":
            command.add_argument("destination", type=Path)
            command.add_argument("--byte-budget", type=int, default=DEFAULT_BYTE_BUDGET,
                                 help="maximum simultaneous intermediate export bytes (default: 64 GiB)")
            command.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES,
                                 help="free-space publication reserve (default: 64 MiB)")
        command.add_argument("--profile", choices=("science", "full-state"), default="science")
        command.add_argument("--status", choices=("running", "completed", "completed_with_warnings",
                            "failed", "cancelled", "paused"), default="running")
    render = commands.add_parser("render", help="render an immutable point artifact generation")
    render.add_argument("artifacts", type=Path, help="artifact directory containing current.json")
    render.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "render":
            from .presentation import committed_generation
            from .render import materialize
            generation, commit, digest = committed_generation(args.artifacts)
            value = materialize(generation, commit, digest, args.destination)
        elif args.command == "export":
            value = export_snapshot(args.results, args.destination, profile=args.profile,
                                    job_status=args.status, byte_budget=args.byte_budget,
                                    reserve_bytes=args.reserve_bytes)
        else:
            value = preview(args.results, profile=args.profile, job_status=args.status)
    except (ContractError, ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
