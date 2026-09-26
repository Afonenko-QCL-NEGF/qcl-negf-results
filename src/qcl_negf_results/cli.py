"""Command line access to immutable scientific result snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qcl_negf_contracts.messages import ContractError
from .export import export_snapshot, preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "export"):
        command = commands.add_parser(name)
        command.add_argument("results", type=Path)
        if name == "export":
            command.add_argument("destination", type=Path)
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
                                    job_status=args.status)
        else:
            value = preview(args.results, profile=args.profile, job_status=args.status)
    except (ContractError, ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
