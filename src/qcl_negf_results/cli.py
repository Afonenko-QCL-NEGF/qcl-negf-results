"""Command line access to one explicitly selected result-v2 source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qcl_negf_contracts.messages import ContractError
from .export_budget import DEFAULT_BYTE_BUDGET, DEFAULT_RESERVE_BYTES


def main() -> None:
    import sys
    # Existing observer lifecycle is owned by P30; do not rewrite its technical
    # cursor or infer scientific capabilities from its launcher success.
    if len(sys.argv) > 1 and sys.argv[1] == 'capture-run':
        capture = argparse.ArgumentParser(prog='qcl-negf-results capture-run')
        capture.add_argument('--spec', type=Path, required=True)
        capture.add_argument('--solver', required=True)
        capture.add_argument('arguments', nargs=argparse.REMAINDER)
        args = capture.parse_args(sys.argv[2:])
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        from .capture import capture_run
        raise SystemExit(capture_run(args.spec, args.solver, arguments))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('preview', 'export'):
        command = commands.add_parser(name)
        command.add_argument('results', type=Path, help='one artifact root')
        if name == 'export':
            command.add_argument('destination', type=Path)
            command.add_argument('--byte-budget', type=int, default=DEFAULT_BYTE_BUDGET)
            command.add_argument('--reserve-bytes', type=int, default=DEFAULT_RESERVE_BYTES)
        command.add_argument('--identity', type=Path, required=True, help='complete canonical Identity JSON')
        selection = command.add_mutually_exclusive_group(required=True)
        selection.add_argument('--state-ref', type=Path, help='exact canonical StateRef JSON')
        selection.add_argument('--pointer', choices=('live', 'recovery'))
        command.add_argument('--profile', choices=('science', 'full-state'), default='science')
    args = parser.parse_args()
    try:
        from .state import StateReader, ResultReadLimits
        from ._capture_io import small_json
        from .export import preview, export_snapshot
        limits = ResultReadLimits()
        identity, _ = small_json(args.identity, limits.maximum_metadata_bytes)
        if args.state_ref:
            ref, _ = small_json(args.state_ref, limits.maximum_metadata_bytes)
            reader = StateReader(args.results, state_ref=ref, expected_identity=identity, limits=limits)
        else:
            reader = StateReader.from_pointer(args.results, expected_identity=identity, pointer=args.pointer, limits=limits)
        with reader:
            if args.command == 'export':
                result = export_snapshot(reader, args.destination, profile=args.profile,
                    byte_budget=args.byte_budget, reserve_bytes=args.reserve_bytes)
            else:
                result = preview(reader, profile=args.profile)
    except (ContractError, ValueError, OSError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
