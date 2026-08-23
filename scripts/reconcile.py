"""Safe SQLite/Chroma reconciliation entry point.

The default path is read-only. Applying repairs requires an existing backup
path supplied by the operator; this script never creates, reads, or prints a
memory backup itself.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from anchor_memory import AnchorMemory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or repair SQLite/Chroma consistency for Anchor Memory"
    )
    parser.add_argument("--db-path", required=True, help="Anchor data directory")
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply vector rebuild/removal actions after a verified backup",
    )
    parser.add_argument(
        "--backup", help="Existing backup archive/path; required with --apply"
    )
    return parser


def main(argv=None, *, memory_factory=AnchorMemory, path_exists=os.path.exists) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.apply and (not args.backup or not path_exists(args.backup)):
        parser.error("--apply requires --backup pointing to an existing backup")

    memory = memory_factory(args.db_path)
    before = memory.reconcile(repair=False)
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "before": before,
    }
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    report["backup_evidence"] = os.path.abspath(args.backup)
    report["applied"] = memory.reconcile(repair=True)
    report["after"] = memory.reconcile(repair=False)
    remaining = (
        report["after"]["missing_vectors"]
        + report["after"]["orphan_vectors"]
        + report["after"]["audit_only_vectors"]
        + report["after"]["mismatched_vectors"]
    )
    report["clean"] = not remaining
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
