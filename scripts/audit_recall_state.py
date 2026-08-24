"""Read-only recall-state audit for operators.

This command intentionally has no apply/repair mode.  It summarizes the
metadata that controls cold-start recall and verifies SQLite/Chroma parity
without printing memory bodies.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from anchor_memory import AnchorMemory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit of Anchor recall metadata and storage parity"
    )
    parser.add_argument("--db-path", required=True, help="Anchor data directory")
    parser.add_argument(
        "--n-salient", type=int, default=3,
        help="Number of salient wakeup items to inspect (default: 3)",
    )
    parser.add_argument(
        "--n-unresolved", type=int, default=3,
        help="Number of unresolved wakeup items to inspect (default: 3)",
    )
    return parser


def _effective_rows(memory):
    rows = memory.db.list_all(limit=10**9)
    return [
        row for row in rows
        if row.get("epistemic_status") != "retracted"
        and row.get("state") != "superseded"
    ]


def build_report(memory, *, n_salient=3, n_unresolved=3):
    """Build a body-free report using read-only methods only."""
    if n_salient < 0 or n_unresolved < 0:
        raise ValueError("wakeup limits must be non-negative")

    rows = _effective_rows(memory)
    salience = Counter(round(float(row.get("salience", 0.5)), 6) for row in rows)
    unresolved_rows = [row for row in rows if bool(row.get("unresolved"))]
    open_question_rows = [row for row in rows if row.get("open_questions")]
    total_open_questions = sum(len(row.get("open_questions") or []) for row in rows)

    wakeup = memory.db.wakeup(
        n_high_emotion=0,
        n_random=0,
        n_salient=n_salient,
        n_unresolved=n_unresolved,
        n_identity=0,
        debug=False,
    )
    reconcile = memory.reconcile(repair=False)
    reconcile_keys = (
        "missing_vectors", "orphan_vectors", "audit_only_vectors",
        "mismatched_vectors",
    )
    reconcile_clean = not any(reconcile.get(key) for key in reconcile_keys)

    return {
        "mode": "read-only",
        "effective_memory_count": len(rows),
        "salience_distribution": dict(sorted(salience.items())),
        "threshold_counts": {
            "gte_0.90": sum(float(row.get("salience", 0.5)) >= 0.90 for row in rows),
            "gte_0.80": sum(float(row.get("salience", 0.5)) >= 0.80 for row in rows),
            "gte_0.70": sum(float(row.get("salience", 0.5)) >= 0.70 for row in rows),
        },
        "unresolved_count": len(unresolved_rows),
        "open_question_memory_count": len(open_question_rows),
        "open_question_count": total_open_questions,
        "wakeup": {
            "salient_ids": [item["memory_id"] for item in wakeup.get("salient", [])],
            "unresolved_ids": [item["memory_id"] for item in wakeup.get("unresolved", [])],
            "salient_limit": n_salient,
            "unresolved_limit": n_unresolved,
        },
        "reconcile": {
            key: reconcile.get(key, []) for key in reconcile_keys
        } | {"clean": reconcile_clean},
    }


def main(argv=None, *, memory_factory=AnchorMemory) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        memory_factory(args.db_path),
        n_salient=args.n_salient,
        n_unresolved=args.n_unresolved,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["reconcile"]["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
