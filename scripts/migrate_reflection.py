"""Dry-run/apply the backward-compatible Reflection schema migration."""

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from anchor_db import AnchorDB


MEMORY_COLUMNS = {"memory_layer", "source_ref", "provenance"}
REFLECTION_TABLES = {
    "reflections", "reflection_sources", "reflection_evidence",
    "decisions", "reflection_effects", "memory_feedback",
}


def inspect(db_file: str) -> dict:
    if not os.path.exists(db_file):
        return {
            "database_exists": False,
            "missing_memory_columns": sorted(MEMORY_COLUMNS),
            "missing_tables": sorted(REFLECTION_TABLES),
        }
    uri = f"file:{os.path.abspath(db_file).replace(os.sep, '/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        memory_columns = set()
        if "memories" in tables:
            memory_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(memories)")
            }
        return {
            "database_exists": True,
            "missing_memory_columns": sorted(MEMORY_COLUMNS - memory_columns),
            "missing_tables": sorted(REFLECTION_TABLES - tables),
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or apply the append-only Reflection schema"
    )
    parser.add_argument("--db-path", required=True, help="Anchor data directory or memories.db")
    parser.add_argument("--apply", action="store_true", help="Apply after a verified backup")
    parser.add_argument(
        "--backup", help="Existing backup archive/path required with --apply"
    )
    args = parser.parse_args()

    db_file = args.db_path
    if not db_file.endswith(".db"):
        db_file = os.path.join(db_file, "memories.db")
    before = inspect(db_file)
    report = {"mode": "apply" if args.apply else "dry-run", "before": before}

    if args.apply:
        if not args.backup or not os.path.exists(args.backup):
            parser.error("--apply requires --backup pointing to an existing backup")
        os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
        AnchorDB(db_file)
        report["backup_evidence"] = os.path.abspath(args.backup)
        report["after"] = inspect(db_file)
        if report["after"]["missing_memory_columns"] or report["after"]["missing_tables"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
