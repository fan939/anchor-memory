"""Idempotently add the independent Drive schema to an Anchor SQLite database."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from anchor_db import AnchorDB


def inspect(db_path: str) -> dict:
    db = AnchorDB(db_path)
    with db._conn() as conn:
        tables = {row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()}
        drive_columns = [row["name"] for row in conn.execute("PRAGMA table_info(drives)").fetchall()]
        link_columns = [row["name"] for row in conn.execute("PRAGMA table_info(drive_links)").fetchall()]
    return {
        "db_path": os.path.abspath(db_path),
        "drives_present": "drives" in tables,
        "drive_links_present": "drive_links" in tables,
        "drive_columns": drive_columns,
        "drive_link_columns": link_columns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path")
    args = parser.parse_args()
    if not os.path.exists(args.db_path):
        sqlite3.connect(args.db_path).close()
    print(inspect(args.db_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
