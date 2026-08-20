import argparse

from anchor_memory import AnchorMemory


parser = argparse.ArgumentParser(description="Compare SQLite and Chroma IDs")
parser.add_argument("--db-path", required=True)
parser.add_argument("--repair", action="store_true")
args = parser.parse_args()

memory = AnchorMemory(args.db_path)
print(memory.reconcile(repair=args.repair))
