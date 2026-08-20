#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: restore.sh /backups/anchor-TIMESTAMP.tar.gz" >&2
    exit 2
fi

archive="$1"
target="${ANCHOR_DB_PATH:-/data/anchor}"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

tar -C "$staging" -xzf "$archive"
restored="$(find "$staging" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
test -f "$restored/memories.db"
test ! -e "$target/memories.db" || {
    echo "Target is not empty. Restore into an empty instance only." >&2
    exit 3
}
mkdir -p "$target"
cp -a "$restored/." "$target/"
echo "Restore complete: $target"
