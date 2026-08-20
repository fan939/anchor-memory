#!/bin/sh
set -eu

source_dir="${ANCHOR_DB_PATH:-/data/anchor}"
backup_root="${ANCHOR_BACKUP_DIR:-/backups}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
destination="${backup_root}/anchor-${stamp}"

mkdir -p "$destination"
cp -a "$source_dir/." "$destination/"
tar -C "$backup_root" -czf "${destination}.tar.gz" "$(basename "$destination")"
rm -r "$destination"
echo "Backup created: ${destination}.tar.gz"
