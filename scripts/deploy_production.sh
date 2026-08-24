#!/bin/sh
# Safe single-service production deploy for the Tencent Cloud layout.
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root, for example: sudo ./scripts/deploy_production.sh v2-<commit>." >&2
    exit 2
fi

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 v2-<release-id>" >&2
    exit 2
fi

release_tag="$1"
case "$release_tag" in
    *[!A-Za-z0-9._-]* | '')
        echo "Release ID may contain only letters, numbers, dot, underscore, and hyphen." >&2
        exit 2
        ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
env_file="${ANCHOR_ENV_FILE:-$project_dir/.env}"
data_volume="${ANCHOR_DATA_VOLUME:-anchor-memory-data}"
model_volume="${ANCHOR_MODEL_CACHE_VOLUME:-anchor-memory-model-cache}"
backup_dir="${ANCHOR_BACKUP_HOST_DIR:-/opt/anchor-backups}"
container_name="anchor-memory"
rollback_name="anchor-memory-rollback-$(date -u +%Y%m%dT%H%M%SZ)-${release_tag}"
image="anchor-memory:${release_tag}"

test -f "$env_file" || {
    echo "Environment file not found: $env_file" >&2
    exit 2
}
docker volume inspect "$data_volume" >/dev/null
docker volume inspect "$model_volume" >/dev/null
mkdir -p "$backup_dir"

cd "$project_dir"
docker build --tag "$image" .

# The backup container sees live data read-only; the archive is written outside
# Docker's volume store for recovery.
docker run --rm --user root \
    --env ANCHOR_BACKUP_DIR=/backups \
    --volume "$data_volume:/data/anchor:ro" \
    --volume "$backup_dir:/backups" \
    "$image" sh scripts/backup.sh >/dev/null

rollback_created=0
if docker container inspect "$container_name" >/dev/null 2>&1; then
    docker stop "$container_name" >/dev/null
    docker rename "$container_name" "$rollback_name"
    rollback_created=1
fi

restore_previous() {
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    if [ "$rollback_created" -eq 1 ]; then
        docker rename "$rollback_name" "$container_name"
        docker start "$container_name" >/dev/null
    fi
}

trap 'restore_previous' EXIT INT TERM HUP

docker run -d --name "$container_name" \
    --restart unless-stopped \
    --env-file "$env_file" \
    --user anchor \
    --memory 2800m \
    --cpus 3.0 \
    --tmpfs /tmp:size=256m,mode=1777 \
    --security-opt no-new-privileges:true \
    --cap-drop ALL \
    --volume "$data_volume:/data/anchor" \
    --volume "$model_volume:/data/model-cache" \
    --publish "127.0.0.1:${ANCHOR_LOCAL_PORT:-8100}:8000" \
    "$image" python anchor_http.py >/dev/null

attempt=0
while [ "$attempt" -lt 36 ]; do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$container_name" 2>/dev/null || true)
    if [ "$status" = "healthy" ]; then
        trap - EXIT INT TERM HUP
        if [ "$rollback_created" -eq 1 ]; then
            echo "Deployed $image; rollback container: $rollback_name"
        else
            echo "Deployed $image."
        fi
        exit 0
    fi
    if [ "$status" = "unhealthy" ]; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 5
done

echo "New container did not become healthy; restoring previous container." >&2
exit 1
