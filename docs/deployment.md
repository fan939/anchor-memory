# Tencent Cloud deployment

Do not expose port 8000 to the public internet. In the Tencent Cloud production
layout, host Nginx is the only public entry point; Anchor binds only to
`127.0.0.1:8100`. The application volume contains private data.

## Prerequisites

- A 64-bit Ubuntu server with at least 4 GB RAM.
- Docker Engine with the Compose plugin.
- A domain whose DNS A record points to the server.
- TCP 80 and 443 allowed in the Tencent Cloud security group. Restrict SSH 22
  to the administrator's IP whenever possible.
- For a mainland China server, complete the required ICP filing before serving
  the domain. Hong Kong and overseas regions do not use mainland ICP filing.

## Configure

Copy `.env.example` to `.env`. Set `ANCHOR_DOMAIN`,
`ANCHOR_PUBLIC_BASE_URL`, and generate `ANCHOR_AUTH_TOKEN` with a cryptographic
random generator. Never paste the token into source control or chat logs.

## Stable production deployment

The production data and model-cache volumes are deliberately external and
named `anchor-memory-data` and `anchor-memory-model-cache`. Never run a Compose
project that creates prefixed replacement volumes: it would start Anchor with
an empty memory database.

Use the deployment script from the checked-out release directory. It builds a
versioned image, creates a backup archive under `/opt/anchor-backups`, keeps the
previous container as a rollback target, and restores it automatically if the
new health check fails:

```sh
sudo ./scripts/deploy_production.sh v2-<commit>
```

The script does not update source code for you. First update the release
directory through your normal Git checkout/release-copy process, then run the
command above. It does not print environment variables, tokens, or memory
content.

Nginx owns ports 80/443 and proxies this service to `127.0.0.1:8100`; do not
start the old Compose Caddy service on this server.

## Validate before starting

Before updating an instance with private data, create and verify a backup, then
run the Reflection migration in dry-run mode. Do not use `--apply` until the
report and backup have been reviewed:

```sh
python scripts/migrate_reflection.py --db-path /data/anchor
```

The SQLite/Chroma consistency check is also read-only by default. Run the
preview first and inspect only IDs, counts, and repair reasons:

```sh
python scripts/reconcile.py --db-path /data/anchor
```

After a verified backup exists, an operator may apply vector-only repairs by
passing the existing backup path. The command rechecks the state after repair
and exits non-zero if anything remains inconsistent:

```sh
python scripts/reconcile.py \
  --db-path /data/anchor \
  --apply \
  --backup /backups/anchor-<timestamp>.tar.gz
```

The reconcile report never includes memory text; do not paste backup paths,
database files, or report contents containing private data into chat. Pending
repair-journal entries remain review items and are not silently replayed.

For a repeatable read-only recall audit, inspect effective-memory counts,
salience thresholds, unresolved/open-question counts, wakeup IDs, and storage
parity without exposing memory bodies:

```sh
python scripts/audit_recall_state.py --db-path /data/anchor
```

This command has no repair or apply mode. A non-zero exit code means the
read-only reconcile section still reports drift; it does not attempt to fix it.

```sh
docker-compose config
sudo docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
curl -fsS https://memory.example.com/healthz
curl -i https://memory.example.com/mcp
```

The unauthenticated MCP request must return `401`. Health should return a small
status response. Readiness must not return private memory content.

## Persistence and recovery

Stop writes before filesystem-level backup. Back up the complete Anchor volume,
including `memories.db`, `chroma/`, and `pinned/`. Test restores only into an
empty volume. Keep encrypted off-server copies and configure retention.

## OAuth boundary

The built-in Bearer guard prevents accidental public exposure during local and
container validation. It is not the final ChatGPT sign-in flow. Put an OAuth
2.1-capable authorization/resource-server gateway in front of `/mcp`, validate
issuer, audience, expiry, and scopes, and retain the fail-closed application
guard until OAuth end-to-end tests pass.

Current OpenAI plugin guidance expects an authenticated private-data MCP server
to implement the MCP OAuth 2.1 authorization contract. Publish protected
resource metadata at `/.well-known/oauth-protected-resource`, publish
authorization-server metadata, preserve the exact `resource` parameter through
the flow, validate tokens on every request, and support a compatible OpenAI
client registration/identification method plus PKCE. Reference:
https://developers.openai.com/plugins/build/auth

Static `ANCHOR_AUTH_TOKEN` remains a staging fail-closed guard only. ICP filing
and TLS readiness alone are not authorization to publish the staging container.

For Auth0-backed production access, set `ANCHOR_AUTH_MODE=oauth`, configure the
exact HTTPS issuer in `ANCHOR_OAUTH_ISSUER`, and set `ANCHOR_OAUTH_RESOURCE` to
the API identifier ending in `/mcp`. The server then publishes protected-resource
metadata at `/.well-known/oauth-protected-resource/mcp` and validates RS256
signatures, issuer, audience, expiry, subject, and the required Anchor scopes.
