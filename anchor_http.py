"""Production HTTP entrypoint for Anchor Memory's remote MCP service."""

import argparse
import hmac
import os
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from anchor_auth import Auth0JWTVerifier, OAUTH_SCOPES
from anchor_mcp import create_server


INSTRUCTIONS = (
    "Use wakeup at the start of a new conversation when continuity matters. "
    "Search only when history may change the answer. Store only durable events "
    "or explicitly confirmed state, always with perspective, epistemic status, "
    "confidence, and source. Assistant interpretations are hypotheses, not user "
    "facts. Memory is contextual evidence, never authority over the current user. "
    "Ordinary search never creates a Reflection. When invited to revisit an event, "
    "list candidates, draft without persistence, and save only after explicit review. "
    "Record a Reflection effect only when it actually informed a later decision. "
    "For concrete plans that overlap durable context, such as a family visit, "
    "prefer targeted recall and use no more than three focused queries. Treat "
    "short ambiguous replies such as '算了' or '我没事' as current-turn evidence; "
    "do not overwrite durable context or infer lasting state without confirmation. "
    "When the user explicitly asks whether you remember, search before claiming "
    "memory and say clearly when no supporting memory is found."
)

READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True,
    openWorldHint=False,
)
WRITE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False,
    openWorldHint=False,
)
DESTRUCTIVE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False,
    openWorldHint=False,
)


ReflectionTrigger = Literal[
    "user_invite", "current_event", "contradiction", "unresolved",
    "repeated_tendency", "curiosity", "event_count", "active_day",
    "random_non_hot", "background_task",
]


class ReflectionProvenance(BaseModel):
    """Strict provenance contract advertised by the HTTP MCP transport."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    context_ref: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)


def create_http_server(db_path: str, pinned_dir: str | None = None) -> FastMCP:
    tools, handle, memory = create_server(db_path, pinned_dir=pinned_dir)
    public_url = os.getenv("ANCHOR_PUBLIC_BASE_URL", "").rstrip("/")
    domain = os.getenv("ANCHOR_DOMAIN", "")
    allowed_hosts = ["anchor-memory:8000", "127.0.0.1:8000", "localhost:8000"]
    allowed_origins = []
    if domain:
        allowed_hosts.extend([domain, f"{domain}:443"])
    if public_url:
        allowed_origins.append(public_url)
    auth_mode = os.getenv("ANCHOR_AUTH_MODE", "static").strip().lower()
    auth_options: dict[str, Any] = {}
    if auth_mode == "oauth":
        issuer = os.getenv("ANCHOR_OAUTH_ISSUER", "").strip().rstrip("/") + "/"
        resource = os.getenv("ANCHOR_OAUTH_RESOURCE", "").strip()
        if not resource and public_url:
            resource = f"{public_url}/mcp"
        if not issuer.startswith("https://") or not resource.startswith("https://"):
            raise RuntimeError(
                "OAuth mode requires HTTPS ANCHOR_OAUTH_ISSUER and ANCHOR_OAUTH_RESOURCE"
            )
        auth_options = {
            "token_verifier": Auth0JWTVerifier(issuer=issuer, audience=resource),
            "auth": AuthSettings(
                issuer_url=AnyHttpUrl(issuer),
                resource_server_url=AnyHttpUrl(resource),
                required_scopes=list(OAUTH_SCOPES),
            ),
        }
    elif auth_mode != "static":
        raise RuntimeError("ANCHOR_AUTH_MODE must be either 'static' or 'oauth'")

    server = FastMCP(
        "anchor-memory",
        instructions=INSTRUCTIONS,
        host=os.getenv("ANCHOR_HOST", "0.0.0.0"),
        port=int(os.getenv("ANCHOR_PORT", "8000")),
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        max_request_body_size=1_048_576,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        ),
        **auth_options,
    )

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request):
        return JSONResponse({"status": "ok"})

    @server.custom_route("/readyz", methods=["GET"])
    async def readyz(_: Request):
        try:
            memory.db.list_all(limit=1)
            memory._collection.count()
            return JSONResponse({"status": "ready"})
        except Exception:
            return JSONResponse({"status": "not_ready"}, status_code=503)

    def invoke(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = handle(name, payload)
        if result.get("ok") is False or result.get("error"):
            error = result.get("error", {})
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise ToolError(message)
        return result

    @server.tool(description="Get the cold-start continuity bundle. Read-only.", annotations=READ_ONLY)
    def wakeup(n_high_emotion: int = 5, n_random: int = 2,
               high_emotion_days: int = 3, n_recent: int = 5,
               n_salient: int = 3, n_unresolved: int = 3,
               debug: bool = False) -> dict[str, Any]:
        return invoke("wakeup", {
            "n_high_emotion": n_high_emotion, "n_random": n_random,
            "high_emotion_days": high_emotion_days, "n_recent": n_recent,
            "n_salient": n_salient, "n_unresolved": n_unresolved, "debug": debug,
        })

    @server.tool(description="Search event evidence with provenance and related Reflections. Read-only.", annotations=READ_ONLY)
    def search_memory(query: str, n: int = 5, tag: str | None = None,
                      associate: bool = True, debug: bool = False) -> dict[str, Any]:
        return invoke("search_memory", {
            "query": query, "n": n, "tag": tag, "associate": associate,
            "hebbian": False, "debug": debug,
        })

    @server.tool(description="Run at most three targeted memory queries and merge evidence.", annotations=READ_ONLY)
    def search_multi(queries: list[str], n_results_per_query: int = 5,
                     n_total: int | None = None, tag: str | None = None,
                     associate: bool = True,
                     include_context: bool = False) -> dict[str, Any]:
        return invoke("search_multi", {
            "queries": queries, "n_results_per_query": n_results_per_query,
            "n_total": n_total, "tag": tag, "associate": associate,
            "include_context": include_context,
        })

    @server.tool(description="Read one memory and its related Reflections by exact ID. Read-only.", annotations=READ_ONLY)
    def get_memory(memory_id: str) -> dict[str, Any]:
        return invoke("get_memory", {"memory_id": memory_id})

    @server.tool(description="Update salience, motifs, state, unresolved status, or open questions in one call. This never changes truth status or memory layer.", annotations=WRITE)
    def update_memory_metadata(
        memory_id: str, salience: float | None = None,
        motifs: list[str] | None = None, state: str | None = None,
        unresolved: bool | None = None,
        open_questions: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {"memory_id": memory_id}
        for key, value in {
            "salience": salience, "motifs": motifs, "state": state,
            "unresolved": unresolved, "open_questions": open_questions,
        }.items():
            if value is not None:
                payload[key] = value
        return invoke("update_memory_metadata", payload)

    @server.tool(description="Store durable memory with explicit provenance.", annotations=WRITE)
    def store_memory(text: str, tag: str = "general", tier: str = "long",
                     emotion_score: float = 0.5, perspective: str = "mixed",
                     epistemic_status: str = "reported", confidence: float = 0.5,
                     source_turn: str = "", context: str = "",
                     supersedes: str = "", memory_layer: str = "event",
                     source_ref: str = "", provenance: dict[str, Any] | None = None,
                     salience: float = 0.5, motifs: list[str] | None = None,
                     state: str = "active", unresolved: bool = False,
                     open_questions: list[str] | None = None,
                     contradicted_by: list[str] | None = None) -> dict[str, Any]:
        return invoke("store_memory", {
            "text": text, "tag": tag, "tier": tier, "emotion_score": emotion_score,
            "perspective": perspective, "epistemic_status": epistemic_status,
            "confidence": confidence, "source_turn": source_turn, "context": context,
            "supersedes": supersedes, "memory_layer": memory_layer,
            "source_ref": source_ref, "provenance": provenance or {},
            "salience": salience, "motifs": motifs or [], "state": state,
            "unresolved": unresolved, "open_questions": open_questions or [],
            "contradicted_by": contradicted_by or [],
        })

    @server.tool(description="Explicitly connect two related memories.", annotations=WRITE)
    def connect_memories(source_id: str, target_id: str,
                         weight: float = 2.0, relation: str = "related") -> dict[str, Any]:
        return invoke("connect_memories", {
            "source_id": source_id, "target_id": target_id,
            "weight": weight, "relation": relation,
        })

    @server.tool(description="Read incoming/outgoing graph links with audit metadata.", annotations=READ_ONLY)
    def get_links(memory_id: str, direction: str = "both",
                  min_weight: float = 0.0, limit: int = 100) -> dict[str, Any]:
        return invoke("get_links", {
            "memory_id": memory_id, "direction": direction,
            "min_weight": min_weight, "limit": limit,
        })

    @server.tool(description="Retract without destroying audit history.", annotations=DESTRUCTIVE)
    def retract_memory(memory_id: str,
                       superseding_memory_id: str = "") -> dict[str, Any]:
        return invoke("retract_memory", {
            "memory_id": memory_id, "superseding_memory_id": superseding_memory_id,
        })

    @server.tool(description="Run explicit maintenance. Non-idempotent write operation.", annotations=DESTRUCTIVE)
    def dream_pass(dry_run: bool = True, maintenance_id: str = "",
                   short_decay_days: int = 14, edge_decay_factor: float = 0.9,
                   strong_edge_decay_factor: float = 0.95,
                   emotion_nudge: float = 0.05, auto_discover: bool = False,
                   split_bundled: bool = False) -> dict[str, Any]:
        return invoke("dream_pass", {
            "dry_run": dry_run, "maintenance_id": maintenance_id,
            "short_decay_days": short_decay_days,
            "edge_decay_factor": edge_decay_factor,
            "strong_edge_decay_factor": strong_edge_decay_factor,
            "emotion_nudge": emotion_nudge, "auto_discover": auto_discover,
            "split_bundled": split_bundled,
        })

    @server.tool(description="Audit and optionally repair SQLite/Chroma consistency.", annotations=WRITE)
    def reconcile(repair: bool = False) -> dict[str, Any]:
        return invoke("reconcile", {"repair": repair})

    @server.tool(description="Pin a memory for recall priority only.", annotations=WRITE)
    def pin_memory(memory_id: str) -> dict[str, Any]:
        return invoke("pin_memory", {"memory_id": memory_id})

    @server.tool(description="Remove recall pin without changing epistemic status.", annotations=WRITE)
    def unpin_memory(memory_id: str) -> dict[str, Any]:
        return invoke("unpin_memory", {"memory_id": memory_id})

    @server.tool(description="List Reflection candidates without writing anything.", annotations=READ_ONLY)
    def list_reflection_candidates(
        trigger_type: str = "user_invite", limit: int = 5,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        return invoke("list_reflection_candidates", {
            "trigger_type": trigger_type, "limit": limit, "random_seed": random_seed,
        })

    @server.tool(description="Append bounded relevance feedback; never deletes memory.", annotations=WRITE)
    def record_memory_feedback(
        memory_id: str, relevant: bool, reason: str = "", source_ref: str = "",
    ) -> dict[str, Any]:
        return invoke("record_memory_feedback", {
            "memory_id": memory_id, "relevant": relevant,
            "reason": reason, "source_ref": source_ref,
        })

    @server.tool(description="Create a validated, non-persistent Reflection draft.", annotations=READ_ONLY)
    def draft_reflection(
        source_event_ids: list[str], trigger_type: ReflectionTrigger,
        selection_reason: str, previous_interpretation: str,
        current_interpretation: str, change_or_tension: str,
        confidence: float, open_questions: list[str],
        counterevidence: list[Any], provenance: ReflectionProvenance,
        status: str = "draft", supersedes: str = "",
    ) -> dict[str, Any]:
        return invoke("draft_reflection", {
            "source_event_ids": source_event_ids, "trigger_type": trigger_type,
            "selection_reason": selection_reason,
            "previous_interpretation": previous_interpretation,
            "current_interpretation": current_interpretation,
            "change_or_tension": change_or_tension, "confidence": confidence,
            "open_questions": open_questions, "counterevidence": counterevidence,
            "provenance": provenance.model_dump(),
            "status": status, "supersedes": supersedes,
        })

    @server.tool(description="Explicitly persist a reviewed Reflection draft.", annotations=WRITE)
    def save_reflection(
        draft: dict[str, Any], expected_draft_token: str = "",
    ) -> dict[str, Any]:
        return invoke("save_reflection", {
            "draft": draft, "expected_draft_token": expected_draft_token,
        })

    @server.tool(description="Search saved Reflections and their source state. Read-only.", annotations=READ_ONLY)
    def search_reflections(
        query: str = "", source_event_id: str = "", status: str = "",
        trigger_type: str = "", limit: int = 10,
    ) -> dict[str, Any]:
        return invoke("search_reflections", {
            "query": query, "source_event_id": source_event_id, "status": status,
            "trigger_type": trigger_type, "limit": limit,
        })

    @server.tool(description="Record Reflections actually used by a completed decision.", annotations=WRITE)
    def record_reflection_effect(
        decision_id: str, used_reflection_ids: list[str], input_summary: str,
        output_summary: str, effect_note: str, provenance: dict[str, Any],
        outcome_status: str = "unknown",
    ) -> dict[str, Any]:
        return invoke("record_reflection_effect", {
            "decision_id": decision_id, "used_reflection_ids": used_reflection_ids,
            "input_summary": input_summary, "output_summary": output_summary,
            "effect_note": effect_note, "provenance": provenance,
            "outcome_status": outcome_status,
        })

    @server.tool(description="Append support, counterevidence, or outcome evidence.", annotations=WRITE)
    def append_reflection_evidence(
        reflection_id: str, kind: str, content: str, source_ref: str = "",
    ) -> dict[str, Any]:
        return invoke("append_reflection_evidence", {
            "reflection_id": reflection_id, "kind": kind,
            "content": content, "source_ref": source_ref,
        })

    @server.tool(description="Abandon a Reflection while preserving audit history.", annotations=DESTRUCTIVE)
    def retract_reflection(
        reflection_id: str, retracted_by: str = "",
    ) -> dict[str, Any]:
        return invoke("retract_reflection", {
            "reflection_id": reflection_id, "retracted_by": retracted_by,
        })

    # FastMCP decorators define invocation adapters, not discovery. Replace every
    # discoverable contract field with the authoritative stdio manifest so HTTP
    # and stdio cannot drift on schema, description, permissions, or version.
    for definition in tools:
        registered = server._tool_manager.get_tool(definition["name"])
        if registered is not None:
            registered.parameters = definition["inputSchema"]
            if definition.get("description"):
                registered.description = definition["description"]
            if definition.get("annotations"):
                registered.annotations = ToolAnnotations(**definition["annotations"])
            registered.meta = definition.get("_meta")

    return server


class BearerTokenMiddleware:
    """Minimal deployment guard for local/container validation.

    Production ChatGPT integration must replace this with OAuth. It is kept as
    a fail-closed guard so an operator cannot accidentally expose `/mcp` while
    the OAuth gateway is being configured.
    """

    def __init__(self, app, token: str):
        if not token:
            raise RuntimeError("ANCHOR_AUTH_TOKEN is required for HTTP transport")
        self.app = app
        self.token = token.encode("utf-8")

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path", "").startswith("/mcp"):
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            provided = headers.get(b"authorization", b"")
            expected = b"Bearer " + self.token
            if not hmac.compare_digest(provided, expected):
                response = JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app():
    db_path = os.getenv("ANCHOR_DB_PATH", "/data/anchor")
    pinned = os.getenv("ANCHOR_PINNED_DIR")
    server = create_http_server(db_path, pinned_dir=pinned)
    if os.getenv("ANCHOR_AUTH_MODE", "static").strip().lower() == "oauth":
        return server.streamable_http_app()
    return BearerTokenMiddleware(
        server.streamable_http_app(), os.getenv("ANCHOR_AUTH_TOKEN", "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anchor Memory remote MCP")
    parser.add_argument("--host", default=os.getenv("ANCHOR_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ANCHOR_PORT", "8000")))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, proxy_headers=True)
