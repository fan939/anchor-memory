"""
Anchor Memory System — MCP Server

Exposes Anchor Memory as an MCP (Model Context Protocol) server.
Any MCP-compatible client (Claude Code, claude.ai, LobeHub, SillyTavern)
can connect and use graph-structured memory with Hebbian learning.

Usage:
    python anchor_mcp.py [--db-path ./my_memory] [--port 3333]
"""

import json
import sys
import os
import uuid
import argparse
from datetime import datetime

# Windows fix: force UTF-8 on stdin/stdout to prevent GBK encoding issues
# (Windows cmd defaults to GBK; mcp_proxy communicates in UTF-8)
if sys.platform == "win32" or (hasattr(sys.stdout, 'buffer') and sys.stdout.encoding and sys.stdout.encoding.upper() != 'UTF-8'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import anchor_pinned


SERVER_VERSION = "1.16.0"
TOOL_SCHEMA_VERSION = "1.6"
TOOL_SCHEMA_META_KEY = "anchor/schema_version"


def create_server(db_path: str = "./anchor_data", pinned_dir: str = None):
    """Create MCP server with Anchor Memory tools.

    pinned_dir: optional directory for the always-injected file layer
    (session_state.md / recent_timeline.md / last_session.md — see
    anchor_pinned.py). When set, wakeup() returns those files too and the
    write_session_state tool becomes functional. Defaults to <db_path>/pinned.
    """

    # Imported here, not at module top: the CLI hook modes (--wakeup-text)
    # must stay on the SQLite-only fast path — importing anchor_memory pulls
    # in sentence_transformers/chromadb (seconds).
    from anchor_memory import AnchorMemory
    from anchor_drive import DriveService
    from anchor_reflection import ReflectionService

    mem = AnchorMemory(db_path=db_path)
    drives = DriveService(mem.db)
    reflection = ReflectionService(mem.db)
    pinned_dir = pinned_dir or os.path.join(db_path, "pinned")

    provenance_schema = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "model": {"type": "string"},
            "thread_id": {"type": "string"},
            "context_ref": {"type": "string"},
            "extractor_version": {"type": "string"},
        },
        "required": ["model", "thread_id", "context_ref", "extractor_version"],
    }
    reflection_draft_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_event_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "trigger_type": {"type": "string", "enum": [
                "user_invite", "current_event", "contradiction", "unresolved",
                "repeated_tendency", "curiosity", "event_count", "active_day",
                "random_non_hot", "background_task",
            ]},
            "selection_reason": {"type": "string"},
            "previous_interpretation": {"type": "string"},
            "current_interpretation": {"type": "string"},
            "change_or_tension": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "counterevidence": {"type": "array", "items": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "object", "properties": {
                        "content": {"type": "string"}, "source_ref": {"type": "string"},
                    }, "required": ["content"], "additionalProperties": False},
                ]
            }},
            "provenance": provenance_schema,
            "status": {"type": "string", "enum": ["draft", "tentative", "shaken", "abandoned"]},
            "supersedes": {"type": "string"},
        },
        "required": [
            "source_event_ids", "trigger_type", "selection_reason",
            "previous_interpretation", "current_interpretation",
            "change_or_tension", "confidence", "open_questions",
            "counterevidence", "provenance",
        ],
    }

    # MCP tool definitions
    TOOLS = [
        {
            "name": "store_memory",
            "description": "Store a new memory. Memories are nodes in a graph — they can be connected to other memories and carry emotional weight.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The memory content. Preserve narrative and original words — memories should read like flashbacks, not file entries."
                    },
                    "tag": {
                        "type": "string",
                        "description": "Category: relationship, identity, emotion, learning, history, project, practical, research, or any custom tag.",
                        "default": "general"
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["core", "long", "short"],
                        "description": "core = permanent. long = kept indefinitely. short = decays after 14 days.",
                        "default": "long"
                    },
                    "emotion_score": {
                        "type": "number",
                        "description": "0.0 (neutral) to 1.0 (intense). How emotionally heavy is this memory? Most are 0.3-0.6. Only truly intense moments get above 0.8.",
                        "default": 0.5
                    },
                    "memory_layer": {
                        "type": "string",
                        "enum": ["core", "dynamic", "event"],
                        "default": "event",
                        "description": "Semantic layer. Core requires confirmed status; Reflection is stored separately."
                    },
                    "perspective": {
                        "type": "string",
                        "enum": ["user", "assistant", "joint", "external", "system", "mixed"],
                        "default": "mixed",
                        "description": "Whose perspective the memory represents."
                    },
                    "epistemic_status": {
                        "type": "string",
                        "enum": ["reported", "observed", "hypothesis", "confirmed", "retracted"],
                        "default": "reported",
                        "description": "Knowledge status; assistant interpretations should normally be hypothesis."
                    },
                    "confidence": {
                        "type": "number", "minimum": 0.0, "maximum": 1.0,
                        "default": 0.5
                    },
                    "source_turn": {
                        "type": "string",
                        "description": "Stable source turn or equivalent provenance locator."
                    },
                    "source_ref": {
                        "type": "string",
                        "description": "Stable source locator, such as a thread/turn or external record."
                    },
                    "provenance": {
                        "type": "object",
                        "description": "Model/window/context/extractor provenance. Never include secrets or full private text."
                    },
                    "supersedes": {
                        "type": "string",
                        "description": "Optional older memory ID this record supersedes."
                    },
                    "connect_to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of memory_ids to explicitly connect this memory to."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional full original text (two-layer storage: text = searchable summary that gets embedded, context = verbatim source). Returned by searches with include_context."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "search_memory",
            "description": "Search memories and return event evidence with provenance and related Reflections. Read-only: it does not cite, create Reflections, or strengthen edges.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Max results.",
                        "default": 5
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag."
                    },
                    "associate": {
                        "type": "boolean",
                        "description": "Follow graph edges to find related memories.",
                        "default": True
                    },
                    "hebbian": {
                        "type": "boolean",
                        "description": "Compatibility field only; MCP search remains read-only and ignores this hint.",
                        "default": False
                    },
                    "debug": {
                        "type": "boolean",
                        "description": "Include ranking internals on each result — raw_distance, citation_boost, emotion_boost, final_score, source ('vector'|'keyword'|'associative'), and edge_weight for associative hops. Use to audit why a given result landed at its rank.",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_multi",
            "description": "Run multiple read-only searches and merge results. It does not create Reflections, citations, or graph edges.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of intent strings to search. Example: for the message '七月去欧洲要不要换4K拍摄', pass ['七月欧洲行', '4K拍摄设置']."
                    },
                    "n_results_per_query": {
                        "type": "integer",
                        "description": "Top-k pulled from each individual search.",
                        "default": 5
                    },
                    "n_total": {
                        "type": "integer",
                        "description": "Final cap after merge. Default n_results_per_query * len(queries)."
                    },
                    "tag": {"type": "string"},
                    "associate": {"type": "boolean", "default": True},
                    "hebbian": {"type": "boolean", "default": False},
                    "include_context": {"type": "boolean", "default": False}
                },
                "required": ["queries"]
            }
        },
        {
            "name": "connect_memories",
            "description": "Explicitly connect two memories. Creates a weighted bidirectional edge (synapse). Use for manual entanglement — connecting memories you know are related.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "weight": {
                        "type": "number",
                        "description": "Connection strength. Hebbian auto-connections are 0.2. Manual entanglement is typically 1.5-3.0. Max 10.0.",
                        "default": 2.0
                    }
                },
                "required": ["source_id", "target_id"]
            }
        },
        {
            "name": "get_neighbors",
            "description": "Get memories connected to a given memory via graph edges. Returns neighbors sorted by edge weight.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "min_weight": {
                        "type": "number",
                        "description": "Minimum edge weight to include.",
                        "default": 0.5
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "get_memory",
            "description": "Read one memory by exact ID with provenance and epistemic status.",
            "inputSchema": {
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"]
            }
        },
        {
            "name": "retract_memory",
            "description": "Retract a memory without deleting its audit history. Use superseding_memory_id when a correction replaces it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "superseding_memory_id": {"type": "string"}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "reconcile",
            "description": "Admin-only dual-store consistency report. Defaults to dry-run; repair removes orphan/audit-only vectors and rebuilds missing active vectors from SQLite authority.",
            "inputSchema": {
                "type": "object",
                "properties": {"repair": {"type": "boolean", "default": False}}
            }
        },
        {
            "name": "delete_memory",
            "description": "Delete a memory and all its edges.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "dream_pass",
            "description": "Preview or run audited maintenance. MCP defaults to dry-run; an explicit maintenance ID makes retries idempotent.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "set_emotion",
            "description": "Set the emotion score of an existing memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "score": {
                        "type": "number",
                        "description": "0.0 (neutral) to 1.0 (intense)."
                    }
                },
                "required": ["memory_id", "score"]
            }
        },
        {
            "name": "set_tier",
            "description": "Change the tier of an existing memory (core/long/short).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "tier": {
                        "type": "string",
                        "enum": ["core", "long", "short"]
                    }
                },
                "required": ["memory_id", "tier"]
            }
        },
        {
            "name": "graph_stats",
            "description": "Get overview stats: total memories, edges, tag distribution, tier distribution, top connected nodes.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "annotate_memory",
            "description": "Add an annotation to a memory. Annotations are append-only — they record how understanding of a memory evolves over time. Searchable. Original memory text is never changed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "The memory to annotate."},
                    "text": {"type": "string", "description": "The annotation text. E.g. '4/18: realized this was about X, not Y.'"}
                },
                "required": ["memory_id", "text"]
            }
        },
        {
            "name": "get_annotations",
            "description": "Get all annotations for a memory, oldest first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "The memory to get annotations for."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "consolidate",
            "description": "Passive Hebbian update — after a conversation, pass key topics to build connections between memories that co-occurred but weren't explicitly searched. Zero LLM token cost. Call at the end of a conversation or session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "conversation_text": {
                        "type": "string",
                        "description": "Key topics from the conversation. E.g. 'talked about her friend Lily, yesterday's dinner (ramen), the cockroach incident, her work project'"
                    }
                },
                "required": ["conversation_text"]
            }
        },
        {
            "name": "store_visual",
            "description": "Store a visual observation as a memory with CLIP embedding. For Anchor Vision integration — lets the system remember what it has seen.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text description of what was seen. E.g. 'red earring, round, small'"},
                    "visual_embedding": {"type": "string", "description": "CLIP embedding as JSON array string."},
                    "tag": {"type": "string", "enum": ["visual", "general"], "description": "Tag. Use 'visual' for visual observations."},
                    "connect_to": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Memory IDs to connect this observation to."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "wakeup",
            "description": "One-call cold start. Returns pinned memories + most recent memories (timestamp order, no emotion filter) + recent high-emotion + random old + unread comments, plus the pinned file layer when configured: session_state (your own rolling state from previous windows), recent_timeline (event ledger), last_session (mechanical tail of the previous window). Call FIRST at the start of a new conversation/window. Does NOT mark unread comments as read — call mark_comments_read separately after processing them.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_recent": {"type": "integer", "description": "How many most-recent memories to return (timestamp order, no emotion filter).", "default": 5},
                    "n_high_emotion": {"type": "integer", "description": "How many recent high-emotion memories to return.", "default": 5},
                    "n_random": {"type": "integer", "description": "How many random old memories to return.", "default": 2},
                    "high_emotion_days": {"type": "integer", "description": "How many days back counts as 'recent' for the high-emotion block.", "default": 3}
                }
            }
        },
        {
            "name": "write_session_state",
            "description": "Write your session_state.md — your own rolling state that carries across windows (what's ongoing, decisions made, current threads, mood). This is the ONLY correct way to update it: the current version is archived automatically before the new one is written, and a continuity header is added so future windows read it as their own state, not a message from someone else. Write the COMPLETE current state (not a diff), in first person. Update it when things change materially and when a conversation wraps up.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The complete new session_state.md content."}
                },
                "required": ["content"]
            }
        },
        {
            "name": "leave_comment",
            "description": "Leave a comment on a memory. The primary mechanism for cross-window messaging — comments left here will surface in the next instance's wakeup() call as unread. Useful for leaving context, decisions, or messages for future-you.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to attach comment to."},
                    "content": {"type": "string", "description": "The comment text."},
                    "author": {"type": "string", "enum": ["ai", "human"], "default": "ai", "description": "Who is leaving the comment."},
                    "reply_to": {"type": "string", "description": "Optional: comment_id this is replying to."}
                },
                "required": ["memory_id", "content"]
            }
        },
        {
            "name": "get_comments",
            "description": "Get all comments on a specific memory (both read and unread). Use this to read the full conversation thread on a memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to fetch comments for."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "mark_comments_read",
            "description": "Mark comments as read so they don't reappear in next wakeup. Call after processing the unread comments returned by wakeup().",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "comment_ids": {"type": "array", "items": {"type": "string"}, "description": "Comment IDs to mark as read."},
                    "reader": {"type": "string", "enum": ["ai", "human"], "default": "ai", "description": "Who is marking as read."}
                },
                "required": ["comment_ids"]
            }
        },
        {
            "name": "pin_memory",
            "description": "Pin any active memory for wakeup() recall priority only. Pinning never changes epistemic status or memory layer; retracted or superseded memories are rejected.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to pin."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "unpin_memory",
            "description": "Remove pinned status from a memory. The memory remains in storage but stops appearing in wakeup()'s pinned section.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to unpin."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "search_annotations",
            "description": "Search across annotation text on memories. Returns matching memory_ids and the annotations themselves. Use when looking for memories by what was added to them later (commentary, corrections, additions).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — words to match against annotation text."},
                    "limit": {"type": "integer", "description": "Max results.", "default": 5}
                },
                "required": ["query"]
            }
        },
        {
            "name": "cite_memory",
            "description": "Explicitly increment usage after a memory actually informed current reasoning. Search itself is read-only and never auto-cites.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory to cite."}
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "list_reflection_candidates",
            "description": "List event seeds that may be worth reviewing. Read-only: a trigger opens an opportunity and never creates a Reflection.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "trigger_type": {"type": "string", "default": "user_invite"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                    "random_seed": {"type": "integer", "description": "Makes random_non_hot selection reproducible."}
                }
            }
        },
        {
            "name": "record_memory_feedback",
            "description": "Append relevance feedback for later ranking. One negative report never deletes or retracts a memory.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "memory_id": {"type": "string"},
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string", "default": ""},
                    "source_ref": {"type": "string", "default": ""}
                },
                "required": ["memory_id", "relevant"]
            }
        },
        {
            "name": "draft_reflection",
            "description": "Validate and normalize a structured Reflection draft. Read-only and never persisted.",
            "inputSchema": reflection_draft_schema,
        },
        {
            "name": "save_reflection",
            "description": "Explicitly persist a previously reviewed Reflection draft after revalidating all event source IDs.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "draft": reflection_draft_schema,
                    "expected_draft_token": {"type": "string"}
                },
                "required": ["draft"]
            }
        },
        {
            "name": "search_reflections",
            "description": "Search saved Reflections by text, source event, status, or trigger. Read-only.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "default": ""},
                    "source_event_id": {"type": "string", "default": ""},
                    "status": {"type": "string", "default": ""},
                    "trigger_type": {"type": "string", "default": ""},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10}
                }
            }
        },
        {
            "name": "record_reflection_effect",
            "description": "Record a decision only after listed Reflections were actually used. Never infer influence after the fact.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "decision_id": {"type": "string"},
                    "used_reflection_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "input_summary": {"type": "string"},
                    "output_summary": {"type": "string"},
                    "effect_note": {"type": "string"},
                    "outcome_status": {"type": "string", "default": "unknown"},
                    "provenance": provenance_schema
                },
                "required": ["decision_id", "used_reflection_ids", "input_summary", "output_summary", "effect_note", "provenance"]
            }
        },
        {
            "name": "append_reflection_evidence",
            "description": "Append support, counterevidence, or an outcome without overwriting the Reflection.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "reflection_id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["support", "counterevidence", "outcome"]},
                    "content": {"type": "string"},
                    "source_ref": {"type": "string", "default": ""}
                },
                "required": ["reflection_id", "kind", "content"]
            }
        },
        {
            "name": "retract_reflection",
            "description": "Mark a Reflection abandoned while retaining its source and decision audit trail.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "reflection_id": {"type": "string"},
                    "retracted_by": {"type": "string", "default": ""}
                },
                "required": ["reflection_id"]
            }
        }
    ]

    # Keep the advertised contract aligned with business validation. The same
    # schemas are reused by the HTTP adapter below, so clients do not discover
    # constraints only after a rejected write.
    tool_map = {tool["name"]: tool for tool in TOOLS}
    store_schema = tool_map["store_memory"]["inputSchema"]
    store_schema["additionalProperties"] = False
    store_schema["properties"]["emotion_score"].update({"minimum": 0.0, "maximum": 1.0})
    store_schema["properties"]["text"]["minLength"] = 1
    store_schema["properties"].update({
        "salience": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5,
                     "description": "Recall importance only; never changes truth status or memory layer."},
        "motifs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True,
                   "description": "Repeated themes or structures used to guide recall."},
        "state": {"type": "string", "enum": ["active", "weakened", "resolved", "superseded"],
                  "default": "active"},
        "unresolved": {"type": "boolean", "default": False},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "contradicted_by": {"type": "array", "items": {"type": "string"}, "uniqueItems": True,
                            "description": "Memory IDs providing explicit counterevidence."},
    })
    store_schema["allOf"] = [{
        "if": {"properties": {"memory_layer": {"const": "core"}}, "required": ["memory_layer"]},
        "then": {"properties": {"epistemic_status": {"const": "confirmed"}},
                 "required": ["epistemic_status"]},
    }]
    tool_map["search_multi"]["inputSchema"]["additionalProperties"] = False
    tool_map["search_multi"]["inputSchema"]["properties"]["queries"].update(
        {"minItems": 1, "maxItems": 3}
    )

    for name in ("search_memory", "get_memory", "get_neighbors", "reconcile",
                 "retract_memory", "delete_memory", "pin_memory", "unpin_memory",
                 "draft_reflection", "search_reflections", "save_reflection",
                 "retract_reflection", "record_memory_feedback",
                 "record_reflection_effect", "append_reflection_evidence"):
        if name in tool_map:
            tool_map[name]["inputSchema"]["additionalProperties"] = False

    tool_map["connect_memories"]["inputSchema"].update({"additionalProperties": False})
    tool_map["connect_memories"]["inputSchema"]["properties"]["weight"].update(
        {"minimum": 0.0, "maximum": 10.0}
    )
    tool_map["connect_memories"]["inputSchema"]["properties"]["relation"] = {
        "type": "string", "enum": ["related", "contradicts", "supports", "supersedes"],
        "default": "related",
    }
    tool_map["wakeup"]["inputSchema"] = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "n_high_emotion": {"type": "integer", "minimum": 0, "maximum": 20, "default": 5},
            "n_random": {"type": "integer", "minimum": 0, "maximum": 20, "default": 2},
            "high_emotion_days": {"type": "integer", "minimum": 1, "maximum": 3650, "default": 3},
            "n_recent": {"type": "integer", "minimum": 0, "maximum": 20, "default": 5},
            "n_salient": {"type": "integer", "minimum": 0, "maximum": 20, "default": 3},
            "n_unresolved": {"type": "integer", "minimum": 0, "maximum": 20, "default": 3},
            "n_identity": {"type": "integer", "minimum": 0, "maximum": 20, "default": 5},
            "n_reflections": {"type": "integer", "minimum": 0, "maximum": 20, "default": 2},
            "include_draft_reflections": {"type": "boolean", "default": False,
                "description": "Draft Reflections are excluded from normal cold start unless explicitly requested."},
            "debug": {"type": "boolean", "default": False,
                      "description": "Audit-only filter explanations; filtered content is never put in normal sections."},
        },
    }
    tool_map["dream_pass"]["inputSchema"] = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "dry_run": {"type": "boolean", "default": True},
            "maintenance_id": {"type": "string", "minLength": 1,
                               "description": "Caller-provided idempotency and audit ID."},
            "short_decay_days": {"type": "integer", "minimum": 1, "default": 14},
            "edge_decay_factor": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 0.9},
            "strong_edge_decay_factor": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 0.95},
            "emotion_nudge": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.05},
            "auto_discover": {"type": "boolean", "default": False},
            "split_bundled": {"type": "boolean", "default": False},
        },
    }
    TOOLS.append({
        "name": "get_links",
        "description": "Read auditable incoming/outgoing memory links, including relation, weight, and timestamps.",
        "inputSchema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "memory_id": {"type": "string"},
                "direction": {"type": "string", "enum": ["incoming", "outgoing", "both"], "default": "both"},
                "min_weight": {"type": "number", "minimum": 0, "maximum": 10, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
            },
            "required": ["memory_id"],
        },
    })
    TOOLS.append({
        "name": "update_memory_metadata",
        "description": "Update salience, motifs, state, unresolved status, or open questions in one call. This never changes epistemic status or memory layer.",
        "inputSchema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "memory_id": {"type": "string"},
                "salience": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "motifs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "state": {"type": "string", "enum": ["active", "weakened", "resolved", "superseded"]},
                "unresolved": {"type": "boolean"},
                "open_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["memory_id"],
            "minProperties": 2,
        },
    })
    TOOLS.append({
        "name": "reconcile_recall_metadata",
        "description": "Preview or explicitly apply synchronization of saved, non-abandoned Reflection open questions to source memories. Never infers from prose and never automatically changes salience.",
        "inputSchema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "dry_run": {"type": "boolean", "default": True,
                            "description": "Preview only unless explicitly set to false."},
                "maintenance_id": {"type": "string", "minLength": 1,
                                   "description": "Required to apply; provides audit and idempotency."},
                "max_candidates": {"type": "integer", "minimum": 1, "maximum": 500,
                                   "default": 100},
            },
            "allOf": [{
                "if": {"properties": {"dry_run": {"const": False}}, "required": ["dry_run"]},
                "then": {"required": ["maintenance_id"]},
            }],
        },
    })
    drive_statuses = ["active", "paused", "satisfied", "abandoned", "superseded"]
    drive_link_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "target_type": {"type": "string", "enum": ["memory", "reflection", "drive"]},
            "target_id": {"type": "string", "minLength": 1},
            "relation": {"type": "string", "enum": [
                "motivated_by", "supported_by", "conflicts_with", "depends_on", "supersedes", "related",
            ]},
            "weight": {"type": "number", "minimum": 0.0, "default": 1.0},
        },
        "required": ["target_type", "target_id", "relation"],
    }
    TOOLS.extend([
        {
            "name": "create_drive",
            "description": "Persist a current Drive without creating an Outbox action or changing Memory/Reflection state.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "content": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "priority": {"type": "integer", "default": 0},
                    "expires_at": {"type": "string", "description": "Optional ISO-8601 datetime."},
                    "decay_policy": {"type": "object", "additionalProperties": True, "default": {"type": "none"}},
                    "source_ref": {"type": "string"},
                    "provenance": {"type": "object", "additionalProperties": True, "default": {}},
                    "open_questions": {"type": "array", "items": {"type": "string"}, "default": []},
                    "links": {"type": "array", "items": drive_link_schema, "default": []},
                },
                "required": ["content", "reason", "strength"],
            },
        },
        {
            "name": "list_drives",
            "description": "List Drives using structured status, priority, strength, and expiry filters. Read-only.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "status": {"type": "array", "items": {"type": "string", "enum": drive_statuses}, "default": ["active"]},
                    "min_strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.3},
                    "min_priority": {"type": "integer", "default": 0},
                    "include_expired": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
            },
        },
        {
            "name": "get_drive",
            "description": "Read one Drive, including provenance, open questions, links, expiry, and effective strength.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {"drive_id": {"type": "string", "minLength": 1}},
                "required": ["drive_id"],
            },
        },
        {
            "name": "update_drive",
            "description": "Explicitly evolve a Drive. Restoring satisfied, abandoned, or superseded to active requires reactivate=true and a reason.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "drive_id": {"type": "string", "minLength": 1},
                    "content": {"type": "string", "minLength": 1}, "reason": {"type": "string", "minLength": 1},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "priority": {"type": "integer"},
                    "status": {"type": "string", "enum": drive_statuses}, "expires_at": {"type": "string"},
                    "decay_policy": {"type": "object", "additionalProperties": True},
                    "open_questions": {"type": "array", "items": {"type": "string"}},
                    "completion_note": {"type": "string"}, "reactivate": {"type": "boolean", "default": False},
                    "reactivation_reason": {"type": "string"},
                },
                "required": ["drive_id"],
                "allOf": [{
                    "if": {"properties": {"reactivate": {"const": True}}, "required": ["reactivate"]},
                    "then": {"required": ["reactivation_reason"]},
                }],
            },
        },
        {
            "name": "link_drive",
            "description": "Add or update an explicit Drive link after validating its target exists.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {"drive_id": {"type": "string", "minLength": 1}, **drive_link_schema["properties"]},
                "required": ["drive_id", "target_type", "target_id", "relation"],
            },
        },
        {
            "name": "review_drives",
            "description": "Read-only Drive hygiene review; never changes Drive state or creates actions.",
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "stale_days": {"type": "integer", "minimum": 1, "default": 30},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                },
            },
        },
    ])
    for tool in TOOLS:
        tool["inputSchema"].setdefault("additionalProperties", False)
    tool_map["search_memory"]["inputSchema"]["properties"]["n"].update(
        {"minimum": 1, "maximum": 50}
    )
    tool_map["search_memory"]["inputSchema"]["properties"]["debug"]["description"] = (
        "Include semantic score, graph/link score, salience, recency, feedback, "
        "state adjustments, final score, and ranking explanation."
    )
    tool_map["set_emotion"]["inputSchema"]["properties"]["score"].update(
        {"minimum": 0.0, "maximum": 1.0}
    )
    tool_map["annotate_memory"]["inputSchema"]["properties"]["text"]["minLength"] = 1
    tool_map["leave_comment"]["inputSchema"]["properties"]["content"]["minLength"] = 1
    tool_map["mark_comments_read"]["inputSchema"]["properties"]["comment_ids"]["minItems"] = 1
    tool_map["search_reflections"]["inputSchema"]["properties"]["status"].update({
        "enum": ["", "draft", "tentative", "adopted", "shaken", "abandoned"]
    })
    trigger_enum = sorted([
        "user_invite", "current_event", "contradiction", "unresolved",
        "repeated_tendency", "curiosity", "event_count", "active_day",
        "random_non_hot", "background_task",
    ])
    tool_map["list_reflection_candidates"]["inputSchema"]["properties"]["trigger_type"]["enum"] = trigger_enum
    tool_map["search_reflections"]["inputSchema"]["properties"]["trigger_type"]["enum"] = [""] + trigger_enum

    read_only_tools = {
        "search_memory", "search_multi", "get_memory", "get_neighbors",
        "get_annotations", "wakeup", "search_annotations",
        "list_reflection_candidates", "draft_reflection", "search_reflections",
        "graph_stats", "get_comments", "get_links", "list_drives", "get_drive", "review_drives",
    }
    destructive_tools = {
        "delete_memory", "dream_pass", "retract_memory", "retract_reflection",
    }
    idempotent_tools = {
        "get_memory", "get_neighbors", "get_annotations", "wakeup",
        "search_annotations", "list_reflection_candidates", "draft_reflection",
        "search_reflections", "graph_stats", "get_comments", "get_links",
        "pin_memory", "unpin_memory", "list_drives", "get_drive", "review_drives",
    }
    for tool in TOOLS:
        name = tool["name"]
        tool["annotations"] = {
            "readOnlyHint": name in read_only_tools,
            "destructiveHint": name in destructive_tools,
            "idempotentHint": name in idempotent_tools,
            "openWorldHint": False,
        }
        # This raw definition is the discovery contract for every transport.
        # Clients can use the version to invalidate a cached tool snapshot.
        tool["_meta"] = {TOOL_SCHEMA_META_KEY: TOOL_SCHEMA_VERSION}

    def handle_tool(name: str, args: dict) -> dict:
        """Execute a tool and return result."""
        def failure(message: str, code: str = "business_error") -> dict:
            return {"ok": False, "error": {"code": code, "message": str(message), "tool": name}}

        try:
            if name == "store_memory":
                confidence = args.get("confidence", 0.5)
                if not 0.0 <= confidence <= 1.0:
                    return failure("confidence must be between 0.0 and 1.0", "validation_error")
                emotion_score = args.get("emotion_score", 0.5)
                if not 0.0 <= emotion_score <= 1.0:
                    return failure("emotion_score must be between 0.0 and 1.0", "validation_error")
                mid = f"mem_{uuid.uuid4().hex[:8]}"
                mem.store(
                    memory_id=mid,
                    text=args["text"],
                    tag=args.get("tag", "general"),
                    tier=args.get("tier", "long"),
                    emotion_score=emotion_score,
                    connect_to=args.get("connect_to"),
                    context=args.get("context", ""),
                    perspective=args.get("perspective", "mixed"),
                    epistemic_status=args.get("epistemic_status", "reported"),
                    confidence=args.get("confidence", 0.5),
                    source_turn=args.get("source_turn", ""),
                    supersedes=args.get("supersedes", ""),
                    memory_layer=args.get("memory_layer", "event"),
                    source_ref=args.get("source_ref", ""),
                    provenance=args.get("provenance", {}),
                    salience=args.get("salience", 0.5),
                    motifs=args.get("motifs", []),
                    state=args.get("state", "active"),
                    unresolved=args.get("unresolved", False),
                    open_questions=args.get("open_questions", []),
                    contradicted_by=args.get("contradicted_by", []),
                )
                return {"memory_id": mid, "status": "stored"}

            elif name == "search_memory":
                results = mem.search(
                    query=args["query"],
                    n_results=args.get("n", 5),
                    tag=args.get("tag"),
                    associate=args.get("associate", True),
                    hebbian=False,
                    no_cite=True,
                    debug=args.get("debug", False),
                )
                return {"memories": results}

            elif name == "search_multi":
                results = mem.search_multi(
                    queries=args["queries"],
                    n_results_per_query=args.get("n_results_per_query", 5),
                    n_total=args.get("n_total"),
                    tag=args.get("tag"),
                    associate=args.get("associate", True),
                    hebbian=False,
                    no_cite=True,
                    include_context=args.get("include_context", False),
                )
                return {"memories": results}

            elif name == "connect_memories":
                mem.db.connect(
                    args["source_id"],
                    args["target_id"],
                    weight=args.get("weight", 2.0),
                    relation=args.get("relation", "related"),
                )
                return {"status": "connected"}

            elif name == "get_neighbors":
                neighbors = mem.db.get_neighbors(
                    args["memory_id"],
                    min_weight=args.get("min_weight", 0.5),
                    limit=args.get("limit", 5),
                )
                return {"neighbors": [dict(n) for n in neighbors]}

            elif name == "get_links":
                return mem.db.get_links(
                    args["memory_id"], direction=args.get("direction", "both"),
                    min_weight=args.get("min_weight", 0.0), limit=args.get("limit", 100),
                )

            elif name == "get_memory":
                row = mem.db.get(args["memory_id"])
                reflections = []
                if row and row.get("memory_layer") == "event":
                    reflections = mem.db.get_reflections_for_event(args["memory_id"])
                return {"memory": row, "reflections": reflections}

            elif name in {"update_memory_metadata", "update_memory_recall_state"}:
                recall_fields = {"salience", "motifs", "state", "unresolved", "open_questions"}
                if not recall_fields.intersection(args):
                    return failure("at least one recall-state field is required", "validation_error")
                return {
                    "status": "updated",
                    "memory": mem.update_recall_state(
                        args["memory_id"], salience=args.get("salience"),
                        motifs=args.get("motifs"), state=args.get("state"),
                        unresolved=args.get("unresolved"),
                        open_questions=args.get("open_questions"),
                    ),
                }

            elif name == "reconcile_recall_metadata":
                return mem.reconcile_recall_metadata(
                    dry_run=args.get("dry_run", True),
                    maintenance_id=args.get("maintenance_id", ""),
                    max_candidates=args.get("max_candidates", 100),
                )

            elif name == "retract_memory":
                ok = mem.retract(
                    args["memory_id"], args.get("superseding_memory_id", "")
                )
                return {"status": "retracted"} if ok else failure("memory not found", "not_found")

            elif name == "reconcile":
                return mem.reconcile(repair=args.get("repair", False))

            elif name == "create_drive":
                return drives.create(**args)

            elif name == "list_drives":
                return drives.list(**args)

            elif name == "get_drive":
                return drives.get(args["drive_id"])

            elif name == "update_drive":
                return drives.update(args.pop("drive_id"), **args)

            elif name == "link_drive":
                return drives.link(args.pop("drive_id"), **args)

            elif name == "review_drives":
                return drives.review(**args)

            elif name == "delete_memory":
                success = mem.delete(args["memory_id"])
                return {"status": "deleted"} if success else failure("memory not found or deletion failed", "not_found")

            elif name == "dream_pass":
                return mem.dream_pass(
                    short_decay_days=args.get("short_decay_days", 14),
                    edge_decay_factor=args.get("edge_decay_factor", 0.9),
                    strong_edge_decay_factor=args.get("strong_edge_decay_factor", 0.95),
                    emotion_nudge=args.get("emotion_nudge", 0.05),
                    auto_discover=args.get("auto_discover", False),
                    dry_run=args.get("dry_run", True),
                    maintenance_id=args.get("maintenance_id", ""),
                    split_bundled=args.get("split_bundled", False),
                )

            elif name == "set_emotion":
                if not mem.db.set_emotion_score(args["memory_id"], args["score"]):
                    return failure("memory not found", "not_found")
                return {"status": "updated"}

            elif name == "set_tier":
                if not mem.db.set_tier(args["memory_id"], args["tier"]):
                    return failure("memory not found", "not_found")
                return {"status": "updated"}

            elif name == "graph_stats":
                total = mem.count()
                all_mems = mem.db.list_all(limit=total)
                tags = {}
                tiers = {}
                for m in all_mems:
                    tags[m.get("tag", "unknown")] = tags.get(m.get("tag", "unknown"), 0) + 1
                    tiers[m.get("tier", "unknown")] = tiers.get(m.get("tier", "unknown"), 0) + 1
                return {
                    "total_memories": total,
                    "tags": tags,
                    "tiers": tiers,
                }

            elif name == "annotate_memory":
                aid = mem.db.annotate(args["memory_id"], args["text"])
                return {"annotation_id": aid, "status": "annotated"}

            elif name == "get_annotations":
                anns = mem.db.get_annotations(args["memory_id"])
                return {"annotations": anns}

            elif name == "consolidate":
                result = mem.consolidate(args["conversation_text"])
                if result.get("status") == "disabled":
                    return failure(result.get("reason", "consolidation disabled"), "disabled")
                return result

            elif name == "store_visual":
                mid = f"vis_{uuid.uuid4().hex[:8]}"
                mem.store(
                    memory_id=mid,
                    text=args["text"],
                    tag=args.get("tag", "visual"),
                    tier="long",
                    emotion_score=0.3,
                    connect_to=args.get("connect_to"),
                )
                if args.get("visual_embedding"):
                    mem.db.set_visual_embedding(mid, args["visual_embedding"])
                return {"memory_id": mid, "status": "stored"}

            elif name == "wakeup":
                result = mem.db.wakeup(
                    n_high_emotion=args.get("n_high_emotion", 5),
                    n_random=args.get("n_random", 2),
                    high_emotion_days=args.get("high_emotion_days", 3),
                    n_recent=args.get("n_recent", 5),
                    n_salient=args.get("n_salient", 3),
                    n_unresolved=args.get("n_unresolved", 3),
                    n_identity=args.get("n_identity", 5),
                    debug=args.get("debug", False),
                )
                # Pinned file layer — session_state (rolling state), timeline
                # (event ledger), tail (previous window, mechanical). Present
                # only when the files exist; MCP-only setups get the same
                # bridges as proxy setups, minus per-turn mechanics.
                for key, fname in (("session_state", anchor_pinned.SESSION_STATE),
                                   ("recent_timeline", anchor_pinned.RECENT_TIMELINE),
                                   ("last_session", anchor_pinned.LAST_SESSION)):
                    text = anchor_pinned.read_file(pinned_dir, fname)
                    if text:
                        result[key] = text
                reflection_bundle = mem.db.wakeup_reflections(
                    limit=args.get("n_reflections", 2),
                    include_drafts=args.get("include_draft_reflections", False),
                )
                result["recent_reflections"] = reflection_bundle["items"]
                result["reflection_policy"] = reflection_bundle["policy"]
                return result

            elif name == "write_session_state":
                archived = anchor_pinned.write_session_state(pinned_dir, args["content"])
                out = {"status": "written", "path": os.path.join(pinned_dir, anchor_pinned.SESSION_STATE)}
                if archived:
                    out["archived_previous"] = archived
                return out

            elif name == "leave_comment":
                cid = mem.db.insert_comment(
                    memory_id=args["memory_id"],
                    content=args["content"],
                    author=args.get("author", "ai"),
                    reply_to=args.get("reply_to"),
                )
                return {"comment_id": cid, "status": "inserted"}

            elif name == "get_comments":
                rows = mem.db.get_comments(args["memory_id"])
                return {"comments": [dict(r) for r in rows]}

            elif name == "mark_comments_read":
                count = mem.db.mark_comments_read(
                    args["comment_ids"],
                    reader=args.get("reader", "ai"),
                )
                return {"status": "marked", "count": count}

            elif name == "pin_memory":
                mem.db.pin(args["memory_id"])
                return {"status": "pinned", "memory_id": args["memory_id"]}

            elif name == "unpin_memory":
                mem.db.unpin(args["memory_id"])
                return {"status": "unpinned", "memory_id": args["memory_id"]}

            elif name == "search_annotations":
                rows = mem.db.search_annotations(args["query"], limit=args.get("limit", 5))
                return {"results": [dict(r) for r in rows]}

            elif name == "cite_memory":
                if not mem.db.cite(args["memory_id"]):
                    return failure("memory not found", "not_found")
                return {"status": "cited", "memory_id": args["memory_id"]}

            elif name == "list_reflection_candidates":
                return reflection.list_candidates(
                    trigger_type=args.get("trigger_type", "user_invite"),
                    limit=args.get("limit", 5),
                    random_seed=args.get("random_seed"),
                )

            elif name == "record_memory_feedback":
                return mem.db.record_memory_feedback(
                    args["memory_id"], args["relevant"],
                    args.get("reason", ""), args.get("source_ref", ""),
                )

            elif name == "draft_reflection":
                return reflection.draft(**args)

            elif name == "save_reflection":
                return reflection.save(
                    args["draft"], args.get("expected_draft_token", "")
                )

            elif name == "search_reflections":
                return {"reflections": mem.db.search_reflections(**args)}

            elif name == "record_reflection_effect":
                return mem.db.record_reflection_effect(
                    decision_id=args["decision_id"],
                    reflection_ids=args["used_reflection_ids"],
                    input_summary=args["input_summary"],
                    output_summary=args["output_summary"],
                    effect_note=args["effect_note"],
                    outcome_status=args.get("outcome_status", "unknown"),
                    provenance=args.get("provenance", {}),
                )

            elif name == "append_reflection_evidence":
                return mem.db.append_reflection_evidence(
                    args["reflection_id"], args["kind"], args["content"],
                    args.get("source_ref", ""),
                )

            elif name == "retract_reflection":
                ok = mem.db.retract_reflection(
                    args["reflection_id"], args.get("retracted_by", "")
                )
                return {"status": "retracted"} if ok else failure("reflection not found", "not_found")

            else:
                return failure(f"Unknown tool: {name}", "unknown_tool")

        except Exception as e:
            return failure(str(e))

    return TOOLS, handle_tool, mem


def run_stdio(db_path: str, pinned_dir: str = None):
    """Run MCP server over stdio (standard MCP transport)."""
    tools, handle_tool, mem = create_server(db_path, pinned_dir=pinned_dir)

    def send(msg):
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def read():
        line = sys.stdin.readline()
        if not line:
            return None
        return json.loads(line.strip())

    # MCP initialization
    while True:
        msg = read()
        if msg is None:
            break

        method = msg.get("method", "")
        id_ = msg.get("id")

        if method == "initialize":
            send({
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "anchor-memory",
                        "version": SERVER_VERSION,
                    }
                }
            })

        elif method == "notifications/initialized":
            pass  # Client acknowledges init

        elif method == "tools/list":
            send({
                "jsonrpc": "2.0",
                "id": id_,
                "result": {"tools": tools}
            })

        elif method == "tools/call":
            tool_name = msg["params"]["name"]
            tool_args = msg["params"].get("arguments", {})
            result = handle_tool(tool_name, tool_args)
            is_error = bool(result.get("ok") is False or result.get("error"))
            send({
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                    "structuredContent": result,
                    "isError": is_error,
                }
            })

        elif method == "ping":
            send({"jsonrpc": "2.0", "id": id_, "result": {}})

        else:
            send({
                "jsonrpc": "2.0",
                "id": id_,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            })


def format_wakeup_text(data: dict) -> str:
    """Format a wakeup() dict as plain text, for hook/prompt injection.

    Used by --wakeup-text: clients with lifecycle hooks (e.g. Claude Code
    SessionStart) can inject this block mechanically instead of relying on
    the model calling the wakeup tool. Empty sections are omitted.
    """
    lines = []

    def file_section(title, text):
        if text:
            lines.append(f"## {title}")
            lines.append(text)
            lines.append("")

    file_section("Session state (your own rolling state)", data.get("session_state"))
    file_section("Previous window (mechanical tail)", data.get("last_session"))
    file_section("Recent timeline", data.get("recent_timeline"))

    def section(title, items, fmt):
        if not items:
            return
        lines.append(f"## {title}")
        for it in items:
            lines.append(fmt(it))
        lines.append("")

    section("Identity / confirmed Core", data.get("identity_snapshot", []),
            lambda m: f"- [{m['memory_id']}] {m['text']}")
    section("Pinned", data.get("pinned", []),
            lambda m: f"- [{m['memory_id']}] {m['text']}")
    section("Recent (newest first)", data.get("recent", []),
            lambda m: f"- [{m['memory_id']}] ({m['timestamp'][:10]}) {m['text']}")
    section("High-salience active memories", data.get("salient", []),
            lambda m: f"- [{m['memory_id']}] (salience {m['salience']:.2f}; "
                      f"motifs: {', '.join(m.get('motifs', [])) or 'none'}) {m['text']}")
    section("Unresolved questions", data.get("unresolved", []),
            lambda m: f"- [{m['memory_id']}] "
                      f"{'; '.join(m.get('open_questions', [])) or m['text']}")
    section("Recall hints (use only when current context warrants recall)",
            data.get("recall_hints", []), lambda hint: f"- {hint}")
    section("Recent high-emotion", data.get("high_emotion", []),
            lambda m: f"- [{m['memory_id']}] (emotion {m['emotion_score']:.2f}) {m['text']}")
    section("Random old", data.get("random_old", []),
            lambda m: f"- [{m['memory_id']}] ({m['timestamp'][:10]}) {m['text']}")
    section("Unread comments", data.get("unread_comments", []),
            lambda c: f"- [{c['comment_id']}] on [{c['memory_id']}] "
                      f"({c['author']}, {c['created_at'][:10]}): {c['content']}")
    section("Reviewed recent Reflections", data.get("recent_reflections", []),
            lambda r: f"- [{r['reflection_id']}] ({r['status']}; confidence "
                      f"{r['confidence']:.2f}) {r['current_interpretation']}")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anchor Memory MCP Server")
    parser.add_argument("--db-path", default="./anchor_data", help="Path to store memory data")
    parser.add_argument("--pinned-dir", default=None,
                        help="Pinned file layer directory (default: <db-path>/pinned)")
    parser.add_argument("--wakeup-text", action="store_true",
                        help="Print wakeup() as plain text and exit (for session-start hooks). "
                             "SQLite-only fast path — no embedder load. Does not start the server.")
    args = parser.parse_args()

    os.makedirs(args.db_path, exist_ok=True)
    pinned = args.pinned_dir or os.path.join(args.db_path, "pinned")

    if args.wakeup_text:
        from anchor_db import AnchorDB
        db = AnchorDB(os.path.join(args.db_path, "memories.db"))
        data = db.wakeup()
        reflection_bundle = db.wakeup_reflections(limit=2, include_drafts=False)
        data["recent_reflections"] = reflection_bundle["items"]
        data["reflection_policy"] = reflection_bundle["policy"]
        for key, fname in (("session_state", anchor_pinned.SESSION_STATE),
                           ("recent_timeline", anchor_pinned.RECENT_TIMELINE),
                           ("last_session", anchor_pinned.LAST_SESSION)):
            text = anchor_pinned.read_file(pinned, fname)
            if text:
                data[key] = text
        print(format_wakeup_text(data))
        sys.exit(0)

    run_stdio(args.db_path, pinned_dir=args.pinned_dir)
