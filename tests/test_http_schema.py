import asyncio
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from anchor_db import AnchorDB
from anchor_mcp import create_server


MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None


class FakeDB:
    def list_all(self, limit=1):
        return []


class FakeCollection:
    def count(self):
        return 0


class FakeMemory:
    db = FakeDB()
    _collection = FakeCollection()


class ManifestMemory:
    def __init__(self, db_path):
        os.makedirs(db_path, exist_ok=True)
        self.db = AnchorDB(os.path.join(db_path, "memories.db"))

    def update_recall_state(self, memory_id, **metadata):
        return self.db.update_recall_state(memory_id, **metadata)


@unittest.skipUnless(MCP_AVAILABLE, "MCP SDK is only installed in the HTTP test environment")
class HttpSchemaTests(unittest.TestCase):
    def test_reflection_adapter_does_not_inject_invoke_and_raises_tool_error(self):
        import anchor_http
        from mcp.server.fastmcp.exceptions import ToolError

        calls = []

        def fake_create_server(db_path, pinned_dir=None):
            def handle(name, payload):
                calls.append((name, payload))
                if name == "search_reflections":
                    return {"ok": False, "error": {"code": "business_error", "message": "rejected"}}
                return {"tool": name}
            return [], handle, FakeMemory()

        with patch.object(anchor_http, "create_server", fake_create_server):
            server = anchor_http.create_http_server("unused")
            asyncio.run(server.call_tool("draft_reflection", {
                "source_event_ids": ["event"], "trigger_type": "user_invite",
                "selection_reason": "reason", "previous_interpretation": "old",
                "current_interpretation": "new", "change_or_tension": "changed",
                "confidence": 0.5, "open_questions": [], "counterevidence": [],
                "provenance": {
                    "model": "test-model", "thread_id": "test-thread",
                    "context_ref": "turns:1-2", "extractor_version": "test-v1",
                },
            }))
            with self.assertRaises(ToolError):
                asyncio.run(server.call_tool("search_reflections", {}))

        draft_payload = calls[0][1]
        self.assertNotIn("invoke", draft_payload)
        self.assertNotIn("handle", draft_payload)

    def test_fastmcp_registers_reflection_tools_and_annotations(self):
        import anchor_http

        def fake_create_server(db_path, pinned_dir=None):
            def handle(name, payload):
                if name == "search_reflections":
                    return {"ok": False, "error": {"code": "business_error", "message": "rejected"}}
                return {"tool": name}
            return [], handle, FakeMemory()

        with patch.object(anchor_http, "create_server", fake_create_server):
            server = anchor_http.create_http_server("unused")
            tools = asyncio.run(server.list_tools())

        by_name = {tool.name: tool for tool in tools}
        expected = {
            "list_reflection_candidates", "draft_reflection", "save_reflection",
            "search_reflections", "record_reflection_effect",
            "append_reflection_evidence", "retract_reflection",
            "record_memory_feedback", "update_memory_metadata", "get_links",
            "pin_memory", "unpin_memory", "reconcile_recall_metadata",
        }
        self.assertTrue(expected.issubset(by_name))
        self.assertTrue(by_name["draft_reflection"].annotations.readOnlyHint)
        self.assertFalse(by_name["save_reflection"].annotations.readOnlyHint)
        self.assertFalse(by_name["record_memory_feedback"].annotations.readOnlyHint)
        self.assertTrue(by_name["retract_reflection"].annotations.destructiveHint)
        self.assertIn("source_event_ids", by_name["draft_reflection"].inputSchema["properties"])
        draft_schema = by_name["draft_reflection"].inputSchema
        self.assertIn("user_invite", draft_schema["properties"]["trigger_type"]["enum"])
        provenance_ref = draft_schema["properties"]["provenance"]["$ref"]
        provenance_name = provenance_ref.rsplit("/", 1)[-1]
        self.assertEqual(
            ["model", "thread_id", "context_ref", "extractor_version"],
            draft_schema["$defs"][provenance_name]["required"],
        )

    def test_http_tools_list_uses_authoritative_raw_schema(self):
        import anchor_http

        authoritative = {
            "type": "object", "additionalProperties": False,
            "properties": {"text": {"type": "string", "minLength": 1}},
            "required": ["text"],
        }
        description = "Authoritative description"
        annotations = {
            "readOnlyHint": False, "destructiveHint": False,
            "idempotentHint": True, "openWorldHint": False,
        }
        meta = {"anchor/schema_version": "test-version"}

        def fake_create_server(db_path, pinned_dir=None):
            return ([{
                        "name": "store_memory", "inputSchema": authoritative,
                        "description": description, "annotations": annotations,
                        "_meta": meta,
                    }],
                    lambda name, payload: {"tool": name}, FakeMemory())

        with patch.object(anchor_http, "create_server", fake_create_server):
            server = anchor_http.create_http_server("unused")
            by_name = {tool.name: tool for tool in asyncio.run(server.list_tools())}

        self.assertEqual(authoritative, by_name["store_memory"].inputSchema)
        self.assertEqual(description, by_name["store_memory"].description)
        self.assertTrue(by_name["store_memory"].annotations.idempotentHint)
        self.assertEqual(meta, by_name["store_memory"].meta)

    def test_every_http_tool_uses_the_stdio_discovery_contract(self):
        import anchor_http

        fake_module = types.ModuleType("anchor_memory")
        fake_module.AnchorMemory = ManifestMemory
        with tempfile.TemporaryDirectory() as tempdir, patch.dict(
            sys.modules, {"anchor_memory": fake_module}
        ):
            raw_tools, handle, memory = create_server(tempdir)
            with patch.object(
                anchor_http, "create_server",
                return_value=(raw_tools, handle, memory),
            ):
                server = anchor_http.create_http_server(tempdir)
                http_tools = asyncio.run(server.list_tools())

        raw_by_name = {tool["name"]: tool for tool in raw_tools}
        http_by_name = {tool.name: tool for tool in http_tools}
        # HTTP intentionally exposes a safer remote subset of legacy stdio
        # maintenance tools. Every remote tool must still come from the raw
        # authoritative manifest and match it field-for-field.
        self.assertTrue(set(http_by_name).issubset(raw_by_name))
        self.assertIn("wakeup", http_by_name)
        for name, exposed in http_by_name.items():
            raw = raw_by_name[name]
            self.assertEqual(raw["description"], exposed.description, name)
            self.assertEqual(raw["inputSchema"], exposed.inputSchema, name)
            self.assertEqual(raw["_meta"], exposed.meta, name)
            for hint, expected in raw["annotations"].items():
                self.assertEqual(expected, getattr(exposed.annotations, hint), name)

    def test_streamable_http_protocol_health_and_auth(self):
        import anchor_http
        from starlette.testclient import TestClient

        def fake_create_server(db_path, pinned_dir=None):
            def handle(name, payload):
                if name == "search_reflections":
                    return {"ok": False, "error": {"code": "business_error", "message": "rejected"}}
                return {"tool": name}
            return [], handle, FakeMemory()

        environment = {
            "ANCHOR_AUTH_TOKEN": "http-test-token",
            "ANCHOR_DB_PATH": "unused",
            "ANCHOR_HOST": "0.0.0.0",
            "ANCHOR_PORT": "8000",
        }
        with patch.object(anchor_http, "create_server", fake_create_server), patch.dict(
            os.environ, environment, clear=False
        ):
            app = anchor_http.create_app()
            with TestClient(app, base_url="http://127.0.0.1:8000") as client:
                self.assertEqual(200, client.get("/healthz").status_code)
                self.assertEqual(200, client.get("/readyz").status_code)
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                }
                initialize = {
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26", "capabilities": {},
                        "clientInfo": {"name": "http-test", "version": "1"},
                    },
                }
                self.assertEqual(
                    401, client.post("/mcp", headers=headers, json=initialize).status_code
                )
                headers["Authorization"] = "Bearer http-test-token"
                response = client.post("/mcp", headers=headers, json=initialize)
                self.assertEqual(200, response.status_code)
                self.assertIn("serverInfo", response.json()["result"])
                tools_response = client.post(
                    "/mcp", headers=headers,
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                )
                self.assertEqual(200, tools_response.status_code)
                names = {tool["name"] for tool in tools_response.json()["result"]["tools"]}
                self.assertIn("save_reflection", names)
                draft_arguments = {
                    "source_event_ids": ["event-http"],
                    "trigger_type": "user_invite",
                    "selection_reason": "HTTP schema test",
                    "previous_interpretation": "old",
                    "current_interpretation": "new",
                    "change_or_tension": "no_change",
                    "confidence": 0.5,
                    "open_questions": [],
                    "counterevidence": [],
                    "provenance": {
                        "model": "test", "thread_id": "test",
                        "context_ref": "test", "extractor_version": "test",
                    },
                }
                draft_call = client.post(
                    "/mcp", headers=headers,
                    json={
                        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "draft_reflection", "arguments": draft_arguments},
                    },
                )
                self.assertEqual(200, draft_call.status_code)
                self.assertFalse(draft_call.json()["result"]["isError"])
                save_call = client.post(
                    "/mcp", headers=headers,
                    json={
                        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "save_reflection", "arguments": {"draft": draft_arguments}},
                    },
                )
                self.assertEqual(200, save_call.status_code)
                self.assertFalse(save_call.json()["result"]["isError"])
                failed_call = client.post(
                    "/mcp", headers=headers,
                    json={
                        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                        "params": {"name": "search_reflections", "arguments": {}},
                    },
                )
                self.assertEqual(200, failed_call.status_code)
                self.assertTrue(failed_call.json()["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
