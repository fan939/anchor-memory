import importlib.util
import os
import unittest
from unittest.mock import patch


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


@unittest.skipUnless(MCP_AVAILABLE, "MCP SDK is only installed in the HTTP test environment")
class OAuthHttpTests(unittest.TestCase):
    def test_oauth_metadata_and_unauthorized_challenge(self):
        import anchor_http
        from starlette.testclient import TestClient

        def fake_create_server(db_path, pinned_dir=None):
            return [], lambda name, payload: {"tool": name}, FakeMemory()

        environment = {
            "ANCHOR_AUTH_MODE": "oauth",
            "ANCHOR_DB_PATH": "unused",
            "ANCHOR_HOST": "0.0.0.0",
            "ANCHOR_PORT": "8000",
            "ANCHOR_PUBLIC_BASE_URL": "https://memory.example.com",
            "ANCHOR_DOMAIN": "memory.example.com",
            "ANCHOR_OAUTH_ISSUER": "https://tenant.example.auth0.com/",
            "ANCHOR_OAUTH_RESOURCE": "https://memory.example.com/mcp",
        }
        with patch.object(anchor_http, "create_server", fake_create_server), patch.dict(
            os.environ, environment, clear=False
        ):
            app = anchor_http.create_app()
            with TestClient(app, base_url="https://memory.example.com") as client:
                metadata = client.get("/.well-known/oauth-protected-resource/mcp")
                self.assertEqual(200, metadata.status_code)
                self.assertEqual(
                    "https://memory.example.com/mcp", metadata.json()["resource"]
                )
                self.assertEqual(
                    ["https://tenant.example.auth0.com/"],
                    metadata.json()["authorization_servers"],
                )
                self.assertEqual(
                    {"anchor:read", "anchor:write", "anchor:admin"},
                    set(metadata.json()["scopes_supported"]),
                )

                response = client.post(
                    "/mcp",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "oauth-test", "version": "1"},
                        },
                    },
                )
                self.assertEqual(401, response.status_code)
                challenge = response.headers.get("WWW-Authenticate", "")
                self.assertIn("oauth-protected-resource/mcp", challenge)


if __name__ == "__main__":
    unittest.main()
