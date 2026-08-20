# ChatGPT remote MCP checklist

1. Finish HTTPS and MCP OAuth 2.1 tests first; never select no authentication for this
   private-memory service.
2. Enable Developer mode in ChatGPT on the web.
3. Create a developer-mode app pointing to `https://<domain>/mcp`.
4. Test `wakeup`, `search_memory`, `get_memory`, and `search_reflections` before enabling write tools.
5. Store one explicit test memory with perspective, epistemic status,
   confidence, and source turn, then inspect it by ID.
6. Retract the test memory and verify the audit record remains.
7. Restart containers and verify persistence.
8. Verify ordinary conversation is not stored automatically and assistant
   hypotheses are not promoted to confirmed user facts.
9. Exercise the neutral Reflection fixture: candidate → draft
   (`persisted=false`) → explicit save → paired event/Reflection retrieval →
   explicit decision link → append counterevidence.

OpenAI's current production guidance requires stable HTTPS Streamable HTTP and,
for private data or user actions, the MCP OAuth authorization flow:

- https://developers.openai.com/plugins/concepts/mcp-server
- https://developers.openai.com/plugins/build/auth

The static Bearer guard in `anchor_http.py` is not the final ChatGPT connection
mechanism.
