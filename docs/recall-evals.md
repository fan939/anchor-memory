# Recall regression cases

These cases guard the boundary between retrieval and interpretation. Passing
them does not mean Anchor can infer a user's inner state; it means retrieval
surfaces relevant evidence without silently rewriting durable memory.

| Input | Expected behavior | Automated boundary |
| --- | --- | --- |
| `周日回家吃饭` | Prefer relevant family context over generic Sunday/text overlap. | Deterministic ranking test checks family context wins with salience and unresolved evidence. |
| `算了` | Treat as an ambiguous current-turn reply, not a durable abandonment judgment. | MCP instructions prohibit durable overwrite without confirmation. |
| `我没事` | Do not erase or supersede known long-term context. | MCP instructions keep short ambiguous replies as current-turn evidence only. |
| `你还记得吗` | Search first; never pretend to remember. | MCP instructions require search before a memory claim and explicit absence reporting. |

The first case exercises backend ranking. The other three are model/tool-use
contract tests because ordinary retrieval is read-only and cannot itself decide
what the user's short reply means. End-to-end model evaluations should reuse
these prompts after any instruction or tool-manifest change.
