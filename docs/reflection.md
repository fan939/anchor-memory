# Reflection: evidence, reinterpretation, and decision provenance

Reflection is a separate append-only interpretation layer. Event Seeds record
what was observed or reported; Reflections record how one or more events were
later understood. Neither record silently overwrites the other.

## Data model

- `memories.memory_layer`: `core`, `dynamic`, or `event`.
- `reflections`: interpretation, trigger, confidence, status, open questions,
  cooldown, provenance, and optional `supersedes` link.
- `reflection_sources`: ordered foreign-key links to real event memories,
  including source status at creation.
- `reflection_evidence`: append-only support, counterevidence, and outcomes.
- `decisions`: short auditable decision summaries and provenance.
- `reflection_effects`: explicit many-to-many links between decisions and the
  Reflections actually used.

Hard deletion of an Event Seed referenced by a Reflection is rejected. Retract
or supersede the event instead; Reflection retrieval then reports
`source_integrity: retracted_source` while retaining the original audit chain.

## Provenance

Memory records accept `source_ref` and a structured `provenance` object.
Reflection drafts and decision links require:

```json
{
  "model": "model or runtime identifier",
  "thread_id": "stable window/thread identifier",
  "context_ref": "turn range or protected context locator",
  "extractor_version": "prompt/tool/curator version"
}
```

Do not put tokens, Authorization headers, private memory bodies, or full chat
transcripts in provenance. Use stable protected locators.

## Explicit minimum workflow

1. Call `list_reflection_candidates`. A trigger opens an opportunity; it does
   not require review and never writes a Reflection.
2. Call `draft_reflection` with source Event Seed IDs, old/current
   interpretation, explicit change or `no_change`, counterevidence, unresolved
   questions, confidence, and provenance. The result has `persisted: false` and
   a content digest.
3. Review the draft, then call `save_reflection`. Source IDs and the draft digest
   are revalidated before the append-only write.
4. Search the event or call `search_reflections`; both event and interpretation
   are returned with separate states.
5. If a later decision actually used the Reflection, call
   `record_reflection_effect` with explicit `used_reflection_ids`. Do not infer
   influence retrospectively.
6. Append later support, counterevidence, or outcomes with
   `append_reflection_evidence`. Use `retract_reflection` to abandon an
   interpretation without deleting its audit history.

`ANCHOR_REFLECTION_AUTO_SAVE=false` is the supported default. There is no
background autonomous writer in the first implementation.

The legacy proxy curator is also disabled by default with
`ANCHOR_AUTO_CURATE=false`. If explicitly enabled, it may only create event
records with conservative status and provenance; it cannot create a Reflection,
mark a user fact confirmed, or promote a record to Core.

## Candidate triggers and rumination guard

Supported triggers include user invitation, current evidence, contradiction,
unresolved questions, repeated tendency, curiosity, event count, active days,
reproducible random non-hot sampling, and an authorized background task.

The initial thresholds are configurable:

```env
ANCHOR_REFLECTION_EVENT_THRESHOLD=5
ANCHOR_REFLECTION_ACTIVE_DAY_THRESHOLD=7
ANCHOR_REFLECTION_COOLDOWN_DAYS=14
ANCHOR_REFLECTION_RANDOM_PROBABILITY=0.05
ANCHOR_REFLECTION_RANDOM_BUDGET=1
ANCHOR_TIMEZONE=Asia/Shanghai
```

Event count and active-day triggers merely surface candidates. Active days use
`ANCHOR_TIMEZONE`; inactive calendar days create nothing. Cooldown is respected
except for an explicit user invitation, contradiction, or current evidence.
Random selection is only performed when that trigger is explicitly requested
and accepts a deterministic seed for testing.

## Migration

Dry-run is the default and opens an existing SQLite database read-only:

```sh
python scripts/migrate_reflection.py --db-path /data/anchor
```

Apply requires evidence of an existing backup path:

```sh
python scripts/migrate_reflection.py \
  --db-path /data/anchor \
  --apply \
  --backup /backups/anchor-TIMESTAMP.tar.gz
```

The migration adds tables and conservative columns. It does not create
Reflections from old data and does not mark old records confirmed. Existing
`tier=core` rows are classified as the `core` layer for compatibility, while
their prior epistemic status is preserved unchanged.

## Threat model and current limits

- Reflection text can be wrong, persuasive, repetitive, or injected through a
  compromised source. Treat it as evidence with provenance, never authority.
- High emotion and frequent retrieval do not increase epistemic confidence.
- Ordinary MCP search is read-only: it does not cite, strengthen edges, or
  create Reflections.
- Relevance feedback is append-only and only applies a bounded ranking penalty
  (one percentage point per net negative, capped at ten points). A negative
  report never deletes, retracts, or rewrites the memory.
- The first implementation uses explainable rules and SQLite text search; it
  does not train a latent trigger or run an autonomous background loop.
- `event_count` currently counts event-layer records, not a learned
  "importance" score. Threshold quality requires later evaluation.
- Core candidates are not automatically promoted. A core write requires
  `epistemic_status=confirmed`, and user-facing facts still require explicit
  confirmation by the caller.
