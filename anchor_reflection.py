"""Reflection service: explicit, auditable reinterpretation of event memories.

Drafting is pure and does not touch storage. Saving revalidates every source ID.
The service deliberately contains no LLM call and no background loop: the MCP
caller supplies an interpretation, then explicitly chooses whether to save it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


TRIGGER_TYPES = {
    "user_invite", "current_event", "contradiction", "unresolved",
    "repeated_tendency", "curiosity", "event_count", "active_day",
    "random_non_hot", "background_task",
}
REFLECTION_STATUSES = {"draft", "tentative", "adopted", "shaken", "abandoned"}
PROVENANCE_KEYS = {"model", "thread_id", "context_ref", "extractor_version"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class ReflectionService:
    def __init__(self, db):
        self.db = db
        self.auto_save = os.getenv("ANCHOR_REFLECTION_AUTO_SAVE", "false").lower() == "true"
        self.event_threshold = max(1, int(os.getenv("ANCHOR_REFLECTION_EVENT_THRESHOLD", "5")))
        self.active_day_threshold = max(1, int(os.getenv("ANCHOR_REFLECTION_ACTIVE_DAY_THRESHOLD", "7")))
        self.cooldown_days = max(0, int(os.getenv("ANCHOR_REFLECTION_COOLDOWN_DAYS", "14")))
        self.random_probability = min(
            1.0, max(0.0, float(os.getenv("ANCHOR_REFLECTION_RANDOM_PROBABILITY", "0.05")))
        )
        self.random_budget = max(
            0, int(os.getenv("ANCHOR_REFLECTION_RANDOM_BUDGET", "1"))
        )
        self.timezone_name = os.getenv("ANCHOR_TIMEZONE", "UTC")
        if self.timezone_name.upper() in {"UTC", "ETC/UTC"}:
            # UTC is built into Python and must work even in minimal Windows
            # runtimes that do not bundle the IANA timezone database.
            self.active_timezone = timezone.utc
        else:
            try:
                self.active_timezone = ZoneInfo(self.timezone_name)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Unknown ANCHOR_TIMEZONE: {self.timezone_name}") from exc

    @staticmethod
    def _validate_provenance(provenance: dict) -> dict:
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        missing = sorted(key for key in PROVENANCE_KEYS if not provenance.get(key))
        if missing:
            raise ValueError("provenance is missing: " + ", ".join(missing))
        return dict(provenance)

    @staticmethod
    def _normalize_counterevidence(items: list | None) -> list:
        result = []
        for item in items or []:
            if isinstance(item, str):
                content = item.strip()
                source_ref = ""
            elif isinstance(item, dict):
                content = str(item.get("content", "")).strip()
                source_ref = str(item.get("source_ref", "")).strip()
            else:
                raise ValueError("counterevidence items must be strings or objects")
            if content:
                result.append({"content": content, "source_ref": source_ref})
        return result

    def list_candidates(
        self, trigger_type: str = "user_invite", limit: int = 5,
        random_seed: int | None = None,
    ) -> dict:
        if trigger_type not in TRIGGER_TYPES:
            raise ValueError(f"Unsupported trigger_type: {trigger_type}")
        now = _now()
        # The unresolved path intentionally applies its stricter eligibility at
        # the database boundary, before cooldown/event-count/text reasoning.
        # Other triggers retain the broader event-review candidate pool.
        rows = self.db.list_reflection_candidate_rows(
            limit=500,
            unresolved_only=(trigger_type == "unresolved"),
        )
        opportunity = self.db.reflection_opportunity_stats()
        latest_text = opportunity.get("latest_reflection_at", "")
        latest = self._parse_utc(latest_text) if latest_text else None
        active_dates = set()
        for row in rows:
            timestamp = self._parse_utc(row.get("timestamp", ""))
            if timestamp and (latest is None or timestamp > latest):
                active_dates.add(timestamp.astimezone(self.active_timezone).date().isoformat())
        opportunity["active_days_since_reflection"] = len(active_dates)
        opportunity["active_day_timezone"] = self.timezone_name
        eligible = []
        for row in rows:
            cooldown_until = row.get("cooldown_until") or ""
            cooling = False
            if cooldown_until:
                try:
                    cooling = datetime.fromisoformat(
                        cooldown_until.replace("Z", "+00:00")
                    ) > now
                except ValueError:
                    cooling = False
            # Explicit invitation and contradiction are evidence-bearing reasons
            # that may bypass a cooldown. Other triggers merely surface an
            # opportunity and respect the rumination guard.
            bypass = trigger_type in {"user_invite", "contradiction", "current_event"}
            if cooling and not bypass:
                continue
            candidate = {
                "memory_id": row["memory_id"],
                "text": row["text"],
                "timestamp": row["timestamp"],
                "source_ref": row.get("source_ref", ""),
                "epistemic_status": row.get("epistemic_status", "reported"),
                "reflection_count": row.get("reflection_count", 0),
                "last_reflection_at": row.get("last_reflection_at"),
                "cooldown_until": cooldown_until,
                "cooldown_bypassed": bool(cooling and bypass),
                "trigger_type": trigger_type,
                "selection_reason": self._selection_reason(trigger_type, row),
            }
            eligible.append(candidate)

        if (
            trigger_type == "event_count"
            and opportunity["new_event_count"] < self.event_threshold
        ):
            eligible = []
        elif trigger_type == "active_day":
            if opportunity["active_days_since_reflection"] < self.active_day_threshold:
                eligible = []
        elif trigger_type == "random_non_hot":
            pool = sorted(eligible, key=lambda item: (item["reflection_count"], item["timestamp"]))
            rng = random.Random(random_seed)
            if rng.random() > self.random_probability or self.random_budget == 0:
                eligible = []
            else:
                rng.shuffle(pool)
                eligible = pool[:self.random_budget]

        return {
            "trigger_type": trigger_type,
            "auto_saved": False,
            "auto_save_enabled": self.auto_save,
            "thresholds": {
                "event_count": self.event_threshold,
                "active_days": self.active_day_threshold,
                "cooldown_days": self.cooldown_days,
                "random_probability": self.random_probability,
                "random_budget": self.random_budget,
            },
            "opportunity": opportunity,
            "candidates": eligible[:max(1, min(limit, 50))],
        }

    @staticmethod
    def _parse_utc(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _selection_reason(trigger_type: str, row: dict) -> str:
        reasons = {
            "user_invite": "The user explicitly invited a review.",
            "current_event": "A current event provides new context for this event seed.",
            "contradiction": "Current evidence may conflict with an earlier interpretation.",
            "unresolved": "The event retains an unresolved question.",
            "repeated_tendency": "A tendency may recur across independent contexts.",
            "curiosity": "A deliberate review was requested without assuming change.",
            "event_count": "The configurable event-count opportunity threshold was reached.",
            "active_day": "The configurable active-day opportunity threshold was reached.",
            "random_non_hot": "A reproducible low-frequency non-hot candidate was sampled.",
            "background_task": "An authorized background task requested a candidate.",
        }
        suffix = f" Existing reflections: {row.get('reflection_count', 0)}."
        return reasons[trigger_type] + suffix

    def draft(
        self, source_event_ids: list[str], trigger_type: str,
        selection_reason: str, previous_interpretation: str,
        current_interpretation: str, change_or_tension: str,
        confidence: float, open_questions: list[str] | None,
        counterevidence: list | None, provenance: dict,
        status: str = "draft", supersedes: str = "",
    ) -> dict:
        if trigger_type not in TRIGGER_TYPES:
            raise ValueError(f"Unsupported trigger_type: {trigger_type}")
        if status not in REFLECTION_STATUSES:
            raise ValueError(f"Unsupported reflection status: {status}")
        if status == "adopted":
            raise ValueError("A draft cannot be adopted before explicit save/review")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not source_event_ids:
            raise ValueError("source_event_ids must not be empty")
        sources = []
        for memory_id in list(dict.fromkeys(source_event_ids)):
            row = self.db.get(memory_id)
            if not row:
                raise ValueError(f"Unknown source event ID: {memory_id}")
            if row.get("memory_layer") != "event":
                raise ValueError(f"Reflection source is not an event: {memory_id}")
            sources.append({
                "memory_id": memory_id,
                "epistemic_status": row.get("epistemic_status"),
                "source_ref": row.get("source_ref", ""),
                "text": row.get("text", ""),
            })
        if not str(current_interpretation).strip():
            raise ValueError("current_interpretation cannot be empty")
        if not str(change_or_tension).strip():
            raise ValueError("change_or_tension must be explicit; 'no_change' is valid")
        normalized = {
            "source_event_ids": [source["memory_id"] for source in sources],
            "trigger_type": trigger_type,
            "selection_reason": str(selection_reason).strip(),
            "previous_interpretation": str(previous_interpretation).strip(),
            "current_interpretation": str(current_interpretation).strip(),
            "change_or_tension": str(change_or_tension).strip(),
            "confidence": float(confidence),
            "open_questions": [str(item).strip() for item in (open_questions or []) if str(item).strip()],
            "counterevidence": self._normalize_counterevidence(counterevidence),
            "provenance": self._validate_provenance(provenance),
            "status": status,
            "supersedes": supersedes or "",
        }
        canonical = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {
            "persisted": False,
            "draft_token": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "draft": normalized,
            "sources": sources,
        }

    def save(self, draft: dict, expected_draft_token: str = "") -> dict:
        normalized = self.draft(**draft)
        if expected_draft_token and expected_draft_token != normalized["draft_token"]:
            raise ValueError("draft token does not match the submitted reflection")
        content = normalized["draft"]
        reflection_id = f"refl_{uuid.uuid4().hex[:12]}"
        cooldown_until = _iso(_now() + timedelta(days=self.cooldown_days))
        return self.db.create_reflection(
            reflection_id=reflection_id,
            cooldown_until=cooldown_until,
            **content,
        )
