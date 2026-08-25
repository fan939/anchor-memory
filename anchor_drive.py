"""Drive MVP: durable intentions and their explicit state evolution.

Drives deliberately do not participate in memory recall, vector storage, or
autonomous execution.  They are a separate, structured record of what remains
worth maintaining, confirming, or completing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid


DRIVE_STATUSES = {"active", "paused", "satisfied", "abandoned", "superseded"}
DRIVE_TARGET_TYPES = {"memory", "reflection", "drive"}
DRIVE_RELATIONS = {
    "motivated_by", "supported_by", "conflicts_with", "depends_on", "supersedes", "related",
}
TERMINAL_DRIVE_STATUSES = {"satisfied", "abandoned", "superseded"}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("datetime must be a valid ISO-8601 value") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _iso(value: datetime) -> str:
    return value.isoformat()


def compute_effective_strength(drive: dict, now: datetime | str | None = None) -> float:
    """Pure decay calculation.  It never persists state or changes the Drive."""
    strength = float(drive["strength"])
    policy = drive.get("decay_policy") or {}
    kind = policy.get("type", "none")
    if kind == "none":
        return strength
    if kind != "linear":
        raise ValueError("decay_policy.type must be none or linear")
    current = _parse_time(now) if isinstance(now, str) else (now or _now())
    endorsed = _parse_time(drive.get("last_endorsed_at"))
    if endorsed is None:
        raise ValueError("Drive last_endorsed_at is required for linear decay")
    grace_days = float(policy.get("grace_days", 0))
    decay_per_day = float(policy.get("decay_per_day", 0))
    floor = float(policy.get("floor", 0.0))
    if grace_days < 0 or decay_per_day < 0 or not 0.0 <= floor <= 1.0:
        raise ValueError("linear decay policy values are out of range")
    elapsed_days = max(0.0, (current - endorsed).total_seconds() / 86400.0 - grace_days)
    return max(floor, min(1.0, strength - elapsed_days * decay_per_day))


class DriveService:
    def __init__(self, db):
        self.db = db

    @staticmethod
    def _questions(values: list | None) -> list[str]:
        return list(dict.fromkeys(str(item).strip() for item in (values or []) if str(item).strip()))

    @staticmethod
    def _validate_policy(policy: dict | None) -> dict:
        policy = dict(policy or {"type": "none"})
        kind = policy.get("type", "none")
        if kind == "none":
            return {"type": "none"}
        if kind != "linear":
            raise ValueError("decay_policy.type must be none or linear")
        normalized = {
            "type": "linear",
            "grace_days": float(policy.get("grace_days", 0)),
            "decay_per_day": float(policy.get("decay_per_day", 0)),
            "floor": float(policy.get("floor", 0)),
        }
        if normalized["grace_days"] < 0 or normalized["decay_per_day"] < 0:
            raise ValueError("linear decay grace_days and decay_per_day must be non-negative")
        if not 0.0 <= normalized["floor"] <= 1.0:
            raise ValueError("linear decay floor must be between 0.0 and 1.0")
        return normalized

    @staticmethod
    def _validate_link(link: dict) -> dict:
        target_type = str(link.get("target_type", ""))
        target_id = str(link.get("target_id", "")).strip()
        relation = str(link.get("relation", ""))
        weight = float(link.get("weight", 1.0))
        if target_type not in DRIVE_TARGET_TYPES:
            raise ValueError("target_type must be memory, reflection, or drive")
        if not target_id:
            raise ValueError("link target_id is required")
        if relation not in DRIVE_RELATIONS:
            raise ValueError("unsupported drive link relation")
        if weight < 0.0:
            raise ValueError("drive link weight must be non-negative")
        return {"target_type": target_type, "target_id": target_id, "relation": relation, "weight": weight}

    @staticmethod
    def _is_expired(drive: dict, now: datetime | None = None) -> bool:
        expires = _parse_time(drive.get("expires_at"))
        return bool(expires and expires <= (now or _now()))

    def _present(self, drive: dict) -> dict:
        result = dict(drive)
        result["is_expired"] = self._is_expired(result)
        result["effective_strength"] = compute_effective_strength(result)
        return result

    def create(self, *, content: str, reason: str, strength: float, priority: int = 0,
               expires_at: str = "", decay_policy: dict | None = None,
               source_ref: str = "", provenance: dict | None = None,
               open_questions: list | None = None, links: list | None = None) -> dict:
        content, reason = str(content).strip(), str(reason).strip()
        if not content or not reason:
            raise ValueError("active Drive requires non-empty content and reason")
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be between 0.0 and 1.0")
        _parse_time(expires_at)
        duplicates = self.db.find_drive_duplicates(content)
        if duplicates:
            return {
                "status": "duplicate_suspected", "created": False,
                "conflict": "An active or paused Drive has the same normalized content.",
                "candidate_ids": [item["drive_id"] for item in duplicates],
            }
        normalized_links = [self._validate_link(item) for item in (links or [])]
        drive = self.db.create_drive({
            "drive_id": f"drv_{uuid.uuid4().hex[:12]}", "content": content, "reason": reason,
            "status": "active", "strength": float(strength), "priority": int(priority),
            "expires_at": expires_at or "", "decay_policy": self._validate_policy(decay_policy),
            "source_ref": str(source_ref), "provenance": dict(provenance or {}),
            "open_questions": self._questions(open_questions), "completion_note": "",
        }, normalized_links)
        return {"status": "created", "created": True, "drive": self._present(drive)}

    def list(self, status: list[str] | None = None, min_strength: float = 0.3,
             min_priority: int = 0, include_expired: bool = False, limit: int = 20) -> dict:
        statuses = status or ["active"]
        if any(item not in DRIVE_STATUSES for item in statuses):
            raise ValueError("unsupported Drive status")
        if not 0.0 <= float(min_strength) <= 1.0:
            raise ValueError("min_strength must be between 0.0 and 1.0")
        return {"drives": [self._present(item) for item in self.db.list_drives(
            statuses, min_strength, min_priority, include_expired, limit,
        )]}

    def get(self, drive_id: str) -> dict:
        drive = self.db.get_drive(drive_id)
        if drive is None:
            raise ValueError("drive not found")
        return {"drive": self._present(drive)}

    def update(self, drive_id: str, *, reactivate: bool = False,
               reactivation_reason: str = "", **updates) -> dict:
        current = self.db.get_drive(drive_id)
        if current is None:
            raise ValueError("drive not found")
        status = updates.get("status")
        if status is not None and status not in DRIVE_STATUSES:
            raise ValueError("unsupported Drive status")
        if "strength" in updates and updates["strength"] is not None:
            if not 0.0 <= float(updates["strength"]) <= 1.0:
                raise ValueError("strength must be between 0.0 and 1.0")
            updates["strength"] = float(updates["strength"])
        if "expires_at" in updates and updates["expires_at"] is not None:
            _parse_time(updates["expires_at"])
        if "decay_policy" in updates and updates["decay_policy"] is not None:
            updates["decay_policy"] = self._validate_policy(updates["decay_policy"])
        if "open_questions" in updates and updates["open_questions"] is not None:
            updates["open_questions"] = self._questions(updates["open_questions"])
        terminal = current["status"] in TERMINAL_DRIVE_STATUSES
        if terminal and status == "active":
            if not reactivate or not str(reactivation_reason).strip():
                raise ValueError("reactivating a terminal Drive requires reactivate=true and reactivation_reason")
        elif reactivate:
            raise ValueError("reactivate=true is only valid when restoring a terminal Drive to active")
        if status == "active" and (not str(updates.get("content", current["content"])).strip()
                                   or not str(updates.get("reason", current["reason"])).strip()):
            raise ValueError("active Drive requires non-empty content and reason")
        updates = {key: value for key, value in updates.items() if value is not None}
        updated = self.db.update_drive(
            drive_id, updates,
            endorsed=(terminal and status == "active" and reactivate),
        )
        if terminal and status == "active":
            self.db.log_event(drive_id, "drive_reactivated", str(reactivation_reason).strip()[:200])
        return {"status": "updated", "drive": self._present(updated)}

    def link(self, drive_id: str, **link) -> dict:
        return {"status": "linked", "drive": self._present(self.db.link_drive(
            drive_id, self._validate_link(link),
        ))}

    def review(self, stale_days: int = 30, limit: int = 50) -> dict:
        if stale_days < 1:
            raise ValueError("stale_days must be at least 1")
        now = _now()
        all_drives = self.db.list_drives(
            list(DRIVE_STATUSES), 0.0, -2**31, True, limit=1000, now=_iso(now),
        )
        links_by_drive = {item["drive_id"]: item["links"] for item in
                          (self.db.get_drive(row["drive_id"]) for row in all_drives)}
        stale_cutoff = now.timestamp() - stale_days * 86400
        active = [item for item in all_drives if item["status"] == "active"]
        duplicate_groups: dict[str, list[str]] = {}
        for item in all_drives:
            if item["status"] in {"active", "paused"}:
                key = " ".join(item["content"].split()).casefold()
                duplicate_groups.setdefault(key, []).append(item["drive_id"])

        def compact(item: dict) -> dict:
            return {"drive_id": item["drive_id"], "status": item["status"], "content": item["content"]}

        return {
            "expired_active": [compact(item) for item in active if self._is_expired(item, now)][:limit],
            "stale": [compact(item) for item in all_drives if item["status"] in {"active", "paused"}
                      and (_parse_time(item["last_endorsed_at"]) or now).timestamp() < stale_cutoff][:limit],
            "conflicts": [compact(item) for item in active if any(
                link["relation"] == "conflicts_with" for link in links_by_drive[item["drive_id"]]
            )][:limit],
            "high_strength_missing_source_links": [compact(item) for item in active if item["strength"] >= 0.8
                and not any(link["relation"] in {"motivated_by", "supported_by"}
                            for link in links_by_drive[item["drive_id"]])][:limit],
            "duplicate_groups": [ids for ids in duplicate_groups.values() if len(ids) > 1][:limit],
            "terminal_missing_completion_note": [compact(item) for item in all_drives
                if item["status"] in {"satisfied", "abandoned"} and not item.get("completion_note", "").strip()][:limit],
        }
