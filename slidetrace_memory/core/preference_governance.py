"""V19 preference claim deduplication and conflict governance.

This module is deliberately additive: it never deletes or rewrites
``memory_records`` or ``memory_contributions``.  LLMs may propose structured
claims and relations, while the deterministic validator owns database changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timezone
from typing import Callable, Iterable, Optional, Sequence


RELATIONS = frozenset({
    "exact_duplicate", "paraphrase_duplicate", "reinforcement", "refinement",
    "subsumes", "exception", "contradiction", "temporal_update", "unrelated",
    "uncertain",
})
CONFLICT_TYPES = frozenset({
    "explicit_retraction", "explicit_supersession", "scope_exception",
    "value_conflict", "priority_conflict", "temporal_conflict", "evidence_conflict",
})
ACTIONS = frozenset({
    "coexist", "supersede", "retract", "conditionalize", "downgrade",
    "hold_unresolved",
})
STATUSES = frozenset({
    "active", "superseded", "retracted", "expired", "conflicted", "archived",
    "compacted",
})
DIRECTIONS = frozenset({"require", "prefer", "avoid", "prohibit"})
PRIORITIES = frozenset({"must", "should", "soft"})
SOURCE_TYPES = frozenset({"explicit_feedback", "inferred", "edit", "acceptance"})


@dataclass(frozen=True)
class PreferenceClaim:
    persona_id: str
    subject: str
    attribute: str
    action_or_value: str
    direction: str
    scope: dict
    conditions: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    priority: str = "soft"
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source_memory_id: str = ""
    source_type: str = "explicit_feedback"
    extraction_confidence: float = 1.0
    confidence: Optional[float] = None
    salience: float = 0.0
    tags: list[str] = field(default_factory=list)
    claim_id: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_boundary(value: str | date | datetime | None, *, end_of_day: bool = False) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.max if end_of_day else time.min)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if len(text) == 10:
                parsed = datetime.combine(date.fromisoformat(text), time.max if end_of_day else time.min)
            else:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid claim validity timestamp: {value!r}") from exc
    else:
        raise ValueError("claim validity timestamps must be ISO dates or datetimes")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def claim_is_current(claim: PreferenceClaim | dict, *, at: str | datetime | None = None) -> bool:
    """Check inclusive UTC bounds; naive times are UTC and date-only ends include the day."""
    value = asdict(claim) if isinstance(claim, PreferenceClaim) else claim
    start = _utc_boundary(value.get("valid_from"))
    end = _utc_boundary(value.get("valid_to"), end_of_day=True)
    if start is not None and end is not None and start > end:
        raise ValueError("valid_from must not be later than valid_to")
    current = _utc_boundary(at) if at is not None else datetime.now(timezone.utc)
    if current is None:
        raise ValueError("at must be an ISO date or datetime when supplied")
    return (start is None or current >= start) and (end is None or current <= end)


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _norm_list(values: Iterable[object]) -> list[str]:
    return sorted({_norm_text(value) for value in values if _norm_text(value)})


def _norm_scope(scope: object) -> dict:
    if isinstance(scope, str):
        return {"kind": _norm_text(scope) or "global"}
    if not isinstance(scope, dict):
        raise ValueError("scope must be a string or object")
    normalized = {str(k).strip().lower(): _norm_text(v) for k, v in scope.items() if _norm_text(v)}
    normalized.setdefault("kind", "global")
    return dict(sorted(normalized.items()))


def normalize_claim(claim: PreferenceClaim) -> dict:
    data = asdict(claim)
    for key in ("persona_id", "subject", "attribute", "action_or_value"):
        data[key] = _norm_text(data[key])
        if not data[key]:
            raise ValueError(f"{key} is required")
    data["direction"] = _norm_text(data["direction"])
    data["priority"] = _norm_text(data["priority"])
    data["source_type"] = _norm_text(data["source_type"])
    if data["direction"] not in DIRECTIONS:
        raise ValueError(f"invalid direction: {data['direction']}")
    if data["priority"] not in PRIORITIES:
        raise ValueError(f"invalid priority: {data['priority']}")
    if data["source_type"] not in SOURCE_TYPES:
        raise ValueError(f"invalid source_type: {data['source_type']}")
    confidence = float(data["extraction_confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("extraction_confidence must be between 0 and 1")
    data["extraction_confidence"] = confidence
    data["confidence"] = confidence if data["confidence"] is None else float(data["confidence"])
    if not 0.0 <= data["confidence"] <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    data["salience"] = float(data["salience"])
    data["tags"] = _norm_list(data["tags"])
    data["scope"] = _norm_scope(data["scope"])
    data["conditions"] = _norm_list(data["conditions"])
    data["exceptions"] = _norm_list(data["exceptions"])
    data["source_memory_id"] = str(data["source_memory_id"] or "")
    # Validate the window on entry, keeping date-only values readable in storage.
    claim_is_current(data)
    return data


def claim_fingerprint(claim: PreferenceClaim | dict) -> str:
    data = normalize_claim(claim if isinstance(claim, PreferenceClaim) else PreferenceClaim(**claim))
    identity = {key: data[key] for key in (
        "persona_id", "subject", "attribute", "action_or_value", "direction",
        "scope", "conditions", "exceptions",
    )}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ensure_preference_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS preference_claims (
            claim_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL UNIQUE,
            persona_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            attribute TEXT NOT NULL,
            action_or_value TEXT NOT NULL,
            direction TEXT NOT NULL,
            scope_json TEXT NOT NULL,
            conditions_json TEXT NOT NULL DEFAULT '[]',
            exceptions_json TEXT NOT NULL DEFAULT '[]',
            priority TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            source_type TEXT NOT NULL,
            extraction_confidence REAL NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            salience REAL NOT NULL DEFAULT 0.0,
            support_count INTEGER NOT NULL DEFAULT 1,
            retrieved_count INTEGER NOT NULL DEFAULT 0,
            exposed_count INTEGER NOT NULL DEFAULT 0,
            executed_count INTEGER NOT NULL DEFAULT 0,
            confirmed_count INTEGER NOT NULL DEFAULT 0,
            rejected_count INTEGER NOT NULL DEFAULT 0,
            contradiction_count INTEGER NOT NULL DEFAULT 0,
            tags_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'active',
            embedding BLOB,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_preference_claims_active
            ON preference_claims(persona_id, status, subject, attribute);
        CREATE TABLE IF NOT EXISTS preference_evidence (
            evidence_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            source_memory_id TEXT,
            source_type TEXT NOT NULL,
            evidence_text TEXT,
            ingestion_key TEXT UNIQUE,
            created_at TEXT NOT NULL,
            FOREIGN KEY(claim_id) REFERENCES preference_claims(claim_id)
        );
        CREATE TABLE IF NOT EXISTS preference_relations (
            relation_id TEXT PRIMARY KEY,
            claim_a TEXT NOT NULL,
            claim_b TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            decision_id TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_preference_relation_pair
            ON preference_relations(claim_a, claim_b, relation_type, active);
        CREATE TABLE IF NOT EXISTS dedup_decisions (
            decision_id TEXT PRIMARY KEY,
            claim_a TEXT,
            claim_b TEXT,
            proposed_relation TEXT NOT NULL,
            proposed_confidence REAL NOT NULL,
            validator_result TEXT NOT NULL,
            final_relation TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            rule_version TEXT NOT NULL,
            model TEXT,
            prompt_version TEXT,
            before_state_json TEXT NOT NULL,
            after_state_json TEXT NOT NULL,
            rolled_back INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conflict_decisions (
            decision_id TEXT PRIMARY KEY,
            claim_a TEXT NOT NULL,
            claim_b TEXT NOT NULL,
            detected_conflict_type TEXT NOT NULL,
            overlapping_scope_json TEXT NOT NULL,
            evidence_ids_json TEXT NOT NULL DEFAULT '[]',
            proposed_action TEXT NOT NULL,
            validator_result TEXT NOT NULL,
            final_action TEXT NOT NULL,
            reason TEXT NOT NULL,
            model TEXT,
            prompt_version TEXT,
            rule_version TEXT NOT NULL,
            before_state_json TEXT NOT NULL,
            after_state_json TEXT NOT NULL,
            rolled_back INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS preference_signal_events (
            event_key TEXT PRIMARY KEY,
            signal TEXT NOT NULL,
            claim_ids_json TEXT NOT NULL,
            count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS preference_status_events (
            event_id TEXT PRIMARY KEY,
            event_key TEXT UNIQUE,
            claim_id TEXT NOT NULL,
            from_status TEXT NOT NULL,
            to_status TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
    """)
    # Safe additive migration for databases created by earlier V19 builds.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(preference_claims)")}
    migrations = {
        "confidence": "REAL NOT NULL DEFAULT 0.5",
        "salience": "REAL NOT NULL DEFAULT 0.0",
        "retrieved_count": "INTEGER NOT NULL DEFAULT 0",
        "exposed_count": "INTEGER NOT NULL DEFAULT 0",
        "executed_count": "INTEGER NOT NULL DEFAULT 0",
        "rejected_count": "INTEGER NOT NULL DEFAULT 0",
        "contradiction_count": "INTEGER NOT NULL DEFAULT 0",
        "tags_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for name, declaration in migrations.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE preference_claims ADD COLUMN {name} {declaration}")
    conn.commit()


def decode_claim_row(row: sqlite3.Row | dict) -> dict:
    """Decode canonical metadata identically for retrieval and governance."""
    value = dict(row)
    value["scope"] = json.loads(value.pop("scope_json"))
    value["conditions"] = json.loads(value.pop("conditions_json"))
    value["exceptions"] = json.loads(value.pop("exceptions_json"))
    value["tags"] = json.loads(value.pop("tags_json", "[]"))
    return value


def _fetch_claim(conn: sqlite3.Connection, claim_id: str) -> dict:
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM preference_claims WHERE claim_id=?", (claim_id,)).fetchone()
    if not row:
        raise KeyError(f"unknown claim: {claim_id}")
    return decode_claim_row(row)


def scopes_compatible(a: dict, b: dict) -> bool:
    a, b = _norm_scope(a), _norm_scope(b)
    for key in set(a) & set(b):
        if key != "kind" and a[key] != b[key]:
            return False
    return True


def scope_specificity(scope: dict) -> int:
    normalized = _norm_scope(scope)
    return sum(1 for key, value in normalized.items() if key != "kind" and value) + (
        0 if normalized.get("kind") == "global" else 1
    )


def scope_applies(claim_scope: dict, requested_scope: dict) -> bool:
    """True only when every claim restriction is present in the request."""
    claim_scope, requested_scope = _norm_scope(claim_scope), _norm_scope(requested_scope)
    if claim_scope.get("kind") == "global":
        return True
    if requested_scope.get("kind") == "global":
        return False
    for key, value in claim_scope.items():
        if key == "kind":
            continue
        if requested_scope.get(key) != value:
            return False
    return True


def ingest_claim(conn: sqlite3.Connection, claim: PreferenceClaim, *, evidence_text: str = "",
                  ingestion_key: Optional[str] = None, update_priority: bool = False,
                  reactivate: bool = False) -> dict:
    """Add evidence; inactive exact matches require an explicit reactivation request."""
    ensure_preference_schema(conn)
    if ingestion_key:
        prior = conn.execute(
            "SELECT e.claim_id,e.evidence_id,c.status FROM preference_evidence e "
            "JOIN preference_claims c ON c.claim_id=e.claim_id WHERE e.ingestion_key=?",
            (ingestion_key,),
        ).fetchone()
        if prior:
            return {"claim_id": prior[0], "evidence_id": prior[1], "created": False,
                    "relation": "idempotent_retry", "decision_id": None,
                    "status": prior[2], "reactivated": False}
    data = normalize_claim(claim)
    if update_priority and data["source_type"] not in {"explicit_feedback", "edit"}:
        raise ValueError("priority changes require explicit feedback or an edit")
    if not isinstance(reactivate, bool):
        raise ValueError("reactivate must be a boolean")
    if reactivate and data["source_type"] not in {"explicit_feedback", "edit"}:
        raise ValueError("reactivation requires explicit feedback or an edit")
    fingerprint = claim_fingerprint(claim)
    now = _now()
    existing = conn.execute(
        "SELECT claim_id,priority,status FROM preference_claims WHERE fingerprint=?", (fingerprint,)
    ).fetchone()
    created = existing is None
    claim_id = (data.get("claim_id") or str(uuid.uuid4())) if created else existing[0]
    old_priority = existing[1] if existing else data["priority"]
    old_status = existing[2] if existing else "active"
    reactivated = bool(reactivate and not created and old_status in {"retracted", "superseded"})
    if reactivate and not created and old_status not in {"active", "retracted", "superseded"}:
        raise ValueError(f"explicit reactivation does not apply to status: {old_status}")
    priority_changed = bool(update_priority and not created and old_priority != data["priority"])
    if created:
        conn.execute("""INSERT INTO preference_claims
            (claim_id,fingerprint,persona_id,subject,attribute,action_or_value,direction,
             scope_json,conditions_json,exceptions_json,priority,valid_from,valid_to,
             source_type,extraction_confidence,confidence,salience,tags_json,status,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            claim_id, fingerprint, data["persona_id"], data["subject"], data["attribute"],
            data["action_or_value"], data["direction"], json.dumps(data["scope"], ensure_ascii=False, sort_keys=True),
            json.dumps(data["conditions"], ensure_ascii=False), json.dumps(data["exceptions"], ensure_ascii=False),
            data["priority"], data["valid_from"], data["valid_to"], data["source_type"],
            data["extraction_confidence"], data["confidence"], data["salience"],
            json.dumps(data["tags"], ensure_ascii=False), "active", now, now,
        ))
    evidence_id = str(uuid.uuid4())
    try:
        conn.execute("""INSERT INTO preference_evidence
            (evidence_id,claim_id,source_memory_id,source_type,evidence_text,ingestion_key,created_at)
            VALUES (?,?,?,?,?,?,?)""", (
            evidence_id, claim_id, data["source_memory_id"], data["source_type"],
            evidence_text, ingestion_key, now,
        ))
        if not created:
            conn.execute("""UPDATE preference_claims SET support_count=support_count+1,
                updated_at=? WHERE claim_id=?""", (now, claim_id))
            if priority_changed:
                conn.execute("UPDATE preference_claims SET priority=? WHERE claim_id=?",
                             (data["priority"], claim_id))
            if reactivated:
                # Restoring lifecycle state does not silently extend a dated rule.
                conn.execute("UPDATE preference_claims SET status='active',updated_at=? WHERE claim_id=?",
                             (now, claim_id))
                conn.execute("""INSERT INTO preference_status_events
                    (event_id,event_key,claim_id,from_status,to_status,reason,created_at)
                    VALUES (?,?,?,?,?,?,?)""", (
                    str(uuid.uuid4()), None, claim_id, old_status, "active",
                    "explicit feedback restores the same preference", now,
                ))
    except sqlite3.IntegrityError:
        evidence_id = conn.execute(
            "SELECT evidence_id FROM preference_evidence WHERE ingestion_key=?", (ingestion_key,)
        ).fetchone()[0]
    decision_id = str(uuid.uuid4())
    final_relation = "new_claim" if created else "exact_duplicate"
    conn.execute("""INSERT INTO dedup_decisions
        (decision_id,claim_a,claim_b,proposed_relation,proposed_confidence,validator_result,
         final_relation,evidence_json,rule_version,before_state_json,after_state_json,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (
        decision_id, claim_id if not created else None, claim_id, final_relation, 1.0,
        "accepted:fingerprint", final_relation, json.dumps([evidence_id]), "v19.1",
        json.dumps({"statuses": {claim_id: old_status} if reactivated else {},
                    "priorities": {claim_id: old_priority} if priority_changed else {}}),
        json.dumps({"statuses": {claim_id: "active"} if reactivated else {}, "claim_id": claim_id, "created": created,
                    "priorities": {claim_id: data["priority"]} if priority_changed else {}}), now,
    ))
    conn.commit()
    return {"claim_id": claim_id, "evidence_id": evidence_id, "created": created,
            "relation": final_relation, "decision_id": decision_id,
            "priority_changed": priority_changed,
            "status": "active" if reactivated else old_status, "reactivated": reactivated}


def find_embedding_candidates(conn: sqlite3.Connection, claim: PreferenceClaim,
                              embedding: Sequence[float], limit: int = 10) -> list[dict]:
    """Embedding is recall-only; hard persona/scope/direction boundaries remain enforced."""
    import pickle
    data = normalize_claim(claim)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""SELECT * FROM preference_claims
        WHERE persona_id=? AND status='active' AND subject=? AND attribute=? AND embedding IS NOT NULL""",
        (data["persona_id"], data["subject"], data["attribute"])).fetchall()
    result = []
    for row in rows:
        candidate = dict(row)
        scope = json.loads(candidate["scope_json"])
        if not scopes_compatible(data["scope"], scope):
            continue
        if data["direction"] != candidate["direction"]:
            continue
        vector = pickle.loads(candidate["embedding"])
        if len(vector) != len(embedding):
            continue
        denom = math.sqrt(sum(float(x) ** 2 for x in vector)) * math.sqrt(sum(float(x) ** 2 for x in embedding))
        score = sum(float(a) * float(b) for a, b in zip(vector, embedding)) / denom if denom else 0.0
        result.append({"claim_id": candidate["claim_id"], "similarity": score})
    return sorted(result, key=lambda item: (-item["similarity"], item["claim_id"]))[:limit]


def validate_relation(conn: sqlite3.Connection, claim_a: str, claim_b: str,
                      proposed_relation: str, confidence: float, evidence: Sequence[str] = (),
                      *, model: str = "", prompt_version: str = "",
                      update_priority_from: Optional[str] = None) -> dict:
    ensure_preference_schema(conn)
    if proposed_relation not in RELATIONS:
        raise ValueError(f"invalid relation: {proposed_relation}")
    a, b = _fetch_claim(conn, claim_a), _fetch_claim(conn, claim_b)
    priority_source = None
    if update_priority_from is not None:
        if update_priority_from not in {claim_a, claim_b}:
            raise ValueError("priority source must be one of the related claims")
        priority_source = a if update_priority_from == claim_a else b
        if priority_source["source_type"] not in {"explicit_feedback", "edit"}:
            raise ValueError("priority changes require explicit feedback or an edit")
    accepted, reason, final = True, "accepted", proposed_relation
    if a["persona_id"] != b["persona_id"]:
        accepted, reason, final = False, "persona_mismatch", "unrelated"
    elif a["subject"] != b["subject"] or a["attribute"] != b["attribute"]:
        accepted, reason, final = False, "subject_or_attribute_mismatch", "unrelated"
    elif not scopes_compatible(a["scope"], b["scope"]):
        accepted, reason, final = False, "incompatible_scope", "unrelated"
    elif proposed_relation in {"exact_duplicate", "paraphrase_duplicate"} and any(
            a[field] != b[field] for field in ("scope", "conditions", "exceptions")):
        accepted, reason, final = False, "different_scope_or_conditions_cannot_merge", "uncertain"
    elif proposed_relation in {"exact_duplicate", "paraphrase_duplicate"} and a["direction"] != b["direction"]:
        accepted, reason, final = False, "direction_flip_cannot_merge", "contradiction"
    elif proposed_relation == "exact_duplicate" and a["fingerprint"] != b["fingerprint"]:
        accepted, reason, final = False, "exact_relation_requires_equal_fingerprint", "uncertain"
    elif proposed_relation == "paraphrase_duplicate" and float(confidence) < 0.9:
        accepted, reason, final = False, "paraphrase_confidence_below_0.9", "uncertain"
    decision_id, now = str(uuid.uuid4()), _now()
    before = {claim_a: a["status"], claim_b: b["status"]}
    after = dict(before)
    priorities_before, priorities_after = {}, {}
    # Only exact/paraphrase may automatically canonicalize.  The loser remains
    # auditable and is marked superseded; evidence rows are copied, never deleted.
    if accepted and final in {"exact_duplicate", "paraphrase_duplicate"}:
        winner, loser = sorted([claim_a, claim_b])
        winner_claim = a if winner == claim_a else b
        # Stable IDs do not choose the user's current priority. Ordinary
        # paraphrases inherit the earlier rule; explicit updates name a source.
        if priority_source is not None:
            priority = priority_source["priority"]
        else:
            # Windows clocks may timestamp consecutive inserts identically.
            # Insertion order, not a random UUID, breaks that tie.
            earlier_id = conn.execute("""SELECT claim_id FROM preference_claims
                WHERE claim_id IN (?,?) ORDER BY created_at,rowid LIMIT 1""",
                (claim_a, claim_b)).fetchone()[0]
            priority = (a if earlier_id == claim_a else b)["priority"]
        if winner_claim["priority"] != priority:
            priorities_before[winner] = winner_claim["priority"]
            priorities_after[winner] = priority
            conn.execute("UPDATE preference_claims SET priority=? WHERE claim_id=?", (priority, winner))
        moved_evidence_ids = [row[0] for row in conn.execute(
            "SELECT evidence_id FROM preference_evidence WHERE claim_id=?", (loser,)
        ).fetchall()]
        conn.execute("UPDATE preference_claims SET status='superseded', updated_at=? WHERE claim_id=?", (now, loser))
        conn.execute("UPDATE preference_evidence SET claim_id=? WHERE claim_id=?", (winner, loser))
        count = conn.execute("SELECT COUNT(*) FROM preference_evidence WHERE claim_id=?", (winner,)).fetchone()[0]
        conn.execute("UPDATE preference_claims SET support_count=?, updated_at=? WHERE claim_id=?", (count, now, winner))
        after[loser] = "superseded"
    else:
        winner = loser = None
        moved_evidence_ids = []
    conn.execute("""INSERT INTO dedup_decisions
        (decision_id,claim_a,claim_b,proposed_relation,proposed_confidence,validator_result,
         final_relation,evidence_json,rule_version,model,prompt_version,before_state_json,
         after_state_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        decision_id, claim_a, claim_b, proposed_relation, float(confidence),
        ("accepted:" if accepted else "rejected:") + reason, final, json.dumps(list(evidence), ensure_ascii=False),
        "v19.1", model, prompt_version,
        json.dumps({"statuses": before, "priorities": priorities_before,
                    "evidence_claims": {evidence_id: loser for evidence_id in moved_evidence_ids}}),
        json.dumps({"statuses": after, "priorities": priorities_after, "canonical_claim": winner}), now,
    ))
    conn.execute("""INSERT OR IGNORE INTO preference_relations
        (relation_id,claim_a,claim_b,relation_type,confidence,evidence_json,decision_id,created_at)
        VALUES (?,?,?,?,?,?,?,?)""", (
        str(uuid.uuid4()), claim_a, claim_b, final, float(confidence),
        json.dumps(list(evidence), ensure_ascii=False), decision_id, now,
    ))
    conn.commit()
    return {"decision_id": decision_id, "accepted": accepted, "final_relation": final,
            "reason": reason, "canonical_claim_id": winner}


def resolve_conflict(conn: sqlite3.Connection, claim_a: str, claim_b: str,
                     conflict_type: str, *, explicit: bool = False,
                     evidence_ids: Sequence[str] = (), proposed_action: str = "hold_unresolved",
                     model: str = "", prompt_version: str = "",
                     preferred_winner: Optional[str] = None) -> dict:
    if conflict_type not in CONFLICT_TYPES or proposed_action not in ACTIONS:
        raise ValueError("invalid conflict type or action")
    a, b = _fetch_claim(conn, claim_a), _fetch_claim(conn, claim_b)
    if (a["persona_id"] != b["persona_id"] or a["subject"] != b["subject"]
            or a["attribute"] != b["attribute"] or not scopes_compatible(a["scope"], b["scope"])):
        action, reason = "coexist", "persona_subject_attribute_or_scope_disjoint"
    elif conflict_type == "explicit_retraction" and explicit:
        action, reason = "retract", "explicit_retraction"
    elif conflict_type == "explicit_supersession" and explicit:
        action, reason = "supersede", "explicit_supersession"
    elif conflict_type == "scope_exception" or scope_specificity(a["scope"]) != scope_specificity(b["scope"]):
        action, reason = "conditionalize", "more_specific_scope_is_exception"
    elif a["source_type"] != b["source_type"] and "explicit_feedback" in {a["source_type"], b["source_type"]}:
        action, reason = "supersede", "direct_user_expression_wins"
    elif (a["priority"] != b["priority"] and "must" in {a["priority"], b["priority"]}) or (
        (a["direction"] == "prohibit") != (b["direction"] == "prohibit")
    ):
        action, reason = "supersede", "hard_priority_wins_same_scope"
    elif conflict_type == "temporal_conflict" and (a["valid_from"] or b["valid_from"]):
        action, reason = "supersede", "explicit_temporal_update"
    else:
        action, reason = "hold_unresolved", "insufficient_deterministic_evidence"
    before = {claim_a: a["status"], claim_b: b["status"]}
    after = dict(before)
    now = _now()
    if preferred_winner is not None:
        if preferred_winner not in {claim_a, claim_b}:
            raise ValueError("preferred_winner must be one of the conflict claims")
        if not explicit or conflict_type not in {"explicit_supersession", "temporal_conflict"}:
            raise ValueError("preferred_winner is allowed only for explicit temporal updates")
    if action == "retract":
        conn.execute("UPDATE preference_claims SET status='retracted',updated_at=? WHERE claim_id=?", (now, claim_a))
        after[claim_a] = "retracted"
    elif action == "supersede":
        if preferred_winner is not None:
            winner = preferred_winner
            loser = claim_b if winner == claim_a else claim_a
        else:
            winner, loser = _choose_winner(a, b)
        conn.execute("UPDATE preference_claims SET status='superseded',updated_at=? WHERE claim_id=?", (now, loser))
        after[loser] = "superseded"
        if preferred_winner is not None and before[winner] != "active":
            # A confirmed explicit update may restore an exact earlier value.
            # Reusing its canonical ID must not leave that winner superseded.
            conn.execute("UPDATE preference_claims SET status='active',updated_at=? WHERE claim_id=?", (now, winner))
            after[winner] = "active"
    elif action == "hold_unresolved":
        conn.execute("UPDATE preference_claims SET status='conflicted',updated_at=? WHERE claim_id IN (?,?)", (now, claim_a, claim_b))
        after = {claim_a: "conflicted", claim_b: "conflicted"}
    decision_id = str(uuid.uuid4())
    validator_result = ("accepted:" if proposed_action == action else "overridden:") + reason
    conn.execute("""INSERT INTO conflict_decisions
        (decision_id,claim_a,claim_b,detected_conflict_type,overlapping_scope_json,
         evidence_ids_json,proposed_action,validator_result,final_action,reason,model,
         prompt_version,rule_version,before_state_json,after_state_json,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        decision_id, claim_a, claim_b, conflict_type,
        json.dumps(a["scope"], ensure_ascii=False, sort_keys=True), json.dumps(list(evidence_ids)),
        proposed_action, validator_result, action, reason, model, prompt_version, "v19.1",
        json.dumps(before), json.dumps(after), now,
    ))
    if action != "coexist":
        relation_type = (
            "exception" if action == "conditionalize" else
            "temporal_update" if conflict_type in {"explicit_supersession", "temporal_conflict"} else
            "contradiction"
        )
        conn.execute("""INSERT OR IGNORE INTO preference_relations
            (relation_id,claim_a,claim_b,relation_type,confidence,evidence_json,decision_id,created_at)
            VALUES (?,?,?,?,?,?,?,?)""", (
            str(uuid.uuid4()), claim_a, claim_b, relation_type, 1.0 if explicit else 0.8,
            json.dumps(list(evidence_ids)), decision_id, now,
        ))
    conn.commit()
    return {"decision_id": decision_id, "action": action, "reason": reason, "after": after}


def _choose_winner(a: dict, b: dict) -> tuple[str, str]:
    source_rank = {"explicit_feedback": 4, "edit": 3, "acceptance": 2, "inferred": 1}
    priority_rank = {"must": 3, "should": 2, "soft": 1}
    def key(item: dict) -> tuple:
        hard_direction = 1 if item["direction"] in {"require", "prohibit"} else 0
        return (source_rank[item["source_type"]], max(priority_rank[item["priority"]], hard_direction * 3),
                item["valid_from"] or "", item["created_at"], item["claim_id"])
    winner, loser = sorted([a, b], key=key, reverse=True)
    return winner["claim_id"], loser["claim_id"]


def rollback_decision(conn: sqlite3.Connection, decision_id: str) -> None:
    row = conn.execute("""SELECT before_state_json,rolled_back,'dedup' AS kind FROM dedup_decisions WHERE decision_id=?
        UNION ALL SELECT before_state_json,rolled_back,'conflict' FROM conflict_decisions WHERE decision_id=?""",
        (decision_id, decision_id)).fetchone()
    if not row:
        raise KeyError(decision_id)
    if row[1]:
        return
    before = json.loads(row[0])
    statuses = before.get("statuses", before)
    for claim_id, status in statuses.items():
        if status not in STATUSES:
            raise ValueError(f"invalid saved status: {status}")
        conn.execute("UPDATE preference_claims SET status=?,updated_at=? WHERE claim_id=?", (status, _now(), claim_id))
    for claim_id, priority in before.get("priorities", {}).items():
        if priority not in PRIORITIES:
            raise ValueError(f"invalid saved priority: {priority}")
        conn.execute("UPDATE preference_claims SET priority=?,updated_at=? WHERE claim_id=?",
                     (priority, _now(), claim_id))
    for evidence_id, original_claim_id in before.get("evidence_claims", {}).items():
        conn.execute("UPDATE preference_evidence SET claim_id=? WHERE evidence_id=?", (original_claim_id, evidence_id))
    for claim_id in statuses:
        count = conn.execute("SELECT COUNT(*) FROM preference_evidence WHERE claim_id=?", (claim_id,)).fetchone()[0]
        conn.execute("UPDATE preference_claims SET support_count=?,updated_at=? WHERE claim_id=?", (count, _now(), claim_id))
    table = "dedup_decisions" if row[2] == "dedup" else "conflict_decisions"
    conn.execute(f"UPDATE {table} SET rolled_back=1 WHERE decision_id=?", (decision_id,))
    conn.execute("UPDATE preference_relations SET active=0 WHERE decision_id=?", (decision_id,))
    conn.commit()


def replay_decision(conn: sqlite3.Connection, decision_id: str) -> None:
    """Deterministically reapply the recorded after-state of a rolled-back decision."""
    row = conn.execute("""SELECT before_state_json,after_state_json,rolled_back,'dedup' AS kind
        FROM dedup_decisions WHERE decision_id=?
        UNION ALL SELECT before_state_json,after_state_json,rolled_back,'conflict'
        FROM conflict_decisions WHERE decision_id=?""", (decision_id, decision_id)).fetchone()
    if not row:
        raise KeyError(decision_id)
    after = json.loads(row[1])
    statuses = after.get("statuses", after)
    for claim_id, status in statuses.items():
        if status not in STATUSES:
            raise ValueError(f"invalid saved status: {status}")
        conn.execute("UPDATE preference_claims SET status=?,updated_at=? WHERE claim_id=?", (status, _now(), claim_id))
    for claim_id, priority in after.get("priorities", {}).items():
        if priority not in PRIORITIES:
            raise ValueError(f"invalid saved priority: {priority}")
        conn.execute("UPDATE preference_claims SET priority=?,updated_at=? WHERE claim_id=?",
                     (priority, _now(), claim_id))
    before = json.loads(row[0])
    canonical_claim = after.get("canonical_claim")
    if canonical_claim:
        for evidence_id in before.get("evidence_claims", {}):
            conn.execute("UPDATE preference_evidence SET claim_id=? WHERE evidence_id=?", (canonical_claim, evidence_id))
    for claim_id in statuses:
        count = conn.execute("SELECT COUNT(*) FROM preference_evidence WHERE claim_id=?", (claim_id,)).fetchone()[0]
        conn.execute("UPDATE preference_claims SET support_count=?,updated_at=? WHERE claim_id=?", (count, _now(), claim_id))
    table = "dedup_decisions" if row[3] == "dedup" else "conflict_decisions"
    conn.execute(f"UPDATE {table} SET rolled_back=0 WHERE decision_id=?", (decision_id,))
    conn.execute("UPDATE preference_relations SET active=1 WHERE decision_id=?", (decision_id,))
    conn.commit()


_SIGNAL_COLUMNS = {
    "retrieved": "retrieved_count", "exposed": "exposed_count",
    "executed": "executed_count", "confirmed": "confirmed_count",
    "rejected": "rejected_count", "contradiction": "contradiction_count",
}


def record_claim_signals(conn: sqlite3.Connection, claim_ids: Sequence[str], signal: str,
                         count: int = 1, *, event_key: Optional[str] = None) -> dict:
    """Atomically record one signal for several claims, with optional idempotency."""
    ensure_preference_schema(conn)
    if count <= 0:
        raise ValueError("count must be positive")
    column = _SIGNAL_COLUMNS.get(signal)
    if not column:
        raise ValueError(f"invalid claim signal: {signal}")
    unique_ids = list(dict.fromkeys(str(claim_id) for claim_id in claim_ids if str(claim_id)))
    if not unique_ids:
        return {"applied": False, "idempotent": False, "claim_ids": []}
    if event_key and conn.execute(
            "SELECT 1 FROM preference_signal_events WHERE event_key=?", (event_key,)).fetchone():
        return {"applied": False, "idempotent": True, "claim_ids": unique_ids}
    placeholders = ",".join("?" for _ in unique_ids)
    found = {row[0] for row in conn.execute(
        f"SELECT claim_id FROM preference_claims WHERE claim_id IN ({placeholders})", unique_ids
    ).fetchall()}
    missing = [claim_id for claim_id in unique_ids if claim_id not in found]
    if missing:
        raise KeyError(f"unknown claims: {missing}")
    salience_delta = count if signal == "confirmed" else -count if signal == "rejected" else -0.5 * count if signal == "contradiction" else 0.0
    confidence_delta = 0.05 * count if signal == "confirmed" else -0.1 * count if signal == "rejected" else -0.05 * count if signal == "contradiction" else 0.0
    now = _now()
    try:
        if event_key:
            conn.execute("""INSERT INTO preference_signal_events
                (event_key,signal,claim_ids_json,count,created_at) VALUES (?,?,?,?,?)""", (
                event_key, signal, json.dumps(unique_ids), count, now,
            ))
        for claim_id in unique_ids:
            conn.execute(f"""UPDATE preference_claims SET {column}={column}+?,
                salience=MAX(0.0,salience+?), confidence=MIN(1.0,MAX(0.0,confidence+?)),
                updated_at=? WHERE claim_id=?""",
                (count, salience_delta, confidence_delta, now, claim_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"applied": True, "idempotent": False, "claim_ids": unique_ids}


def record_claim_signal(conn: sqlite3.Connection, claim_id: str, signal: str, count: int = 1) -> dict:
    """Record one signal; exposure/retrieval never increase salience."""
    record_claim_signals(conn, [claim_id], signal, count)
    return get_claim_signals(conn, claim_id)


def set_claim_status(conn: sqlite3.Connection, claim_id: str, status: str, *,
                     reason: str = "", event_key: Optional[str] = None) -> dict:
    """Apply an auditable, idempotent lifecycle transition used by local callers."""
    ensure_preference_schema(conn)
    if status not in STATUSES:
        raise ValueError(f"invalid claim status: {status}")
    if event_key:
        prior = conn.execute(
            "SELECT from_status,to_status FROM preference_status_events WHERE event_key=?",
            (event_key,),
        ).fetchone()
        if prior:
            return {"changed": False, "idempotent": True, "from": prior[0], "to": prior[1]}
    row = conn.execute("SELECT status FROM preference_claims WHERE claim_id=?", (claim_id,)).fetchone()
    if not row:
        raise KeyError(f"unknown claim: {claim_id}")
    before = row[0]
    if before == status:
        return {"changed": False, "idempotent": False, "from": before, "to": status}
    now = _now()
    try:
        conn.execute("UPDATE preference_claims SET status=?,updated_at=? WHERE claim_id=?",
                     (status, now, claim_id))
        conn.execute("""INSERT INTO preference_status_events
            (event_id,event_key,claim_id,from_status,to_status,reason,created_at)
            VALUES (?,?,?,?,?,?,?)""", (
            str(uuid.uuid4()), event_key, claim_id, before, status, reason, now,
        ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"changed": True, "idempotent": False, "from": before, "to": status}


def get_claim_signals(conn: sqlite3.Connection, claim_id: str) -> dict:
    ensure_preference_schema(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute("""SELECT direction,salience,confidence,retrieved_count,
        exposed_count,executed_count,confirmed_count,rejected_count,contradiction_count
        FROM preference_claims WHERE claim_id=?""", (claim_id,)).fetchone()
    if not row:
        raise KeyError(f"unknown claim: {claim_id}")
    return dict(row)


def active_claims_for_context(conn: sqlite3.Connection, persona_id: str, *,
                              scope: Optional[dict] = None, top_k: int = 15,
                              token_budget: int = 1200,
                              tokenizer: Optional[Callable[[str], Sequence]] = None,
                              record_retrieval: bool = False) -> list[dict]:
    """Return only resolved active canonical claims within scope and budget."""
    ensure_preference_schema(conn)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""SELECT * FROM preference_claims
        WHERE persona_id=? AND status='active' ORDER BY
        CASE priority WHEN 'must' THEN 3 WHEN 'should' THEN 2 ELSE 1 END DESC, salience DESC,
        confirmed_count DESC, support_count DESC, updated_at DESC""", (persona_id,)).fetchall()
    requested_scope = _norm_scope(scope or {"kind": "global"})
    current_time = datetime.now(timezone.utc)
    selected, used = [], 0
    for row in rows:
        item = decode_claim_row(row)
        if not claim_is_current(item, at=current_time):
            continue
        item_scope = item["scope"]
        if not scope_applies(item_scope, requested_scope):
            continue
        text = f"{item['direction']} {item['subject']} {item['attribute']}: {item['action_or_value']}"
        cost = len(tokenizer(text)) if tokenizer else max(1, math.ceil(len(text) / 4))
        if used + cost > token_budget:
            continue
        item["scope"] = item_scope
        item["text"] = text
        item["token_cost"] = cost
        selected.append(item)
        used += cost
        if len(selected) >= top_k:
            break
    if record_retrieval:
        for item in selected:
            record_claim_signal(conn, item["claim_id"], "retrieved")
    return selected


def cluster_monitor(conn: sqlite3.Connection) -> dict:
    """Read-only fragmentation and persona/scope isolation report."""
    conn.row_factory = sqlite3.Row
    sizes = [row[0] for row in conn.execute(
        "SELECT COUNT(*) FROM cluster_members GROUP BY cluster_id"
    ).fetchall()]
    cross_persona = 0
    mixed_scope = 0
    if sizes:
        clusters = {}
        for row in conn.execute("""SELECT cm.cluster_id,m.tags FROM cluster_members cm
            JOIN memory_records m ON m.id=cm.memory_id"""):
            try:
                tags = json.loads(row[1] or "[]")
            except (TypeError, json.JSONDecodeError):
                tags = []
            persona = next((tag.split(":", 1)[1] for tag in tags if str(tag).startswith("persona:")), None)
            scope = next((tag.split(":", 1)[1] for tag in tags if str(tag).startswith("scope:")), None)
            bucket = clusters.setdefault(row[0], {"personas": set(), "scopes": set()})
            if persona:
                bucket["personas"].add(persona)
            if scope:
                bucket["scopes"].add(scope)
        cross_persona = sum(len(bucket["personas"]) > 1 for bucket in clusters.values())
        mixed_scope = sum(len(bucket["scopes"]) > 1 for bucket in clusters.values())
    sorted_sizes = sorted(sizes)
    p95 = sorted_sizes[min(len(sorted_sizes) - 1, math.ceil(len(sorted_sizes) * .95) - 1)] if sizes else 0
    uncrystallized = conn.execute(
        "SELECT COUNT(*) FROM knowledge_clusters WHERE skill_slug IS NULL OR skill_slug=''"
    ).fetchone()[0] if sizes else 0
    skill_sources = {
        row[0]: row[1] for row in conn.execute("""SELECT k.skill_slug,COUNT(cm.memory_id)
            FROM knowledge_clusters k LEFT JOIN cluster_members cm ON cm.cluster_id=k.cluster_id
            WHERE k.skill_slug IS NOT NULL AND k.skill_slug!='' GROUP BY k.skill_slug""").fetchall()
    } if sizes else {}
    evolution_counts = {row[0]: row[1] for row in conn.execute(
        "SELECT status,COUNT(*) FROM evolution_records GROUP BY status"
    ).fetchall()} if sizes else {}
    return {"cluster_count": len(sizes), "singleton_ratio": (sum(x == 1 for x in sizes) / len(sizes)) if sizes else 0.0,
            "average_size": (sum(sizes) / len(sizes)) if sizes else 0.0, "p95_size": p95,
            "uncrystallized_ratio": (uncrystallized / len(sizes)) if sizes else 0.0,
            "cross_persona_cluster_count": cross_persona, "mixed_scope_cluster_count": mixed_scope,
            "skill_source_memory_counts": skill_sources,
            "recrystallization_success_count": evolution_counts.get("recrystallized", 0),
            "recrystallization_rejected_count": sum(
                count for status, count in evolution_counts.items() if "reject" in status
            )}
