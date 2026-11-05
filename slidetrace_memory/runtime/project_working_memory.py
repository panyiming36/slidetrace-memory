"""Governed project working memory for V20.

The durable V19 canonical store remains the long-term memory.  This module
adds a project-scoped, auditable layer with a different lifetime:

* long-term memory is retrieved at project start; hosts may load a new page
  context once without repeating retrieval on later rounds;
* round feedback is normalized and governed inside the project without a new
  embedding or retrieval call;
* every fresh generator session receives only the current active working set;
* only verified-and-confirmed stable claims are promoted to long-term memory
  when the project closes.

No row is destructively deleted.  Superseded, retracted and rejected material
stays available in the project archive together with its evidence and decision
trail.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__ == "slidetrace_memory.runtime":
    from ..core.preference_governance import (
        PreferenceClaim, claim_fingerprint, claim_is_current, decode_claim_row,
        normalize_claim, resolve_conflict, scope_applies, scope_specificity,
        scopes_compatible,
    )
else:
    from engine.core.preference_governance import (
        PreferenceClaim, claim_fingerprint, claim_is_current, decode_claim_row,
        normalize_claim, resolve_conflict, scope_applies, scope_specificity,
        scopes_compatible,
    )

if __package__:
    from .memory_out import build_memory_out_variants
    from .semantic_memory import RELATION_MIN_CONFIDENCE
    from .tokenization import count_tokens
else:
    from memory_out import build_memory_out_variants
    from semantic_memory import RELATION_MIN_CONFIDENCE
    from tokenization import count_tokens


WORKING_STATUSES = {"active", "superseded", "retracted", "conflicted", "archived"}
PERSISTENCE = {"session_only", "candidate_long_term", "confirmed_long_term"}
MERGE_RELATIONS = {"paraphrase_duplicate", "reinforcement"}
REPLACE_RELATIONS = {"refinement", "subsumes", "contradiction", "temporal_update"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_claim_payload(value: PreferenceClaim | dict) -> dict:
    claim = value if isinstance(value, PreferenceClaim) else PreferenceClaim(**value)
    data = normalize_claim(claim)
    # Database-only and binary fields must never leak into a working package.
    return {key: data[key] for key in (
        "persona_id", "subject", "attribute", "action_or_value", "direction",
        "scope", "conditions", "exceptions", "priority", "valid_from", "valid_to",
        "source_type", "extraction_confidence", "confidence", "salience", "tags",
    )}


def _claim_from_long_term(item: dict, persona_id: str) -> dict:
    return _safe_claim_payload({
        "persona_id": persona_id,
        "subject": item.get("subject", "preference"),
        "attribute": item.get("attribute", "instruction"),
        "action_or_value": item.get("action_or_value") or item.get("text", ""),
        "direction": item.get("direction", "prefer"),
        "scope": item.get("scope") or {"kind": "global"},
        "conditions": item.get("conditions", []),
        "exceptions": item.get("exceptions", []),
        "valid_from": item.get("valid_from"),
        "valid_to": item.get("valid_to"),
        "priority": item.get("priority", "soft"),
        "source_type": item.get("source_type", "explicit_feedback"),
        "extraction_confidence": item.get("extraction_confidence", 1.0),
        "confidence": item.get("confidence", 0.5),
        "salience": item.get("salience", 0.0),
        "tags": item.get("tags", []),
    })


def ensure_project_working_schema(conn: sqlite3.Connection) -> None:
    """Create additive V20 tables without changing V19 canonical tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS project_sessions (
            session_id TEXT PRIMARY KEY,
            persona_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            task_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            satisfied INTEGER,
            final_round INTEGER,
            long_term_retrieval_id TEXT,
            long_term_retrieval_count INTEGER NOT NULL DEFAULT 0,
            summary_json TEXT,
            created_at TEXT NOT NULL,
            closed_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_project_session_open
            ON project_sessions(persona_id, task_id, session_id);

        CREATE TABLE IF NOT EXISTS project_context_retrievals (
            session_id TEXT NOT NULL,
            scope_json TEXT NOT NULL,
            retrieval_id TEXT NOT NULL,
            long_term_claim_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY(session_id, scope_json),
            FOREIGN KEY(session_id) REFERENCES project_sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS project_working_claims (
            working_claim_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            canonical_claim_id TEXT,
            fingerprint TEXT NOT NULL,
            claim_json TEXT NOT NULL,
            origin TEXT NOT NULL,
            persistence TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            introduced_round INTEGER NOT NULL DEFAULT 0,
            last_exposed_round INTEGER,
            support_count INTEGER NOT NULL DEFAULT 1,
            exposed_count INTEGER NOT NULL DEFAULT 0,
            executed_count INTEGER NOT NULL DEFAULT 0,
            confirmed_count INTEGER NOT NULL DEFAULT 0,
            rejected_count INTEGER NOT NULL DEFAULT 0,
            contradiction_count INTEGER NOT NULL DEFAULT 0,
            superseded_by TEXT,
            promoted_claim_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES project_sessions(session_id),
            UNIQUE(session_id, fingerprint)
        );
        CREATE INDEX IF NOT EXISTS idx_project_working_active
            ON project_working_claims(session_id, status, introduced_round);

        CREATE TABLE IF NOT EXISTS project_working_evidence (
            evidence_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            working_claim_id TEXT,
            round_no INTEGER NOT NULL,
            evidence_text TEXT NOT NULL,
            ingestion_key TEXT UNIQUE,
            classification TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS project_working_decisions (
            decision_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            round_no INTEGER NOT NULL,
            new_working_claim_id TEXT,
            candidate_working_claim_id TEXT,
            proposed_relation TEXT NOT NULL,
            proposed_confidence REAL NOT NULL,
            final_action TEXT NOT NULL,
            reason TEXT NOT NULL,
            proposal_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS project_working_signal_events (
            event_key TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            signal TEXT NOT NULL,
            working_claim_ids_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()


class ProjectWorkingMemory:
    """Project-scoped governance above a :class:`LocalMemoryPlugin`."""

    def __init__(self, plugin, persona_id: str, *, token_limit: int | None = None,
                 tool_store=None):
        if token_limit is not None and token_limit <= 0:
            raise ValueError("token_limit must be positive")
        self.plugin = plugin
        self.persona_id = str(persona_id)
        self.db_path = Path(plugin.db_path)
        self.token_limit = int(token_limit) if token_limit is not None else None
        self.tool_store = tool_store
        self.session_id: str | None = None
        self.task: dict | None = None
        self._governance_claims: dict[str, dict] = {}
        with self._connect() as conn:
            ensure_project_working_schema(conn)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _require_open(self) -> str:
        if not self.session_id:
            raise RuntimeError("project working memory has not been started")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM project_sessions WHERE session_id=?",
                (self.session_id,),
            ).fetchone()
        if not row or row[0] != "open":
            raise RuntimeError("project working memory is not open")
        return self.session_id

    def _context_scope(self, task: dict) -> dict:
        """Combine the project boundary with the current page identifiers."""
        requested = dict((self.task or {}).get("scope") or {})
        requested.update(task.get("scope") or {})
        for field in ("domain", "audience", "task_id", "project_id", "page_id", "page_type"):
            value = task.get(field)
            if value is not None and str(value).strip():
                requested[field] = value
        if not requested.get("project_id") and requested.get("task_id"):
            requested["project_id"] = requested["task_id"]
        requested = dict(sorted(
            (str(key).strip().lower(), " ".join(str(value).strip().lower().split()))
            for key, value in requested.items() if value is not None and str(value).strip()))
        requested.setdefault("kind", "global")
        if requested["kind"] == "global" and any(
                requested.get(field) for field in ("task_id", "project_id", "page_id", "page_type")):
            requested["kind"] = "task"
        return requested

    def _load_governance_claims(self, conn, task: dict, session_id: str) -> dict[str, dict]:
        requested = self._context_scope(task)
        current_time = datetime.now(timezone.utc)
        governance = {}
        for row in conn.execute("""SELECT claim_id,persona_id,subject,attribute,
                action_or_value,direction,scope_json,conditions_json,exceptions_json,
                priority,valid_from,valid_to,source_type,extraction_confidence,
                confidence,salience,tags_json FROM preference_claims
                WHERE persona_id=? AND status='active' ORDER BY claim_id""", (self.persona_id,)):
            item = decode_claim_row(row)
            if not claim_is_current(item, at=current_time) or not scope_applies(item["scope"], requested):
                continue
            working_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL, f"{session_id}:long-term:{item['claim_id']}"))
            governance[working_id] = {"canonical_claim_id": item["claim_id"],
                                      "claim": _claim_from_long_term(item, self.persona_id)}
        return governance

    def start_project(self, task: dict, *, session_id: str | None = None) -> dict:
        if self.session_id is not None:
            raise RuntimeError("another project is already active")
        if not task.get("task_id") or not task.get("topic"):
            raise ValueError("task_id and topic are required")
        sid = session_id or (
            f"v20:{self.persona_id}:{task['task_id']}:{uuid.uuid4().hex[:12]}"
        )
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM project_sessions WHERE session_id=?", (sid,)).fetchone():
                raise ValueError("project session already exists; use resume_project")
        retrieval_id = f"{sid}:long-term-start"
        # Default experiment callers retrieve once. Hosts can explicitly load a
        # new page context later; exposure is recorded only after budget selection.
        long_term = self.plugin.retrieve(
            self.persona_id,
            task.get("memory_query") or task["topic"],
            scope=self._context_scope(task),
            top_k=15,
            min_score=0.35,
            token_limit=self.token_limit,
            retrieval_id=retrieval_id,
            record_exposure=False,
        )
        now = _now()
        with self._connect() as conn:
            ensure_project_working_schema(conn)
            # This is a local metadata view, not a second semantic retrieval.
            # Omitted claims become working rows only when feedback links one.
            governance = self._load_governance_claims(conn, task, sid)
            conn.execute(
                """INSERT INTO project_sessions
                   (session_id,persona_id,task_id,task_json,status,
                    long_term_retrieval_id,long_term_retrieval_count,created_at)
                   VALUES (?,?,?,?,?,?,1,?)""",
                (sid, self.persona_id, task["task_id"], _json(task), "open",
                 retrieval_id, now),
            )
            conn.execute("""INSERT INTO project_context_retrievals
                (session_id,scope_json,retrieval_id,long_term_claim_count,created_at)
                VALUES (?,?,?,?,?)""", (
                sid, _json(self._context_scope(task)), retrieval_id,
                len(long_term.get("claims", [])), now,
            ))
            for item in long_term.get("claims", []):
                working_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{sid}:long-term:{item['claim_id']}",
                ))
                payload = governance.get(working_id, {}).get(
                    "claim", _claim_from_long_term(item, self.persona_id))
                conn.execute(
                    """INSERT INTO project_working_claims
                       (working_claim_id,session_id,canonical_claim_id,fingerprint,
                        claim_json,origin,persistence,status,introduced_round,
                        created_at,updated_at)
                       VALUES (?,?,?,?,?,'long_term','confirmed_long_term',
                               'active',0,?,?)""",
                    (working_id, sid, item["claim_id"], claim_fingerprint(payload),
                     _json(payload), now, now),
                )
            conn.commit()
        self.session_id = sid
        self.task = dict(task)
        self._governance_claims = governance
        return {
            "session_id": sid,
            "task_id": task["task_id"],
            "long_term_retrieval_id": retrieval_id,
            "long_term_claim_count": len(long_term.get("claims", [])),
            "long_term_retrieval_count": 1,
            "governance_candidate_count": len(governance),
        }

    def refresh_context(self, task: dict) -> dict:
        """Load one new page scope once, preserving all existing working decisions."""
        sid = self._require_open()
        if self.task and task.get("task_id") != self.task.get("task_id"):
            raise RuntimeError("project/task mismatch")
        requested = self._context_scope(task)
        scope_json = _json(requested)
        with self._connect() as conn:
            context = conn.execute("""SELECT retrieval_id,long_term_claim_count
                FROM project_context_retrievals WHERE session_id=? AND scope_json=?""",
                (sid, scope_json)).fetchone()
            session = conn.execute("SELECT * FROM project_sessions WHERE session_id=?", (sid,)).fetchone()
            if context is not None:
                return {"loaded": False, "session_id": sid, "scope": requested,
                        "long_term_retrieval_id": context["retrieval_id"],
                        "long_term_claim_count": context["long_term_claim_count"],
                        "long_term_retrieval_count": session["long_term_retrieval_count"],
                        "added_claim_count": 0}
            # Older open sessions already performed their base retrieval before
            # this table existed. Register that scope without retrieving again.
            if requested == self._context_scope(self.task):
                count = conn.execute("""SELECT COUNT(*) FROM project_working_claims
                    WHERE session_id=? AND origin='long_term'""", (sid,)).fetchone()[0]
                conn.execute("""INSERT INTO project_context_retrievals
                    (session_id,scope_json,retrieval_id,long_term_claim_count,created_at)
                    VALUES (?,?,?,?,?)""", (sid, scope_json, session["long_term_retrieval_id"], count, _now()))
                conn.commit()
                return {"loaded": False, "session_id": sid, "scope": requested,
                        "long_term_retrieval_id": session["long_term_retrieval_id"],
                        "long_term_claim_count": count,
                        "long_term_retrieval_count": session["long_term_retrieval_count"],
                        "added_claim_count": 0}
        retrieval_id = f"{sid}:context:{scope_json}"
        long_term = self.plugin.retrieve(
            self.persona_id, task.get("memory_query") or task.get("topic") or self.task["topic"],
            scope=requested, top_k=15, min_score=0.35, token_limit=self.token_limit,
            retrieval_id=retrieval_id, record_exposure=False,
        )
        now = _now()
        added = 0
        with self._connect() as conn:
            governance = self._load_governance_claims(conn, task, sid)
            for item in long_term.get("claims", []):
                working_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{sid}:long-term:{item['claim_id']}"))
                payload = governance.get(working_id, {}).get(
                    "claim", _claim_from_long_term(item, self.persona_id))
                # Inactive rows stay inactive: refreshing another page cannot
                # undo a retraction or a replacement made inside this project.
                cursor = conn.execute("""INSERT OR IGNORE INTO project_working_claims
                    (working_claim_id,session_id,canonical_claim_id,fingerprint,claim_json,
                     origin,persistence,status,introduced_round,created_at,updated_at)
                    VALUES (?,?,?,?,?,'long_term','confirmed_long_term','active',0,?,?)""",
                    (working_id, sid, item["claim_id"], claim_fingerprint(payload),
                     _json(payload), now, now))
                added += cursor.rowcount
            conn.execute("""INSERT INTO project_context_retrievals
                (session_id,scope_json,retrieval_id,long_term_claim_count,created_at)
                VALUES (?,?,?,?,?)""", (sid, scope_json, retrieval_id,
                                       len(long_term.get("claims", [])), now))
            conn.execute("""UPDATE project_sessions
                SET long_term_retrieval_count=long_term_retrieval_count+1 WHERE session_id=?""", (sid,))
            count = conn.execute("SELECT long_term_retrieval_count FROM project_sessions WHERE session_id=?",
                                 (sid,)).fetchone()[0]
            conn.commit()
        self._governance_claims.update(governance)
        return {"loaded": True, "session_id": sid, "scope": requested,
                "long_term_retrieval_id": retrieval_id,
                "long_term_claim_count": len(long_term.get("claims", [])),
                "long_term_retrieval_count": count, "added_claim_count": added,
                "governance_candidate_count": len(self._governance_claims)}

    def resume_project(self, session_id: str) -> dict:
        """Attach an open persisted project without retrieval or new memory rows."""
        sid = str(session_id).strip()
        if not sid:
            raise ValueError("session_id is required")
        if self.session_id is not None and self.session_id != sid:
            raise RuntimeError("another project is already active")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM project_sessions WHERE session_id=?", (sid,)).fetchone()
            if row is None:
                raise KeyError(f"unknown project session: {sid}")
            if row["persona_id"] != self.persona_id:
                raise ValueError("project session belongs to another persona")
            if row["status"] != "open":
                raise RuntimeError("only an open project can be resumed")
            task = json.loads(row["task_json"])
            governance = self._load_governance_claims(conn, task, sid)
            for context in conn.execute("""SELECT scope_json FROM project_context_retrievals
                    WHERE session_id=?""", (sid,)):
                context_task = {**task, "scope": json.loads(context["scope_json"])}
                governance.update(self._load_governance_claims(conn, context_task, sid))
            long_term_count = conn.execute("""SELECT COUNT(*) FROM project_working_claims
                WHERE session_id=? AND origin='long_term'""", (sid,)).fetchone()[0]
            last_round = conn.execute("""SELECT MAX(round_no) FROM (
                SELECT round_no FROM project_working_evidence WHERE session_id=?
                UNION ALL SELECT introduced_round FROM project_working_claims WHERE session_id=?
                UNION ALL SELECT last_exposed_round FROM project_working_claims WHERE session_id=?
            )""", (sid, sid, sid)).fetchone()[0]
        already_attached = self.session_id == sid
        self.session_id = sid
        self.task = task
        self._governance_claims = governance
        return {"session_id": sid, "task_id": row["task_id"], "status": "open",
                "resumed": True, "idempotent": already_attached,
                "last_round": int(last_round or 0),
                "long_term_retrieval_id": row["long_term_retrieval_id"],
                "long_term_retrieval_count": row["long_term_retrieval_count"],
                "long_term_claim_count": long_term_count,
                "governance_candidate_count": len(governance)}

    def _active_rows(self, *, include_inactive_time: bool = False) -> list[dict]:
        sid = self._require_open()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM project_working_claims
                   WHERE session_id=? AND status='active'""",
                (sid,),
            ).fetchall()
        result = []
        current_time = datetime.now(timezone.utc)
        for row in rows:
            item = dict(row)
            item["claim"] = json.loads(item.pop("claim_json"))
            if not include_inactive_time and not claim_is_current(item["claim"], at=current_time):
                continue
            result.append(item)
        return result

    @staticmethod
    def _rank_key(row: dict) -> tuple:
        claim = row["claim"]
        priority = {"must": 0, "should": 1, "soft": 2}
        origin = {"round_feedback": 0, "task_constraint": 1, "long_term": 2}
        return (
            origin.get(row["origin"], 3),
            -int(row["introduced_round"]),
            priority.get(claim.get("priority", "soft"), 3),
            -int(row["confirmed_count"]),
            -int(row["executed_count"]),
            row["working_claim_id"],
        )

    def get_generation_memory(self, task: dict, round_no: int) -> dict:
        sid = self._require_open()
        if self.task and task.get("task_id") != self.task.get("task_id"):
            raise RuntimeError("project/task mismatch")
        active_rows = sorted(self._active_rows(include_inactive_time=True), key=self._rank_key)
        current_time = datetime.now(timezone.utc)
        rows = [row for row in active_rows if claim_is_current(row["claim"], at=current_time)]
        time_filtered = [{"id": row["working_claim_id"], "filter_reasons": ["outside_validity_window"]}
                         for row in active_rows if not claim_is_current(row["claim"], at=current_time)]
        effective_task = {**task, "scope": self._context_scope(task)}
        rows, boundary_filtered = self._apply_scoped_exceptions(rows, effective_task)
        boundary_filtered = time_filtered + boundary_filtered
        claims = []
        for row in rows:
            claim = dict(row["claim"])
            claim.update({
                "claim_id": row["working_claim_id"],
                "working_origin": row["origin"],
                "introduced_round": row["introduced_round"],
                "persistence": row["persistence"],
                "canonical_claim_id": row["canonical_claim_id"],
            })
            claims.append(claim)

        tool_out = ""
        tool_ids: list[str] = []
        tool_tokens = 0
        if self.tool_store is not None:
            tool_budget = (
                180 if self.token_limit is None
                else min(180, max(1, self.token_limit // 5))
            )
            tool_result = self.tool_store.retrieve(
                tool_name=str(task.get("tool_name", "ppt-master")),
                operation=str(task.get("tool_operation", "generate_and_render")),
                context=str(task.get("topic", "")),
                token_limit=tool_budget,
            )
            tool_out = tool_result["memory_out"]
            tool_ids = tool_result["experience_ids"]
            tool_tokens = tool_result["tokens"]

        separator_reserve = 8 if tool_out else 0
        working_budget = (
            None if self.token_limit is None
            else max(1, self.token_limit - tool_tokens - separator_reserve)
        )
        variants = build_memory_out_variants(
            claims, token_limit=working_budget,
            partition_resolver=self.plugin.partition_resolver,
        )
        self._persist_working_partitions(variants["partition_decisions"])
        exposed_ids = variants["claim_ids"]
        memory_out = variants["partitioned"]
        if tool_out:
            memory_out = memory_out.rstrip() + "\n\n" + tool_out
        actual_tokens = count_tokens(memory_out) if tool_out else variants["partitioned_tokens"]
        if self.token_limit is not None and actual_tokens > self.token_limit:
            raise RuntimeError(
                f"combined working/tool memory exceeds budget: {actual_tokens} > "
                f"{self.token_limit}")

        scope_json = _json(effective_task["scope"])
        event_key = (f"{sid}:round:{round_no}:scope:{scope_json}:"
                     f"claims:{_json(sorted(exposed_ids))}:exposed")
        self._record_signal(exposed_ids, "exposed", event_key=event_key,
                            round_no=round_no)
        canonical_ids = [
            row["canonical_claim_id"] for row in rows
            if row["working_claim_id"] in exposed_ids and row["canonical_claim_id"]
        ]
        if canonical_ids:
            self.plugin.record_exposure(canonical_ids, event_key=event_key + ":long-term")

        selected = [item for item in claims if item["claim_id"] in exposed_ids]
        session = self._session_row()
        return {
            "memory_out": memory_out,
            "control": variants["control"],
            "claims": selected,
            "trace": {
                "retrieval_id": f"{sid}:working-round:{round_no}:scope:{scope_json}",
                "source": "project_working_memory",
                "long_term_retrieval_id": session["long_term_retrieval_id"],
                "long_term_retrieval_count": session["long_term_retrieval_count"],
                "scope": effective_task["scope"],
                "candidate_ids": [item["claim_id"] for item in claims],
                "injected_ids": exposed_ids,
                "tool_experience_ids": tool_ids,
                "injected_tokens": actual_tokens,
                "token_limit": self.token_limit,
                "partition_decisions": variants["partition_decisions"],
                "filtered": boundary_filtered + [
                    {"id": item["claim_id"], "filter_reasons": ["token_budget"]}
                    for item in claims if item["claim_id"] not in exposed_ids
                ],
            },
        }

    def _apply_scoped_exceptions(self, rows: list[dict], task: dict) -> tuple[list[dict], list[dict]]:
        """Expose a validated audience exception without deleting its default."""
        requested = task.get("scope") or {"kind": "global"}
        applicable = [row for row in rows if scope_applies(row["claim"]["scope"], requested)]
        hidden = {row["working_claim_id"]: "scope_mismatch" for row in rows
                  if row not in applicable}
        by_working = {row["working_claim_id"]: row for row in applicable}
        by_canonical = {row["canonical_claim_id"]: row for row in applicable
                        if row["canonical_claim_id"]}
        pairs = []
        with self._connect() as conn:
            for decision in conn.execute(
                    "SELECT new_working_claim_id,candidate_working_claim_id "
                    "FROM project_working_decisions WHERE session_id=? "
                    "AND final_action='conditionalize_and_coexist'", (self.session_id,)):
                if decision[0] in by_working and decision[1] in by_working:
                    pairs.append((by_working[decision[0]], by_working[decision[1]]))
            if by_canonical:
                for relation in conn.execute(
                        "SELECT claim_a,claim_b FROM preference_relations "
                        "WHERE relation_type='exception' AND active=1"):
                    if relation[0] in by_canonical and relation[1] in by_canonical:
                        pairs.append((by_canonical[relation[0]], by_canonical[relation[1]]))
        for first, second in pairs:
            a, b = first["claim"], second["claim"]
            if not scopes_compatible(a["scope"], b["scope"]):
                continue
            if any(a.get(field, []) != b.get(field, []) for field in ("conditions", "exceptions")):
                # Scope fields cannot establish arbitrary free-text conditions.
                continue
            left, right = scope_specificity(a["scope"]), scope_specificity(b["scope"])
            if left == right:
                # Free-text conditions require semantic interpretation; do
                # not guess their truth from incidental keyword overlap.
                continue
            default = first if left < right else second
            hidden[default["working_claim_id"]] = "scoped_exception_applies"
        return ([row for row in applicable if row["working_claim_id"] not in hidden],
                [{"id": claim_id, "filter_reasons": [reason]} for claim_id, reason in hidden.items()])

    def _session_row(self) -> dict:
        sid = self._require_open()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM project_sessions WHERE session_id=?", (sid,),
            ).fetchone()
        return dict(row)

    def _semantic_candidates(self) -> list[dict]:
        rows = sorted(self._active_rows(), key=self._rank_key)
        with self._connect() as conn:
            known = {row[0] for row in conn.execute(
                "SELECT working_claim_id FROM project_working_claims WHERE session_id=?",
                (self.session_id,))}
        rows.extend({"working_claim_id": claim_id, "claim": item["claim"],
                     "candidate_source": "canonical_governance"}
                    for claim_id, item in self._governance_claims.items() if claim_id not in known)
        return [{
            "claim_id": row["working_claim_id"],
            "subject": row["claim"]["subject"],
            "attribute": row["claim"]["attribute"],
            "action_or_value": row["claim"]["action_or_value"],
            "direction": row["claim"]["direction"],
            "scope": row["claim"]["scope"],
            "conditions": row["claim"].get("conditions", []),
            "exceptions": row["claim"].get("exceptions", []),
            "priority": row["claim"]["priority"],
            # Working sets are deliberately small.  Candidate enumeration is
            # complete and no fake embedding similarity is claimed.
            "candidate_source": row.get("candidate_source", "project_working"),
        } for row in rows]

    def _persist_working_partitions(self, decisions: list[dict]) -> None:
        tags = {"planning": "structure", "drawing": "style", "instructions": "instructions"}
        pending = [item for item in decisions if item.get("source") == "llm"]
        if not pending:
            return
        with self._connect() as conn:
            for item in pending:
                row = conn.execute("""SELECT claim_json FROM project_working_claims
                    WHERE session_id=? AND working_claim_id=?""",
                    (self.session_id, item["claim_id"])).fetchone()
                if row:
                    claim = json.loads(row[0])
                    claim["tags"] = sorted(set([*claim.get("tags", []), tags[item["bucket"]]]))
                    conn.execute("""UPDATE project_working_claims SET claim_json=?
                        WHERE working_claim_id=?""", (_json(claim), item["claim_id"]))
            conn.commit()

    def ingest_raw_feedback(self, task: dict, round_no: int, feedback: str, *,
                            ingestion_key: str | None = None,
                            session_id: str | None = None) -> dict:
        sid = self._require_open()
        if self.task and task.get("task_id") != self.task.get("task_id"):
            raise RuntimeError("project/task mismatch")
        if self.plugin.semantic_analyzer is None:
            raise RuntimeError("semantic_analyzer is required for project feedback")
        if not feedback or not feedback.strip():
            raise ValueError("feedback is required")
        clean = self.plugin._sanitize(feedback)
        key = ingestion_key or (
            f"v20-working:{sid}:{round_no}:{hashlib.sha256(clean.encode()).hexdigest()}"
        )
        audit_key = f"v20-working-batch:{sid}:{key}"
        with self._connect() as conn:
            audit = conn.execute(
                "SELECT proposal_json,result_json FROM semantic_ingestion_decisions "
                "WHERE ingestion_key=?", (audit_key,),
            ).fetchone()
            if audit:
                prior_result = json.loads(audit["result_json"])
                if not prior_result.get("pending"):
                    result = {**prior_result, "idempotent": True}
                    if "results" in result:
                        result["results"] = [{**item, "idempotent": True}
                                             for item in result["results"]]
                    return result
            else:
                # Keep completed single-item records from earlier versions usable.
                prior = conn.execute(
                    "SELECT session_id,working_claim_id,classification "
                    "FROM project_working_evidence WHERE ingestion_key=?", (key,),
                ).fetchone()
                if prior:
                    if prior["session_id"] != sid:
                        raise ValueError("feedback ingestion key belongs to another project")
                    return {"stored": bool(prior["working_claim_id"]), "idempotent": True,
                            "working_claim_id": prior["working_claim_id"],
                            "classification": prior["classification"]}

        scope = dict(task.get("scope") or {"kind": "global"})
        for name in ("task_id", "project_id", "page_id", "page_type", "domain", "audience"):
            if name not in scope and task.get(name):
                scope[name] = task[name]
        if "page_ids" not in scope and task.get("page_ids"):
            scope["page_ids"] = list(task["page_ids"])
        context = (
            f"task:{task['task_id']};round:{round_no};domain:{task.get('domain', '')};"
            f"ingestion_session:{session_id or ''}"
        )
        candidates = self._semantic_candidates()
        proposal = (json.loads(audit["proposal_json"]) if audit else
                    self.plugin.semantic_analyzer.analyze(
                        feedback=clean, persona_id=self.persona_id, scope=scope,
                        context=context, candidates=candidates))
        classification = proposal["classification"]
        # Save the validated batch before applying its items. A retry can reuse
        # the same proposal and skip completed items without dropping the rest.
        if not audit:
            self.plugin._write_semantic_audit(
                ingestion_key=audit_key, feedback=clean, classification=classification,
                candidates=candidates, proposal=proposal, result={"pending": True})
        if not proposal.get("claims"):
            evidence_id = str(uuid.uuid4())
            with self._connect() as conn:
                prior = conn.execute(
                    "SELECT session_id FROM project_working_evidence WHERE ingestion_key=?", (key,),
                ).fetchone()
                if prior and prior["session_id"] != sid:
                    raise ValueError("feedback ingestion key belongs to another project")
                if not prior:
                    conn.execute(
                        """INSERT INTO project_working_evidence
                           (evidence_id,session_id,working_claim_id,round_no,evidence_text,
                            ingestion_key,classification,created_at)
                           VALUES (?,?,NULL,?,?,?,?,?)""",
                        (evidence_id, sid, round_no, clean, key, classification, _now()),
                    )
                    self._write_decision(
                        conn, round_no=round_no, new_id=None, candidate_id=None,
                        relation="unrelated", confidence=0.0,
                        action="reject_non_preference",
                        reason=str(proposal.get("reason", "")), proposal=proposal,
                    )
                conn.commit()
            result = {"stored": False, "classification": classification,
                      "reason": proposal.get("reason", ""), "idempotent": bool(prior)}
        else:
            results = []
            multiple = len(proposal["claims"]) > 1
            for index, item in enumerate(proposal["claims"]):
                raw_claim = dict(item)
                item_classification = raw_claim.pop("classification", classification)
                update_priority = raw_claim.pop("priority_update", False)
                partition = raw_claim.pop("partition")
                partition_reason = raw_claim.pop("partition_reason")
                relation = dict(raw_claim.pop("relation"))
                raw_claim.setdefault("conditions", [])
                raw_claim.setdefault("exceptions", [])
                raw_claim.setdefault("source_type", "explicit_feedback")
                raw_claim.setdefault("extraction_confidence", raw_claim.get("confidence", 0.7))
                raw_claim["tags"] = sorted(set([
                    *raw_claim.get("tags", []),
                    {"planning": "structure", "drawing": "style",
                     "instructions": "instructions"}[partition],
                    f"partition_reason:{partition_reason}",
                    "semantic:llm_extracted", "lifetime:project_working",
                ]))
                # Earlier items may already have replaced a recalled candidate.
                # Follow its local replacement; all axis/scope checks still run
                # inside ingest_claim before any merge or supersession.
                candidate_id = str(relation.get("candidate_id") or "")
                visited = set()
                with self._connect() as conn:
                    while candidate_id and candidate_id not in visited:
                        visited.add(candidate_id)
                        row = conn.execute(
                            "SELECT status,superseded_by FROM project_working_claims "
                            "WHERE session_id=? AND working_claim_id=?", (sid, candidate_id),
                        ).fetchone()
                        if not row or row["status"] == "active":
                            break
                        if row["status"] != "superseded" or not row["superseded_by"]:
                            candidate_id = ""
                            relation["type"] = "uncertain"
                            break
                        candidate_id = row["superseded_by"]
                relation["candidate_id"] = candidate_id
                result_item = self.ingest_claim(
                    task, round_no, clean,
                    PreferenceClaim(persona_id=self.persona_id, **raw_claim),
                    relation=relation, proposal=proposal,
                    ingestion_key=f"{key}:claim:{index}" if multiple else key,
                    update_priority=update_priority,
                    classification=item_classification,
                )
                results.append(result_item)
            result = (results[0] if not multiple else {
                "stored": any(item["stored"] for item in results),
                "classification": classification, "results": results,
                "working_claim_ids": list(dict.fromkeys(
                    item["working_claim_id"] for item in results if item.get("working_claim_id"))),
                "idempotent": all(item.get("idempotent", False) for item in results),
                "embedding_calls": 0,
            })
        self.plugin._write_semantic_audit(
            ingestion_key=audit_key, feedback=clean, classification=classification,
            candidates=candidates, proposal=proposal, result=result)
        return result

    def ingest_claim(self, task: dict, round_no: int, feedback: str,
                     claim: PreferenceClaim | dict, *, relation: dict | None = None,
                     proposal: dict | None = None,
                     persistence: str = "candidate_long_term",
                     ingestion_key: str | None = None,
                     update_priority: bool = False,
                     classification: str = "stable_preference") -> dict:
        """Govern a structured claim without embedding it."""
        sid = self._require_open()
        if self.task and task.get("task_id") != self.task.get("task_id"):
            raise RuntimeError("project/task mismatch")
        if classification not in {"stable_preference", "one_off_incident"}:
            raise ValueError("working claims must be stable or project-local requirements")
        if persistence not in PERSISTENCE:
            raise ValueError(f"invalid persistence: {persistence}")
        payload = _safe_claim_payload(claim)
        if classification == "one_off_incident":
            # A local edit may override a default inside this project, but it
            # must never be mistaken for an update to the durable default.
            if payload["scope"].get("kind") not in {"task", "project", "page"}:
                payload["scope"] = {**payload["scope"], "kind": "task",
                                    "task_id": str(task["task_id"]).strip().lower()}
                current_page = task.get("page_id") or (task.get("scope") or {}).get("page_id")
                if current_page:
                    payload["scope"].update({"kind": "page", "page_id":
                                            " ".join(str(current_page).strip().lower().split())})
            persistence = "session_only"
        elif payload["scope"].get("kind") in {"task", "project", "page"} or any(
                payload["scope"].get(name) for name in ("task_id", "project_id", "page_id")):
            persistence = "session_only"
        if update_priority and payload["source_type"] not in {"explicit_feedback", "edit"}:
            raise ValueError("priority changes require explicit feedback or an edit")
        if payload["persona_id"] != self.persona_id:
            raise ValueError("working claim cannot cross persona boundary")
        fingerprint = claim_fingerprint(payload)
        relation = relation or {"candidate_id": "", "type": "unrelated",
                                "confidence": 0.0, "explicit": False}
        relation_type = str(relation.get("type", "unrelated"))
        relation_confidence = float(relation.get("confidence", 0.0))
        # Treat JSON null as the absence of a candidate.  str(None) ==
        # "None" is truthy and would otherwise trigger a bogus database
        # lookup after semantic validation has already accepted the proposal.
        candidate_id = str(relation.get("candidate_id") or "")
        relation["candidate_id"] = candidate_id
        threshold = RELATION_MIN_CONFIDENCE.get(relation_type, 0.0)
        if (relation_type not in {"unrelated", "uncertain"}
                and relation_confidence < threshold) or (
                relation_type == "temporal_update" and not relation.get("explicit")):
            relation_type = "uncertain"
            candidate_id = ""
        key = ingestion_key or (f"v20-working:{sid}:{round_no}:{fingerprint}"
                                + (f":priority:{payload['priority']}" if update_priority else ""))
        now = _now()

        with self._connect() as conn:
            prior = conn.execute(
                "SELECT session_id,working_claim_id,classification FROM project_working_evidence "
                "WHERE ingestion_key=?", (key,),
            ).fetchone()
            if prior:
                if prior["session_id"] != sid:
                    raise ValueError("feedback ingestion key belongs to another project")
                return {"stored": bool(prior["working_claim_id"]), "created": False,
                        "working_claim_id": prior["working_claim_id"],
                        "classification": prior["classification"],
                        "idempotent": True, "embedding_calls": 0}
            candidate = None
            if candidate_id:
                candidate = conn.execute(
                    "SELECT * FROM project_working_claims "
                    "WHERE session_id=? AND working_claim_id=?",
                    (sid, candidate_id),
                ).fetchone()
                if candidate is None and candidate_id in self._governance_claims:
                    item = self._governance_claims[candidate_id]
                    previous = item["claim"]
                    if all(payload[field] == previous[field] for field in ("persona_id", "subject", "attribute")) \
                            and scopes_compatible(payload["scope"], previous["scope"]):
                        conn.execute("""INSERT INTO project_working_claims
                            (working_claim_id,session_id,canonical_claim_id,fingerprint,claim_json,
                             origin,persistence,status,introduced_round,created_at,updated_at)
                            VALUES (?,?,?,?,?,'long_term','confirmed_long_term','active',0,?,?)""",
                            (candidate_id, sid, item["canonical_claim_id"], claim_fingerprint(previous),
                             _json(previous), now, now))
                        candidate = conn.execute("SELECT * FROM project_working_claims WHERE working_claim_id=?",
                                                 (candidate_id,)).fetchone()
                    else:
                        relation_type, candidate_id = "uncertain", ""
                if candidate is None:
                    if candidate_id:
                        raise ValueError("relation candidate is not in the active project")
                elif candidate["status"] != "active":
                    raise ValueError("relation candidate must be active")
            if candidate is not None:
                previous = json.loads(candidate["claim_json"])
                same_boundary = all(payload.get(field) == previous.get(field) for field in (
                    "scope", "conditions", "exceptions"))
                same_axis = all(payload[field] == previous[field] for field in (
                    "persona_id", "subject", "attribute"))
                if not same_axis or not scopes_compatible(payload["scope"], previous["scope"]):
                    relation_type, candidate_id, candidate = "uncertain", "", None
                elif relation_type in MERGE_RELATIONS and (
                        not same_boundary or payload["direction"] != previous["direction"]):
                    relation_type, candidate_id, candidate = "uncertain", "", None
                elif relation_type in REPLACE_RELATIONS and not same_boundary:
                    # A more specific audience/condition is an exception, not
                    # permission to erase the default outside that boundary.
                    relation_type = "exception"
            exact = conn.execute(
                """SELECT working_claim_id,status,claim_json FROM project_working_claims
                   WHERE session_id=? AND fingerprint=?""", (sid, fingerprint),
            ).fetchone()
            if exact:
                target = exact["working_claim_id"]
                reactivated = exact["status"] != "active"
                if reactivated:
                    # A restored preference is a new current request. Its old
                    # execution/acceptance evidence cannot approve this version.
                    conn.execute(
                        """UPDATE project_working_claims SET executed_count=0,
                           confirmed_count=0,last_exposed_round=NULL,
                           origin='round_feedback',persistence=?,introduced_round=?
                           WHERE working_claim_id=?""", (persistence, round_no, target),
                    )
                priority_change = self._update_working_priority(
                    conn, target, json.loads(exact["claim_json"]), payload,
                    round_no, persistence) if update_priority else None
                conn.execute(
                    """UPDATE project_working_claims SET support_count=support_count+1,
                       status='active',superseded_by=NULL,updated_at=?
                       WHERE working_claim_id=?""", (now, target),
                )
                if (reactivated and candidate_id and candidate_id != target and
                        relation_type in REPLACE_RELATIONS):
                    candidate_status = conn.execute(
                        """SELECT status FROM project_working_claims
                           WHERE session_id=? AND working_claim_id=?""",
                        (sid, candidate_id),
                    ).fetchone()
                    if candidate_status and candidate_status["status"] == "active":
                        conn.execute(
                            """UPDATE project_working_claims SET status='superseded',
                               superseded_by=?,updated_at=? WHERE working_claim_id=?""",
                            (target, now, candidate_id),
                        )
                self._insert_evidence(conn, target, round_no, feedback, key,
                                      classification)
                self._write_decision(
                    conn, round_no=round_no, new_id=target, candidate_id=target,
                    relation="exact_duplicate", confidence=1.0,
                    action=("update_priority" if priority_change else
                            "reactivate_exact" if reactivated else "reinforce_existing"),
                    reason="deterministic fingerprint match",
                    proposal=({**(proposal or {}), "priority_change": priority_change}
                        if priority_change else proposal or {}),
                )
                conn.commit()
                return {"stored": True, "created": False, "merged": True,
                        "working_claim_id": target, "classification": classification,
                        "relation": "exact_duplicate", "reactivated": reactivated,
                        "idempotent": False,
                        "embedding_calls": 0}

            if candidate is not None and relation_type in MERGE_RELATIONS:
                target = candidate["working_claim_id"]
                priority_change = self._update_working_priority(
                    conn, target, previous, payload, round_no, persistence) if update_priority else None
                conn.execute(
                    """UPDATE project_working_claims SET support_count=support_count+1,
                       updated_at=? WHERE working_claim_id=?""", (now, target),
                )
                self._insert_evidence(conn, target, round_no, feedback, key,
                                      classification)
                self._write_decision(
                    conn, round_no=round_no, new_id=target if priority_change else None, candidate_id=target,
                    relation=relation_type, confidence=relation_confidence,
                    action="update_priority" if priority_change else "semantic_merge",
                    reason="high-confidence semantic equivalence",
                    proposal=({**(proposal or {}), "priority_change": priority_change}
                        if priority_change else proposal or {}),
                )
                conn.commit()
                return {"stored": True, "created": False, "merged": True,
                        "working_claim_id": target, "classification": classification,
                        "relation": relation_type, "idempotent": False,
                        "embedding_calls": 0}

            working_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sid}:{key}"))
            conn.execute(
                """INSERT INTO project_working_claims
                   (working_claim_id,session_id,canonical_claim_id,fingerprint,
                    claim_json,origin,persistence,status,introduced_round,
                    created_at,updated_at)
                   VALUES (?,?,?,?,?,'round_feedback',?,'active',?,?,?)""",
                (working_id, sid, None, fingerprint, _json(payload), persistence,
                 round_no, now, now),
            )
            self._insert_evidence(conn, working_id, round_no, feedback, key,
                                  classification)

            action = "add_active"
            if candidate is not None and relation_type in REPLACE_RELATIONS:
                old_id = candidate["working_claim_id"]
                conn.execute(
                    """UPDATE project_working_claims
                       SET status='superseded',superseded_by=?,updated_at=?
                       WHERE working_claim_id=?""", (working_id, now, old_id),
                )
                if relation_type == "contradiction":
                    conn.execute(
                        """UPDATE project_working_claims
                           SET contradiction_count=contradiction_count+1
                           WHERE working_claim_id IN (?,?)""", (old_id, working_id),
                    )
                action = "supersede_old_with_current_feedback"
            elif candidate is not None and relation_type == "exception":
                action = "conditionalize_and_coexist"
            elif relation_type == "uncertain":
                action = "add_active_relation_unresolved"

            self._write_decision(
                conn, round_no=round_no, new_id=working_id,
                candidate_id=candidate_id or None, relation=relation_type,
                confidence=relation_confidence, action=action,
                reason=("current explicit project feedback outranks an older working "
                        "claim" if action.startswith("supersede") else
                        "validated working-memory insertion"),
                proposal=proposal or {},
            )
            conn.commit()
        return {"stored": True, "created": True, "merged": False,
                "working_claim_id": working_id, "classification": classification,
                "relation": relation_type, "final_action": action,
                "persistence": persistence, "idempotent": False,
                "embedding_calls": 0}

    def _update_working_priority(self, conn, claim_id, previous, incoming, round_no, persistence) -> dict | None:
        if previous["priority"] == incoming["priority"]:
            return None
        counters = conn.execute("""SELECT executed_count,confirmed_count FROM project_working_claims
            WHERE working_claim_id=?""", (claim_id,)).fetchone()
        change = {"from": previous["priority"], "to": incoming["priority"],
                  "executed_before": counters[0], "confirmed_before": counters[1]}
        previous["priority"] = incoming["priority"]
        previous["source_type"] = incoming["source_type"]
        # Keep the canonical priority unchanged until this project is accepted.
        conn.execute("""UPDATE project_working_claims SET claim_json=?,origin='round_feedback',
            persistence=?,introduced_round=?
            WHERE working_claim_id=?""", (_json(previous), persistence, round_no, claim_id))
        return change

    def _insert_evidence(self, conn, claim_id: str | None, round_no: int,
                         feedback: str, ingestion_key: str,
                         classification: str) -> None:
        conn.execute(
            """INSERT INTO project_working_evidence
               (evidence_id,session_id,working_claim_id,round_no,evidence_text,
                ingestion_key,classification,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), self.session_id, claim_id, round_no, feedback,
             ingestion_key, classification, _now()),
        )

    def _write_decision(self, conn, *, round_no: int, new_id: str | None,
                        candidate_id: str | None, relation: str,
                        confidence: float, action: str, reason: str,
                        proposal: dict) -> None:
        conn.execute(
            """INSERT INTO project_working_decisions
               (decision_id,session_id,round_no,new_working_claim_id,
                candidate_working_claim_id,proposed_relation,proposed_confidence,
                final_action,reason,proposal_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), self.session_id, round_no, new_id, candidate_id,
             relation, confidence, action, reason, _json(proposal), _now()),
        )

    def _record_signal(self, claim_ids: Iterable[str], signal: str, *,
                       event_key: str, round_no: int | None = None) -> dict:
        sid = self._require_open()
        ids = list(dict.fromkeys(str(item) for item in claim_ids if str(item)))
        if signal not in {"exposed", "executed", "confirmed", "rejected"}:
            raise ValueError(f"invalid working-memory signal: {signal}")
        column = signal + "_count"
        with self._connect() as conn:
            prior = conn.execute(
                "SELECT 1 FROM project_working_signal_events WHERE event_key=?",
                (event_key,),
            ).fetchone()
            if prior:
                return {"updated": 0, "idempotent": True}
            known = {row[0] for row in conn.execute(
                "SELECT working_claim_id FROM project_working_claims WHERE session_id=?",
                (sid,),
            ).fetchall()}
            unknown = sorted(set(ids) - known)
            if unknown:
                raise KeyError(f"unknown working claims: {unknown}")
            for claim_id in ids:
                if signal == "exposed" and round_no is not None:
                    conn.execute(
                        f"""UPDATE project_working_claims SET {column}={column}+1,
                            last_exposed_round=?,updated_at=?
                            WHERE working_claim_id=?""",
                        (round_no, _now(), claim_id),
                    )
                else:
                    conn.execute(
                        f"""UPDATE project_working_claims SET {column}={column}+1,
                            updated_at=? WHERE working_claim_id=?""",
                        (_now(), claim_id),
                    )
            conn.execute(
                """INSERT INTO project_working_signal_events
                   (event_key,session_id,signal,working_claim_ids_json,created_at)
                   VALUES (?,?,?,?,?)""",
                (event_key, sid, signal, _json(ids), _now()),
            )
            conn.commit()
        return {"updated": len(ids), "idempotent": False}

    def record_execution(self, claim_ids: Iterable[str], *, event_key: str) -> dict:
        result = self._record_signal(claim_ids, "executed", event_key=event_key)
        self._propagate_long_term_signal(claim_ids, "executed", event_key)
        return result

    def confirm(self, claim_ids: Iterable[str], *, event_key: str) -> dict:
        result = self._record_signal(claim_ids, "confirmed", event_key=event_key)
        self._propagate_long_term_signal(claim_ids, "confirmed", event_key)
        return result

    def reject(self, claim_ids: Iterable[str], *, event_key: str) -> dict:
        return self._record_signal(claim_ids, "rejected", event_key=event_key)

    def retract(self, working_claim_id: str, *, round_no: int, reason: str,
                event_key: str) -> dict:
        """Explicitly retract one project claim while retaining its history."""
        sid = self._require_open()
        if not reason.strip():
            raise ValueError("retraction reason is required")
        with self._connect() as conn:
            prior = conn.execute(
                "SELECT 1 FROM project_working_signal_events WHERE event_key=?",
                (event_key,),
            ).fetchone()
            if prior:
                return {"changed": False, "idempotent": True}
            row = conn.execute(
                """SELECT status FROM project_working_claims
                   WHERE session_id=? AND working_claim_id=?""",
                (sid, working_claim_id),
            ).fetchone()
            if not row:
                raise KeyError(f"unknown working claim: {working_claim_id}")
            if row["status"] != "active":
                raise ValueError(
                    f"only active working claims can be retracted; got {row['status']}")
            conn.execute(
                """UPDATE project_working_claims SET status='retracted',
                   rejected_count=rejected_count+1,updated_at=?
                   WHERE working_claim_id=?""", (_now(), working_claim_id),
            )
            conn.execute(
                """INSERT INTO project_working_signal_events
                   (event_key,session_id,signal,working_claim_ids_json,created_at)
                   VALUES (?,?,?,?,?)""",
                (event_key, sid, "retracted", _json([working_claim_id]), _now()),
            )
            self._write_decision(
                conn, round_no=round_no, new_id=None,
                candidate_id=working_claim_id, relation="explicit_retraction",
                confidence=1.0, action="retract",
                reason=reason, proposal={"explicit": True},
            )
            conn.commit()
        return {"changed": True, "idempotent": False,
                "working_claim_id": working_claim_id, "status": "retracted"}

    def _propagate_long_term_signal(self, claim_ids: Iterable[str], signal: str,
                                    event_key: str) -> None:
        ids = list(dict.fromkeys(claim_ids))
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT canonical_claim_id FROM project_working_claims
                    WHERE working_claim_id IN ({placeholders})
                    AND canonical_claim_id IS NOT NULL""", ids,
            ).fetchall()
        canonical = [row[0] for row in rows]
        if not canonical:
            return
        if signal == "executed":
            self.plugin.record_execution(canonical, event_key=event_key + ":long-term")
        elif signal == "confirmed":
            self.plugin.confirm(canonical, event_key=event_key + ":long-term")

    def record_tool_events(self, task: dict, round_no: int,
                           events: Iterable[dict]) -> list[dict]:
        if self.tool_store is None:
            return []
        sid = self._require_open()
        return self.tool_store.ingest_events(
            events, session_id=sid, task_id=task["task_id"], round_no=round_no)

    def finish_project(self, *, satisfied: bool, final_round: int,
                       archive_path: str | Path | None = None) -> dict:
        """Archive the project and promote only confirmed stable claims."""
        sid = self._require_open()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM project_working_claims
                   WHERE session_id=? ORDER BY introduced_round,created_at""",
                (sid,),
            ).fetchall()
            evidence_rows = conn.execute(
                """SELECT * FROM project_working_evidence
                   WHERE session_id=? ORDER BY round_no,created_at""", (sid,),
            ).fetchall()
            decision_rows = conn.execute(
                """SELECT * FROM project_working_decisions
                   WHERE session_id=? ORDER BY round_no,created_at""", (sid,),
            ).fetchall()

        promoted = []
        skipped = []
        priority_updates = {row["new_working_claim_id"]:
                            json.loads(row["proposal_json"])["priority_change"]
                            for row in decision_rows if row["final_action"] == "update_priority"}
        for raw in rows:
            row = dict(raw)
            if row["origin"] != "round_feedback":
                continue
            reason = None
            if not satisfied:
                reason = "project_not_accepted"
            elif row["status"] != "active":
                reason = f"status:{row['status']}"
            elif row["persistence"] != "candidate_long_term":
                reason = "session_only"
            elif row["confirmed_count"] < 1:
                reason = "not_confirmed"
            elif row["executed_count"] < 1:
                reason = "not_verified_executed"
            elif row["working_claim_id"] in priority_updates:
                change = priority_updates[row["working_claim_id"]]
                if row["executed_count"] <= change["executed_before"]:
                    reason = "priority_not_reverified"
                elif row["confirmed_count"] <= change["confirmed_before"]:
                    reason = "priority_not_reconfirmed"
            if reason:
                skipped.append({"working_claim_id": row["working_claim_id"],
                                "reason": reason})
                continue
            evidence = next((dict(item) for item in reversed(evidence_rows)
                             if item["working_claim_id"] == row["working_claim_id"]), None)
            claim_payload = json.loads(row["claim_json"])
            result = self.plugin.ingest_feedback(
                evidence["evidence_text"] if evidence else claim_payload["action_or_value"],
                claim_payload,
                context=(f"promoted from project {self.task['task_id']}; "
                         f"verified and persona-confirmed at round {final_round}"),
                ingestion_key=f"v20-promote:{sid}:{row['working_claim_id']}",
                update_priority=row["working_claim_id"] in priority_updates,
                reactivate=True,
            )
            canonical_id = result["claim_id"]
            signal_writers = {
                "exposed": self.plugin.record_exposure,
                "executed": self.plugin.record_execution,
                "confirmed": self.plugin.confirm,
                "rejected": self.plugin.reject,
                "contradiction": self.plugin.record_contradiction,
            }
            # The working claim did not have a canonical ID when these events
            # originally occurred. Carry the audited history across promotion
            # through the canonical signal API, with deterministic idempotency.
            for signal, writer in signal_writers.items():
                if row["canonical_claim_id"] == canonical_id:
                    # Existing canonical signals were already propagated live.
                    continue
                for index in range(int(row[f"{signal}_count"])):
                    writer(
                        [canonical_id],
                        event_key=(f"v20-promote:{sid}:{row['working_claim_id']}:"
                                   f"{signal}:{index + 1}"),
                    )
            promoted.append({"working_claim_id": row["working_claim_id"],
                             "canonical_claim_id": canonical_id})

        canonical_updates = self._consolidate_relations(
            rows, decision_rows, promoted) if satisfied else []
        now = _now()
        status_counts_before = {}
        signal_totals = {
            "exposed": 0, "executed": 0, "confirmed": 0,
            "rejected": 0, "contradiction": 0,
        }
        for row in rows:
            status_counts_before[row["status"]] = status_counts_before.get(row["status"], 0) + 1
            for signal in signal_totals:
                signal_totals[signal] += int(row[f"{signal}_count"])
        decision_action_counts = {}
        for row in decision_rows:
            decision_action_counts[row["final_action"]] = (
                decision_action_counts.get(row["final_action"], 0) + 1)
        with self._connect() as conn:
            for item in promoted:
                conn.execute(
                    """UPDATE project_working_claims
                       SET promoted_claim_id=?,persistence='confirmed_long_term',
                           status='archived',updated_at=? WHERE working_claim_id=?""",
                    (item["canonical_claim_id"], now, item["working_claim_id"]),
                )
            conn.execute(
                """UPDATE project_working_claims SET status='archived',updated_at=?
                   WHERE session_id=? AND status='active'""", (now, sid),
            )
            counts = {row["status"]: row["count"] for row in conn.execute(
                """SELECT status,COUNT(*) AS count FROM project_working_claims
                   WHERE session_id=? GROUP BY status""", (sid,),
            ).fetchall()}
            summary = {
                "schema_version": "v20-project-archive-1",
                "session_id": sid,
                "persona_id": self.persona_id,
                "task_id": self.task["task_id"],
                "satisfied": bool(satisfied),
                "final_round": int(final_round),
                "long_term_retrieval_count": self._session_row()["long_term_retrieval_count"],
                "working_claim_count": len(rows),
                "decision_count": len(decision_rows),
                "evidence_count": len(evidence_rows),
                "promoted": promoted,
                "canonical_updates": canonical_updates,
                "skipped_promotion": skipped,
                "status_counts_before_archive": status_counts_before,
                "status_counts_after_archive": counts,
                "decision_action_counts": decision_action_counts,
                "signal_totals": signal_totals,
                "closed_at": now,
            }
            conn.execute(
                """UPDATE project_sessions SET status='archived',satisfied=?,
                   final_round=?,summary_json=?,closed_at=? WHERE session_id=?""",
                (int(bool(satisfied)), final_round, _json(summary), now, sid),
            )
            conn.commit()

        archive = {
            "summary": summary,
            "claims": [dict(row) for row in rows],
            "evidence": [dict(row) for row in evidence_rows],
            "decisions": [dict(row) for row in decision_rows],
        }
        if archive_path is not None:
            target = Path(archive_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(
                archive, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self.session_id = None
        self.task = None
        self._governance_claims = {}
        return summary

    def _consolidate_relations(self, rows, decisions, promoted) -> list[dict]:
        """Carry accepted working-state changes into the durable store.

        Only a verified, confirmed replacement can retire a durable default.
        Follow intermediate refinements so a multi-round change also closes
        the original long-term version. Session-only overrides do not do so.
        """
        by_id = {row["working_claim_id"]: dict(row) for row in rows}
        canonical = {
            row["working_claim_id"]: row["canonical_claim_id"] for row in rows
            if row["canonical_claim_id"]
        }
        canonical.update({item["working_claim_id"]: item["canonical_claim_id"]
                          for item in promoted})
        updates = []
        for original in rows:
            old_id = original["canonical_claim_id"]
            if not old_id:
                continue
            if original["status"] == "retracted":
                result = self.plugin.set_status(
                    old_id, "retracted", reason="accepted explicit project retraction",
                    event_key=f"v20-consolidate:{self.session_id}:{old_id}:retracted")
                updates.append({"old_claim_id": old_id, "action": "retract", **result})
                continue
            current = dict(original)
            visited = set()
            while current["status"] == "superseded" and current["superseded_by"]:
                next_id = current["superseded_by"]
                if next_id in visited or next_id not in by_id:
                    break
                visited.add(next_id)
                current = by_id[next_id]
            new_id = canonical.get(current["working_claim_id"])
            if not new_id or new_id == old_id or not any(
                    item["working_claim_id"] == current["working_claim_id"]
                    for item in promoted):
                continue
            with self.plugin._connect() as conn:
                result = resolve_conflict(
                    conn, old_id, new_id, "explicit_supersession", explicit=True,
                    proposed_action="supersede", preferred_winner=new_id)
            updates.append({"old_claim_id": old_id, "new_claim_id": new_id, **result})
        for decision in decisions:
            if decision["final_action"] != "conditionalize_and_coexist":
                continue
            new_id = canonical.get(decision["new_working_claim_id"])
            old_id = canonical.get(decision["candidate_working_claim_id"])
            if not old_id or not new_id or old_id == new_id:
                continue
            with self.plugin._connect() as conn:
                result = resolve_conflict(conn, old_id, new_id, "scope_exception",
                                          proposed_action="conditionalize")
            updates.append({"old_claim_id": old_id, "new_claim_id": new_id, **result})
        return updates


class ProjectWorkingMemoryAdapter:
    """Closed-loop adapter exposing the V20 project lifecycle."""

    def __init__(self, working: ProjectWorkingMemory, *, archive_root: str | Path | None = None):
        self.working = working
        self.archive_root = Path(archive_root) if archive_root else None
        self.signal_log: list[dict] = []

    def start_project(self, task: dict) -> dict:
        return self.working.start_project(task)

    def get_generation_memory(self, task: dict, round_no: int) -> dict:
        return self.working.get_generation_memory(task, round_no)

    def retrieve(self, task: dict) -> dict:
        # Compatibility only; the V20 runner calls get_generation_memory with
        # the actual round number.
        return self.get_generation_memory(task, 1)

    def write_feedback(self, task: dict, round_no: int, feedback: str, *,
                       session_id: str | None = None) -> dict:
        return self.working.ingest_raw_feedback(
            task, round_no, feedback, session_id=session_id)

    def record_execution(self, claim_ids: list[str], event_key: str) -> None:
        self.working.record_execution(claim_ids, event_key=event_key)
        self.signal_log.append({"event": "executed", "ids": list(claim_ids),
                                "key": event_key})

    def confirm(self, claim_ids: list[str], event_key: str) -> None:
        self.working.confirm(claim_ids, event_key=event_key)
        self.signal_log.append({"event": "confirmed", "ids": list(claim_ids),
                                "key": event_key})

    def record_tool_events(self, task: dict, round_no: int, events: list[dict]) -> list[dict]:
        return self.working.record_tool_events(task, round_no, events)

    def retract(self, working_claim_id: str, *, round_no: int,
                reason: str, event_key: str) -> dict:
        return self.working.retract(
            working_claim_id, round_no=round_no, reason=reason,
            event_key=event_key)

    def finish_project(self, task: dict, *, satisfied: bool, final_round: int) -> dict:
        archive_path = None
        if self.archive_root is not None:
            archive_path = self.archive_root / f"{task['task_id']}.project_memory.json"
        return self.working.finish_project(
            satisfied=satisfied, final_round=final_round,
            archive_path=archive_path)
