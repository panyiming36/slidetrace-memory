"""Host-facing API over the existing SlideTrace working and canonical stores."""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .core.preference_governance import (
    PreferenceClaim, decode_claim_row, normalize_claim, scope_applies,
)
from .runtime.local_memory_plugin import LocalMemoryPlugin
from .runtime.project_working_memory import ProjectWorkingMemory
from .runtime.semantic_memory import SemanticFeedbackAnalyzer


def _public_claim(claim: dict) -> dict:
    return {key: value for key, value in claim.items()
            if key != "embedding" and not key.startswith("_")}


class PptMemory:
    """One user's memory plugin. The host supplies generation and acceptance.

    Feedback adapts the open project's working memory. Reusable changes enter
    canonical memory only after execution, confirmation, and project acceptance.
    Use one instance per open project; the SQLite file persists across instances.
    """

    def __init__(self, db_path: str | Path, *, user_id: str,
                 embedder: Callable[[str], list[float]],
                 complete_json: Callable[[str], dict] | None = None,
                 embedding_model: str | None = None,
                 token_limit: int | None = 1200, sanitizer=None):
        self.user_id = " ".join(str(user_id).strip().lower().split())
        if not self.user_id:
            raise ValueError("user_id is required")
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model = (embedding_model or getattr(embedder, "model", None)
                                or "host-embedding")
        self._embedder = embedder
        self._last_round = 0
        self._ensure_embedding_profile()
        with self._connect() as conn:
            dims = conn.execute("SELECT dims FROM ppt_memory_embedding_profile WHERE id=1").fetchone()[0]
        analyzer = (SemanticFeedbackAnalyzer(
            complete_json=complete_json, model=getattr(complete_json, "model", "host"),
        ) if complete_json is not None else None)
        self._plugin = LocalMemoryPlugin(
            self.db_path, embedder=self._embed, semantic_analyzer=analyzer,
            sanitizer=sanitizer, expected_embedding_dims=dims,
            embedding_text_mode="claim",
        )
        self._working = ProjectWorkingMemory(
            self._plugin, self.user_id, token_limit=token_limit,
        )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_embedding_profile(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS ppt_memory_embedding_profile
                (id INTEGER PRIMARY KEY CHECK(id=1), model TEXT NOT NULL, dims INTEGER,
                 text_mode TEXT NOT NULL DEFAULT 'claim')""")
            row = conn.execute("SELECT model,text_mode FROM ppt_memory_embedding_profile WHERE id=1").fetchone()
            if row is not None and row["model"] != self.embedding_model:
                raise ValueError("this store uses another embedding model; keep its model or use a new store")
            if row is not None and row["text_mode"] != "claim":
                raise ValueError("this store uses a different embedding text mode; use its original configuration")
            if row is None:
                populated = conn.execute("""SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='preference_claims'""").fetchone()
                if populated and conn.execute("SELECT 1 FROM preference_claims LIMIT 1").fetchone():
                    raise ValueError("use a new plugin store; this existing store has no embedding-model profile")
                conn.execute("INSERT INTO ppt_memory_embedding_profile(id,model) VALUES(1,?)",
                             (self.embedding_model,))

    def _embed(self, text: str) -> list[float]:
        vector = [float(value) for value in self._embedder(text)]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding must be a non-empty vector of finite numbers")
        if not any(vector):
            raise ValueError("embedding must not be an all-zero vector")
        with self._connect() as conn:
            row = conn.execute("SELECT dims FROM ppt_memory_embedding_profile WHERE id=1").fetchone()
            if row["dims"] is not None and row["dims"] != len(vector):
                raise ValueError("embedding dimensions changed for this memory store")
            if row["dims"] is None:
                conn.execute("UPDATE ppt_memory_embedding_profile SET dims=? WHERE id=1", (len(vector),))
        if hasattr(self, "_plugin"):
            self._plugin.expected_embedding_dims = len(vector)
        return vector

    @property
    def session_id(self) -> str | None:
        return self._working.session_id

    def start_project(self, project_id: str, topic: str, *,
                      domain: str | None = None, audience: str | None = None,
                      page_ids: Iterable[str] | None = None,
                      scope: dict | None = None) -> dict:
        if not str(project_id).strip() or not str(topic).strip():
            raise ValueError("project_id and topic are required")
        requested = dict(scope or ({"kind": "domain", "domain": domain}
                                   if domain else {"kind": "project"}))
        if requested.get("kind") == "global":
            requested["kind"] = "project"
        requested.update({"project_id": str(project_id), "task_id": str(project_id)})
        if audience:
            requested["audience"] = audience
        task = {"task_id": str(project_id), "project_id": str(project_id),
                "topic": topic, "scope": requested}
        if domain:
            task["domain"] = domain
        if audience:
            task["audience"] = audience
        if page_ids is not None:
            task["page_ids"] = list(dict.fromkeys(str(value) for value in page_ids))
        task["memory_query"] = " ".join(filter(None, (topic, domain, audience)))
        result = self._working.start_project(task)
        self._last_round = 0
        return result

    def resume_project(self, session_id: str) -> dict:
        result = self._working.resume_project(session_id)
        with self._connect() as conn:
            row = conn.execute("""SELECT MAX(value) FROM (
                SELECT MAX(round_no) AS value FROM project_working_evidence WHERE session_id=?
                UNION ALL SELECT MAX(last_exposed_round) FROM project_working_claims WHERE session_id=?)""",
                (session_id, session_id)).fetchone()
        self._last_round = int(row[0] or 0)
        return result

    def list_projects(self, *, open_only: bool = True) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""SELECT session_id,task_id,status,satisfied,final_round,
                created_at,closed_at FROM project_sessions WHERE persona_id=?"""
                + (" AND status='open'" if open_only else "") + " ORDER BY created_at",
                (self.user_id,)).fetchall()
        return [dict(row) for row in rows]

    def _task(self, *, page_id: str | None = None, page_type: str | None = None) -> dict:
        self._working._require_open()
        task = dict(self._working.task)
        task["scope"] = dict(task["scope"])
        if page_id is not None:
            page_id = str(page_id)
            normalize = lambda value: " ".join(str(value).strip().lower().split())
            if task.get("page_ids") and normalize(page_id) not in {
                    normalize(value) for value in task["page_ids"]}:
                raise ValueError("page_id is outside the current project")
            task["page_id"] = page_id
            task["scope"]["page_id"] = page_id
        if page_type is not None:
            task["page_type"] = str(page_type)
            task["scope"]["page_type"] = str(page_type)
        task["memory_query"] = " ".join(str(value) for value in (
            task["topic"], task.get("domain"), task.get("audience"), task.get("page_type"),
        ) if value)
        return task

    @staticmethod
    def _round(round_no: int) -> int:
        if isinstance(round_no, bool) or not isinstance(round_no, int) or round_no < 1:
            raise ValueError("round_no must be a positive integer")
        return round_no

    def context(self, *, round_no: int, page_id: str | None = None,
                page_type: str | None = None) -> dict:
        """Return memory_out and claim IDs; the host appends it to its prompt."""
        round_no = self._round(round_no)
        task = self._task(page_id=page_id, page_type=page_type)
        self._working.refresh_context(task)
        result = self._working.get_generation_memory(task, round_no)
        self._last_round = max(self._last_round, round_no)
        result["claims"] = [_public_claim(claim) for claim in result["claims"]]
        return result

    def feedback(self, text: str, *, round_no: int, page_id: str | None = None,
                 page_type: str | None = None, feedback_id: str | None = None) -> dict:
        """Extract all feedback items into the project's governed working set."""
        round_no = self._round(round_no)
        task = self._task(page_id=page_id, page_type=page_type)
        self._working.refresh_context(task)
        self._last_round = max(self._last_round, round_no)
        return self._working.ingest_raw_feedback(
            task, round_no, text,
            ingestion_key=(f"{self.session_id}:feedback:{feedback_id}" if feedback_id else None),
        )

    def feedback_claim(self, claim: dict, *, round_no: int,
                       feedback: str | None = None, relation: dict | None = None,
                       temporary: bool = False, feedback_id: str | None = None,
                       priority_update: bool | None = None,
                       page_id: str | None = None, page_type: str | None = None) -> dict:
        """Use structured input when the host already extracts preference claims."""
        round_no = self._round(round_no)
        task = self._task(page_id=page_id, page_type=page_type)
        self._working.refresh_context(task)
        payload = dict(claim)
        supplied_user = payload.pop("persona_id", self.user_id)
        if str(supplied_user).strip().lower() != self.user_id:
            raise ValueError("claim belongs to another user")
        payload.setdefault("scope", {"kind": "global"})
        payload.setdefault("source_type", "explicit_feedback")
        # Hosts often reuse analyzer-schema claims; the analyzer-only fields
        # are governed by feedback_claim's own parameters instead.
        for auxiliary in ("classification", "partition", "partition_reason"):
            payload.pop(auxiliary, None)
        inline_relation = payload.pop("relation", None)
        inline_update = payload.pop("priority_update", None)
        if relation is None and isinstance(inline_relation, dict):
            relation = inline_relation
        if priority_update is None:
            priority_update = inline_update if inline_update is not None else False
        if not isinstance(priority_update, bool):
            raise ValueError("priority_update must be a boolean")
        SemanticFeedbackAnalyzer._validate_claim_scope(payload["scope"], current_scope={
            **task["scope"], "page_ids": task.get("page_ids", []),
        })
        payload = normalize_claim(PreferenceClaim(persona_id=self.user_id, **payload))
        persistence = ("session_only" if temporary or payload["scope"].get("kind") in
                       {"project", "task", "page"} else "candidate_long_term")
        self._last_round = max(self._last_round, round_no)
        return self._working.ingest_claim(
            task, round_no, feedback or payload["action_or_value"], payload,
            relation=relation, persistence=persistence,
            update_priority=priority_update,
            classification="one_off_incident" if temporary else "stable_preference",
            ingestion_key=(f"{self.session_id}:feedback:{feedback_id}" if feedback_id else None),
        )

    def record_outcome(self, *, round_no: int, implemented_ids: Iterable[str] = (),
                       accepted_ids: Iterable[str] = (), rejected_ids: Iterable[str] = (),
                       outcome_id: str | None = None) -> dict:
        """Record host evidence. Merely retrieving a claim never confirms it."""
        round_no = self._round(round_no)
        sid = self._working._require_open()
        implemented = list(dict.fromkeys(implemented_ids))
        accepted = list(dict.fromkeys(accepted_ids))
        rejected = list(dict.fromkeys(rejected_ids))
        if not set(accepted).issubset(implemented):
            raise ValueError("accepted_ids must also be implemented in this round")
        if set(accepted) & set(rejected):
            raise ValueError("a claim cannot be both accepted and rejected")
        with self._connect() as conn:
            known = {row["working_claim_id"] for row in conn.execute("""SELECT working_claim_id
                FROM project_working_claims WHERE session_id=? AND status='active'
                AND last_exposed_round=?""", (sid, round_no))}
        if not (set(implemented) | set(accepted) | set(rejected)).issubset(known):
            raise ValueError("outcome IDs must come from this project's exposed claims for this round")
        key = f"{sid}:round:{round_no}:outcome:{outcome_id or uuid.uuid4().hex}"
        result = {}
        for name, ids, writer in (("executed", implemented, self._working.record_execution),
                                  ("confirmed", accepted, self._working.confirm),
                                  ("rejected", rejected, self._working.reject)):
            result[name] = writer(ids, event_key=f"{key}:{name}")
        self._last_round = max(self._last_round, round_no)
        return result

    def finish_project(self, *, satisfied: bool, final_round: int | None = None) -> dict:
        if not isinstance(satisfied, bool):
            raise ValueError("satisfied must be a boolean supplied by the host")
        return self._working.finish_project(
            satisfied=satisfied, final_round=self._round(
                self._last_round if final_round is None else final_round),
        )

    def list_memories(self, *, include_inactive: bool = False,
                      include_working: bool = False) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM preference_claims WHERE persona_id=?"
                + ("" if include_inactive else " AND status='active'") + " ORDER BY updated_at,claim_id",
                (self.user_id,)).fetchall()
            canonical = [_public_claim(decode_claim_row(row)) for row in rows]
            working = []
            if include_working and self.session_id:
                for row in conn.execute("SELECT * FROM project_working_claims WHERE session_id=?",
                                        (self.session_id,)):
                    item = dict(row)
                    item["claim"] = json.loads(item.pop("claim_json"))
                    working.append(item)
        return {"canonical": canonical, "working": working}

    def forget(self, claim_id: str, *, reason: str = "user requested removal") -> dict:
        """Stop using a preference in canonical memory and open working copies.

        Historical evidence is retained. ``claim_id`` may be a canonical or
        current working ID from this user's list/context response.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT claim_id FROM preference_claims WHERE claim_id=? AND persona_id=?",
                               (claim_id, self.user_id)).fetchone()
            working = None
            if row is None:
                working = conn.execute("""SELECT w.* FROM project_working_claims w
                    JOIN project_sessions s ON s.session_id=w.session_id
                    WHERE w.working_claim_id=? AND s.persona_id=?""",
                    (claim_id, self.user_id)).fetchone()
                if working is None:
                    raise KeyError("unknown preference for this user")
            canonical_id = (row["claim_id"] if row else
                            working["canonical_claim_id"] or working["promoted_claim_id"])
        result = (self._plugin.set_status(canonical_id, "retracted", reason=reason)
                  if canonical_id else {"changed": False})
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            open_ids = [row["session_id"] for row in conn.execute(
                "SELECT session_id FROM project_sessions WHERE persona_id=? AND status='open'",
                (self.user_id,))]
            affected = {claim_id} | ({canonical_id} if canonical_id else set())
            targets, frontier = set(), set(affected)
            while frontier and open_ids:
                marks = ",".join("?" * len(frontier))
                sess_marks = ",".join("?" * len(open_ids))
                rows = conn.execute(f"""
                    SELECT w.working_claim_id FROM project_working_claims w
                    WHERE w.session_id IN ({sess_marks}) AND w.status!='retracted'
                      AND (w.working_claim_id IN ({marks})
                           OR w.canonical_claim_id IN ({marks})
                           OR w.working_claim_id IN (
                               SELECT d.new_working_claim_id
                               FROM project_working_decisions d
                               WHERE d.candidate_working_claim_id IN ({marks})))
                """, (*open_ids, *frontier, *frontier, *frontier)).fetchall()
                found = {row[0] for row in rows} - targets
                targets |= found
                frontier = found
            changed_working = 0
            if targets:
                marks = ",".join("?" * len(targets))
                changed_working = conn.execute(
                    f"""UPDATE project_working_claims SET status='retracted',updated_at=?
                        WHERE working_claim_id IN ({marks}) AND status!='retracted'""",
                    (now, *targets)).rowcount
        self._working._governance_claims = {
            key: item for key, item in self._working._governance_claims.items()
            if key not in targets and item["canonical_claim_id"] not in affected
        }
        return {**result, "changed": bool(result.get("changed") or changed_working),
                "claim_id": claim_id, "canonical_claim_id": canonical_id,
                "status": "retracted"}

    def revise(self, claim_id: str, *, round_no: int, action_or_value: str,
               priority: str | None = None) -> dict:
        """Propose an explicit revision; commit it through the usual evidence path."""
        self._working._require_open()
        with self._connect() as conn:
            canonical = conn.execute("SELECT * FROM preference_claims WHERE claim_id=? AND persona_id=?",
                                     (claim_id, self.user_id)).fetchone()
        if canonical is not None:
            previous = decode_claim_row(canonical)
            if previous["status"] != "active":
                raise ValueError("revise requires an active preference; submit feedback to restore an old one")
            task = self._task(page_id=previous["scope"].get("page_id"),
                              page_type=previous["scope"].get("page_type"))
            if not scope_applies(previous["scope"], task["scope"]):
                raise ValueError("this preference does not apply to the current project")
            self._working.refresh_context(task)
        with self._connect() as conn:
            row = conn.execute("""SELECT claim_json,working_claim_id,status FROM project_working_claims
                WHERE session_id=? AND (working_claim_id=? OR canonical_claim_id=?)
                ORDER BY status='active' DESC LIMIT 1""", (self.session_id, claim_id, claim_id)).fetchone()
            if row:
                if row["status"] != "active":
                    raise ValueError("revise requires an active working preference")
                payload, candidate_id = json.loads(row["claim_json"]), row["working_claim_id"]
            else:
                found = next(((key, item) for key, item in self._working._governance_claims.items()
                              if item["canonical_claim_id"] == claim_id), None)
                if found is None:
                    raise KeyError("preference is not in this project's applicable memory")
                candidate_id, item = found
                payload = dict(item["claim"])
        payload["action_or_value"] = action_or_value
        payload["source_type"] = "edit"
        if priority is not None:
            payload["priority"] = priority
        return self.feedback_claim(
            payload, round_no=round_no, feedback=f"Change this preference to: {action_or_value}",
            relation={"candidate_id": candidate_id, "type": "temporal_update",
                      "confidence": 1.0, "explicit": True},
            priority_update=priority is not None,
            page_id=payload["scope"].get("page_id"),
            page_type=payload["scope"].get("page_type"),
        )
