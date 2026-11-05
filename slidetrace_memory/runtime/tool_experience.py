"""Minimal, evidence-gated tool-experience memory for V20.

This channel is deliberately separate from persona preferences.  A tool
experience can be created only from a real error followed by a successful fix
and an explicit verification pass.  Normal successful calls contribute cheap
process statistics but do not inflate the memory store.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__:
    from .tokenization import count_tokens
else:
    from tokenization import count_tokens


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _signature(value: object) -> str:
    text = _norm(value)
    # Remove volatile paths, line numbers and UUID-like fragments while
    # retaining the deterministic error class/message.
    text = re.sub(r"[a-z]:\\[^\s]+", "<path>", text, flags=re.I)
    text = re.sub(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<uuid>", text)
    text = re.sub(r"\bline\s+\d+\b", "line <n>", text)
    text = re.sub(r"\b\d{3,}\b", "<n>", text)
    return text[:1000]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", _norm(text)))


def ensure_tool_experience_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tool_events (
            event_id TEXT PRIMARY KEY,
            event_key TEXT UNIQUE,
            session_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            round_no INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            tool_version TEXT NOT NULL DEFAULT '',
            operation TEXT NOT NULL,
            event_type TEXT NOT NULL,
            error_signature TEXT NOT NULL DEFAULT '',
            details_json TEXT NOT NULL,
            related_error_event_id TEXT,
            verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tool_events_task
            ON tool_events(task_id, round_no, tool_name, operation);

        CREATE TABLE IF NOT EXISTS tool_experiences (
            experience_id TEXT PRIMARY KEY,
            fingerprint TEXT UNIQUE NOT NULL,
            tool_name TEXT NOT NULL,
            tool_version TEXT NOT NULL DEFAULT '',
            operation TEXT NOT NULL,
            environment_json TEXT NOT NULL DEFAULT '{}',
            error_signature TEXT NOT NULL,
            successful_fix TEXT NOT NULL,
            applicable_when TEXT NOT NULL,
            verification_steps_json TEXT NOT NULL,
            source_event_ids_json TEXT NOT NULL,
            success_count INTEGER NOT NULL DEFAULT 1,
            failure_count INTEGER NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 0.6,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tool_experience_lookup
            ON tool_experiences(tool_name, operation, status);

        CREATE TABLE IF NOT EXISTS tool_experience_retrievals (
            retrieval_id TEXT PRIMARY KEY,
            tool_name TEXT NOT NULL,
            operation TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            candidate_ids_json TEXT NOT NULL,
            injected_ids_json TEXT NOT NULL,
            injected_tokens INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()


class ToolExperienceStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        with self._connect() as conn:
            ensure_tool_experience_schema(conn)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def ingest_events(self, events: Iterable[dict], *, session_id: str,
                      task_id: str, round_no: int) -> list[dict]:
        results = []
        known_keys: dict[str, str] = {}
        for index, raw in enumerate(events, 1):
            event = dict(raw)
            event_type = str(event.get("event_type", "success"))
            if event_type not in {"success", "error", "fix_verified", "fix_failed"}:
                raise ValueError(f"invalid tool event type: {event_type}")
            event_key = str(event.get("event_key") or
                            f"{session_id}:{round_no}:{index}")
            related_key = str(event.get("related_error_event_key", ""))
            related_id = known_keys.get(related_key)
            if related_key and related_id is None:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT event_id FROM tool_events WHERE event_key=?",
                        (related_key,),
                    ).fetchone()
                related_id = row[0] if row else None
            result = self.record_event(
                event_key=event_key,
                session_id=session_id,
                task_id=task_id,
                round_no=round_no,
                tool_name=str(event.get("tool_name", "ppt-master")),
                tool_version=str(event.get("tool_version", "")),
                operation=str(event.get("operation", "generate_and_render")),
                event_type=event_type,
                error=event.get("error") or event.get("error_signature", ""),
                details=event,
                related_error_event_id=related_id,
                verified=bool(event.get("verified", event_type == "fix_verified")),
            )
            known_keys[event_key] = result["event_id"]
            if event_type == "fix_verified":
                if not related_id:
                    raise ValueError("fix_verified requires a related error event")
                result["experience"] = self._create_experience(
                    error_event_id=related_id,
                    fix_event_id=result["event_id"],
                    successful_fix=str(event.get("successful_fix", "")).strip(),
                    applicable_when=str(event.get("applicable_when", "")).strip(),
                    verification_steps=list(event.get("verification_steps", [])),
                    environment=dict(event.get("environment", {})),
                )
            results.append(result)
        return results

    def record_event(self, *, event_key: str, session_id: str, task_id: str,
                     round_no: int, tool_name: str, operation: str,
                     event_type: str, error: object = "", details: dict | None = None,
                     tool_version: str = "", related_error_event_id: str | None = None,
                     verified: bool = False) -> dict:
        signature = _signature(error) if event_type in {"error", "fix_failed"} else ""
        event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"v20-tool-event:{event_key}"))
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT event_id FROM tool_events WHERE event_key=?", (event_key,),
            ).fetchone()
            if existing:
                return {"event_id": existing[0], "created": False, "idempotent": True}
            conn.execute(
                """INSERT INTO tool_events
                   (event_id,event_key,session_id,task_id,round_no,tool_name,
                    tool_version,operation,event_type,error_signature,details_json,
                    related_error_event_id,verified,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (event_id, event_key, session_id, task_id, int(round_no), tool_name,
                 tool_version, operation, event_type, signature,
                 json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                 related_error_event_id, int(bool(verified)), _now()),
            )
            conn.commit()
        return {"event_id": event_id, "created": True, "idempotent": False}

    def _create_experience(self, *, error_event_id: str, fix_event_id: str,
                           successful_fix: str, applicable_when: str,
                           verification_steps: list, environment: dict) -> dict:
        if not successful_fix:
            raise ValueError("verified fix requires successful_fix")
        if not verification_steps:
            raise ValueError("verified fix requires verification_steps")
        with self._connect() as conn:
            error = conn.execute(
                "SELECT * FROM tool_events WHERE event_id=?", (error_event_id,),
            ).fetchone()
            fix = conn.execute(
                "SELECT * FROM tool_events WHERE event_id=?", (fix_event_id,),
            ).fetchone()
            if not error or error["event_type"] != "error":
                raise ValueError("related event is not a recorded tool error")
            if not fix or fix["event_type"] != "fix_verified" or not fix["verified"]:
                raise ValueError("tool experience requires a verified fix event")
            identity = {
                "tool_name": error["tool_name"],
                "tool_version": error["tool_version"],
                "operation": error["operation"],
                "error_signature": error["error_signature"],
                "successful_fix": _norm(successful_fix),
            }
            fingerprint = hashlib.sha256(
                json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            existing = conn.execute(
                "SELECT experience_id FROM tool_experiences WHERE fingerprint=?",
                (fingerprint,),
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE tool_experiences SET success_count=success_count+1,
                       confidence=MIN(1.0,confidence+0.1),updated_at=?
                       WHERE experience_id=?""", (_now(), existing[0]),
                )
                conn.commit()
                return {"experience_id": existing[0], "created": False,
                        "reinforced": True}
            experience_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL, f"v20-tool-experience:{fingerprint}"))
            now = _now()
            conn.execute(
                """INSERT INTO tool_experiences
                   (experience_id,fingerprint,tool_name,tool_version,operation,
                    environment_json,error_signature,successful_fix,applicable_when,
                    verification_steps_json,source_event_ids_json,confidence,
                    status,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (experience_id, fingerprint, error["tool_name"], error["tool_version"],
                 error["operation"], json.dumps(environment, ensure_ascii=False,
                 sort_keys=True), error["error_signature"], successful_fix,
                 applicable_when or error["error_signature"],
                 json.dumps(verification_steps, ensure_ascii=False),
                 json.dumps([error_event_id, fix_event_id]), 0.7, "active", now, now),
            )
            conn.commit()
        return {"experience_id": experience_id, "created": True,
                "reinforced": False}

    def retrieve(self, *, tool_name: str, operation: str, context: str,
                 token_limit: int = 180) -> dict:
        if token_limit <= 0:
            raise ValueError("token_limit must be positive")
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM tool_experiences WHERE status='active'
                   AND (tool_name=? OR tool_name='*')
                   AND (operation=? OR operation='*')""",
                (tool_name, operation),
            ).fetchall()
        context_terms = _tokens(context)
        candidates = []
        for raw in rows:
            row = dict(raw)
            terms = _tokens(" ".join((row["error_signature"], row["applicable_when"],
                                      row["successful_fix"])))
            overlap = len(context_terms & terms) / max(1, len(context_terms | terms))
            row["score"] = 0.6 * float(row["confidence"]) + 0.4 * overlap
            candidates.append(row)
        candidates.sort(key=lambda item: (-item["score"], item["experience_id"]))
        selected = []
        lines = []
        for row in candidates:
            steps = json.loads(row["verification_steps_json"])
            line = (
                f"- 遇到 {row['applicable_when']}：{row['successful_fix']}；"
                f"修复后验证：{'；'.join(map(str, steps))}"
            )
            proposal = "【工具执行经验】\n" + "\n".join([*lines, line])
            if count_tokens(proposal) > token_limit:
                continue
            selected.append(row)
            lines.append(line)
        memory_out = "【工具执行经验】\n" + ("\n".join(lines) or "- 无")
        injected_tokens = count_tokens(memory_out)
        retrieval_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tool_experience_retrievals
                   (retrieval_id,tool_name,operation,context_hash,candidate_ids_json,
                    injected_ids_json,injected_tokens,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (retrieval_id, tool_name, operation,
                 hashlib.sha256(context.encode("utf-8")).hexdigest(),
                 json.dumps([row["experience_id"] for row in candidates]),
                 json.dumps([row["experience_id"] for row in selected]),
                 injected_tokens, _now()),
            )
            conn.commit()
        return {
            "memory_out": memory_out if selected else "",
            "experience_ids": [row["experience_id"] for row in selected],
            "tokens": injected_tokens if selected else 0,
            "candidate_ids": [row["experience_id"] for row in candidates],
            "retrieval_id": retrieval_id,
        }

    def metrics(self, *, task_id: str | None = None) -> dict:
        where = " WHERE task_id=?" if task_id else ""
        args = (task_id,) if task_id else ()
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT event_type,COUNT(*) AS count FROM tool_events{where} "
                "GROUP BY event_type", args,
            ).fetchall()
            verified = conn.execute(
                f"SELECT COUNT(*) FROM tool_events{where}" +
                (" AND verified=1" if where else " WHERE verified=1"), args,
            ).fetchone()[0]
        counts = {row["event_type"]: row["count"] for row in rows}
        return {
            "tool_errors": counts.get("error", 0),
            "verified_repairs": counts.get("fix_verified", 0),
            "failed_repairs": counts.get("fix_failed", 0),
            "successful_calls": counts.get("success", 0),
            "verification_events": verified,
        }
