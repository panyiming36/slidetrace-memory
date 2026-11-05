"""Small SQLite and text dependencies for the installable memory plugin."""
from __future__ import annotations

import json
import pickle
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class MemoryRecord:
    id: str
    memory_type: str
    content: str
    context: str
    resolution: str
    tier: str
    hit_count: int
    status: str
    created_at: datetime
    last_hit_at: datetime
    embedding: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class Database:
    """Persist original feedback alongside the shared canonical claim tables."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_records (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                resolution TEXT,
                tier TEXT DEFAULT 'warm',
                hit_count INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active',
                generated_skill_slug TEXT,
                tags TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_hit_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_memory_record(self, record: MemoryRecord) -> str:
        with self._get_connection() as conn:
            conn.execute("""INSERT INTO memory_records (
                id,memory_type,content,context,resolution,tier,hit_count,status,
                tags,embedding,created_at,last_hit_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                memory_type=excluded.memory_type,content=excluded.content,
                context=excluded.context,resolution=excluded.resolution,
                tier=excluded.tier,hit_count=excluded.hit_count,
                status=excluded.status,tags=excluded.tags,
                embedding=excluded.embedding,last_hit_at=excluded.last_hit_at,
                updated_at=CURRENT_TIMESTAMP""", (
                    record.id, record.memory_type, record.content,
                    record.context, record.resolution, record.tier,
                    record.hit_count, record.status,
                    json.dumps(record.tags, ensure_ascii=False),
                    pickle.dumps(record.embedding) if record.embedding else None,
                    record.created_at.isoformat(), record.last_hit_at.isoformat(),
                ))
        return record.id


class DataSanitizer:
    """Remove obvious credentials without rewriting presentation preferences.

    A host may pass its own ``sanitize(text)`` implementation for additional
    policies. This default requires no language model or downloaded NLP data.
    """

    _PATTERNS = (
        (r"\bsk-[A-Za-z0-9_-]{20,}\b", "<LLM_API_KEY>"),
        (r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b", "<GITHUB_TOKEN>"),
        (r"\bAKIA[A-Z0-9]{16}\b", "<AWS_ACCESS_KEY>"),
        (r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer <REDACTED_TOKEN>"),
        (
            r"(?i)(\b(?:api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*)"
            r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
            r"\1<REDACTED_CREDENTIAL>",
        ),
    )

    def sanitize(self, text: str) -> str:
        value = str(text or "")
        for pattern, replacement in self._PATTERNS:
            value = re.sub(pattern, replacement, value)
        return value
