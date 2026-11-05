"""In-process V20 long-term memory plugin.

V20 keeps the V19 governed store intact and adds a project working-memory
layer above it.  This module remains the durable canonical store; project
feedback is promoted here only after verified execution and persona
confirmation.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sqlite3
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

if __package__ == "slidetrace_memory.runtime":
    from ..storage import Database, MemoryRecord
    from ..core.preference_governance import (
        PreferenceClaim, active_claims_for_context, claim_is_current,
        decode_claim_row, ensure_preference_schema, get_claim_signals,
        ingest_claim, normalize_claim, record_claim_signals, resolve_conflict,
        scopes_compatible, scope_specificity, set_claim_status, validate_relation,
    )
else:
    from engine.core.database import Database
    from engine.core.models import MemoryRecord
    from engine.core.preference_governance import (
        PreferenceClaim, active_claims_for_context, claim_is_current,
        decode_claim_row, ensure_preference_schema, get_claim_signals,
        ingest_claim, normalize_claim, record_claim_signals, resolve_conflict,
        scopes_compatible, scope_specificity, set_claim_status, validate_relation,
    )

if __package__:
    from .memory_out import build_memory_out_variants
    from .semantic_memory import RELATION_MIN_CONFIDENCE, ensure_semantic_schema
    from .tokenization import get_tokenizer
else:
    from memory_out import build_memory_out_variants
    from semantic_memory import RELATION_MIN_CONFIDENCE, ensure_semantic_schema
    from tokenization import get_tokenizer


class LocalMemoryPlugin:
    """A local deployment with online-time semantics.

    Structured claims can use ``ingest_feedback`` directly. Raw prose uses an
    injected semantic analyzer whose proposals remain behind deterministic
    candidate, scope, confidence and lifecycle validators. No raw memory row is
    ever injected; retrieval returns only governed active claims.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        embedder: Optional[Callable[[str], list[float]]] = None,
        sanitizer=None,
        trace_path: str | Path | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
        expected_embedding_dims: int | None = None,
        semantic_analyzer=None,
        scope_router=None,
        partition_resolver=None,
        embedding_text_mode: str = "feedback",
    ):
        self.db_path = Path(db_path)
        self.db = Database(str(self.db_path))
        if sanitizer is None:
            if __package__ == "slidetrace_memory.runtime":
                from ..storage import DataSanitizer
            else:
                from engine.utils.sanitizer import DataSanitizer
            sanitizer = DataSanitizer()
        self.sanitizer = sanitizer
        self._embedder = embedder
        self.embedding_base_url = (
            embedding_base_url or os.environ.get("ULTRON_EMBEDDING_BASE_URL", "")
        ).rstrip("/")
        self.embedding_api_key = embedding_api_key or os.environ.get(
            "ULTRON_EMBEDDING_API_KEY", ""
        )
        self.embedding_model = embedding_model or os.environ.get(
            "ULTRON_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"
        )
        if expected_embedding_dims is not None and expected_embedding_dims <= 0:
            raise ValueError("expected_embedding_dims must be positive")
        self.expected_embedding_dims = expected_embedding_dims
        # Only HTTP embeddings are cached. Injected test embedders may be
        # intentionally stateful and must retain their original semantics.
        self._http_embedding_cache: dict[str, tuple[float, ...]] = {}
        self.semantic_analyzer = semantic_analyzer
        self.scope_router = scope_router
        self.partition_resolver = partition_resolver
        if embedding_text_mode not in {"feedback", "claim"}:
            raise ValueError("embedding_text_mode must be feedback or claim")
        self.embedding_text_mode = embedding_text_mode
        self.trace_path = Path(trace_path) if trace_path else None
        with self._connect() as conn:
            ensure_preference_schema(conn)
            ensure_semantic_schema(conn)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _sanitize(self, text: str) -> str:
        return str(self.sanitizer.sanitize(text or "") or "")

    def _embed(self, text: str) -> list[float]:
        if self._embedder is not None:
            vector = self._embedder(text)
        else:
            cache_key = f"{self.embedding_model}\0{text}"
            cached = self._http_embedding_cache.get(cache_key)
            if cached is not None:
                return list(cached)
            if not self.embedding_base_url:
                raise RuntimeError("embedding base URL is not configured")
            body = json.dumps({"model": self.embedding_model, "input": text}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.embedding_api_key:
                headers["Authorization"] = f"Bearer {self.embedding_api_key}"
            request = urllib.request.Request(
                f"{self.embedding_base_url}/embeddings", data=body, headers=headers,
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
            vector = payload["data"][0]["embedding"]
        result = [float(value) for value in vector]
        if not result:
            raise RuntimeError("embedding provider returned an empty vector")
        if (self.expected_embedding_dims is not None
                and len(result) != self.expected_embedding_dims):
            raise RuntimeError(
                "embedding dim mismatch: "
                f"got {len(result)}, expected {self.expected_embedding_dims}"
            )
        if self._embedder is None:
            self._http_embedding_cache[cache_key] = tuple(result)
        return result

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
        return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0

    def ingest_feedback(
        self,
        feedback: str,
        claim: PreferenceClaim | dict,
        *,
        context: str = "",
        resolution: str = "",
        ingestion_key: str | None = None,
        embedding: list[float] | None = None,
        update_priority: bool = False,
        reactivate: bool = False,
    ) -> dict:
        """Sanitize, embed and commit evidence+canonical claim synchronously."""
        if not feedback or not feedback.strip():
            raise ValueError("feedback is required")
        with self._connect() as conn:
            ensure_preference_schema(conn)
            if ingestion_key:
                prior = conn.execute("""SELECT e.claim_id,e.evidence_id,e.source_memory_id
                    FROM preference_evidence e WHERE e.ingestion_key=?""", (ingestion_key,)).fetchone()
                if prior:
                    return {
                        "created": False, "idempotent": True, "claim_id": prior["claim_id"],
                        "evidence_id": prior["evidence_id"], "memory_id": prior["source_memory_id"],
                    }

        value = claim if isinstance(claim, PreferenceClaim) else PreferenceClaim(**claim)
        clean_feedback = self._sanitize(feedback)
        clean_context = self._sanitize(context)
        clean_resolution = self._sanitize(resolution)
        clean_claim = replace(
            value,
            subject=self._sanitize(value.subject),
            attribute=self._sanitize(value.attribute),
            action_or_value=self._sanitize(value.action_or_value),
            conditions=[self._sanitize(item) for item in value.conditions],
            exceptions=[self._sanitize(item) for item in value.exceptions],
        )
        if self.embedding_text_mode == "claim":
            semantic_scope = {key: value for key, value in clean_claim.scope.items()
                              if key not in {"project_id", "task_id", "page_id"}}
            embedding_text = "\n".join((
                f"{clean_claim.direction} {clean_claim.subject} {clean_claim.attribute}: "
                f"{clean_claim.action_or_value}",
                json.dumps(semantic_scope, ensure_ascii=False, sort_keys=True),
                " ".join(clean_claim.conditions), " ".join(clean_claim.exceptions),
            ))
        else:
            embedding_text = "\n".join(filter(None, (
                clean_feedback, clean_context, clean_resolution, clean_claim.action_or_value,
            )))
        vector = embedding if embedding is not None else self._embed(embedding_text)
        vector = [float(item) for item in vector]
        if not vector:
            raise ValueError("embedding must not be empty")
        if (self.expected_embedding_dims is not None
                and len(vector) != self.expected_embedding_dims):
            raise ValueError(
                "embedding dim mismatch: "
                f"got {len(vector)}, expected {self.expected_embedding_dims}"
            )
        memory_id = (
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"v19-local:{ingestion_key}"))
            if ingestion_key else str(uuid.uuid4())
        )
        now = datetime.now()
        tags = list(dict.fromkeys([
            *clean_claim.tags,
            f"persona:{clean_claim.persona_id}",
            f"scope:{clean_claim.scope.get('kind', 'global')}",
            f"direction:{clean_claim.direction}",
            "channel:preference",
        ]))
        self.db.save_memory_record(MemoryRecord(
            id=memory_id, memory_type="preference", content=clean_feedback,
            context=clean_context, resolution=clean_resolution, tier="warm",
            hit_count=0, status="active", created_at=now, last_hit_at=now,
            embedding=vector, tags=tags,
        ))
        clean_claim = replace(clean_claim, source_memory_id=memory_id)
        with self._connect() as conn:
            result = ingest_claim(
                conn, clean_claim, evidence_text=clean_feedback, ingestion_key=ingestion_key,
                update_priority=update_priority,
                reactivate=reactivate,
            )
            conn.execute(
                "UPDATE preference_claims SET embedding=? WHERE claim_id=?",
                (pickle.dumps(vector), result["claim_id"]),
            )
            conn.commit()
        return {**result, "memory_id": memory_id, "idempotent": False}

    def _semantic_candidates(self, persona_id: str, vector: list[float], scope: dict,
                             limit: int = 8) -> list[dict]:
        """Embedding recall only; persona and compatible-scope bounds are deterministic."""
        with self._connect() as conn:
            rows = conn.execute("""SELECT claim_id,subject,attribute,action_or_value,
                direction,scope_json,priority,embedding,conditions_json,exceptions_json,valid_from,valid_to
                FROM preference_claims
                WHERE persona_id=? AND status='active' AND embedding IS NOT NULL""",
                (persona_id,),
            ).fetchall()
        candidates = []
        for row in rows:
            if not claim_is_current(dict(row)):
                continue
            item_scope = json.loads(row[5])
            if not scopes_compatible(scope, item_scope):
                continue
            try:
                candidate_vector = pickle.loads(row[7])
            except Exception as exc:
                if self.expected_embedding_dims is not None:
                    raise RuntimeError(
                        f"claim {row[0]} has an unreadable embedding") from exc
                continue
            if (self.expected_embedding_dims is not None
                    and len(candidate_vector) != self.expected_embedding_dims):
                raise RuntimeError(
                    f"claim {row[0]} embedding dim mismatch: "
                    f"got {len(candidate_vector)}, expected "
                    f"{self.expected_embedding_dims}")
            candidates.append({
                "claim_id": row[0], "subject": row[1], "attribute": row[2],
                "action_or_value": row[3], "direction": row[4], "scope": item_scope,
                "priority": row[6], "similarity": self._cosine(vector, candidate_vector),
                "conditions": json.loads(row[8]), "exceptions": json.loads(row[9]),
            })
        candidates.sort(key=lambda item: (-item["similarity"], item["claim_id"]))
        return candidates[:limit]

    def _write_semantic_audit(self, *, ingestion_key: str | None, feedback: str,
                              classification: str, candidates: list[dict], proposal: dict,
                              result: dict) -> None:
        import hashlib
        decision_id = str(uuid.uuid4())
        feedback_hash = hashlib.sha256(feedback.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            ensure_semantic_schema(conn)
            conn.execute("""INSERT OR REPLACE INTO semantic_ingestion_decisions
                (decision_id,ingestion_key,feedback_hash,classification,candidates_json,
                 proposal_json,result_json,model,prompt_version,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                decision_id, ingestion_key, feedback_hash, classification,
                json.dumps(candidates, ensure_ascii=False, sort_keys=True),
                json.dumps(proposal, ensure_ascii=False, sort_keys=True),
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                str(proposal.get("_model", "")),
                str(proposal.get("_prompt_version", "v19-semantic-ingest-1")),
                datetime.utcnow().isoformat() + "Z",
            ))
            conn.commit()

    def ingest_raw_feedback(self, feedback: str, persona_id: str, *, scope: dict,
                            context: str = "", ingestion_key: str | None = None) -> dict:
        """Extract, partition and govern one raw feedback item in a single semantic call."""
        if self.semantic_analyzer is None:
            raise RuntimeError("semantic_analyzer is required for raw feedback")
        if not feedback or not feedback.strip():
            raise ValueError("feedback is required")
        if ingestion_key:
            with self._connect() as conn:
                ensure_semantic_schema(conn)
                prior = conn.execute(
                    "SELECT result_json FROM semantic_ingestion_decisions WHERE ingestion_key=?",
                    (ingestion_key,),
                ).fetchone()
                if prior:
                    return {**json.loads(prior[0]), "idempotent": True}

        clean_feedback = self._sanitize(feedback)
        clean_context = self._sanitize(context)
        vector = self._embed("\n".join(filter(None, (clean_feedback, clean_context))))
        candidates = self._semantic_candidates(persona_id, vector, scope)
        proposal = self.semantic_analyzer.analyze(
            feedback=clean_feedback, persona_id=persona_id, scope=scope,
            context=clean_context, candidates=candidates,
        )
        classification = proposal["classification"]
        if not proposal.get("claims"):
            result = {
                "stored": False, "classification": classification,
                "reason": str(proposal.get("reason", "")), "idempotent": False,
            }
            self._write_semantic_audit(
                ingestion_key=ingestion_key, feedback=clean_feedback,
                classification=classification, candidates=candidates,
                proposal=proposal, result=result,
            )
            return result

        results = []
        for index, raw in enumerate(proposal["claims"]):
            item = dict(raw)
            item_classification = item.pop("classification", classification)
            if item_classification != "stable_preference":
                results.append({
                    "stored": False, "classification": item_classification,
                    "reason": "temporary requirements need an open project working memory",
                    "idempotent": False,
                })
                continue
            item_key = (ingestion_key if len(proposal["claims"]) == 1 else
                        f"{ingestion_key}:item:{index}" if ingestion_key else None)
            results.append(self._ingest_semantic_claim(
                item, proposal=proposal, persona_id=persona_id,
                feedback=clean_feedback, context=clean_context,
                vector=vector, ingestion_key=item_key,
            ))
        result = (results[0] if len(results) == 1 else {
            "stored": any(item["stored"] for item in results),
            "classification": classification, "results": results,
            "claim_ids": [item["claim_id"] for item in results if item.get("claim_id")],
            "idempotent": False,
        })
        self._write_semantic_audit(
            ingestion_key=ingestion_key, feedback=clean_feedback,
            classification=classification, candidates=candidates,
            proposal=proposal, result=result,
        )
        return result

    def _ingest_semantic_claim(self, item: dict, *, proposal: dict,
                               persona_id: str, feedback: str, context: str,
                               vector: list[float], ingestion_key: str | None) -> dict:
        """Apply one validated atomic item and return its canonical survivor."""
        update_priority = item.pop("priority_update", False)
        partition = item.pop("partition")
        partition_reason = item.pop("partition_reason")
        relation = item.pop("relation")
        item.setdefault("conditions", [])
        item.setdefault("exceptions", [])
        item.setdefault("source_type", "explicit_feedback")
        item.setdefault("extraction_confidence", item.get("confidence", 0.7))
        tags = list(item.pop("tags", []))
        tags.extend([
            {"planning": "structure", "drawing": "style", "instructions": "instructions"}[partition],
            f"partition_reason:{partition_reason}", "semantic:llm_extracted",
        ])
        item["tags"] = tags
        claim = PreferenceClaim(persona_id=persona_id, **item)
        candidate_id = str(relation.get("candidate_id") or "")
        relation_type = str(relation.get("type", "unrelated"))
        if update_priority and relation_type in {"reinforcement", "paraphrase_duplicate", "exact_duplicate"}:
            # An explicit change in force updates the existing requirement,
            # even when the user's wording changes with it.
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM preference_claims WHERE claim_id=?",
                                   (candidate_id,)).fetchone()
            previous = decode_claim_row(row) if row else None
            incoming = normalize_claim(claim)
            identity = ("persona_id", "subject", "attribute", "direction", "scope", "conditions", "exceptions")
            if previous is None or previous["status"] != "active" or any(
                    incoming[field] != previous[field] for field in identity):
                raise ValueError("priority changes must preserve the existing requirement boundary")
            claim = replace(claim, action_or_value=previous["action_or_value"])
        stored = self.ingest_feedback(
            feedback, claim, context=context,
            ingestion_key=ingestion_key,
            embedding=vector if self.embedding_text_mode == "feedback" else None,
            update_priority=update_priority,
            reactivate=bool(relation_type == "temporal_update" and relation.get("explicit")),
        )
        relation_result = None
        relation["candidate_id"] = candidate_id
        if candidate_id and candidate_id != stored["claim_id"]:
            confidence = float(relation.get("confidence", 0.0))
            threshold = RELATION_MIN_CONFIDENCE.get(relation_type, 0.0)
            if confidence < threshold or (
                relation_type == "temporal_update" and not relation.get("explicit")
            ):
                relation_type = "uncertain"
            model = str(proposal.get("_model", ""))
            prompt_version = str(proposal.get("_prompt_version", ""))
            with self._connect() as conn:
                if relation_type == "paraphrase_duplicate":
                    relation_result = validate_relation(
                        conn, candidate_id, stored["claim_id"], relation_type,
                        confidence, [stored["evidence_id"]], model=model,
                        prompt_version=prompt_version,
                        update_priority_from=stored["claim_id"] if update_priority else None,
                    )
                elif relation_type == "contradiction":
                    relation_result = resolve_conflict(
                        conn, candidate_id, stored["claim_id"], "value_conflict",
                        proposed_action="hold_unresolved", model=model,
                        prompt_version=prompt_version,
                    )
                elif relation_type == "temporal_update":
                    relation_result = resolve_conflict(
                        conn, candidate_id, stored["claim_id"], "explicit_supersession",
                        explicit=bool(relation.get("explicit")), proposed_action="supersede",
                        model=model, prompt_version=prompt_version,
                        preferred_winner=stored["claim_id"] if relation.get("explicit") else None,
                    )
                elif relation_type == "exception":
                    relation_result = resolve_conflict(
                        conn, candidate_id, stored["claim_id"], "scope_exception",
                        proposed_action="conditionalize", model=model,
                        prompt_version=prompt_version,
                    )
                elif relation_type != "unrelated":
                    relation_result = validate_relation(
                        conn, candidate_id, stored["claim_id"], relation_type,
                        confidence, [stored["evidence_id"]], model=model,
                        prompt_version=prompt_version,
                    )
        canonical_id = ((relation_result or {}).get("canonical_claim_id")
                        or stored["claim_id"])
        with self._connect() as conn:
            current = conn.execute(
                "SELECT status FROM preference_claims WHERE claim_id=?", (canonical_id,),
            ).fetchone()
        return {
            "stored": True, "classification": "stable_preference",
            "claim_id": canonical_id, "submitted_claim_id": stored["claim_id"],
            "memory_id": stored["memory_id"],
            "status": current["status"] if current else None,
            "partition": partition, "partition_reason": partition_reason,
            "relation": relation_result, "idempotent": False,
        }

    def _available_domains(self, persona_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT scope_json FROM preference_claims WHERE persona_id=? AND status='active'",
                (persona_id,),
            ).fetchall()
        domains = set()
        for row in rows:
            value = json.loads(row[0])
            domain = value.get("domain") or value.get("value")
            if value.get("kind") == "domain" and domain:
                domains.add(str(domain))
        return sorted(domains)

    def _persist_llm_partitions(self, decisions: list[dict]) -> None:
        tag_for = {"planning": "structure", "drawing": "style", "instructions": "instructions"}
        llm_items = [item for item in decisions if item.get("source") == "llm"]
        if not llm_items:
            return
        with self._connect() as conn:
            for item in llm_items:
                row = conn.execute(
                    "SELECT tags_json FROM preference_claims WHERE claim_id=?", (item["claim_id"],),
                ).fetchone()
                if not row:
                    continue
                tags = json.loads(row[0] or "[]")
                tags = sorted(set([*tags, tag_for[item["bucket"]], "partition:llm_backfill"]))
                conn.execute(
                    "UPDATE preference_claims SET tags_json=?,updated_at=? WHERE claim_id=?",
                    (json.dumps(tags, ensure_ascii=False), datetime.utcnow().isoformat() + "Z", item["claim_id"]),
                )
            conn.commit()

    def retrieve(
        self,
        persona_id: str,
        query: str,
        *,
        scope: dict | None = None,
        top_k: int = 15,
        min_score: float = 0.35,
        token_limit: int | None = None,
        retrieval_id: str | None = None,
        record_exposure: bool = True,
    ) -> dict:
        """Return a canonical-only injection package and an auditable trace."""
        if not query or not query.strip():
            raise ValueError("query is required")
        if top_k <= 0 or (token_limit is not None and token_limit <= 0):
            raise ValueError("top_k must be positive and token_limit, when set, must be positive")
        query_vector = self._embed(query)
        encoding = get_tokenizer()
        route_meta = {"source": "caller" if scope is not None else "global_default"}
        effective_scope = scope
        if effective_scope is None and self.scope_router is not None:
            routed = self.scope_router.resolve(persona_id, query, self._available_domains(persona_id))
            route_meta = {
                "source": routed.pop("_route_source", "llm"),
                "confidence": routed.pop("_route_confidence", None),
            }
            effective_scope = routed
        effective_scope = effective_scope or {"kind": "global"}
        with self._connect() as conn:
            candidates = active_claims_for_context(
                conn, persona_id, scope=effective_scope, top_k=10000,
                token_budget=1_000_000_000, tokenizer=encoding.encode,
            )
            inactive = conn.execute("""SELECT claim_id,status FROM preference_claims
                WHERE persona_id=? AND status!='active'""", (persona_id,)).fetchall()
            active_rows = conn.execute(
                "SELECT claim_id,valid_from,valid_to FROM preference_claims WHERE persona_id=? AND status='active'",
                (persona_id,),
            ).fetchall()
            all_active_ids = {row["claim_id"] for row in active_rows}
            outside_time = {row["claim_id"] for row in active_rows
                            if not claim_is_current(dict(row))}

        priority = {"must": 1.0, "should": 0.7, "soft": 0.4}
        scored = []
        for claim in candidates:
            try:
                vector = pickle.loads(claim["embedding"]) if claim.get("embedding") else []
            except Exception as exc:
                if self.expected_embedding_dims is not None:
                    raise RuntimeError(
                        f"claim {claim['claim_id']} has an unreadable embedding") from exc
                vector = []
            if self.expected_embedding_dims is not None:
                if not vector:
                    raise RuntimeError(
                        f"claim {claim['claim_id']} has no embedding")
                if len(vector) != self.expected_embedding_dims:
                    raise RuntimeError(
                        f"claim {claim['claim_id']} embedding dim mismatch: "
                        f"got {len(vector)}, expected {self.expected_embedding_dims}")
            semantic = self._cosine(query_vector, vector)
            score = (
                0.75 * max(0.0, semantic)
                + 0.15 * float(claim.get("confidence", 0.5))
                + 0.10 * priority.get(claim.get("priority", "soft"), 0.4)
            )
            scored.append({**claim, "semantic_score": semantic, "final_score": score,
                           "filter_reasons": []})
        scored.sort(key=lambda item: (-item["final_score"], item["claim_id"]))
        eligible = [item for item in scored if item["final_score"] >= min_score]
        # Relevance chooses an axis; an applicable, validated scope exception
        # chooses its current version before the top-k cutoff is applied.
        by_id = {item["claim_id"]: item for item in scored}
        exceptions = {}
        with self._connect() as conn:
            for left, right in conn.execute("""SELECT claim_a,claim_b FROM preference_relations
                    WHERE relation_type='exception' AND active=1"""):
                if left not in by_id or right not in by_id:
                    continue
                a, b = by_id[left], by_id[right]
                if any(a.get(field, []) != b.get(field, []) for field in ("conditions", "exceptions")):
                    continue
                if scope_specificity(a["scope"]) == scope_specificity(b["scope"]):
                    continue
                broad, narrow = sorted((a, b), key=lambda item: scope_specificity(item["scope"]))
                if not scopes_compatible(broad["scope"], narrow["scope"]):
                    continue
                current = exceptions.get(broad["claim_id"])
                if current is None or (scope_specificity(narrow["scope"]), narrow["final_score"], narrow["claim_id"]) > (
                        scope_specificity(current["scope"]), current["final_score"], current["claim_id"]):
                    exceptions[broad["claim_id"]] = narrow
        selected, selected_ids = [], set()
        for item in eligible:
            while item["claim_id"] in exceptions:
                item = exceptions[item["claim_id"]]
            if item["claim_id"] not in selected_ids:
                selected.append(item)
                selected_ids.add(item["claim_id"])
            if len(selected) >= top_k:
                break
        selected_ids = {item["claim_id"] for item in selected}
        for item in scored:
            if item["claim_id"] not in selected_ids:
                item["filter_reasons"].append(
                    "scoped_exception_applies" if item["claim_id"] in exceptions else
                    "below_threshold" if item["final_score"] < min_score else "top_k"
                )

        variants = build_memory_out_variants(
            selected, token_limit=token_limit, partition_resolver=self.partition_resolver,
        )
        self._persist_llm_partitions(variants["partition_decisions"])
        exposed_ids = variants["claim_ids"]
        event_id = retrieval_id or str(uuid.uuid4())
        with self._connect() as conn:
            record_claim_signals(
                conn, [item["claim_id"] for item in selected], "retrieved",
                event_key=f"{event_id}:retrieved",
            )
            if record_exposure:
                record_claim_signals(
                    conn, exposed_ids, "exposed", event_key=f"{event_id}:exposed",
                )
        trace = {
            "retrieval_id": event_id, "persona_id": persona_id, "query": query,
            "candidate_ids": [item["claim_id"] for item in scored],
            "scores": [{
                "id": item["claim_id"],
                "semantic_score": round(item["semantic_score"], 6),
                "final_score": round(item["final_score"], 6),
                "filter_reasons": item["filter_reasons"],
            } for item in scored],
            "filtered": [
                {"id": row["claim_id"], "filter_reasons": [f"status:{row['status']}"]}
                for row in inactive
            ] + [
                {"id": claim_id, "filter_reasons": [
                    "outside_validity_window" if claim_id in outside_time else "scope_mismatch"]}
                for claim_id in sorted(all_active_ids - {item["claim_id"] for item in scored})
            ] + [
                {"id": item["claim_id"], "filter_reasons": item["filter_reasons"]}
                for item in scored if item["filter_reasons"]
            ],
            "injected_ids": exposed_ids,
            "injected_tokens": variants["partitioned_tokens"],
            "token_limit": token_limit,
            "scope": effective_scope,
            "scope_route": route_meta,
            "partition_decisions": variants["partition_decisions"],
            "exposure_recorded": bool(record_exposure),
        }
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
        return {
            "memory_out": variants["partitioned"], "control": variants["control"],
            "claims": [{key: value for key, value in item.items() if key != "embedding"}
                       for item in selected if item["claim_id"] in exposed_ids],
            "trace": trace,
        }

    def record_execution(self, claim_ids: Iterable[str], *, event_key: str | None = None) -> dict:
        with self._connect() as conn:
            return record_claim_signals(conn, list(claim_ids), "executed", event_key=event_key)

    def record_exposure(self, claim_ids: Iterable[str], *, event_key: str | None = None) -> dict:
        """Record actual prompt exposure separately from retrieval.

        The V20 project layer retrieves long-term memory exactly once at job
        start with ``record_exposure=False``.  It then calls this method for
        the durable claims that are actually injected in each fresh round.
        """
        with self._connect() as conn:
            return record_claim_signals(
                conn, list(claim_ids), "exposed", event_key=event_key)

    def confirm(self, claim_ids: Iterable[str], *, event_key: str | None = None) -> dict:
        with self._connect() as conn:
            return record_claim_signals(conn, list(claim_ids), "confirmed", event_key=event_key)

    def reject(self, claim_ids: Iterable[str], *, event_key: str | None = None) -> dict:
        with self._connect() as conn:
            return record_claim_signals(conn, list(claim_ids), "rejected", event_key=event_key)

    def record_contradiction(self, claim_ids: Iterable[str], *,
                             event_key: str | None = None) -> dict:
        with self._connect() as conn:
            return record_claim_signals(
                conn, list(claim_ids), "contradiction", event_key=event_key)

    def set_status(self, claim_id: str, status: str, *, reason: str = "",
                   event_key: str | None = None) -> dict:
        with self._connect() as conn:
            return set_claim_status(
                conn, claim_id, status, reason=reason, event_key=event_key,
            )

    def signals(self, claim_id: str) -> dict:
        with self._connect() as conn:
            return get_claim_signals(conn, claim_id)
