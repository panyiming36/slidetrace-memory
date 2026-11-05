"""Build V19 partitioned and unpartitioned memory injection packages."""
from __future__ import annotations

import json
from typing import Callable, Iterable

if __package__:
    from .tokenization import count_tokens
else:
    from tokenization import count_tokens


def _tags(claim: dict) -> set[str]:
    raw = claim.get("tags", claim.get("tags_json", []))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [raw]
    return {str(tag).strip().lower() for tag in (raw or []) if str(tag).strip()}


def _bucket_with_source(claim: dict) -> tuple[str, str]:
    tags = _tags(claim)
    descriptor = " ".join(
        str(claim.get(key, "")).lower() for key in ("subject", "attribute")
    )
    if any(tag == "structure" or tag.startswith("structure:") for tag in tags):
        return "planning", "tag"
    if any(tag == "style" or tag.startswith("style:") for tag in tags):
        return "drawing", "tag"
    if any(tag == "instructions" or tag.startswith("instructions:") for tag in tags):
        return "instructions", "tag"
    if any(
        word in descriptor for word in (
            "structure", "layout", "outline", "结构", "布局", "大纲", "页面规划",
            "叙事", "章节", "信息密度", "density", "narrative",
        )
    ):
        return "planning", "dictionary"
    if any(
        word in descriptor for word in (
            "style", "color", "font", "visual", "风格", "配色", "字体", "视觉",
            "图表", "公式", "渲染", "chart", "formula", "render", "标注", "背景",
            "字号", "间距", "曲线",
        )
    ):
        return "drawing", "dictionary"
    return "instructions", "default"


def _bucket(claim: dict) -> str:
    return _bucket_with_source(claim)[0]


def _prepare_partitions(
    claims: Iterable[dict],
    partition_resolver: Callable[[list[dict]], dict[str, str]] | None,
) -> tuple[list[dict], list[dict]]:
    """Resolve only legacy claims missed by both stored tags and the dictionary."""
    prepared = [dict(claim) for claim in claims]
    unresolved = [
        claim for claim in prepared if _bucket_with_source(claim)[1] == "default"
    ]
    llm_result = partition_resolver(unresolved) if unresolved and partition_resolver else {}
    decisions = []
    valid = {"planning", "drawing", "instructions"}
    tag_for = {"planning": "structure", "drawing": "style", "instructions": "instructions"}
    for claim in prepared:
        bucket, source = _bucket_with_source(claim)
        claim_id = str(claim.get("claim_id", ""))
        if source == "default" and claim_id in llm_result:
            proposed = str(llm_result[claim_id]).strip().lower()
            if proposed in valid:
                bucket, source = proposed, "llm"
                tags = list(_tags(claim))
                tags.append(tag_for[bucket])
                claim["tags"] = sorted(set(tags))
        claim["_memory_bucket"] = bucket
        decisions.append({"claim_id": claim_id, "bucket": bucket, "source": source})
    return prepared, decisions


def _line(claim: dict) -> str:
    text = str(claim.get("text") or "").strip()
    if not text:
        text = (
            f"{claim.get('direction', 'prefer')} {claim.get('subject', '')} "
            f"{claim.get('attribute', '')}: {claim.get('action_or_value', '')}"
        ).strip()
    qualifiers = []
    scope = claim.get("scope") or {}
    if scope.get("kind") != "global":
        bounds = [f"{key}={value}" for key, value in scope.items()
                  if key != "kind" and value]
        if bounds:
            qualifiers.append("适用：" + ", ".join(bounds))
    for field, label in (("conditions", "条件"), ("exceptions", "例外")):
        values = claim.get(field) or []
        if values:
            qualifiers.append(label + "：" + "；".join(str(value) for value in values))
    suffix = " [" + " | ".join(qualifiers) + "]" if qualifiers else ""
    return f"- {text}{suffix}"


def _render_partitioned(claims: Iterable[dict]) -> str:
    buckets = {"planning": [], "drawing": [], "instructions": []}
    for claim in claims:
        buckets[str(claim.get("_memory_bucket") or _bucket(claim))].append(_line(claim))
    return "\n\n".join((
        "【规划区】\n" + ("\n".join(buckets["planning"]) or "- 无"),
        "【绘制区】\n" + ("\n".join(buckets["drawing"]) or "- 无"),
        "【使用说明】\n" + ("\n".join(buckets["instructions"]) or "- 无"),
    ))


def _render_control(claims: Iterable[dict]) -> str:
    return "【未分区对照】\n" + ("\n".join(_line(c) for c in claims) or "- 无")


def build_memory_out_variants(
    claims: Iterable[dict], *, token_limit: int | None = None, model: str = "gpt-4o-mini",
    partition_resolver: Callable[[list[dict]], dict[str, str]] | None = None,
) -> dict:
    """Render one common claim set; an explicit token limit remains optional."""
    if token_limit is not None and token_limit <= 0:
        raise ValueError("token_limit must be positive")
    prepared, partition_decisions = _prepare_partitions(claims, partition_resolver)
    selected: list[dict] = []
    truncated = False
    if token_limit is None:
        selected = prepared
    else:
        for claim in prepared:
            proposal = [*selected, claim]
            if max(count_tokens(_render_partitioned(proposal), model),
                   count_tokens(_render_control(proposal), model)) > token_limit:
                truncated = True
                continue
            selected = proposal
    partitioned = _render_partitioned(selected)
    control = _render_control(selected)
    selected_ids = {str(claim.get("claim_id", "")) for claim in selected}
    return {
        "partitioned": partitioned,
        "control": control,
        "claim_ids": [str(c.get("claim_id", "")) for c in selected],
        "token_limit": token_limit,
        "partitioned_tokens": count_tokens(partitioned, model),
        "control_tokens": count_tokens(control, model),
        "truncated": truncated,
        "partition_decisions": [
            item for item in partition_decisions if item["claim_id"] in selected_ids
        ],
    }
