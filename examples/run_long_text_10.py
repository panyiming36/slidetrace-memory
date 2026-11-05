"""Ten sequential text-only host tasks using the real installable memory core.

The semantic callback and vectors are fixtures. SQLite, scope filtering,
feedback governance, retrieval, exposure, confirmation, project close, restart,
revision and forgetting are real plugin calls. No network or slide rendering.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidetrace_memory import PptMemory


AXES = {
    "title": ("页面标题", "措辞", "planning"),
    "background": ("页面", "背景", "drawing"),
    "avatar": ("人物头像", "形状", "drawing"),
    "unit": ("财务数据", "金额单位", "instructions"),
    "loss_color": ("财务亏损", "颜色", "drawing"),
    "font": ("正文", "字号", "drawing"),
    "footer": ("页脚", "状态标识", "instructions"),
}
GLOBAL = {"kind": "global"}
PEOPLE = {"kind": "page_type", "page_type": "people"}
FINANCE = {"kind": "domain", "domain": "finance"}
EXTERNAL_FINANCE = {**FINANCE, "audience": "external"}

MESSAGES = {
    "T01": "以后所有PPT用浅色背景、结论式标题，人物介绍页统一用圆形头像。",
    "T02": "财务数据以后以万元为单位，亏损用红色；仅这份PPT临时改深色背景。",
    "T03": "收到，谢谢，继续。",
    "T04": "以后人物介绍页改用方形头像，原先圆形要求作废。",
    "T05": "对外部受众的财务报告，金额改用元；内部财务报告仍按万元。",
    "T06": "以后所有PPT改红色背景；这份PPT第三页加内审草稿标记。",
    "T07": "以后所有PPT正文字号不小于28。",
    "T08": "人物介绍页方形头像改成可选偏好，不再是原来的建议优先级。",
    "T10": "以后人物介绍页重新使用圆形头像；标题继续用结论句。",
}

TASKS = [
    ("T01", "集团公司与人物介绍", "company", "internal", ["title", "people", "process"], "学习三条偏好"),
    ("T02", "季度财务经营回顾", "finance", "internal", ["title", "data", "table"], "财务专属规则＋本项目深底"),
    ("T03", "品牌营销团队与活动方案", "campaign", "external", ["title", "people", "process"], "跨领域复用＋临时要求消退＋噪声"),
    ("T04", "产品团队人物与能力介绍", "product", "internal", ["title", "people", "process"], "圆形明确改成方形"),
    ("T05", "对外年度财务数据说明", "finance", "external", ["title", "data", "table"], "外部受众的金额单位例外"),
    ("T06", "内部财务预算审核草案", "finance", "internal", ["title", "data", "table"], "内部默认恢复＋未接受项目不晋升"),
    ("T07", "对外财务业务结果简报", "finance", "external", ["title", "data", "table"], "工作反馈后重启恢复＋外部例外持续"),
    ("T08", "招聘人物与组织介绍", "recruiting", "internal", ["title", "people", "process"], "隔多个项目仍记得＋优先级降低"),
    ("T09", "研发团队与人物介绍", "engineering", "internal", ["title", "people", "process"], "待生效修订后忘掉整个工作链"),
    ("T10", "新部门人物与协作介绍", "company", "internal", ["title", "people", "process"], "重新学习旧偏好＋重复要求不膨胀"),
]


class LocalVectors:
    model = "offline-feature-vector-10-task"

    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        words = ("人物", "people", "finance", "财务", "背景", "标题", "单位", "字号", "external")
        lowered = text.lower()
        return [2.0] + [0.7 if word in lowered else 0.0 for word in words]


class ScriptedSemantics:
    model = "offline-scripted-feedback"

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        match = re.search(r"^反馈: (.*)$", prompt, re.MULTILINE)
        if match is None:
            raise ValueError("unexpected semantic repair prompt; inspect the fixture contract")
        feedback = match.group(1)
        tid = next((key for key, text in MESSAGES.items() if text == feedback), None)
        if tid is None:
            raise ValueError(f"no scripted response for feedback: {feedback}")
        scope_match = re.search(r"^当前 scope: (.*)$", prompt, re.MULTILINE)
        current = json.loads(scope_match.group(1))
        candidates = []
        for line in prompt.splitlines():
            candidate = re.match(
                r"- id=(\S+) .*?subject=(.*?) attribute=(.*?) direction=(.*?) "
                r"priority=(.*?) scope=(\{.*?\}) value=(.*?) conditions=", line)
            if candidate:
                cid, subject, attribute, direction, priority, bounds, value = candidate.groups()
                candidates.append({"id": cid, "subject": subject, "attribute": attribute,
                                   "scope": json.loads(bounds), "value": value})

        def claim(axis, value, bounds, *, relation=None, temporary=False,
                  priority="should", priority_update=False):
            subject, attribute, partition = AXES[axis]
            options = [c for c in candidates if c["subject"] == subject and c["attribute"] == attribute]
            options.sort(key=lambda c: (c["scope"] == bounds, c["value"] == value), reverse=True)
            old = options[0] if options else None
            kind = relation if old and relation else "unrelated"
            return {
                "classification": "one_off_incident" if temporary else "stable_preference",
                "subject": subject, "attribute": attribute, "action_or_value": value,
                "direction": "prefer", "scope": dict(bounds), "priority": priority,
                "conditions": [], "exceptions": [], "confidence": 0.95,
                "partition": partition, "partition_reason": "离线场景固定分析结果",
                "priority_update": priority_update,
                "relation": {"candidate_id": old["id"] if kind != "unrelated" else "",
                             "type": kind, "confidence": 0.99 if old else 0.0,
                             "explicit": kind in {"temporal_update", "reinforcement"}},
            }

        if tid == "T01":
            claims = [claim("background", "浅色背景", GLOBAL),
                      claim("title", "结论式标题", GLOBAL), claim("avatar", "圆形头像", PEOPLE)]
        elif tid == "T02":
            claims = [claim("unit", "万元", FINANCE), claim("loss_color", "红色", FINANCE),
                      claim("background", "深色背景", {"kind": "task", "task_id": current["task_id"]},
                            relation="exception", temporary=True)]
        elif tid == "T03":
            return {"classification": "noise", "reason": "礼貌回应不是偏好", "claims": []}
        elif tid == "T04":
            claims = [claim("avatar", "方形头像", PEOPLE, relation="temporal_update")]
        elif tid == "T05":
            claims = [claim("unit", "元", EXTERNAL_FINANCE, relation="exception")]
        elif tid == "T06":
            claims = [claim("background", "红色背景", GLOBAL, relation="temporal_update"),
                      claim("footer", "内审草稿", {"kind": "page", "task_id": current["task_id"],
                                                   "page_id": "p3"}, temporary=True)]
        elif tid == "T07":
            claims = [claim("font", "28", GLOBAL)]
        elif tid == "T08":
            claims = [claim("avatar", "方形头像", PEOPLE, relation="reinforcement",
                            priority="soft", priority_update=True)]
        elif tid == "T10":
            claims = [claim("avatar", "圆形头像", PEOPLE),
                      claim("title", "结论式标题", GLOBAL, relation="reinforcement")]
        else:
            raise ValueError(tid)
        return {"classification": "stable_preference", "reason": "固定文字场景",
                "claims": claims}


class Simulation:
    def __init__(self, output):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = output / "ten_tasks.memory.db"
        if self.db.exists():
            raise RuntimeError("choose an unused output directory; do not reuse a previous run")
        self.vectors, self.semantics = LocalVectors(), ScriptedSemantics()
        self.tasks, self.checks = [], []
        self.task_id = "setup"
        self.started = time.perf_counter()
        self.memory = self.new_instance()

    def new_instance(self):
        return PptMemory(self.db, user_id="long-text-user", embedder=self.vectors,
                         complete_json=self.semantics, token_limit=1200)

    def check(self, name, condition, detail=None):
        item = {"task": self.task_id, "name": name, "passed": bool(condition), "detail": detail}
        self.checks.append(item)
        if not condition:
            raise AssertionError(f"{self.task_id}: {name}: {detail}")

    def canonical(self, axis=None, *, active=True):
        items = self.memory.list_memories(include_inactive=not active)["canonical"]
        if axis:
            subject, attribute, _ = AXES[axis]
            items = [c for c in items if c["subject"] == subject and c["attribute"] == attribute]
        return items

    def pages(self, round_no, page_types):
        pages = []
        for index, page_type in enumerate(page_types, 1):
            packet = self.memory.context(round_no=round_no, page_id=f"p{index}", page_type=page_type)
            fields = {"title": "叙述式标题", "background": "深色背景", "avatar": "默认矩形头像",
                      "unit": "未指定", "loss_color": "自动", "font": "22", "footer": "无"}
            implemented = []
            for claim in packet["claims"]:
                axis = next((key for key, (subject, attribute, _) in AXES.items()
                             if claim["subject"] == subject and claim["attribute"] == attribute), None)
                if axis:
                    fields[axis] = claim["action_or_value"]
                    implemented.append(claim["claim_id"])
            pages.append({"page_id": f"p{index}", "page_type": page_type, "fields": fields,
                          "implemented_ids": implemented, "packet": packet})
        return pages

    @staticmethod
    def field(pages, key, index=0):
        return pages[index]["fields"][key]

    @staticmethod
    def axes(pages):
        return {(c["subject"], c["attribute"]) for page in pages for c in page["packet"]["claims"]}

    def run_task(self, definition):
        tid, topic, domain, audience, page_types, focus = definition
        self.task_id = tid
        opening = self.memory.start_project(tid, topic, domain=domain, audience=audience,
                                            page_ids=["p1", "p2", "p3"])
        before = self.pages(1, page_types)
        row = {"task": tid, "topic": topic, "focus": focus, "opening": opening,
               "before": before, "feedback": MESSAGES.get(tid), "satisfied": tid != "T06"}
        self.tasks.append(row)
        if tid != "T01":
            self.check("跨项目保留浅色背景", all(p["fields"]["background"] == "浅色背景" for p in before))
            self.check("跨项目保留结论式标题", all(p["fields"]["title"] == "结论式标题" for p in before))
        if tid in {"T03", "T04"}:
            self.check("旧人物偏好能跨项目取回", self.field(before, "avatar", 1) == "圆形头像")
        if tid in {"T03", "T08", "T09", "T10"}:
            self.check("财务专属规则未带入其他域", AXES["unit"][:2] not in self.axes(before))
        if tid == "T03":
            self.check("上题临时深底未泄漏", all(p["fields"]["background"] != "深色背景" for p in before))
        if tid in {"T06", "T07"}:
            self.check("受众单位按范围正确恢复", self.field(before, "unit", 1) == ("万元" if tid == "T06" else "元"))
        if tid == "T07":
            self.check("未接受的红底修改未进入下题", self.field(before, "background") == "浅色背景")
            self.check("上题页脚未泄漏", all(p["fields"]["footer"] == "无" for p in before))
        if tid in {"T08", "T09", "T10"}:
            self.check("字号偏好跨项目持久保留", all(p["fields"]["font"] == "28" for p in before))
        if tid == "T08":
            self.check("隔四个项目仍保留方形要求", self.field(before, "avatar", 1) == "方形头像")
        if tid == "T10":
            self.check("忘记后的头像规则不再取回", AXES["avatar"][:2] not in self.axes(before))

        if tid == "T09":
            cid = self.canonical("avatar")[0]["claim_id"]
            row["revision"] = self.memory.revise(cid, round_no=1, action_or_value="六边形头像")
            row["forget"] = self.memory.forget(cid)
        else:
            options = ({"page_id": "p2", "page_type": "people"}
                       if tid in {"T01", "T04", "T08", "T10"} else {})
            row["feedback_result"] = self.memory.feedback(MESSAGES[tid], round_no=1,
                                                           feedback_id=f"{tid}-feedback", **options)
            if tid == "T03":
                self.check("礼貌噪声不写为偏好", row["feedback_result"]["stored"] is False)
            if tid in {"T04", "T08"}:
                old = self.canonical("avatar")[0]
                self.check("执行确认前长期库尚未被改写", old["action_or_value"] ==
                           ("圆形头像" if tid == "T04" else "方形头像") and old["priority"] == "should")
            if tid == "T07":
                session = self.memory.session_id
                calls = self.vectors.calls
                self.memory = self.new_instance()
                resumed = self.memory.resume_project(session)
                self.check("反馈后跨实例恢复同一项目", resumed["session_id"] == session)
                self.check("恢复不重复向量检索", self.vectors.calls == calls)
                row["resumed"] = resumed

        after = self.pages(2, page_types)
        row["after"] = after
        if tid == "T01":
            self.check("一段反馈三条全部进入工作记忆", len(row["feedback_result"]["results"]) == 3)
            self.check("新头像要求进入人物页", self.field(after, "avatar", 1) == "圆形头像")
        if tid == "T02":
            self.check("财务单位与颜色全部注入", self.field(after, "unit", 1) == "万元" and
                       self.field(after, "loss_color", 1) == "红色")
            self.check("临时深底在本项目全部页面生效", all(p["fields"]["background"] == "深色背景" for p in after))
        if tid == "T04":
            self.check("明确改口后仅用方形", self.field(after, "avatar", 1) == "方形头像" and not any(
                c["action_or_value"] == "圆形头像" for p in after for c in p["packet"]["claims"]))
        if tid == "T05":
            self.check("外部例外覆盖万元默认", self.field(after, "unit", 1) == "元" and not any(
                c["subject"] == AXES["unit"][0] and c["action_or_value"] == "万元"
                for p in after for c in p["packet"]["claims"]))
        if tid == "T06":
            self.check("页脚临时要求只影响第三页", [p["fields"]["footer"] for p in after] == ["无", "无", "内审草稿"])
            self.check("待拒绝修改先在项目内生效", self.field(after, "background") == "红色背景")
        if tid == "T07":
            self.check("重启后未晋升反馈仍可应用", all(p["fields"]["font"] == "28" for p in after))
        if tid == "T08":
            current = [c for p in after for c in p["packet"]["claims"] if c["subject"] == AXES["avatar"][0]]
            self.check("优先级降低应用于工作记忆", current and all(c["priority"] == "soft" for c in current))
        if tid == "T09":
            self.check("忘记同时清掉未提交修订", AXES["avatar"][:2] not in self.axes(after))
        if tid == "T10":
            self.check("原来撤下的圆形要求可以重新学习", self.field(after, "avatar", 1) == "圆形头像")

        implemented = sorted({cid for p in after for cid in p["implemented_ids"]})
        row["outcome"] = self.memory.record_outcome(
            round_no=2, implemented_ids=implemented,
            accepted_ids=implemented if row["satisfied"] else [], outcome_id=f"{tid}-r2")
        row["finish"] = self.memory.finish_project(satisfied=row["satisfied"], final_round=2)
        row["store_after"] = self.memory.list_memories(include_inactive=True)["canonical"]
        active = [c for c in row["store_after"] if c["status"] == "active"]
        self.check("没有项目/页临时规则混入长期库", not any(c["scope"].get("task_id") or
                   c["scope"].get("project_id") or c["scope"].get("page_id") for c in active))
        if tid == "T02":
            self.check("临时深底未覆盖长期浅底", [c["action_or_value"] for c in self.canonical("background")] == ["浅色背景"])
        if tid == "T04":
            self.check("验收后旧圆形退休", [c["action_or_value"] for c in self.canonical("avatar")] == ["方形头像"])
        if tid == "T05":
            self.check("默认和受众例外均保留", {c["action_or_value"] for c in self.canonical("unit")} == {"万元", "元"})
        if tid == "T06":
            self.check("项目拒绝则零新条目晋升", row["finish"]["promoted"] == [])
            self.check("拒绝修改不退休原浅底", [c["action_or_value"] for c in self.canonical("background")] == ["浅色背景"])
        if tid == "T08":
            self.check("验收后明确降级长期生效", self.canonical("avatar")[0]["priority"] == "soft")
        if tid == "T09":
            self.check("被忘记的头像与六边形均不晋升", self.canonical("avatar") == [])
        if tid == "T10":
            self.check("重新学习的圆形长期状态恢复", [c["action_or_value"] for c in self.canonical("avatar")] == ["圆形头像"])
            self.check("重复标题反馈不新增长期副本", len(self.canonical("title")) == 1)
        row["active_long_term"] = len(active)
        self.save_task(row)
        print(f"{tid} completed; active_long_term={len(active)}; checks={len(self.checks)}", flush=True)

    def save_task(self, row):
        target = self.output / "tasks"
        target.mkdir(exist_ok=True)
        (target / f"{row['task']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [f"{row['task']} {row['topic']}", f"检查重点：{row['focus']}",
                 f"反馈：{row.get('feedback') or '先修订为六边形，再明确忘记原偏好及其工作链'}"]
        for stage in ("before", "after"):
            lines.append("\n" + ("第一轮（既有记忆）" if stage == "before" else "第二轮（处理本题反馈后）"))
            for page in row[stage]:
                lines.append(f"{page['page_id']} [{page['page_type']}] " + "；".join(
                    f"{key}={value}" for key, value in page["fields"].items()))
        lines += [f"\n项目接受：{row['satisfied']}", f"结束后活跃长期条目：{row['active_long_term']}"]
        (target / f"{row['task']}.txt").write_text("\n".join(lines), encoding="utf-8")

    def save_summary(self, error=None):
        passed = sum(item["passed"] for item in self.checks)
        tokens = [p["packet"]["trace"]["injected_tokens"] for t in self.tasks
                  for stage in ("before", "after") for p in t.get(stage, [])]
        summary = {"completed_tasks": sum("finish" in t for t in self.tasks),
                   "checks_passed": passed, "checks_total": len(self.checks),
                   "real_model_calls": 0, "ppt_generated": 0,
                   "scripted_semantic_calls": self.semantics.calls,
                   "fake_embedding_calls": self.vectors.calls,
                   "elapsed_seconds": round(time.perf_counter() - self.started, 3),
                   "total_injected_context_tokens": sum(tokens),
                   "mean_injected_context_tokens": round(sum(tokens) / len(tokens), 2) if tokens else 0,
                   "error": error, "checks": self.checks,
                   "final_store": self.memory.list_memories(include_inactive=True)["canonical"]}
        (self.output / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# 10 任务连续文字模拟", "",
                 "同一个 SQLite 库连续运行 10 个任务，每题三页文字、两轮。没有生成 PPT、调用真实模型或联网。",
                 "反馈分析结果和向量为离线脚本桩；真实运行的是插件的写入、检索、治理、作用域、执行确认、晋升及恢复。",
                 "文本页面由固定宿主规则应用实际取回的 claim；执行与接受是模拟宿主信号，不代表真实页面质量。", "",
                 f"完成：{summary['completed_tasks']}/10；检查：{passed}/{len(self.checks)}；用时：{summary['elapsed_seconds']} 秒。",
                 f"真实模型调用：0；三页文本上下文平均注入：{summary['mean_injected_context_tokens']} token/次（无 API 费用）。", "",
                 "| 任务 | 内容 | 重点 | 活跃长期条目 | 本题检查 |",
                 "|---|---|---|---:|---|" ]
        for row in self.tasks:
            checks = [c for c in self.checks if c["task"] == row["task"]]
            lines.append(f"| {row['task']} | {row['topic']} | {row['focus']} | {row.get('active_long_term', '未结束')} | "
                         f"{sum(c['passed'] for c in checks)}/{len(checks)} |")
        lines += ["", "## 每项检查", ""]
        lines.extend(f"- {'通过' if c['passed'] else '失败'} · {c['task']} · {c['name']}" for c in self.checks)
        if error:
            lines += ["", "## 中断原因", "", "```text", error, "```"]
        (self.output / "模拟报告.md").write_text("\n".join(lines), encoding="utf-8")
        texts = ["10 个任务的连续文字过程", "同一记忆库，三页文字/题；采用离线语义桩和假向量。",
                 "没有调用模型、联网或生成 PPT。页面字段来自插件实际取回的记忆。", ""]
        texts.extend((self.output / "tasks" / f"{row['task']}.txt").read_text(encoding="utf-8")
                     for row in self.tasks if "finish" in row)
        texts += ["\n最终活跃长期记忆："]
        texts.extend(f"{claim['subject']}／{claim['attribute']}：{claim['action_or_value']}；"
                     f"范围={json.dumps(claim['scope'], ensure_ascii=False)}；优先级={claim['priority']}"
                     for claim in summary["final_store"] if claim["status"] == "active")
        (self.output / "10任务文字过程.txt").write_text("\n\n".join(texts), encoding="utf-8")
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    simulation = Simulation(arguments.output_dir)
    try:
        for definition in TASKS:
            simulation.run_task(definition)
    except Exception:
        error = traceback.format_exc()
        simulation.save_summary(error)
        print(error, file=sys.stderr)
        return 1
    summary = simulation.save_summary()
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"checks", "final_store"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
