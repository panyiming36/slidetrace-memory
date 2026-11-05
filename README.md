# SlideTrace PPT 记忆插件

版本：0.1.0。Python 3.10 或以上。

这是给现有 PPT 生成程序调用的 Python 记忆插件。宿主生成和修改页面，插件维护用户偏好、当前项目工作记忆及跨项目长期记忆，并返回可以放进提示词的上下文。

## 安装

Python 3.10 或以上。在插件源码目录执行：

```powershell
python -m pip install .
```

必需的第三方依赖只有 `tiktoken`。SQLite 和 HTTP 客户端使用 Python 标准库，不安装生成器、实验评委、基线系统或 NLP 模型。

## 配置

插件接受宿主已有的两个回调：

- `embedder(text) -> list[float]`：为写入与检索生成向量，同一库使用同一模型。首次调用时锁定维度；全零或含非有限值的向量会被拒绝。
- `complete_json(prompt) -> dict`：将自然语言反馈拆成原子偏好并提出关系。插件校验后再更新记忆。返回值必须是一个对象：`classification` 取 `stable_preference / one_off_incident / noise / uncertain`（noise 与 uncertain 不允许携带 claims）；每个 claim 必须包含 `subject`、`attribute`、`action_or_value`、`direction`、`scope`、`priority`、`confidence`、`partition`（`planning / drawing / instructions` 之一）、`partition_reason`、`relation`（对象，含 `candidate_id / type / confidence / explicit`；`type` 非 `unrelated / uncertain` 时必须给出召回候选的 `candidate_id`）与可选布尔 `priority_update`。scope 的取值边界：`global` 不得带其他键；其余 `kind` 必须落在宿主当前边界内（例如 `page` 级必须带本项目已知的 `project_id/task_id + page_id`，`page_type` 只能使用当前上下文出现过的类型名）。校验失败时插件会把错误拼进修复提示重试（默认最多 3 次）。

也可以使用内置的兼容 OpenAI 接口客户端：

```python
import os
from slidetrace_memory import PptMemory, ChatClient, EmbeddingClient

memory = PptMemory(
    "./data/ppt_memory.db",
    user_id="user-001",
    embedder=EmbeddingClient(
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.environ["EMBEDDING_API_KEY"],
        model=os.environ["EMBEDDING_MODEL"],
    ),
    complete_json=ChatClient(
        base_url=os.environ["CHAT_BASE_URL"],
        api_key=os.environ["CHAT_API_KEY"],
        model=os.environ["CHAT_MODEL"],
    ),
)
```

地址填写服务商的 API 根地址；需要 `/v1` 时包含它。这里只调用反馈分析与向量接口，不生成 PPT。同一用户在不同会话继续使用同一个数据库路径和 `user_id`；不同用户由 `user_id` 隔离。

`ChatClient` 默认超时为 120 秒。响应较慢的思考型模型可显式设置 `timeout=600`；本轮 GLM-5.3-Flash 接入验证使用了该配置。

## 宿主接入流程

```python
project = memory.start_project(
    "project-001", "公司介绍", domain="company_profile",
    page_ids=["p1", "p2", "p3"],
)
session_id = project["session_id"]  # 保存它，重启后可以续接。

packet = memory.context(round_no=1, page_id="p2", page_type="people")
# 宿主将 packet["memory_out"] 加入生成提示词，自己生成和展示页面。

memory.feedback(
    "以后人物介绍页用圆形头像；这次第二页暂时用深色背景。",
    round_no=1, page_id="p2", page_type="people", feedback_id="feedback-001",
)
packet = memory.context(round_no=2, page_id="p2", page_type="people")
# 宿主再生成页面，使用 packet["claims"] 中的工作记忆 ID 关联实际执行情况。
```

宿主确认页面实现后，回传实际命中的 ID；用户接受后，回传其中被接受的 ID：

```python
def on_host_evaluation(implemented_ids, accepted_ids, rejected_ids):
    return memory.record_outcome(
        round_no=2,
        implemented_ids=implemented_ids,
        accepted_ids=accepted_ids,
        rejected_ids=rejected_ids,
        outcome_id="project-001-round-2",
    )

# 宿主判断项目已被接受后调用；不接受则 satisfied=False。
# memory.finish_project(satisfied=True, final_round=2)
```

这些 ID 来自当前轮 `context()` 的 `claims`，不是由插件自动判定页面已实现。`accepted_ids` 必须同时出现在本轮 `implemented_ids` 中。宿主可以来自实际页面检查和用户确认，无需安装本项目的实验判卷器。

一轮生成涉及多页时，对各页调用 `context()`，然后将该轮真实命中的 ID 去重后统一回传。只处理一次回传时使用稳定的 `outcome_id`；确实有多次独立检查时使用不同 ID。

## 沿用的记忆规则

- 新反馈先进入当前项目的工作记忆，在下一次获取上下文时生效。
- 可复用偏好只有在项目接受、该条有执行和确认记录时，才进入长期库。
- “这次”“这一页”等临时要求留在对应项目或页面，项目结束不晋升为长期默认。
- 改口沿用替代、例外与优先级治理；局部例外不抹掉其他场景的默认偏好。
- 过期和未到生效时间的条目不注入。条件、例外、作用域随上下文一起返回。

宿主应提供稳定的 `domain`、`page_type` 等名称。同一“人物介绍页”不要在不同调用中无故改成不同类型名。跨领域通用的页型偏好可使用 `scope={"kind": "page_type", "page_type": "people"}`；特定领域偏好用 `domain`，当前项目或页面用 `project`/`task`/`page`。

## 恢复与管理

```python
memory.list_projects()                     # 找到尚未结束的项目。
memory.resume_project(saved_session_id)     # 新实例继续原项目，不重复新建。
memory.list_memories(include_working=True)  # 长期记忆和当前项目工作记忆。
memory.revise(claim_id, round_no=3, action_or_value="人物介绍使用方形头像")
memory.forget(claim_id)                     # 立即停用，保留历史记录供查看。
```

修改偏好仍走工作记忆、执行确认与项目接受的原有过程。明确忘记会同时停用长期条目和开放项目里的对应工作副本。之后重新提出同一要求，需重新应用和接受，才会恢复为长期偏好。

宿主已经有结构化偏好时，可以不配置 `complete_json`，调用 `feedback_claim()`：提供 `subject`、`attribute`、`action_or_value`、`direction`，以及需要的 `scope`、`priority`、条件和例外。向量回调仍用于长期写入与检索。若传入的 dict 混入了分析器专有字段（`classification`、`partition`、`partition_reason`、`relation`、`priority_update`），插件会忽略或转用对应参数（`relation`、`priority_update` 形参优先），不会报错。

## 当前交付状态

本版完成多 agent 静态审查及接入验证。接入方记录了安装成功、31 项离线冒烟通过，以及真实 embedding 与 LLM 的记忆端到端验证。本轮另执行 5 项定向回归，均通过，覆盖工作项撤回、修订链和结构化字段兼容。各轮结果分别记录，未将重叠场景合并计数。实际 PPT 宿主的页面生成与反馈联调尚未进行。

仓库另附可运行的 5 项定向回归及连续 10 任务文字模拟（70 项检查通过）。命令、结果与验证范围见 [验证说明](docs/VALIDATION.md)。
