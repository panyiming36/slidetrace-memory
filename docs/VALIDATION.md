# 插件验证

## 可直接运行的离线验证

在仓库根目录安装后运行：

```shell
python -m pip install .
python -m unittest discover -s tests -v
python examples/run_long_text_10.py --output-dir runs/ten_tasks
```

第一次命令安装插件及 tokenizer 依赖。后两项使用固定假向量和脚本化语义回调，不发出 embedding、聊天模型或 PPT 生成请求。tokenizer 的首次加载可能需要缓存其公开编码文件。

### 定向回归：5/5

- 忘记尚未晋升的工作偏好。
- 忘记尚未晋升的完整修订链。
- 长期偏好修订后，忘记原偏好及对应工作副本。
- 结构化反馈兼容分析器附加字段。
- 显式优先级参数优先于内联字段。

### 连续文字轨迹：10/10 任务，70/70 检查

同一 SQLite 库连续处理十个项目，每题三页文字、两轮。覆盖跨项目保留、领域及受众范围、临时要求、改口、拒绝项目、跨实例恢复、优先级降低、忘记及重新学习。最终有七条活跃长期偏好。

记录：[逐项报告](validation/ten_tasks_report.md)、[完整文字过程](validation/ten_tasks_transcript.txt)。脚本支持用新输出目录重跑，不读取既有用户记忆。

## 真实模型接入记录

接入方已记录兼容接口的真实 Qwen3-Embedding-8B 与 GLM-5.3-Flash 验证：持久偏好晋升并在新项目取回，临时页面要求未晋升。本仓库不含 API key 或私密端点，离线回归不依赖这些服务。

以上验证针对记忆流程和接口。文字轨迹的页面执行与接受由模拟宿主提供，未测量真实 PPT 页面排版质量。
