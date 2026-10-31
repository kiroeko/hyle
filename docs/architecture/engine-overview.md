# 推理引擎概览

本页是`engine/`的源码地图。分工与执行合同见[整体架构](inference-design.md)，运行方法见[引擎指南](../user/inference.md)，
当前任务只看[进度看板](../developer/inference-progress.md)。

## 分层

```text
engine/main.cc              CLI转入具体模型
engine/models/              模型制品、数学、完整状态和静态适配
engine/inference/           请求、缓存、程序IO、步骤推进与资源交接的共同机制
engine/reporting/           报告格式
Graph → Compute／后端       唯一的图执行、设备资源、完成与退休机制
```

依赖单向向下。Compute／Graph不解释模型文件或生成政策，也可由独立C++消费者使用。
模型新增自己的数学与状态适配；共同机制不以Qwen的形状、MTP阶段或状态布局作为公共固定要求。

## 共同机制

| 要修改的职责 | 先看源码 |
|---|---|
| 请求身份、接纳与turn推进 | [request_loop.h](../../engine/inference/request_loop.h)、[request_window.h](../../engine/inference/request_window.h)、[request_progress.h](../../engine/inference/request_progress.h) |
| 输出、背压与停止 | [output_delivery.h](../../engine/inference/output_delivery.h)、[request_stop_policy.h](../../engine/inference/request_stop_policy.h) |
| 前缀缓存与真实版本接纳 | [prefix_directory.h](../../engine/inference/prefix_directory.h)、[session_cache.h](../../engine/inference/session_cache.h)、[version_admission.h](../../engine/inference/version_admission.h) |
| 冷形状、具体程序与物理槽 | [cold_shape_plan.h](../../engine/inference/cold_shape_plan.h)、[prepared_program_family.h](../../engine/inference/prepared_program_family.h)、[program_io.h](../../engine/inference/program_io.h)、[program_slots.h](../../engine/inference/program_slots.h) |
| 步骤推进、后继与失败 | [step_execution.h](../../engine/inference/step_execution.h)、[request_wave_execution.h](../../engine/inference/request_wave_execution.h)、[execution_episode.h](../../engine/inference/execution_episode.h) |
| 生成接受与采样 | [mtp_acceptance.h](../../engine/inference/mtp_acceptance.h)、[sampling_graph.h](../../engine/inference/sampling_graph.h)、[sampling_cursor.h](../../engine/inference/sampling_cursor.h) |
| 端点、输出与完整状态搬运 | [backend.h](../../engine/inference/backend.h)、[prepared_output.h](../../engine/inference/prepared_output.h)、[state_transfer.h](../../engine/inference/state_transfer.h)、[host_state_storage.h](../../engine/inference/host_state_storage.h) |

这些机制组合原Graph／Compute资源和原Invocation；它们不再建立一套设备执行器，也不取代模型完整状态的发布权威。

## 模型与冷准备

Qwen的模型私有实现位于`engine/models/qwen35/`：

| 部分 | 入口 |
|---|---|
| 本地输入与权重 | `prepare.py`、`prepared_roles.h`、`weights.h/.cc`及具体codec导入 |
| 几何、精度与状态容量 | `model_geometry.h`、`precision_profile.h`、`cache_precision.h`、`cache_plan.h` |
| 冷构图与主干数学 | [graph_builder.h/.cc](../../engine/models/qwen35/graph_builder.h)、[target_graph.h/.cc](../../engine/models/qwen35/target_graph.h)、`block_graph.h`、`mtp_graph.h` |
| Family、Phase与辅助程序 | [family_factory.h/.cc](../../engine/models/qwen35/family_factory.h)、[phase.h](../../engine/models/qwen35/phase.h)、[phase_preparation.cc](../../engine/models/qwen35/phase_preparation.cc)、`auxiliary_programs.h/.cc` |
| 完整状态与请求适配 | `session_state.h`、`host_image.h`、`session_transfer.h`、`generation_ledger.h`及`main.cc`中的Model／Session与静态适配 |

`main.cc`仍包含热Model、请求适配和CLI，不代表完整冷数学仍在那里。诊断消费者复用相同实现，按测试目标显式构建。
其它模型组件检验共同合同，不需要照搬Qwen目录内部的Family／Phase结构。

## 一次decode如何推进

```text
RequestLoop选就绪请求 → 按兼容条件分组 → 选择已准备程序
  → drive_steps控制prepare／submit／finish及后继去留
  → 原Completion真实完成 → 模型取完整结果并做必要修复
  → 发布状态／pending／RNG／提交见证 → 请求观察一次并交付
```

同步、普通连续及MTP目标步骤共用推进机制，模型保留实际数学、Source边界和发布顺序。
MTP可以是融合或递归执行，不必固定为三次提交。取消、背压及准备拒绝不能提前回收已接纳工作，
后继记录命中也不能跳过实际Queue、资源权限或完整绑定检查。

## 能力边界

- 完整文本入口仍以Qwen为主；GLM／Kimi／DeepSeek已有组件消费者，尚非第二完整引擎模型。
- 连续执行按实际后端、采样和状态资格选择，auto不适用时保留原单步路径；物理载体数、在途深度和批成员数分别表达。
- Backend当前以一个主端点及可选CPU辅助端点为主；Graph多端点表达不等于多GPU分片／跨Context等待或多机已接通。
- 通用loader、C++ tokenizer、server／应用API与图片产品入口尚未交付；现有CLI使用准备后的权重和token IDs。

各模型／制品实际范围见[模型覆盖](../implementation/model-operators.md)，性能结果见[decode资料](../developer/performance/decode.md)。
