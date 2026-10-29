# 推理引擎概览

本页说明 `engine/` 目录的组成、一次请求的执行路径，以及当前的结构性限制，是进入
[推理系统整体架构](inference-design.md)之前的地图。运行方法见[推理引擎指南](../user/inference.md)，
缩写见[术语表](../glossary.md)。

HYLE的产品目标是完整推理引擎，当前开发重点是它的底层：计算、执行、状态、缓存和请求能力。
Qwen3.5是真实模型驱动，用来检验这些能力能否串起来；通用模型加载、tokenizer与服务器API仍未完成。
当前正在收敛共同执行架构，再集中处理性能；具体正在做的任务只在[推理进度](../developer/inference-progress.md)维护。

## 分层

引擎建立在 Compute 和 Graph 之上，自身分两层：模型无关的共同机制，以及具体模型的实现。

```text
engine/main.cc                 CLI 入口，转到 qwen35::run
engine/models/qwen35/          Qwen3.5 模型：权重、几何、构图、状态、Model/Session、请求循环适配
engine/inference/              模型无关的共同机制：请求、缓存、接受规则、后端装配、状态搬运
engine/reporting/              执行报告输出
src/compute*, src/graph*       算子、设备、资源寿命、图编译与执行，不解释模型文件和生成策略
```

依赖方向是单向的：`src/` 与 `include/` 中没有文件引用 `engine/`。引擎可以在源码树中构建，
也可以作为独立工程链接已安装的 HYLE SDK。

## 共同机制（`engine/inference/`）

| 机制 | 文件 | 职责 |
|---|---|---|
| 请求循环 | `request_loop.h` | 单 owner 的请求身份、FIFO 接纳、轮次推进、暂停与唤醒、取消；模型通过静态绑定的适配器提供容量、恢复、执行和退休 |
| 请求窗口与进度 | `request_window.h`、`request_progress.h` | 有界请求槽与世代、每轮输入和输出额度、已提交输出的一次观察 |
| 输出交付 | `output_delivery.h` | 输出 credit／ACK 游标，实现背压 |
| 停止条件 | `request_stop_policy.h` | 额外停止 token、忽略模型 EOS |
| 前缀目录 | `prefix_directory.h`、`session_cache.h` | token radix 树、LRU、pin；载荷是模型自己的完整检查点 |
| 版本接纳 | `version_admission.h` | 按真实驻留版本和预算决定能否接纳新会话头 |
| 合批 | `batch_indices.h`、`batch_dispatch.h` | 在一波请求里按兼容条件分组，选出可一起执行的成员 |
| MTP 接受 | `mtp_acceptance.h` | 给定验证结果、额度、EOS 和上下文，计算合法接受前缀；Qwen、GLM、DeepSeek 共用 |
| 采样 | `sampling_graph.h`、`sampling_cursor.h` | 温度、top-k、top-p 的图构造；Philox 随机地址 |
| 后端装配 | `backend.h` | 冷选择 CPU／CUDA／Apple／Vulkan 端点及可选 CPU 辅助端点，给 Graph 提供存储域和队列 |
| 读回与输出 | `readback.h`、`prepared_output.h` | 按真实 dtype 和存储取回结果 |
| 状态搬运 | `state_transfer.h`、`host_state_storage.h`、`host_storage.h` | 会话完整状态在设备与主机内存之间换出、换回，等待真实退休后才交还写权限 |
| 状态写入块复用 | `reusable_output_storage.h` | 回收已退休的旧状态写入块，减少每步分配 |
| 失败与退休 | `evaluation_failure.h`、`retirement_observation.h` | 定位单个成员的失败；区分"已退休"与"观察到的状态" |

共同机制不解释模型的数学结构或状态布局。它们可以拥有执行端点、存储和保活句柄：例如
`Backend` 持有 Context／Queue，状态搬运通过原 Compute／Graph 接口提交复制并跟踪退休。
模型适配层负责说明完整状态由哪些部分组成；共同机制负责请求、预算、交付和寿命，
不另建一套设备执行器。

## Qwen3.5 模型实现（`engine/models/qwen35/`）

| 部分 | 主要文件 | 内容 |
|---|---|---|
| 输入准备 | `prepare.py`、`gguf_roles.py`、`nf4_state.py` | Python：核对本地模型文件，生成权重偏移清单与 token ID |
| 权重角色与格式 | `prepared_roles.h`、`prepared_gguf_roles.h`、`projection_binding.h`、`fp8_weight_import.h`、`nf4_weight_import.h` | 物理角色、量化编码、投影绑定；不在加载时展开整库 |
| 冷加载 | `weights.h`、`weights.cc` | 权重存储与冷导入 |
| 几何与精度 | `model_geometry.h`、`precision_profile.h`、`cache_precision.h` | 层数、维度、注意力层位置，源格式决定的计算精度，KV 缓存编码 |
| 放置 | `ffn_placement.h` | `--cpu-ffn-layers` 决定哪些 FFN 权重留在主机 |
| 构图 | `main.cc` 中的 `Recipe`、`GraphAssembly`，`mtp_graph.h` | 主干、MTP 草稿头、采样图 |
| 状态 | `session_state.h` | 分页 KV（每页 16 行）、conv／GDN 状态池、快照与容量规划 |
| 缓存与搬运适配 | `prefix_cache.h`、`host_image.h`、`session_transfer.h` | 把共同机制落到 Qwen 的状态布局上 |
| Model／Session | `main.cc` | 已加载模型、会话、准备／提交／完成／发布的执行事务 |
| 请求适配 | `main.cc` 中的 `GenerationRequestAdapter` | 把 Qwen 的 Model 接到共同 `RequestLoop` |
| CLI | `main.cc` 中的 `qwen35::run` | 参数解析、日程文件、报告 |

`main.cc` 仍集中包含构图、`Model` 事务、请求适配和 CLI。其余诊断程序（`*_probe.cc`）通过
`#include "main.cc"` 复用同一实现，只在显式目标中构建。

## 一次 decode 步的路径

```text
RequestLoop 选出就绪请求 → 按阶段、行数、采样配置分组（batch_dispatch）
  → Model::prepare_evaluations：校验、写入 token/位置、页布局、状态绑定、replay 准备
  → submit：提交到原 Queue／原生执行源，必要时重放 CUDA Graph
  → 观察 Completion 并确认原执行已退休，成功后读回选中的 token
  → 发布：新状态版本、pending、RNG 游标、已消费输入一起原子提交
  → RequestProgress 观察新输出，OutputDelivery 交付，决定下一轮或结束
```

MTP 有不同执行形状。单草稿融合路径把提议与验证接在同一次图执行内；递归多草稿路径先产生
草稿，再执行 target 验证。`mtp_acceptance.h` 根据验证结果计算合法接受前缀，模型适配层随后
选择完整续算状态，并按 draft 的 frontier／lag 决定是否需要 catchup。不能把所有 MTP 路径
理解为固定的“提议、验证、修复”三次提交。影响本次状态提交的可失败检查在发布之前完成，
发布本身不抛异常；发布后的报告或诊断失败不能倒写已经提交的状态。

## 当前的结构性限制

- **只有一个完整模型家族。** GLM、DeepSeek、Kimi 等目前是 `tests/compute_models/` 里的组件消费者，
  用来检验共同机制，还没有接成引擎入口。新增家族要在 `engine/models/` 下实现自己的 Model 适配。
- **完整执行编排仍集中在Qwen。** 共同请求、缓存和资源原语已提取，但形状、执行槽及部分事务编排仍在
  `qwen35::Model`；下一步要把能共用的机制移出，而不是让新模型复制它们。
- **流水只覆盖部分配置。** 当前冷资源有3个物理IO载体，支持的CUDA普通／greedy K1 MTP路径默认auto，
  以至多两步候选重叠主机准备和设备执行；其它配置保留单步路径。物理载体数量、在途候选数与批内成员数是不同概念。
- **合批有兼容条件。** 有效长度与varlen能力已实现，CPU／CUDA已有相关有限检查；请求仍按阶段、布局和采样政策等分组，
  不是把任意prefill与decode请求混在一张图里。
- **新形状仍有冷准备成本。** 接纳前准备有界形状族；已知工作可以预热，尚未准备的新形状仍可能需要编译。
- **MTP功能与流水覆盖分别推进。** 成员可有独立采样、停止及动态logits输入；这不表示所有组合都已有设备后继或相同速度。
- **输入需要预处理。** 没有通用模型目录加载和 C++ tokenizer，输入由 `prepare.py` 生成。

这些限制和对应的性能影响在[推理进度](../developer/inference-progress.md)与
[decode 性能资料](../developer/performance/decode.md)中跟踪。

## 相关文档

- [推理系统整体架构](inference-design.md)：模型、会话、请求、批次的职责划分，以及提交、缓存、取消的合同。
- [Compute](compute-design.md)、[Graph](graph-design.md)、[原生路线](compute-routes.md)：引擎下面的执行层。
- [模型与算子覆盖](../implementation/model-operators.md)：各模型、量化制品的实际检查范围。
- [Qwen 模型运行说明](../../engine/models/qwen35/README.md)：CLI 参数、日程文件、报告字段与容量公式。
