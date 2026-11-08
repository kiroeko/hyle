# 推理引擎概览

本页是`engine/`的源码地图。分工与执行合同见[整体架构](inference-design.md)，运行方法见[引擎指南](../user/inference.md)，
当前任务以用户指令和[进度看板](../developer/inference-progress.md)为准。

## 推理内核与完整引擎

推理引擎负责把应用请求组织为完整的推理过程。内核之外的引擎部分负责模型制品的识别和读取、
输入输出与tokenizer、请求与会话管理、跨请求调度、资源接纳与淘汰政策、结果交付及应用入口。
HTTP服务可以作为引擎外围适配器；现有通用loader、C++ tokenizer和服务入口尚未交付。

推理引擎内核负责将已接纳的推理工作变为计算结果和新状态。它包括模型和设备的冷准备、模型计算、
prefill／decode及推测步骤、采样、模型状态与缓存操作、执行依赖、完成观察和真实退休。
模型特有的数学、量化布局、状态和合法融合属于内核实现，仍放在相应模型或后端内；
模型无关的Graph／Compute等基础库提供原有执行、资源和权限机制，也可以独立使用。
这里的“推理内核”是系统职责，不等于一个CUDA或Metal kernel，也不按现有目录机械划分。

边界按责任区分：制品解析属于引擎外围，权重绑定、布局及程序准备属于内核；
跨请求的接纳、分组和淘汰政策属于引擎外围，内核负责已接纳工作内部的执行依赖和实际资源；
外围提供采样参数和停止政策，内核执行采样并与模型状态一起提交RNG。
缓存政策与物理状态共享同一真实所有权和退休事实，不另建执行器或账簿。

当前入口仍是Qwen驱动，模型、会话和请求装配有交叉；本定义不表示代码已完成拆分，也不声明稳定的引擎API。
先整体设计通用引擎及其执行边界，再按实际实现和验证结果分步冻结；目标和迁移方向见[通用引擎设计](engine-design.md)。
冻结后的具体模型、kernel、融合和策略仍可演进；合同变更须说明影响并完成相关验证。

### 冻结判定

内核冻结以当前实际实现和经过验证的执行合同为准，至少满足：

1. **职责与依赖**：记录冷准备、模型执行、状态发布、错误和退休的所有者及调用边界。模型数学、
   完整状态和专用快路留在模型实现；共同执行机制不依赖具体模型，Graph／Compute不依赖引擎。
   外围不直接修改内核的在途事务或权限；报告与用户输入不反向成为模型数学的依赖。
2. **路径与门**：后端、模型及制品提供各自的事实和能力；默认策略、正确性约束与执行时资源校验各有明确归属。
   可提前确定的路径在冷期选择，解释不可用原因并保留真实回落；每步仍须检查的Source、权限与退休事实不能提前假定。
   不以某个已测模型的名字或某次跑分伪装共同层的正确性限制。
3. **扩展成本**：在M2之前通过代码和接口合同作理论审查，核对新模型的状态、数学、量化和执行方式不必套用Qwen内部布局，
   新后端可在自身实现和能力边界接入。不要求下载或运行其它模型，也不为尚未交付的组合预建基类或抹掉专用优化；
   能力尚未实现时明确拒绝或走已验证的路径。理论审查不能证明新模型或后端已受支持。
4. **正确性与生命周期**：对实际消费者验证普通与MTP、贪心与已支持的采样，以及相关配置的完整输出或数值质量、
   RNG、状态版本、有序发布、取消／失败和真实退休。使用真实设备或真实模型检查时分别记录其覆盖，不能用合成检查代签。
5. **构建与性能**：标准构建及相关测试须实际通过，基础库仍可独立构建。以同窗交错、每臂至少三个独立进程及同程序A/A
   对照默认完整推理进程，并分别看prefill／decode；确认改动没有超出实测噪声的退化，不把局部kernel计时当成吞吐结论。

记录实际验证的系统、设备、模型、制品和推理方式；未验证组合不视为支持承诺。冻结合同及其变更纪律，
不冻结某种模型内部文件布局、单个GPU kernel、融合或优化策略。

内核边界、范围和已知缺口完成验证后才能标为冻结；这不表示未实测的模型或设备已兼容，
也不将源码内的临时适配声明为稳定的公开API。引擎整体冻结同样记录实际交付范围、接口、
生命周期、正确性与性能；两个层次都完成后再进入M2。

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
| 完整状态与请求适配 | `session_state.h`、`host_image.h`、`session_transfer.h`、`generation_ledger.h`、[model_core.h](../../engine/models/qwen35/model_core.h)中的Model／Session，以及`main.cc`中的请求适配 |

`model_core.h`是当前模型内部入口，不是已冻结的公共引擎API。`main.cc`仍包含请求适配和CLI；诊断消费者复用相同实现，按测试目标显式构建。
理论审查共同合同是否允许其它模型使用自己的状态和执行结构，不要求照搬Qwen目录内部的Family／Phase结构。

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

当前Qwen入口的`run_generation_sessions`装配请求和冷计划，`GenerationRequestAdapter`连接共同请求循环与模型步骤；
`Model`准备、提交和完成原始工作，`Session`持有已提交的完整状态头，`WorkingState`执行状态交换。
设备草稿交给目标验证器时，由`Model`准备原验证步骤、绑定实际输入并观察所属资源退休；
请求适配器仍负责请求身份和账本，不直接写入该验证步骤的输入字段。
预填和解码后继的原Phase、物理槽及捕获候选由`Model`检查；请求适配器决定请求是否继续。
捕获候选只说明已有录制可能覆盖该输入，实际Source、权限与完整绑定仍由原Graph路径验证。
提交记录通过原模型的同步发布见证与状态头衔接；失败、取消和后继都沿原执行及退休路径处理。
设备队列、Source、Completion与读写权限仍由原Graph／Compute机制负责。

`Model`和`Session`已可由不包含CLI或报告实现的独立消费者使用，部分原有测试仍直接包含`main.cc`。
`Model`仍含部分诊断职责，执行请求也携带用于请求错误归因的身份；
`GenerationStepAdapter`还直接读取模型在途步骤的Phase、Source和完成状态，并轮询原执行。
它目前是Qwen专用适配器，不能直接当成模型无关的外围接口；这些访问的所有权和生命周期仍须审查，
不能仅因原模型方法负责提交和退休就认定边界已经冻结。
权重清单只从模型读取事实，JSON格式与路由原因由外围报告保有。

## 能力边界

- 完整文本入口仍以Qwen为主；GLM／Kimi／DeepSeek已有组件消费者，尚非第二完整引擎模型。
- 连续执行按实际后端、采样和状态资格选择，auto不适用时保留原单步路径；物理载体数、在途深度和批成员数分别表达。
- Backend当前以一个主端点及可选CPU辅助端点为主；Graph多端点表达不等于多GPU分片／跨Context等待或多机已接通。
- 通用loader、C++ tokenizer、server／应用API与图片产品入口尚未交付；现有CLI使用准备后的权重和token IDs。

各模型／制品实际范围见[模型覆盖](../implementation/model-operators.md)，性能结果见[decode资料](../developer/performance/decode.md)。
