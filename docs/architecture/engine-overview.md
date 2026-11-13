# 推理引擎概览

本页是`engine/`的源码地图。分工与执行合同见[整体架构](inference-design.md)，运行方法见[引擎指南](../user/inference.md)，
当前任务以用户指令和[进度看板](../developer/inference-progress.md)为准。

## 推理内核与完整引擎

推理引擎负责把应用请求组织为完整的推理过程。内核之外的引擎部分负责模型制品的识别和读取、
输入输出与tokenizer、请求与会话管理、跨请求调度、资源接纳与淘汰政策、结果交付及应用入口。
HTTP服务作为引擎外围适配器。当前本地文本CLI和Responses文本子集使用Python tokenizer／模板与常驻模型进程，
支持冷期具名方案、进程内会话、可信前缀域与派生制品持久化。模型来源通过静态提供者分派，目前完整适配器是Qwen；
C++ tokenizer和完整服务API尚未交付。

推理引擎内核负责将已接纳的推理工作变为计算结果和新状态。它包括模型和设备的冷准备、模型计算、
prefill／decode及推测步骤、采样、模型状态与缓存操作、执行依赖、完成观察和真实退休。
模型特有的数学、量化布局、状态和合法融合属于内核实现，仍放在相应模型或后端内；
模型无关的Graph／Compute等基础库提供原有执行、资源和权限机制，也可以独立使用。
这里的“推理内核”是系统职责，不等于一个CUDA或Metal kernel，也不按现有目录机械划分。

边界按责任区分：制品解析属于引擎外围，权重绑定、布局及程序准备属于内核；
跨请求的接纳、分组和淘汰政策属于引擎外围，内核负责已接纳工作内部的执行依赖和实际资源；
外围提供采样参数和停止政策，内核执行采样并与模型状态一起提交RNG。
缓存政策与物理状态共享同一真实所有权和退休事实，不另建执行器或账簿。

token CLI与文本工作进程共用公共模型提供者、装载配置和请求入口；Qwen的有限日程、重放和报告位于显式诊断目标。
公共合同与模型内部结构分别维护，不将内部C++接口声明为稳定的安装后ABI。
本阶段的固定范围与验收见[通用引擎设计](engine-design.md)。
冻结后的具体模型、kernel、融合和策略仍可演进；合同变更须说明影响并完成相关验证。

### 冻结判定

本阶段R1–R4已关闭，职责与合同按[冻结范围和验证](engine-design.md#架构审查与冻结修复基线)收口。下列要求用于保持这些边界，不把全部未来模型、平台和产品功能自动加入本轮完成条件。

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
engine/main.cc              静态登记提供者，调用公共应用入口
engine/application.h        token CLI／worker的参数、装载与请求装配
engine/model_preparation.py  冷期制品准备器登记
engine/asset_store.py        不可变文件复用、校验和原子发布
engine/text/                tokenizer、模板和协议无关的文本适配
engine/text/app.py          本地文本CLI和服务装配
engine/text/server.py       HTTP路由、连接和流式传输
engine/text/responses.py    Responses文本协议适配
engine/text/configuration.py 冷期方案与请求选择
engine/text/sessions.py     响应历史、原生会话借用、TTL与回收
engine/models/              模型制品、数学、完整状态和静态适配
engine/inference/           请求、缓存、程序IO、步骤推进与资源交接的共同机制
engine/reporting/           报告格式
Graph → Compute／后端       唯一的图执行、设备资源、完成与退休机制
```

依赖单向向下。Compute／Graph不解释模型文件或生成政策，也可由独立C++消费者使用。
模型新增自己的数学与状态适配；共同机制不以Qwen的形状、MTP阶段或状态布局作为公共固定要求。
[源码边界检查](../../tests/tooling/inference_architecture_boundaries_test.py)覆盖共同层和后端的直接include、部分生成脚本的静态导入、文本层静态导入及请求策略中的后端宏；它不能代替构建依赖闭包、动态加载或运行时行为验证。
本机CUDA的Compute和Graph公开示例已分别作为独立C++程序链接现有共享库并运行，未链接引擎；这不代替两者的单组件构建或安装包验证。Windows上另以`HYLE_COMPONENTS=mem`独立构建并安装共享库，已安装SDK的C++示例通过；该验证不包含Compute或Graph。

## 共同机制

| 要修改的职责 | 先看源码 |
|---|---|
| 请求身份、接纳与turn推进 | [request_admission.h](../../engine/inference/request_admission.h)、[request_loop.h](../../engine/inference/request_loop.h)、[request_window.h](../../engine/inference/request_window.h)、[request_progress.h](../../engine/inference/request_progress.h) |
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
| 完整状态与请求适配 | `session_state.h`、`host_image.h`、`session_transfer.h`、`generation_ledger.h`、[model_core.h](../../engine/models/qwen35/model_core.h)中的Model／Session，以及`generation_state.h`／`generation_runtime.cc`中的请求适配 |

`model_core.h`是模型内部入口，不是公共引擎API。正常产品编译`generation_runtime.cc`，不链接Qwen的完整报告格式器；
冷路线归`mtp_default_policy.h`、`decode_route_policy.h`和`device_mtp_planning.h`。
`generation_diagnostics.h`／`generation_report.cc`消费完成观察；`main.cc`与`diagnostic_main.cc`组成显式`hyle_qwen35_diagnostic`入口。
部分原有私有探针继续包含这一诊断消费者。
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

Qwen的`execute_generation_sessions`消费公共批次与已选择的执行准备，冷程序／录制准备分别经`prepare_generation_programs`／`prepare_generation_recordings`；
`GenerationRequestAdapter`连接共同请求循环与模型步骤。诊断包装器`run_generation_sessions`单独接入完成观察，不决定状态发布；
`Model`准备、提交和完成原始工作，`Session`持有已提交的完整状态头，`WorkingState`执行状态交换。
设备草稿交给目标验证器时，由`Model`准备原验证步骤、绑定实际输入并观察所属资源退休；
请求适配器仍负责请求身份和账本，不直接写入该验证步骤的输入字段。
预填和解码后继的原Phase、物理槽及捕获候选由`Model`检查；请求适配器决定请求是否继续。
捕获候选只说明已有录制可能覆盖该输入，实际Source、权限与完整绑定仍由原Graph路径验证。
提交记录通过原模型的同步发布见证与状态头衔接；失败、取消和后继都沿原执行及退休路径处理。
设备队列、Source、Completion与读写权限仍由原Graph／Compute机制负责。

`Model`和`Session`已可由不包含CLI或报告实现的独立消费者使用，部分原有测试仍直接包含`main.cc`。
`Model`仍含部分诊断职责，执行请求也携带用于请求错误归因的身份。
`GenerationStepAdapter`是Qwen内部适配器，直接读取模型持有的在途步骤、Phase和完成状态；
其原始步骤指针只在当前驱动调用内借用，跨调用保留的是请求描述和游标，恢复时重新核对模型的原步骤。
Source的共享引用只用于保活，不取代原Graph的权限与退休；失败路径交回模型退休原工作，未退休时保留原所有者。
这些内部耦合不属于公共模型合同。本机 CUDA 和 Qwen3.5-2B BF16 上，已检查提交后 Source 元数据捕获失败、
外部所有者释放后的退休和同会话重试；双请求接力还检查了背压及已提交后继被丢弃后存活请求的状态和退休。
这些检查没有注入原生 GPU 故障，不能代替完整失败注入和冻结验收。
权重清单只从模型读取事实，JSON格式与路由原因由外围报告保有。

## 能力边界

- 完整文本入口仍以Qwen为主；GLM／Kimi／DeepSeek已有组件消费者，尚非第二完整引擎模型。
- 连续执行按实际后端、采样和状态资格选择，auto不适用时保留原单步路径；物理载体数、在途深度和批成员数分别表达。
- Backend当前以一个主端点及可选CPU辅助端点为主；Graph多端点表达不等于多GPU分片／跨Context等待或多机已接通。
- token CLI使用准备后的权重和token IDs，与worker共用提供者装载；文本CLI与Responses文本服务使用本地tokenizer和模板。
  通用loader、C++ tokenizer、完整服务API和图片产品入口尚未交付；实际字段见[文本接口指南](../user/text-api.md)。

各模型／制品实际范围见[模型覆盖](../implementation/model-operators.md)，性能结果见[decode资料](../developer/performance/decode.md)。
