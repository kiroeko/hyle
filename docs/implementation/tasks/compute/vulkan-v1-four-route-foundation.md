# R6 Vulkan：原生计算程序、资源与 Graph

## Closure card / CANDIDATE

- owner：`compute_route_vulkan`、`graph_route_vulkan`；身份已在 `9059ad30` 登记为 QUEUED。
- baseline：`9059ad30`，保留 R5 不可变候选 `1636365b` 及其仍在运行的串行验收。
- trigger：[四路线合同](four-route-foundation-refactor.md)的 R6 已获实施授权，当前无 Vulkan 产品路线。
- 三个有限问题已按下列源码事实裁决，V1–V7 固定为 AUDITED FIX。新路线的独立实现不改变 R5
  候选或抢占它的重型验收；公开产品 roots 的下一次改动仍须另绑定当时的交付 owner。
  登记身份和本卡不计为已实现路线，也没有创建空 target。
- 主纵切：常驻矩阵权重、变化 activation、门控 FFN 的 Direct 与 Graph；借用已登记 ggml Vulkan
  的成熟矩阵/门控实现。具体数值和物理格式在审计中按真实 shader 合同固定，不把其它路线的四位格式套入。
- 正交纵切：数值/资源管线中成熟计算与 HYLE 自定义 SPIR-V 片段组合，真实 upload/compute/download、
  ready native buffer handoff、两个运行槽、保留旧结果和关闭回收。
- 依赖方向：Compute 的操作/shape/Completion/ResultSlot、Error/Mem/Platform；Graph 只冷读取
  DAG 并降成 Vulkan Program。Direct 无 Graph 依赖，不复用 CPU/CUDA/Apple 的执行器或 Queue。
- 范围：原生 instance/device/queue、device memory 与 staging、pipeline/descriptor/command
  program、运行槽、失败完成和回收；成熟计算接入、同设备扩展、固定默认规则、Graph、三交付与文档。
- non-goals：算子大全、模型/KV/batching/serving、跨 GPU/跨设备模型执行、CPU 模型 fallback、
  HIP、任意动态图、旧 API 兼容、未知硬件的 SOTA 声明。多 queue/family 仅在实际公开入口承诺时纳入。
- 目标验证域：Windows RTX 4070 原生 Vulkan；Apple Silicon 的 MoltenVK portability。WSL/Linux
  先核对实际 loader/device，区分编译交付、原生 GPU、转换层与软件 device；无设备时如实保留资格边界。
  AMD/Intel 无当前实机，不从 NVIDIA 外推；Mini 只作正确性，不冒充 Air 性能机。
- 调查预算：60 分钟，限下面三个具名问题；最多一个确有未决原生约束的短 admission 程序。
  已有两 pass 探针不重跑、不计产品。两次具名 shader/build 接入纠正；新基础缺陷定向绑定 owner。
- 性能预算：P0；API/算法/规则冻结后的 Direct、Graph 各一次 P1，原 5% 非劣门与方向分开，
  n=15，仅资格成立而区间未决时最多 n=31；最多一次有源码归因的机制纠正。禁止按 benchmark
  创建公共能力、改统计核心或用不同数值/完成责任签输赢。
- 验证：有限 FIX 的真实 native 正确性、独立 oracle、非法域/OOM/部分接纳/晚到错误/取消/关闭、
  并发槽与保活、适用 host 插桩及 Vulkan validation、native handoff、两系统三交付/relocation、
  OFF 无 SDK、public headers、默认规则与完整成本 P1；不故意 GPU hang/device-loss。
- stop：有限问题裁决后固定 AUDITED FIX，不继续全架构研究；具体必需原生能力/环境不能兑现时
  记录精确 blocker 与恢复条件。任何适用未运行项不签 PASS。R7 不首次接线本路线的验收。

## 有限审计问题

1. **成熟实现边界**：已登记 ggml `456172ec733a135778adcd32d00e576a58232e45` 的 public
   Vulkan API 没有 device/queue/buffer import/export；完整 runtime 每次 graph_compute 编码。
   比较受审 shader 子集与窄 runtime bridge 的完整生命周期/融合/维护成本，选择一个默认承载。
   Vulkan 的 `vk::SystemError` 不能与旧 ggml CPU abort blocker 混为一谈；CPU owner 继续 BLOCKED。
2. **资源和执行合同**：固定 native owner、外部借用方保活、recorded/managed、descriptor 更新、
   pending fence、memory visibility、non-coherent flush/invalidate、cancel/close。普通入口必须
   自己处理必要回收，不要求用户重复手写 glue。Graph 中间资源按实际 lifetime 复用。
3. **实际能力与交付**：核对 loader/driver/device、shader compiler、Vulkan-Headers/Volk/glslang
   pins、MoltenVK 与启用 feature；compiler 识别扩展不等于设备可用。固定实际可维护的 shader
   变体与合法基线，按设备/架构/driver/SDK/数值/shape 冷绑定，热态不查库或计时。

## 初始能力盘点（SCOPED 阶段）

| 能力族 | 层级与来源 | 当前状态 | 审计方向 |
|---|---|---|---|
| 原生 buffer/memory、pipeline、descriptor、commands、fence | Vulkan 原子机制；数值与资源管线 | MISSING | BORROW Vulkan；SELF 必要 RAII/错误与组合边界 |
| MatMul、门控完整片段 | 成熟复合计算；ggml/llama.cpp FFN | MISSING | BORROW 受审成熟实现，保留适用融合/硬件专用分支 |
| 自定义 SPIR-V 与 native handoff | 原生组合；资源流送/数值管线 | MISSING | SELF 窄冷准备与保活合同，复用同 device/queue |
| Program/Invocation/Completion/owning result | 复合生命周期；Direct 与 Graph | 共同结果语义已可用，路线 MISSING | ROUTED 既有 Completion/ResultSlot；SELF Vulkan 原生状态 |
| Graph 冷规划、资源复用与低成本重放 | 复合功能；固定 shape DAG | 路线 MISSING | SELF 冷 lowering；不建统一热执行器 |
| 模型格式、调度与跨 GPU 策略 | 应用策略 | OUT_OF_SCOPE | APPLICATION |

下节固定普通 C++ 调用、资源与失败边界、默认规则、有限 shader/格式域、交付 shape、
完整成本账单、代码/测试 owner 和有限 FIX。此前 SCOPED 阶段未写产品代码。

## 注册阶段验证状态（历史）

注册时所有 R6 产品门仍为 UNPROVEN。注册 QUEUED 身份的 selector 原始计划保存在
`.scratch/validation/vulkan-r6-identities-plan.json`；中央 manifest 触发的 broad build gates 未执行，
不计 PASS，也不为只有身份登记的改动另开一批重型构建。真实依赖/target 接入后按冻结候选的
实际影响面执行与复用证据。当前 docs/closure 与状态迁移检查通过；R5 的运行中验收保持独立。

## 有限审计检查点

[源码/编译器/环境记录](../../../../evidence/research/vulkan-r6-source-admission/README.md)已取得：
Windows RTX 4070 与 Air MoltenVK 的实际版本/能力；四个原样 F32 shader 变体编译成功。
GLSL 的扩展声明不等于最终 SPIR-V 必需能力，且部分 F32 输入的 cooperative-matrix 变体内部转为
FP16，须分开数值合同。两环境尚缺已枚举的 Khronos validation layer；Mini/WSL 原生条件继续按
本卡定向准备。这些不计产品完成；下列裁决冻结之后才进入实现。

## 审计裁决与 API / lifecycle freeze

**承载**：采用 ggml 的受审 shader 子集和 HYLE 原生 Vulkan 程序。完整 ggml runtime 的公共接口
缺少所需的 buffer/device handoff，且每次重新编码；本轮不维护穿透其私有 owner/pool 的桥。
矩阵/门控算法 BORROW，HYLE 只实现 route 的资源、录制、完成与 Graph 组合。初始 typed 域为二维
F32 MatMul 和 F32 SiLU 门控；完整 F32 FFN 是真实消费形状，既不冒充量化 carrier，也不改共享
Operation schema 来容纳 benchmark。其它格式/算子明确拒绝；未来受审 SPIR-V 可沿同一 native seam 接入。

ggml `mul_mm.comp`、`mul_mat_vec.comp` 与 `swiglu.comp` 保留原样来源与许可证；按实际 artifact
能力选择 shared-memory 或 subgroup 变体。默认不隐式把 F32 降成 FP16，也不把 cooperative matrix
输入为 F32 就当作同数值。需要 RHS 物理转置时，成熟 copy/stride 片段承担转换：不可变常量可冷
packing 并共享，变化 RHS 在程序中计入真实转换、scratch 和 barrier。不能把 ggml `[N,K]` 权重
布局直接解释成普通 MatMul 的 `[K,N]`，或让普通调用者为匹配 benchmark 改数据语义。

**公开形状**：普通路线头为 `compute/vulkan.h`，Graph 头为 `graph/vulkan.h`；SDK 类型只在 native
路线头出现。Context 冷创建/记录 Environment，Tensor 是带描述的原生 buffer owner，Program 是
不可变原生制品，Invocation 是独立 command/descriptor/fence/scratch/结果槽。普通 prepare 接受
typed Operation 和输入规格/常量，bind 接受 ready 同设备 Tensor；submit 返回已有 OwnedResult。
GraphBuilder → `graph::vulkan::compile` → 相同 Program/Invocation；用户不提供 Provider 顺序。

Direct 单算子不依赖 Graph。完整 native 数值/资源管线还有冷 Recorder：组合已经准备的 typed
计算和显式自定义 SPIR-V，持有其真实 buffer/pipeline owner，记录成一个原生程序。它服务具名
native handoff 消费，不创造第二套通用 DAG/搜索器。专家固定 caller buffers 与普通每槽生成输出
分别声明 ownership；共用原生提交/完成机制，不能把 caller destination 伪装成不可覆盖的生成结果。

**资源与并发**：一个 Context 选择一个真实 compute queue/family；queue host 访问由对象局部互斥
保护。每槽 command pool/descriptor pool/fence 独立，pending 或旧生成结果 lease 存在时不覆盖。
代码/常量共享，可变 scratch 分槽。Graph 冷期使用拓扑/liveness 复用已结束的中间空间；未证明
in-place 安全时，不把当前 node 仍读取的输入空间分给该 node 的输出。

Device-local 是正常计算 backing；上传/下载使用真实 staging，按 memory type 做 flush/invalidate。
`mem::QuotaAllocator::Reservation` 可计入实际 VkDeviceMemory 分配字节，driver/pipeline 私有内存
另列，不能宣称整个 GPU 的硬预算。复用 Platform DynLib 和现有 Error、Completion/ResultSlot、
Mem；不复制 OS loader、quota 或基础等待机制。

已锁定 Volk 虽有 instance/device tables，其 `volkLoadInstanceTable` 仍写全局
`vkGetDeviceProcAddr`。本轮需要独立 Context/native interop，选择通过 Platform DynLib 取得
`vkGetInstanceProcAddr`，冷保存必要的 instance/device PFN；不引入会和外部 caller 竞争的全局
Volk dispatch 状态。PFN 是 Vulkan 原生调用入口，不是逐 kernel Provider 裁决。

Native handoff 限同 VkDevice、同 queue family、已完成且 memory visibility 已交接的 buffer；
借入方必须交付真实 owner token。公开 native queue 访问须沿 Context 的同步边界；未完成外部工作、
不同 family 的 ownership transfer、跨设备/跨 API 内存桥不偷偷 host wait 或自动复制。没有确证的
桥明确拒绝。普通结果保有原生 buffer，显式 download 的同步/搬运成本可发现；结果 owner 使 native
device/loader 存活，Context 关闭后不再接受计算。

**完成与失败**：每槽 fence 在接纳前创建。热提交不创建 pipeline/command/descriptor 或重新查询
device。成功 submit 才进入 pending；取消不会 reset/free pending 资源。poll/wait 检查实际 fence，
失败不发布可读结果。Completion 析构不隐藏 wait：Context 保有 abandoned 工作，后续 collect/提交
或 close 推进回收；没有为 Vulkan 强加 MLX 固定 owner 线程。close 停止计算接纳并排空本 Context，
结果/buffer 的 native owners 可继续存活。

按 [vkQueueSubmit](https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueSubmit.html)，
提交 OOM 不改变被引用资源/同步状态；不能对未接纳、未 signal 的 fence 无条件等待。
按 [lost-device 合同](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-lost-device)，
设备丢失后的数据不可信，子对象仍须显式清理。其它观察错误不能冒充 quiescence：保留 pending owner，
经 fence/idle 确认真实终态后才交付原始错误和释放，禁止异常路径提前 reset/destroy。

**成本**：冷期做 shader/数值/limits 匹配、必要 packing、pipeline、内存计划和命令录制。重复执行
只做已声明的绑定更新、一次 queue submit、fence 查询/等待及结果寿命管理；每个 dispatch 的 driver
编码不留在热态。变化 payload 的 upload/download 单列或在完整旅程内计费，不能从两侧任选一侧
移出。常量转换、staging、同 GPU 片段边界与 shader-write/read barrier 计入完整方案。

## 审计冻结的有限 FIX

| ID | 实现与 owner | 必须验收 |
|---|---|---|
| V1 | Compute Vulkan：真实可选 target、锁定 Vulkan-Headers 与窄 ggml shader 来源、host shader 编译及嵌入资源；OFF 不探测 | 原始来源/许可证、online/offline、无 CPU/CUDA/MLX 引入，source/static/shared 的真实普通入口 |
| V2 | Compute Vulkan：native Context/Tensor、staging、队列同步、每槽状态、native handoff、Completion/结果保活与 close | 两槽/保留结果/并发、非一致内存、quota/OOM、部分初始化/提交/观察失败、取消/abandon/排空 |
| V3 | Compute Vulkan：二维 F32 MatMul、SwiGLU、所需私有格式转换与成熟 shader 规则；HYLE 自定义 SPIR-V 扩展 | 独立 oracle、尾部/非方阵/变化 RHS、冷常量 packing、无隐式 F16、同设备真实组合 |
| V4 | Compute Vulkan：普通 typed prepare 与冷 native Recorder | Direct 无 Graph 依赖、无需排序/profile、固定命令重用；专家资源合同可见 |
| V5 | Graph Vulkan：typed DAG 冷 lowering、常量共享、合法性/固定 shape、liveness 与资源复用 | 完整 FFN、重复执行无 DAG 遍历、custom schema、非法域/效果显式拒绝、无过早 alias |
| V6 | 两 owner：有限原生验证、插桩/validation、同合同成本资格和冻结候选 P1 | RTX 4070 + MoltenVK 实算；三交付、普通默认规则、正确性先于采样；未测硬件不外推 |
| V7 | 两 owner及单独交付 owner：公开 roots、source/static/shared 样例、用户指南和七门收口 | native 资源/源码/许可证可迁移，CUDA/Apple 既有候选不重测；R7 只复核 |

V1–V7 不授权公共治理/统计核心修改。API/算法尚在实现阶段，P1 不启动；规则最后以相同数值、
资源和完成合同下最强适用成熟完整片段及 same-kernel direct 成本为基线。ggml 的不同精度自然路线
另报；无完整收益证据时保留合格成熟实现，不生造赢家。AMD/Intel、FP16/cooperative-matrix 等
未准入数值/设备域保持明确未资格化，不能将其编译/枚举结果补成 PASS。

## 实现阶段接线记录（未验收）

原生 Context/Tensor、typed Program/Invocation、Recorder、完成/取消/回收和 Graph 冷 lowering
已经写入工作树，尚未签 GPU 正确性或路线完成。严格 MSVC 语法检查覆盖八个产品单元及新增的
原生正确性/生命周期测试；实际 GPU、validation、三交付和 P1 仍待执行。

普通运行槽的可重绑定 descriptor 使用设备真实支持且显式启用的
`descriptorBindingStorageBufferUpdateAfterBind`，layout/pool 同时采用对应 flag。普通无此
feature 的 Context 不能创建需要更新绑定的 Invocation；固定 caller buffers 的 Recorder 保留
其原生路径。Windows RTX 4070 与 Air M4 的实际 feature/limit inventory 支持该前提。依据
[vkUpdateDescriptorSets](https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateDescriptorSets.html)，
这避免在更新普通 descriptor 后误复用已失效 command buffer；pending 状态仍禁止更新。

实际 selector 已正确确认两个产品 owner 为 AUDITED，但新 vendor 前缀和原生 ON 配置需要
[独立的有限策略登记](../build/vulkan-r6-validation-policy-v1.md)。只登记既有 JSON 字段，不修改
selector 执行引擎或性能仪器。上游两个 shader include 的末尾空行按原始 blob 保留；源码
身份优先于重新格式化 vendor，产品自身的 diff 空白检查单独执行。

### Windows 原生检查点（2026-09-17，仍 AUDITED）

[实际 GPU 记录](../../../../evidence/correctness/vulkan-r6-windows-native/README.md)：13 项 Compute、
3 项 Graph 及两个逐头 CTest 全过；Khronos 1.4.357 与 synchronization validation 已实际启用，
无 VUID/SYNC-HAZARD。完整 FFN、变化 RHS/转置、同设备 custom SPIR-V、保留旧结果、失败/取消/
关闭及两线程槽回放均有真实执行。测试内非一致内存分支不冒充物理非一致 memory-type 资格。

首轮明确暴露矩阵 push 中 `k_split` 应为归约长度 K，而非分片数量 1；另纠正版本生成变量和
MSVC 测试环境读取。按源码根因修复后一次完整复验通过，保留全部失败记录，未改原样 vendor、
数值门或性能预算。源码输入已由 controller 哈希固定；这仍是实现检查点，不是 P1 候选。

R5 已独立 CLOSED，后续重型验证可串行使用 Air。MoltenVK、host 插桩、公开 facade/三交付、
规则/P0/P1 与终审仍须完成；中央构建导致的 broad selected gates 尚未执行，不计 PASS。

公开 facade 和跨路线交付 owner 已另立[有限交付卡](../build/vulkan-r6-product-delivery.md)，
按当时十三个既有 roots 绑定，不借旧 Apple CLOSED 卡授权 Vulkan 变更。

### MoltenVK 与 host 插桩检查点

[Air 原始证据](../../../../evidence/correctness/vulkan-r6-air-native-host/README.md)：同一 16 项
真实 native GTest、两个逐头检查及 Khronos synchronization validation 通过。ASan/UBSan/TSan
分别实际编译并运行全部 16 项测试，各保留 12 个已插桩编译单元的命令和 executable hash；
只签 HYLE host，外部 loader/driver/GPU 未插桩。

唯一新增产品修正是 shader 生成脚本兼容系统 Python 3.9；Windows 八个生成文件逐字节相同，
没有改变 shader、数值门或 API。产品候选尚未冻结，未启动性能采样。普通 public roots 和
Direct/Graph FFN+custom 同源样例已接线，Windows installed static 首次真实计算通过；
relocation、其余交付形态与两系统完整闭包仍按交付卡验收，不由此签 R6 完成。

### 两系统三交付检查点

`60de4212` 的[实际交付证据](../../../../evidence/correctness/vulkan-r6-three-delivery/README.md)
已完成 Windows / Air source/static/shared 的普通 Direct/Graph FFN、自定义 SPIR-V、变化输入、
重复执行和独立 oracle；验证 SDK relocation/offline、Windows DLL 邻置以及 Air producer 隐藏。
这些工作分别在两系统串行完成，不重跑此前已通过的 native/host 插桩。

R6 当前未完成项为默认规则与完整成本的性能资格、中央构建所选门的执行/有效复用及候选终审。
本次只归档交付检查点，不改变算法/API、不启动 P1，也不把路线或十三交付 owner 写成 CLOSED。

### V3 / V6 性能前源码裁决

[性能合同](vulkan-r6-performance-contract.md)固定了独立 ggml strict-F32 shader 参考、各入口
相同生命周期及单列的裸原生成本。初次预审把上游 host recipe 的 BK16 当成 F32 shader 的实际
归约分块，此判断已被下文 Air 实测纠正；同时接入不依赖隐藏 padding 的完整 tile vector-load
变体。shader 原样、公开 API、owner、运行状态与完成协议不变；未启动 P1。

同轮发现冷 packing 后仍持有不再引用的原始 dense 权重；V3 在全部 lowering/共享处理完成后
释放最终 dispatch 不使用的常量 owner。public owner/quota 与后续真实计算测试覆盖该修正，
仍被其它 dispatch 使用的常量继续保活。Windows 当前 14 Compute + 3 Graph 和逐头检查通过，
Khronos synchronization validation 无错误；完整 K4096/H11008/O4096、M1/M16 的 Direct/Graph
及原生成本参考全部通过两组输入的独立 double oracle。
原始命令、输入/制品身份和日志见 [F32 检查点](../../../../evidence/correctness/vulkan-r6-f32-preflight/README.md)。

两个入口、两个机器 profile 的性能合同与模块局部 A/A validator 已接线，复用公共机器合同和
nanobench parser；Windows 子进程证明 P 核 caller/mask，Air 证明 interactive QoS。资格矩阵仍为
两侧实现×两种 placement×执行正反序×分配正反序，共每入口/profile 16 个进程；未运行项不签 PASS。

Air `82c14c7e` 补验推翻了上段初次 P0 的 BK 判断：shader 的 `DATA_A_F32` 分支硬编码 BK32，
host recipe 的 ID3=16 不参与它的计算；按 16 估算共享内存导致 128 tile 超过 32 KiB。
本卡 V3 定向修正实际资源账单，保留原始 shader、合法完整 tile 分支与常量释放；不修改数值门
或性能预算。失败原始记录保留，正式 P1 继续等待修正后的两平台正确性。

## 不可变候选 `8e45b104`

V1–V5 的 API、算法/规则、资源与运行状态实现已固定在
`8e45b10423686a0753aaa9d461c4ef3b117412c5`。V6/V7 的验收继续执行，不能把候选登记当作路线完成。
[修正后的原始证据](../../../../evidence/correctness/vulkan-r6-f32-corrected/README.md)保存 Air 失败与
修正后的两平台结果：Windows / Air 各 17 项 native GTest、两个逐头检查和真实 synchronization
validation 通过；Air 四个完整 FFN 程序的两组输入/全输出独立 double oracle 通过。Windows 同
形状 oracle 复用前一检查点，修正只影响 128 tile 在 32 KiB 设备上的准入，不改变 M1/M16 shader
或 Windows 48 KiB 设备的选择。

下一步按四份 Direct/Graph × W1/M1 合同运行统一仪器：每入口先一次有限 A/A，通过后 n15，只有
资格成立而区间未决时最多 n31；原生边界成本单列为不可排名诊断。适用的最终交付/selector 门、
候选红队与全部性能仍未签 PASS。十三 facade 交付 owner 保持自己的活动卡，不恢复旧施工队列。

首个 W1 selector 已通过前置 docs/closure、configure 与两个原生 factor。P1 在构建阶段遇到
独立 peer 的 Windows 反斜杠被 CMake 的 `CONFIGURE_DEPENDS` 写入无效转义；此时没有 A/A 或
性能样本。只在 Vulkan benchmark 的 CMake 局部规范化该路径，保留失败日志；产品 `8e45b104`
不变，修正后的 benchmark commit 作为测量制品身份，原性能预算未消耗。

### Windows Direct P1 检查点

[实际统一仪器证据](../../../../evidence/performance/vulkan-r6-w1-direct-8e45b104/README.md)
已通过 A/A、公共 controls 与输入/制品/host 一致性。两个完整 FFN shape 的 16 项原 5% 非劣门
全部 PASS，n=15 停止；严格方向均为 INCONCLUSIVE，不声称 HYLE 算法胜出。测量快照
`3894b72e`，产品候选仍为 `8e45b104`。Windows Graph 与 Air 两入口、原生边界诊断、最终交付/
selector 和路线终审未由本检查点签证，两个 Vulkan owner 继续 CANDIDATE。

同一 `loc-report.py` 口径在 clean `87a2da74` 统计 Compute + Graph 全部自实现为 36,765 行、
129 个文件，不含测试/bench/vendor；包含尚未 CLOSED 的 Vulkan 实现，LOC 不等于完成度。

### Windows Graph P1 与剩余验收

[Windows Graph 原始证据](../../../../evidence/performance/vulkan-r6-w1-graph-8e45b104/README.md)
同样通过 A/A、公共控制项和身份检查；两个 shape 的 16 项原 5% 非劣门全过，n=15 停止。
严格方向全部 INCONCLUSIVE。Windows Direct/Graph 的正式 P1 均已完成，不因后续文档或状态
修改重测；原生服务边界的单列成本诊断仍待执行。

Air 首次 selector 在创建 Context 前发现 loader 路径未传入，尚未开始 A/A/P1。
实际进程探查证明 `/usr/bin/python3` 启动会丢弃该环境变量，直接使用它解析后的
`/Library/Developer/CommandLineTools/usr/bin/python3` 可保留；只修正任务启动器并保留失败，
不修改产品或公共仪器。Air 两入口性能、最终三交付/selector、适用 ASan 和终审仍待完成。

该启动器修正使 Air 两个 native factor 全部通过；进入 A/A 后，公共性能仪器按既有合同清理
`DYLD_LIBRARY_PATH`，因此仍在第一次 native Context 创建时拒绝启动，正式样本由 A/A guard
拒绝计时。保留这次完整日志；只在 Vulkan benchmark 的 Apple 冷配置中定位 loader，并通过
已有 `ContextOptions.loader_path` 为候选/参考使用同一路径。没有弱化环境清理、修改产品 API、
shader 或公共统计工具；新的 Apple benchmark 快照独立记身份，Windows `3894b72e` P1 不重跑。

两次启动失败的[完整保留记录](../../../../evidence/correctness/vulkan-r6-air-loader-setup/README.md)
已归档并校验；修正后的 Apple 测量快照为 `a41d43d5`。V1–V7 的
[候选源码复核](../../../../evidence/review/vulkan-r6-candidate/README.md)完成一遍，未发现新的产品
缺陷；已校正总入口文档。源码 review 不代替尚未完成的运行、交付和性能门。

[WSL 实际盘点](../../../../evidence/research/vulkan-r6-wsl-inventory/README.md)只枚举到 CPU 软件
device llvmpipe；不计原生 GPU 或路线验收。当前 Windows / Air 的验证域不外推到 Linux GPU、
AMD/Intel 或其它数值格式。

### Air Direct P1 检查点

[Air Direct 原始证据](../../../../evidence/performance/vulkan-r6-m1-direct-8e45b104/README.md)
在 `a41d43d5` 测量快照完成：16 个 A/A、120 个产品进程、30 个 control 进程，资格与全部身份
检查通过；M1/M16 共 16 项原 5% 非劣门全部 PASS，n=15 停止。M1 严格分层为 7 WIN / 1
INCONCLUSIVE，M16 为 4 WIN / 4 INCONCLUSIVE；两个 workload 整体方向均未决，不签算法胜出。
完整归档已在 Windows 复核逐项 hash。产品候选仍是 `8e45b104`；Air Graph 已串行接续，原生边界
诊断、最终交付/selector、ASan 及终审仍未由本检查点签证。两个 Vulkan owner 继续 CANDIDATE。

### 四份正式 P1 已完成

[Air Graph 原始证据](../../../../evidence/performance/vulkan-r6-m1-graph-8e45b104/README.md)
也完成 16 个 A/A、120 个产品及 30 个 control 进程；全部资格/身份检查及 16 项原 5% 非劣门
PASS，n=15 停止。M1 为 2 WIN / 6 INCONCLUSIVE，M16 为 4 LOSS / 4 INCONCLUSIVE；后者的
四个 peer-allocation-first 分层 median 延迟约高 1.2%–1.5%，原始负向和通过非劣门的事实分别保存。
两个 workload 的整体方向仍未决，不签算法胜出，不追加采样。

至此 Windows / Air × Direct / Graph 四份正式 P1 全部完成并通过；产品仍固定为 `8e45b104`。
原生边界成本诊断已开始按既定顺序串行执行；最终 ASan、两系统三交付、中央 selector 与终审仍待
完成，两个 Vulkan owner 继续 CANDIDATE。后续文档、状态或证据修改不触发这四份 P1 重跑。

### Windows 原生边界诊断完成

[Windows Direct/Graph 原始记录](../../../../evidence/performance/vulkan-r6-w1-native-boundary-8e45b104/README.md)
各完成四个顺序分层 × 15 个独立进程，oracle、machine/input/artifact 一致性与收集通过，未扩样。
按冻结合同保持 NATURAL_ROUTE / NOT_COMPARABLE；裸 submit/fence 缺少完整取消/quarantine/结果
lease 服务，不签非劣或排名。全部分层、原始日志和成本差额已归档；Air 相同有限诊断串行接续。
这项证据不改变产品候选、四份正式 P1 或其余未完成门，路线仍为 CANDIDATE。

### 全部边界诊断与最终 ASan 完成

[Air Direct/Graph 诊断](../../../../evidence/performance/vulkan-r6-m1-native-boundary-8e45b104/README.md)
各完成四层 × 15 个独立进程、两个完整 FFN shape，oracle/机器/制品与收集检查通过；至此两系统
四项自然成本诊断全部结束，保持 NATURAL_ROUTE / NOT_COMPARABLE，不增加 NI 或算法排名。
原始正负差额、顺序关系和参考责任差异均保留，不追加采样。

[最终 host ASan](../../../../evidence/correctness/vulkan-r6-final-asan/README.md)在 `a41d43d5`
验证快照实际覆盖 12 个插桩编译单元、14 Compute + 3 Graph GTest，全部通过。新增常量 owner
释放的寿命路径已补验；旧 UBSan/TSan 仅继承未变化的具体 host 路径，候选/数量与外部 GPU
未插桩限制按证据保留。两项归档在 Windows 重算全部条目 hash，通过。

产品仍为 `8e45b104`，两个路线 owner 继续 CANDIDATE；剩余最终 Windows/Air 三交付、中央
selector 和 terminal 汇总按既定串行任务执行，不能由上述结果提前签路线完成。

### 最终 source/static/shared 完成

[最终两系统交付](../../../../evidence/correctness/vulkan-r6-final-delivery/README.md)已通过；相同
完整源码包在 Windows / Air 各生成并实际运行六个普通消费者，覆盖 Direct/Graph 完整 FFN、
custom、变化输入与回放。SDK relocation、隐藏原 source/build、offline、许可证与导出、Windows
DLL 邻置均完成。详细 D1–D3 归属仍在独立交付卡，十三 owner 的旧实现/性能保持继承。

当前只剩已经启动的中央 broad selector 和 terminal 汇总；四份正式 P1、四项自然成本、最终
ASan 和两系统三交付均已执行完成。没有为收口增加采样或未知设备资格，两个 Vulkan owner
继续 CANDIDATE，等待最后适用门的实际 verdict。

### 最后 selector 的具名测试问题

原 Windows batch 已通过 84 门，在 Sched 的 251 项测试中遇到一个旧测试顺序错误，剩余 26 门
没有执行。[原始失败](../../../../evidence/correctness/sched-first-error-test-order/README.md)已完整
保存：typed `JoinHandle` 结果就绪早于 scope 完整清理/首错发布，旧测试没有建立后者的顺序。
[定向测试维护](../sched/sched-first-error-test-order.md)只改测试同步，不改变 Sched 或 R6 产品。
补验该用例后独立续跑尚未执行门；原 BLOCKED result 与新测试快照分别保留，不重跑 84 个已通过门
或任何 P1。路线仍为 CANDIDATE，直到覆盖证据完整。
