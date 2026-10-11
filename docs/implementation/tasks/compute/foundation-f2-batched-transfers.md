# F2：准备后的原生批量范围搬运

> 状态：AUDITED（2026-10-11）；F2 七 FIX 的形状/owner/有限实现已固定，开始集中接线。
> 实施前[原子/复合接缝源码审计](foundation-f2-source-audit.md)为定界输入，未授权产品修改或签运行资格。
> component / owner：拟 compute、compute_backend_cpu/cuda、
> compute_route_cpu/cuda/apple/vulkan；只有普通 Graph 需要本批 native binding 接缝时再绑定对应 owner。

## 共同执行合同

### 当前实际启动绑定

- 启动 baseline / 改前 registry ref：`7f53bf532b044803ea36f792196e446c66b58ac4`，Windows canonical
  工作树干净；F1 七 owner 已 CLOSED，产品 source 与其固定 `3034fc595` 相同，不回退后续证据。
- 具名 reopen trigger：本 goal 原 F2-01至07 的新增 prepared bulk ranges/explicit phases/ready
  public journey；[源码审计](foundation-f2-source-audit.md)已证明已有 single-copy/TransferSlot
  不能完整承载 bounded ranges、batch alias/producer/index lifetime 与一阶段真实完成。
- 当前确定绑定 `compute`、`compute_backend_cuda`、`compute_route_apple`、`compute_route_vulkan`，
  仅从 CLOSED 进入 SCOPED。CPU 原 byte copy/QueueSequence 先 BORROW；没有实际改动理由的
  `compute_backend_cpu`、`compute_route_cpu/cuda`、Graph 与全部 Provider/lowerer 保持 CLOSED。
  若审计发现确需新 public factory/graph seam，再以具体事实合法绑定 owner，不提前重开整闭包。
- 原七 FIX、主量化权重/状态与正交资源流送/分块矩阵、四路线双向/同设备 copy、二组二槽、
  原格式/BF16数学及三交付范围不缩减。未运行任何 F2 产品构建、数学/故障或 P1，不把只读审计当 PASS。
- SCOPED 首轮源码/形状预算45分钟：沿已固定 peer 的三段源码与实际 SDK/native owner 核对，
  冻结 physical identity/alias、端点/phase/group C++形状、F1 guard/producer/prefix 交接、错误/关闭、
  冷热成本、public/内部边与普通 sample。原一次反例补读、两次有限修复/验收预算不增加或重置。
  未完成 AUDITED/有限实现 ledger 前不写产品。新全库边界或公共语义矛盾立即定界/裁决，不无限研究。

- 基线：`edb2714384037b12a3d1e69839aa895fea17cec1`；设计见
  [资源与流送设计](../../../architecture/compute-resource-streaming-design.md)。
- 本卡已作为上述四 owner 的当前 AUDITED record；其他 owner 按 registry。共同控制首轮见下表；
  原生程序、数学与完整交付验证尚未运行。
  以下 AUDIT/有限实现 ledger 固定后才改产品，后续新公共语义须重新裁决。
- 主纵切：量化推理的外部状态/权重范围及真实成熟数学消费。
  正交纵切：渲染资产流送和分块矩阵管线；不加入 token/layer/KV policy。
- 复用 Mem/Util/Sched/IO/Compute/native 完成，不增加复制 kernel、数学算子、模型 loader 或第二套 runtime。
- 全部产品接线集中后冻结候选，批量执行适用检查；小修只运行影响门，同身份证据复用。
- 四路线资源/程序/完成独立。Windows 为唯一产品真相；CPU Win/WSL/Air，CUDA Win/WSL，
  Apple Air，Vulkan Win/Air；WSL Vulkan 按实际 GPU 资格，不以旧例外代签。
- 三交付：source/installed static/shared 普通 Direct+Graph 消费、SDK 迁址、ON/OFF 新头、路线缺席诊断、
  Windows DLL 同目录。不让普通用户添加内部 factor/provider/stream 开关。
- 默认 P0：有界资源、热分配/提交/完成计数与成本结构；没有 timed 性能赢家承诺。
  出现需性能裁决的具体反例才在稳定候选上按原统一仪器固定一次同合同有预算 P1，
  未资格/未运行如实记，不能无限换姿势或升级仪器。
- build1/test1、重型串行、启动前确认内存余量；复用源码缓存/SDK 构建树，完成后只清本批临时产物。
- 每轮 selected gates、`check-module-closure.py`、`check-docs.py`；
  registry 变化带 `--previous-ref`。文档/状态阶段不构建/P1。提交推送用 `[skip ci]`。
- 调研纠正预算：至多一个具体反例的一次补读、两次有限修复/验收；新公共语义或边界变化必须重新定界。
  所有 FIX/适用门完成则 terminal；不能把研究/文档/旧 CLOSED 计作本轮产品完成。

## 需求与完整能力

实际来源：vLLM cpu/gpu_worker 的 CopyPlan 143–185、transfer_async 534、
cache_kernels.cu 110–194；SGLang l2_transfer 的 producer/index/layout 依赖。
复制任务不是只接受“所有整 tensor 具有相同尺寸”：
host/device 粒度不同、部分首尾块、多个 backing、有限 index/range 更新和两个方向均需表达。

本批 MUST：CPU host copy，CUDA H2D/D2H/同设备 D2D，Apple 原资源 CPU/GPU 可见性与 blit，
Vulkan staging/device 双向及同设备合法 copy。无跨 GPU/peer/multi-GPU；不为 GPU 动态 IDs 新写 kernel。

## API / usability freeze

- 一次 prepare 固定 resource endpoints、capacity/metadata/groups/native version 与 copy 路线。
  每批绑定 host-known 源/目标范围或固定 catalogue IDs；不动态发现资源/Provider，不做时间比赛。
- source 可被多次读；destination 不能交叠；源写依赖不得在 independent batch 中隐式排序。
  exact self-copy/no-op 与空批处理清楚；相依搬运由显式 phase 与 completion 依赖表达。
- 保持原编码 bytes，description/物理容量/alias 分开校验；typed binding 不能用半编码块冒充 tensor。
  需要 native padding 时必须明确已分配范围，不能越过邻块。
- 一阶段一个真实 Completion，保留 F1 guards 和指针/index metadata；
  CPU 同步完成也遵循完整生命周期，不伪装 GPU 异步。native partial acceptance 不丢 prefix。
- CUDA batch 与 fallback 在冷准备按 SDK/driver/stream 能力固定。
  STREAM 是 producer 依赖默认；ANY 仅在严格已证 host source 条件采用，不按方向名称猜测。
  Metal/Vulkan 各用原 command/queue，原有 Vulkan family/可见性 owner 保留。
- 单组/分组可分别返回 ready；不保证物理重叠或全同 buffer pair 能一条 native command，
  但禁止每 copy 重建 plan/event/Completion。报告真实 native calls/groups/waits/staging。

## 有限 FIX 集

### 当前 AUDIT / 形状冻结

- AUDIT baseline / 改前 registry ref：`58bc5d72b`。沿原 seven FIX，实物 source/peer/SDK 证据见
  [审计记录](foundation-f2-source-audit.md)，没有扩大调查文件或更新 dependency pin。
- 共同字节描述/校验/有限 permission/control 属 `compute`；CPU/CUDA 只消费原 Context/QueueSequence，
  CUDA 的 batch ABI 冷解析与 admitted native enqueue 属 `compute_backend_cuda`。
  Apple/Vulkan 拥有自己的 native Program/Invocation/Submission，分别复用原 Runtime/Job 和 Run/Device。
  共同 detail helper 只校验、权限、实际退休/ready，不 dispatch native work，不造万能 executor。
- 不改现有 BackendVTable、Completion 公共布局或旧 ordinary submit 的常态字段；CUDA 借原 versioned
  BackendExtensionRef 进行冷资格及原序列录制。只为新 F2 dependency 入口识别原 owner 的 actual native
  identity；旧借用入口与 F1 normal API 语义保留。外部未知 callback/Completion 不猜 native fence。

#### 自然 C++ 入口与独立运行槽

共同值放在 `hyle/compute/batch_transfer.h`：`CopyResourceId/CopyPhaseId`、`CopyRange`、`CopyPhase`、
`CopyResourceSpec`、`CopyBatchOptions`、`CopyBatchReport`。范围值为 source/destination IDs、各自 byte offset、
byte count 与 phase ID；phase 有显式先行阶段 mask 与应用 group tag，不含 layer/token/model。
`CopyRange::bytes` 是 logical bytes，`physical_bytes=0` 表示 exact copy；显式非零 padding 必须不小于
logical、属于两个已拥有范围且参与全部 alias/预算校验。零 logical copy 不接受非零 padding。
phase 数最多64，dependency 必须指向先前阶段且覆盖真实跨阶段 hazards；同 phase 是 independent copies。

CPU/CUDA 共同 `compute::CopyBatchProgram/CopyBatchInvocation/CopyBatchSubmission` 的原执行入口是
Context/Queue；Apple/Vulkan 同名类型在其各自 namespace、native factory/context 下，资源/执行 owner 独立。
它们共享普通字节描述与验证合同，不通过 common Queue 执行 MLX/Vulkan。

```cpp
auto program = CopyBatchProgram::prepare(context, resource_specs, options).value();
auto first = CopyBatchInvocation::create(program, queue).value(); // native routes 用原 Context owner
auto second = CopyBatchInvocation::create(program, other_queue).value();
HYLE_RETURN_IF_ERROR(first.bind_source(source_id, ready_read));
HYLE_RETURN_IF_ERROR(first.bind_mutable(destination_id, std::move(destination_write)));
HYLE_RETURN_IF_ERROR(first.bind_producer(source_id, std::move(producer))); // 需要时显式交接
HYLE_RETURN_IF_ERROR(first.prepare_replay(phases, ranges));
auto submitted = first.submit().value();
auto ready = submitted.output(phase_id, destination_id, offset, bytes); // 未真成功时 Unavailable/error
HYLE_RETURN_IF_ERROR(submitted.wait());
```

所有 create/prepare/bind 返回 `Result/Status`；move-only invocation/submission，program 可共享不可变制品。
program 冷固定 resource domain/capacity、native ABI/可用性与 metadata 上限；每 invocation 冷预置资源绑定、
ranges/physical interval 索引、phase/producer/use/observation 槽。默认 resources16/copies64/phases4，
显式有限扩容只在 cold create；至少两 invocation 独立 backing/在途槽，不将两业务槽折成同一个 mutable owner。
source 绑定 ReadLease；mutable source/destination 接管 WriteLease，失败不消费。program/context/resource/allocator
保持原借用寿命；实际 leases、index/descriptor、移动进来的 producer Completion 自动保活到本次真实最后使用。

`prepare_replay(phases,ranges)` 在原固定资源内更新 host-known IDs/offsets/bytes，不重选 Provider/driver ABI。
catalogue IDs 等价于有限 caller range 索引，不支持 GPU-produced IDs。新增资源绑定为显式冷操作，须没有在途
native work/未放弃的 submission；旧 exported ReadLease 阻止对应块再次取得 writer。reset/close/drain/in_flight/
report 的正常与 empty/moved 行为明确，failed slot 不自动复用；close 停接纳，drain 等实际已接受 work。
每 invocation 一份 submission 冷槽；旧 submission 保留该槽，避免热扩建 callback/资源表。

#### Alias、phase 与逐组 ready

- 原资源 ID 不是 physical identity。CPU 用实际 host address；CUDA 用 context/address domain 与原 byte pointer；
  Metal 用实际 shared contents/native backing；Vulkan 用 Device/memory或已知 native backing/offset。
  native adapter 冷确定 canonical span，拒绝不能证明用途/alias 的外部 writable carrier，不以 wrapper 地址代签。
- 全批 checked addition/容量/usage/device、各 phase destination 不交叠、source→destination overlap 同批拒绝；
  重复 source 允许。exact self-copy 与零 byte ranges 在接纳前消除，空批有合法 inline completion。
  多 phase 的 read/write hazards 必须由显式 dependency closure 覆盖；不得靠提交数组顺序默默接受相依批内 copy。
- interval 索引冷预置，bind/更新做按 physical key/start 排序及线性 scan/union，O(R log R)；不做所有 range 两两比较。
  每 phase used-root/读写角色在 bind 去重，submit 不重复范围排序。F1 现有 bounded dedup 的 O(N²) 成本不能藏进
  O(R+G)：新 F2 使用已证明唯一 root 的内部准入，保留 permission/lock/ref 检查，O(N+R+G)（N为使用资源数）。
- 每 phase 一个真实 native Completion。所有 native descriptors/可能引用的 data 保留到其原退休 callback；
  partial acceptance 保存 prefix 和首错；whole submission 与 per-phase ready 分开，已成功独立组保留有效值。
- mutable B 的 A→B、B→C 由 F2 内部、可证明 ordered read-use 交接；实际写 guard 退休/成功确认后，
  可以发布 B 的 ReadLease，仍在途的后继只读 uses 继续保活。普通 F1 public write/read/publish 入口不放宽。
  group 只发布相应 lease block 的最后写已实际退休的数据；同 block 未来还要写时不能提前发布，普通多组
  pipeline 使用独立块。失败/取消/unknown 不能发布；新 read 之后无后继 writer，旧 read 阻止复用。
  内部 fast admission 由 cold root identity 去重；旧 public准入的搜索路径不增加 ordered分支。
  WriteLease mutable view 在原 domain lock 下另查 root uniqueness，阻止新 ordered readonly native use 期间的
  mutable alias；这是 lease opt-in view 的一次原 refcount读取成本，不给普通借用 TensorView/Queue增加字段。
- producer 必须有合法 owner 与显式 native/inline 依赖。新 F2 resolve 只识别原 retained owner 并转交实际
  pending identity，known error/unknown 已移交退休仍是错误；没有可用 native route 时明确拒绝或显式报告已选
  host boundary，不悄悄同步。程序/完成 metadata 不进入数学 Provider selection。
  新内部 `RetainedCompletion::native_dependency` 只剥离已知 own proxy（有界32层），不 poll/wait、不把
  compound/custom vtable 冒充 CUDA event；新的 `QueueSequence::enqueue_resolved_wait` 保留原借用 producer
  身份/lifetime，known error在新接纳前拒绝。旧 enqueue_wait/Completion布局及旧 submit 路径不改。

#### Native 固定路线

CPU 沿原 byte copy/原同步序列，phase completion inline；CUDA 在13.3实际 ABI冷解析 `_v2` 或
同 stream 原 memcpy fallback，默认 STREAM，legacy/缺席能力不在热路径尝试后换策略。所有 host-known
pointer/size/attribute 数组冷固定，fallback O(R) calls 如实记。原 QueueSequence 负责接纳/event/首错/退休。
Apple 的 blit 使用原 MLX stream/CommandEncoder 与原 Context Job owner，MCB 单次使用/同 queue排序、
visibility/error/真实等待明确；原 host→shared 边界与有限 staging 费用分别记录，不新建全局 Queue/进展线程。
Vulkan 固定 pair grouping/VkBufferCopy、原 Run/fence/queue family/cache visibility；可用 transfer queue沿原
handoff合同，legacy/imported 合法 compute queue或明确拒绝。tail 仅在已拥有的 source/destination physical
capacity 内合法扩展，不能覆盖邻块；不新增 repack/copy kernel。每份报告列实际 copies/calls/groups/host waits、
logical/physical/staging/metadata/unknown bytes，opaque driver/vendor 成本不冒充零。

### 有限实现 ledger（不新增 FIX ID）

| 原 FIX | 允许的产品接线 | 验收与停止边界 |
|---|---|---|
| F2-01 | 共同 descriptors/checked validator、各自 program/invocation/submission 与冷固定 metadata/预算/绑定 | R=0/1/多、IDs/capacity/overflow/OOM/失败不消费、二组二槽、ranges/catalogue replay不增长 |
| F2-02 | canonical physical spans、排序/scan independent alias、explicit phase closure、自复制消除 | 跨 wrapper alias、重叠写/相依批拒绝、重复 source、显式 A→B→C 与空批/边界 |
| F2-03 | CPU借原录制；CUDA原 backend cold batch/fallback、native producer依赖/descriptor保活 | 双向/同设备 byte equality、cold资格一次、native counts/STREAM ordering与失败prefix |
| F2-04 | Apple原Job/MLX native stream blit、shared visibility、合法staging与ready/退休 | 非CPU冒充GPU；实际MCB/取消/失败/两组数学与allocator/native成本 |
| F2-05 | Vulkan原Run的pair regions/record/submit/flush/invalidate/handoff与范围检查 | 真GPU、family/usage/tail、legacy/imported边界、一个phase completion/首错与排空 |
| F2-06 | compute内F2专用预去重/ordered read准入、写退休后发布、retained native依赖桥与每phase actual latch | public F1回归；pending后继read保活、旧group、部分成功、OOM/取消/放弃/unknown/close/drain/slot失效 |
| F2-07 | 普通facade样例与指南、原Direct/Graph数学消费者、P0账、头/absence/source/static/shared | 原编码/BF16/量化真实值，三平台/三交付/迁址/DLL、完整矩阵/七问/红队；未验不PASS |

每项都先实现完整已冻结语义；native 细节允许有限补读，出现新 owner/公开形状/常态成本等级变化先退回审计。
本次 AUDIT 是 SOURCE/设计裁决，全部七 FIX 的运行 verdict 仍 UNPROVEN，不据此关闭 F2。

### 首批精准验证声明

共同控制接线阶段仅选 `compute` 的新 range/phase/lease handoff/proxy依赖协议 tests 与既有
`ResourceLeases.*:Completion.*` 相关回归，新 `CopyRanges.*:CopyLeaseHandoff.*:CopyDependency.*` 控制协议与 header sole-include；复用本批 Windows共同/WSL compute构建，
build1/test1、4GiB、configure300/build1800/exe120，不全算子/P1。真实 CUDA/Apple/Vulkan故障与数学、
二组二槽及三交付在本包产品接线后按稳定候选集中运行，日志/预算不因阶段提交重置。
CUDA copy backend 接缝实现期另选原 `compute_backend_cuda` 构建树的 `CudaCopyBatch.*`，
只验证固定13.0 `_v2` ABI、同Context physical identity、H2D/D2D/D2H byte equality和有限replay；
没有签整体permission/controller/数学/故障资格。该门同样build1/test1和上述内存/时间上限，
新内部 `detail/copy_backend.h` 独立编译；全部端到端门继续待候选冻结。
公共CPU/CUDA控制对象接线阶段增加原CUDA构建树的 `CudaCopyProgram.*`：
二slot三阶段、last-write output、实际pending后继读、放弃/close/reset、失误不消费、空批/旧句柄及冷metadata回放账。
CPU对等门仍用原CPU构建树的 `CpuCopyProgram.*`；两者只签对应实现期接缝，不代签最终数学/三交付。
native slot冷创建加上本slot需要的原event容量；HYLE metadata计实际requested live bytes，
含共享program与其全部live slots一次，opaque driver/event内部费用分别记unknown，有限ResourceBudget必须显式允许。
报告将已冻结的wait账显式分成 `native_waits`（实际接纳的device dependency）与 `host_waits`，
`native_calls` 计copy entry-point；phase只导入声明的直接先行边，hazard校验仍用传递闭包。
Vulkan native录制实现期另选原Windows route构建树的 `VulkanCopyRecording.*`：实际GPU的
buffer-pair regions合并、H2D/D2D/D2H三phase byte equality、二独立Run、重复source/self/empty、
whole-phase overlap/对齐/容量/foreign Device和冷metadata replay；只签native接缝。
仍build1/test1、configure300/build1800/exe120和4GiB；每phase复用原Run，合格idle host upload
沿原transfer→compute handoff，其他沿原compute queue；未知external backing/transfer usage先明确拒绝。
Vulkan公共Program/Invocation/Submission、native producer/权限/故障、数学及交付继续待验，不拿此门代签。
原owned view的 `memory` 空是避免双重释放的正常形状，`bind_view_buffer` 已保留实际allocation_bytes、
VkBuffer与byte offset；F2冷copy span沿该已有来源证明，使用Device+原VkBuffer+offset作canonical identity。
原allocator每个root单独绑定Memory，views保持同handle；未知external借入为allocation_bytes=0，
仍因copy usage/alias无证明明确拒绝。首批门包括实际lease binding/import_readonly view回放与unknown外部边界。
公共Vulkan对象接线复用compiled `copy_control` 的root角色/预去重准入/逻辑覆盖算法；
该helper不拥有Queue/native submit，CPU/CUDA仍沿原执行，Vulkan每phase仍沿原Run。
Vulkan producer冷资格只接受同Runtime已完整入队的原native Run/inline；managed/async/unknown不猜prefix。
实际子phase退休后用原nonblocking poll合并producer/先行checked status，再退休本phase写guard，错误不发布。
native Run callback尾部仍可能释放其control block，metadata采用原ResourceBudget每allocation保活，
不把program内嵌counter的地址留给最后callback之后的free；实际known requested bytes及opaque费用分别报告。
本阶段增加 `VulkanCopyProgram.*`（二slot/三phase/原producer/old result/reset/close与真实故障prefix），
精确复跑 `VulkanCopyRecording.*`、34项common，及共享控制抽取直接影响的原 `CpuCopyProgram.*` /
`CudaCopyProgram.*:CudaCopyBatch.*`；仍串行原树/固定时限/4GiB，无全算子/P1。
首次configure在中央include owner检查发现新native控制state直接使用已关闭 `util_core::RefCounted`，
缺少route的direct private edge。实际borrow理由是原子refcount及最后native callback之后的metadata寿命；
沿中央组件图只为当前AUDITED `compute_route_vulkan`补 `PRIVATE_DEPS util_core`，
不重开Util、不实现私有refcount、不修改checker/公共工具或普通用户开关。该组件早已在Compute传递闭包，
公开形状/owner/方向不变；保留首次configure失败，不据此重置本卡预算。
放弃全部句柄后的先行phase观察沿原Run的nonblocking observe，不能poll已清空的公开Completion。
该实际回收门核对原Runtime.collect、每phase use归零、每allocation保活的metadata完全释放，
仍不把GPU已执行和完成句柄已退休混为一件事。仅暴露私有poll helper，不增加进展线程或执行层。
Apple沿原Runtime/Job FIFO及原MLX stream：原math Job在native同步后才退休，F2在该owner上
结束旧encoder、创建一次原command buffer的blit pass、录制有限ranges，再由原CommandEncoder.synchronize
commit/检查实际MCB结果。MLX buffer为untracked，因此不跳过原owner的实际同步边界；每native phase的
worker host wait和vendor单次MCB/descriptor成本如实计入，不伪装提交者等待或新并行runtime。
offset/size按[原Metal buffer copy规范](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy%28from%3Asourceoffset%3Ato%3Adestinationoffset%3Asize%3A%29)
在macOS均为4-byte约束；不同wrapper使用实际MTLDevice+MTLBuffer+byte origin检查alias。
首批Air门固定为 `AppleCopyRecording.*` 与原 `AppleLeaseLifetime.*`、34项common控制回归：
原GPU blit byte equality、三stage双向/同device、二slot/冷metadata replay、overlap/self/empty/offset/tail，
configure300/build1800/exe120、build1/test1与4GiB。仅签此native接缝；公共对象/producer/数学/完整交付仍待验。
首构建 `eb7aed17f` 在const report方法的mutex限定失败，未运行native测试；保留原configure/build日志。
对opt-in recording的观测mutex加 `mutable`，不改变同步/原worker/API语义，再沿完整archive同树取证，
不把编译器报错或新日志目录当作新产品轮/预算重置。
Apple公共对象仍按已冻结同名program/invocation/submission、固定resources/ranges/phases及两个独立slot；
借compiled copy_control、原ResourceBudget每allocation保活和原util_core refcount（显式private edge，
不重开Util/改变普通开关）。原worker blit前作纯状态predicate，失败/取消停止后继native记录，
phase guard只在实际成功MCB/FIFO退休后发布；empty dependency沿原Job marker而非伪造GPU完成。
worker preflight与公共output/wait的同Completion观察由每opt-in invocation的递归mutex串行；
阻塞wait先等待原recording observer，释放观察锁供worker运行，再裁决实际退休，不形成worker/调用者循环。
本阶段Air门扩到 `AppleCopyProgram.*`（两slot/replay/producer/较早组/取消/放弃/close/reset），
复跑 `AppleCopyRecording.*:AppleLeaseLifetime.*` 与34项common；独立头、数学和三交付留稳定候选集中门。
公共Apple首遍 `b7779933a` configure通过，编译发现新compiled source缺少直接StdAllocator include；
补明确的 `hyle/mem/std_allocator.h`，原FAIL保留，同机器/命令/完整archive重验；未运行不写PASS。
首遍增加owned-view用例的fixture将raw-copy合法4-byte offset传给既有storage borrow factory，
后者还要求native storage offset alignment；保留这次FAIL，fixture改为合法zero-origin外部carrier，
再检验F2的unknown usage拒绝，不修改旧factory或放宽其合同。

### 当前实施与首轮证据（2026-10-11）

原始日志、candidate/tree、命令、exit code 与逐日志 SHA-256 保存在
[共同控制收据](../../../../evidence/correctness/foundation-f2-common-controls/receipt.json)。
这些是 Windows MSVC Release/shared/source 的实现期控制协议检查，不能代签原生搬运或其他交付。

| 产品段 | 固定检查点 | 已运行结果 | 尚未证明 |
|---|---|---|---|
| 范围/物理 alias/显式阶段 | `8107b6a23` | 7新用例含500组独立模型比较，相关回归合计27项；2头独立编译 PASS | native usage、程序与replay |
| 有序读保活/最后写退休后发布 | `651c202fa` | 4新协议用例，相关回归合计31项；3头独立编译 PASS | 真实GPU逐阶段退休/数学消费 |
| retained producer identity/错误保留 | `73bae6e65` | 3新协议用例，相关回归合计34项；5头独立编译 PASS | 原生producer依赖及partial prefix |

三个检查点 build1/test1，启动前实际可用内存均超过4GiB；没有P1。
selector 的原生反向消费者与交付门按本卡集中阶段继续待验，保留完整 selected/skipped 清单。
七FIX仍是部分实现，program/invocation/submission、CPU/CUDA/Apple/Vulkan native batch、
二组二槽/真实编码数学/故障/三交付/P0与最终红队尚未完成；本卡维持 AUDITED。

CUDA接缝首构建在测试callback的返回类型推导失败（`HYLE_RETURN_IF_ERROR` 的传播代理与Status混合）；
原生产库编译通过，未运行GPU测试。有限纠正1：fixture callback显式返回 `Status`，保留原失败日志并
在同机器/树/命令重验；此处计入原两次有限纠正，不用新日志目录重置预算。
同机有限纠正后 `3c83f434d` 的2个 CUDA byte-route 用例与新内部头 PASS。
[原始收据](../../../../evidence/correctness/foundation-f2-cuda-backend/receipt.json)含首次失败与重验，
实际 RTX4070/driver617.14；三轮H2D→D2D→D2H的9个phase实际9次driver batch calls，
无CPU代跑。console首遍没有记录GTest property，唯一一次同binary XML补验只补调用数，未改变产品/预算。
这仍只证明backend ABI/byte/replay接缝；public controller、权限/故障、数学、CPU/Apple/Vulkan和完整交付待验。
冷slot追加原Queue identity防误用后，`52e437c8e` 同一精准门2项再次PASS，
仍是9个phase/9次native batch；同Context另一Queue也在native接纳前拒绝，证据列入同一收据。

公共对象首轮 `1c1ce5f26` build/header通过；两槽/三阶段/old-result/固定metadata用例通过，
早ready用例错误地在pool.drain关闭之后期待writer；range/空批用例的异常经单例复现与Ninja deps核对，
定位到验证树的MSVC locale头依赖丢失/混合新旧ABI。
[独立验证副本blocker](../tooling/foundation-f2-windows-header-dependency-blocker.md)记录事实与有限恢复。
有限纠正2：fixture按关闭合同核对回收，scratch runner英文include输出并精确清理两个目标的artifact重建；
原失败/诊断继续保留，新 public/common运行资格尚未取得，不增加或重置原两次纠正预算。
`4d937b2f8` 有限重验PASS：3个公共CUDA用例、2个原生byte接缝用例及34项common回归，
3头独立编译、真实header deps全部通过；没有热metadata增长，真实gated producer的较早组可以read，
后继readonly use仍保活，放弃两控制句柄后metadata实际归零/旧reader最后回收。
[公共对象首轮收据](../../../../evidence/correctness/foundation-f2-public-program/receipt.json)保留两次attempt、
单例diagnostic与依赖故障。CPU、Apple/Vulkan、实际数学/故障全矩阵/三交付仍待验，维持AUDITED。
CPU `d16f07f92` 的MSVC Release/static/source首轮同样PASS：3个公共program用例、34项common回归与3头，
复用原同步byte copy，无伪异步；二slot三轮不增长metadata，4次copy/3phase/零native与host waits，
跨read子view的physical self-copy实际消除，later write不提前发布、失败绑定不消费及empty/OOM均通过。
[CPU收据](../../../../evidence/correctness/foundation-f2-cpu-program/receipt.json)保留精确tree/命令/deps/log SHA。
当前CPU/CUDA公共对象完成此首轮实现期检查；WSL/Air、Apple/Vulkan程序、真实数学与完整故障/交付矩阵
尚未通过。没有把这些首轮用例计成七FIX全完成或CANDIDATE；四owner仍AUDITED。
Vulkan `d256ca5b1` 的原录制门3项PASS；补齐owned-view来源后 `a09c9dd32` 的新增fixture在
旧storage factory的alignment失败，`f65a5ed3a` 使用合法carrier后4项全部PASS。
[原录制收据](../../../../evidence/correctness/foundation-f2-vulkan-recording/receipt.json)保留三个checkpoint及失败，
每次34项common回归/实际header deps/独立头通过，XML实机为RTX4070而非software adapter。
原生双向/同设备字节、同pair合并、独立pairs、3次replay无metadata增长、物理alias与合法readonly view已取证；
尚无Vulkan公共租约Program/Invocation/Submission与producer/完整故障/数学/其他主机/三交付资格。
公共Vulkan接线 `f7e49d924` 的8项门PASS；补actual Run观察后 `3391e2bd5` 的9项PASS，
包括双slot三phase、原retained producer+pending mutable源、真实QueueSubmit后缀拒绝/前组保留、
empty dependency的真实prefix marker、失败不消费，以及全部句柄放弃后原collect使use/metadata归零。
该段共享控制抽取的原CPU3项/CUDA5项精准回归同candidate PASS，四次成功构建各34项common、
实际header deps和3头通过；没有数学/完整其他平台/SDK交付P0终审资格。
[公共Vulkan收据](../../../../evidence/correctness/foundation-f2-vulkan-public/receipt.json)保留首次direct edge
configure FAIL、WORKTREEIntroducing未归属的selector BLOCKED及提交后owners明确AUDITED的READY，
没有改公共checker；中央图导致的扩展component/package门列入稳定候选集中阶段，当前不代写PASS。
CPU/CUDA/Vulkan公共program首轮接线已取证，Apple及本卡完整矩阵仍未完成，维持AUDITED。
Apple native接缝 `18e1eac2e` 首轮PASS：Mac16,12/16GiB/macOS27.0.1/AppleClang21、
3个新Metal blit与3个既有native lease生命周期用例、34项common回归；实际三stage GPU字节与3轮replay，
HYLE冷metadata稳定，macOS offset/size/alias/self/empty及readonly imported view边界通过。
[Air原始收据](../../../../evidence/correctness/foundation-f2-apple-native/receipt.json)保留首次const-mutex
compile FAIL与有限重验、完整archive SHA/commit/tree和实际机器/命令/log SHA，非Windows或mock替代。
原worker每phase一次CommandEncoder.synchronize明确计入host wait，vendor MCB/descriptor成本仍UNKNOWN；
不宣称所有vendor热分配为零或物理重叠。独立头/Apple公共控制对象、producer/完整故障/数学/三交付待验。
Apple公共对象 `2d491c141` 的Air首轮PASS：4项program/slot/phase/producer/early independent group/
取消/放弃/reset/no-consume用例，复跑原3项blit、3项AppleLeaseLifetime和34项common。
双slot/3replay的实际requested metadata稳定，原FIFO实际native producer与writer guard正确交接；
真实GPU primitive边界被阻塞时，旧组仍可读，queued取消/丢弃不会提前释放，原close后metadata归零。
[公共Apple收据](../../../../evidence/correctness/foundation-f2-apple-public/receipt.json)含首构建FAIL、完整archive
SHA/commit/tree、硬件/工具链/命令/log SHA；当前四路线公共入口首轮均有实物证据。
四owner仍AUDITED：实际BF16/原编码量化的Direct/Graph消费、fault全矩阵、平台/头/三交付/迁址/P0终审
与稳定候选红队尚未完成，不能从这里跳到CANDIDATE或CLOSED；F3仍未实施。

| FIX | 必须实现 | 验收 |
|---|---|---|
| F2-01 | CopyBatch/phase 的冷准备、固定资源域/metadata/byte budget 和范围绑定 | 稳定 replay 改 IDs/ranges，满容量/零批/溢出/OOM，无重新搜索 |
| F2-02 | 全批 alias/跨 range 独立性、native usage/方向/对齐检查，self-copy 消除 | 共享 source 允许；重复/交叠写及 A→B→C 同批拒绝；phase 链正确 |
| F2-03 | CPU/CUDA native 批量与冷资格 fallback | 真 byte equality、原生使用计数、严格 producer ordering；不热 cuGetProcAddress |
| F2-04 | Apple 原生 batch/blit/ready，与共享内存 CPU 可见性 | 不隐式绕过同步/把 host 写当 GPU 已完成，pending 显式 |
| F2-05 | Vulkan pair grouping、command/fence、family/cache 依赖与明确 tail | 一个提交 prefix；legacy/imported storage 拒绝/合法 fallback；不复制 native runtime |
| F2-06 | F1 guard 交接、partial acceptance、取消/丢弃/close、slot 退休与再绑定 | 第一项已接受后第二项失败，旧 bytes/index 不重用；真实未知退役路径 |
| F2-07 | Direct/Graph 数学消费、逐组 ready、P0/三交付/文档 | 至少二组二槽，成熟数学消费复制后的真实量化/BF16数据，原规则不变 |

## 预声明验证与停止

普通数学接线阶段选新增 `samples/package/graph/batched_transfers`：先在Windows原CPU/CUDA
源码消费树以普通 `hyle::graph` 运行两组/两槽/三次replay，BF16 dense及原GGUF Q8_0
先host→device→device→host，逐字节读回，再由原Direct/Graph MatMul消费device ready leases。
CPU用同合同host变体，不伪造device能力。此阶段不签Apple/Vulkan或SDK交付；两者接同一
样例后再沿卡内全矩阵取证。build1/test1、4GiB、configure300/build1800/exe180，冷metadata
allocator只统计CopyBatch自身；成熟数学准备/Graph编译成本不混入copy replay账。

有限矩阵覆盖 R=0/1/多范围、同/不同 buffer pair、单/多 phase、重复 source、坏 ID、overlap、
host/device 原编码、aligned/tail、pending 与 retained result。
每路线对一次 staged upload、一次 D2H/read 与 D2D/host 合法变体作数学消费；故障只装饰真实 native calls。
复用现有 native transfer/math fixture，不建立新 benchmark 军团。

P0：prepare allocation/budget 与 bind 校验 O(R log R) 或更好（列出实际算法），
submit O(R+G) 的 native commands/一次 phase completion；热元数据容量固定。
native batch 缺席 fallback O(R) native calls 是明确成本，不虚报单 driver call或性能赢家。
若需优化/计时，先源码/同完成账单，再一份不可变候选、一项完整旅程、有界门；不无限 GPU A/A。

本卡完成后即可供 F3 使用；不等 F3 才首次验证三交付/四路线。
运行期 GPU-produced索引、repack kernel、模型 cache 规则不在本卡。
