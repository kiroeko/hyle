# R7：四路线基础终审、诊断与清理

## Closure card / CANDIDATE

- baseline：`14e78c45cd58c2836ffc6b3bcc0394fdb4fbc220`，Windows canonical 工作树干净。
- owner：`compute`、`graph`、`compute_route_apple`、`graph_route_apple`、
  `compute_route_vulkan`、`graph_route_vulkan`。旧 terminal 与候选见
  [继承快照](../../../../evidence/correctness/compute-graph-r7/inherited-baseline.json)。
- trigger：[七包计划](four-route-foundation-refactor.md) 的最小 opt-in Obs 尚无实际调用；
  [Graph §7](../../../architecture/graph-design.md#7-依赖构建与诊断) 要求独立编译 metadata，
  Apple/Vulkan 目前仅返回 Program，CPU/CUDA report 尚缺节点数值政策。
  这是已承诺能力的具体缺口，不因“再看一遍”重开。
- 普通任务：冷编译一个已支持 FFN/custom 图，保存可解释的规则、数值、资源、并发和拒绝诊断，
  释放 Graph/Program/payload 后仍可读；需要排障时启用主机范围跟踪，平时完全关闭。
- 主纵切：四路线普通 FFN 的 prepare/compile → bind → submit → checked wait。
  正交纵切：已交付的非模型 custom/native 资源管线及拒绝/释放；诊断不能保活任一设备 owner。
- 直接依赖：既有 Compute/Graph、Error/Obs trace 门；不修改依赖实现。构建先决问题由
  [独立小卡](../build/optional-factor-trace-gate-v1.md)承担。
- 范围：六个 owner 的最小 host zones；冷 metadata；有限旧入口/当前指南/证据汇总。
  不改默认计算规则、native executor、Program/Invocation 热布局、算法、Provider 选优或失败协议。
- 非目标：算子目录、模型引擎、GPU timestamp/per-kernel profiler、常开计数器、全局注册表、
  新工具框架、重新计时 CPU 成本例外、Mem/Obs/Sched 全库复审。
- 用户补充（2026-09-17）：一并核对旧 Sched `structured-spawn-join` 的不等责测试设计。
  范围限该测试及四路线已有配对比较入口/合同/当前结论；不等价的两边不能继续作为主动 A/B
  排名或验收测试，即使标了 diagnostic 也不能被重新误用。历史原始数据保留，必要时退役
  现行误导入口；不通过修改 Scheduler、删减产品责任或重写公共性能核心来“修复”测试。
- 性能：P0。关闭 trace 的热路径消除证明、源码/成本与代表性 Release codegen；冷 report 线性
  metadata 复制独立计入 prepare；Vulkan 常量去重另有 O(C log C) 冷排序。继承已测计算路径的 P1，
  原正负方向和设备/版本限制不变。
  若出现未预见热布局/算法/成本改变，退回 AUDITED 并依法决定适用验证，不能偷借旧证据。
- 调查预算：已登记参考模型与具体源码一遍；只补实现细节，不重新研究四路线架构。
  一份组合不可变候选启动适用普通门；失败仅定向修正受影响项，0 次新增性能采样。
- 验证：默认 OFF 和 Tracy ON 的真实 Direct/Graph、头/导出/三交付；Windows CPU/CUDA/Vulkan、
  macOS Apple/MoltenVK、WSL CPU/CUDA 相关路径；其余平台只继承可证明未变路径。
  report 正常/拒绝/reset、Graph/Program/constant/custom owner 释放后的可读性、metadata OOM。
  适用 host ASan/UBSan；无新共享可变状态时以原 TSan 加静态并发证明继承，不虚称重新运行。
  selector 对中央 manifest/helper 的 broad gates 如实执行，重型串行、build 2 / CTest 1。
- stop：有限 FIX 完成且七门有证据后结案，通知基础架构完成并停止；未知硬件保持未资格化。
  Mem v10、ggml CPU BLOCKED 与 compute_sched QUEUED 按原状态，不扩开新队列。

## 审计边界

| 能力族 | 当前事实 | 动作 |
|---|---|---|
| 各路线普通计算、规则、资源与三交付 | R1–R6 已各自验收；用户不提供 timing profile/Provider 排序 | KEEP，按 receipt 与实际源码核对 |
| 编译诊断 | CPU/CUDA 已独立持有 route/copy/sync/storage；native 路线缺 report，节点数值字段缺席 | FIX，冷值语义，不保留 native owner |
| 跟踪 | Obs 已有编译 OFF/Tracy ON；六 owner 没有真实范围事件 | BORROW，使用现有静态名称 host zones |
| 旧 API/资源可行性规划 | 共同 Compute/Graph 有 CPU/CUDA 活消费者，R2 已移除 timing profile 竞争 | KEEP 有需求机制；只删有确切反例的残留 |
| CPU decode 原比较 | 裸 direct 服务不等价，已在 `14e78c45` 对称修正八项分类 | KEEP 成本诊断/原始数据，不重测或重开 Sched |
| 不等责配对测试 | 旧 Sched typed join 对 Marl 较轻服务；四路线保留的裸 native 诊断须与同合同入口分清 | 有界核对现行入口并冻结清理，禁止苹果对橘子形成新 A/B 验收 |
| 全库/模型策略 | 不属于本轮基础功能 | OUT_OF_SCOPE |

## 有限 FIX 与 API 冻结（2026-09-17）

SCOPED 提交 `ecfed35fd9d1d6d5c70f201213c98f2f4c2f10e6`。已完成上述具体源码一遍；
没有发现需要改变四路线执行架构的反例。以下集合冻结后开始实现，不追加算子或测量任务。

| ID | 证据 / 允许改动 | 验收 |
|---|---|---|
| F1 | 六 owner 实现文件调用现有 `HYLE_TRACE_ZONE_NAMED`，中央与 factor 声明 TRACE；CPU/CUDA 实际入口在共同 compute/graph；Apple 含 native owner 执行范围；Vulkan 含冷准备与真实 submit | 默认 OFF 无状态/分配/诊断调用，ON 真实入口可编译运行且单一 Tracy client；仅 host 范围，非 GPU timing |
| F2 | 共享纯值 `CompileNodeReport`（原节点 ordinal、OperationKind、OperationTraits）；CPU/CUDA 现有独立 report 增加 indexed node 数值/累加政策；native CompileOptions 增加只在 compile 借用的 report 指针 | 现有 ReportState 与冷 metadata 可变，Program/Invocation/Completion 热布局不变；不复制 Operation custom payload |
| F3 | Apple/Vulkan 原生 CompileReport 各自持有 Status、环境字符串、输入/输出描述、live node metadata、固定实现/数值/边界/并发政策和本路线可证明的资源事实 | 独立 owning，编译前 reset，失败不残留成功统计；保留 report 后 Graph/Program/constant/custom owner 真正释放；cold OOM 返回 error |
| F4 | 当前指南、前六包证据对照与旧入口校准；已无消费者的旧 timing profile 面不恢复，实际资源可行性规划保留；修正 R5 当前 Mem 尾项措辞 | 每个 MUST 有实际证据或明确限制，不以研究/旧 CLOSED 代替实际能力 |
| F5 | 用户要求的有界比较清理：Sched v1 `structured-spawn-join` 裸对照、CPU Graph 四 fused/四 natural 裸对照，CUDA/Apple/Vulkan 六个 bare-native cost targets/专用分支退出现行 A/B；历史源码/合同/数据以冻结提交追溯 | 现行入口不再产出不等责双边计时；完整服务对照及独立数值 oracle 保留；不改产品、不新计时、不把移除数据改成 WIN/PASS |

### F2/F3 值语义与成本

- `include/hyle/graph/compile_report.h` 仅定义共享节点语义值，不含执行器或 runtime 类型。
  `CompileReport::node(index)` 查询 CPU/CUDA 冷报告；原节点数与数值 traits 使用 GraphState 的
  `kind()/traits()`，包括 symbolic View，O(1) 查询、O(nodes) 冷复制，不重新运行 schema/payload。
- `graph::apple::CompileReport` 与 `graph::vulkan::CompileReport` 是各自可复制的冷值结构。
  `CompileOptions::report=nullptr` 为默认；只借用到 compile 返回，不进入 native callback/制品。
  错误使用现有 `Status`（只持有错误文本/原生错误值，不持有 tensor/native owner），不造私有错误协议。
- Apple：报告 MLX managed whole-region、F32/affine-Q4 及 typed traits、就绪常量/原生结果无影子拷贝、
  每槽独立但 Context owner FIFO 执行；logical constant bytes 只按声明 views 计数。MLX fusion、
  私有 heap、driver 和运行输出实际字节保持 unknown，不能把 logical bytes 当 RSS/硬配额。
- Vulkan：窄 `detail::PlanFacts` 可选冷输出从最终 prepare_plan 取得实际 dispatch 数、每槽 storage
  block capacities、去重 HYLE Buffer owner 的常量容量/已知 owned VkDeviceMemory 与 borrowed 标志。
  Graph 不从共享 Context quota 前后差估算资源，不把外借 buffer 或 opaque driver 空间算成零。
  规则、strict F32、cold packing、compute barriers/ready handoff、独立 native slot 责任单独说明。
- 默认不构建 native report，不新增常态 metadata/atomic/lock/registry；公共 C++ 值对象遵守已有
  同编译器/STL 的三交付约定。新增不属于 public API 的诊断助手保持 owner 私有。

### 七问与验收适用性

1. 外部依赖仍为已审锁定 Tracy/oneDNN/CUDA/MLX/ggml shader，无版本升级或新 vendor。
2. 主 FFN 与正交 custom/resource 两条旅程覆盖报告与跟踪；图外 Direct 仍直接用各 native 路线。
3. 复用 Obs 宏、Error Status、现有 shape/traits 与 prepare_plan，不建立另一套观测或内存计费器。
4. P0 只新增冷复制；OFF hooks 消除，运行布局不变。F5 的同合同资格先按源码责任核对，计时不能
   证明合同等价；原 LOSS/INCONCLUSIVE 和 CPU 成本取舍如实保留。
5. 不恢复旧 C/FFI、不制造统一 kernel vtable；共享 metadata 不要求共用 Queue。
6. 中央 TRACE helper 矛盾由独立工具卡修复；实际 OFF/ON consumer 与 source/static/shared 验证。
7. 指南说明 host zones、report lifetime、unknown memory 与明确失败；不要求用户理解内部 factor。

新增语义风险为 cold metadata OOM、reset/拒绝与 owner 意外保活，须有行为测试及适用 sanitizer。
跨线程执行算法不变，原竞态/失败测试复用需要逐路径证明；新 trace ON 含外部 profiler，仅签 host
观察功能，不继承性能数据。F5 只做构建/独立 oracle 与入口负检查，不做新的 P1 比较。

## 预声明验收与停止

七门逐项使用实际 evidence，未运行不得写 PASS。本轮把前六包默认规则、普通消费者、真实 native
错误寿命、三交付、有限 P1 和 Mem 尾项按原要求对应到明确 receipt；不把历史 CLOSED、文档提交、
mock 或 copy 当成新路线完成。新增诊断只使用元数据，不加入 GPU 全设备内存/内部融合的虚假精确值。

每次 registry 迁移运行 `check-module-closure.py --previous-ref <迁移前提交>`；文档/状态只运行
selector 所选 docs/closure，保留 skipped reasons。终局核对当前性能账本与 owner binding；施工期间
原账本是 R6 终局快照，不能用它代替本卡当前 registry 状态。LOC 使用统一脚本统计整个 Compute +
Graph，不含测试/bench/vendor；baseline 为 **36,765 行 / 129 个代码文件**。

## 不可变候选与验收进展（2026-09-17）

六个 owner 进入 CANDIDATE，绑定 `87ae583407cbb5ed13108c3837b020e6e343a3ca`。F1–F5 的产品 API、
算法、运行对象布局和活动比较入口已冻结；F4 的最终 MUST 索引与终局证据尚待剩余验收。
本轮产品来自 `e84a7ad3`，其后三次提交仅修正新增测试的语义假设和 descriptor 形状；
`src/include/cmake/samples/bench/scripts` 逐树一致，不触发新的性能采样。

[已归档 P0 与配置证据](../../../../evidence/correctness/compute-graph-r7/README.md)覆盖九个 B1
正反 configure、四份 trace-only 运行源、十五份未变算法/依赖树，以及 29 个真实 Release OFF
函数的指令与调用目标比较。只对已核实的源码位置身份做归一化，未掩盖执行成本变化。
统一脚本统计本候选整个 Compute + Graph **37,035 行 / 130 个代码文件**，不含测试/bench/vendor。

Windows 普通门在不可变验证副本执行；原失败和定向测试修正后的续跑记录分别保留，不伪造原整批
PASS。剩余适用项目为跨平台 report 寿命/OOM 与 host sanitizer、真实 TRACE OFF/ON 三交付、F5
fixture 构建和独立数值 oracle、最终红队与证据汇总。当前不能签 CLOSED 或宣布基础架构完成。

[MUST / evidence 索引](four-route-foundation-acceptance.md)已逐项连接 R1–R6 的真实 consumer、规则、
寿命、交付与性能记录，保留具体平台和格式限制；R7 未运行项仍明确待验。
[候选后的有限源码复核](../../../../evidence/correctness/compute-graph-r7/source-review.md)已完成，
未新增产品 FIX，不代替尚未完成的运行验收。

### Windows 选定门完成

[111 门覆盖记录](../../../../evidence/correctness/compute-graph-r7/windows-selected-identity.json)已完成：
39 个原批 PASS、9 个测试修正后的既有 PASS、63 个最终续跑 PASS 去重覆盖全部选定门，七个 skipped
reasons 与原始失败分别保留。十五个公共产品的三交付全部通过；主 factor/header 使用 Ninja，续跑
package smoke 实际使用同 MSVC 的 MSBuild，身份如实记录。没有重测已完成 P1。

Mac ASan/UBSan 已按既定串行任务启动；其后仍需实际完成 TRACE OFF/ON 三交付与 F5 fixture 门。
Windows 覆盖完成不代替这些尚未运行的验收，六个 owner 继续 CANDIDATE。

### Mac host sanitizer 完成

[ASan/UBSan 记录](../../../../evidence/correctness/compute-graph-r7/air-host-identity.json)在同一候选的
Air 验证副本上分别通过 7 个程序 / 395 个 GTest，49 个具名插桩编译单元，另有六个公共头门。
Apple/Vulkan 报告在 constant/custom/Context owner 释放后仍可读，reset、OOM 与重试均实测通过；
Vulkan borrowed/owned 统计和 quota 回收、Generic Graph 逐分配点回滚也通过。

只签 HYLE host C++，不签 vendor/driver/GPU 插桩；TSan 保持源码限定继承。Mac 的 TRACE OFF 源码
普通消费者已通过，其余 OFF/ON 静态/动态及 Windows/WSL 交付和 F5 fixture 门按已启动的串行任务执行。

### Mac OFF/ON 三交付与 F5 完成

[Mac 交付](../../../../evidence/correctness/compute-graph-r7/air-delivery-identity.json)已完成 TRACE OFF/ON
各三种交付 × Compute/Graph，共十二条真实消费旅程；同一源码 ZIP、SDK relocation、隐藏原 producer、
单一 Tracy client、普通 FFN/custom 与报告均通过。
[F5 fixture](../../../../evidence/correctness/compute-graph-r7/air-fixtures-identity.json)完成八个程序构建、
四个 Apple/Vulkan 数值 oracle 与退役入口负检查，0 次新增 P1。

fixture 首次因验证脚本混用 product/developer build 入口在配置期被拒绝，原失败保留；改用独立构建
目录后通过。产品候选与公共工具不变。剩余仅 Windows/WSL OFF/ON 三交付、Windows F5 fixture、终局
证据与账本绑定，继续按既定低并行串行任务执行。
