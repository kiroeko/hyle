# R7：四路线基础终审、诊断与清理

## Closure card / SCOPED

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
  metadata 工作独立计入 prepare。继承已测计算路径的 P1，原正负方向和设备/版本限制不变。
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

具体 API、owner/常态成本与有限 FIX 在 AUDITED 后冻结；SCOPED 不授权产品代码。

## 预声明验收与停止

七门逐项使用实际 evidence，未运行不得写 PASS。本轮把前六包默认规则、普通消费者、真实 native
错误寿命、三交付、有限 P1 和 Mem 尾项按原要求对应到明确 receipt；不把历史 CLOSED、文档提交、
mock 或 copy 当成新路线完成。新增诊断只使用元数据，不加入 GPU 全设备内存/内部融合的虚假精确值。

每次 registry 迁移运行 `check-module-closure.py --previous-ref <迁移前提交>`；文档/状态只运行
selector 所选 docs/closure，保留 skipped reasons。终局核对当前性能账本与 owner binding；施工期间
原账本是 R6 终局快照，不能用它代替本卡当前 registry 状态。LOC 使用统一脚本统计整个 Compute +
Graph，不含测试/bench/vendor；baseline 为 **36,765 行 / 129 个代码文件**。
