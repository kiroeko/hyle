# G06 metadata 与 decode 成本收口

状态：**SCOPED，2026-10-16**。本卡沿[完整十项目标](../project/inference-gaps-goal-execution-2026-10.md)，
不缩减 CPU、单 GPU、CPU＋单 GPU、单/多请求或四原生路线。先完成四组共同审计和形状冻结，
再集中实施、批量检查、按实际失败定向修正；当前不授权产品修改。

## 当前 owner 与 baseline

canonical baseline：`10129b90bbfc4d85c25e1595c823962b5a9c8146`；本轮前产品输入分别为：

| owner | 原产品 candidate | 当前阶段与边界 |
|---|---|---|
| compute_batch | `982676cb05a6e6244e66e0a596347aa96043617b` | 原 G04/G05 聚合 BLOCKED 后恢复到本卡 SCOPED；布局、上传与权限准入 |
| graph_batch | `982676cb05a6e6244e66e0a596347aa96043617b` | 同上；共同 metadata 与原 region 依赖 |
| CPU Provider/runtime、FlashInfer、Apple/Vulkan native controller 及三个 hybrid | 各 registry 的原 candidate/terminal | 本阶段只读，不自动转移施工权；找到具体必要修改后，在本卡固定范围并逐 owner 合法进入相应阶段 |

G04/G05 数学域与全部原失败、费用、未知、正确性/交付证据继续保留。
[CE28 成本检查](../../../../evidence/architecture/inference-g04-g05-cuda-ce28-producer-stage/consumer-cost-checkpoint.json)
已终止，没有新 FI material 双钥匙；原 CUDA prefill 非劣 UNRESOLVED 不冒充 PASS。
按总卡允许的真实依赖调整，G06 可先推进；共有费用降低不保证 CUDA 相对比率改善。

## 真实任务与有限范围

主消费者为 Qwen attention 的单请求与多请求 Q1 decode，以及已有混合 prefill/decode；
公开旅程为 `BatchSchema → BatchLayoutVersion → PagedArenaRead/原 pending carrier →
BatchInvocation/Graph coordinator → checked Completion → owning output → close/drain`。
请求选择、bucket、KV 管理与采样策略仍由引擎拥有。

正交消费者为两个真正 attention node 共享同一不可变布局、保留不同 native workspace 的
数值 DAG；改变请求顺序/加入/结束、shared physical pages 和旧结果存续反证特例化。

四组清单沿原总卡冻结，不新增第五组机制：

| ID | 已有缺口与承载裁决 | 实施前必须冻结的决定 |
|---|---|---|
| G06-M | 每版全容量分配/清零，每步全 metadata staging/upload。SELF，compute_batch | active snapshot 与原 fixed-capacity metadata 导出的关系；immutable generation、inactive 字节合同、changed ranges、失败保旧版、预算与线程安全。不能仅省 warm copy 而把全容量费用藏到每 token 的 version 创建 |
| G06-S | Graph 在逐 node 上传之后才共享 mirror。SELF，graph_batch/compute_batch，复用原 F2/G03 | 在首上传前按同 generation、schema、resource、queue/原依赖合格分组；各 node plan/workspace 独立，reader 活到真实退休；不以换队列后的旧 ready 推定新依赖 |
| G06-L | 每页普通 `LeaseUseBatch::read` 重扫已准入表，最坏 O(G²)。BORROW 原内部 `read_unique` | 按原 lease block identity/generation 去重，保 permission、poison、writer/reader 和 pending 顺序。`LeaseBackingInfo.identity` 是共享 backing，不能代替 lease identity；不得修改公共 F1 检查来签通过 |
| G06-D | 四路线实际 topology/runtime 与 Q1 planning/capture 成本。BORROW 数学、SELF 必要状态桥 | 核真正原生 plan key 和编译期长度，不能只放松 key 或因 PrefillPlan 名称判缺 decode；host-known 不暗中 D2H，device-produced 仍经 checked boundary；capture 单独取证 |

原 CUDA plan 只依赖 Q/KV indptr；page IDs/tail 不参与该计划是正确行为，应保留。
Apple/Vulkan 当前编译期切片保留布局 owner，不能在长度改变时复用含旧常量的 Program。
所有已关闭数学、F1/F2、scheduler 和 queue 继续复用，不新建 planner/线程池/通用 schema。

固定 peer 来源与精确文件 SHA 已在上述只读证据中重新核对：SGLang `7224fce1` 的
host-known plan/active buffers、vLLM `98f72111` 的 decode wrapper/tensor-core 默认、
llama.cpp `cbe28bb3` 的层外 attention input 创建与逐层消费。它们只证明成熟形状，
不代签 HYLE 的异步、capture 或速度资格，也不要求四路线使用相同 native 对象。

## 审计阶段与停止条件

当前 SCOPED 只做源代码、真实 consumer、有限 public 调用草图和成本反证；无产品编译/P1。
审计需一次性回答四组的 owner、依赖、公开对象/identity、ownership、失败/取消/关闭、
完成与并发、冷热分配/搬运/等待，再列唯一有限 KEEP/FIX/BORROW/REJECT 表进入 AUDITED。
普通 C++ 使用不要求用户组合 benchmark 私有 Provider；不保旧接口兼容。

调查、实现、验证各以 2h 为检查点，按用户常设授权有限追加且保原累计，不重置失败/未知。
到点必须记录已有结果、剩余工作与具体下一阶段；无新机制证据时停止该调查，不扩为
CUDA tile/阈值搜索。若真实依赖阻断某 owner，精确记录恢复条件，仍推进其余独立工作；
goal 不因单次构建慢或一个模块未决而停止。

## 预声明验收

P0 裁决 active/changed/uploaded bytes、上传次数、plan/replan、host waits、warm allocation、
页权限准入复杂度和 native workspace。验收用真实操作与原 report/allocator/native 观察，
不为某个 benchmark 行添加公共字段。P1 仅在 API/算法冻结后按实际必要成本问题定界，
不能把旧 G04 CUDA 合同通过视为 G06 完成，亦不为状态/说明变化重跑它。

- 保原 public single/multi、Direct/Graph、三联合和四 native 准入域；数学与 G03 证据仅按
  实际未变输入借用。空/容量/稳定 tail/跨页/请求重排/加入结束/同 shape 不同 layout 必测。
- 同 layout 两 attention node 检查一次共享上传、独立 workspace、队列依赖和真实退休；
  foreign schema/generation/resource、旧 output、关闭与未完成 ticket 不能错误复用。
- 完整 lease identity 重复/同 backing 不同 block/不同 generation、权限冲突、poison、
  producer 失败和 pending writer→reader 继续拒绝或传播原错误，不以去重绕过原 F1/F2。
- host-known 不发生隐式 D2H；device-produced 的未完成/失败事实不能发布 ready；
  OOM/fault injection 保旧版与已接受前缀，cancel/abandon/unknown 不假退休。
- 受影响 source/独立公开头/搬迁 static/shared/直接消费者与适用 ASan/UBSan/TSan；
  native 数学改变才扩相应设备风险，具体 target/命令在 AUDITED 时固定。Apple 只在实查
  身份的 MacBook Air 上，Windows/WSL/Air heavy 串行，最多 4 编译 worker。
- 使用 change-impact selector，保 selected/skipped 原因；不重写公共 checker、构建或性能核心。
  registry 每轮带实际 previous-ref；至少 docs/closure/diff 检查。未运行项不记 PASS。

七门、红队、LOC、完整剩余支持/负空间与最终性能方向在 immutable candidate 后实际裁决；
当前没有任何 G06 产品或运行 PASS，完成项仍为 G01/G02/G03。
