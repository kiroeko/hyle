# G09 完成驱动修正

状态：**SCOPED，2026-10-17**。用户明确要求“我们先修复g09”。
baseline `8119dc6c2c02f5b3e588cc6e3b64b36b41222c59`；原 G09 产品候选
`3524413272cb52d06af5e64d634edbfef09b9864`、原功能/交付/CUDA P1 与三组未资格结果保留。
本卡替代本轮三个 owner 的施工入口，不重新授权旧 profiler/重试脚本。

## 需求与范围

- owner：`graph_route_cpu_vulkan`、`graph_route_cpu_apple`、`compute_route_apple`。
  前两者由 BLOCKED 回 SCOPED；Apple native owner 因真实组合消费者缺少主动完成通知而重开。
- trigger：[R29 因果诊断](../tooling/inference-performance-recovery-2026-10-17.md)在完整数学/
  结果/退休不变的 Windows Vulkan 消费旅程中确认周期观察策略有毫秒级成本。
  诊断占用 Scheduler worker，不可直接采用。Apple 原完成者虽主动执行，hybrid 仍按 timer 观察。
- 主纵切：同一 DAG 的独立 CPU/native 数学支路及 join，两个插入顺序；两个独立 invocation
  slots 的 CPU→GPU→CPU microbatch。反偏置纵切：批量数值分支与必要串行链，无 token 身份。
- 普通调用：Graph/placement→compile→create_invocation→bind→prepare→submit→checked owning
  result→release→close。native 数学、Queue、transfer/page、LeaseSubmission 和结果 owner 复用。
- 有限范围：让原 fiber 由真实完成/有界等待执行者唤醒，保 single worker、多在途、新 ready
  native、错误/取消和 accepted prefix 实际退休；处理通知/等待任务尾部后才允许复用。
- 非目标：G04/G05、CUDA 数学/Graph、公共 Scheduler/BlockingPool/Completion 改造、新线程池、
  新 Provider、全库重新资格化、采样/估计量/阈值调整。
- 直接依赖：原 Graph frontier/leases、Sched CV/Event/BlockingPool、原 Apple/Vulkan 完成者。
  直接消费者为现有 hybrid contracts/pipeline、native page/state joint、公开 SDK sample。

## 准入与暂定形状

能力是原 G09 联合进展的复合机制，SELF 只拥有协调；等待、数学、资源、取消和退休 BORROW
原 owner，模型分组/请求策略仍为 APPLICATION。所有既有公开入口继续归原 J1–J4 能力族。

| 项 | 已证不足 | 本轮有限设计 |
|---|---|---|
| W1 Vulkan | Runtime 无后台 observer；1ms timer 在无 ready native job 时观察 pending | 借 caller-owned BlockingPool；每 RunState 冷内嵌一个 waiter；不在 Sched worker 上阻塞原 Completion |
| W2 Apple | 原 FIFO worker 完成后，hybrid 未收到主动通知 | 原 invocation/copy owner 提供内部 opaque 通知，Graph 保有 node/epoch/CV；Compute 叶不依赖 Sched |
| W3 退休/错误 | queued job 取消不等于 GPU 完成；common Completion 的 const 观察会修改 handle | 冻结独占观察、destroy/release 尾部、拒绝/取消/close/异常进度与引用归还协议 |
| W4 使用/交付 | Vulkan 当前创建入口没有借用池 | 冷创建改为 create_invocation(Scheduler&, CPU Queue&, BlockingPool&)；无兼容双入口，迁移真实调用方 |

已有[最小设计与两条独立源码审查](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g09-production-fix-design.md)
作为本轮输入；`subscribe_retirement`/attach_lifetime 不会主动观察 GPU，不作为主动通知源。
Apple 单订阅/已完成竞态与 Vulkan pool 拒绝/queued shutdown 的唯一具体协议尚须审完，
在本卡与 registry 进入 AUDITED 前不改产品。

公共调用需要的 pool 必须活过 invocation/所有 waiter destroy/accepted native work；thread/stack
由原 pool 上限约束并明确计费。正常热路径不创建 JoinState/job、不新增 executor 或每 op fiber。
API/owner、失败/并发/常态成本在 AUDITED 冻结后批量实现，不边实现边改变合同。

## 预算与停止

- 设计审计 90 分钟检查点；实现与普通验证 4 小时检查点。慢 native 构建按原同配置保留累计
  时间，不用另目录或新卡重置原性能失败。每条路线只采用本轮冻结的一种机制。
- 普通失败最多三轮定向修复；架构/API/正常成本变化回 AUDITED。发现必须改公共等待/统计核心
  则留独立具名 blocker，不在本卡扩建。不能用更短 timer、忙轮询、更多 workers 掩盖进展问题。
- 重型任务全局串行；Windows/WSL/Air 最多 4 核/jobs、BelowNormal/nice10，native 编译最多2
  top-level jobs。Apple 仅 MacBook Air，先读取实际硬件/系统/工具链。
- API/算法和 correctness 固定成不可变 candidate 后，独立审查新有限测量入口。
  复用原 normal/holdout、四顺序、5%/10% workload A/A 与 n15→31 合同；不追加 epochs/n、
  不改尺寸/Provider/精度/估计量。新候选每未资格 profile 最多一次新入口资格尝试，任一
  preflight/A/A 失败即停止该 profile 并保原阻塞；不在模块内继续开发仪器。
- 本轮结构/功能修复和原 G09 聚合 P1 分开裁决。适用门未过写 BLOCKED/INCONCLUSIVE，
  不由单机有限提速或旧未变 CUDA PASS 代签新路线。

## 预声明验证

普通验证先 selector，仅改动 owner/真实直接消费者与公开闭包；旧未变叶证据按身份复用。

| 风险/交付 | 有限门 |
|---|---|
| 真实数学/进展 | 现有 hyle_cpu_vulkan_contracts / hyle_cpu_apple_contracts 和 pipeline；1/2 workers、两个 slots×65 步、两插入顺序、必要串行/纯 CPU 留出 |
| 新等待/通知 | 已完成/登记竞态、active waiter 时出现新 native ready、多 pending、同步/异步通知、generation/slot reuse，所有新路径用定向测试而非 sleep 证明 |
| 生命周期/运营失败 | cancel/abandon/close、旧 output、先错后仍在途；pool submit 拒绝/queued shutdown/running job、cold OOM/首 worker 拒绝；无早发布/无假退休 |
| 真硬件 | Windows Vulkan validation、Air Apple Metal validation、Air Vulkan；原数值/只读/页与状态 joint 直接消费者 |
| 宿主风险 | 改动 owner 的 ASan/UBSan/TSan 适用配置，runtime 不支持项按真实原因/替代门记录，不冒充未运行 PASS |
| C++ 交付 | 两个 hybrid 与 Apple native 受影响公开头、source/static/shared 迁址 consumer；未变 vendor/SDK 复用，不能拿旧映像签新 API |
| 性能/P0 | 冷 O(V+E)/有限 waiter/node 元数据、原 pool/thread 预算；warm allocation、观察独占、通知/锁/wake/copy；固定 candidate 后原完整 consumer 合同与资格 |
| 维护 | docs、closure --previous-ref、write-status、diff；最终统一 LOC/独立红队 |

SCOPED 以上门均待本轮运行；审计后补精确 filters/构建入口与适用风险，不默认通过。

## 七问与阶段

1. vendor/version/license 均不变，native 数学沿原成熟 owner；本轮不升级依赖或引入 fallback。
2. 只补原 J1–J4 的通知/等待/退休旅程；两种分支、两 slots 和串行留出共同约束形状。
3. 复用已有 file-stream 的 embedded BlockingJob destroy 边界、Sched CV/Event 与 Apple FIFO
   完成者；不复制执行器/定时器/资源权限。
4. 新线程/栈与冷节点费用公开，原所有性能结果保留；新收益待正式资格。
5. 普通公共面仍为 C++；内部 opaque callback 不构成 Native FFI 产品。
6. 按实际 headers/export/直接依赖检查 source/static/shared，未变依赖不全量重建。
7. Vulkan pool 借用及关闭顺序写在类型附近、guide 与公开 sample，Apple 保原普通调用。

当前仅 SCOPED；有限 FIX、API freeze、候选、运行 verdict、红队/LOC 在后续实际完成时填写。
