# F1 共同租约核初验（Windows）

- 子候选：`db93f2e069ba5b22098d2fba21bc0604ece01a57`，compute owner AUDITED。
- 新入口 resource_leases.h：有界 LeasePool、ReadLease/WriteLease、跨pool LeaseUsePool/Batch。
- 复用原 RangeAllocator、ConcurrentBlockPool、RefCounted、ResourceBudget 与 CompletionRetention。
  无新 allocator、kernel、reaper 或执行队列；创建后控制容量固定。
- 基础共享/独占/区间回收、abandon、失败不发布、quota/OOM回退、拒绝attach保留ticket：
  [common.log](common.log)的69项通过；新测试5项，原测试64项。
- 真实 CUDA resource backing + 原increment helper + native Completion：
  [cuda.log](cuda.log)的29项通过；新测试1项，原shipping CUDA测试28项，实际写入结果42。
- 新resource_leases.h单独self-contained编译通过。完整compute头组在此exact factor配置中，
  原cuda_weights.h缺少SDK include上下文而失败；未写该组PASS，按整包正常route SDK上下文补验。
- 本次源码review修正read self-assignment和native回收尾部：drain后仍借用resource的callback访问
  纳入已有domain/usepool锁与计数协议；回收slot复用父级Mem机制。
  初次quota测试错误地给Budget自身1-byte控制额度；调整为可建budget且pool超额度，回退账单通过。

这是共同核初验，未验证typed readonly导入、Apple/Vulkan原生桥、自动Direct/Graph绑定、
整包三交付、WSL/Air、完整并发/红队及P1；不代表F1或goal已完成。
旧普通TensorView/Invocation布局未改，tensor()/view()是显式borrow，自动保活由后续leased适配兑现。

[统一LOC](loc-current.json)self-check：Compute+Graph非测试68,866行/303文件，
全库非测试164,093行/654文件。新头与共同核属于明确生命周期机制，不是新增算子。
