# Foundation A：CPU+CUDA 有限精确表接缝

## Closure card

- component / owner：`graph_route_cpu_cuda`，仅冷 typed factory；Graph 共用精确表与原 compiler。
- baseline：636d61fce9ea266aca9218105daf924af66447c0。
- trigger：A-02 要求三 hybrid 的同一有限 exact coverage 旅程；原 CPU+CUDA 的
  declared-domain+exact variants 要求 general route，不能代表只覆盖两个点的原生表。
- consumer：同 [四路线卡](foundation-a-route-symmetry.md) 的有限 batch/资产矩阵形状。
- scope：inline `compile_variants` → ExactNativeVariants<graph::Executable>，每点调用原
  cpu_cuda::compile。原执行/结果、placements、effects、retirement 与 general compiler 保持原合同。
- non-goals：新 executor、Queue/Provider、运行时 JIT、隐式 padding/general、公共工具重写。
- 依赖/消费者：既有 Graph/compute_route_cuda/CPU；本轮新普通 CPU+CUDA shape consumer。
- P0：冷有限表/原 artifact，不给 submit 增查表或常态 metadata；无 P1。
- 预算：一轮静态审计、一 FIX；configure300/build1800/test120/consumer180 秒，失败不重置。
  重型门串行，库按可用内存 j4/j2/j1，test1。
- 验证：Win/WSL CPU+CUDA 定向 exact/coverage/旧结果/math source 与 installed static/shared；
  新头/缺 CUDA/迁址按实际 package seam；共用 shape guard/OOM 复用 A 共同证据；docs/closure/diff。
- stop：原 compiler 任一点无法准备即拒绝全表；无法完成适用门则 BLOCKED/INCONCLUSIVE。

当前 SCOPED，未写本 owner 产品代码。有限 A-CUDA-01：有限 typed factory 与真实 consumer。
七门仍 UNPROVEN；原 H1 CLOSED 只证明旧已签范围。
