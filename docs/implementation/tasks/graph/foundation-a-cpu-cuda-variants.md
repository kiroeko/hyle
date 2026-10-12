# Foundation A：CPU+CUDA 有限精确表接缝

本轮有限接缝已在[最终收口](foundation-a-final-closure.md)CLOSED；以下为阶段事实。

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

当前 AUDITED。有限 A-CUDA-01：有限 typed factory 与真实 consumer。
七门仍 UNPROVEN；原 H1 CLOSED 只证明旧已签范围。

审计冻结：Graph 的 kDeclaredDomain 必须保有全域 general artifact；此入口明确 exact-only。
拒绝非空 options.shape/exact_variants 或非 kExact 策略。返回共享 Executable handle，沿原
graph::Invocation 与 cpu_cuda::Invocation::create 取得独立 slots；无新 runtime/结果对象。
各点先共同 metadata 预检，再原 compiler 准备；缺任一点/重复签名拒绝整表，选择不编译。

原 CompiledStorageLimits/CompilationStorageLimits/InvocationStorageLimits 是 family-wide
合同，逐点调用不能把限额重复算成多个额度。本最小入口拒绝有限 compiled_storage、
compilation_storage 或 invocation_storage 配额，用户需使用已有共享 ResourceBudget；
共同表 metadata 也通过其 metadata_allocator 收费。原普通 compile/general 配额不变，
不宣称 ResourceBudget 覆盖未知 native 私有内存。对每点原 search caps 仍是每点合同，
总工作明确为有限表长度乘各点 cap，不叫原 general compiler 的累计搜索 family。

七问：原锁/许可证及 Provider KEEP；主/正交有限 shape 复合 cold glue SELF/BORROW；
共享求值和执行器；P0 冷 O(V(D+N+E)+V²D)、选 O(DV)、warm 原 native；C++ typed 无新 FFI；
三交付/缺路线/头实际验证待做；指南明示 finite-only、显式预算与原 native 限制。

Win source 首批 PASS：RTX4070 / CUDA13.3，两个有限精确形状、表外拒绝、CPU→CUDA→host
数学、旧 result 阻止复用，以及不将有限 compiler 配额按点分裂。原 compiler/Queue/Provider
没有新增执行路径；已通过不代表 WSL/installed/缺席/迁址或最终七门完成。

Candidate：546b0d2468e5de29031c34668f4afa4bc31b9a17；finite typed factory/自然消费者已固定，
原 budgets 与 failures 保持。剩余适用 WSL/三交付/缺席/迁址及终审需实际证据。

[Win static SDK](../../../../evidence/correctness/foundation-a-sdk-cpu-cuda-static/README.md) 已通过
普通 public consumer、新头、真正迁址、runtime和缺 Apple/Vulkan 诊断；shared/WSL/终局仍待验。

候选定向验证追加：Windows static/shared和WSL source/static/shared已全部完成真实RTX4070
finite/hole/CPU→CUDA→host/old-result/quota拒绝；WSL记录见
[source](../../../../evidence/correctness/foundation-a-wsl-cuda-source/README.md)、
[static](../../../../evidence/correctness/foundation-a-sdk-wsl-cuda-static/README.md)、
[shared](../../../../evidence/correctness/foundation-a-sdk-wsl-cuda-shared/README.md)。
共同ASan/UBSan42项、P0/LOC已签，原锁/失败预算保留；仅终局对证/红队与父卡一起裁决。
