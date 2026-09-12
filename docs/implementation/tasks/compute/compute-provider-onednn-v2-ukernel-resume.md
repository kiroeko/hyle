# Compute oneDNN provider v2：BRGEMM ukernel 恢复轮

## Closure card（SCOPED）

- component / owner / round / kind：`compute_provider_onednn` / Compute /
  `compute-provider-onednn-v2-ukernel-resume` / `RESUME`；
- baseline：`d6a93086b4cdeb6b412f62c103ff9fb5d61f8902`；previous terminal：
  [`compute-provider-onednn-v1`](compute-provider-onednn-v1.md)；
- 恢复事实：锁定的 oneDNN `v3.13.2` 已包含默认关闭的
  `ONEDNN_EXPERIMENTAL_UKERNEL` 与公开 C header `dnnl_ukernel.h`。x64 BRGEMM ukernel 在
  `finalize/generate` 冷路径创建固定 JIT artifact，`get_scratchpad_size` 报告 caller buffer，warm
  `execute` 只在该 buffer 尾部填写 batch offsets 后调用已生成 kernel；verbose 关闭时源码没有 heap
  allocation。它是 v1 恢复条件要求的 caller-owned/precreated 等价成熟 route。相反，同版本 AArch64
  ukernel 的每次 execute 仍构造 `std::vector`，不能据此解除 ARM blocker；
- 正常 consumer 任务：x86-64 推理或工程用户为固定 dense f32 row-major
  `A[M,K] * B[K,N] -> C[M,N]` prepare 一次，在同一个 immutable artifact 上创建独占 invocation，
  使用 caller-owned 64-byte-aligned scratchpad 重复执行；
- 主纵切：ggml/llama.cpp、vLLM、SGLang 与 ds4 的 decode `M=1/4` 和短 prefill `M=32`；
  正交纵切为普通工程 GEMM；两者都只使用公开 Compute MatMul journey；
- scope：审计实验 C API 的版本/维护边界；以 scratch prototype 证明支持 shape、正确性、warm allocation
  和串行成本；冻结 HYLE-owned executor tiling、per-task ukernel/scratch 与 nested-progress 合同；决定
  x64-only provider capability 和 AArch64 native fallback；随后才可用 ukernel 替换 v1 ordinary primitive
  execute 路线，并重做三交付、导出、许可证和 P1；
- non-goals：公开暴露 `dnnl_*`、post-op/quantized/f16/bf16、persistent weight packing、oneDNN Graph、
  修改 Compute MatMul 语义、为 ARM 保留会 warm allocation 的 primitive/ukernel、Native FFI；
- 直接依赖 / 消费者：仍为 `compute_provider_onednn -> compute`；oneDNN 是私有、精确锁定的实现依赖；
  direct Compute 与 Graph 共用已有 Provider/PreparedOperation/OperationInvocation；
- tier / machine / margin：`P1`；W1-WIN 固定 P-core，以 direct oneDNN ukernel 作机制对照、HYLE provider
  作完整 API 路线，5% non-inferiority；Windows/WSL x86-64 做 correctness/package/sanitizer。Apple ARM
  只验证关闭或 fallback 的诚实交付，不签 oneDNN ukernel 能力或性能；
- 调查与纠正预算：C API/header、x64/AArch64 execute、CMake/export、threading/tiling 各扫描一次；一个
  scratch prototype；API/实现最多两轮，性能根因最多两轮，每个正式 workload 最多一次 qualification；
- 预声明验证：M={1,4,32} 的正常 shape、至少一个非整块 holdout、zero/overflow/unsupported、差分、
  同 artifact 并发 invocation、nested executor progress、workspace size/alignment/OOM、六个 v1 allocation
  probe shape、source/static/shared relocation、shared vendor export hygiene、offline lock/license、P0/P1、
  red-team、closure/docs/diff；
- stop：若 x64 ukernel warm path仍分配或隐藏 copy/JIT，实验 ABI 无法由精确 pin 私有化，HYLE-owned
  threading 必须改变公开 API/形成反向依赖，或同合同 prefill 不能达到 5% non-inferiority 且没有明确
  kernel/tiling根因，则回到 `BLOCKED`；不得恢复 ordinary primitive、放宽门或用 decode 胜利掩盖 prefill。

## SCOPED 审计清单

1. 用实际构建/符号和 allocation probe 验证 `ONEDNN_EXPERIMENTAL_UKERNEL`，不能只凭 header 宣称恢复；
2. 冻结串行 full-shape 与显式 executor 的 tile 数、边界 variant、每 task scratchpad 和 artifact ownership；
3. 证明同一 immutable ukernel 是否可并发执行；若上游没有合同，HYLE 必须按 task 拥有独立 artifact；
4. 明确 x64 feature detection、unsupported/fallback 与 AArch64 负空间；不得把“编译成功”写成 capability；
5. 盘点 v1 engine/stream/memory/threadpool metadata 的 `DELETE/KEEP`，避免同时维护两条同义路线；
6. 冻结有限 `FIX/BORROW/DELETE/DEFER` ledger、自然 C++ journey 和公平 P1 逐阶段 charging。

在上述审计完成并依法进入 `AUDITED` 前，不修改 provider 产品或 oneDNN 构建开关。

