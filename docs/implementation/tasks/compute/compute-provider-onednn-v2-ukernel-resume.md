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

## AUDITED 结果与有限改动集（2026-09-12）

### 上游与 scratch prototype 事实

- 实际以 MSVC/Release 启用 `ONEDNN_EXPERIMENTAL_UKERNEL=ON` 后，C header 和
  `dnnl_brgemm_{create,finalize,generate,get_scratchpad_size,execute,destroy}` 均编译、静态链接并运行；
- full-shape f32 probe 对 `M={1,4,32}` 正常形状及 `7x129x257` holdout 与 scalar reference 一致；v1
  登记的六个 `M={1,4,32}` × `{4096x4096,5120x13824}` 形状均在一次 warm-up 后报告
  `warm_new=0`。x64 execute 源码同时确认 verbose OFF 时只向 caller scratch 写 batch element 并调用
  预生成 kernel，不存在 malloc/new/vector/shared owner；因此恢复条件不是 timing 推断；
- probe 的点值只用于机制预检，未资格化且受主频/系统负载明显影响，不形成 P1。它确认 raw full-shape
  ukernel 对 M=32 有达到 ordinary oneDNN primitive 同量级的窗口，而把 K/N 机械切成 64 小块会明显增加
  串行成本；不能把官方“小块原子”误解为所有 shape 都必须手工 K-block；
- oneDNN 文档明确 ukernel 是供调用方组合 blocking/threading 的 sequential operation，并要求多线程时
  A/B offsets 与 scratch 避免 cache-line sharing。x64 `execute` 为 const、只读 kernel artifact，全部 per-call
  可变 batch metadata 位于 caller scratch；同 artifact、独占 output tile/scratch 的并发路线成立；
- 当前 oneDNN `6b7f1fbbb18e` 仍保留相同 x64 C API 与无分配 execute 形状，降低精确 pin 的近期迁移风险，
  但 API 仍标为 experimental。它只在 provider 内私有使用，升级必须重新审计源码/符号/探针；
- AArch64 v3.13.2 及当前源码的 execute 都构造 `std::vector<brgemm_batch_element_t>`。本轮不 patch vendor，
  oneDNN factor 收窄到 x86-64；ARM 上 AUTO 关闭、ON 明确失败，Compute native 以及后续 MLX/Metal 路线
  继续提供产品能力。

### 有限 ledger

| ID | 问题与证据 | 裁决与允许改动 | 验收 |
|---|---|---|---|
| OD2-01 | ordinary primitive warm 创建 execution context/vector | `DELETE` engine/descriptor/primitive/stream/memory/threadpool bridge；`BORROW` x64 BRGEMM C ukernel | 产品源码不再调用 primitive execute；六形状 allocation probe |
| OD2-02 | raw ukernel 需要 caller-owned run metadata | plan 冷路径 create/finalize/generate；invocation 映射 workspace；execute 使用 batch=1 的 stack offsets | prepare/create-invocation 可失败；warm 零 HYLE/oneDNN allocation/JIT/copy |
| OD2-03 | 显式 executor 不能丢失 | concurrency=1 使用 full-N kernel；并行时冷路径把 N 划成至多 concurrency 个、正常宽度不小于 64 的 contiguous tile，只生成 full/tail 两种 kernel | 每 task 独占 64-byte scratch/output tile；同步返回、nested progress、并发差分 |
| OD2-04 | decode 已有更合适 native leaf | serial `M<16` 返回 `Unimplemented`，沿既有 ordered fallback 进入 HYLE native；并行 provider 仍可按 N tile 使用明确 executor | provider report/fallback 如实；M=1/4 不为 provider benchmark 强行绕路 |
| OD2-05 | AMX context 属执行线程 | 每个 executor task 在本线程 set hardware context，完成其 tile 后在同线程 release；错误用一个 atomic status 汇总 | 无跨线程 context teardown；失败不抛异常、不隐藏部分成功 |
| OD2-06 | AArch64 路线违反 blocker | CMake capability 收窄 x86-64，不保留 ordinary primitive 或 ARM ukernel 路线 | ON 诊断、AUTO 跳过；ARM header/native fallback journey |
| OD2-07 | vendor runtime/build 仍按旧 primitive 配置 | 私有 build 打开 ukernel，CPU runtime 改 `SEQ`，保留 MatMul-only/static/cache-off/concurrent-exec；static SDK 补 ukernel header 检查 | OFF 不下载；source/static/shared relocation、vendor export hygiene |
| OD2-08 | 实验 API 维护成本 | 精确 version assert/lock 不变；C API 不进入 HYLE header，升级触发源码与六形状重审 | shared 无 vendor symbol；用户只见既有 `onednn::provider()` |

这八项是全部产品改动；不增加公开类型、operation、属性或 benchmark-only route。

### API、所有权与成本冻结

公开 C++ journey 保持 v1 的 `onednn::provider({.executor=...}) -> Context -> prepare ->
create_invocation(workspace) -> submit`。`Options` 和 Compute provider SPI 不改形状：用户选择 provider，
冷路径按 capability/shape 允许 native fallback，steady path 不再选择。

- Context 只借用 `ParallelExecutorRef`；不再拥有 oneDNN engine；
- Prepared artifact 拥有最多 full/tail 两个不可变 ukernel handle、shape/tile 和每-task scratch stride；销毁时
  冷路径 destroy handle；
- Invocation 保留 prepared，并把 caller workspace 映射到终态；每个 task 的 scratch slice 与 output tile
  不重叠；tensor view 仍只借用到同步 Completion；
- prepare/JIT 为冷路径且可有 oneDNN allocation；create-invocation 只有既有 HYLE owner allocation/map；
  warm submit 为 O(MKN)，外加 O(task count) 的 stack/atomic dispatch metadata，零 heap、packing、copy、
  descriptor、stream、memory handle和 route search；
- output 在 backend failure 后可部分写入，沿既有 Compute submit failure 合同；executor callback 必须同步、
  infallible、noexcept 并支持 nested progress。

### P1 冻结

完整 provider wrapper 与 direct oneDNN C ukernel 使用相同预生成 artifact、shape/layout/precision、caller
scratch、task count、tile 和 completion；两侧同样 charge set/release hardware context 与全部 tile execute，
只把 create/finalize/generate、storage allocation 和 warm-up留在 setup。该层证明 wrapper tax 不劣于 5%。
另列 HYLE-native 与 provider 的 `M=1/4/32` 双顺序结果，用来验证冷 route 选择；不得把 native decode 胜利
和 provider prefill 胜负合并成一个平均值。A/A、known-delta、affinity/frequency、placement 与 provenance
未资格化则为 `INCONCLUSIVE/BLOCKED`，不允许再次改 API。

## Audit verdict

恢复事实已由实际构建、正确性、六形状 allocation probe 与精确源码共同验证；OD2-01--OD2-08 是有限、
不改变普通 Compute/Graph 语义的实现集。允许进入 `AUDITED` 并修改 provider/CMake/tests；任何 vendor
patch、ARM execute、公开 tiling knob、K-partition reduction、weight packing 或新 dtype 必须停下另立卡。

