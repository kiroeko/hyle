# Compute oneDNN provider v2：BRGEMM ukernel 恢复轮

## Closure card（CANDIDATE）

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

## 实现 checkpoint（2026-09-12）

- provider 产品实现已删除 engine/descriptor/primitive/stream/memory/threadpool bridge，改为 plan-owned
  x64 BRGEMM base/wide kernel、Invocation-owned caller scratch mapping 和每 task 的 stack offsets；默认串行
  `M<16` 在 prepare 时回落 native，显式 executor 按 N 维形成至多 concurrency 个均衡连续 tile；
- oneDNN 构建已改为 x64-only、`SEQ` runtime、`ONEDNN_EXPERIMENTAL_UKERNEL=ON`，仍保持精确
  v3.13.2 lock、static/MatMul-only/cache-off/concurrent-exec；static SDK 检查同时要求普通 C header 与
  `dnnl_ukernel.h`，shared SDK 继续禁止 vendor header 泄漏；
- 中央组件审计曾拒绝 provider 直接包含 Mem alignment header。实现改用固定 64-byte ukernel 私有算术，
  因而实际依赖仍为 `compute_provider_onednn -> compute`，没有为了两行算术扩大产品闭包；
- W1-WIN/MSVC Release 的 shared 定向 provider/header/Graph 为 3/3 PASS；覆盖 16-row serial ukernel、
  `7x17x257` 的四任务 base/wide 分片与 nested-progress executor、M=1/4 serial native fallback、transpose/
  empty fallback、workspace 缺失/不足/未对齐、metadata OOM、同 artifact 并发 Invocation 和 Graph scratch
  planning；static provider/header 为 2/2 PASS；
- static 产品专用、默认不随 ctest 自动执行的分配探针在完整 HYLE
  `OperationInvocation::submit()+Completion::wait()` 边界运行六个
  `M={1,4,32} x {4096x4096,5120x13824}` 形状，全部 `global new=0`，一次运行约 2.0 秒；这补强了裸
  ukernel prototype，不用 shared-DLL 符号解析伪造零分配；
- Compute package sample 已扩为真正 prepare/allocate-workspace/submit/verify 的 oneDNN MatMul consumer；
  selected-component smoke 仅在 native x64 强开 provider，native ARM 明确 OFF 并验证 HYLE-native 产品，
  不再让 x64-only factor 阻断 Apple Compute 交付。
- 正式 P1 已冻结为 `compute-onednn-provider-p1-v1`：同一进程内用独立但同值的 caller-owned storage、
  相同 `M=32,K=N=4096` row-major f32 shape、full-N 预生成 kernel、scratch、串行 task 与 hardware-context
  边界，对比完整公开 HYLE submit/wait 和 direct oneDNN C ukernel；双执行顺序、双构造顺序与代码布局
  扰动均为必需 strata。合同 schema、精确 v3.13.2 peer lock 与两个 benchmark target 已通过编译；一次
  unmanaged 冒烟中完整 HYLE 边约慢 3%，只证明仪器能运行且候选落在 5% 门附近，不是资格化结果，
  不进入性能总账。
- immutable candidate `71f9de0c` 的首次 runner 调用误带 `--formal`。W1-WIN profile 明确尚未预钉死
  Formal compiler/linker/SDK/std library 八项身份，因而 `build-contract` 正确 fail closed，bundle 只能为
  `DIAGNOSTIC`。该无资格 bundle 的 8/8 产品 strata 均为 `PARITY` 且非劣门 PASS，A/A、5%/10%
  known-delta、timer-floor 与 placement 也全部 PASS，但不能据此签发 P1。它没有形成产品 FAIL、没有改变
  candidate，也没有触发性能调参；依照 instrument v1 第 8、9 节，只允许保留原因并以普通 engineering
  `qualify-platform` 重做，不能把观察到的工具链身份事后登记为 Formal expected。
- 随后的普通 engineering qualification 绑定 clean source `5e078305`（产品字节仍为 `71f9de0c`）与 clean
  oneDNN `71094df45190`，在 W1-WIN/MSVC 19.50、Release/static/LTO-off 下于 n=15 按停止规则结束：8/8
  strata 的 median candidate/peer ratio 为 `0.9925--1.0134`，全部 99.26% CI 落在 5% 等价带内，combined
  verdict=`PARITY`、non-inferiority=`PASS`；A/A、5%/10% known-delta、timer-floor、placement、artifact
  identity 与 build contract 全部 PASS。有效 bundle 已严格重放为 `QUALIFIED/CURRENT/EXACT`，保存在
  [`compute-onednn-provider-p1-v1-5e078305-w1-win`](../../../../evidence/performance/compute-onednn-provider-p1-v1-5e078305-w1-win/)；
  这只签发 x64 serial prefill provider 边界，不继承到 decode、并行 executor 或其他 dtype/backend。

## Candidate checkpoint（2026-09-12）

- immutable product candidate：`71f9de0cce8e6dd93a1bf3b2c6eb8f1e1f1b2cdb`。它固定 provider API、
  x64 BRGEMM 算法、artifact/workspace layout、native fallback 与常态成本；后续 harness、状态、证据和
  文档修正不改变该产品 candidate；
- Windows/MSVC 19.50 的修正版 selected-component smoke 已明确探测 `AMD64`，三个 cache 都记录
  `HYLE_COMPUTE_ONEDNN=ON` 与 `DNNL_CPU_RUNTIME=SEQ`。source、relocated shared、relocated static
  consumer 均实际创建 oneDNN Context、准备 16x4x4 MatMul、确认 provider identity/no fallback 并验证
  数值，全部 PASS；static SDK 带 `dnnl.lib`、ukernel headers 与 oneDNN license，shared SDK 不泄漏
  vendor headers；
- shared-export hygiene 已增加 `dnnl_*` 负合同。实际 source build 的 12 个 HYLE libraries/1364 exports
  与 installed shared build 的 6 个 HYLE libraries/682 exports 均通过，无 HYLE C、mimalloc、oneDNN 或
  raw fcontext 实现符号泄漏；
- 正式 P1、Windows 定向 correctness/allocation 和 Windows 三交付已经完成，有限 OD2-01--OD2-08
  产品实现集没有未处理项，因此依法进入 `CANDIDATE`，不再以验证结果微调产品。

候选后的剩余验证现已完成；它们没有修改公开 API、layout、算法、常态 metadata、依赖版本或产品
源码。终局只补充候选绑定证据、状态和文档，不重新运行已经完成的 P1。

## 七问终局答案

1. **第三方路径**：唯一产品第三方是私有、锁定的 oneDNN v3.13.2
   `71094df45190`，Apache-2.0；lock、archive/hash、source cache、online/offline 和许可证投影均由中央
   dependency contract 管理。provider 仅在 x86-64 可构建，ARM 的 `AUTO` 诚实关闭，显式 `ON` 在
   configure 期给出不支持诊断；oneDNN 升级会触发源码、符号、六形状分配和交付重审。
2. **功能完整**：本 factor 精确拥有 fixed-shape、dense row-major f32 MatMul 的 x64 BRGEMM provider，
   覆盖真实 short-prefill 与普通 GEMM；串行 `M<16`、transpose、empty 和不支持形状沿同一个公开
   ordered-provider journey 回到 HYLE-native/SIMD。量化、f16/bf16、post-op、persistent packing、ARM
   provider 和 region partition 是已登记的独立后续能力，不被本 factor 冒充为已支持。
3. **内部复用**：直接复用 Compute 的 Tensor/Operation、Provider、PreparedOperation、Invocation、
   Completion、typed Status 和 native fallback，没有另建 tensor、queue、error、thread pool 或 runtime。
   私有 ukernel 所需固定 64-byte scratch 算术不泄漏为公共内存机制，也不为两行局部算术扩大到 Mem
   依赖。
4. **性能**：prepare 冷路径承担 capability、JIT、kernel generation 和 scratch sizing；warm submit 为
   O(MKN)+O(task count)，零 heap、packing、copy、JIT、descriptor/stream 创建和 provider search。
   W1-WIN 同一 oneDNN ukernel 的完整 HYLE wrapper 对 direct C ukernel 的八个 order/placement strata
   全为 `PARITY`，ratio `0.9925--1.0134`，99.26% CI 总范围 `0.9573--1.0485`，固定 5% non-inferiority
   全部 `PASS`；它只签 serial f32 short-prefill，不外推到 decode、并行或其他 dtype/backend。
5. **交付边界**：只有 C++ `onednn::provider()`；公开头不出现 oneDNN 类型，没有历史 C/Native FFI。
   shared export 实测拒绝 `hyle_*` C 入口、`dnnl_*`、mimalloc 和 raw fcontext 符号。
6. **构建交付**：Windows 与 Linux x64 的 source embedding、relocated shared、relocated static consumer
   都实际选择并执行 oneDNN provider；MacBook Air 与 Mac mini 的 source/shared/static 均验证 ARM
   feature-off native 产品，显式 feature-on 均按合同失败。static SDK 携带所需 ukernel headers/library/
   license，shared SDK 不泄漏 vendor headers/symbols；离线正负合同通过。
7. **易用与文档**：用户只需显式构造 provider、放入 Context、prepare、按报告申请 workspace 后 submit；
   provider 的 x64/f32/layout 范围、executor 借用寿命、64-byte workspace、同步完成、partial-output failure
   和 native fallback 均在 Compute guide 与公开声明附近说明，没有公开 tiling 或 benchmark knob。

## Terminal record（2026-09-12）

不可变产品候选是 `71f9de0cce8e6dd93a1bf3b2c6eb8f1e1f1b2cdb`。候选到终局的
`include/`、`src/` 与产品依赖配置差异为空；唯一 CMake 差异是 selected-component 测试脚本的 script-mode
host 探测修正，不改变 shipping target。适用验证如下：

| Gate | 候选绑定证据 | Verdict |
|---|---|---|
| scope | OD2-01--OD2-08 全部完成；未增加公开 operation、vendor 类型或测试入口 | `PASS` |
| architecture | `compute_provider_onednn -> compute` 唯一 HYLE 边；plan-owned immutable ukernel、invocation-owned run state/workspace、ordered native fallback | `PASS` |
| correctness | Windows/MSVC 与 Linux/GCC 14 provider/header/Graph、六形状 warm allocation probe；Linux ASan 使用默认 fat SIMD，UBSan/TSan 使用 exact scalar SIMD 且同一 x64 oneDNN JIT 路线，均零 report；同 artifact 并发 invocation 与 nested executor 通过 | `PASS` |
| delivery | Windows/Linux selected Compute source/shared/static relocated consumers通过；两台 Apple ARM 的三交付 feature-off 通过、feature-on 明确失败；实际导出、offline lock/license 正负合同通过 | `PASS` |
| usability_docs | Compute user guide、distribution 与本卡描述完整公开 journey、owner、workspace、fallback 和限制 | `PASS` |
| red_team | 有界复核 ownership、destroy order、同 artifact concurrency、tile/scratch 隔离、AMX thread context、OOM/overflow、partial output、vendor leakage 和 benchmark seam；未发现候选产品反证 | `PASS` |
| engineering_performance | P0 成本结构通过；`compute-onednn-provider-p1-v1` 为 `QUALIFIED/CURRENT/EXACT`，8/8 `PARITY` 且固定 5% non-inferiority 全过 | `PASS` |

代表性维护命令/receipt：

```text
cmake -DHYLE_SMOKE_COMPONENT=compute -P cmake/HyleCiSelectedComponentSmoke.cmake
ctest --test-dir <linux-targeted> -L compute_provider_onednn --output-on-failure
ctest --test-dir <linux-asan|ubsan|tsan> -R 'hyle_(compute_provider_onednn|graph)_test' --output-on-failure
cmake -P cmake/HyleCiOfflineDependencyContract.cmake
python scripts/hyle-perf.py verify --contract bench/contracts/compute-onednn-provider-p1-v1.json --manifest evidence/performance/compute-onednn-provider-p1-v1-5e078305-w1-win/manifest.json
```

Linux selected smoke 通过完整 source/shared/static relocated consumer；定向 provider/header/Graph 为
`3/3 PASS`，六形状 allocation 为 `1/1 PASS`，ASan/UBSan/TSan 的 provider+Graph 各 `2/2 PASS`。
MacBook Air `Mac16,12` 与 Mac mini `Mac16,10` 的 exact archive selected smoke 均通过三交付，且
`HYLE_COMPUTE_ONEDNN=ON` 都以 `unsupported for Darwin/aarch64` 明确失败。远程副本只用于验证，没有
反向覆盖 Windows canonical checkout。

公开限制：本 factor 不提供 ARM、quantized、f16/bf16、post-op、persistent packing、通用并行 reduction
或 region fusion；`dnnl_ukernel.h` 是精确 pin 下的私有 experimental dependency；Options 中 executor 是
borrow，必须活过 Context；每次 invocation 需要报告大小且 64-byte aligned 的 caller workspace；backend
failure 后 output 可以部分写入。只有 oneDNN 升级或 C API/source 变化、真实消费者需要本 factor 范围内
缺失的 dtype/layout/region、可复现 correctness/lifetime/export 问题、公开合同/算法/交付改变，或合格同
合同证据显示 material regression 才能重开。

`python scripts/loc-report.py --module compute_provider_onednn --commit 71f9de0c --self-check --json`
为 `PASS`：候选全库 `564,724` 行、自实现不含测试 `99,000` 行、本 factor 实现 `393` 行。七门全部
`PASS`，`compute_provider_onednn` v2 裁决为 **`CLOSED`**。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
