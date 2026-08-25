# compute-09：ggml_bridge 算子面扩展 —— 通用图执行器 + elementwise/norm 算子（step-1）+ FFN block 权重常驻（step-2）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-05（ggml_bridge：桥接机制 + matmul 首消费者）|
| **状态** | ✅ **step-1 + step-2 完成**。**step-1（2026-07-09）**：通用图执行器 `elementwise_blocking` + 4 算子(`rms_norm`/`silu`/`add`/`mul`)。**step-2（2026-07-10，WSL GPU RTX4070+CPU + MSVC GPU+CPU）**：`FfnBlock` —— 一条完整 LLaMA 式 FFN 子块,**权重常驻显存**(`ggml_backend_alloc_ctx_tensors_from_buft`,构造时拷入一次),每次 forward 只喂激活。双后端对拍 host scalar 参考 + 权重常驻证明 + move 语义覆盖 + 2 death test。**step-3(attention:rope/softmax/permute)→ backlog #102**。|

> **一句话**:compute-05 只暴露 matmul 一个算子("桥接机制证明")让 HYLE 显得像玩具。step-1 立**通用算子执行器**,证明"借更多 ggml 算子"是**一句建图**的成本;step-2 用它拼出**真 FFN block + 权重常驻**——这才是 matmul 缺的真推理机制(matmul 每调用重拷两个操作数,真推理的权重必须常驻)。

## 为什么做（纠正一个原则误用）

**"算子门面按消费者驱动逐步加"这条原则被误用了**(用户 2026-07-09 指出):
- **正当场景**:成本高、设计有歧义、要不要都不确定的东西等消费者(如 close-02 序列化的 schema 编译器)。
- **误用**:一条推理链的标准算子集是**确定的、低成本的(bridge 内一句建图)、无设计歧义的**(ggml 都定义好了)。对这种"等消费者"是偷懒,不是纪律——而且**循环论证**:只暴露 matmul → 无法当任何推理消费者的底座 → 永远等不到消费者。
- **纠正**:**推理链本身就是消费者**。不该等虚构的下游,该按"一条真实 transformer block 的构成"驱动算子铺开。step-1 先立机制 + 4 个无权重算子,step-2 铺完整 block。

## 做了什么（step-1）

**通用图执行器 `elementwise_blocking`**(`src/compute/ggml_bridge.cc`):matmul 的 `matmul_batch_blocking` 里除中间那句 `ggml_mul_mat` 全是样板(建 no_alloc ctx → 建 tensor + set_input → 算子 → set_output → build_forward_expand → gallocr_alloc_graph → tensor_set H2D → graph_compute → tensor_get D2H → free)。执行器把它写一次,算子用一个 `EwOp` enum 选中间那句:

| 算子 | ggml 调用 | 语义 |
|---|---|---|
| `ggml_rms_norm_f32` | `ggml_rms_norm(ctx,a,eps)` | `x / sqrt(mean(x²)+eps)`(eps 在 sqrt 内)|
| `ggml_silu_f32` | `ggml_silu(ctx,a)` | `x / (1+exp(-x))` |
| `ggml_add_f32` | `ggml_add(ctx,a,b)` | 逐元素(residual)|
| `ggml_mul_f32` | `ggml_mul(ctx,a,b)` | 逐元素(norm-weight/FFN gate)|

**边界(step-1 刻意不做)**:
- **权重常驻**:这 4 个是**无权重**的 activation-shaped 算子(n 进 n 出),沿用 matmul"每次拷输入"模型。权重常驻(`ggml_backend_alloc_ctx_tensors_from_buft`,GgmlBackend 从无状态升级为持有独立 weight buffer)是 step-2 decoder block 的前置,那里才需要。
- **单行 rms_norm**:1D tensor,归一化整个 n。真实 block 的 per-row(每 token 各自归一化)是 step-2 多维张量的事。

## 验收

- 4 算子端到端(host in → ggml op on ggml VRAM → host out,经 spawn_blocking + gallocr)对拍**独立 host scalar 参考**,RTX 4070 CUDA + CPU 双后端。
- n=323(17×19,非 8/16/64 倍数)压 SIMD 尾部。
- 契约 death test:silu null dst / add n=0 / add null b / rms_norm eps<0 各 fail-fast("HYLE fatal")。
- WSL GPU+CPU 17/17 + MSVC 17/17。

## 对抗式审查（2026-07-09 红队 + 自审,4 项全修）

不跳 CHECKLIST 第 4 步(compute-07 教训)。算子语义逐个对照 ggml 源码通过(rms_norm 公式/eps 位置、silu、add/mul 无意外广播),但揪出 4 项:

| # | 类别 | 问题 | 修 |
|---|---|---|---|
| 1 | 测试 | 4 算子零契约 death test(matmul 有 6 个)| 补 4 death test |
| 2 | 测试 | `kN=320` 注释"非 SIMD 倍数"**错**(320=2⁶×5 被 8/16/64 整除,不测尾部)| 改 n=323 |
| 3 | 契约 | `eps<0` 会在 pool 线程触发 ggml abort(非 HYLE fail-fast)| 公开层加 `HYLE_FATAL_ASSERT(eps>=0)` |
| 4 | 测试 | elementwise 无并发用例 | 判定:同 `ew_spawn→spawn_blocking→per-fiber backend` 路径,matmul 的 `ConcurrentIndependentBackends`+TSan 已覆盖机制,不重复造近似 harness(登记,非补) |

**教训重申**:测试通过≠对抗式审查。#2 尤其打脸——自审时以为 320"够奇怪"没验算,红队一算 320 恰好整除所有 SIMD 宽度。

## 做了什么（step-2：FFN block + 权重常驻）

**`FfnBlock`**(`include/hyle/compute/ggml_bridge.h` + `.cc`):一条完整 LLaMA 式前馈子块,权重常驻显存。对输入 `x=[rows×d]`:

```
h    = rms_norm(x) ⊙ w_norm                 (RMSNorm eps,再乘 per-feature 权重)
gate = silu(h · w_gateᵀ)                     h[rows×d] · w_gate[ffn×d]ᵀ → [rows×ffn]
up   =      h · w_upᵀ                        → [rows×ffn]
out  = x + (gate ⊙ up) · w_downᵀ             down:[rows×ffn]·w_down[d×ffn]ᵀ → [rows×d]
```

全在**一张 ggml 图、一次 submit**(finding 4)。

**真架构点 = 权重常驻**(matmul 缺的真推理机制):
- matmul 每调用把两个操作数都 H2D 重拷;真推理里权重**跨 forward 不变**,必须常驻,每次只喂激活。
- `FfnBlock::create` 建一个 `no_alloc` 权重 ctx(只放 4 个 weight tensor 的元数据),用 `ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft)` 一次性分配**独立的常驻 weight 显存 buffer**,`ggml_backend_tensor_set` 拷入一次。此后 `forward` 只 H2D 激活 `x`、D2H 结果 `out`。
- **两条独立 device 路**:weight buffer(常驻,非 gallocr)vs 激活/中间张量(gallocr,每次 forward 复用)。这就是 llama.cpp 的模型。GgmlBackend 从无状态升级为 `FfnBlock` 持有 `wctx_`/`wbuf_`/`weights_`。
- weight tensor **不** `ggml_set_input`(那是 gallocr 激活语义);只有 `gx` set_input、最终 `gout` set_output(中间张量设 set_output 会破坏 gallocr 复用)。

**GPU fp32 matmul 精度(非 bug,已诊断)**:CUDA 后端用融合/降精度 matmul 累加,与 ggml-CPU 跑**同一张图**在大值上差 ~0.19%(实测 |v|~18 处 abs 3.5e-2),residual 里的相消把它放大到近零输出 ~1.5% 相对。→ 后端感知容差:GPU rel 3e-2/abs 2e-2、**CPU rel 2e-3/abs 1e-3(CPU 仍是形状正确性的严格闸门)**。不是"放宽到绿"——GPU 与 ggml-CPU 在大值上一致到 0.19% 坐实是真 GPU fp32 matmul 精度。

## 对抗式审查（step-2，2026-07-10 红队 subagent + 自审）

红队逐条 trace 了 create 的 5 个失败早返回(无泄漏/无双 free)、dtor free 顺序(weights→wbuf→wctx→gallocr→backend,ggml 所有权安全)、move-assign 的 `~FfnBlock()` idiom(平凡布局安全)、图张量预算(9 张 < 16)、mul_mat 的 ne0 契约(逐个匹配)、w_norm 广播、set_input/output、并发(各 block 独占 be/galloc/weights,无 global/static)、ref_ffn 与图数学逐位一致——**核心 1-9 类全部确认无 CRITICAL/泄漏/契约违反**。揪出 3 项:

| # | 类别 | 问题 | 修 |
|---|---|---|---|
| 1 | 编译（MSVC）| 类已 `HYLE_COMPUTE_API`,成员再标导致 C2487(dll 接口不一致)| 去掉成员上冗余的 `HYLE_COMPUTE_API`(与 GgmlBackend 一致)|
| 2 | 测试 | `create()` 契约 assert(d>0/ffn>0/eps>=0/weight 非空)无 death test | 补 `FfnBadDim`/`FfnNullWeight` |
| 3 | 测试 | move 语义无覆盖 | `run_ffn` 里 `FfnBlock mvd = std::move(*blk)`,所有 forward 走 moved-into block |

另发现一个 `-Wunused-parameter`(WSL GCC):`ffn` 传进 `ffn_forward_blocking` 后未用(所有 ffn-形状中间张量都从常驻 weight tensor 的 ne 推导)→ 删该参数。**双平台 warning 揪出的真死参,不是噪音**。

## 验收

- **step-1**:4 算子端到端对拍 host scalar 参考,RTX 4070 CUDA + CPU 双后端;n=323 压尾部;4 契约 death test。
- **step-2**:FFN block 端到端对拍 host `ref_ffn`(kD=32/kFfn=88/kRows=5,非圆整压形状);**权重常驻证明**(第二次 forward 换激活复用同一常驻权重,不 re-create);move 语义;2 契约 death test。
- **双平台**:WSL GPU RTX4070+CPU 21/21(含 6 death test)+ MSVC GPU+CPU 9/9(非 death)。

## 明确不做（step-3 去向）

| 不做 | 去向 |
|---|---|
| ~~decoder block 端到端 + 权重常驻~~ | ✅ **step-2 完成**(FfnBlock + `ggml_backend_alloc_ctx_tensors_from_buft`)|
| rope/flash_attn/softmax/permute/cont(attention 半块)| **step-3 backlog #102**(形状杂耍风险高,签名已调研见任务 #100;step-2 权重常驻机制可复用)|
| 通用算子 builder(暴露 ggml 图)| ⛔ 否决(泄漏 ggml 类型 + 过度设计);对外走高层 block API |

## 交付物

- `include/hyle/compute/ggml_bridge.h`（step-1: +4 算子声明 + friend + 契约注释；step-2: +`FfnBlock` 类，零 ggml 类型泄漏）
- `src/compute/ggml_bridge.cc`（step-1: +`elementwise_blocking` + `EwOp`/`EwJob`/`ew_spawn` + 4 wrapper；step-2: +`FfnWeights` TU 结构 + `ffn_forward_blocking` 一图 + `FfnBlock::create`/move/dtor/forward）
- `tests/compute/ggml_bridge_test.cc`（step-1: elementwise 对拍 + 4 death test；step-2: `FfnBlockDefaultDevice`/`FfnBlockCpu` 对拍 `ref_ffn` + 权重常驻证明 + move 覆盖 + `FfnBadDim`/`FfnNullWeight` death test）

## 参考

- compute-05 里程碑(桥接机制 + 内存边界 + spawn_blocking + n_threads)。
- borrowing-ggml-memory-model.md(host 归 HYLE / device 归 ggml / 权重常驻 llama.cpp 模式)。
- ggml 算子签名:`.deps/ggml-src/include/ggml.h`(rms_norm:1371/silu:1185/add:889/mul:955)+ 实现 `ops.cpp:3813`(rms_norm 公式)。
- 任务 #100 调研报告(decoder block 9 算子集 + 权重常驻机制 + block API 形态)。
