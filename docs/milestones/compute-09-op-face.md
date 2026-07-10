# compute-09 step-1：ggml_bridge 算子面扩展 —— 通用图执行器 + elementwise/norm 算子

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-05（ggml_bridge：桥接机制 + matmul 首消费者）|
| **状态** | ✅ **step-1 完成（2026-07-09，WSL GPU+CPU + MSVC，17/17 含 4 death test + 红队闭环）**。抽出通用图执行器 `elementwise_blocking`,把 matmul 那套建 ctx→gallocr→H2D→compute→D2H→free 样板写一次,新增 4 个算子(`rms_norm`/`silu`/`add`/`mul`)各只贡献一句建图。RTX 4070 CUDA + CPU 双后端对拍 host scalar 参考。**step-2(decoder block + 权重常驻)未做**。|

> **一句话**:compute-05 只暴露 matmul 一个算子("桥接机制证明")让 HYLE 显得像玩具。step-1 立**通用算子执行器**,证明"借更多 ggml 算子"是**一句建图**的成本,不是复制 50 行——为 step-2 的完整 decoder block 铺路。

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

## 明确不做（step-2 去向）

| 不做 | 去向 |
|---|---|
| decoder block 端到端 + 权重常驻 | **step-2**(GgmlBackend 有状态化 + weight buffer 隔离 gallocr) |
| rope/flash_attn/softmax/permute/cont | step-2(block 需要,签名已调研见任务 #100)|
| 通用算子 builder(暴露 ggml 图)| ⛔ 否决(泄漏 ggml 类型 + 过度设计);对外走高层 block API |
| 多维/per-row norm | step-2(多维张量)|

## 交付物

- `include/hyle/compute/ggml_bridge.h`(+4 算子声明 + friend + 契约注释;零 ggml 类型泄漏)
- `src/compute/ggml_bridge.cc`(+`elementwise_blocking` 通用执行器 + `EwOp`/`EwJob`/`ew_spawn` + 4 wrapper)
- `tests/compute/ggml_bridge_test.cc`(+`ElementwiseOpsDefaultDevice`/`ElementwiseOpsCpu` 对拍 host 参考 + 4 death test)

## 参考

- compute-05 里程碑(桥接机制 + 内存边界 + spawn_blocking + n_threads)。
- borrowing-ggml-memory-model.md(host 归 HYLE / device 归 ggml / 权重常驻 llama.cpp 模式)。
- ggml 算子签名:`.deps/ggml-src/include/ggml.h`(rms_norm:1371/silu:1185/add:889/mul:955)+ 实现 `ops.cpp:3813`(rms_norm 公式)。
- 任务 #100 调研报告(decoder block 9 算子集 + 权重常驻机制 + block API 形态)。
