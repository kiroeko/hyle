# SIMD v6：provider 升级与推理消费回放

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由当前产品事实定向重开。它不把“上游发布了新 commit”本身当作需求，也不重新打开
> 387-operation / 422-scope 的无界性能笛卡尔积。完成 `SCOPED` 与 `AUDITED`、冻结有限 FIX 集以前，
> 不修改 SIMD 产品源码、公开 API 或 provider 路由。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`simd` / SIMD /
  `simd-v6-provider-consumer-replay` / `REOPEN`；
- reviewed closure：`simd-v5-target-ownership`；candidate
  `d015b0de1cf6d5899da8336fef94104974bcd90e`；end
  `c3a8b0beb24a9bfce95f9117fc7df6f194b0e966`；product identity anchor
  `195aa9a3d2ce90db1ddb249a5319edb351224988`；
- replay / baseline：`b03fd9503885818a41f91375d761eed5d54238ff`，2026-09-06；
- reviewer：Codex；前态 registry snapshot SHA-256：
  `279C31AFA7238B416637EE3841574230D476BA077488E3CE4C279BBF6B1D587E`；
- trigger `SIMD-V6-DELIVERY-DOC-CONTRADICTION`：实际 CMake package registry、selected-component
  smoke 与 `samples/package/simd/cpp` 都把 `hyle::simd_expert`、`target.h`、`fat_kernel.h`、锁定的
  Highway/xsimd headers 交付给 relocated static/shared SDK；当前用户指南和构建架构也如此说明，
  但当前维护指南的“交付与负空间”却明确声称这些只存在于完整源码交付。相同产品的公开交付合同
  互相矛盾，会使真实消费者错误选择集成方式；
- affected-provider input：产品当前锁定 Highway
  `f503c485cec6b4f2f596fa60d83484797c9eeb5c`、xsimd
  `3b3151ff665514a00a437cc985f47b7de99b772b`，VQSort adaptation origin
  `83eff39371e5e6494b305b253f8a3c06f831faf4`。当前 Highway
  `18e449b642599b8f51c03ca10173fdc59864e4f2` 对 FastLog、NaN reduction、f16 math、ARM
  u8×s8 primitive 与 VQSort 有相关变化；这些只构成有限影响面审计输入，尚不预判全部进入产品；
- current consumer identities：ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`、llama.cpp
  `7620399f58aebfd2196b74021f9581bcf7218cb9`、DwarfStar/ds4
  `9ab705347c1775e7599ede7eb81a6255ec7dccb5`、vLLM
  `a1541f5742a29864a80087af313ad460066a1524`、SGLang
  `6cee9285a3dd43e1f0270818aef7bf01a0568863`。旧 v3 能力证据没有覆盖这组当前形状。vLLM/SGLang
  只补充推理 engine/custom-op/CPU-offload 的真实需求，不作为第三个 SIMD provider peer，也不把
  Python/Triton/CUDA surface 机械复制成 HYLE API；
- replay verdict：**`PRODUCT_TRIGGER`**。确定触发仅是交付文档矛盾；provider/consumer 差距在
  `AUDITED` 前可以裁决为 `FIX/DEFER/REJECT`，不能借此无界扩张 SIMD 或侵入 Compute。

## Closure card（`SCOPED`）

- scope：校正 source/static/shared 的 ordinary/expert 交付说明；比较产品锁定 Highway 与当前
  `18e449b6...` 后，只审计变化触及的 math/search/sort、target/provider codegen 和 VQSort
  adaptation；把当前四组推理证据（独立 ggml + llama.cpp、ds4、vLLM、SGLang）的需求归入
  ordinary SIMD、expert primitive、Compute operator 或 engine policy；对实际受影响的正常路径续签能力、
  复杂度与 P1；
- non-goals：不复制 Highway/xsimd 的完整寄存器 API，不恢复旧 register DSL，不为 benchmark 增加
  API/layout/metadata，不把 block-quant layout、GEMM、attention、RoPE、模型 sampler 或 fused tensor
  operator 错归 SIMD，不动未成熟 Compute/Graph，不建立 C/Native FFI 或跨工具链稳定 C++ ABI，不重写
  公共性能仪器，也不开放 422 scope × type × size × ISA 全组合搜索；
- direct dependencies / consumers：公开依赖保持 `core, platform`；Platform 是 CPU/OS facts 唯一 owner，
  SIMD 拥有 compiled/selected kernel policy。当前直接产品消费者是 Compute 的 private edge；ordinary
  与 expert package sample 是交付消费者。Container 只有出现具名可复用 kernel 时才建立真实边，不能
  因“可能更快”制造依赖；
- normal consumer workloads：连续 f32/f64 row 算术与归约、f16/bf16/fp8 搬运、AI activation 与稳定
  softmax、vocabulary argmax/top-k/ordering、scan/bitpack/RNG；长度覆盖量化 block/hidden row/vocabulary
  三个数量级。ggml/llama.cpp/ds4/vLLM/SGLang 的 block quant/dot、Hadamard、RoPE、MoE routing、
  paged attention 与 fusion 只验证 expert seam 或记录到 Compute/engine owner，不用不同合同 benchmark
  逼迫 ordinary API；
- capability peers：provider/portable primitive 使用 Highway
  `18e449b642599b8f51c03ca10173fdc59864e4f2` 与 xsimd
  `3b3151ff665514a00a437cc985f47b7de99b772b`；ordering 只使用同一 Highway VQSort origin/current
  对照。独立 ggml + llama.cpp、ds4、vLLM 和 SGLang 是 `REAL_CONSUMER` 差距输入，不冒充同合同强
  peer；每个能力族仍最多两个强 peer；
- tier / performance：`P1`，5% material non-inferiority margin。P0 先核对遍历数、复杂度、allocation、
  dispatch、target flags 和最终 provider 层数；P1 只跑本卡锁定的 affected family sentinels 与真实三档
  size，正反顺序和空白对照不合格时不得改 route；默认 Highway，汇编相同、差异小于 5%、地址/顺序/
  批次反转或有限测量仍不可裁决时都停止并选 Highway；
- machines / instruments：W1-WIN 为 P1 primary，W1-WSL 与 M1-MAC 为 target/correctness/delivery
  holdout；`closure-correctness-v1` + `closure-perf-v1`，复用现有 SIMD route/codegen/operation 工具，
  不在本轮改造公共 instrument；
- validation gates：ordinary correctness、特殊值/alias/tail、route ledger 387/422/0、有限 winner
  correctness/codegen、baseline SIGILL safety、exact/fixed 零 runtime selector、fat 每 whole-buffer 至多
  一次缓存函数调用、Windows/WSL/macOS、适用 sanitizer、source/relocated static/shared ordinary+expert+
  fixed+fat、license/dependency lock、当前直接 consumer、C++ 易用性/文档、旧 register/C/FFI 泄漏和红队；
- budget：总调查与施工最多 6 engineering hours，其中能力/provider/consumer 审计 2 小时、实现与
  correctness/delivery 2 小时、P1 与最多两次机制诊断 2 小时；最多三个 affected family group、两项
  算法/route 修改。超出即以 `BLOCKED/INCONCLUSIVE` 或明确 `DEFER` 终止，不扩展实验空间；
- expected output：capability-family map、provider/version affected map、`KEEP/FIX/BORROW/DEFER/REJECT`
  ledger、自然 ordinary/expert 边界、三交付合同、有限性能 verdict、候选与 terminal receipt；
- acceptable limitations：SVE/RVV 没有真实机器时只签 compile/correctness，不签 P1 SOTA；模型专用
  operator 留给 Compute/engine；Formal 保持独立 `PENDING`。

`SCOPED` 只冻结上述边界和预算。完成 capability/public-surface/direct-consumer 双向追踪并把所有发现
裁决进有限 ledger 后才能进入 `AUDITED`；当前仍未授权产品实现。

## 审计结果（`AUDITED`）

### Reference receipt

2026-09-06 在 `C:\public\project-hyle` 确认下列 checkout 均与各自 `origin` 对应的当前
HEAD 一致且工作树为 clean：

| 证据组 / provider | clean commit | 本轮作用 |
|---|---|---|
| standalone ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | 底座能力、CPU 行算子和复杂度形状 |
| llama.cpp | `7620399f58aebfd2196b74021f9581bcf7218cb9` | 模型图中的 RMSNorm、GLU、sampling 与 DSA top-k 真实旅程 |
| ds4 | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` | 紧凑 CPU/统一内存引擎的 row norm、clamped SwiGLU 和小-k expert route |
| vLLM | `a1541f5742a29864a80087af313ad460066a1524` | CPU activation/RMSNorm 与 runtime/custom-op 的 dtype/fusion 需求 |
| SGLang | `6cee9285a3dd43e1f0270818aef7bf01a0568863` | serving/kernel registry 中的 GLU、RMSNorm、MoE/sampling top-k 需求 |
| Highway | `18e449b642599b8f51c03ca10173fdc59864e4f2` | 直接 product provider 候选升级和受影响 API/codegen 审计 |
| xsimd | `3b3151ff665514a00a437cc985f47b7de99b772b` | 当前锁定独立 math peer；本轮无新产品差异 |

`llama.cpp` 随仓 ggml 身份由该 llama.cpp commit 自身锁定，不用 standalone ggml 的更新
commit 伪装 llama.cpp 已消费。vLLM/SGLang 主要提供 GPU/Triton/tensor 形状，只作
consumer/demand evidence；它们不是 ordinary CPU pointer/count 性能 peer。

### 四组推理证据与能力族裁决

| 能力族 | 四组证据中的真实 shape | HYLE baseline | owner / verdict |
|---|---|---|---|
| 连续 row algebra/reduction | ggml CPU vec 与各引擎 row kernel 均需 add/mul/scale/sum/dot/extrema | `ops.h`/`search.h` 已有一次 whole-buffer route | SIMD `COMPLETE/KEEP` |
| storage conversion | 三引擎反复在 f32/f16/bf16/fp8 storage 之间移动 | `dtype.h`/`dtype_ops.h` 覆盖公开 storage graph | SIMD `COMPLETE/KEEP`；block format 归 format/Compute |
| 基本激活 | ggml CPU 有 GELU exact/tanh/quick、SiLU；vLLM CPU 对 floating storage 提供同族；SGLang 有统一 kernel key | 只有 f32，且缺 quick GELU | SIMD `PARTIAL/FIX` |
| gated activation | ggml 有 split/packed ReGLU、GeGLU exact/tanh/quick、SwiGLU；vLLM CPU 有 `silu_and_mul`/`gelu_and_mul`；ds4 直接 clamped SwiGLU；SGLang 有同义 kernel | ordinary SIMD 缺失 | model-independent split row 归 SIMD `MISSING/FIX`；stride/thread/quant fusion 归 Compute |
| row normalization | ggml CPU 有 LayerNorm/RMSNorm；llama.cpp 多模型直接消费；ds4 有 weighted/unweighted/head RMSNorm；vLLM CPU 支持 floating storage；SGLang 有 RMSNorm registry | ordinary SIMD 缺失 | 连续单 row 归 SIMD `MISSING/FIX`；tensor/fused residual/group/model variant 归 Compute |
| stable softmax/log-softmax | 各引擎均使用，但常与 mask/attention 融合 | f32 单 row 已完整 | SIMD `COMPLETE/KEEP`；mask/online/attention 归 Compute |
| float score + original index top-k | ggml `ggml_top_k` 以 index scratch 做 partial sort；llama.cpp sampler/DSA 取 token/position index；ds4 `topk_desc` 返回 expert index；vLLM/SGLang 都交付 weight+id | float `partial_sort` 破坏 score 且丢 index，`argsort` 强制全排序与 O(n) scratch | 非破坏、有 index 的单 row top-k 归 SIMD `MISSING/FIX`；top-p/categorical/MoE policy 归 engine |
| full/stable ordering | ggml/llama 有 argsort，基础数值处理还需 full/partial/select | VQSort adaptation + stable radix/argsort 已完整 | SIMD `COMPLETE/KEEP` |
| block quant/dot/GEMV/GEMM | 四组证据都有 layout-specific/tiled/fused kernels | expert seam 可写 primitive，无 HYLE tensor/format owner | `ROUTED` 到 format + Compute；不进 ordinary SIMD |
| RoPE/attention/KV/Mamba/MoE fusion | vLLM/SGLang/ggml/llama/ds4 都有 backend/tensor 实现 | expert fixed/fat seam 可原型化 | `ROUTED` 到 Compute/engine |
| sampling/RNG policy | llama/vLLM/SGLang 有 top-k/top-p/categorical policy | SIMD 已有 caller-owned bulk uniform stream | random substrate `COMPLETE`；policy 归 engine |
| target/provider/expert SDK | 运行时需要 baseline-safe dispatch，自定义 kernel 需 fixed/fat target | ordinary/exact/fixed/fat 均存在，但维护文档错误声称 expert 只能 source | 产品 `COMPLETE`，delivery docs `FIX` |

具体证据位置为 ggml `include/ggml.h` 及 `src/ggml-cpu/{vec.cpp,vec.h,ops.cpp}`，llama.cpp
`src/llama-sampler.cpp` 和 `src/models/{deepseek4,glm-dsa,qwen4exp}.cpp`，ds4 `ds4.c`的
`rms_norm_*`/SwiGLU/`topk_desc`，vLLM `csrc/cpu/{activation,layernorm}.cpp` 及 MoE top-k 路径，
SGLang `python/sglang/kernels/ops/{elementwise,moe}` 与 batch-invariant RMSNorm 路径。

### Public-surface trace

| 已有公开面 | 所有者 / 现状 | 本轮处理 |
|---|---|---|
| `types.h`, `dtype.h`, `dtype_ops.h`, `narrow_ops.h` | HYLE storage/layout/semantic，Highway 是私有 lowering | `KEEP`；续签新 provider correctness |
| `ops.h`, `search.h`, `scan.h` | ordinary pointer/count row 算法 | `KEEP`；新 norm 复用同一 target substrate |
| `math.h` | real/complex/activation/softmax | `FIX`：补 current consumer 需要的 quick/narrow/gated/norm 自然 row 形状 |
| `sort.h` | destructive ordering + stable radix/argsort | `FIX`：补不破坏 score 的 typed top-k result，不改现有 sort 语义 |
| `bitpack.h`, `random.h` | canonical codec 与 caller-owned PRNG | `KEEP` |
| `runtime.h`, `target.h` | selected compiled target diagnosis | `KEEP`；Platform 只提供 facts |
| `fat_kernel.h`, `hyle::simd_expert` | 源码/static/shared 的 target-bound vendor primitive seam | 产品 `KEEP`，维护文档 `FIX` |
| provider operation manifest/routes | 387 operation / 422 scope / 0 pending 的 expert route 账本 | 只回放 Highway delta 触及项；新 whole-row composition 不伪装 register op |

所有 installed SIMD header 均已映射到上表；没有 C header、`extern "C"`、Native FFI 或恢复旧
register façade 的理由。

### Provider/version affected map

| upstream delta | 对 HYLE 的实际影响 | verdict |
|---|---|---|
| `math-inl.h` f16 限制与新 `f16_math-inl.h` | 已有 f32/f64 ordinary math 语义未改；新 narrow inference kernel 以 f32 accumulate/compute 后明确舍入 | `BORROW` 新 lock，不盲目改精度合同 |
| FastLog 重写 | HYLE ordinary `log` 使用 qualified `math-inl.h`，不使用 FastLog | `NOT_AFFECTED` |
| `ReduceMinOrNaN/ReduceMaxOrNaN` | 只返回 value，不能代替 HYLE first-NaN/index/tie 合同 | `KEEP` 现 search，expert 自然获得新 primitive |
| ARM u8×s8 `PerBlock2x2MatMul` | 是 matrix/tile primitive，不是 ordinary row API | expert 暴露，具体核归 Compute `ROUTED` |
| `WidenMulAccumulate` 扩展 | 旧 `ReorderWidenMulAccumulate` 仍存在；manifest 已分别记录两者 | `KEEP/BORROW` |
| VQSort current delta | HYLE 已独立回移 zero/one-key 与 zero-k underflow 修复；其余主要是 upstream base-case/debug/build 改造，不解决已复现产品缺陷 | 保留 bounded HYLE adaptation；本轮不做无证据整体 rebase |

### 冻结的有限 FIX 集

| ID | action | 边界 / 验证 |
|---|---|---|
| `SIMD-V6-F01` | 修正 expert SDK source/static/shared 矛盾，使 architecture、user、maintenance 和 package sample 一致 | 不改交付产品；跑 relocated static/shared ordinary+expert smoke |
| `SIMD-V6-F02` | 补 basic activation 的 quick/narrow storage 缺口，并提供 split-input ReGLU/GeGLU/SwiGLU row | f32/f16/bf16 storage，f32 compute，O(n)/O(1)，单次 whole-row dispatch；不包 tensor stride/thread/backward/quant fusion |
| `SIMD-V6-F03` | 提供 weighted/unweighted RMSNorm 与 affine/non-affine LayerNorm row | f32/f16/bf16 storage，f32 accumulation，O(n)/O(1)，无 heap/lock；fused residual/group/model variant 留 Compute |
| `SIMD-V6-F04` | 提供非破坏 f32 score top-k，返回 score+`uint32_t` original index | result storage 即 O(k) workspace；k==1/有界 MoE k<=8 为 O(n)，其余 O(n log k + k log k)，descending、NaN-last、tie 按原 index；不做 sampler policy |
| `SIMD-V6-F05` | 把直接 Highway product/bench lock 升到 `18e449b6...`，重签受影响 compile/correctness/codegen/license/package | xsimd 不动；VQSort integrated origin 不伪装成整体 rebase |
| `SIMD-V6-F06` | 新增三个 family sentinel 的正确性、复杂度、alias/special-value、三交付和 P1 证据，并更新 architecture/user/maintenance/performance 文档 | 仅三档 normal size；peer adapter 先对齐语义和计费边界，不用 apples-vs-oranges 结果改 API |

未进入 FIX 集的项目均已有去向：block quant/GEMM/attention/RoPE/Mamba/tensor fusion 记到
Compute/format/engine；GLU backward 是未来 training consumer trigger；无当前缺陷的 VQSort 整体同步
与 422-scope 重跑被 `DEFER/REJECT`。本轮不得再新增第四个能力族。

### Owner 七问（AUDITED 直答）

1. **第三方：**Highway 是直接 header-only product mechanism 与 expert delivery dependency；xsimd 是两个已锁定
   MSVC AVX2 math exception的 product mechanism；VQSort adaptation 是已登记 integrated third-party source。四组
   推理 checkout 只是 evidence，不是产品依赖。
2. **功能：**三个未处置 `MUST`已被 F02–F04 有限冻结；其余四组推理差距均已
   `COMPLETE` 或路由到具名 owner，没有第四个未处置 SIMD `MUST`。
3. **内部复用：**Core 给编译/断言 primitive，Platform 只给 CPU/OS facts；SIMD 保留 target/provider 唯一
   decision owner。新 row 数学复用同一 Highway target substrate，top-k 不为一个局部 heap 拉入
   Container/Mem。
4. **性能：**F02/F03 为 O(n)/O(1) 且不做多次公开 dispatch；F04 在 k<=8 与 ds4 的真实 bounded-k
   insertion 同为 O(n)，更大 k 使用 O(n log k)/O(k)，且不强迫 ggml 式 O(n) index scratch。P1 只在同语义、
   同输出、同 scratch 与同 ISA 下裁决；当前尚未运行，所以仍是 `PLANNED`。
5. **交付/C ABI：**source/static/shared 是相同 C++ 产品；expert provider headers 确实随 static/shared SDK
   交付。无 C ABI/Native FFI，不声称跨工具链稳定 C++ ABI。
6. **易用性/文档：**新 API 使用具义 dtype + operation 名、显式 pointer/count、epsilon、alias 与
   workspace 合同；不暴露 vendor/tensor/model 类型。F01/F06 尚待完成，所以现在不宣称通过。
7. **红队：**已确认最高风险是 narrow 舍入/特殊值、norm 零长/非法 epsilon/alias、top-k
   NaN/tie/index 上界/复杂度、baseline SIGILL、exact/fat codegen 和 relocated expert delivery；它们已全部
   进入 F02–F06 验证，尚未写成 `PASS`。

Audit verdict：**`AUDITED`**。capability-family map、public-surface trace 和有限 FIX 集已冻结；
从此只允许实施 F01–F06。任何新公开能力或第四 family 必须退回 `SCOPED/AUDITED`，
不得在 benchmark 中静默扩张。SCOPED registry snapshot SHA-256：
`459ABA732B440D57593B28DE12D817A68D24A37F2F42AD02B773F7DE99D513A3`。

## Candidate 施工进展

截至 2026-09-06，F01–F05 的产品实现已经落地，F06 已完成候选正确性和正式 P1 合同接线：

- ordinary API 已加入 quick GELU、f16/bf16 GELU/SiLU、f32/f16/bf16 split ReGLU/GeGLU/SwiGLU、
  weighted/unweighted RMSNorm、affine/non-affine LayerNorm，以及非破坏 f32 score+index top-k；
- activation/GLU/norm 都沿既有 per-target math body，一次 whole-row dispatch、O(n)/O(1)、无
  allocation/lock；top-k 的 bounded MoE `k <= 8` 为 fixed insertion，其余为 heap/full-sort；
- Highway product lock、benchmark checkout 和 route ledger 当前 provider identity 已统一为
  `18e449b6...`；387 operation / 422 scope / 0 pending 的机械检查通过，历史 `f503...` evidence
  仍保持历史身份，不被改写；
- Windows MSVC 19.50 的 shared-fat 与 static exact-AVX2 编译均通过，static exact SIMD suite 为
  5/5 PASS；package sample 已引用 GLU、norm 和 indexed top-k；
- 新 P1 合同 `simd-inference-row-p1-v1` 锁定 FFN=11008、hidden=4096、experts=256/k=6 三个
  完整推理 row，双侧对齐 storage、输出、scratch、ISA、allocation 和计费边界，禁止用不同语义行
  裁决产品；一次完整 row 调用是一个 measurement operation，元素数不冒充独立 callback batch；
- 旧 build tree 保留自动 `FETCHCONTENT_SOURCE_DIR_HIGHWAY` 的独立构建缺陷已记录到
  [`build finding`](../build/build-dependency-lock-cache-refresh-finding-2026-09.md)，不在 SIMD 轮
  顺手修改公共 resolver。

尚未签发 `CANDIDATE`：需要先形成干净 product commit/tree。之后只剩 W1-WIN 正式 P1、WSL/macOS
正确性与交付、relocated source/static/shared、适用 sanitizer、最终文档/红队与 terminal receipt；
若 P1 出现 material FAIL，最多执行本卡允许的一次机制诊断，不扩展成逐 operation 计时。
