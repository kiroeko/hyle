# simd-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

> 本页是 `simd` 当前唯一 closure card。此前 `simd-01` 至 `simd-25`、旧
> `simd-module-closure.md`、旧 performance 表和 architecture 中的“终态/完成”只作历史事实与审计输入，
> 不授权产品修改，也不提供当前性能证书。机械状态只看
> [`module-status.json`](../../module-status.json)。

## Closure card（`AUDITED`）

- component / owner / round ID：`simd` / CPU SIMD owner / `simd-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`29b77897c0a14c92dd1baa49dccb41033dc835cd` /
  `278527f61607682396637d44d594e9dd65b75dd0`
- 进入原因与 demand/defect ID：
  - `SIMD-CAP-01`：38,024 行现有实现来自多轮 benchmark/peer 补齐，尚未按真实用户任务重新证明
    capability family 完整性；当前 guide 已明确留下 narrow sort 等未实现边；
  - `SIMD-API-01`：whole-buffer、runtime/exact/native/fat-kernel、D-tag 与 narrow surface 很宽，
    普通用户面、expert source seam 和 internal selector 需要重新分层，而不是把 387 个名字视作同类 API；
  - `SIMD-INFER-API-02`：AUDITED 后的 consumer challenge 证明首轮把 `ops.h` 的
    `add/dot/saxpy` 误写成 whole-buffer arithmetic 完整；llama.cpp/ggml
    `7584430716ee229751771ed0d6bbcb780d105eeb` 的实际 CPU row seam 与 Qwen3/Qwen3.5/3.6
    图要求普通 source/prebuilt 用户自然取得基础 elementwise/reduction，而不是被迫先写一个 fat-kernel；
  - `SIMD-PERF-01`：机械提取当前为 387 个 public/selector 名、344 个 benchable、314 个曾被 bench
    source 覆盖，但当前 report matrix 为 0；历史点值和“全部打平/SOTA”不能继承；
  - `SIMD-CORRECT-01`：scalar/SSE4/AVX2/AVX-512/NEON/SVE/RVV、partial/tail、mask、NaN、narrow、
    alias 与 dispatch safety 的证据来自不同旧轮，必须绑定同一候选；
  - `SIMD-DEP-01`：直接底层依赖是 Core/Platform；半成品 compute 调 whole-buffer，container 通过
    build-private `simd_vec_headers` 使用两个 target-bound kernel，修改公开/专家 seam 会影响真实边；
  - `SIMD-DOC-01`：architecture/user/history 仍混有旧 C ABI、过期 LOC、provisional 与互相冲突的
    “终态”叙事；
  - `SIMD-DELIVERY-01`：fat、exact、native/fat-kernel 与 source/static/shared 的实际可组装性尚未在
    当前 build/package 合同下统一验收。
- 真实 consumer task / normal workload：推理/工程计算对连续 tensor/buffer 做 mixed-precision dot、
  dtype 搬运、elementwise/activation/softmax、scan/search、sort/top-k/select、bitpack 与批量非加密随机；
  source 用户在固定 target TU 写有 partial tail 的 transform/gather/compress/widen-dot kernel；预编译
  用户发布一个 fat binary，已知硬件用户选择 exact target。基准 size strata 固定为适用操作的
  `n=31/4096/1048576`；sort 另用 `n=4096,k=10`，bitpack 用 `n=4096,bits=11/13`。不以旧 bench
  中存在的行决定 API 或 workload。
- 范围：`include/hyle/simd/**`、`src/simd/**`、`tests/simd/**`、`bench/simd/**`；simd component、
  exact/fat/native build contract、package/source delivery；Core/Platform 直接依赖；container 的两个
  build-private vec consumer 与 compute 的代表性 whole-buffer consumer；architecture/user/performance/
  maintenance 文档。下游模块只做受影响回归，不借本轮宣告其收口。
- 非目标：GPU/accelerator backend、矩阵乘/卷积/attention/model operator、完整 BLAS/FFT/linear algebra、
  container 的 UTF-8/probe policy、密码协议、C API/Native FFI、稳定 C++ ABI、任意编译器混用、per-ISA
  DLL、为了 API 数或 benchmark 点值新增常态 metadata、Formal release 证书。
- 直接依赖 / 直接消费者：`PUBLIC core,platform`；产品消费者为 `compute`（PRIVATE batch API，未完成）
  和 `container`（PRIVATE_BUILD `simd_vec_headers`，不拉完整 simd product）。不存在 math/error 直接边；
  是否应新增只能由 audit 中的真实公开 include/失败合同证明。
- tier 与理由：`P1`。batch kernel 与 target-bound primitive 预期高频，但尚无完成的推理引擎 profile
  把某行提升到 P2；本轮签正常 replay 的工程非劣，不宣称端到端 token/s。
- material margin / resource budget：同合同 5% non-inferiority；单线程采样固定最高 performance core，
  编译最多 4 jobs，不运行无关全库 benchmark，不并发两个性能 workload。小于 5% 的点差只有可归因机制
  且能传播到正常 consumer 才能驱动产品。
- 总调查预算 / 性能预算：6 小时 / 90 分钟。3 小时内必须冻结 capability ledger；若范围不能收成一个
  有限产品批次，记录具名 blocker/follow-up，而不是继续扩表。
- 最多复现、机制诊断、产品修改次数：每个 material finding 一次反序/复现和一次 codegen/profile
  机制检查；整个 frozen `FIX` 集一个合并产品批次；一次完整候选重跑。仪器问题单独登记，不在 SIMD
  轮次重写统计核心。
- machine profile：W1-WIN、W1-WSL、M1-MAC。W1-WIN 是 P1 qualification machine；M1-MAC 提供
  NEON 正确性与补充同合同数据；WSL 提供 GCC/sanitizer。AVX-512 SDE 与 SVE/RVV QEMU 只证明执行
  correctness/隔离，不伪装真硅性能。
- instrument profile：`closure-perf-v1`。
- strong peers（每项 capability 最多两个，均为本地只读 checkout）：
  - portable D-tag/native/dispatch/sort/bitpack：Highway
    `239071390b18ca70f8fadde90ebefa6d8b12cb69`；
  - vector transcendental：xsimd `3b3151ff665514a00a437cc985f47b7de99b772b` 与 SLEEF
    `7623d6cfa2712462880fa63a4d0f0b5f775d1a83`，先资格化精度合同再比较；
  - 推理 dtype/batch/dispatch 形状：llama.cpp
    `7584430716ee229751771ed0d6bbcb780d105eeb` 与 oneDNN
    `5dc67b7f37b1215d8c1f054e99c1b7cfd40317a9`，用于任务/算法形状，不做不同合同的点值胜负。
- 无同合同 peer 时的预算及禁止声明：以 scalar oracle、复杂度/bytes moved/分配/dispatch/codegen 和
  预冻结 consumer replay 裁决；不得声称未测 ISA、未测 dtype/op、整个模块或端到端推理 SOTA。
- 适用 correctness / delivery / performance gates：public-header self-contained；zero/null-if-zero、
  partial/tail/guard-page、alignment/alias、integer overflow、NaN/Inf/signed-zero/ULP/total-order、RNG state、
  sort/select/top-k/bitpack property/differential；runtime init reentrancy；ASan/UBSan；x86 scalar/SSE4/AVX2
  与 macOS NEON 真机，AVX-512/SVE/RVV 执行门；component/source/relocated static/shared、fat/exact/native
  consumer；container/compute affected regression；P1 A/A、known-delta、order/placement 与 5% NI；当前
  architecture/user/performance/maintenance 文档；独立红队。
- 预期输出：capability-family map；全部 public header/symbol 的普通/expert/internal trace；与 peer/
  north-star 的有限 gap ledger；冻结 `FIX` 集；不可变 candidate；三平台/backend/delivery/consumer
  manifest；有资格的分能力性能 verdict；当前 architecture/user guide；红队、LOC 与 terminal receipt。
- 停止条件 / 可接受限制：所有 `MUST` 为 `COMPLETE/ROUTED`，不存在 `PARTIAL + KEEP`；全部适用 gate
  有诚实 verdict。无真硅的 AVX-512/SVE/RVV 可以只保留 correctness 与明确的未签性能限制，但不得对它们
  宣称非劣；任何适用 P1 row 未运行、仪器无资格或 material regression 未解都使聚合性能
  `INCONCLUSIVE/FAIL` 并进入 `BLOCKED`。Native FFI 和模型算子保持 negative space。

## SCOPED baseline facts

- immutable baseline LOC：全库 306,880；自实现含测试 124,275；自实现不含测试 83,119；SIMD 实现
  38,024（108 files；`include/hyle/simd + src/simd`）。LOC 只描述规模，不证明缺口或完成度。
- 产品外部语法消费：compute 的 `add/dot/saxpy/exp`；container 的 bitset/string target-bound vec
  kernel。其余调用是 tests/bench/docs。
- `scripts/simd-op-coverage.py` 在 baseline 自检通过：344 benchable 中 source 覆盖 314（91.3%），
  30 个均已分类；但没有当前历史 matrix row，所以只能作为 audit inventory，不能继承 performance verdict。
- 当前 C++ guide 自报 narrow sort 未实现，architecture/user 顶部自报 `NOT_REVIEWED`；这是 audit
  seed，不是已经冻结的 `FIX`。

## AUDITED revision 2：推理 row seam 校正

用户在首轮 `AUDITED` 后指出历史“易用性补齐”没有真正对账推理消费者。该 finding 不改变冻结的
组件范围或 `matrix/attention/model operator -> compute` 边界，但击穿了 `SIMD-OPS-01 = COMPLETE`
的审计结论，因此本轮在写 candidate 前重新执行受影响的 top-down/bottom-up audit：

- 同层 peer 不是 `ggml_op` 的张量枚举，而是 llama.cpp 当前提交的
  `ggml/src/ggml-cpu/vec.h`：机械提取为 95 个 typed `ggml_vec_*` 名；f16/f32 重复、copy/fill、
  backward 与 fused model expression 不能机械变成 95 个 HYLE API；
- Qwen3 dense decode 的基础 row 需求是 residual add、gate multiply/scale、RMS reduction、activation、
  softmax、argmax/top-k 与 mixed/quantized projection。Qwen3.5/3.6 又加入 sigmoid gate、SSM/conv 与
  gated normalization；其中连续 elementwise/reduction 是 SIMD 基础能力，tensor stride/threading、
  RMSNorm/GLU 的 operator fusion、RoPE/SSM/attention、quant block repack/dequant-dot/GEMM 属 compute；
- `std::copy/fill` 已有成熟编译器/runtime 路径；模型无关但 performance-sensitive 的自定义融合继续走
  `native::transform + hyle_add_fat_kernel`。普通用户不应为基础 `subtract/multiply/divide/scale/sum`
  和一元代数重写该机制；
- 因而保留“SIMD 是连续 row/buffer 基础层，不是第二个 ggml tensor runtime”的边界，同时撤销
  “现有 whole-buffer arithmetic 已完整”的错误结论。以下 map、ledger、FIX 集与 workload matrix
  是 revision 2 的冻结审计输出。

## Capability-family coverage map

| Family | admitted user journeys / evidence | variants and boundary | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| Target selection and delivery | portable fat binary, known-target exact build, source TU fixed target; current package contract and Highway dispatch model | process-wide immutable selection; no per-lane dispatch; source/static/shared C++ | `dispatch.h`, `runtime.h`, `target.h`, `fat_kernel.h` | `COMPLETE` | no per-ISA DLL, stable cross-toolchain ABI or Native FFI |
| Register/D-tag kernel construction | expert quantization, codec, filter and transform kernels; Highway portable-register family | fixed/scalable tag, partial/tail, mask, load/store, arithmetic, conversion, swizzle, gather/compress, crypto primitive | `vec.h` plus one selected `backend/*.h` | `COMPLETE/ROUTED` | missing convenience compositions are expressible with one existing primitive composition; protocols are not owned here |
| Native source algorithms | caller-authored fixed-target loops without reimplementing tail/control flow | unary/binary transform, generate/replace/find/count/copy-if, scan/search/random; exact alias contracts | `native/*.h`, `native.h`, `hyle_target_simd` | `COMPLETE` | runtime callbacks are rejected because they prevent inlining/vectorization |
| Whole-buffer arithmetic and mixed dot | Qwen residual/gate/scale and row reduction; ggml CPU vec seam; current compute add/dot/saxpy | f32/f64 base unary/binary/scalar algebra and sum; f16/bf16/i8/i16 dot; zero/tail/exact alias | `ops.h`, `native/ops.h`, runtime/exact dispatch | `PARTIAL` | ordinary API lacks subtract/multiply/divide/scale/add-scalar/sum and abs/negate/square/sqrt; tensor layout, fused normalization/GLU, GEMM and quant block policy belong to compute |
| Storage dtype and narrow numeric primitives | inference storage conversion, narrow lane arithmetic/search/reduction | f16/bf16/fp8 scalar and batch conversion; f16/bf16 D-tag arithmetic/compare/total order | `dtype.h`, `dtype_ops.h`, `narrow_*.h` | `COMPLETE` | narrow model activations and top-k remain compute-policy routes until a concrete storage/rounding journey is admitted |
| Real/complex math and AI elementwise | current compute exp plus Qwen/ggml GELU/SiLU/sigmoid/softmax row shapes | f32/f64 scalar-oracle accuracy, SoA complex; elementwise exact alias; softmax in-place or out-of-place; no hidden allocation | `math.h`, per-target math kernels | `PARTIAL` | GELU/sigmoid/SiLU allocate O(n) hidden storage and compose multiple passes; softmax/log-softmax are not target-bound and force in-place storage |
| Search and extrema | engineering find/count/min/max plus inference logits argmax; Highway `IndexOfMin/Max` and llama/ggml argmax | all ordinary scalar lane types; first tie; empty returns `n`; floating NaN propagation must identify first NaN | `search.h`, `native/search.h`, runtime table | `PARTIAL` | value extrema exist, extrema indices do not |
| Scan | prefix processing and current expert source kernel | inclusive/exclusive, in-place/out-of-place, scalar/ISA fallback, overflow as lane modular arithmetic | `scan.h`, `native/scan.h` | `COMPLETE` | segmented/business-policy scans are higher-layer algorithms |
| Full/top-k/select sort | engineering sort and inference score/top-k/MoE routing | integer/floating/KV/U128, ascending/descending, NaNs-last ordinary float order | `sort/partial_sort/select` and target VQSort | `COMPLETE` | stability is deliberately separate |
| Stable radix and permutation | stable numeric ordering and argsort without moving payloads; existing public promise plus standard stable ordering | caller-reused scratch and allocating convenience; u/i16/32/64, f32/f64; first equal key preserved; 32-bit index bound explicit | current `radix_sort/argsort` | `PARTIAL` | strict-aliasing UB; OOM violates stable/noexcept contract; hidden allocation only; type family and index bound incomplete |
| Bit packing | quantized integer block storage and transport | u8/u16/u32, 1..lane bits, fixed byte layout independent of scalable VL, non-overlap | `bitpack.h` | `COMPLETE` | compression formats/protocol framing belong to their owners |
| Bulk non-cryptographic random | simulation/init/shuffle source data | explicit state, deterministic stream, fill u64/f32/f64, jump/split ownership | `random.h`, `native/random.h` | `COMPLETE` | entropy and cryptographic RNG are out of scope |
| Tensor/model/GPU operations | Qwen3/Qwen3.5/3.6 north-star inference engine | strided tensor scheduling, RMSNorm/GLU fusion, matrix/attention/RoPE/SSM/quant block/device ownership | compute or accelerator backend | `OUT_OF_SCOPE/ROUTED` | SIMD supplies reusable contiguous row/buffer kernels and custom fat-kernel seam, not a second ggml tensor runtime |

### Public-surface trace

| Header/surface | tier and reverse demand | audit verdict |
|---|---|---|
| `types.h`, `dtype.h` | ordinary public storage types/conversion oracles | `KEEP`; value-only, trivially-copyable types, no C/FFI promise |
| `dispatch.h`, `runtime.h`, `target.h`, `fat_kernel.h` | ordinary diagnostics/control plus expert fixed-target delivery | `KEEP`; Runtime is C++ type erasure/diagnostics, not a stable ABI; table version changes with search repair |
| `vec.h` | expert D-tag DSL; 387 deduplicated public/selector names across eight parsed headers | `KEEP/ROUTE`; 344 benchable names and 30 classified non-independent names are a mechanical inventory, not 344 independent product demands |
| `backend/*.h` | target-selected implementation included by `vec.h` | `INTERNAL`; users do not include a backend directly; duplicate selector names are not extra public journeys |
| `native.h`, `native/*.h`, `ops.h` | expert source loops and ordinary whole-buffer kernels | `PARTIAL -> FIX`; expert composition is sound, but ordinary basic algebra/reduction is materially incomplete for the inference row seam |
| `dtype_ops.h`, `narrow_float.h`, `narrow_ops.h`, `narrow_algorithm.h` | ordinary batch conversion plus expert narrow lane algorithms | `KEEP`; narrow model operators/sort are documented routes, not silent promises |
| `math.h` | ordinary real/complex batch and inference activation API | `PARTIAL -> FIX` for the six AI entry points; other math stays under existing accuracy/property gates |
| `search.h` | ordinary whole-buffer find/count/extrema/order | `PARTIAL -> FIX` with `index_of_min/index_of_max` for the existing type matrix |
| `scan.h` | ordinary whole-buffer scan | `KEEP`; native body and dispatched wrapper share one algorithm |
| `sort.h` | ordinary sort/top-k/select and explicit stable radix/argsort | `PARTIAL -> FIX` only for stable radix/permutation ownership, failure and coverage; VQSort surface stays unchanged |
| `bitpack.h` | ordinary stable-layout pack/unpack | `KEEP`; format and overlap contract remain explicit |
| `random.h` | ordinary state-owned bulk PRNG | `KEEP`; not an entropy API |

The baseline `scripts/simd-op-coverage.py --check` inventory is self-consistent: 387 deduplicated
register/selector names, 344 benchable, 314 represented in source parity benches, and all 30 remaining
names classified. This is the bottom-up trace for `vec.h`; it does not grant each spelling an independent
benchmark row or make current performance claims.

## Capability / gap ledger

| ID | class | exact source / normal shape | ownership, failure and resource target | coverage | action / severity | verification / negative space |
|---|---|---|---|---|---|---|
| `SIMD-DISPATCH-01` | `MUST` | existing fat/exact/native consumer journeys; current build/package contract | Platform owns CPU facts; SIMD owns immutable selection; one batch-boundary indirect call, no per-element metadata | `COMPLETE` | `KEEP` | source/static/shared, fat/exact/native consumers and ISA isolation |
| `SIMD-REG-01` | `MUST` | current container target-bound consumers and north-star custom kernels | caller owns buffers/alias; compile-time target; O(n), zero allocation | `COMPLETE` | `KEEP` | header/backend/property/codegen matrix; representative same-TU Highway rows, not name-count SOTA |
| `SIMD-REG-02` | `GENERAL` | current Highway adds `MaskedMulSub*`, `Slide*Or`, `Lookup32/64` convenience | register composition or small-table policy; no current consumer | `ROUTED/PARTIAL` | `REJECT` from v1 product batch | masked/slide forms use `IfThenElse` plus existing ops; admit LUT only with a named codec/dequant journey and second mature source/material mechanism |
| `SIMD-INTDIV-01` | `GENERAL` | Highway current all-width/floor invariant division; HYLE existing i/u32 `Divisor` | reusable parameter object; zero divisor fail-fast; no current product consumer | `PARTIAL` | `DEFER` | i/u32 remains documented; all-width/floor requires a separately admitted consumer/second source, not peer parity alone |
| `SIMD-OPS-01` | `MUST` | Qwen residual/gate/scale/row reduction; llama.cpp ggml CPU vec seam; oneDNN binary/reduction shapes | pointer/count; output may exactly alias an input; O(n), O(1), zero allocation/lock | `PARTIAL` | `FIX` / capability+usability | retain add/dot/saxpy; add f32/f64 subtract/multiply/divide/scale/add-scalar/sum and abs/negate/square/sqrt across native/exact/runtime; standard copy/fill stays standard-library route |
| `SIMD-DTYPE-01` | `MUST` | llama/oneDNN storage conversion and quantized inference | scalar oracle + dispatched batch; explicit rounding/special values | `COMPLETE` | `KEEP` | exhaustive fp8, sampled f16/bf16, hardware/fallback and representative P1 conversion rows |
| `SIMD-NARROW-01` | `GENERAL` | Highway f16 sort/math and llama narrow activations; HYLE first-class storage dtype | compute owns tensor storage/accumulation/rounding policy; users can convert explicitly | `ROUTED` | `DEFER` | no v1 narrow sort/activation API; reopen when a concrete tensor journey cannot use conversion/fusion without material extra traffic |
| `SIMD-ACT-01` | `MUST` | reproducible current source plus llama CPU GELU/SiLU/softmax shapes | elementwise exact alias; softmax exact alias or disjoint output; `n==0`; O(n), O(1) extra storage, no heap/lock; scalar tail preserves stated accuracy | `PARTIAL` | `FIX` / architecture+performance+OOM+usability | target-bound fused GELU/tanh-GELU/sigmoid/SiLU and vectorized softmax/log-softmax; add out-of-place softmax overloads; allocation probe, property and qualified consumer timing |
| `SIMD-INFER-ROUTE-01` | `MUST` | Qwen3/Qwen3.5/3.6 graph and ggml tensor/operator layer | tensor owns strides/layout/workspace/threading; compute owns fusion and quant block formats | `ROUTED` | `KEEP` / architecture boundary | RMSNorm/LayerNorm/GLU fusion, RoPE, attention/SSM, quantized GEMM/repack remain compute/private until their own closure; SIMD custom source seam must be sufficient to prototype them without public model types |
| `SIMD-SEARCH-01` | `MUST` | Highway `IndexOfMin/Max`; llama/ggml argmax | borrowed input; first tie; empty=`n`; first NaN for floating; O(n), O(1), no allocation | `MISSING` | `BORROW/FIX` / capability | all ten existing search types, runtime/exact/native routes, differential/property and Highway finite same-contract row |
| `SIMD-SORT-01` | `MUST` | reproducible strict-aliasing and OOM/stability defects in existing public stable API | caller owns keys/output; scratch borrow or allocating convenience; no hidden allocation on scratch path | `PARTIAL` | `FIX` / correctness+usability | generic object-preserving digit scatter, overlap/capacity/index boundary, sanitizer/property/OOM-shape tests |
| `SIMD-SORT-02` | `MUST` | existing sort scalar family and stable permutation journey | u/i16/32/64 and f32/f64; float finite order with all NaNs stable-last; 32-bit index means `n<=UINT32_MAX` | `PARTIAL` | `FIX` / API completeness | fill missing radix/argsort overloads; explicit workspace query/span overload; baseline/std qualified P1 rows |
| `SIMD-SCAN-01` | `MUST` | engineering prefix workload and existing public promise | caller buffers; modular integer semantics; O(n), zero allocation | `COMPLETE` | `KEEP` | property/tail/backend plus one family P1 row |
| `SIMD-BITPACK-01` | `MUST` | quantized storage `bits=11/13`, fixed format | non-overlap; exact byte count; O(n), zero allocation | `COMPLETE` | `KEEP` | layout oracle, scalable-VL invariance and one family P1 row |
| `SIMD-RNG-01` | `MUST` | bulk non-crypto initialization/simulation | caller-owned state; deterministic stream; O(n), zero allocation/lock | `COMPLETE` | `KEEP` | state/property and one family P1 row |
| `SIMD-DEP-01` | `MUST` | central component graph and real includes/calls | only `PUBLIC core,platform`; container private vec seam; compute ordinary batch seam | `COMPLETE` | `KEEP` | graph/target agreement and affected consumer builds; no error/math/mem dependency added |
| `SIMD-DOC-01` | `MUST` | current architecture/user pages contain provisional, old C ABI/peer/LOC and historical terminal claims | architecture owns durable decision; user guide owns current selection/contracts; history stays implementation-only | `PARTIAL` | `FIX` / usability | stale-term scan and docs checker; rewrite instead of extending stage chronology |

### Frozen `FIX` set

Exactly one combined product batch is authorized:

1. replace the six AI entry points with allocation-free target-bound kernels; fuse elementwise
   arithmetic with the chosen per-target transcendental, vectorize softmax/log-softmax reductions,
   and add out-of-place overloads while preserving exact in-place use and the public numerical contract;
2. add first-tie `index_of_min/index_of_max` to ordinary and native search, all existing scalar lane
   types, exact/runtime dispatch and Runtime table version;
3. rebuild stable radix/argsort around caller-owned typed scratch so no path aliases an object's
   representation; provide throwing allocating conveniences, non-allocating `span` overloads and the
   complete u/i16/32/64 + f32/f64 stable permutation family, with explicit 32-bit index limit and
   stable-last NaN semantics;
4. complete the ordinary f32/f64 contiguous algebra/reduction foundation in `ops.h` and the native,
   exact and runtime routes: subtract/multiply/divide, scale/add-scalar, sum, abs/negate/square/sqrt;
   keep copy/fill in the standard library and keep tensor/fused-model policy in compute;
5. update affected tests/bench/build/package samples and replace the provisional SIMD architecture,
   user and performance narratives with the current ordinary/expert/internal boundary and honest
   per-family evidence.

No integer-division width expansion, narrow tensor operator/sort, Lookup32/64, arbitrary peer-name
parity, tensor/GPU/model operation, fused normalization/GLU/quant-block kernel, C/Native FFI,
stable ABI, central instrument rewrite or unrelated
component change is authorized. A new material finding returns the round to `SCOPED/AUDITED`; it is
not appended while implementing.

### Frozen P1 workload matrix

`n=31` is a correctness/tail and dispatch-overhead check, not a timing verdict. Qualified W1-WIN
timing uses randomized block order and the shared instrument on the following normal rows; cache-scale
uses `n=4096`, streaming uses `n=1048576` where listed:

- same-TU expert kernel: transform plus gather/compress/widen-dot at `4096`, compared with Highway;
- whole-buffer: f32 add/multiply/divide/scale/sum/dot and f16/bf16/i8 dot at `4096` and applicable
  streaming size; basic unary/binary families use one representative qualified peer row each rather
  than one benchmark per spelling;
- dtype/math: f32↔f16/bf16 conversion, f32 exp, tanh-GELU, SiLU and in-place softmax at `4096`;
- search/scan: f32 count-equal, f32 argmax and u32 inclusive scan at `4096` and streaming size;
- ordering: f32 full sort, `partial_sort(n=4096,k=10)`, scratch u32 radix and f32 argsort at `4096`;
- bitpack/random: u16 `bits=11/13` and f32 fill at `4096`.

Only semantically and numerically qualified peer rows receive the 5% non-inferiority verdict. Other
rows use the pre-frozen candidate-vs-baseline resource/throughput budget and are labelled
`NOT_COMPARABLE`, never silently counted as a peer tie. One family row failing materially after the
allowed mechanism diagnosis blocks the aggregate P1 verdict; 314 historical source-bench names are
not promoted into current claims.

## Candidate

尚未产生。

## Verification manifest

尚未执行；SCOPED baseline inventory 不是 gate `PASS`。

## 独立红队

只在 candidate 冻结后执行，30 分钟上限。

## Closure decision

本轮从 `AUDITED` 终止为 **`BLOCKED`**，没有产生 product candidate，也没有执行或继承任何 P1
性能 verdict。阻断事实不是某个 SIMD 算法未知，而是冻结施工单无法在自己的预算和停止规则内诚实执行：

- 一个 combined batch 同时修改 activation/transcendental、search/runtime ABI、stable ordering、普通
  row arithmetic 和全部当前文档，明显超过“超过一个工作日先拆分”的公共方法；
- 聚合 P1 matrix 又把相互独立的 backend、dtype/math、search/scan、ordering、bitpack/random 绑成同一
  terminal gate，历史上已经导致超过 72 小时的参数/仪器重跑；
- 已出现 search、sort、activation 三组未提交草稿，但它们没有共同 candidate、完整 gate 或可继承的
  性能证据。继续实现只会让新的产品事实先于可执行合同。

blocker owner：`simd/simd-epic-control-plane`。

恢复条件：建立一个仍以 `simd` 为单一公共产品、但由有界顺序 ticket 执行的 RESUME round；每张票冻结
自己的路径、consumer、正确性与少量 P1 family 行，公共仪器不可在产品票内改写；现有草稿必须先得到
`KEEP/REWORK/DROP` 归属。恢复轮的总账见
[`SIMD/Math epic`](simd-epic-v1.md)，顺序队列见 [tickets](tickets/README.md)。

当前草稿裁决：consumer audit revision 2 保留为新轮输入；extrema-index search 进入 SIMD-09 重审；
typed-scratch radix/argsort 进入 SIMD-11 重审；两个 target-bound activation 草稿进入 SIMD-08 重审；
integrated-source hash 在最终内容确定后重新生成。`REWORK` 不表示实现或 API 已被接受。

已知限制：

- 当前只能引用 baseline 上已有 SIMD 功能，不能声称 revision 2 缺口已经修复；
- 没有当前 candidate、聚合 P1 verdict、跨平台性能结论或模块级 SOTA 证书；
- architecture/user/performance 文档仍需按新 epic 分流，旧历史结论不能作为当前用户承诺；
- C/Native FFI、稳定跨工具链 C++ ABI、tensor/model/GPU operator 继续属于明确负空间。

产品触发条件仍为：真实 consumer 暴露 correctness/security/usability/能力缺口；公开 API/layout、算法、
direct dependency、vendor/ISA 语义或 source/static/shared 交付改变；成熟同合同 peer 证明更优复杂度或正常
consumer 上 material advantage。满足上述 resume condition 是恢复审查，不代表这些触发已被证明。

本轮没有接受任何产品改动，LOC 绑定 immutable baseline
`29b77897c0a14c92dd1baa49dccb41033dc835cd`：全库 306,880；自实现含测试 124,275；自实现不含测试
83,119；SIMD 实现 38,024。terminal end/registry anchor 由后续 registry transition 记录。

<!-- MODULE CLOSURE RECEIPT: BLOCKED -->
