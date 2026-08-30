# SIMD v2 vendor-first resume closure

<!-- MODULE CLOSURE RECORD: TERMINAL SUPERSEDED -->

> 当前产品状态只以 [`module-status.json`](../../module-status.json) 为准。本页是从 terminal
> [`simd-v1`](simd-v1-closure.md) 恢复的新一轮冻结卡；旧 task/evidence 只作审计输入，不授权产品改动。
> 施工顺序和单票停止规则见 [`SIMD epic`](simd-epic-v1.md)。

> 2026-08-30 terminal verdict：`BLOCKED/SUPERSEDED_BY_OWNER_RESCOPE`。在 SIMD-12 删除旧 register
> mechanism 前，current xsimd f32 `tanh` 已复现 material advantage；owner 随后明确要求 Highway、xsimd
> 与 HYLE 多套候选长期并存，并对旧 387 个 operation 全量逐项裁决。该新需求与本卡“Highway 为默认产品
> 机制、xsimd 仅具名补位、抽样后删除”的冻结范围不兼容。本轮没有 candidate，禁止按本页余下 FIX 清单
> 继续删除；恢复记录见 [`simd-v3-provider-routing`](simd-v3-provider-routing.md)。

## Closure card（`AUDITED`）

- component / owner / round ID：`simd` / SIMD owner / `simd-v2`；task kind：`RESUME`。
- 前一 round / blocker / resume condition：`simd-v1` 从 `AUDITED` 停在 `BLOCKED`，因为一个 closure card
  同时绑定五组产品施工与跨 family P1 矩阵，无法在预算内诚实执行。恢复条件要求建立有界顺序 tickets、
  归属所有草稿、禁止产品票改中央仪器，并按 capability family 资格化性能。SIMD-00～03 已建立控制面、
  完成 Math/vendor 审计并用三平台隔离 proof 冻结迁移边界，满足恢复条件。
- baseline commit / tree：`5becc37ff546e64351b00c56ac6eb39eee7fec3f` /
  `9f2a3c0dd653c16823e3aec7eca0d960dbca0b08`。工作树中的 search、ordering、activation 与 dependency-hash
  草稿不是 baseline 或 candidate，只是 SIMD-08/09/11 的 `REWORK/REGENERATE` 输入。
- 进入原因与 demand IDs：
  - `SIMD-V2-VENDOR-01`：约 2.5 万行平行 register/backend/native 机制与 checked-in vendor ports让 HYLE 重复承担
    Highway/xsimd 的 ISA、compiler 与算法升级；用户要求成熟 vendor 机制优先；
  - `SIMD-V2-DELIVERY-01`：普通 C++ 用户仍需要自然 pointer/span/buffer API、一次 runtime dispatch，
    source-primary expert 需要 direct/fixed/fat kernel；不能把 vendor 类型泄漏到普通 ABI；
  - `SIMD-V2-INFER-01`：未来 Qwen/ggml-shaped CPU 推理需要连续 row algebra、dtype/mixed dot、activation、
    extrema/search、scan、bitpack/random 与 ordering primitives；tensor/layout/threading/GEMM 仍由 Compute；
  - `SIMD-V2-GAPS-01`：simd-v1 revision 2 已复现 row-op、allocation-free activation、extrema index 与稳定
    radix/argsort 的 `PARTIAL/MISSING`；
  - `SIMD-V2-DOC-01`：architecture/user/performance 中仍有 port ownership、历史 C ABI、provisional 与整模块
    SOTA 叙事，需要随接受的产品形状分流。
- 真实 consumer / normal workload：
  - 首席 source-primary 消费者是未来 HYLE 量化推理引擎：固定/动态 ISA 的连续 row kernel，常用
    `n=4096` cache-scale 与适用 `n=1048576` streaming buffer；小 tail 只作 correctness/dispatch 诊断；
  - 当前直接语法消费者是 Container 的 target-bound primitives 与 Compute 的 ordinary batch seam；二者只在
    其实际使用路径变化时迁移，不把下游半成品 closure 并入本轮；
  - llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`、ggml
    `36da57138425487184aa1da2eee2cde155909c6f` 是 north-star consumer/差异输入，不冒充现有 HYLE 生产调用。
- 范围：按 SIMD-04～12 顺序完成数学 vendor 合并、register/backend 删除或薄适配、scalable/runtime/exact/
  source/static/shared 交付、dtype/narrow/row ops、transcendental/activation、search/scan、bitpack/random、
  ordering/radix/top-k，以及最终文档、六问、红队与 closure；将 Highway 作为中央锁定的直接产品依赖，
  xsimd/SLEEF 只在具名 gap 中补位。
- 非目标：GPU backend，matrix/attention/RoPE/SSM/GEMM、tensor layout/workspace/threading、完整 BLAS/FFT/
  linear algebra、复制全部 ggml/Highway spellings、per-ISA DLL、C/Native FFI、稳定跨工具链 C++ ABI、
  PowerPC/LoongArch 新平台、中央 closure/performance 仪器重写和 Formal certification。
- 直接依赖 / 消费者：当前 `simd PUBLIC core,platform`；本轮允许新增中央锁定 Highway product dependency，
  xsimd 或 SLEEF 只有 SIMD-04 审计证明 Highway gap 才可加入。Platform 保持 CPU/OS facts owner；Mem/Error/
  Time 不因迁移进入热路径依赖。直接消费者只验证 Container/Compute 的受影响路径。
- tier 与理由：`P1`。ordinary batch、dtype、search、scan、random 与 ordering 是可能高频 runtime primitives；
  expert/fixed kernel 可进入推理热路径。没有端到端引擎 profile，任何 family 都不得升级为 P2。
- material margin / resource budget：
  - 同合同 peer 或 immutable baseline 的工程 non-inferiority margin 为 5%；小于 5% 且无可传播机制的差异
    不改变 API/layout/常态 metadata；
  - contiguous hot kernels 目标为 O(n)、O(1) extra storage、零隐藏 allocation/copy/lock/syscall；ordering 的
    workspace 与复杂度按其单票合同显式计费；
  - ordinary runtime 只允许每 batch 一次间接调用，不能重复 CPU detection/target bit scan；fixed/expert
    route 不支付 runtime dispatch；
  - 同 capability 的 candidate shared artifact 或 clean selected-component build time 若较 baseline 增长超过
    20%，必须由新增能力/target 明确计费，否则为 material resource failure。
- 总调查 / 性能预算：SIMD-04～11 各最多一个工程日、P1 60 分钟；SIMD-12 在 2026-08-30 owner
  “不兼容旧实现”重定界和本次计划复核后，**既有消耗不重置**，剩余硬上限 8 工程时、P1 60 分钟。
  达到剩余上限仍不能删除通用 legacy mechanism 或形成候选即 `BLOCKED`，不得继续研究或重置预算。
- 最多实验 / 修复：每票一次正反序复现、一次 codegen/profile 机制检查、一个有界产品批次和一次重跑；
  不发明第五种 benchmark 姿势。公共工具失格登记独立 tool task 并使适用 gate `INCONCLUSIVE/BLOCKED`。
- machine profile：W1-WIN 负责资格化 P1；W1-WIN、W1-WSL、M1-MAC 负责适用 correctness 与
  source/static/shared；SVE/RVV 无真硅时只签 cross-compile/emulator correctness，不签性能。
- instrument profile：`closure-perf-v1`；correctness 使用当前模块测试、sanitizer、ISA isolation、artifact
  inspection 与 consumer journey。SIMD-03 的隔离 spike 只证明架构可行，不继承为产品性能 PASS。
- strong peers（按 capability 最多两个）：
  - register/dispatch/search/bitpack/VQSort：Highway
    `f503c485cec6b4f2f596fa60d83484797c9eeb5c`；ordering 可再使用同合同标准库算法；
  - transcendental/accuracy：Highway 上述 identity 与 xsimd
    `3b3151ff665514a00a437cc985f47b7de99b772b`；SLEEF
    `7623d6cfa2712462880fa63a4d0f0b5f775d1a83` 只在二者无法满足精度合同后另行 rescope；
  - inference row shape：llama.cpp/ggml 是 consumer source；oneDNN
    `a244fe249f408a8f6ae61ad800e28fa64a8fb074` 只为同合同 batch primitive 提供第二 peer，不与 Highway
    同时形成三个 peer。
- 无同合同 peer 时的 budget / 禁止声明：使用同交付形状的 immutable HYLE baseline 与该票冻结的
  consumer composition；必须保持 5% non-inferiority、复杂度档与资源预算。新能力只有绝对/结构预算时
  只能声明“通过该 consumer budget”，不得声明 SOTA、全面快于 ggml/Highway 或跨平台冠军。
- correctness gates：normal、zero/empty、odd/tail、capacity/overflow、invalid input、alias/overlap、NaN/
  infinity/signed-zero/rounding、workspace/OOM、deterministic state、concurrency/reentrancy（仅适用面）、
  property/differential、ASan/UBSan、真实或隔离 ISA、public-header self-containment 与 direct consumer。
  move/destroy/cancellation 只在最终公开 owning state 确实存在时适用，不制造虚构 gate。
- delivery gates：selected SIMD source embedding、installed static/shared C++23 consumers；Windows
  export/import 和 runtime staging、Unix loader-relative lookup、offline locked dependency、exact/fat/source
  target discovery、中央 component graph 与 target link 一致。不得恢复旧 C/FFI artifact。
- performance gates：每票只跑其预声明 family rows，source 对 source、runtime 对等价 runtime、同 ISA/
  accuracy/layout/alias/workspace。最终只重放受后续 backend 改动影响的 family 与冻结 sentinel，不重跑
  314-name 笛卡尔积。
- 预期输出：vendor/source manifest、删除/保留表、有限 `FIX` ledger、自然 ordinary/expert C++ journeys、
  受影响 correctness/delivery/P1 evidence、不可变 candidate、六问、四项 LOC 与 terminal receipt。
- 停止条件 / 可接受限制：所有准入 `MUST` 为 `COMPLETE/ROUTED` 且适用 gate 通过才可关闭。经典 MSVC
  AVX-512、SVE/RVV 真硅、固定 wire format 等可以成为具名最小 gap；未知或 material failure 必须阻塞
  对应 ticket/aggregate。Formal 保持 `PENDING`，不阻塞工程 closure。

前态 registry 明确快照为
`C:\hyle-scratch\simd-v2\transitions\module-status-before-scoped.json`，SHA-256
`69EF4F377862977AFC7749D23FE116FB16B3091E0A79C07F37F6B1BF06939E19`。

## AUDITED capability-family map

本轮在 `a5e9abe6fd475e0c288a5b99c0016844aee02a59` 的 `SCOPED` 输入上完成两遍审计：自顶向下使用
Qwen/ggml row journey、C++ scan/random 语义、Highway 与 oneDNN；自底向上枚举 34 个 installed public
header、`Runtime` 表、387-name register inventory、当前 Container/Compute 语法消费者和全部 integrated
source 路径。参考库只形成下面的具名能力族，不按 symbol 求并集。

| family | admitted journeys / semantic variants | lifecycle、boundary 与正常资源 | current route | coverage | missing edge / v2 destination |
|---|---|---|---|---|---|
| target selection and delivery | unknown-CPU ordinary batch；known-target exact；expert source/fat kernel；actual-target diagnostics | process-wide immutable selection；ordinary 每 batch 一次 indirect call；source/static/shared/offline | public `Runtime` table、Platform aliases、HYLE per-ISA TUs | `PARTIAL` | 函数表和 CPU aliases 泄漏实现且多处各自选择；SIMD-05/06 改为 private table + 小型实际-target诊断，Highway 是 kernel authority |
| expert register/kernel construction | Container string/bitset 两条当前消费者；未来 codec/quant/filter/fused row | compile-time target；vector type 不跨 TU/shared boundary；O(n)、零 allocation | `vec.h` + 七套 backend + `native/*` | `PARTIAL` | 约 2.5 万行平行机制；迁移真实消费者后物理删除，expert 直接使用锁定 Highway，不保留 HYLE register adapter 产品 |
| source algorithms | fixed-target transform/generate/find/count/copy-if 与自定义 tail | caller owns buffers；exact alias only；无 runtime callback | HYLE templates built on D-tag | `PARTIAL` | 任务形状成立但平行模板面没有独立价值；ordinary 能力走 HYLE batch API，自定义 kernel 走 direct Highway fixed/fat source seam，旧 native surface 删除 |
| ordinary f32/f64 row algebra | residual/add、gate multiply、scale/bias、SAXPY、sum/dot/norm composition | exact alias 或 disjoint；O(n)/O(1)；零 heap/lock/copy | `dot/saxpy/add` | `PARTIAL` | 缺 sub/mul/div/scale/add-scalar/affine/sum/abs/negate/square/sqrt；SIMD-07 补齐，不把 tensor stride/fusion 下沉 |
| storage dtype conversion | f32 与 f16/bf16/e4m3/e5m2；storage-to-storage without full temporary；mixed dot | explicit RNE/saturation/NaN；exact alias only when widths equal；O(n)/O(1) | scalar types + asymmetric named batch calls | `PARTIAL` | 现有 f16→fp8 特例使转换图单向；SIMD-07 提供一个可发现的完整 hub/直接路线并保留 specialty kernels |
| narrow numeric source work | target-bound f16/bf16 arithmetic、search、sum、total order | register widen/operate/narrow；whole-buffer tensor policy belongs higher layer | expert narrow tags + ordinary search/sum | `COMPLETE/ROUTED` | expert task 完整；普通 narrow activation/sort/quant policy 路由 Compute，具体 CPU half-row consumer 出现才准入 |
| real/complex transcendental | f32/f64 real unary/binary；SoA complex；fast in-range expert contract | documented accuracy/domain/special values；exact alias or disjoint；O(n)/O(1) | HYLE façade over Highway/xsimd-derived ports | `COMPLETE` | public family 保留；SIMD-04 重新资格化后 direct vendor 或最小 delta，不按某一个域逐 op 选库 |
| model-independent activation | GELU variants、sigmoid/SiLU、ReLU/leaky-ReLU/ELU/hard-sigmoid/hard-swish、softmax/log-softmax | elementwise exact alias；softmax in/out；O(n)/O(1)；noexcept 路径不得分配 | six entries，四个 elementwise 当前在 `noexcept` 中分配 O(n) | `PARTIAL` | 重写为 target-bound allocation-free kernels；补 ggml+oneDNN 共同证明的前向集合和 out-of-place softmax；backward/Mish/Softplus 保持路由或 GENERAL |
| search and extrema | equality/count/find；min/max/minmax；first argmin/argmax；sortedness | empty=`n` for indices；first tie/first NaN；O(n)/O(1) | ordinary/native，extrema-index 是未接受草稿 | `PARTIAL` | SIMD-09 完成十类型 runtime/exact/native 和浮点合同；不引入 predicate callback runtime API |
| additive scan | in-place and out-of-place inclusive/exclusive sum；exclusive initial value | exact alias or disjoint；integer modular；O(n)/O(1) | public only in-place zero-init，native only in-place | `PARTIAL` | C++ standard + ggml cumsum 证明 out-of-place 是基本旅程；SIMD-09 补齐，custom op 路由 `std::*_scan`/expert source |
| fixed-width bit packing | portable persisted/transport bytes and same-process quant scratch | canonical little-endian layout；1..lane bits；non-overlap；O(n)/O(1) | runtime tier-dependent untagged bytes | `PARTIAL` | ordinary API 在不同 CPU 无法解码且不携带 format，属于危险半项；SIMD-10 固定 canonical layout，native quant layout 留给 Compute |
| bulk non-crypto random | deterministic u64/f32/f64 fill；copy/reseed；parallel independent stream | caller-owned mutable state；not thread-safe；O(n)/O(1)；no hidden TLS | eight-stream `RandomState` + fill | `PARTIAL` | xoshiro jump 与标准 engine discard/split 旅程缺公开路线；SIMD-10 增加明确 advance/split 并说明 copy/concurrency |
| full/partial/select ordering | no-allocation full sort、top-k prefix、nth/select；numeric/KV/U128；NaNs-last | in place；O(n log n) expected/worst contract as peer；no hidden heap | VQSort port + benchmark-selected streaming/radix routes | `PARTIAL` | ordinary `sort(n>=4096)` 偷分配 O(n) scratch；small-k route 需 adversarial complexity 资格化；SIMD-11 direct VQSort wrapper 或删除策略分支 |
| stable radix and permutation | stable numeric sort/argsort；caller scratch and allocating convenience | O(bytes*n), O(n) explicit workspace；OOM throws only on convenience；overlap rejected | typed-scratch 草稿 + current allocator path | `PARTIAL` | 草稿尚未成为 candidate；完成 type/NaN/signed-zero/alias/OOM/index contracts，不允许 ordinary sort 隐式调用 |
| tensor/model/quant/GPU | Qwen tensor layout、RMSNorm/GLU fusion、RoPE/attention/SSM、quant block/GEMM、device | stride/workspace/threading/model ownership | Compute/consumer/accelerator | `ROUTED/OUT_OF_SCOPE` | SIMD 提供连续 row 与 expert kernel seam；不复制 ggml tensor/backend 或其量化格式 |

## Public-surface trace

| public surface | mapped family / current consumer | AUDITED action |
|---|---|---|
| `types.h`, `dtype.h`, `dtype_ops.h` | storage types/conversion；Compute mixed row north-star | `FIX/BORROW` in SIMD-07；HYLE owns value/rounding contract，vendor owns mechanism |
| `dispatch.h`, `runtime.h`, `target.h`, `fat_kernel.h` | target/delivery；仓库外无直接语法消费者 | `FIX/REMOVE_INTERNALS` in SIMD-05/06；不再公开 function table 或转发 Platform detector |
| `vec.h`, `backend/*.h` | expert register family；Container 只有 string/bitset 两条 direct consumer | `MIGRATE_THEN_DROP` in SIMD-05；387 names 是删除 inventory，不是兼容清单 |
| `native.h`, `native/*.h` | task-shaped source algorithms/row primitives | owner 重定界为 `DELETE`；ordinary whole-range helper 已由 batch API 承担，自定义源码 kernel 直接使用锁定 Highway |
| `narrow_float.h`, `narrow_algorithm.h`, `narrow_ops.h` | narrow expert work及 ordinary search/sum | `KEEP/BORROW` in SIMD-07；不扩张 tensor/activation policy |
| `ops.h` | Compute 当前 add/dot/saxpy；Qwen/ggml row algebra | `FIX` in SIMD-07，形成完整普通 row seam |
| `math.h` | Compute 当前 exp；real/complex engineering 与 AI row | `BORROW/FIX` in SIMD-04/08；移除 hidden heap，保留 HYLE accuracy/alias API |
| `search.h` | extrema/search；当前 extrema-index 草稿 | `FIX` in SIMD-09，草稿仅作 REWORK input |
| `scan.h` | prefix sum | `FIX` in SIMD-09，补 out-of-place/init；不造 runtime callback |
| `bitpack.h` | generic integer codec；无当前 product consumer | `FIX/BORROW` in SIMD-10；ordinary bytes 改为 canonical，不再把 target-dependent layout 伪装成普通格式 |
| `random.h` | bulk deterministic RNG；无当前 product consumer | `FIX/KEEP_HYLE` in SIMD-10；stream semantic owner 留在 HYLE，kernel mechanism 可借 Highway |
| `sort.h` | full/top-k/stable ordering；无当前非测试 consumer | `FIX/BORROW` in SIMD-11；VQSort direct，workspace/稳定语义由 HYLE 拥有 |

所有 installed public header 已映射到上表；源码树没有 SIMD C header、`extern "C"` bridge、FFI option、
managed sample 或 C package metadata。shared library 的 decorated C++ exports 不是 C ABI。

## Capability / gap ledger

表中 `owner` 顺序为 semantic / mechanism-artifact / decision-delivery；正常指针任务均为 caller-owned、
`n==0` no-op、`n>0` non-null，除非对应 family 行另述。

| ID / kind / source | normal shape | owner、failure 与复杂度 | coverage / action / ticket | verification / negative space |
|---|---|---|---|---|
| `SIMD-VENDOR-01` MUST；Highway proof + maintenance defect | ordinary 4K/streaming batch，expert fused TU | HYLE / Highway / SIMD+build；one dispatch，O(n)，zero hidden cost | `PARTIAL` / `BORROW` / 04–06 | dependency/license/offline、codegen、三交付；不复制 full vendor API |
| `SIMD-REG-01` MUST；Container + north-star source kernels | fixed/scalable register loop | HYLE routing / Highway / Highway adapter；compile-time target | `PARTIAL` / `FIX` / 05 | migrate string+bitset、property/codegen；crypto/codec protocols 不归 SIMD |
| `SIMD-DISPATCH-01` MUST；delivery contract | process runtime、exact artifact、source fat kernel | HYLE / Highway+minimal gap / one private table；thread-safe immutable init | `PARTIAL` / `FIX` / 06 | source/static/shared/offline/exact；no per-ISA DLL/FFI/stable cross-toolchain ABI |
| `SIMD-OPS-01` MUST；Qwen/ggml + oneDNN row | f32/f64 n=4096 and streaming | HYLE / Highway kernels / SIMD；exact alias/disjoint，O(n)/O(1) | `PARTIAL` / `FIX` / 07 | property/alias/tail + unary/binary/reduction sentinel；tensor strides/fusion Compute |
| `SIMD-DTYPE-01` MUST；existing promise + oneDNN reorder | batch among f32 and four narrow storage types | HYLE rounding / Highway+specialty TU / SIMD；no full temp | `PARTIAL` / `FIX` / 07 | exhaustive fp8, sampled f16/bf16, HW fallback；quant scale/block policy Compute |
| `SIMD-NARROW-01` GENERAL；ggml half rows | expert register arithmetic and ordinary storage search | HYLE / Highway adapter / SIMD；explicit rounding | `COMPLETE/ROUTED` / `KEEP+DEFER` / 07 | current narrow property; reopen for concrete material half-row traffic |
| `SIMD-MATH-01` MUST；existing engineering + Highway/xsimd | qualified f32/f64 real/complex batch | HYLE accuracy / qualified vendor / SIMD；O(n)/O(1) | `COMPLETE` / `BORROW` / 04 | domain/ULP/special/alias + limited sentinels；no per-input winner switch |
| `SIMD-ACT-01` MUST；Qwen/ggml + oneDNN | 4K elementwise/softmax row | HYLE / Highway math + HYLE composition / SIMD；O(n)/O(1)，no heap | `PARTIAL` / `FIX` / 08 | property/allocation/alias/softmax stability + P1; backward/fused norm/GLU Compute |
| `SIMD-ACT-02` GENERAL；oneDNN-only current delta | Mish/Softplus/clip/pow activation spellings | Compute/standard composition until second demand | `ROUTED` / `DEFER` | reopen with second mature source or named consumer; no API-union copying |
| `SIMD-SEARCH-01` MUST；Highway + llama/ggml argmax | ten scalar types，first tie/NaN | HYLE / Highway algo / SIMD；O(n)/O(1) | `MISSING` at baseline, draft `PARTIAL` / `FIX` / 09 | differential/property + f32 count/argmax P1；runtime predicates rejected |
| `SIMD-SCAN-01` MUST；C++ standard + ggml cumsum | in/out inclusive/exclusive plus init | HYLE / Highway-register composition / SIMD；O(n)/O(1) | `PARTIAL` / `FIX` / 09 | alias/modular/float tolerance + u32 sentinel；segmented/custom op routed |
| `SIMD-BITPACK-01` MUST；generic codec + persisted model input | canonical u8/u16/u32 1..width | HYLE wire / Highway or minimal delta / SIMD；O(n)/O(1) | `PARTIAL` / `FIX` / 10 | cross-tier byte oracle, 11/13, VL independence; quant block/framing Compute |
| `SIMD-RNG-01` MUST；xoshiro reference + standard engine lifecycle | bulk fill and independent worker state | HYLE stream / minimal HYLE+vendor ops / SIMD；caller state, no TLS/lock | `PARTIAL` / `FIX` / 10 | deterministic sequence/split/copy/concurrency docs + f32 sentinel；not crypto/entropy/distributions |
| `SIMD-SORT-01` MUST；Highway VQSort + top-k journey | no-heap full/partial/select | HYLE order / direct VQSort / SIMD；no hidden allocation | `PARTIAL` / `FIX` / 11 | adversarial distribution, NaN/KV/U128, P1；no automatic stable-radix allocation |
| `SIMD-SORT-02` MUST；existing stable API + standard stable order | typed scratch radix/argsort | HYLE / HYLE stable radix / SIMD；O(bytes*n), O(n) explicit scratch | `PARTIAL` / `FIX` / 11 | sanitizer/OOM/overlap/type/index + qualified rows；alloc convenience may throw |
| `SIMD-INFER-ROUTE-01` MUST；Qwen/ggml graph | tensor/model/device work | Compute/consumer / backend owner / Compute delivery | `ROUTED` / `KEEP` | later Compute closure；no public model types in SIMD |
| `SIMD-FFI-01` build policy | C++ source/static/shared only | build owns negative product surface | `COMPLETE` / `KEEP_NEGATIVE` / 12 | symbol/header/option scan；private OS C boundaries allowed |
| `SIMD-DOC-01` MUST；stale current pages | ordinary/expert choice and traps | SIMD docs owner | `PARTIAL` / `FIX` / every ticket + 12 | docs/stale scan；history never becomes current authority |

## Frozen `FIX` set and ticket stop lines

只有下列产品变化获授权；新发现不得直接施工，必须退回 `AUDITED` 重裁决。

1. SIMD-04：direct Highway math qualification；xsimd 仅保留具名 accuracy/capability gap；删除等价 port。
2. SIMD-05：迁移 Container 两条 D-tag consumer；删除平行 fixed backend/387-name surface，真实 specialty gap
   只留最小 TU/adapter。
3. SIMD-06：私有化 function table，去掉 Platform detector re-export，统一一次 target authority；完成 scalable、
   exact/source/static/shared/offline delivery，不恢复 FFI。
4. SIMD-07：完成 f32/f64 row algebra 与 storage conversion graph；保留 dtype rounding、alias 与 specialty
   hardware，narrow tensor policy继续路由。
5. SIMD-08：接受或重写 activation 草稿为无分配 target kernel；补有限常用前向集合及 out-of-place
   softmax/log-softmax。
6. SIMD-09：完成 extrema index；补 in/out scan 与 exclusive init；不新增 callback runtime。
7. SIMD-10：把普通 bitpack 改为跨 tier canonical bytes；为 RNG 增加 advance/split 生命周期并记录线程合同。
8. SIMD-11：ordinary full/partial/select 不隐藏 O(n) allocation；资格化或移除 benchmark-derived strategy；完成
   stable radix/argsort workspace、OOM、overlap、type/order，最终 direct-wrap VQSort。
9. SIMD-12：完成 SIMD-05 已明确递延的 Container 两条 consumer 退出、Ordering-owned VQSort mechanism
   内聚和通用 `vec/native/backend` 物理删除；随后只修集成 blocker，冻结 candidate、三交付/consumer/
   correctness/P1 family evidence、文档六问与 terminal receipt。该清算属于原冻结 FIX，不得新增能力、
   Container 公共 API 或 benchmark shape。

工作树草稿裁决不变：extrema→SIMD-09 `REWORK`，activation headers→SIMD-08 `REWORK`，typed scratch
ordering→SIMD-11 `REWORK`，dependency content hash→最终内容后 `REGENERATE`。任何草稿都不是 candidate。

## AUDITED 六问直答（施工前）

1. **第三方：**当前产品路径含 Highway `83eff393…` 和 xsimd `5d2490f…` integrated-source port；冻结目标
   Highway 是 `f503c485…` direct product dependency。semantic owner 仍是 HYLE；SIMD-04～11 改变实际
   product identity 后旧性能证据不能继承。
2. **功能：**尚不完整；上表所有 `PARTIAL/MISSING` 已落到 04～11 的有限 FIX，没有未处置的本轮 `MUST`。
   tensor/quant/model/device 已完整路由 Compute/consumer，不伪装为 SIMD 缺口。
3. **性能：**本轮为 `P1 PLANNED`，尚无 candidate verdict，不能声称 non-inferior/SOTA。每票只签同合同
   family；wrapper proof 只证明架构可行。
4. **旧 C ABI / Native FFI：**源码树为清洁负空间；没有 SIMD C 产品面。历史 build artifact 不构成合同。
5. **构建与导入：**当前旧实现能构建不证明迁移候选；direct Highway 后必须重签 source/static/shared、
   offline、Windows staging、Unix lookup、exact/fixed/fat consumer。
6. **C++ API 与文档：**普通 whole-buffer 形状方向正确，但 public Runtime/dispatch、bitpack format、hidden
   allocation、单向 dtype 与若干半项 journey 不自然；它们正是冻结 FIX。当前 user guide 标为 provisional，
   SIMD-12 前不得当成终态指南。

## Candidate

尚未产生；不得继承 simd-v1 或工作树草稿为 candidate。

## Verification manifest

尚未执行。SCOPED 输入、SIMD-03 proof 与历史测试均不得写成当前 gate `PASS`。

## Closure decision

双向 capability/public-surface 审计已完成，有限 FIX 集已冻结，当前裁决为 `AUDITED`。下一步只允许按
SIMD-04～12 顺序施工；任何票到预算即 `DONE/BLOCKED`，不得借本页继续扩大产品面。

本次 `SCOPED -> AUDITED` 前态 registry 快照为
`C:\hyle-scratch\simd-v2\transitions\module-status-before-audited.json`，SHA-256
`7CA35837CA4BE36DD84D1847996EF71FCF3A9DBF40E8189D99FEF516CB7CAD94`。

<!-- MODULE CLOSURE RECEIPT: ACTIVE -->
