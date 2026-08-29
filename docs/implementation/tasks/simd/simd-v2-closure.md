# SIMD v2 vendor-first resume closure

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前产品状态只以 [`module-status.json`](../../module-status.json) 为准。本页是从 terminal
> [`simd-v1`](simd-v1-closure.md) 恢复的新一轮冻结卡；旧 task/evidence 只作审计输入，不授权产品改动。
> 施工顺序和单票停止规则见 [`SIMD epic`](simd-epic-v1.md)。

## Closure card（`SCOPED`）

- component / owner / round ID：`simd` / SIMD owner / `simd-v2`；task kind：`RESUME`。
- 前一 round / blocker / resume condition：`simd-v1` 从 `AUDITED` 停在 `BLOCKED`，因为一个 closure card
  同时绑定五组产品施工与跨 family P1 矩阵，无法在预算内诚实执行。恢复条件要求建立有界顺序 tickets、
  归属所有草稿、禁止产品票改中央仪器，并按 capability family 资格化性能。SIMD-00～03 已建立控制面、
  完成 Math/vendor 审计并用三平台隔离 proof 冻结迁移边界，满足恢复条件。
- baseline commit / tree：`5becc37ff546e64351b00c56ac6eb39eee7fec3f` /
  `9f2a3c0dd653c16823e3aec7eca0d960dbca0b08`。工作树中的 search、ordering、activation 与 dependency-hash
  草稿不是 baseline 或 candidate，只是 SIMD-08/09/11 的 `REWORK/REGENERATE` 输入。
- 进入原因与 demand IDs：
  - `SIMD-V2-VENDOR-01`：约 1.9 万行平行 register backend 与 checked-in vendor ports 让 HYLE 重复承担
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
- 总调查 / 性能预算：SIMD-04～11 各最多一个工程日、P1 60 分钟；SIMD-12 最多一个工程日、P1 90 分钟；
  全轮上限 9 个工程日与 9.5 小时性能取证。任何票未到结论即按自己的 `DONE/BLOCKED` 停止，不把余额
  借给另一票无限调查。
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
  export/import 和 runtime staging、Unix loader-relative lookup、offline locked dependency、exact/fat/native
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

## Pre-audit capability inputs

以下只是 `SCOPED` 输入；SIMD-04 开始前必须完成自顶向下/自底向上审计并冻结有限 `FIX` 集，才能进入
`AUDITED`。当前状态不能授权产品修改。

| family | admitted journey | current coverage | required v2 decision |
|---|---|---|---|
| target selection/delivery | ordinary runtime、fixed/exact、expert source、fat kernel | `COMPLETE` but mechanism affected | 保留 HYLE contract/Platform facts，迁移 target bodies，证明三交付与一次 dispatch |
| register/D-tag | custom codec/quant/filter/fused source kernel | `COMPLETE` but duplicate ownership | consumer trace 后 direct Highway、thin adapter 或具名 gap；不默认保留 387-name DSL |
| ordinary row algebra/mixed dot | residual/gate/scale/reduction、storage dot | `PARTIAL` | 完成准入 f32/f64 algebra/reduction，保留 dtype-specific specialty gap |
| dtype/narrow | f16/bf16/fp8 conversion、mixed storage | `COMPLETE` | 迁移机制并保留 rounding/special-value contract |
| real/complex math/activation | exp/log/trig 与 GELU/SiLU/softmax row | `PARTIAL` | vendor accuracy 资格化；activation O(n)/O(1) 且无隐藏 heap |
| search/extrema/scan | find/count/min/max/argmax/prefix | `PARTIAL` | extrema index 完整；scan 只迁移不扩张 |
| bitpack/random | quant block bits、deterministic bulk fill | `COMPLETE` | 固定 wire/stream contract 对等才 wrap，否则最小 gap |
| full/top-k/stable ordering | sort/select、stable radix/argsort/workspace | `PARTIAL` | VQSort direct wrapper；修 alias/OOM/type/workspace 与稳定 float order |
| tensor/model/GPU | Qwen tensor graph、attention、GEMM、device | `ROUTED/OUT_OF_SCOPE` | 保持 Compute/consumer ownership；SIMD 只提供连续 row/source seam |

## Capability / gap ledger

在进入 `AUDITED` 前由 SIMD-04 的前置审计写入；覆盖状态与动作分离，`PARTIAL/MISSING + KEEP` 非法。

## Candidate

尚未产生；不得继承 simd-v1 或工作树草稿为 candidate。

## Verification manifest

尚未执行。SCOPED 输入、SIMD-03 proof 与历史测试均不得写成当前 gate `PASS`。

## Closure decision

当前只裁决为 `SCOPED`。下一合法状态是完成双向 capability/public-surface 审计与有限 FIX 冻结后进入
`AUDITED`，或出现具名 blocker 时进入 `BLOCKED`。

<!-- MODULE CLOSURE RECEIPT: ACTIVE -->
