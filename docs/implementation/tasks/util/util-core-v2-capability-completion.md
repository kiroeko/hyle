# util_core v2 能力补全与跨目标修复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `util-core-v2-capability-completion` 的活动冻结卡。当前机械状态只看
> [`module-status.json`](../../module-status.json)；v1 与 capability replay 仅作为历史证据，不能覆盖
> 2026-09-01 已接受的全局 capability-factorization 结论。

## Closure card

- component / owner / round ID：`util_core` / Util-core maintainers /
  `util-core-v2-capability-completion`；
- task kind：`REOPEN`；
- baseline commit：`6e75266a06de7f6dafecda3f452b07285e40f6fc`；
- 进入原因与 demand/defect ID：全局 Stage B 把 `UTILC-A-09/12/15/17` 明确归入 `B-F04
  KEEP_PRIME`；其中 RISC-V pause 是支持目标上的可复现机制缺陷，其余三项是成熟 peer 与 HYLE
  同步 consumer 共同证明的能力缺口；
- 真实 consumer task / normal workload：ggml/llama.cpp CPU threadpool 的 RISC-V polling；Platform
  visitor、Registry/Container synchronous iteration 与 Sched bulk call-only callback；thread-confined
  plan/AST/object graph ownership；required pointer parameter；
- 范围：补齐上述四项 header-only C++ 能力，复验既有 Ref/SpinLock P1 合同、三平台正确性、RISC-V
  编译/指令、source/static/shared selected-component 交付、文档和依赖；
- 非目标：RW/adaptive/parking/fiber mutex、owning callable、optional-ref 同义类型、secure entropy、weak/
  aliasing ownership、跨仓 consumer 迁移、C/Native FFI、稳定 C++ ABI、benchmark 仪器重写、Formal；
- 直接依赖 / 直接消费者：仍为 `util_core -> PUBLIC core`；现有直接消费者 Mem/Sched/Graph；新能力
  先形成公开自然词汇，不在本轮越界重写其他 owner；
- tier 与理由：`P1`，因为既有 Ref retain/release 与短锁 handoff 是已签署热路；新增借用类型为 P0
  layout/codegen，不能凭合成点值制造 API；
- material margin / resource budget：既有 W1-WIN 同合同 P1 非劣门 5%；新增类型必须零分配、无常态
  runtime branch，`FunctionRef` 两个机器字，`NotNull<T*>` 一个机器字，thread-confined counter 一个
  `uint32_t`；
- 总调查预算 / 性能预算：本轮一个有限实现周期；只复验 v1 两个 P1 normal shapes，并做新能力的
  P0 layout/codegen/assembly 检查；不做亚纳秒搜索；
- 最多复现、机制诊断、产品修改次数：每个冻结缺陷一次最小复现、一次机制诊断、一次产品修复；
  既有 P1 gate 若 materially fail，最多一次反序确认与一次机制诊断，然后回 `AUDITED` 或阻塞；
- machine profile：correctness `W1-WIN/W1-WSL/M1-MAC`；工程性能 `W1-WIN`；RISC-V 使用支持的
  compile/QEMU profile，不冒充真机 timing；
- instrument profile：`closure-perf-v1`；
- strong peers：Abseil `5e12b490a2009cb223dd75a70fd0213847b82f9c` 与 Folly
  `176c088840724715b2c3c6211b2b347611676002`（FunctionRef/borrow）；Boost
  `78505eab9b203e3e26e83893d0685f160133fa0f`（intrusive counter policies）；oneTBB
  `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` 与 ggml
  `d4716378882593333721eb33f153144b6885caf2`（short backoff/RISC-V pause）；llama.cpp
  `9d817213a0975020775efe6c458822616826f376` 是推理 consumer reference；均为 2026-09-02
  本地 clean locked checkout；
- 使用原型 provenance：`REAL_CONSUMER + NORTH_STAR_JOURNEY + STANDARD`；
- API 语义档位与选择前提：同步调用期借用 `FunctionRef`；异步/存储继续 owning callable 或 owner
  fn+ctx；required pointer 用 `NotNull<T*>`，nullable 用 `T*`，fallible non-null 用 `Result<T&>`；跨线程
  ownership 用默认 `RefCounted<T>`，明确 thread-confined 才 opt in `ThreadUnsafeRefCounted<T>`；
- 适用 gates：scope、architecture、correctness、delivery、usability/docs、red team、P1 engineering；
- 预期输出：两个新 public headers、Ref policy 与 RISC-V hint 修复、tests/codegen、更新后的 architecture/
  guide/performance evidence 与 terminal receipt；
- 停止条件：四项 `FIX` 全部通过，或出现无法在冻结合同内解决的 correctness/delivery/P1 blocker；
  不以“继续研究”收尾。

## Capability-family coverage map

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | state | missing edge / negative space |
|---|---|---|---|---|---|---|
| statistical RNG | inference sampling、shuffle、parallel stream | explicit/TLS、bounded/full-range、state/jump | caller/TLS owns mutable stream | `Rng` / standard distributions | `COMPLETE` | secure bytes routed to Platform/security owner |
| lexical cleanup | rollback、early-return cleanup | move、dismiss/invoke、construction failure | guard owns callable until once execution | `ScopeGuard` | `COMPLETE` | typed resource and exception-only guards route to RAII/standard |
| intrusive strong ownership | one-word hot owner、custom teardown | atomic/thread-confined、adopt/retain/casts | object embeds count；owner carries +1 | `RefCounted` + `Ref` | `PARTIAL` | built-in thread-confined policy missing |
| short spin/hint | tiny lock state and polling | x86/Arm/RISC-V hint、bounded yield | lock owner controls hold/wake policy | `SpinLock` + `cpu_relax` | `PARTIAL` | supported RISC-V currently silently no-ops |
| call-only type erasure | synchronous visitor/bulk callback | value/void/noexcept、function/lambda、const/mutable | callee borrows only for call；never stores | repeated templates/raw fn+ctx | `MISSING` | common two-word borrowed callable absent |
| borrow vocabulary | required/nullable/fallible borrow | const conversion、temporary/null rejection | caller retains referent lifetime | `T&`/`T*`/`Result<T&>` | `PARTIAL` | rebinding required pointer has no discoverable non-null type |
| diagnostic type identity | local compile-time spelling/hash | cv/ref/array/function/incomplete | static compiler spelling；not wire identity | `type_name/type_hash` | `COMPLETE` | runtime/persistent identity routed elsewhere |
| broader synchronization/callable | RW/park/fiber/owning async callback | owner-specific progress/storage/cancel | state machine owns wait/storage lifetime | standard/Platform/Sched/owner | `ROUTED` | deliberately not a util_core API union |

## Capability / gap ledger

| ID | class | demand kind | exact source | normal shape | owner | lifetime/concurrency | complexity/resource | state | action | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `UC2-RISCV-01` | MUST | defect + real consumer | supported HYLE RISC-V target；ggml CPU polling | atomic polling/short spin | util_core header | hint only；no synchronization | O(1)，one ISA hint，zero state/allocation | PARTIAL | FIX | high | RISC-V compile/codegen/QEMU；no silicon timing claim |
| `UC2-FREF-01` | MUST | peers + repeated consumers | Abseil/Folly；Platform/Registry/Sched sync callbacks | pass borrowed callable by value for one synchronous call | util_core header | referent outlives call；no store/return/async | O(1)，2 words，zero copy/allocation | MISSING | FIX | high | callable matrix、temporary lifetime diagnostic、size/codegen；no owning semantics |
| `UC2-REFLOCAL-01` | MUST | peer + performance foundation | Boost safe/unsafe counter policies；thread-confined graph/plan journey | type-level non-atomic counter opt-in | util_core header | single-thread precondition；same custom destroy | O(1)，one `uint32_t`，no atomic/branch | PARTIAL | FIX | medium | ownership suite for both modes；release assembly contains no atomic RMW |
| `UC2-BORROW-01` | MUST | peers + cross-domain route | Folly not_null；Abseil nullability；Error `Result<T&>` | required rebindable pointer parameter | util_core header | non-owning；null invariant fatal；no temporary lifetime extension | O(1)，1 word，zero allocation/branch after construction | PARTIAL | FIX | medium | conversion/comparison/hash/layout/death tests；raw pointer remains nullable |
| `UC2-BACKOFF-01` | GENERAL | duplicate scan | oneTBB atomic backoff；Container/Sched private policies | owner-specific spin/yield/park escalation | owning state machine | wake/deadline/cancel differ | no extra common metadata | ROUTED | REJECT public class | low | keep only common `cpu_relax`; do not force consumer migration |
| `UC2-RW-01` | GENERAL | peer breadth | oneTBB/Folly RW/adaptive locks | ordinary thread shared/parking locks | standard/Platform/Sched | fairness/progress/fiber differ | owner-specific | ROUTED | REJECT | low | std mutex/RW + Platform wait + Sched fiber route |
| `UC2-OPTREF-01` | GENERAL | peer breadth | Abseil optional_ref | nullable reference | language/standard route | non-owning nullable | one pointer | COMPLETE route | REJECT | low | use `T*`; do not add a synonym without consumer-specific semantics |

- existing public-surface reverse trace：v1 五个 headers remain；two new headers enter the same factor；Ref default
  remains atomic and source-compatible；SpinLock public shape remains unchanged；
- semantic duplicate scan：Platform/Sched/Container 的 stored callback 或 park policy 不是 FunctionRef/backoff
  同合同；Error `Result<T&>` 已承载 fallible non-null borrow；
- public/internal/higher-layer decision：`FunctionRef`、`NotNull` 与 counter policy 是 public C++ vocabulary；
  `SpinBackoff` remains private；
- frozen `FIX` set：only `UC2-RISCV-01`、`UC2-FREF-01`、`UC2-REFLOCAL-01`、`UC2-BORROW-01`
  plus tests、delivery、documentation and existing P1 replay；
- adjacent backlog：no consumer migration in this round；future owners may adopt only when their own lifetime/semantic
  audit proves equivalence。

## API / usability freeze

- status：`FROZEN`；
- `FunctionRef<R(Args...)>` and `FunctionRef<R(Args...) noexcept>` are non-null, trivially copyable call-only
  views. They accept lvalue/rvalue callable for the duration of a parameter call and function pointers, use
  `std::invoke` semantics, preserve `noexcept` at the signature, and have no default/null state. Returning or storing
  one beyond the full expression remains caller misuse and is stated explicitly；
- `NotNull<Pointer>` is admitted only when `Pointer` is a raw object/function pointer. It is one word, cannot default
  construct or construct/assign from `nullptr`, checks dynamic null once as a fatal invariant, supports safe pointer
  conversion, `get/*/->`, identity comparison and hash. It does not own or extend lifetime and deliberately does not
  wrap smart pointers；
- `RefCountMode::{kThreadSafe,kThreadUnsafe}` is a type-level policy on `RefCounted<Sub, Mode>`；the default stays
  thread-safe. `ThreadUnsafeRefCounted<Sub>` is the discoverable opt-in alias. `Ref<T>` remains policy-agnostic via
  `incref/decref`; crossing threads with the unsafe base is a contract violation；
- `cpu_relax()` adds RISC-V Zihintpause emission for GCC/Clang, using the standardized mnemonic when the feature macro
  is available and the instruction encoding fallback used by ggml otherwise. It remains a hint with no memory-order
  semantics；
- public dependency remains Core only；both new headers are selected/installed by `hyle::util` and internal
  `hyle_util_core` closure；Native FFI remains not admitted；
- benchmark cannot change these APIs/layouts. Internal invoker spelling、counter policy helper and inline choices may
  change if tests/codegen preserve the frozen contract；
- if a normal consumer proves two-word callback overhead materially fails a same-contract peer, return to `AUDITED`；
  do not add metadata or a second “fast” API inside benchmark work。

## Owner 六问（AUDITED 直答）

1. **第三方产品路径：**none；all peers are clean source references. Product target remains header-only on Core and
   standard library/compiler ISA mechanisms.
2. **推理能力：**the four fixes close all admitted `B-F04` partial/missing rows. Sampling distributions, secure
   entropy, parking/fiber synchronization and owning async callbacks are complete routes to other owners, not missing
   util_core APIs.
3. **性能：**P1 remains limited to frozen Ref handoff and short SpinLock workloads on W1-WIN at 5%. New callback,
   borrow and unsafe-counter claims are structural layout/codegen only; RISC-V gets compile/instruction correctness,
   not simulated timing. No all-operation/cross-platform/SOTA claim.
4. **旧 C ABI / Native FFI：**none admitted; no C header, bridge, unmanaged export or managed sample is introduced.
5. **交付：**source/static/shared selected Util product must all expose the same seven util_core headers; the factor
   itself remains INTERFACE and creates no fake binary artifact.
6. **C++ 易用性/文档：**guide must add a selection row and recipes for callback lifetime, required vs nullable vs
   fallible borrow, atomic vs thread-confined ownership and RISC-V hint boundary; architecture must update capability,
   complexity and maintenance contracts.

## Candidate

- candidate-freeze input：四项 frozen FIX 已落地；MSVC 19.50、clang-cl 20.1、GCC 14.2 focused
  suite 与逐头 self-contained 全部通过；AppleClang 21 手工独立编译 80/80；RISC-V rv64gc 与
  rv64gc_zihintpause 分别确认标准编码和 `pause` mnemonic，static sample 在 QEMU 成功退出；
- structural evidence：MSVC/GCC optimized probe 中默认 counter 保留 atomic RMW、thread-confined
  counter 是 plain integer、NotNull unwrap 是 pointer identity；selected Util source/static/shared journey
  已安装并消费两个新 header；
- product commit / tree：`d42d992cf99232669f60422f30fb3b50ec9ec4eb` /
  `9944f099679503df632b6335e69154656164b30b`；
- public contract changes：`FunctionRef`、`NotNull`、thread-confined RefCounted policy 与 RISC-V pause；
- candidate tests / harness：focused unit/header suite、maintenance-only codegen object、v2 P1 contract；

## Verification manifest

### Correctness risk matrix

| Risk | state | evidence | limit |
|---|---|---|---|
| normal callable/pointer/ref/spin | PASS | candidate focused suite + self-contained headers on MSVC/GCC；AppleClang 80/80 | public synchronous contract only |
| empty / zero / null | PASS | deleted construction static asserts + dynamic null/empty death tests | no nullable FunctionRef/NotNull |
| capacity / overflow | PASS | Ref zero/overflow/underflow and batch ownership suite | counter remains `uint32_t` |
| move / destroy / lifetime | PASS | move-only/mutable/const callable、custom teardown、borrow conversion；Clang lifetime diagnostic caught invalid stored temporary member pointer | referent lifetime remains caller-owned |
| concurrency | PASS | candidate TSan Ref concurrent release + five SpinLock tests 6/6 | unsafe mode intentionally single-thread |
| property / stress | PASS | existing RNG distribution/state and Ref/Spin contention suite | no fuzz-only API |
| real ISA/backend | PASS | x64/Arm builds；RISC-V two march codegen + QEMU static sample | no RISC-V silicon timing |

### Candidate correctness、sanitizer、delivery 与 consumers

| Journey | Result |
|---|---|
| W1-WIN MSVC 19.50 Release/WERROR focused unit + seven header self-contained | `2/2 PASS` |
| W1-WIN clang-cl 20.1 Release/WERROR focused unit + headers | `2/2 PASS` |
| W1-WSL GCC 14.2 Release/WERROR focused unit + headers | `2/2 PASS` |
| M1-MAC AppleClang 21 independent manual compile | `80/80 PASS` |
| exact candidate GCC 14 ASan / UBSan | `2/2 PASS` / `2/2 PASS` |
| exact candidate GCC 14 TSan applicable concurrent subset | `6/6 PASS`，no race report |
| selected Util source + relocated shared + relocated static consumer | `PASS`；两个新 header 均 installed/consumed |
| exact candidate Mem/Sched/Graph consumer closure | `7/7 PASS`；含两个 header tests 与三个 Graph tests |

WSL/macOS/RISC-V 树都是 disposable validation copy，canonical product 仍只来自 Windows checkout。util_core
保持 header-only INTERFACE factor，因此没有虚构 static/shared binary；三种 journey 验的是同一 `hyle::util`
产品安装/源码消费面。

### W1-WIN P1 qualification

- clean candidate/peers/contract identity、完整 strata 与 bundle hash 见
  [`benchmarks-util-core-v2`](../../evidence/performance/benchmarks-util-core-v2.md)；independent verifier 返回
  `QUALIFIED / EXACT / CURRENT`，paths verified；
- Ref owning-handoff 的四 strata median advantage ratio 为 1.0991--1.1011，最低 99.26% CI lower
  1.0781，combined `WIN`；
- short state transition 的四 strata ratio 为 0.9986--1.0000，最低 CI lower 0.9622，combined
  `PARITY`；两项全部通过 5% non-inferiority；
- A/A、5%/10% known-delta、timer-floor、placement、artifact/build identity 与 consumer provenance 全
  `PASS`，n=15 已资格化而未触发 n=31；
- FunctionRef/NotNull/thread-confined counter 只签 size、allocation、branch/codegen 等 P0 结构合同；没有
  用新 microbenchmark 反向改变 API。

### Hostile red team 与 Owner 六问终答

1. **第三方：`PASS`。**util_core 产品闭包仍只有 Core；Abseil/Folly/Boost/oneTBB/ggml/llama.cpp 仅是
   clean locked reference，公开头没有第三方 include/type，标准库和 ISA intrinsic/asm 是机制。
2. **功能：`PASS`。**`UC2-RISCV/FREF/REFLOCAL/BORROW` 全部 `COMPLETE`，未处置 `MUST` 为 0；secure
   entropy、parking/fiber lock、owning async callable、weak/aliasing ownership 均有明确 owner 或拒绝理由。
3. **性能：`PASS`。**既有两项正常 P1 consumer 通过 5% 门；新能力的 O(1)、零分配/常态字段和 codegen
   结构通过。结论不外推为全操作、跨平台或端到端推理 SOTA。
4. **旧 C ABI / Native FFI：`CLEAN`。**没有 C header/schema/bridge/managed sample/undecorated product
   export。test-only `extern "C"` probe 只稳定 object inspection 名称，不安装、不链接入 SDK。
5. **构建/交付：`PASS`。**中央图仍是 `util_core PUBLIC_DEPS core`，七个 header 由同一 factor 清单选择、
   安装和导出；source/static/shared consumer 与直接消费者均通过，没有新增 binary 或依赖环。
6. **C++ 易用性/文档：`PASS`。**guide 以同步/异步、required/nullable/fallible、thread-safe/confined 的真实
   选择组织 API，并写明 temporary、存储、跨线程、null、RISC-V hint 陷阱；architecture/task/performance
   分别承担长期设计、事实和有限证据，公开面没有 benchmark seam。

红队逐文件复核 baseline→candidate 产品 diff、依赖、header inventory、FFI/export、生命周期和性能比较
合同；候选后产品 diff 为零，剩余 medium-or-higher finding 为 0。唯一发现的维护缺陷是 LOC canonical
boundary 漏登记两个新 header；只更新 `loc-report.py` 的边界清单并增加 tooling regression test，未改变
candidate 产品 tree 或统计核心。

### Candidate LOC

`python scripts/loc-report.py --module util_core --commit d42d992cf99232669f60422f30fb3b50ec9ec4eb
--self-check --json` 由 cloc 2.08 从不可变 tree 复算并通过 self-check：

| Metric | code |
|---|---:|
| 全库代码 | 419,146 |
| 自实现包含测试 | 101,555 |
| 自实现不包含测试 | 63,267 |
| util_core 实现 | 967 |

### Aggregate gates

| Gate | state | evidence / limit |
|---|---|---|
| scope | PASS | only four frozen FIX；candidate product unchanged after freeze |
| architecture | PASS | Core-only prime、no third-party product path/cycle |
| correctness | PASS | three native platforms、sanitizers、RISC-V codegen/QEMU |
| delivery | PASS | header factor + selected source/static/shared product journeys |
| usability/docs | PASS | natural C++ task shapes、recipes、pitfalls、current design/evidence |
| red team | PASS | dependency/API/FFI/lifetime/codegen/comparison attack；MEDIUM+ 0 |
| engineering performance | PASS | W1-WIN two frozen consumers、all strata 5% NI |

## Closure decision

- current decision：`CANDIDATE`；七项 aggregate gate 全 `PASS`、四项 MUST 全 `COMPLETE`、无 blocker，已满足
  `CANDIDATE -> VERIFIED` 输入；
- Formal 独立 `PENDING`，不属于工程关闭条件；terminal receipt、接受限制与 reopen triggers 待写。
