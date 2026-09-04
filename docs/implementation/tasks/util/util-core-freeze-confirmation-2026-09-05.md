# Util-core 冻结确认 — 2026-09-05

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对既有 `CLOSED` 产品的只读确认，不是一次无触发重开，不建立新的产品 candidate，也不替换
> [`util-core-v2-capability-completion`](util-core-v2-capability-completion.md) terminal record。当前状态真相仍是
> [`module-status.json`](../../module-status.json)；其中的 `freeze_confirmation` 只把本次确认绑定到
> 明确的仓库快照和本文证据。

## 绑定身份与范围

- confirmation：`util-core-freeze-2026-09-05`；marker：`FROZEN`；
- reviewed repository commit / tree：
  `bb79f6f4986bfb9ad26509b7a60e98d4a3d7e223` /
  `a699943f3c17f7f518904e68dbb2ae0dff431b24`；
- frozen util_core candidate / tree：
  `d42d992cf99232669f60422f30fb3b50ec9ec4eb` /
  `9944f099679503df632b6335e69154656164b30b`；
- candidate 到 reviewed commit 的七个公开实现头只有 `ref_counted.h` 中 `Allocator` → `AllocatorRef`
  一处 owner 名称注释校准；声明、算法、对象布局与生成代码不变；模块 tests 和 benchmark 为 zero diff；
- 中央 factor/product graph 与 Util package assertion 在 candidate 后调整，因此当前精确 factor、三平台
  source/static/shared/relocated Util 消费和全图组件合同均重新执行，不从旧 receipt 推断；
- product modifications in this confirmation：none。

## 质因数与依赖裁决

util_core 是当前内部组件图中的**质因数 factor**。中央图声明
`_hyle_declare_component(util_core PUBLIC_DEPS core ...)`，精确选择得到且只得到 `core,util_core`。
七个能力头都是 header-only、Core-only、没有自己的二进制生命周期、可选后端或第三方实现依赖；继续按
头文件拆 target 不会解除依赖环、隔离重依赖或降低实际编译成本，因为消费者本来就只 include 所需头，
反而会把一个最低层工具词汇拆成更多构建概念。

普通用户不选择内部 factor，而是按产品心智链接 `hyle::util`，再按任务 include
`hyle/util/<capability>.h`。聚合 Util 产品还包含 Registry/demangle 及其 Error、Platform、Mem、Container
闭包；这是用户产品聚合的传递依赖，不是 util_core 实现依赖。内部 Mem/Sched/Graph 则可精确依赖
util_core，避免为一个 SpinLock 或 pause hint 反向拉入整个 Util runtime。

## 七问终答

### 1. 第三方产品路径

`PASS`。util_core 的直接和传递产品闭包只有 Core 与 C++ 标准库/compiler ISA mechanism；公开头没有
Abseil、Folly、Boost、oneTBB、ggml 或 llama.cpp include/type，也没有 vendored object 或 runtime
dispatch。`<random>`、atomic、compiler intrinsic/asm 是标准或工具链机制，不是第三方产品依赖。

本轮 capability 扫描使用 2026-09-05 已刷新 remote-tracking identities：Abseil
`9e15d3d6cc19ae4703f3ad7fa413a789580ec1fc`、Folly
`6df3ef0f4d97f1bb7dd77b92080ff612a93ec894`、Boost
`cb94f33fcb13e90922905c2d8105508e71fa6db4`、oneTBB
`e00c68dcac0e83947c9b54f6d5304c0616c9639f`、ggml
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` 与 llama.cpp
`8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`。这些只用于能力/机制比较；P1 结果仍绑定原 contract 中的
Boost/oneTBB peer identity，不能用最新 tip 冒充性能 provenance。

### 2. 功能超集与推理消费

`PASS`。当前能力按自然任务而非 peer 符号并集组成：

| 用户任务 | 当前形状 | 完整边界 |
|---|---|---|
| 可复现统计随机流、bounded draw、shuffle、标准分布组合 | `Rng` / `RngState` | xoshiro256++、Lemire u32/u64、float/double、URBG、state/discard/jump/long-jump |
| 每线程普通非安全随机流 | `thread_rng()` | 首次 best-effort seed，稳态无锁；不承诺 crypto/跨 DSO stream identity |
| 词法 rollback/cleanup | `ScopeGuard` / `HYLE_DEFER` | once、dismiss/invoke、move 与 construction/transfer failure safety |
| 单指针侵入式强所有权 | `RefCounted` / `ThreadUnsafeRefCounted` + `Ref` | adopt/retain、copy/move/reset/release、casts、identity order/hash、custom destroy |
| 一次同步调用期借用任意 callable | `FunctionRef<Sig>` | 两机器字、function/object/member invoke、value/void/reference、`noexcept` |
| required 且可 rebind 的 raw-pointer borrow | `NotNull<T*>` | 一机器字、const/base/void/function pointer、identity/hash；不拥有 lifetime |
| 极短线程临界区和 atomic polling hint | `SpinLock` / `cpu_relax` | Lockable、TTAS/backoff、x86/Arm/RISC-V hint；不提供公平、parking 或 fiber wait |
| 编译器本地静态诊断标签 | `type_name/type_hash` | constexpr spelling/hash；不作为 wire、plugin、持久或无碰撞 identity |

llama.cpp 当前相关真实形状仍是采样器的标准 RNG，ggml 当前相关形状仍是 CPU threadpool polling hint；
HYLE 的 URBG 互操作和 `cpu_relax` 已覆盖自有推理引擎需要的底层任务。signed distribution、Gaussian/
discrete distribution 与 ranges algorithm 直接由标准库组合 `Rng`，安全随机由 Platform 承载，token
top-k/top-p 等分布变换属于推理 owner；这些不是应复制进 util_core 的半项能力。最新参考增量没有产生
新的共同 `MUST`，未处置 `MUST` 为 0。

### 3. 现有模块与加速设施复用

`PASS`。逐项反查如下：

| Needed mechanism | Existing candidate | Verdict | Reason |
|---|---|---|---|
| compiler attributes、fatal invariant、lifetime annotation、macro hygiene | Core | `REUSE` | 直接使用 `HYLE_FORCE_INLINE`、`HYLE_FATAL_ASSERT`、`HYLE_LIFETIME_BOUND`、`HYLE_NO_UNIQUE_ADDRESS` 与 Core preprocessor contract |
| callable invocation、iterator/distribution、atomic 与 ownership vocabulary | C++ standard library | `REUSE` | 使用 `std::invoke`、URBG/distribution、iterator、atomic、comparison/hash；不复制标准通用设施 |
| secure entropy | Platform | `BORROW/ROUTED` | `thread_rng` 明确只是 header-only best-effort statistical seed；安全 bytes 走 `platform::secure_random`，反向依赖 Platform 会成环 |
| fallible borrow、allocation/constructor failure | Error / Mem / owning module | `BORROW/ROUTED` | `Result<T&>` 与 allocator-aware factory 由上层 owner 完整表达；util_core 不为 `Ref` 引入 Error/Mem 反向边 |
| owning/async callback | standard / owning module | `BORROW/ROUTED` | `FunctionRef` 只承载同步 call-only borrow；存储、排队、取消必须由 owner 持有 callable lifetime |
| blocking/fair/RW/fiber synchronization | standard / Sched | `BORROW/ROUTED` | `SpinLock` 只拥有 compact short-lock 合同；park/wake/deadline/cancel 不是同一 primitive |
| SIMD bulk random/fill | SIMD / inference owner | `NOT_APPLICABLE` | 当前公开任务是 scalar URBG/state；没有跨 buffer 的已准入布局，强加 SIMD dispatch 会增加常态状态和依赖 |
| Container、reclamation、topology/NUMA、clock | 相邻 HYLE 模块 | `NOT_APPLICABLE` | 七个 primitive 不拥有这些数据结构或 runtime decision；没有第二套同义机制 |

模块内部的能力也不应为了“自我复用”强行互套：owning `ScopeGuard` 不能换成 borrowed `FunctionRef`，
FunctionRef 的 object/function union 也不是 `NotNull<T*>` 的单一 raw-object-pointer合同。生产反向扫描确认
Mem/Sched/Graph 正确复用 `SpinLock/cpu_relax`；其它新词汇是否迁移由各 consumer 自己的生命周期审计
决定，不能越界改写已关闭模块。没有发现 material 重复机制、遗漏的加速设施或第二 runtime authority。

### 4. 性能与复杂度

`PASS / P1_ENGINEERING`，material margin 为 5%，但不宣称 universal SOTA。保留的
`util-core-p1-v2` bundle 已重新执行 verifier，结果为
`QUALIFIED / EXACT / CURRENT`、`paths_verified=true`、
`current_certification_policy_eligible=true`；manifest、summary、observations 与 contract-lock SHA-256
分别为 `d42e407a730bd65d7afd84a6e5904eba9f110b4462fc0cffaf5031fadde48dd2`、
`9dd623ed2457d9b3277be435ef7aa5c81fc2af2c10c473df3875d5a2e53ffe67`、
`6aeb3d8bb25e6d3153171d57f4eaf835a1cedb56e6c06d051444d7254c212d01` 与
`ecf4f4760de7d2134cd0694b13931dd3a519ba101f24cc24905e140cab0892b8`。

冻结的 W1-WIN 同合同结果保持：Ref owning handoff 对 Boost 为约 10% `WIN`，短状态转移 SpinLock 对
oneTBB 为 `PARITY`，全部 strata 通过 5% non-inferiority。当前 optimized MSVC probe 仍生成 atomic
`lock inc`、thread-confined plain `inc`、NotNull pointer identity 与 x86 `pause`，没有 wrapper、DLL hop、
allocation 或 runtime dispatch。其余 primitive 采用 P0 机制裁决：RNG draw/range expected O(1)、shuffle
O(n)、Ref/FunctionRef/NotNull/ScopeGuard O(1)、type name 为 compile-time view；SpinLock 在争用下没有
有界公平性，因此只准入短低争用合同。候选后的实现、benchmark、Core compiler contract 与锁定 peer
均无 material 漂移，不重跑纳秒点值。

### 5. 旧 C ABI / Native FFI

`CLEAN`。没有 C header/schema/bridge、undecorated export、managed sample 或 Native FFI option。这里是
模板、inline ownership/borrow 与 compiler-local type spelling，不是自然稳定的跨语言 runtime seam；HYLE
当前只承诺匹配 headers/toolchain/STL 的 C++ source/static/shared 消费。

### 6. 构建与交付

`PASS`。util_core 是 INTERFACE factor，不制造虚构 static/shared binary；七个头在同一 factor inventory
中自包含并随 Util 产品安装。内部精确 factor 只解析 `core,util_core`，且故意拒绝 `HYLE_INSTALL=ON`；
安装是完整产品 journey。普通用户三种形态都链接 `hyle::util`，内部模块才使用精确 factor。

### 7. C++ 易用性与文档

`PASS`。[`user/util`](../../../user/util.md) 先按真实需求说明该选 RAII/ScopeGuard、Ref/shared_ptr、
FunctionRef/owning callable、NotNull/reference/pointer/Result、Rng/secure random、SpinLock/mutex/Sched，随后给出
可复制用法和 lifetime、null、thread、exception、OOM、contention、DSO 与 type-hash 陷阱。
[`util design`](../../../architecture/util-design.md) 负责长期边界、复杂度和 factor/product 设计；历史 task
只保留已发生事实。公开 API 没有 `fast/light/bench` 分叉、benchmark metadata 或测试专用入口。

## 当前快照验证

Windows 权威工作树中的无关既有修改没有进入 reviewed commit；WSL/macOS 使用该提交的 disposable native
copy，远端内容没有回写产品。

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| exact util_core factor | resolved=`core,util_core` | same | same | `PASS` |
| Release unit + self-contained headers | `2/2 PASS`；binary 内 80/80 case | `2/2 PASS` | `2/2 PASS` | `PASS` |
| optimized P0 codegen | atomic/local/borrow/pause probe `PASS` | candidate evidence retained | candidate evidence retained | `PASS` |
| Util source + relocated shared/static consumer | every journey `PASS` | every journey `PASS` | every journey `PASS` | `PASS` |
| current central component contract | `PASS` | selected journey | selected journey | `PASS` |
| retained P1 evidence | verifier `QUALIFIED / EXACT / CURRENT` | not a timing claim | not a timing claim | `PASS` |

候选的 sanitizer、RISC-V compile/codegen/QEMU 证据继续有效：相关 util_core tests/实现与 Core compiler header
为 zero runtime diff。本次没有把未重跑项写成新的跨平台 timing 或 Formal 证书；Formal 保持 `PENDING`。

## 冻结裁决与 LOC

没有发现新的准入能力缺口、复杂度或性能缺陷、benchmark-shaped API、内部复用缺口、易用性/文档错误、
旧 C ABI 或 build/package integration 缺陷。util_core 因此保持 `CLOSED`，并在 reviewed repository snapshot
上附加 `FROZEN` 标记。只有 registry 中既有 `reopen_triggers` 的新事实才能解除；peer API 数量、LOC、
单次微基准或无新事实的“再看一遍”本身不能。

`python scripts/loc-report.py --module util_core --commit
bb79f6f4986bfb9ad26509b7a60e98d4a3d7e223 --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 439,453 |
| 自实现不含测试 | 70,651 |
| util_core 实现 | 967 |
