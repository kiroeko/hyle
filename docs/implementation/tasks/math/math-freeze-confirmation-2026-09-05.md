# Math 冻结确认 — 2026-09-05

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对既有 `CLOSED` 产品的只读 replay，不是无触发重开，不建立新 product candidate，也不替换
> [`math-v2-checked-arithmetic`](math-v2-checked-arithmetic.md) terminal record。当前状态真相仍是
> [`module-status.json`](../../module-status.json)；其中的 `freeze_confirmation` 只把本次新标准确认绑定到
> 明确仓库快照和本文证据。

## Replay 身份与边界

- task / reviewer / date：`math-freeze-2026-09-05` / Codex primary reviewer / 2026-09-05；
- task kind / verdict：closed-product replay / **`CONFIRMED`**；
- reviewed closure round：`math-v2-checked-arithmetic`；closure task：
  [`math-v2-checked-arithmetic`](math-v2-checked-arithmetic.md)；
- frozen product candidate / tree：
  `1a9e680cb5a25ccc0c20786f4635625eef2d7ab0` /
  `010323120bc6eb07b5a07e7d3e6de1725968d314`；
- closure end / registry anchor：
  `0ab83f780087802699bae9cb2a330c74b832922a` /
  `a1c771d7674bb5c7313ee27781ce0f09da48487e`；
- replay-reviewed repository commit / tree：
  `ed39ba8df2202fa55d577241e5abf40871d92ad9` /
  `2fbb9e7fade1c54bf06d2c29440fb1caa666c6f9`；
- executable validation snapshot：
  `cfc470fccebfbc1afe4c57b4466e5a1863fa8b95` /
  `dfeeb9a1134510995e1d3ff57cd506114778ee93`。它与 reviewed commit 之间只差本轮两份 Math 文档的
  使用说明，产品、tests、CMake、sample 和工具均为 zero diff；
- candidate 到 reviewed commit 的 `include/hyle/math`、`src/math` 和 package sample 为 zero diff；只有
  两个 test-only warning-clean commits 调整 oracle fixture；
- replay budget：120 分钟；只允许能力/依赖/API/正确性/P0/交付/文档/红队确认，不修改产品或公共仪器；
- product modifications：none。文档只补充 `std::in_range` 与逐维 product 的既有组合路线。

修改 registry 前的明确快照保存在 canonical checkout 外：
`C:\hyle-scratch\math-freeze-20260905\module-status-before-freeze.json`，SHA-256
`94139DD47A475569FF45901E77626122D85C4A2058CFCD0DA2D62488038B85F8`。

## 最新参考与真实消费

本轮先把 clean reference checkout 快进到各自跟踪分支，再锁定对象：

| Role | Identity | 本轮结论 |
|---|---|---|
| llama.cpp | `427291b5b34cd914a31b3fd3b61a68f6184f4b9f`；bundled ggml tree `9bf0d932917ba3ccfe7e3d9fd1723390dc870ec0` | tensor/backend 仍反复执行 shape product、byte sizing、range end、ceil/round/align；没有新 scalar semantic family |
| standalone ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | 独立实现确认相同推理 sizing 需求 |
| Folly | `54e827f8cf5b93569df581a0031caf8930968d60` | `CheckedMath.h` 相对 v2 pinned identity 为 zero diff；binary checked primitive 与显式组合仍是成熟路线 |
| CCCL | `486de1c44daf7d1a343cbbf0f5477e9bea8ad600` | checked/mixed/saturating numeric surface 仍完整；新增主要是 `isclose` 与 saturation 实现细化，没有形成 HYLE 新共同 `MUST` |

llama.cpp/ggml 的具体正常操作序列是“读取 signed tensor extents → 验证并转换到 size type → 逐维
checked product → checked element-size multiplication → checked base/end → 分配、映射或 launch”。这条
`PROVISIONAL_ARCHETYPE` 已由现有七个入口与标准 C++ 组合完整覆盖；它不冒充当前尚不存在的 HYLE
推理引擎 trace。

## 能力超集终审

| Capability family | 真实任务与必要语义 | 当前自然路线 | Coverage / action |
|---|---|---|---|
| checked integral add/subtract/multiply | shape、stride、offset、byte count；same-type non-bool；成功精确、overflow/underflow 可发现 | `try_add` / `try_subtract` / `try_multiply` | `COMPLETE / KEEP` |
| non-negative ceil division | block、tile、thread partition；零/负 divisor 与负 signed value 可发现 | `try_ceil_div`；可信 invariant 用 `ceil_div` | `COMPLETE / KEEP` |
| representable multiple round-up | padded extent；非正 multiple 与越界结果可发现 | `try_round_up_to_multiple`；可信 invariant 用 `round_up_to_multiple` | `COMPLETE / KEEP` |
| mixed width / signedness conversion | 模型 `int64_t` extent 转 allocation size | C++23 `std::in_range<To>` + 显式 cast，再调用 same-type helper | `ROUTED / KEEP`；不重包 `checked_cast` |
| multi-dimensional product / `base + count * stride` | 顺序 sizing，拒绝第一个不可表示中间值 | 重复 binary `try_multiply`，再 `try_add` | `ROUTED / KEEP`；不固化 arity/layout |
| generic checked div/mod | 除零与 signed `min/-1`，但当前没有统一正常消费 | 非负 partition 走 `try_ceil_div`；其它领域 owner 先验证自己的 divisor/rounding | `OUT_OF_SCOPE`；peer 名称不能替代消费合同 |
| saturation/clamp/fixed-point/mixed-result | policy、result type、device contract 依消费域变化 | 标准/CCCL、SIMD 或领域 owner | `OUT_OF_SCOPE`；当前未处置 `MUST` 为 0 |
| ordinary bit/scalar/floating math | 标准 scalar 合同 | `<bit>`、`<cmath>`、`<numbers>`、`<numeric>` | `ROUTED / KEEP`；不重命名标准库 |
| lane/row/buffer/model math | dtype、accuracy、dispatch、layout、fusion | SIMD / Compute / consumer | `OUT_OF_SCOPE`；Math 不作为 SIMD 差集 |

Folly 的 variadic add/mul-add 可以由现有 binary primitive 等价组合；CCCL 与 ggml 的真实 sizing 也采用
显式组合。没有第二个 strong peer 加真实 journey 证明固定 arity convenience 是独立能力，故不增加
`try_multiply_add` 或 variadic product。CCCL 的 device、heterogeneous、saturating surface 服务另一套
result/policy/backend 合同；把它按 API 数量复制到 host scalar control plane 会制造错误形状。

## 质因数、依赖与内部复用

Math 继续是不可再拆的 header-only **质因数 factor**。中央图声明
`_hyle_declare_component(math PUBLIC_DEPS core)` 与 `_hyle_declare_product(math FACTORS math)`；三平台
精确选择都解析为且只解析为 `core,math`。把 checked 与 fail-fast 七个小函数再拆 target 不解除依赖环、
不隔离重依赖或 binary 生命周期，反而增加用户和维护成本。

| Needed mechanism | Existing owner | Verdict | Reason |
|---|---|---|---|
| release-surviving invariant | Core | `REUSE` | `ceil_div`/`round_up_to_multiple` 直接复用统一 fatal contract |
| scalar vocabulary、range query | C++23 standard library | `REUSE` | concepts、optional、limits、bit representation 与 `std::in_range` 已拥有语义 |
| runtime overflow lowering | supported compiler / ISA | `LOCAL MECHANISM` | builtin 或 full-width multiply intrinsic，不是第三方 product dependency |
| recoverable diagnostic payload | Error | `NOT_APPLICABLE` | 纯值 predicate 用 `optional<T>`；没有 OS/context payload 或 allocation |
| address alignment / allocation | Mem | `BORROW/ROUTED` | Math 只做整数 multiple；pointer provenance、power-of-two alignment 和 allocation 归 Mem |
| vector/device/model math | SIMD / Compute / consumer | `BORROW/ROUTED` | accuracy、dtype、dispatch、kernel/layout policy 不下沉 scalar factor |
| Platform/Time/Obs/Container/reclamation | adjacent factors | `NOT_APPLICABLE` | 七个无状态值函数不拥有这些机制，也不存在可复用的同合同 hot facility |

全树反查没有 HYLE 产品语法 consumer；package sample 只证明交付。Mem-vm、Platform-fs 等尚待其自身
新标准轮的 owner 仍有局部 overflow guard，这构成后续 adoption 输入，不构成 Math 产品缺口，也不授权
本 replay 越界修改它们。冻结的 Time 饱和 deadline、Obs limiter 与 Error representation sizing 都有不同
failure/policy 合同，不是应强行改写成 Math optional 的重复 authority。

产品第三方依赖为零；`<intrin.h>` 是 MSVC toolchain mechanism，GoogleTest 只进入 tests。Math 不直接或
间接把 Folly、CCCL、llama.cpp/ggml、Highway 或 xsimd 带入产品闭包。

## C++ 形状、Native FFI 与文档

`scalar.h` 的七个 public template 按任务分为 checked `try_` 和 trusted fail-fast 两条路线。名称直接表达
operation，返回 `optional<T>` 没有 out-pointer/null contract；same-type constraint 主动拒绝隐式
signed/unsigned promotion。[`user/math`](../../../user/math.md) 已给出 tensor sizing、逐维 product、
`std::in_range`、mul-add 组合、ceil/round 选择，以及 typed literal、unsigned subtract、pointer alignment、
negative rounding 和 floating tolerance 陷阱。[`math design`](../../../architecture/math-design.md) 记录长期
scalar/SIMD/Compute owner 边界。公开面没有 `fast/light/bench` 分叉、测试 metadata、umbrella namespace
或 internal compiler seam 泄漏。

旧 C ABI / Native FFI 为 **`CLEAN / N/A`**：产品没有 C header/schema/bridge、未装饰 C export、managed
sample、专用 option/target/package metadata 或只为 FFI 支付的 layout/indirection。codegen probe 的
`extern "C"` 只稳定 test-only 反汇编符号名，既不安装也不链接到 SDK。模板能力由匹配工具链的 C++
consumer 在自身 TU 实例化；header-only factor 不伪造动态或静态 Math binary。

## 正确性与独立红队

red-team reviewer / date / bounded budget：Codex primary reviewer / 2026-09-05 / 30 分钟。
风险矩阵重新攻击了 signed/unsigned 8/16/32/64-bit zero/min/max、四个 overflow 方向、unsigned
underflow、signed `min * -1`、零/负 divisor、最大值 round-up、`constexpr`、bool/mixed-type rejection、
MSVC full-width multiply、portable fallback、optional failure、public-header self-containment 和 package
实例化。现有 tests 对 int8/uint8 全输入空间用 wider exact oracle 对拍，对全部标准固定宽度覆盖 boundary；
WSL 又在 UBSan 下执行。

| Risk | Evidence | Verdict / limitation |
|---|---|---|
| exact result and arithmetic boundary | exhaustive 8-bit oracle + 8/16/32/64 boundary/constexpr tests | `PASS` |
| UB / compiler-specific fallback | WSL GCC 14.2 UBSan；MSVC x64 与 Apple arm64 runtime implementation | `PASS` |
| invalid rounding input | zero/negative divisor/multiple、negative signed value、unrepresentable result | `PASS` |
| allocation/OOM/lifetime/move/cancel | pure inline value functions without owner or async work | `NOT_APPLICABLE` |
| concurrency/reentrancy | no global、TLS or shared state | `PASS` by structure |
| ASan/TSan | no memory ownership or synchronization protocol | `NOT_APPLICABLE`；未运行不写 PASS |
| public header / type constraints | three-platform self-contained compile and compile-time constraints | `PASS` |
| downstream HYLE product consumer | none exists | `NOT_APPLICABLE` for execution；explicit provisional archetype retained |

红队 finding 只有两项：`MATH-DOC-01`（low）已在 reviewed commit 补成可复制的 mixed-width/multi-dim
recipe；`MATH-MSVC-FRAME-02`（informational）确认 Core 丰富 fatal diagnostic 使 opaque fail-fast probe 的
MSVC reserved frame 从旧证据的 40 bytes 变为 216 bytes，但正常路径仍只增加一对固定 `sub/add rsp`，
没有 frame memory traffic、allocation、额外 divide 或 indirect call。checked 路线无 frame；GCC 将失败
路径拆到 cold section，AppleClang 正常路径直接 return。没有 profile/material contribution，故这不是
API/算法 trigger；若真实 consumer 证明其达到 material contribution，registry 已有 reopen trigger。

剩余 medium-or-higher finding、correctness/security blocker、未处置 `MUST`、重复 decision authority 和
benchmark-shaped surface 均为 0。

## P0 性能裁决

工程 tier 保持 **`P0 STRUCTURAL PASS`**；timing peer comparison 明确为
**`NOT_APPLICABLE`**，不宣称 nanosecond ranking 或 universal SOTA。

- 所有入口 O(1)、零 allocation/lock/syscall/atomic/global registry/常态 metadata/runtime dispatch；
- checked add/sub 的 runtime 只有一次目标 arithmetic operation 和 overflow predicate；checked multiply
  在 x86 使用 `mul/imul` full-width/overflow flag，在 arm64 使用低半积加 `umulh/smulh`，均无 divide、
  helper call 或 DLL hop；
- ceil/round 每次 opaque runtime divisor 至多一次 integer divide；checked 入口没有 failure sink；
- constant divisor 和 constexpr 可由消费 TU 继续优化；未知 compiler 的 portable multiply fallback 可以
  先比较再乘并可能使用 divide，这是已登记的非 shipping limitation。

当前 optimized probe object SHA-256：W1-WIN MSVC 19.50 x64
`254868DB24C2318FFD66D2172E48CA9DC6F3C7587F7D73F7731DED6348857C4E`；W1-WSL GCC 14.2 x86-64
`06c17872939fccb87ceaf5b50177cc2fded1d82dc424b7a37c1b295cec0565fd`；M1-MAC AppleClang 21 arm64
`b6e6a4724a7c633e5745c1a40d3ad3ff963c53055045c8280e6cc5d7a87308c7`。三者都复现以上结构。

## 当前树构建与交付

所有可执行验证都来自 reviewed product-equivalent 的 clean/disposable snapshot；Windows 权威工作树中的
无关既有 Container/performance 修改没有进入。WSL 与 Mac 使用原生临时副本，不回写产品。

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| exact Math factor | resolved=`core,math` | same | same | `PASS` |
| unit + self-contained header | Release/Werror `2/2` | RelWithDebInfo/Werror/UBSan `2/2` | Release/Werror `2/2` | `PASS` |
| optimized P0 code shape | current object inspected | current standalone Release object inspected | current object inspected | `PASS` |
| source consumer | `1/1` | `1/1` | `1/1` | `PASS` |
| relocated shared SDK consumer | `1/1` | `1/1` | `1/1` | `PASS` |
| relocated static SDK consumer | `1/1` | `1/1` | `1/1` | `PASS` |
| installed inventory | Core + Math headers/CMake/license；zero Math binary/vendor | same | same | `PASS` |

中央 component/product declarations、target 的真实 `PUBLIC core` edge 与 package sample 一致。自最近一次
Build v12 全图 component contract 到 reviewed commit，CMake/build/package paths 为 zero diff；本轮又在
三端真实解析并消费 exact `math` closure，因此没有用历史图替代本模块当前交付门。

## 冻结裁决与 LOC

本轮没有发现新的 admitted capability gap、错误复杂度、material performance defect、benchmark-shaped
API、内部复用遗漏、C++ 易用性缺陷、文档错误、旧 C ABI 或 build/package integration defect。Replay
verdict 为 **`CONFIRMED`**：`math` 保持 `CLOSED`，并在 reviewed repository snapshot 上取得新标准
**`FROZEN`** 标记。只有 registry 既有 reopen trigger 的新事实才能解除；peer symbol 数量、LOC、一次
微基准或没有新事实的“再看一遍”本身不能。

`python scripts/loc-report.py --module math --commit
ed39ba8df2202fa55d577241e5abf40871d92ad9 --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 442,388 |
| 自实现不含测试 | 70,870 |
| Math 实现 | 222 |
