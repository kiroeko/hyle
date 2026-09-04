# Error 冻结确认 — 2026-09-04

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对既有 `CLOSED` 产品的只读确认，不是一次无触发重开，不建立新的产品 candidate，也不替换
> [`error-v5-capability-completion`](error-v5-capability-completion.md) terminal record。当前状态真相仍是
> [`module-status.json`](../../module-status.json)；其中的 `freeze_confirmation` 只把本次确认绑定到
> 明确的仓库快照和本文证据。

## 绑定身份与范围

- confirmation：`error-freeze-2026-09-04`；marker：`FROZEN`；
- reviewed repository commit / tree：
  `94acc0c260a899980b690aa96b5dbfbe57ea07c7` /
  `457daeb77df3915f6ae969b41e8cadd1edfb41d7`；
- frozen Error product candidate / tree：
  `d61f0710ec0e465b31d458257e2587fdfe37ed52` /
  `305b2cb5947627dfb5b9697c939205b6c26cc6b7`；
- candidate 到 reviewed commit 的 Error-owned public implementation、test、benchmark 与 package sample：
  zero diff；
- reviewed current-tree change outside those paths：中央 factor/product graph 与构建合同曾重排，因此当前
  source/static/shared 选择、安装头、动态符号和组件合同均重新执行，不从旧 receipt 推断；
- product modifications in this confirmation：none。Architecture 只把一段 v5 验证前的旧时态改为已完成
  事实和未来续签规则。

## 能力、接口与真实消费确认

Error 当前拥有同一套 recoverable-failure 语义：完整 canonical `StatusCode`、8B owning `Status`、UTF-8
message/native detail、copy/move/COW、`Update`/`IgnoreError`/`WithContext`、owning `Result<T>`、非空借用
`Result<T&>`、标准 expected-shaped observers/monads、传播宏、caller-buffer/owning/bounded dynamic
format，以及 errno、Win32、WinSock、HRESULT 和 `std::error_code` 冷边界。推理引擎的 factory、模型加载、
backend 初始化、registry borrow、任务图错误交接和运营诊断都能用这些自然任务形状表达。

当前 clean reference identities 仍为 llama.cpp
`c7bda030e7faee594dbe7550185e857351ad405d`（bundled ggml tree
`25e2241ed891c98cd26af5c47246c79b430a49cd`）、standalone ggml
`d4716378882593333721eb33f153144b6885caf2`、Abseil
`536d30bfea05a09c7561cbba749ca7f738d46821` 与 Folly
`80071a1f79e110b34a2cf7a3a4d01a59e823d697`。锁定 Abseil peer 到当前版本的 `Status`/`StatusOr`
实现零改动，只有一个 `statusor_test.cc` 变更；锁定 Folly 的 expected/error carrier 路径零改动。llama.cpp
增量中的新 assertion/fatal 使用仍属于 Core，不可恢复的 server exception spelling 仍由 host boundary
拥有，没有形成 Error 的新准入 `MUST`。

payload/cause/source-location、`MultiError`、runtime domain registry、async pending/cancellation、fatal、logging、
backend-specific translator 与 Native FFI 分别因没有共享消费者合同，或已有 Sched/Core/Obs/backend/host
owner 而不进入 Error。它们不是遗漏；具名消费者以后证明需要时，只按对应能力族局部重开。

公开接口仍按任务拆成 `status.h`、`result.h`、`format.h` 与 `system_error.h`。值、必有非空借用、可空
pointer/view、native error 和 standard adapter 的选择，deep const、temporary/lifetime、custom category/DSO
unload、OOM/no-throw、截断与传播宏陷阱都在 [`user/error`](../../../user/error.md) 就近说明。未发现
benchmark/fast-vs-slow seam、常态 benchmark metadata、旧 C header/schema/bridge、managed sample 或 Native
FFI option。

## 现有模块与加速设施复用裁决

> 2026-09-05 按通用 closure 第 3 问增补；复核基线为
> `59d8046b38d580081a716f166a496c4a13c5986d`，Error candidate 到该提交的 product/test/benchmark/package
> sample 仍为 zero diff。本节不修改产品或替换原冻结身份。

| Needed mechanism | Existing candidate | Verdict | Reason |
|---|---|---|---|
| fail-fast、compiler attributes、macro hygiene | Core | `REUSE` | `HYLE_FATAL_ASSERT`、`HYLE_FORCE_INLINE` 与 preprocessor contract 已直接复用 |
| value error / expected-shaped transport | Error / C++ standard library | `LOCAL + REUSE` | Error 是该语义 owner；`Result<T>` 复用 `std::expected`，format/char conversion 复用标准库，不再造通用 engine |
| Status heap/COW/refcount | Mem、Util-core `RefCounted` | `LOCAL` | Mem/Platform 位于 Error 上层；Status 是 8B tagged value、可变长单块 allocation 与 specialized COW，不能套用 CRTP object ownership 而不改变 layout/teardown |
| errno/Win32/WinSock/HRESULT 文本边界 | Platform | `LOCAL` | Platform 公开依赖 Error；这里的私有 OS call 是 bootstrap adapter，反向依赖会成环 |
| UTF-8 验证与短文本复制 | SIMD/simdutf/Container | `NOT_APPLICABLE` | 只处于冷 native/custom-error translation；上层 SIMD/Container 会成环并扩大所有 Error consumer，`memcpy` 由 compiler/CRT 提供平台优化 |
| clock、container、reclamation、scheduler | 相邻 HYLE 模块 | `NOT_APPLICABLE` | Error 的已准入任务不拥有这些机制或 runtime decision |

源码/include 与中央闭包反查没有发现 Error 私造已有 clock、allocator、container、CPU/topology、wait、hash
或 dispatch 机制。唯一看似可复用的 atomic refcount 与 UTF-8 路径都因产品表示、冷路径和依赖方向不等价；
不存在遗漏的 material 加速设施或第二 runtime decision authority。该项 verdict 为 `PASS`。这里的“尽量
复用”不要求最底层 Error 为提高引用数依赖其消费者；未来出现合法更低 factor 或 profile 证明冷路径成为
P2 时，才按既有 dependency/performance trigger 局部重开。

## 质因数裁决

Error 是当前内部依赖图中的**质因数 factor**。中央图声明
`_hyle_declare_component(error PUBLIC_DEPS core)`，用户 product 声明
`_hyle_declare_product(error FACTORS error)`；选择 Error 得到且只得到 `core,error`。Error 没有第三方产品
依赖，其四个公开头共同拥有一个 value-error representation、生命周期和二进制 seam。把 native adapter、
format 或 Result 再拆成 target 不会解除依赖环、隔离可选第三方或形成独立生命周期，反而会复制 Status
协议和增加交付心智。因此它在当前能力超集下不再做内部 factor 分裂；“质因数”不表示以后有新依赖事实时
永远禁止局部重构。

## 当前树验证

Windows 权威工作树中的无关既有修改没有触及 Error 路径；WSL/macOS 使用 reviewed commit 的 disposable
native copy，远端内容没有回写产品。

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| Release Error/tool contract | `20/20 PASS`；Error label `2/2` | `20/20 PASS`；Error label `2/2` | `20/20 PASS`；Error label `2/2` | `PASS` |
| public-header/self-contained compile | `PASS` | `PASS` | `PASS` | `PASS` |
| source + relocated shared/static consumers | every journey `PASS` | every journey `PASS` | every journey `PASS` | `PASS` |
| selected product closure | products/factors=`core,error` | products/factors=`core,error` | products/factors=`core,error` | `PASS` |
| current central component contract | `PASS` | selected journey | selected journey | `PASS` |
| docs / closure registry | `check-docs PASS`；`check-module-closure PASS` | repository contract | repository contract | `PASS` |

当前 Windows shared DLL 导出 22 个 MSVC decorated C++ symbol，与 frozen candidate 数量一致；没有 undecorated
C export。source/static/shared 的 C++ spelling 相同，SDK 仍要求匹配 headers、binary、toolchain/STL，且不
承诺稳定跨工具链 C++ ABI。

## 性能与冻结裁决

Error remains `P1 ENGINEERING PASS`，material margin 为 5%。保留的
`error-source-consumer-v2` bundle 已重新执行 deep verifier：manifest、summary、contract-lock 的 SHA-256
分别仍为 `12685d8f398d4f55c9f05c71d54a3f5054ee3b99f21147778dfe91644f8b7eff`、
`3303032a19b5cf036733ac2627626631f8460147b26432453cb60e5a6421cd45` 与
`04b41774b914124c3189083dee962a46819e8312256bcd5ad51cb2fb5ca10612`；结果为
`qualification=QUALIFIED`、`paths_verified=true`、`provenance_compatibility=EXACT`、
`current_certification_policy_eligible=true`。候选后的 Error product、benchmark 和 locked peer 实现没有
material 改动，当前 build-system diff 只重排其他 factor/catalog，不改变 Error 源、compile option 或 P1
合同。

五个同合同 consumer 的全部四种 order/placement strata 仍通过 5% non-inferiority：TaskGraph success 为
`PARITY`，三个 owning failure handoff 与 fixed-table `Result<T&>` borrow 为 `WIN`。冷 translator、bounded
format 和 context storage 保持 O(1)/bounded 或按最终文本字节线性的既定复杂度。该结论只签 W1-WIN、
MSVC 19.50、锁定 Abseil 与五个 workload；不声称跨平台、所有操作、端到端推理、Formal 或 universal SOTA。

没有发现新的准入能力缺口、复杂度/性能缺陷、benchmark-shaped API、易用性缺陷、旧 C ABI、文档事实错误
或 build/package integration 缺陷。Error 因此保持 `CLOSED`，并在 reviewed repository snapshot 上附加
`FROZEN` 标记。只有 registry 中既有 `reopen_triggers` 的新事实才能解除；peer API 数量、LOC、新 benchmark
或无新事实的“再看一遍”本身不能。

## Snapshot LOC

`python scripts/loc-report.py --module error --commit
94acc0c260a899980b690aa96b5dbfbe57ea07c7 --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 439,155 |
| 自实现包含测试 | 111,779 |
| 自实现不包含测试 | 70,651 |
| Error 实现 | 2,178 |
