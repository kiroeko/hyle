# Core 对 module-closure-v1 的控制样本回放

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

> **记录性质：`HISTORICAL CONFORMANCE AUDIT`。** 本页记录 2026-08-26 对既有 Core `CLOSED`
> 结论的只读产品回放和方法修复，不声明当前状态；唯一当前状态源是
> [`module-status.json`](../../module-status.json)。本轮没有重构或修改 Core 产品代码。

## 审计合同与边界

- component / round：`core` / `core-method-replay-v1`；
- reviewed product state / candidate：`CLOSED` / `b241a228d3311f0faa65cde68123de7da3688c2f`；
- reviewed repository commit：`1a465e23ea69a6e613840cf8d30d33054463ae1c`；
- 进入原因：用一个已经有充分技术证据的底层模块反证闭环框架，而不是寻找理由再次重构；
- normal consumer：HYLE native C++ 模块通过 public headers 使用 compiler/invariant/cache primitives，
  以及 source/relocated static/shared Core package consumer；
- 范围：能力来源、API/owner/failure、依赖边、P0 成本结构、候选身份、交付、文档、红队和停止条件；
- 非目标：新增 Core 能力、改变 API/layout/codegen、为 Core 制造 nanobenchmark、重新宣称下游收益；
- tier：`P0 / closure-correctness-v1`；计时预算为 0，material nanobench margin 不适用；
- 回放预算：90 分钟、一个 fresh Windows Core build、一个 source/static/shared journey 集、三个并行
  只读 reviewer、零产品修改；旧 Core 轮没有可证明的 v1 预算，本文不事后伪造；
- peer snapshot：本轮实际复看本地只读镜像 Abseil
  `81c993af4fb88b3a51829a796df02251fa27102f`（`absl/base/{attributes,optimization,prefetch}.h`、
  `internal/hardening.h`）、Highway `239071390b18ca70f8fadde90ebefa6d8b12cb69`
  （`hwy/{base,cache_control}.h`）与 Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`
  （`folly/CPortability.h` 和 cache-layout 使用点）；每项裁决最多使用两个实现 peer。compiler vendor
  文档只核 spelling/退化，不作为第三个实现 peer。本轮没有借 peer 引入新 Core 能力；它们用于反证
  现有 portability/prefetch/fail-fast 形状没有明显能力或复杂度缺口；
- 停止条件：product diff 为零、候选与 raw evidence 身份成立、当前受影响门通过、独立红队无产品
  trigger；方法 finding 可以修框架/证据，但不能越界改 Core。

## 独立复现的能力裁决

| ID | demand kind / exact source | normal shape | semantic / artifact / delivery owner | 复核裁决 | negative space |
|---|---|---|---|---|---|
| CORE-INV-01 | real consumers：Error `Result`、Container B-tree 与其他 fatal invariants | Debug/Release 都存活的 fail-fast，可借用静态消息 | core / core headers / build contract | `KEEP`；条件一次求值，消息只在失败时求值 | recoverable 失败必须走 error；不是 logging/crash reporter |
| CORE-COMP-01 | real consumers：SIMD、container、mem、error 等 native hot paths | 跨 MSVC/GCC/Clang 的 inline/noinline/restrict/branch/loop/layout seam | core / core headers / build contract | `KEEP`；标准 C++ 已自然覆盖的旧 macro 不恢复 | hint 不是保证；optimizer promise 不能验证输入 |
| CORE-LAYOUT-01 | real consumers：shared pool、SPSC/并发 shards；标准 interference practice | compile-target constructive/destructive recommendation | core / core header / build+platform | `KEEP` | runtime cache topology 归 platform；不序列化为跨 build ABI |
| CORE-PREFETCH-R | real consumers：Swiss/vqsort | single-address temporal read intent | core / core header / consumer module | `KEEP` | 具体收益、距离和带宽污染由 owning workload 证明 |
| CORE-PREFETCH-NW | peer-general：Abseil/成熟 compiler primitive；当前无生产调用 | NTA read / write intent | core / core header / future consumer | `KEEP` as peer-derived `GENERAL`，不冒充已有 consumer | 不因此扩张 selector、metadata 或 benchmark-only API |
| PKG-VERSION-01 | build/package 与三种 package journey | header identity constants | build semantic / root header artifact / package delivery | `BORROW/ADJACENT` | 不制造 version-only runtime 或 FFI |
| CORE-FFI-01 | foreign runtime 假设，无成立 runtime task | compile-site attribute/layout/prefetch | N/A | `REJECT` | Native FFI 只准入有真实 type-erased runtime task 的模块 |
| CORE-INTR-01 | time 的 allocation-free list | independent `container_intrusive` target | container / container header / build graph | `MOVED`，Core 不再拥有 | 不让 time 依赖完整 container/mem stack |

公开入口、用户选择、UB 前置条件、borrow/lifetime、ABI 与失败边界在
[`user/core`](../../../user/core.md) 可直接找到；仓库扫描没有发现已删除旧 Core API 的产品残留。

## 当前候选与验证复放

`b241a22..1a465e2` 对 `include/hyle/core/**`、`include/hyle/version.h`、`src/core/**`、Core tests、
codegen/symbol audit 和 package sample 的产品输入为零 diff。候选 tree 复算为
`e73472e2cefedd70e41eca6243871b2343db7d74`；历史 manifest 的 25/25 raw SHA-256 匹配。

| Gate | Verdict | 本轮新跑 / 复核 | 限制 |
|---|---|---|---|
| current Windows correctness/codegen | `PASS` | fresh VS 18/MSVC 19.50 Core-only Release；16/16（含 4 Core、header self-contained、codegen/parser、closure/LOC tooling） | 本轮未把全库、compute、graph、Vulkan 拉入 |
| source journey | `PASS` | isolated parent `add_subdirectory` consumer 1/1；唯一产品 target 为 INTERFACE Core | current Windows |
| relocated shared SDK journey | `PASS` | installed consumer 1/1；Core binary artifact=0；imported target 为 INTERFACE | current Windows |
| relocated static SDK journey | `PASS` | installed consumer 1/1；Core binary artifact=0；imported target 为 INTERFACE | current Windows |
| affected representative consumers | `PASS` | candidate 后发生相关头变化的 `container_intrusive` + `time`，10/10 | 影响集选择，不冒充全库重新签署 |
| historical platform evidence identity | `PASS` | candidate tree、25 raw hashes、Windows/Linux/macOS manifest identity | hash/identity 复核，不冒充本轮重跑 Linux/macOS/sanitizer |
| P0 cost structure | `PASS` | normal path compile-time/inline O(1)，无 allocation/lock/syscall/registry/常态 metadata；release codegen intent 已绑定 | consumer hint 收益仍归 consumer |
| docs / state | `PASS` after fixes | closure checker、deep Git identity、docs link/whitespace checks | 语义过时仍需 reviewer；链接器不能理解自然语言 |
| independent red team | `PASS` for product | 三个 reviewer 分别审 capability、verification、adversarial evidence | 不追认旧轮曾具备 v1 reviewer ledger |

Correctness 风险矩阵中：normal、header isolation、fatal failure、release structure、delivery 为 `PASS`；
empty/zero 由 constexpr/layout tests 覆盖；Core 无 owning runtime state，因此容量/OOM、move/destroy、并发
state、取消/关闭为 `NOT_APPLICABLE`；历史 ASan/UBSan 与真实 Apple arm64 evidence 只作身份复核。

本轮代表命令：

```text
cmake -S . -B .scratch/core-method-replay-win -G "Visual Studio 18 2026" -A x64
  -DHYLE_COMPONENTS=core -DHYLE_BUILD_TESTS=ON -DHYLE_BUILD_BENCH=OFF
  -DHYLE_INSTALL=OFF -DHYLE_STATIC=ON -DHYLE_WERROR=ON
cmake --build .scratch/core-method-replay-win --config Release --parallel 2
ctest --test-dir .scratch/core-method-replay-win -C Release --output-on-failure --parallel 2
python scripts/check-docs.py
python scripts/loc-report.py --module core --commit b241a228d3311f0faa65cde68123de7da3688c2f --self-check --json
```

## 红队发现与处置

Reviewers：Planck the 2nd（capability）、Euclid the 2nd（verification）、Banach the 2nd（independent
red team），由 `/root` 集成；日期 2026-08-26；每个并行 reviewer 的 v1 默认上限为 30 分钟。本页不把
旧任务中“无 MEDIUM+”的一句话反称为可审计的 v1 红队。

| Finding | Severity | Disposition | Evidence / result |
|---|---|---|---|
| registry 的 `end_commit=b923105` 早于真正签署提交 | high / evidence | `fixed` | 改为 `49512b7`；以只读 Git 检查 task、candidate 与 ancestry，并记录 `1a465e2` registry anchor |
| candidate 全库 LOC 记为 297,471，`cloc 2.08` 复算为 296,932 | high / evidence | `fixed` | 同一 `b241a22` 快照复算；后三项 141,867 / 96,105 / 233 原值正确 |
| immutable LOC 的未锚定 regex 会把 `third_party/**/src` 算成自实现 | high / instrument | `fixed` | commit tree 只物化一次，并把 Git 精确相对文件表交给 cloc；新增 nested vendor 反例 |
| 旧 task/performance 页复制当前 `SIGNED/QUALIFIED` 状态 | medium / docs | `fixed` | 历史页面改为 `HISTORICAL`，当前状态只由 registry/status 投影表达 |
| `core-design` 仍称三平台/package/red team 未完成 | medium / docs | `fixed` | architecture 只保长期设计，状态转到 canonical registry |
| closure checker 被扩张成 Git/CMake 证明系统 | high / method | `rejected and removed` | 恢复 455 行薄检查器；Git identity 用只读命令，CMake/package 语义由真实 journey tests 负责 |
| LOC 路径零 diff 不能证明 build/package 语义 | high / boundary | `fixed by ownership` | source/static/shared 与 component graph 使用实际 CMake/package tests，不从 closure 文本解析器推断 |
| `BLOCKED` 的旧 end 指向仍为 AUDITED/TBD 的 candidate commit | high / evidence | `fixed` | container_intrusive end 改绑首次保存 BLOCKED receipt 的 `1a465e2`；CLOSED/BLOCKED 均须 terminal receipt |
| P0 profile 只是一个字符串 | high / method | `fixed` | 新增受检 `closure-correctness-v1` 定义，并要求每个 instrument ID 映射文档 |
| 模板漏 demand kind、negative space、三类 owner、逐项 correctness | medium / method | `fixed` | methodology/template 同步，peer 上限改为每项 capability |
| 旧 Core 轮没有 v1 closure card、预算和逐 finding red team | high / provenance | `accepted` | 本轮独立重放，不伪造历史合规 |
| 底层组件“所有直接消费者”会膨胀成全库测试 | medium / method | `fixed` | P0 profile 要求按 diff/公开暴露/失败半径选择并解释代表集 |

## 裁决

- Core 产品 verdict：`CONFIRMED`；保持 `CLOSED`，没有 `PRODUCT_TRIGGER`，Core 实现改动为 0；
- 历史过程 verdict：不能声称旧 `core-v2` 原生执行过 module-closure-v1；它被诚实导入为 legacy；
- 性能 verdict：P0 structural `PASS`；没有 Core nanobenchmark 或全局 SOTA 声明；
- 现有 negative space：runtime topology、crash reporting、consumer hint policy、Native FFI 仍不属于 Core；
- reopen triggers：真实 correctness/security/consumer foundation defect；supported compiler/target 使合同错误；
  直接依赖或 delivery boundary 改变语义；同合同 strong peer 证明更好复杂度档或 material consumer 优势。

这个结果证明的是：当前方法能让一个已关闭模块在固定边界内被复现、能发现自己的证据造假空间，
并且只修方法/证据而不扭曲产品。它不把一次 retrospective replay 改写成旧工程过程从一开始就完美。
