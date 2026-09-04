# Core 冻结确认 — 2026-09-04

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对既有 `CLOSED` 产品的只读确认，不是一次无触发重开，不建立新的产品 candidate，也不替换
> [`core-v5-assert-macro-contract`](core-v5-assert-macro-contract.md) terminal record。当前状态真相仍是
> [`module-status.json`](../../module-status.json)；其中的 `freeze_confirmation` 只把本次确认绑定到
> 明确的仓库快照和本文证据。

## 绑定身份与范围

- confirmation：`core-freeze-2026-09-04`；marker：`FROZEN`；
- reviewed repository commit / tree：
  `8f42e57c00df77b07528c55541d406b6fc8d78c8` /
  `d673aeae014fb2e86cb8abe403d0d68378bc37cc`；
- frozen Core product candidate / tree：
  `4b7f28c55e9ca118cf87b80058ce291261033ec5` /
  `074617e1b1dc9d11843493c1e2240da7fd798d9a`；
- current Core-owned implementation/test paths relative to the product candidate：zero diff；
- reviewed current-tree change outside those paths：the central factor/product graph and its build contracts were
  refactored after the Core candidate, so current source/static/shared selection and the whole component contract
  were replayed rather than inferred from the old receipt；
- product modifications in this confirmation：none。

## Capability、形状与红队确认

本次按当前 Core 边界重新检查全部公开 family：bounded fatal/debug invariant、fatal observer、current-TU
environment facts、compiler contracts、compile-time cache-layout recommendation，以及 temporal、one-shot 和
write prefetch intent。每个公开入口仍可反向追溯到正常 HYLE/推理引擎开发任务；公开名称、类型、对象
layout 和常态 metadata 中未发现 benchmark/test/fast-vs-slow seam。

本地 clean reference identities 为 llama.cpp
`c7bda030e7faee594dbe7550185e857351ad405d`（bundled ggml tree
`25e2241ed891c98cd26af5c47246c79b430a49cd`）、standalone ggml
`d4716378882593333721eb33f153144b6885caf2`、Abseil
`536d30bfea05a09c7561cbba749ca7f738d46821`、Folly
`80071a1f79e110b34a2cf7a3a4d01a59e823d697`、Highway
`63f4b16cda024e2adff8a18dee81e067f8a87d61` 和 oneTBB
`843162db8cf4b2531a4a6147b6ea3fd860d44668`。针对 Core family 的当前差异扫描没有产生新准入 `MUST`：
llama.cpp/ggml 新增的 invariant 调用仍由现有 always-on family 覆盖；printf-varargs abort、完整 crash
policy、runtime ISA/topology、allocation、logging 和 recoverable failure 仍分别路由到 Obs/host、Platform、
Mem 与 Error，不应下沉 Core。peer 的 feature 数量、标准 C++ 已直接覆盖的 spelling 和 vendor 私有宏不
构成 Core API 需求。

接口仍按任务拆成 `assert.h`、`environment.h`、`compiler.h` 与 `cache.h`；只有必须跨 compiler/preprocessor
表达的能力使用宏，其余使用 typed C++ value/function。Release/Debug expression parity、1–8 typed
fields、同步 borrowed observer、plugin unload/quiescence、optimizer-promise UB 与 prefetch lifetime 陷阱
都已由 [`user/core`](../../../user/core.md) 就近说明。没有旧 C ABI、Native FFI、umbrella header、Core
archive/DSO 或只为测试存在的公开入口。

## 当前树验证

所有验证都在 reviewed commit 的 clean/disposable copy 上完成；Windows 权威工作树中的无关既有修改
没有进入验证快照。

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| Release Core/tool contract | `18/18 PASS`，Core label `6/6` | `18/18 PASS`，Core label `6/6` | `18/18 PASS`，Core label `6/6` | `PASS` |
| Debug Core/tool contract | `17 PASS + release-only codegen SKIP` | candidate product evidence retained；Core paths unchanged | candidate product evidence retained；Core paths unchanged | `PASS` |
| optimized P0 codegen | disabled Debug / fatal hot-cold / prefetch structural probes pass | 同合同 probe pass | 同合同 probe pass | `PASS` |
| source + relocated shared/static consumers | every journey `2/2 PASS` | every journey `2/2 PASS` | every journey `2/2 PASS` | `PASS` |
| current central component contract | `PASS` | `PASS` | `PASS` | `PASS` |
| docs / closure registry | `check-docs PASS`；`check-module-closure PASS` | repository contract | repository contract | `PASS` |

Current configure resolves `HYLE_COMPONENTS=core` to exactly product `core` and factor `core`。The central manifest
still declares `_hyle_declare_component(core)` with no dependency and `_hyle_declare_product(core FACTORS core)`；
tests-off source/package journeys enable no Core runtime artifact。The current factorization therefore did not leak
Container ISA、exception policy、Trace、Testing、Threads or an unrelated factor into the shipped Core contract。

## 性能与冻结裁决

Core remains `P0 STRUCTURAL PASS`；timing peer comparison is `NOT_APPLICABLE`。Current optimized probes preserve
zero runtime work for disabled Debug、one cold failure edge for successful fatal assertions、compile-time environment/
layout values and at most one target prefetch hint or no-op。There is no allocation、lock、syscall、runtime dispatch
or normal object metadata。Failure I/O、cache-hit rate and downstream throughput are deliberately not promoted to a
module-wide SOTA claim；the owning consumer must qualify those effects。

No new admitted capability gap、performance defect、benchmark-shaped API、usability defect、documentation defect or
build/package integration defect was found。Core therefore remains `CLOSED` and is additionally marked
`FROZEN` at the reviewed repository snapshot。Only an existing `reopen_triggers` fact may remove this hold；a new
benchmark、peer symbol count、LOC difference or request to “look again” alone does not。

## Snapshot LOC

`python scripts/loc-report.py --module core --commit
8f42e57c00df77b07528c55541d406b6fc8d78c8 --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 439,073 |
| 自实现包含测试 | 111,779 |
| 自实现不包含测试 | 70,651 |
| Core 实现 | 762 |
