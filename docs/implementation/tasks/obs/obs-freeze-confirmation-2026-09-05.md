# Obs 冻结确认 — 2026-09-05

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 本文确认刚完成的 `obs-v4-structured-event` terminal round，没有无触发重开产品，也不建立第二套状态。
> 当前状态真相仍是 [`module-status.json`](../../module-status.json)；其中的 `freeze_confirmation` 将新标准
> `FROZEN` 标记绑定到明确仓库快照和本文证据。

## 绑定身份与边界

- confirmation：`obs-freeze-2026-09-05`；marker：`FROZEN`；
- reviewed repository commit / tree：
  `080753874828f5bb8d973b665c27df77d9eaa269` /
  `bd89b1dd9cc78d1b7017a3c79b92b4012277bddf`；
- frozen v4 product candidate / tree：
  `14c82a18f8f00b892479ae16abba4d8320d5915c` /
  `7594a2f13271ba7956c3e685e24c6302a91f015b`；
- candidate 到 reviewed commit 的 `include/hyle/obs`、`src/obs`、Obs component edge 与 package sample
  为 zero product diff；中间提交只保存 verification、terminal receipt、registry 与 Build v12 终态；
- terminal task：[`obs-v4-structured-event`](obs-v4-structured-event.md)；performance：
  [`benchmarks-obs`](../../evidence/performance/benchmarks-obs.md)。

Obs 是低层同步 observability producer **质因数**：公开依赖 Core/Error/Time，私有依赖 Mem-reclaim；
Trace ON 借用中央锁定 Tracy。它拥有 typed borrowed log record、同步 sink lifecycle、call-site control 和
native trace seam，不拥有 metrics、异步队列、持久化、配置树、任务上下文或 crash reporting。

## 七问冻结直答

### 1. 第三方产品路径

`PASS`。Trace OFF 无第三方产品实现；Trace ON 唯一借用 Tracy v0.14.1 /
`30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c`，由中央 lock/license/cache 与单 client target 管理。
Abseil 只作 ordinary benchmark peer，Folly 只作架构参考，llama.cpp/ggml 只作推理消费 provenance；它们
不进入 Obs 产品。标准库及 Windows/POSIX terminal/debugger API 属平台合同。

### 2. 功能超集与推理消费

`PASS`。当前能力覆盖 level/exact message、allocation-free format、同步多 sink、typed RAII registration、
quiescent remove/flush/default sink、六类 call-site/显式 rate control、category/event、string/signed/unsigned/
double/bool attributes、独立 correlation attributes、标准/外来 source、optional Unix timestamp、Error status
表达 recipe 与 Tracy trace seam。llama.cpp model-load、slot/decode identity/rate/depth/status 及 ggml level/text
callback 都可自然承载。owning/async logging、metrics 与自动 task context 已完整路由，未处置 `MUST` 为 0。

### 3. 内部复用与加速设施

`PASS`。Core compiler/environment/fatal、Error Status/Result、Time UnixTimestamp、Mem-reclaim 短 snapshot
guard 均已复用；Tracy 为 feature-ON borrow。callback quiescence 的 per-sink active ref 是 Obs 自有且不同
合同的机制；固定 arrays 使通用 allocator/container 不适用。Platform 尚无 console/debugger 同合同 seam，
因此 default sink 直接借 OS API。不存在重复的 clock/error/reclamation/trace decision authority。

### 4. 拟真、公平性能

`PASS`。W1-WIN `closure-perf-v1` 的三份 retained contract 均为 `QUALIFIED/EXACT/CURRENT`：ordinary
every-N 对 Abseil 四层 `WIN`，ratio `5.5077–5.6112`；完整 typed model-load progress 四层 `PARITY`，
ratio `0.9715–0.9821`；八 producer/一个 adapter registry throughput/chunk-p99 对 pre-v4 两层
`INCONCLUSIVE`，ratio `1.0460–1.0536`。全部 required strata 通过预声明 5% non-inferiority。只签上述
normal journeys，不外推全面或跨平台 SOTA。

### 5. 旧 C ABI / Native FFI

`CLEAN`。没有公开 C header/schema/bridge、managed sample、专用 option/target/package metadata 或只为
FFI 支付的 layout/indirection。Windows DLL 24 个 export 和 macOS dylib export 都是 C++ decorated symbols；
header template 所需 `detail` compiled seam 也不是 Native FFI。

### 6. Source / static / shared 构建与导入

`PASS`。Windows MSVC 19.50、WSL GCC 14.2、macOS AppleClang 21 均验证 product
`core,error,obs` / factor `core,error,mem_reclaim,time,obs`，并分别通过 source embedding、relocated
installed shared、relocated installed static consumer 与 Trace ON tests。三平台完整 component contract、
Windows all-product feature-OFF build 通过；Build v12 已把 append-only identity ledger 与 O(V+E) target
拓扑构造解耦，未选模块不会被连带构建。

### 7. C++ API 易用性与文档

`PASS`。[`user/obs`](../../../user/obs.md) 先按 error/log/rate-control/structured record/sink/trace 任务选型，
给出普通、推理、Status、adapter 与 Tracy recipe，并明确 borrow、temporary、callback concurrency、
remove/flush、default sink、脱敏、async copy 和网络陷阱；[`obs-design`](../../../architecture/obs-design.md)
记录长期 owner/依赖/成本。API 没有 benchmark mode、`fast/light` 分叉、旧 `LogField` alias 或 internal
registry seam 泄漏。

## 独立红队复核

`PASS`。终态红队检查了 snapshot publication→active-reference handoff、self/cross remove、slot reuse、
callback recursion、borrowed string/span、timestamp、default renderer、Trace OFF/ON、三平台 sanitizer/交付、
component topology、动态符号与三份性能 comparison class。guard 在 callback 前退出，第二个 grace period
关闭旧 snapshot handoff；因此没有自等待或用慢 callback 阻塞 Mem-reclaim。candidate 后产品为 zero diff，
剩余 medium-or-higher finding、material 性能 FAIL、未处置 `MUST`、重复 authority 与交付 blocker 均为 0。

## 冻结裁决与 LOC

没有新的能力缺口、错误复杂度、benchmark-shaped API、内部复用遗漏、C++ 易用性/文档错误、伪 C ABI
或 build/package 缺陷。Obs 保持 `CLOSED` 并取得本次新标准 `FROZEN` 标记；只有 registry 已列 reopen
trigger 的新事实才能解除。

`python scripts/loc-report.py --module obs --commit
14c82a18f8f00b892479ae16abba4d8320d5915c --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 441,812 |
| 自实现不含测试 | 70,870 |
| Obs 实现 | 1,454 |
