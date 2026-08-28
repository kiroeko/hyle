# Build/package system v4：util 依赖边现树复签

<!-- PROJECT UNIT REVIEW: ACTIVE / AUDITED -->

> Build/package 是开发单元而不是可链接组件，不进入 `module-status.json`。本页只处理 v3 关闭后由
> util-runtime 收口触发的构建边变化；长期合同仍见
> [`build-system.md`](../../../architecture/build-system.md)，v3 task 保留历史事实。

## Closure card（`AUDITED`）

- unit / owner / round：`build-package` / build/package maintainers / `build-v4-01`；kind：`REOPEN`。
- 前一 round / trigger：`build-v3-01` / `BUILD-V4-UTIL-EDGE-01`。util-runtime candidate
  `469356394a185b5201675fd76ff1a10a4050528b` 删除了不再存在的 `util -> util_core` 直接产品边；组件/
  target 边变化命中 v3 明示 reopen trigger。
- baseline：build v3 implementation candidate
  `42a6747d345c5f908648bc2e904ac942bc81a4c9`；本轮 current product candidate 为
  `469356394a185b5201675fd76ff1a10a4050528b` / tree
  `eabcba1fe1cffb484b20c377af0423028e2f9062`。
- 真实 journeys：选择 `util` 的源码父工程、relocated static/shared SDK；all-component CPU 开发树；
  完整 source package；在线/离线 component graph 与许可证投影。
- 范围：中央 manifest 的这一条 direct-edge 删除及其 selected closure、真实 target link、header owner、
  source/static/shared、all-component CPU、offline/lock/license/source package、no-op/invalidation 与当前文档。
- 非目标：重写 CMake、改变任何 option/依赖 lock/workflow、运行 CUDA/Vulkan/TLS/HTTP2、替产品模块签
  功能/性能、启动托管 GitHub full dispatch、Native FFI/C API 或发布级 Formal 证书。
- tier / performance：`P0` / `closure-correctness-v1`。目标仍是 O(V+E) closure、selected-only graph、
  no-op 零重编以及 leaf 变化不扩散；只判断结构与数量级，不声明构建系统 SOTA。
- machine：W1-WIN canonical、W1-WSL、M1-MAC；三端独占 scratch，`--parallel 2`。远端只接收 Windows
  已推送提交，不接收远端产品改动。
- 预算：最多两小时、一次机制诊断、一个必要 correctness/delivery 修复及一次重跑。若冻结候选需要
  产品修改，先退回 audit；不得为刷新点值扩建仪器。
- strong contracts：CMake 3.25+ target/package/FetchContent 模型；HYLE 中央 component manifest、
  selected-component smoke 与 source/static/shared 用户 journey。无同合同 peer，不产生 SOTA 声明。
- 适用门：manifest schema/无环/direct target/header owner；unknown/非法 option；offline positive/
  negative、lock/license；util source/static/shared relocation；三平台 all-component CPU build/test/no-op；
  clean source package 脱离 Git；docs/tooling。真实 optional backend、AArch64/clang-cl 和托管 CI 为未改变
  机制的 accepted limit，不伪称运行。
- 停止条件：以上适用门通过则 `CLOSED`；任一当前 correctness/delivery blocker 则 `BLOCKED`。

## 有限 audit ledger

| ID | requirement / evidence | disposition | 本轮验证 |
|---|---|---|---|
| BUILD-V4-EDGE-01 | `registry.h` 已删除 util_core/ref/type-name/preprocessor 使用；中央 direct edge 同步删除 | `KEEP / REPLAY` | include/target graph contract；不得因 `mem -> util_core` 的传递闭包伪造 direct edge |
| BUILD-V4-DELIVERY-01 | `util` 公开模板仍直接使用 core/error/mem/container，compiled demangle 仍由同 target 交付 | `KEEP / REPLAY` | source embedding、relocated shared/static util consumer |
| BUILD-V4-GRAPH-01 | 删除一条边不能造成缺 owner、环、selected closure 或最终 target 漂移 | `KEEP / REPLAY` | current component contract + all-component CPU 三平台 |
| BUILD-V4-SUPPLY-01 | lock/offline/license/source-package 实现没有漂移，但完整产品图 identity 已改变 | `KEEP / REPLAY` | current contract positive/negative + clean current source package |
| BUILD-V4-PERF-01 | direct edge 删除不应制造常态重编、全图失效或非 selected target | `KEEP / REPLAY` | 三平台 no-op；source-package scratch 的 util leaf invalidation |
| BUILD-V4-CI-01 | workflow 实现零漂移；额度有限且此次没有 workflow/action 变化 | `ACCEPTED LIMIT` | YAML/static contract；不启动 hosted full dispatch |

审计 `42a6747d..46935639` 的 build/package 专用 43 文件边界只有
`cmake/HyleComponents.cmake` 一处一行修改；option、dependency lock、offline/package、tool wrapper 与
workflow 均零漂移。该删除由 util-runtime 真实 public include 边变化驱动，不是 benchmark 或构建点值
驱动。冻结产品 `FIX` 集为空；验证若暴露 blocker 才允许退回本节重新裁决。

## Verification manifest

候选冻结后填写；未运行项不得写 `PASS`。

## Closure decision

当前为 `AUDITED`，尚无 terminal verdict。
