# Build/package system v4：util 依赖边现树复签

<!-- PROJECT UNIT REVIEW: CLOSED -->

> Build/package 是开发单元而不是可链接组件，不进入 `module-status.json`。本页只处理 v3 关闭后由
> util-runtime 收口触发的构建边变化；长期合同仍见
> [`build-system.md`](../../../architecture/build-system.md)，v3 task 保留历史事实。

## Closure card（`CLOSED`）

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

- [x] W1-WIN current component contract：schema/无环、中央 direct edge 与真实 target edge、header
  owner、unknown/非法 option、offline positive/negative、dependency lock 与 license 投影全部通过。
- [x] W1-WIN selected `util`：fresh source embedding、relocated shared SDK、relocated static SDK consumer
  全部构建并运行通过。解析闭包仍因 `mem -> util_core` 间接包含 `util_core`，但 direct-edge contract
  准确确认 `util` 不再伪造对它的直接依赖。
- [x] current all-component CPU shipping graph：W1-WIN MSVC 19.50 为 305 steps、62/62；W1-WSL
  GCC 14.2 为 322 steps、62/62；M1-MAC AppleClang 21 / Unix Makefiles 为 60/60。三端均为
  WERROR、昂贵 feature 显式 OFF、`--parallel 2`；这里只签构建/组装，不替未关闭产品模块签能力。
- [x] clean current source package：commit
  `16dd7d80e9702b28758669e595c5907e30501218` 生成 zip 与 SHA-256，manifest/legal payload 通过，
  解包后脱离 Git 以 all-component、WERROR、feature-OFF 完成 125-step product build。
- [x] no-op / invalidation：Windows Ninja 0.146/0.088/0.086s、WSL Ninja 三次 0.02s，均明确
  `no work to do`；macOS Makefiles 1.03/0.77/0.78s，遍历 target 但零编译。只 touch source-package
  解包副本的 `src/util/demangle.cc` 后，仅重编该 TU 并重链 `hyle_util`。
- [x] 当前 workflow/YAML 静态合同由 component/tooling tests 覆盖；为保留 GitHub Actions 额度，
  没有启动 hosted full dispatch。当前 CMake/workflow/lock/option 实现相对 v3 零漂移，因此这项仍是
  明示的 `ACCEPTED LIMIT`，不是借历史绿灯伪称现树运行。
- [x] LOC 在 immutable product candidate `469356394a185b5201675fd76ff1a10a4050528b`
  上由 canonical `build-package` boundary 自检通过：全库 297,318；自实现含测试 122,237；自实现
  不含测试 82,124；build/package 实现 7,726。

Windows leaf invalidation 首次从未导入 MSVC 环境的普通 PowerShell 直接调用 `cmake --build`，按既有
合同因找不到标准头失败；用受支持的 `scripts/hyle-windows.ps1` 入口重跑后得到上述精确失效结果。
这既没有暴露产品缺陷，也再次确认维护者不能绕过已文档化的 Windows toolchain bootstrap。

## Closure decision

状态：`CLOSED`。触发本轮的唯一产品变化是删除已经不存在的 `util -> util_core` direct edge；当前
manifest、target、selected source/static/shared consumer、三平台 CPU 图、offline/lock/license、完整
source package 与增量失效合同全部一致，冻结 `FIX` 集为空。build/package 仍保持中央 manifest 一次
声明、模块 target 真实投影、独占 scratch 并行施工和三种 C++ 交付形状。

保留边界：真实 CUDA/Vulkan/TLS/HTTP2、AArch64/clang-cl 与 hosted full dispatch 本轮未运行，不能
由 CPU 三平台证据外推。以下任一事件重开本单元：43 文件 build/package 实现边界变化；组件/公开头/
target edge 变化；依赖 lock、license/source-package 投影、option 或 workflow 变化；新增交付形式、
工具链、平台或具名 Native FFI 消费者；出现 selected root、relocation、offline、no-op/invalidation 或
并行独占 cache 的可复现失败。

terminal receipt：`build-v4-01` 对 product candidate
`469356394a185b5201675fd76ff1a10a4050528b` / tree
`eabcba1fe1cffb484b20c377af0423028e2f9062` 的全部适用门为 `PASS`，无 blocker；本页所在提交是
terminal end receipt。
