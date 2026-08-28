# Build/package v5：selected-closure 独立施工修复

<!-- PROJECT UNIT REVIEW: ACTIVE -->

> Build/package 是开发单元，不是可链接组件，因此不进入 `module-status.json`。本轮由
> [`build/package capability replay`](build-package-capability-replay-v1.md) 的真实
> `PRODUCT_TRIGGER` 启动；长期合同仍归
> [`build-system.md`](../../../architecture/build-system.md)。

## Closure card（`CANDIDATE`）

- unit / owner / round / kind：`build-package` / build/package maintainers /
  `build-v5-selected-closure` / `REOPEN`；
- baseline：`c3b32d1e2d251c6ed971ee4ada7aba129af2bc41`；前一 round：`build-v4-01`；
- trigger / evidence：`BUILD-CAP-PARALLEL-01`。只选择 `core` 时，未选择的 `graph` 增加一条非法
  cross-component include 仍会令 configure 失败；精确复现、机制与 peer 差异见回放记录；
- real consumers：两个互不共享 cache 的模块维护者；授权源码父工程显式选择一个产品根；all-component
  CI 继续作为完整图审计；
- scope：组件 root 解析顺序、manifest 闭包复杂度、public/source include-owner audit 的 selected
  作用域、正负 regression、对应 architecture/user/developer 说明和受影响 current-tree delivery；
- non-goals：改变任何组件/target edge、公开 C++ API、package target、option、dependency lock、offline/
  license/source package、workflow、Native FFI；重写 CMake；跑 optional hardware 或 build-time SOTA；
- direct consumers/dependencies：全部 source-embed 根；`HyleComponents.cmake` 向
  `hyle_add_module()`/package/tests 提供 `HYLE_BUILD_*` 和 `HYLE_RESOLVED_COMPONENTS`。不改变产品依赖图；
- tier / performance：`P0 / closure-correctness-v1`。目标是 manifest closure 只访问 O(V+E) 元数据，
  include audit 只读 `S_selected`；不比较一次 cold configure 毫秒值；
- machines/instrument：W1-WIN canonical checkout，CMake 4.4 + MSVC 19.50/Ninja；portable CMake 逻辑在
  W1-WSL/GCC 14 复证 component contract。M1-MAC 不改变平台专有路径，本轮若不可用明确保留为未运行，
  不冒充 PASS；
- peers/contracts：CMake 3.25+ target/subdirectory model；oneTBB
  `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 与 Abseil
  `81c993af4fb88b3a51829a796df02251fa27102f` 的 source-embedding isolation；
- budget：实现与本地验证最多 90 分钟；一个算法修正、一次受影响重跑；不得顺手扩建公共 closure 或
  performance 仪器；
- gates：selected/unselected 正负 fixture；current component graph contract；fresh `core` source/static/
  shared relocated smoke；all-component feature-OFF configure；docs/closure/LOC；
- stop：以上适用门通过则 `CLOSED`；未选源码仍能阻断，或 selected 漏边不再 fail closed，则
  `BLOCKED`。产品代码变更超出下表时退回 `AUDITED`。

## Capability/gap ledger 与冻结 FIX

| ID | demand / defect | semantic/artifact/delivery owner | target and cost | coverage | action / verification |
|---|---|---|---|---|---|
| BUILD-V5-SELECT-01 | source root 必须在 include audit 前解析；未选模块不能阻断 | build/package / `HyleComponents.cmake` / source delivery | manifest schema 全量验证；include scan 只读 selected closure | `PARTIAL` | `FIX`：先解析 roots/closure，再扫描 `HYLE_RESOLVED_COMPONENT_LIST` |
| BUILD-V5-COMPLEXITY-01 | 当前 fixed-point 与全库 source scan 不符合冻结的结构目标 | build/package | closure O(V+E)，source scan O(S_selected) | `PARTIAL` | `FIX`：利用已验证 topological manifest 单次反向传播 dependency marks；不做 timing 调参 |
| BUILD-V5-FAILCLOSED-01 | 限制作用域不能放松 selected direct-edge 审计 | build/package / contract fixture | selected 非法 include 在 target 建立前失败 | `COMPLETE but at risk` | `KEEP + VERIFY`：最小 selected probe 必须失败且命中正确诊断 |
| BUILD-V5-PARALLEL-01 | 未选模块中间态不影响另一个 root | build/package / contract fixture | unselected invalid source 存在时 core configure 成功 | `MISSING` | `FIX`：最小 unselected probe 必须成功；all-component 仍会检查全库 |
| BUILD-V5-DELIVERY-01 | source/static/shared target/package identity 不随审计顺序变化 | build/package | target/headers/relocation 不变 | `COMPLETE` | `KEEP + VERIFY`：fresh selected Core 三种 journey；current component contract |
| BUILD-V5-DOC-01 | 用户/维护者必须知道 selected scan、非 CMake 与 binary archive route | docs / build-package | 无产品成本 | `PARTIAL` | `FIX`：architecture 复杂度与边界；distribution 说明 bridge/install-prefix 归档路线 |

冻结产品 `FIX` 只有 `BUILD-V5-SELECT-01` 与 `BUILD-V5-COMPLEXITY-01` 的同一局部重排；其余为
regression/文档。不得修改 component edge、option、package、dependency 或 workflow。

## Candidate freeze

- product candidate / tree：`83bfa5566325e679da11528f93e50a46d2512abd` /
  `4d5c2455d23400ad9db66047410c65d62d844598`；
- product delta：`HyleComponents.cmake` 先验证 manifest/解析 roots，以已验证的 topological order 单次
  反向传播 dependency marks，再只对 `_hyle_resolved` 做 public/source include audit；组件名、edge、
  `HYLE_BUILD_*`、target、option 和 package surface 不变；
- regression delta：component contract 增加两个最小 child-project fixture；未选 graph 的非法 include
  配置 core 必须成功，selected core 的同类漏边必须命中原 fail-closed 诊断；
- documentation delta：architecture 说明全局 manifest 与 selected source audit 的边界；用户指南明确
  非 CMake bridge 和 binary install-prefix 归档路线；
- P0 mechanism：dependency propagation 从 growing-set fixed point 改成一次 topological reverse pass；
  source text 从 `S_all` 收束为 `S_selected`。没有计时调参、常态 runtime metadata 或产品 API 变化；
- pre-freeze evidence：W1-WIN current component contract `PASS`；Core source/shared/static/relocated consumer
  各 `1/1 PASS`；W1-WSL CMake 3.28/GCC 14 同一 current component contract `PASS`；docs/closure/diff
  checks `PASS`。

Candidate 后若改变以上算法、作用域或正常 source journey，退回 `AUDITED`。后续只允许补验证、receipt
和 LOC。

## Verification / terminal receipt

待当前 candidate 的 macOS portable configure、最终 Windows current-tree replay、文档/闭包与 immutable
LOC 完成后填写。
