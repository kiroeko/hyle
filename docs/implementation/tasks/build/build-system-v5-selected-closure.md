# Build/package v5：selected-closure 独立施工修复

<!-- PROJECT UNIT REVIEW: CLOSED -->

> Build/package 是开发单元，不是可链接组件，因此不进入 `module-status.json`。本轮由
> [`build/package capability replay`](build-package-capability-replay-v1.md) 的真实
> `PRODUCT_TRIGGER` 启动；长期合同仍归
> [`build-system.md`](../../../architecture/build-system.md)。

## Closure card（`CLOSED`）

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

## Verification manifest

| Gate | Verdict | 当前 candidate 证据 |
|---|---|---|
| selected/unselected regression | `PASS` | 三平台 component contract 都确认：未选 graph 的非法 include 不阻断 core；selected core 的同类漏边仍命中 direct-edge fail-closed 诊断 |
| complete manifest / selected closure | `PASS` | W1-WIN、W1-WSL、M1-MAC 的 current `HyleCiComponentContract.cmake` 全部通过；unknown/empty/normal-cache、拓扑、owner、target edge、offline/lock/license 等既有门未回退 |
| source/static/shared delivery | `PASS` | W1-WIN fresh Core selected smoke：source、relocated shared、relocated static consumer 各 `1/1`；Core 保持 header-only、零 binary artifact |
| all-component audit | `PASS` | W1-WIN fresh feature-OFF configure 解析全部 16 个组件；all root 仍读取/审计完整 public/source tree，optional backend 零探测 |
| cross-platform CMake semantics | `PASS` | W1-WSL CMake 3.28/GCC 14 + Ninja；M1-MAC CMake 4.3/AppleClang 21 + Unix Makefiles。Mac 显式 `MACOSX_DEPLOYMENT_TARGET=14.4`，符合既有 embedding-parent 合同 |
| P0 cost structure | `PASS` | dependency marks 单次 topological reverse propagation；source audit 只读 `S_selected`；无 runtime 路径、allocation、lock、syscall、metadata 或计时驱动修改 |
| product identity | `PASS` | candidate 后 canonical 43 文件 build/package boundary 零 diff；component edge、option、package、lock、offline 与 workflow 均未改变 |
| docs / closure | `PASS` | architecture/user route 已更新；279 个 Markdown、825 个相对链接和 16-component closure schema 全部通过 |
| immutable LOC | `PASS` | candidate `83bfa5566325e679da11528f93e50a46d2512abd` 由 canonical boundary self-check 复算 |

Mac 首次直接运行 contract 未设置父工程 deployment floor，按设计在 platform 配置期拒绝；显式提供
文档要求的 14.4 后同一 candidate 全通过。这不是重试调参或掩盖失败，而是验证 source-embedding parent
必须声明已有 public OS floor。真实 CUDA/Vulkan/TLS/HTTP2、AArch64/clang-cl、完整 source package 和
hosted Actions 本轮未重跑：对应实现边界零 diff，也不受 selected source audit 顺序影响；它们保持 v4
证据/手工 workflow 的 accepted limit，不能由本轮 CPU 合同外推。

## Closure decision

状态：`CLOSED`。`BUILD-CAP-PARALLEL-01` 已修复：manifest 本身仍全局 fail closed，product source
只按 selected closure 审计，all-component CI 仍覆盖全树。这样独占 scratch 现在同时提供 cache 隔离和
语义隔离；一个未选模块的施工中 include 不再阻断另一个模块，selected 模块漏边也没有被放过。

能力回放中的其余结论维持：三种 C++ 交付、relocatable package、dependency lock/offline/license、
source bundle、owner-scoped option、CI 分层、exact pre-1.0 version 与 Native FFI `N/A` 均没有新缺口。
非 CMake/package archive 任务已有用户可发现 route，不因 oneTBB/Abseil feature 数量制造半接口。

重开条件：canonical build/package boundary、组件/公开头/target edge、option/lock/workflow 或交付形状
改变；selected root 再次受未选源码/cache/依赖阻断；selected 漏边不再 fail closed；具名非 CMake/
Native FFI/新平台消费者出现；或正常构建出现可复现的数量级/全图 invalidation 回归。一次 cold 点值、
CI 额度、pkg-config/CPack 的存在或未收口产品模块本身不能单独重开。

terminal receipt：`build-v5-selected-closure` 对 candidate
`83bfa5566325e679da11528f93e50a46d2512abd` / tree
`4d5c2455d23400ad9db66047410c65d62d844598` 的全部适用门为 `PASS`，无 blocker；本页所在提交保存
terminal end receipt。

Candidate immutable LOC：全库代码 **299,028**；自实现包含测试 **122,321**；自实现不包含测试
**82,176**；build/package 实现 **7,782** 行。LOC 只说明规模，不形成完成度或性能证书。
