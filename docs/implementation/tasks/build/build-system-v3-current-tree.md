# Build/package system v3 current-tree 收口

<!-- PROJECT UNIT REVIEW: AUDITED -->

> Build/package 是贯穿所有组件的开发单元，不是可链接产品组件，因此不伪造
> `module-status.json` 条目。本页按模块收口方法保存当前轮状态与证据；长期合同仍归
> [`build-system.md`](../../../architecture/build-system.md)。v2 的 `VERIFIED` 记录只作历史输入。

## Closure card（`AUDITED`）

- unit / owner / round：`build-package` / build/package maintainers / `build-v3-01`；kind：`REFACTOR`
  （current-tree evidence rebind，不预设产品实现修改）。
- baseline commit：`6c7d5aeaa1616b3040c94aaa56f105e0e467824c`。
- 进入原因与 defect ID：`BUILD-V3-FRESH-01`。v2 只到 `VERIFIED`；依赖升级后性能页明确撤销旧图的
  current-certificate 资格，而模块源码、测试、benchmark 与 package consumer 又继续演进。
- 真实 journeys：授权源码消费者 `add_subdirectory()`；selected-root installed static/shared SDK 与
  relocation；两个互不共享 cache 的模块施工目录；锁定依赖的 online/offline/source-package 投影；
  新模块自动进入 CPU baseline 和 package/component contract。
- 范围：当前 HEAD 的中央组件图及直接 target 边、header ownership、C++ source/static/shared、
  selected component、并行隔离、option/feature fail-closed、dependency lock/offline/license/source
  package、CI 分层、用户/维护文档和正常 configure/reconfigure/no-op/invalidation 结构。
- 非目标：重写 CMake、每模块独立 project、Native FFI/C API、替 compute/graph 补产品能力、真实
  CUDA/Vulkan/TLS/HTTP2 设备或协议认证、sanitizer/ISA 性能证书、发布级 Formal 证书。
- 直接消费者：全部产品组件、tests/bench、package samples、CI 和获得源码授权的父工程；本轮代表根为
  `core`、`error`、`container_intrusive`，all-component CPU 只证明当前构建形状而不签未收口模块能力。
- tier / performance：`P0`。无 runtime 热路径；复杂度目标是组件闭包 O(V+E)、一次 manifest 投影、
  selected-only configure/build、no-op 零重编、模块改动不无故使全图失效。计时只作数量级/失效诊断，
  不比较纳秒或宣称构建系统 SOTA。
- 总预算：审计与验证最多两小时；Windows、WSL、macOS 各一轮 bounded CPU/build contract，统一
  `--parallel 2`。最多一次机制诊断、一个当前 correctness/delivery 修复与一次受影响重跑；不增加
  第五种交付姿势。
- machine / instrument：Windows canonical checkout；`W1-WIN`、`W1-WSL`、`M1-MAC`；P0
  `closure-correctness-v1` 加 CMake component/selected-delivery/source-package contracts。WSL/macOS
  只接收 Windows 单向同步的当前提交，不接受远端本地产品修改。
- strong contracts：CMake 3.25+ target/package/FetchContent 官方模型；HYLE 已冻结的中央 component
  manifest 与 source/static/shared 用户 journey。无同合同 peer，不产生 peer/SOTA 声明。
- correctness/delivery gates：manifest schema/无环/直接边/header owner；normal/empty/unknown/错误
  option；独占 cache 并行；source/static/shared + relocation；offline positive/negative；source package
  脱离 Git；license/export closure；Windows/WSL/macOS CPU baseline；文档与 closure/tooling tests。
- 预期输出：有限 current-tree gap ledger、current candidate identity、逐风险 verdict、已知限制、重开
  条件、terminal receipt 和 build 专用 LOC。
- 停止条件：所有适用 P0 门通过则 `CLOSED`；同一 current-tree 在任一适用平台或交付形状存在具名
  correctness/delivery blocker 则 `BLOCKED`。远端暂时离线是验证 blocker，不授权继续改 CMake。

## Current-tree audit ledger

| ID | requirement | current evidence | disposition |
|---|---|---|---|
| BUILD-V3-GRAPH-01 | 单一组件 manifest 同时产生选择闭包和真实 target 边 | Windows current-tree component contract 验证 schema、拓扑、header owner、最终 target 边与 selected closure；`core` fresh configure 只解析 `core` | `KEEP / PASS` |
| BUILD-V3-DELIVERY-01 | source/static/shared、selected roots、relocation、并行独占 cache | `core`、`error`、`container_intrusive` 的 source/shared/static/relocated smoke 全通过；Windows/WSL 为 62/62、macOS 为 60/60 的 all-component CPU 回归通过 | `KEEP / PASS` |
| BUILD-V3-SUPPLY-01 | lock、offline、license、source package 同源且 fail closed | component contract 的 offline positive/negative、lock/license 门通过；commit `0fbb5ac8` 的完整 source zip 脱离 Git、WERROR、all-component feature-OFF 构建通过 | `KEEP / PASS` |
| BUILD-V3-CI-01 | 当前 CI 分层覆盖现有组件且不无故消耗额度 | workflow YAML 可解析且合同检查通过；push/PR 只跑 Linux 高信号门，三平台/package/feature 只在 full dispatch；最近一次托管成功为 `a8db371c`，本轮因额度不用旧绿灯冒充现树运行 | `KEEP / ACCEPTED LIMIT` |
| BUILD-V3-PERF-01 | configure/no-op/invalidation 无数量级或全图泄漏 | all-component no-op：Windows Ninja 0.163s、WSL Ninja 0.09s 且均为 `no work to do`；macOS Makefiles 1.08s、零编译；scratch 中只 touch `util/demangle.cc` 后仅重编该 TU 并重链 `hyle_util` | `KEEP / PASS` |
| BUILD-V3-DOC-01 | architecture/developer/user/历史证据各守职责 | architecture、distribution、三平台/offline setup 与历史性能页完成职责审计；265 个 Markdown、746 个相对链接合同通过；性能页只需绑定本轮 freshness receipt | `DOC FIX` |

## Audit freeze

- build/package 实现候选：`42a6747d345c5f908648bc2e904ac942bc81a4c9`；完整 tree
  `0ba864aee7d52d22ea1b46063d28a56c9fbaae0e`。
- 从该候选到本轮 scoped HEAD，专用 43 文件边界（workflow、根 CMake/presets、`cmake/`、offline
  init 与三个 build/tooling 入口）`git diff` 为空；依赖升级和后续模块施工没有偷偷改动构建产品。
- 冻结的产品 `FIX` 集为空。唯一 `DOC FIX` 是把历史性能页的 freshness gate 绑定到本轮 current-tree
  复签；不得为了得到新点值而改 CMake，也不启动托管 CI 消耗余额。
- `ACCEPTED LIMIT`：本轮没有运行真实 CUDA/Vulkan/TLS/HTTP2、AArch64 cross、clang-cl 或托管 full
  dispatch。相应机制未发生实现漂移，仍由手工 workflow 保留；任一实现/lock/runner 变化即重开。

旧 v2 数字、GitHub 绿灯或历史完整构建均未单独充当 current-tree `PASS`；本页的 verdict 来自上述
现树合同、当前三平台 CPU 组装和当前源码包旅程。
