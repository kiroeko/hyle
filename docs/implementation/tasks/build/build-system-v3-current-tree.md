# Build/package system v3 current-tree 收口

<!-- PROJECT UNIT REVIEW: SCOPED -->

> Build/package 是贯穿所有组件的开发单元，不是可链接产品组件，因此不伪造
> `module-status.json` 条目。本页按模块收口方法保存当前轮状态与证据；长期合同仍归
> [`build-system.md`](../../../architecture/build-system.md)。v2 的 `VERIFIED` 记录只作历史输入。

## Closure card（`SCOPED`）

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
| BUILD-V3-GRAPH-01 | 单一组件 manifest 同时产生选择闭包和真实 target 边 | audit pending | pending |
| BUILD-V3-DELIVERY-01 | source/static/shared、selected roots、relocation、并行独占 cache | audit pending | pending |
| BUILD-V3-SUPPLY-01 | lock、offline、license、source package 同源且 fail closed | audit pending | pending |
| BUILD-V3-CI-01 | 当前 CI 分层覆盖现有组件且不无故消耗额度 | audit pending | pending |
| BUILD-V3-PERF-01 | configure/no-op/invalidation 无数量级或全图泄漏 | audit pending | pending |
| BUILD-V3-DOC-01 | architecture/developer/user/历史证据各守职责 | audit pending | pending |

本轮尚未冻结 `FIX`；任何实现修改必须先完成审计并把相应行裁决为 `FIX`。旧 v2 数字、GitHub 绿灯或
历史完整构建都不能单独充当 current-tree `PASS`。
