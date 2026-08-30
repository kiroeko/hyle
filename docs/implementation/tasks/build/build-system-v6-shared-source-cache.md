# Build/package v6：共享锁定源码缓存

<!-- PROJECT UNIT REVIEW: CLOSED -->

> Build/package 是开发单元，不进入 `module-status.json`。长期合同仍归
> [`build-system.md`](../../../architecture/build-system.md)；本页只授权本轮有限施工。

## Closure card（`CLOSED`）

- unit / owner / round / kind：`build-package` / build/package maintainers /
  `build-v6-shared-source-cache` / `REOPEN`；
- baseline：`7d456d2542d367a596fc740d706e7096cf11cc08`；前一轮：`build-v5-selected-closure`；
- trigger：用户明确要求平台/CPU 无关的 FetchContent 源码在构建树之间共用；新机器第一次在线
  configure 应自动取得缺失源码，之后重复 configure/build 不重新下载；依赖还要可统一查看、准备、
  更新检查和清理，并移除仓库根的旧缓存和生成目录。当前默认
  `<build>/_deps` 同时保存 source/build/subbuild，删除 build 会丢失源码；仓库根 `.deps/` 又是已过期、
  非权威的旧缓存，需求可复现；
- real consumers：同一 checkout 的 MSVC/clang-cl、Debug/Release、sanitizer 与独立模块 build tree；
  source embedding 父工程和断网准备者；
- scope：FetchContent 锁定 archive 的共享 source/archive cache、online configure cache-miss 自动填充、
  per-build binary/subbuild 分离、显式 override/offline 优先级、并发/中断安全、
  `status/sync/check-updates/prune` 管理入口、architecture/user/developer 文档与确认生成目录清理；
- non-goals：共享任何 object/library/generated CMake state；改变组件边、公开 C++ API、产品算法、
  Native FFI、系统 SDK 探测或 benchmark peer checkout；自动追踪不稳定 upstream HEAD；借缓存任务重开
  `core/error/container_intrusive/util_core/time/platform/mem/obs/simd`；
- tier：P0。结构目标是每个 lock identity 最多一次网络下载/展开，cache hit 只做 fail-closed 身份校验；
  不按一次 configure 点值调参；
- budget / stop：实现 2 小时、验证 45 分钟；只允许下列 FIX。两个 fresh build 不能复用同一 source，
  cache corruption 未 fail closed，或 embedding/offline 合同回退则 `BLOCKED`，不扩建第二套包管理器。

## 本轮冻结的取得语义

源码与编译产物仍然分层，但“编译前准备”不要求用户先手工运行命令：

1. 显式 `FETCHCONTENT_SOURCE_DIR_<NAME>` 优先，供父工程、离线包和本地开发覆盖；
2. 否则查询 `HYLE_DEP_SOURCE_CACHE` 中与中央 lock 完全一致的 immutable identity；
3. 命中时只校验并复用，configure 和 build 均不联网、不重新展开；
4. 在线 configure 未命中时自动调用同一套 `sync` 原语，校验 archive SHA、展开树和 license 后再交给
   FetchContent；网络取得只发生在 configure 的依赖准备阶段，不混入编译步骤；
5. CMake 标准 `FETCHCONTENT_FULLY_DISCONNECTED=ON` 未命中时禁止联网，并给出可执行的预准备命令；
6. `sync` 也可由用户/CI 提前执行；它与 configure 自动填充共享锁、staging 和验证实现，不能形成第二套
   下载逻辑。

因此“第一次 configure 自动下载”和“源码是跨 build 的公共资源”并不冲突：configure 只是确保资源
存在，FetchContent 的 binary/subbuild 状态仍严格属于各自 build tree。

## 依赖 freshness 审计（2026-08-30）

九个已关闭模块的常态第三方产品输入只有：Mem 的 in-tree mimalloc；Obs/Mem 在 `HYLE_TRACE=ON` 时的
Tracy；SIMD 的 Highway、xsimd 与 adapted Highway VQSort。GoogleTest 和 benchmark peers 不是产品
依赖。官方 refs/API 对账结果：

- mimalloc `v3.5.0`、Tracy `v0.14.1`、GoogleTest `v1.18.0` 都是当前最新 release；
- Highway `f503c485...` 与 xsimd `3b3151ff...` 都等于当前官方默认分支 HEAD；
- `highway_port` 是带 origin/content hash 的改造源码，不属于 FetchContent，也不自动追 HEAD；其升级按
  SIMD receipt 的 VQSort reopen trigger 独立进行；
- 其他中央 lock 项仍按 release train/tag/snapshot 策略管理；默认分支比 release tag 新不自动构成升级。

本轮没有已证明需要修改的 dependency revision。版本 freshness 是本轮审计事实，不写成“永远最新”的
构建承诺；更新仍须原子刷新 commit/tree/archive/expanded tree/license 与受影响验证。

## Gap ledger 与冻结 FIX

| ID | 当前缺口 | 冻结 action | 验证 |
|---|---|---|---|
| `BUILD-V6-CACHE-01` | source 与 platform/toolchain build output 混在 `<build>/_deps` | 新增 `HYLE_DEP_SOURCE_CACHE`；archive/source 按 lock identity 共享，binary/subbuild 留在 `HYLE_DEP_BUILD_DIR` | 两个 fresh build 使用同一 canonical source；各自 binary dir 不同 |
| `BUILD-V6-CACHE-02` | 删除 build 后会重新下载；旧根 `.deps` 又不受 lock 管理 | 在线 configure miss 自动在锁内执行统一 `sync`，校验 archive SHA、展开树和 license；hit 不联网但重验源码身份；offline miss fail closed | 第一次在线 configure 自动取得；第二次 disconnected configure 成功；archive/source mtime 不变 |
| `BUILD-V6-CACHE-03` | 并发 configure/中断可能留下半棵源码 | per-identity file lock、独立 staging、验证后 rename；不完整 staging 不可成为命中 | corruption/partial fixture fail closed；重新 sync 可恢复 |
| `BUILD-V6-CACHE-04` | 依赖位置、版本检查和清理没有一个入口 | CMake script-mode tool 提供 `status/sync/check-updates/prune`；prune 只删除 cache marker 下不再由当前 lock 引用的 identity，并要求显式确认 | status/missing/current/stale fixture；无确认拒绝 prune；更新检查不自动改 lock |
| `BUILD-V6-OVERRIDE-01` | 自动 cache 可能抢走 offline/父工程显式 source | 非空 `FETCHCONTENT_SOURCE_DIR_*` 始终优先并走既有严格 verifier；fully-disconnected miss 不联网 | existing offline/component contract；explicit override fixture |
| `BUILD-V6-DOC-01` | 文档仍称 per-build source 是有意行为、根 `.deps` 只是兼容 | 更新架构、依赖/offline/排障说明；明确共享的是 immutable source，不是 build artifact | `check-docs.py` |
| `BUILD-V6-CLEAN-01` | canonical checkout 有大量 ignored build/scratch/object 与旧 `.deps` | 在产品提交之外删除已确认生成物；不删除 `.claude` 或未知目录，不触碰 tracked 文件 | 清理前 dry-run inventory；清理后根目录与 `git status` |

任何 dependency revision、组件/target、产品源、公开头或模块 route 变化都超出本卡，必须另立升级/模块
任务。候选后只允许补验证、receipt 和事实文档。

## Required gates

1. cache contract：online configure miss 自动填充、hit、disconnected hit、offline miss 拒绝、corruption、
   explicit override、concurrent/stale/prune safety；
2. current component/offline/package registry contract；
3. 两个 fresh SIMD selected source configure/build，分别证明共享 Highway/xsimd source 与独立 binary；
4. fresh Core source/static/shared 仍不取得任何依赖；
5. `check-module-closure.py`、`check-docs.py`、`git diff --check`；
6. terminal receipt、限制、维护命令和 build/package LOC。

## Candidate freeze

- product candidate / tree：`750c9892daede4bfab4a5c0a1b7ef7db481b6954` /
  `9bdfd99670ddd7b36d22e27f4769774a9e1a022c`；
- source resource：`HYLE_DEP_SOURCE_CACHE` 只保存中央 lock identity 对应的官方 archive、完整校验后的
  immutable 展开树和管理 metadata；默认位于平台用户 cache，也可由 CMake 变量或同名环境变量指定；
- build state：`HYLE_DEP_BUILD_DIR` 只保存当前 build 的 binary/subbuild，Hyle 只在自己的目录作用域设置
  `FETCHCONTENT_BASE_DIR`，不会改写 embedding parent；旧 `HYLE_DEPS_DIR` 明确拒绝并给出迁移信息；
- population：configure miss 与 `sync` 共用 per-identity lock、partial archive、staging、完整 archive/tree/
  license verifier 和验证后 rename；cache hit 仍重验 identity，但不下载、不展开、不改源码；
- offline：`FETCHCONTENT_FULLY_DISCONNECTED=ON` 缺 source 但存在有效 archive 时可本地重建，source 与
  archive 都缺失时 fail closed；显式 `FETCHCONTENT_SOURCE_DIR_*` / `HYLE_TRACY_DIR` 优先且仍严格校验；
- management：统一工具提供 `status/sync/check-updates/prune`；具名 `HYLE_DEP_REPAIR=YES` 只重建选中
  identity，保留 SHA 正确的 archive；prune 要求 marker 与显式确认，只删除当前 lock 不再引用的 identity；
- CI/product integration：普通构建、产品形状 smoke、locked-feature/Tracy smoke 都走同一 resolver；
  GitHub Actions 以 dependency-lock hash 缓存共享 source tree，不再维护第二套 archive downloader；
- cleanup：canonical checkout 顶层 39 个已确认 ignored build/cache/object 目标与仓库内旧 `.deps` 已删除，
  tracked 文件未被清理；`.claude` 保留，历史证据仍引用且本轮继续使用的 `C:\hyle-scratch` 不冒充废料删除。

Candidate 后产品取得语义、变量、锁/发布算法或正常 consumer journey 变化必须退回 `AUDITED`；只补
验证可移植性、terminal receipt 和事实记录不改变 product identity。

## Verification manifest

| Gate | Verdict | candidate 证据 |
|---|---|---|
| first online configure / disconnected hit | `PASS` | W1-WIN fresh empty cache：SIMD configure 自动下载并校验 Highway+xsimd，20.1 s 完成；第二个 fresh build 在 fully-disconnected 下 1.3 s 配置成功，展开目录 file count/mtime 不变。时间只描述事件，不形成性能门 |
| two build isolation | `PASS` | W1-WIN 两个 fresh SIMD source build 都完整生成 `hyle_simd`；共同 source cache 为 `C:/public/project-hyle/.deps`，各自 `HYLE_DEP_BUILD_DIR` 分别位于自己的 `_deps` |
| cache lifecycle / red team | `PASS` | 真实 Highway lock identity：并发首次初始化、并发 populate、interrupted staging、disconnected hit、源码损坏拒绝、保留 archive 的离线 repair、missing 拒绝、unconfirmed/confirmed prune、unmarked 外来目录拒绝全部通过；Windows 连续 3 轮，WSL/CMake 3.28.3 再通过 1 轮 |
| embedding / override / offline / registry | `PASS` | W1-WIN current `HyleCiComponentContract.cmake` 全通过；两个 embedding fresh build 共用 source、各自 build state，第二个断网；标准 SOURCE_DIR 覆盖安全压过任意同名 first declaration；既有 offline 与 package/license registry 正负门未回退 |
| dependency-free delivery | `PASS` | fresh Core source、relocated shared、relocated static consumer 各 1/1；指定的空 `HYLE_DEP_SOURCE_CACHE` 从未创建 |
| current CPU product shape | `PASS` | shipping-layout `core,util,obs,time` 闭包构建完成，40/40 CTest；GoogleTest source 位于共享 cache，build tree 只生成 `_deps/googletest-build` |
| Tracy / feature package | `PASS` | 默认 `HYLE_TRACE=ON` 自动取得 Tracy 并构建 Obs；locked-feature `trace` scope 的 static/shared SDK、许可证/开发头、搬迁后 consumer 各 1/1 |
| dependency freshness tool | `PASS` | 2026-08-30：Highway/xsimd lock 等于 upstream HEAD；Tracy v0.14.1、GoogleTest v1.18.0、mimalloc v3.5.0 以 release pin 报告 HEAD 差异但不把默认分支误判为 release upgrade |
| workflow / docs / closure | `PASS` | workflow YAML 与 PowerShell parser 通过；329 个 Markdown、930 个相对链接、16-component closure schema 与 `git diff --check` 通过 |
| immutable LOC | `PASS` | candidate 上 `loc-report.py --module build-package --self-check` 通过 |

验证过程中修复了两个真实状态机缺口：断网 resolver 原先会在有效 archive 存在时过早拒绝 source repair；
Windows 两个首次 configure 还可能在目录创建的瞬间命中 `EXISTS/IS_DIRECTORY` TOCTOU。另一个 WSL
`SIGPIPE` 只属于并发合同把两个 CMake 进程串成输出管道的仪器错误；candidate 用静默子进程保留真实
并发和双退出码裁决，没有修改产品算法。

本轮没有运行 M1-MAC、hosted GitHub Actions、locked-feature `adapters/all`、真实 CUDA/Vulkan/HTTP2
或完整 source-package smoke。前两项分别保留为跨平台/托管执行限制；后几项没有改变产品 adapter、lock
或 package registry 语义，不能借本轮 trace/CPU 证据外推为已运行。CI workflow 已静态校验，未消耗用户
所剩不多的 hosted Actions 额度。

## Closure decision

状态：`CLOSED`。共享的是平台无关、中央锁定且每次使用前可重验的 source resource；任何 object、library、
generated CMake state 仍属于单个 build。新机器第一次在线 configure 开箱即用，后续 configure/build 和
断网 fresh build 复用同一 source；显式父工程/离线输入仍有最高优先级，但不能绕过 provenance。
版本查看、预取、修复和清理集中到同一入口，普通产品与 CI 不再各自实现下载器。

重开条件：中央 dependency lock/schema 或 source class 改变；CMake/FetchContent 新版本击穿 override、
scope、file-lock 或 atomic-publish 语义；cache hit 重新下载/展开或跨 build 泄漏 binary state；并发首次
配置产生 partial/corrupt tree；显式 override/offline/package/source bundle 回退；新平台无法表达默认
cache；或正常 consumer 出现可复现的 material configure/build 回归。单次网络抖动、upstream HEAD 变化、
某个未选 optional backend 缺 SDK 或一次冷配置点值不单独重开。

terminal receipt：`build-v6-shared-source-cache` 对 candidate
`750c9892daede4bfab4a5c0a1b7ef7db481b6954` / tree
`9bdfd99670ddd7b36d22e27f4769774a9e1a022c` 的全部本轮适用门为 `PASS`，无 blocker；本页所在提交保存
terminal end receipt。

Candidate immutable LOC：全库代码 **402,475**；自实现包含测试 **96,041**；自实现不包含测试
**59,898**；build/package 实现 **8,810** 行。LOC 只说明规模，不形成完成度或性能证书。
