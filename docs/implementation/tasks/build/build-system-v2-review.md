# Build/package system v2 review

<!-- PROJECT UNIT REVIEW: VERIFIED -->

> 状态：`VERIFIED`。Build/package 是独立开发单元，但不是可链接产品组件；它不进入
> `HyleComponents.cmake` 或 `module-status.json`。本页按模块收口协议保存本轮冻结范围、有限
> `FIX` 集与证据，长期设计仍归 [`build-system.md`](../../../architecture/build-system.md)。

## Closure card

- unit / owner / round：`build-package` / build/package maintainers / `build-v2-01`
- kind / baseline：`REFACTOR` / `7971f35d3bf91664013163ba06ae27d8daf2dd3b`
- 进入原因：产品只承诺 C++ source/static/shared，但机器合同仍编译、导出、安装并测试旧 C/FFI；
  固定 preset build tree 也不适合多个模块或 agent 并行施工。
- 真实 journeys：源码 `add_subdirectory()` 的单组件消费者；只含所选闭包的 static/shared SDK；
  两个并行模块审查在互不共享 CMake cache 的目录中独立 configure/build/test。
- 范围：组件图和 header 所有权、三种 C++ 交付、旧 C/FFI 清除、选项语义、开发入口、package/CI
  合同、预发布版本兼容策略及相应文档。
- 非目标：不重写 CMake；不把每个模块变成独立顶层 project；不裁决 compute/io/simd 的产品能力；
  不实现尚无消费者的 Native FFI；不删除 vendored C、汇编或 OS callback 所需的真实 C ABI。
- 直接消费者：全部产品组件、测试/benchmark、安装包样例、CI、源码授权客户的父工程。
- tier：`P0`。本单元没有运行时热路径；性能门是图复杂度、依赖获取、configure/no-op build、增量
  invalidation 与并行隔离，不用纳秒 benchmark。
- 预算：审计 90 分钟；三批原子实现，每批至多一个主要设计修正；本地验证总计不超过两小时。
  无法在该预算内证明的 feature/backend 语义登记给所属模块，不在本轮继续研究。
- machine / instrument：canonical Windows checkout；CMake configure/build/ctest、安装后独立 consumer、
  `check-docs.py`、`check-module-closure.py`、Git diff。远端 CI 只在额度允许时复证，不替代本地 verdict。
- peers/contracts：CMake target/package/FetchContent 官方模型（项目最低 3.25）；本仓库成熟的
  target-based source/static/shared journey。这里比较所有权和复杂度档，不比较文件行数。
- 停止条件：冻结 `FIX` 全部完成并通过适用门，或精确记录 blocker；不因相邻模块尚未收口而重写其
  feature policy。

## Capability / gap ledger

| ID | demand / normal shape | verdict | severity | 本轮裁决与验证 |
|---|---|---|---|---|
| BUILD-01 | 一份中央组件图同时产生选择闭包与真实 `PUBLIC/PRIVATE` target 边 | `KEEP` | P0 | 保留并加严：manifest 生成真实 target 边，配置阶段复核最终 target 图；component contract PASS |
| BUILD-02 | C++ source/static/shared 三种交付、relocation、runtime staging | `KEEP` | P0 | C++ selected-root source/shared/static + relocation consumer 均 PASS |
| BUILD-03 | 当前产品没有 native-C API 或 Native FFI | `FIX` | P0 | 原生 C++ 类型不再借 C schema；旧 C header/bridge/fixture/managed probe/package journey 已删除，防复发扫描已加入 |
| BUILD-04 | 模块公开头的所有权与直接依赖可机械对账 | `FIX` | P0 | manifest 精确登记特殊 owner；公开头和实现 `.cc/.cu/.inc/...` 的直接 include 对账；安装仅投影 selected closure headers |
| BUILD-05 | 两个模块/agent 可并行开发 | `FIX` | P1 | Windows wrapper 接受绝对独占 `-BinaryDir` 与 `-Component`；core/error 两个 scratch 同时配置 PASS |
| BUILD-06 | 单组件根能独立 source build/test 及 static/shared package consume | `FIX` | P1 | 新增参数化 selected-root smoke；container 的三种交付、relocation 与不引入 compiled simd 均 PASS |
| BUILD-07 | option 表达产品选择且默认配置确定、轻量 | `FIX` | P1 | 后端默认 OFF、测试默认 shipping layout；owner-scoped cache、父 normal/cache 规范化、exact ISA 与 benchmark peer scope 均有合同 |
| BUILD-08 | CI 反馈与额度匹配风险 | `FIX` | P1 | PR/main 只留 Linux 高信号基线；cross/clang-cl/三平台/package/feature 为手工扩展门；本轮未消耗托管额度 |
| BUILD-09 | 0.x 不承诺未冻结的 C++ binary compatibility | `FIX` | P1 | package 使用 `ExactVersion`；pre-1.0 shared loader identity 使用完整 `0.y.z` |
| BUILD-10 | 依赖来源、哈希、许可证及 offline 投影 | `KEEP` | P0 | 本轮只复证，不借机重写锁系统 |
| BUILD-11 | CMake 调用方不可信时的 attestation/token 防护 | `DEFER` | P2 | 当前没有可复现产品缺陷；另开 dependency-lock 简化任务才允许改，不能与 C/组件迁移混在一起 |
| BUILD-12 | 把 build/package 放进产品 component registry | `REJECT` | P0 | 它不可链接且没有产品 artifact；本任务即受检开发单元记录，不伪造 target/component |
| BUILD-13 | 用 `BUILD_SHARED_LIBS` 或每模块独立 `project()` 简化 | `REJECT` | P0 | 会污染父工程或复制图事实；保留 HYLE-owned linkage 与一个可组合 project |

## 冻结 FIX 集与批次

1. `native-c-removal`：BUILD-03，以及被它直接阻断的 C++ native type/测试/文档迁移。
2. `component-delivery`：BUILD-04、05、06。
3. `options-ci-version`：BUILD-07、08、09；同步 architecture/user/developer 文档。

批次外的新发现只在 correctness、security 或当前构建阻断时进入本轮；其他内容写入 backlog。

## Verification manifest

- [x] `include/hyle/c`、产品 `c_*.cc`、旧 C/managed fixtures 不存在；合法的 vendored C/assembly/
  OS `extern "C"` 不受影响。
- [x] Windows all-component shipping layout `62/62`、显式 instrumentation `63/63`；公开头 self-contained。
- [x] core/error 两个不同 scratch root 同时配置，分别只解析 `core` 与 `core,error`。
- [x] container selected-root source + static/shared installed C++ consumer；两种包 relocation 后运行通过。
- [x] all-component CPU configure/build 回归（昂贵 feature 显式 OFF）。
- [x] option 负向/规范化、GTest 复用、pre-1.0 exact package、target platform 与 CI 分层合同通过。
- [x] `python scripts/check-module-closure.py` 与 `python scripts/check-docs.py`。
- [x] clean commit `42a6747` 生成完整源码包；解压后的 all-component、WERROR、feature-OFF build 通过。
- [x] terminal immutable snapshot 四项 LOC；build/package 使用文档定义的专用边界。

本地最终门使用 MSVC 19.50 + Ninja、CMake 4.4.0；`HyleCiComponentContract.cmake`、selected
container delivery smoke、all-component 两种测试布局与文档/闭包/LOC self-check 均通过。Linux、macOS、
AArch64 与 locked optional backend 本轮只静态复核 workflow，未把未运行的托管门写成 PASS。

## Reopen triggers / accepted limits（终态填写）

- Native FFI 只有在具名跨语言 consumer、窄运行时对象模型和绑定 owner 同时存在时另立开发单元；
  不恢复整库镜像式 C API。
- compute、graph 仍未完成；本轮只保证其现有构建形状未被伪 C 清理意外破坏，不签其产品能力。
- container 当前通过仅在 build tree 存在的 `simd_vec_headers` target 借用 fixed-ISA header，不把 compiled
  simd 伪装成产品依赖。把 expert Vec DSL 正式拆成 `simd_vec` 产品组件是可行的，但还需要独立 consumer
  shape、可安装 exact-ISA target helper 与 SIMD 模块审查，不能在 build/package 单元里替 SIMD 做决定。
- io/compute/graph 的当前公开头直接暴露 sched 类型，因此这些根在无 fcontext target 上诚实失败；只有
  后续模块审查拆出真正 sched-free 的公开形状后，组件图才允许变成条件边。
- BUILD-11 的 dependency attestation 简化保持 `DEFER`；现有锁、离线与许可证 fail-closed 合同继续保留。
