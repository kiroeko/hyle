# Build/package system v2 review

<!-- PROJECT UNIT REVIEW: ACTIVE -->

> 状态：`AUDITED`。Build/package 是独立开发单元，但不是可链接产品组件；它不进入
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
| BUILD-01 | 一份中央组件图同时产生选择闭包与真实 `PUBLIC/PRIVATE` target 边 | `KEEP` | P0 | 保留 `HyleComponents` + final target graph audit；component contract |
| BUILD-02 | C++ source/static/shared 三种交付、relocation、runtime staging | `KEEP` | P0 | 保留；用 C++ installed consumers 复证 |
| BUILD-03 | 当前产品没有 native-C API 或 Native FFI | `FIX` | P0 | 先把 time/obs/io/simd 的原生类型移出 C schema，再删 `include/hyle/c`、`c_*.cc`、C/managed fixtures、符号审计与伪 package journey；加入防复发合同 |
| BUILD-04 | 模块公开头的所有权与直接依赖可机械对账 | `FIX` | P0 | 中央 manifest 登记 owned headers；扫描所有直接 `hyle/<owner>` include，不再只特判 Core；安装只投影 selected closure headers |
| BUILD-05 | 两个模块/agent 可并行开发 | `FIX` | P1 | 提供显式 module + 独占 scratch root 的跨平台入口；禁止把固定 preset build tree 当共享 cache |
| BUILD-06 | 单组件根能独立 source build/test 及 static/shared package consume | `FIX` | P1 | 参数化 selected-root smoke；all-component smoke 仍是整图门，不能替代该证据 |
| BUILD-07 | option 表达产品选择且默认配置确定、轻量 | `FIX` | P1 | 昂贵可选 backend 默认 OFF；普通测试默认 shipping layout，instrumentation 显式开启且不得污染 benchmark；嵌入默认不隐式组装 all；只在 owner 组件入图时声明 feature option |
| BUILD-08 | CI 反馈与额度匹配风险 | `FIX` | P1 | push/PR 保留基线与文档合同；cross、clang-cl exact、locked-feature/package 改为手工扩展门；不把未运行门写成 PASS |
| BUILD-09 | 0.x 不承诺未冻结的 C++ binary compatibility | `FIX` | P1 | package version 使用 exact compatibility；每个 pre-1.0 release 使用完整 loader identity |
| BUILD-10 | 依赖来源、哈希、许可证及 offline 投影 | `KEEP` | P0 | 本轮只复证，不借机重写锁系统 |
| BUILD-11 | CMake 调用方不可信时的 attestation/token 防护 | `DEFER` | P2 | 当前没有可复现产品缺陷；另开 dependency-lock 简化任务才允许改，不能与 C/组件迁移混在一起 |
| BUILD-12 | 把 build/package 放进产品 component registry | `REJECT` | P0 | 它不可链接且没有产品 artifact；本任务即受检开发单元记录，不伪造 target/component |
| BUILD-13 | 用 `BUILD_SHARED_LIBS` 或每模块独立 `project()` 简化 | `REJECT` | P0 | 会污染父工程或复制图事实；保留 HYLE-owned linkage 与一个可组合 project |

## 冻结 FIX 集与批次

1. `native-c-removal`：BUILD-03，以及被它直接阻断的 C++ native type/测试/文档迁移。
2. `component-delivery`：BUILD-04、05、06。
3. `options-ci-version`：BUILD-07、08、09；同步 architecture/user/developer 文档。

批次外的新发现只在 correctness、security 或当前构建阻断时进入本轮；其他内容写入 backlog。

## Verification manifest（待执行）

- [ ] `include/hyle/c`、产品 `c_*.cc`、旧 C/managed fixtures 不存在；合法的 vendored C/assembly/
  OS `extern "C"` 不受影响。
- [ ] 受影响模块 C++ tests，至少 Windows selected roots；公开头 self-contained。
- [ ] 两个不同 scratch root 并行配置不同组件，cache/依赖目录互不覆盖。
- [ ] selected-root source + static/shared installed C++ consumer；relocation。
- [ ] all-component CPU configure/build 回归（昂贵 feature 显式 OFF）。
- [ ] option 负向合同、pre-1.0 exact package version、CI workflow syntax/分层检查。
- [ ] `python scripts/check-module-closure.py` 与 `python scripts/check-docs.py`。
- [ ] terminal immutable snapshot 四项 LOC；build/package 使用文档定义的专用边界。

## Reopen triggers / accepted limits（终态填写）

- Native FFI 只有在具名跨语言 consumer、窄运行时对象模型和绑定 owner 同时存在时另立开发单元；
  不恢复整库镜像式 C API。
- compute、graph 仍未完成；本轮只保证其现有构建形状未被伪 C 清理意外破坏，不签其产品能力。

