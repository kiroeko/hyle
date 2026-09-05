# Frozen foundation 联合复验（2026-09）

<!-- PROJECT REPLAY RECORD: CLOSED -->
<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对已经带 `freeze_confirmation.marker=FROZEN` 的底层质因数组件做的只读 current-product replay。
> 旧 `CLOSED`、旧 freeze 页和旧 benchmark 只作输入，不能直接生成本轮结论。只有可复现 finding 命中
> 既有 reopen trigger，受影响组件才进入自己的 `REOPENED -> SCOPED -> AUDITED` 产品轮。

## 身份与固定范围

- baseline commit / tree：
  `6755bcd6dcb8b5f1d21ca1ccafba3fe070a6ee4d` /
  `449f5808e40e03d95a772468fff3582989f287a9`；
- immutable validation source：`27aa252fd66c6ae51e7a6b1dab5030bce6183436` / tree
  `57e6be3c2cd3c42cbd6838e4d51b1e311a649661`；两者之间只增加本 replay card，没有 scoped
  product/CMake 变化；本轮后续修改也只涉及 evidence、registry 与文档；
- reviewer / date：Codex / 2026-09-05；
- dirty-worktree exclusion：Container/Mem/Util performance replay 的既有未提交文件不属于本轮 baseline，
  不进入构建、结论或提交；所有产品判断使用上述 immutable Git tree；
- scope：`core`、`error`、`math`、`container_intrusive`、`util_core`、`time`、`timer_queue`、
  `mem_reclaim`、`obs`、`platform`、`platform_thread`、`platform_fs`；
- Math/Util-core 的 `ff23fc18` 回放可作为本轮刚完成的输入，但仍检查其与其余组件的 dependency/build
  组合；Time 不继承旧确认，照常复验；
- non-goals：重新发散功能超集、上层 Mem/Container/Util-runtime/Sched/IO 产品收口、Compute/Graph、
  Native FFI、Formal 证书、公共性能仪器改造和为了点值改变接口/layout。

## 依赖合同

| factor | public dependencies | private dependencies |
|---|---|---|
| core | — | — |
| error | core | — |
| math | core | — |
| container_intrusive | core | — |
| util_core | core | — |
| time | core | — |
| timer_queue | core, time, container_intrusive | — |
| mem_reclaim | core | — |
| obs | core, error, time | mem_reclaim |
| platform | core, error | — |
| platform_thread | core, error, time | platform |
| platform_fs | core, error, time | platform |

这张表必须同时与中央 component graph、真实 target link、公开 header include 和安装后的传递依赖一致。
“尽量复用”不是按类型名替换：只有完整 ownership/lifetime/error/concurrency/complexity 合同一致且依赖方向
合法，才要求迁移到较低 factor；不为隐藏一条表达式制造反向依赖。

## 固定检查与裁决规则

### 1. 接口、易用性与测试污染

- 从普通 C++ consumer、现有 user guide/sample 和下游调用反查 public surface；逐项检查命名、推导、
  ownership/lifetime、nullable/failure、thread-safety、`noexcept`、误用诊断和标准类型互操作；
- 扫描固定 N、地址/placement、执行顺序、peer、benchmark-only fast mode、常态测量 metadata 和只为
  harness 存在的 overload/字段；测试可以调用 API，测试参数不得反向决定 API；
- 功能超集只做回归：已有 admitted family 若 `PARTIAL` 或自然 journey 实际不可用才形成 trigger；普通
  feature 愿望进入 watchlist，不在本 replay 扩 scope。

### 2. 性能

- P0：检查复杂度、资源次数、对象 layout、分支/锁/分配/动态分发和 optimized code shape；没有合理
  same-contract peer 的 primitive 不制造纳秒榜；
- P1：先逐格证明 ownership、lifetime、failure、thread-safety、progress、charged phases、work amount、
  delivery 与 consumer provenance 等价。比较不是同合同就记 `NOT_COMPARABLE/METHOD_GAP`，不能用绿色
  non-inferiority 掩盖苹果与橘子；
- 合同、candidate、peer、compiler、instrument 与 retained artifact 均未 materially 改变时，先用 deep
  verifier 复验；有 material drift 或证据不可验证时，每个合同最多一次 fresh n=15 qualification；
- 稳定 material `FAIL` 只允许一次机制诊断。找到产品原因则触发模块 reopen；找不到则保留具名 blocker，
  不修改 workload、margin、API 或 layout 追绿。

### 3. Build/CMake、依赖与下游回放

- exact baseline 运行中央 component contract；三平台运行 12-factor selected closure、逐头与适用
  sanitizer/WERROR；
- consumer-facing `core/error/math/time/platform/obs` 及内部 factor 所属的 `container/util/mem` 聚合关系
  检查 source/static/shared。header-only factor 不制造伪 binary journey；
- 扫描下层后续新增 API 与本 12 个 factor 的当前调用：breaking drift、错误调用或完整同合同重复机制是
  trigger；纯 additive API 不强迫迁移；
- 当前 CMake 选择器、公开/私有 edge、安装 export 和 source embedding 必须表达同一依赖图。

## 批次、预算与停止条件

1. `core/error/math/container_intrusive/util_core`；
2. `time/timer_queue`；
3. `platform/platform_thread/platform_fs`；
4. `mem_reclaim/obs`；
5. 组合 CMake、交付和跨批次 dependency replay。

每批最多 60 分钟只读审计和验证；全部 P1 fresh sampling 合计最多 90 分钟；公共仪器不在本轮修改。
组件满足全部适用门时更新其 freeze confirmation 到本轮 reviewed identity；出现具体产品 trigger 时只重开
该组件。任何未运行项不写 `PASS`；本项目 replay 自身不批量改产品状态。

## 结果矩阵

| factor | interface/usability | pollution | P0/P1 | build/delivery | dependency/reuse | verdict |
|---|---|---|---|---|---|---|
| core | `PASS` | `PASS` | `PASS/P0` | `PASS` | `PASS` | `FROZEN` |
| error | `PASS` | `PASS` | `PASS/P1` | `PASS` | `PASS` | `FROZEN` |
| math | `PASS` | `PASS` | `PASS/P0` | `PASS` | `PASS` | `FROZEN` |
| container_intrusive | `PASS` | `PASS` | `PASS/P1` | `PASS` | `PASS` | `FROZEN` |
| util_core | `PASS` | `PASS` | `PASS/P1` | `PASS` | `PASS` | `FROZEN` |
| time | `PASS` | `PASS` | `PASS/P0` | `PASS` | `PASS` | `FROZEN` |
| timer_queue | `PASS` | `PASS` | `PASS/P1` | `PASS` | `PASS` | `FROZEN` |
| mem_reclaim | `PASS` | `PASS` | `PASS/P1` | `PASS` | `PASS` | `FROZEN` |
| obs | `PASS` | `PASS` | `PASS/P1-LIMITED` | `PASS` | `PASS` | `FROZEN` |
| platform | `PASS` | `PASS` | `PASS/P0` | `PASS` | `PASS` | `FROZEN` |
| platform_thread | `PASS` | `PASS` | `PASS/P0` | `PASS` | `PASS` | `FROZEN` |
| platform_fs | `PASS` | `PASS` | `PASS/P0` | `PASS` | `PASS` | `FROZEN` |

## 当前源码与接口复核

- 从各 factor 的既有 `freeze_confirmation.reviewed_commit` 到 immutable validation source，逐一对
  factor-owned public headers 和 implementation 做 `git diff --name-only`，12 项均为
  `EXACT_PRODUCT_PATH`；没有 public API、layout、算法或正常成本漂移。
- 逐头复核 consumer guide/sample 与公开 surface：命名、推导、owner/lifetime、nullable/failure、
  concurrency、`noexcept` 与标准类型互操作仍一致。公开头没有固定 benchmark N、peer 名、placement/
  order、measurement metadata、benchmark-only fast mode 或为 harness 新增的 overload/字段。
- Core 保持无可恢复错误/日志/容器策略的最小编译基础；Error 保持 owning/borrowed Status/Result 与
  system adapters；intrusive factor 保持 caller-owned stable-address membership；Util-core 保持
  non-owning callable/resource/borrow 基元；这些边界没有为了提高复用率强行合并。
- Time 产品自然聚合 `time + timer_queue + container_intrusive`；Obs public 使用 Core/Error/Time 并
  private 使用 `mem_reclaim`；Platform 产品聚合 host/thread/fs 三 factor。只有 ownership、lifetime、
  concurrency 与复杂度合同相同才复用下层 factor；没有发现重复实现同一完整合同或非法反向依赖。

## 性能复核与“苹果对橘子”裁决

- Core、Math、Time、Platform 三 factor 的签发层级仍是 P0：复杂度、资源次数、layout、分支/锁/分配、
  dynamic dispatch 与 optimized code shape 未漂移。Time host 与 Platform host/thread/fs 的旧 P1
  只作具名 normal-route 补充，不把它们提升成“整个模块 SOTA”。
- Error、Container-intrusive、TimerQueue、Mem-reclaim、Obs 和 Util-core 的 retained bundles 均由
  当前统一 verifier 深验证为 `QUALIFIED / CURRENT / paths_verified`；candidate、peer、compiler、
  instrument、contract 与 raw artifact 未改，因此没有按规则重复采样。
- 旧包引用的大 consumer 文件发生 12 条非语义 SHA 漂移；逐项 materiality review 与精确 hash 边见
  [current-source provenance review](../../evidence/performance/provenance-reviews/frozen-foundation-current-source-2026-09.md)。
  当前 resolver 全部返回 `REVIEWED_NON_MATERIAL`；没有 wildcard，也没有把上层模块收口状态带入结论。
- Error/Abseil carrier、intrusive/Boost、TimerQueue/Seastar、reclamation/Folly 以及 Platform 的锁定
  Win32/oneTBB-shaped/Folly-shaped 行，在 ownership、lifetime、thread/progress、工作量与 charged
  phase 上仍为同合同。声明只覆盖各自冻结 journey 和 5% 工程 margin。
- Obs 复核发现两条历史 contract 的 `SAME_CONTRACT` 标签过强：ordinary 路线的 mandatory metadata
  不同，structured 路线的并发 admission 保证不同。不可变旧包不回写；性能页已撤销这两条 peer
  non-inferiority 外推，只保留 natural-route 点值/用户预算。受 `mem_reclaim` 影响的 concurrent
  registry 同实现回归仍为同合同且通过 5% non-inferiority，因此 Obs 的受限 P1 工程门保持 `PASS`，
  但不形成 universal SOTA 声明。

## Build、交付与跨平台证据

- W1-WIN：exact 12-factor、Release/shared、WERROR fresh configure/build；93 targets，46/46 tests
  `PASS`，包括 12 个实现测试、逐公开头、适用 codegen/symbol/no-exception 与 tooling tests。
- W1-WSL：同一 archive、GCC 14.2、Release/shared、WERROR、UBSan，42/42 `PASS`。
- M1-MAC：同一 archive、AppleClang 21 arm64、Release/shared、WERROR、UBSan，42/42 `PASS`。
- 当前中央 component contract `PASS`：中央 graph 与真实 target 的 public/private edge、split-header
  owner、factor/product visibility、选择闭包、安装 export、source embedding 与 dependency source
  cache 合同一致。
- W1-WIN 对 `core/error/math/time/obs/platform/mem/container/util` 九个用户产品逐一执行 selected
  product smoke；每个产品的源码嵌入、重定位 shared SDK consumer、重定位 static SDK consumer 全部
  `PASS`。后三个产品只用于证明其内部 frozen factor 的聚合/交付可达，不据此把尚未按新标准复核的
  Mem/Container/Util-runtime 产品整体签成 frozen。

## 最终裁决

本联合 replay 没有发现命中既有 reopen trigger 的产品事实。12 个 factor 保持 `CLOSED + FROZEN`；
本结论更新的是当前证据身份和声明边界，不伪造一次产品重构，也不把旧 `CLOSED` 自动升级成新冻结。
后续只能由各自 registry 中的具名 reopen trigger 重开；新增 peer、命名偏好、单次纳秒点值或“再看一遍”
均不足以重开。
