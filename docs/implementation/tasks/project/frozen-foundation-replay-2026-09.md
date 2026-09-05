# Frozen foundation 联合复验（2026-09）

<!-- PROJECT REPLAY RECORD: ACTIVE -->

> 这是对已经带 `freeze_confirmation.marker=FROZEN` 的底层质因数组件做的只读 current-product replay。
> 旧 `CLOSED`、旧 freeze 页和旧 benchmark 只作输入，不能直接生成本轮结论。只有可复现 finding 命中
> 既有 reopen trigger，受影响组件才进入自己的 `REOPENED -> SCOPED -> AUDITED` 产品轮。

## 身份与固定范围

- baseline commit / tree：
  `6755bcd6dcb8b5f1d21ca1ccafba3fe070a6ee4d` /
  `449f5808e40e03d95a772468fff3582989f287a9`；
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
| core | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| error | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| math | `PASS` | `PASS` | `PASS/P0` | `PENDING` | `PENDING` | `PENDING` |
| container_intrusive | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| util_core | `PASS` | `PASS` | `PASS/P1` | `PENDING` | `PENDING` | `PENDING` |
| time | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| timer_queue | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| mem_reclaim | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| obs | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| platform | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| platform_thread | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| platform_fs | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |

