# 能力完整性标准存量采用记录 — 2026-08

<!-- PROJECT AUDIT RECORD: ACTIVE -->

> 本任务把 2026-08-28 接受的
> [基础库能力完整性决策](../../../architecture/capability-completeness.md)应用到已经实现或已经结案的
> 开发单元。它是项目级审计清单，不替代
> [`module-status.json`](../../module-status.json)，也不因为方法升级而批量改写产品状态或历史 task。

## 进入原因与边界

- baseline：`e99aae855d517ece9d2685b43a9f454699d5c821`；
- defect：新标准首次只落到了活动的 `time-v1`，没有回答 `core/error/util` 等既有实现是否也覆盖了
  各自已经声称提供的完整用户旅程；
- 范围：`core`、`error`、`util_core`、`util`、`container_intrusive` 与 build/package 的只读能力回放，
  以及 `time/platform/mem/obs` 应采用哪一种正常施工路径；
- 非目标：不回写旧 closure task，不用文档补表冒充产品验证，不批量重开 `CLOSED`，不在本任务中
  修产品或重跑纳秒 benchmark，不替 `container/simd/sched/io/compute/graph` 提前收口；
- 每个存量回放的固定输出：capability-family map、现有 public-surface trace、每族覆盖状态、
  `CONFIRMED/METHOD_GAP/PRODUCT_TRIGGER`、受影响 consumer/dependency 和下一动作；
- 每单元调查预算：实现和公开面不超过 30 分钟；build/package 不超过 45 分钟。每个能力族最多两个
  已钉死 strong peers，不发明新 benchmark 姿势。预算到达仍无法裁决 material `MUST` 时记
  `METHOD_GAP`，而不是继续研究。

## 存量采用矩阵

| Unit | current product/unit state | 已有证据 | 新标准尚缺什么 | 采用路径 / 当前 verdict |
|---|---|---|---|---|
| `core` | `CLOSED` | v2 demand ledger、全部公开入口反向追踪、P0 cost/delivery；另有一次只读 methodology replay | 新回放已显式覆盖 family map、public trace、成熟 peer 差异、重复 compiler seam 与三种当前交付 | [`core-capability-replay-v2`](../core/core-capability-replay-v2.md)；`CONFIRMED`，实现改动 0 |
| `error` | `CLOSED` | v3 对 Status/Result/context/native/format/OOM/lifetime 的宽 ledger 与 public trace；v4 有合格 source-consumer 性能恢复 | 新回放已逐族签署构造、传播、组合、失败、teardown、边界与负空间；WinSock/typed rich error 等均有 owner route/trigger | [`error-capability-replay-v1`](../error/error-capability-replay-v1.md)；`CONFIRMED`，实现改动 0 |
| `util_core` | `CLOSED` | RNG/Ref/ScopeGuard/SpinLock/type-name 五族、peer/consumer/复杂度和 public trace | 新回放已签署每族取得、组合、失败、边界、teardown 与完整 owner route；校准四项易用性陷阱和概率测试 | [`util-core capability replay`](../util/util-core-capability-replay-v1.md)；`CONFIRMED`，API/算法/布局改动 0 |
| `util` | `CLOSED` (`util-runtime-v2`) | Registry/demangle 的用户旅程、peer、复杂度、资源和负空间 | 回放确认 Registry 完整，同时发现 owning demangled-text 与两处文档缺口；v2 已补齐并通过三平台正确性、交付、易用性、红队和 P0 结构门 | [`util runtime capability replay`](../util/util-runtime-capability-replay-v1.md) 得到 `PRODUCT_TRIGGER`；[`v2`](../util/util-runtime-v2-closure.md) 已 `CLOSED`，Registry 产品零改动 |
| `container_intrusive` | `CLOSED` | caller-owned list、auto/safe hook、membership transfer、public trace、Boost 同合同证据 | 需把 list family 的 lifecycle/边界/迭代失效与更宽 intrusive framework route 显式判为覆盖状态 | read-only capability replay；`PENDING` |
| build/package | v2 unit `VERIFIED`，持续 workstream | component graph、source/static/shared、offline、并行 scratch、options/CI/version/依赖锁合同 | 旧 unit ledger 没有 family coverage 轴；后续 v3/v4 变更需汇入当前 capability map，不能只复用 v2 表 | read-only development-unit replay；`PENDING` |
| `time` | `AUDITED` | 已按新标准建立 family map/public trace，冻结十项 ledger FIX | 实现、candidate 与全部验证尚未完成 | 继续正常 `AUDITED -> CANDIDATE`，不另做 replay |
| `platform` / `mem` / `obs` | `QUEUED` | 旧实现和历史审查只作输入 | 全部新标准证据尚未建立 | 轮到时走完整正常状态流，不补签旧结论 |

`container`、`simd`、`sched`、`io` 本来就尚未按当前状态机收口；它们将直接采用新标准，不属于本次
“旧 CLOSED 结论回放”。`compute/graph` 仍是未完成产品，不能用存量回放代替建设。

## 回放顺序与停止方式

1. `core`：最小控制样本，验证新 family map 能从既有证据快速形成，而不会再次重构几百行 primitive；
2. `error`：检查基础失败值的既有能力是否真的完整，并重新确认 negative space；
3. `util_core`（已 `CONFIRMED`）、`util`：分别审五个 header-only primitives 与 Registry/demangle，
   禁止再混成一个 util；
4. `container_intrusive`：在 `time` 进入 candidate 前复核其唯一通用直接依赖；
5. build/package：汇总 v2 以后当前交付系统，而不是把模块表面错误解释成交付能力。

每项若 `CONFIRMED`，新增回放记录后保持 `CLOSED`；若只是证据不足，写 `METHOD_GAP` 和精确补证条件，
产品状态不变；若发现真实缺失旅程、错误 owner、半项公开能力或复杂度档缺口，写
`PRODUCT_TRIGGER`，随后为受影响范围创建新 round，依法 `REOPENED -> SCOPED`。产品修复不在本项目
清单中偷偷发生。

## 当前项目结论

`core`、`error`、`util_core` 已分别完成存量回放，verdict 都为 `CONFIRMED`；三者都没有因方法升级
重构产品。util_core 只校准 installed contract/user guide 与一个概率测试，API、算法、布局和运行时代码
不变。`util` 回放得到 `PRODUCT_TRIGGER`：Registry 维持原结论，demangle 缺两个成熟来源共同提供的
owning-text 常用路线；其 v2 易用性切片现已修复并 `CLOSED`。`time-v1` 已直接使用新
capability-family 格式。
`container_intrusive/build-package` 在逐项回放完成前，不能描述成已经按新标准证明完整；其既有状态也
不会仅因方法差异自动失效。
