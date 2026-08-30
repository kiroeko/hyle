# SIMD/Math 重构与收口 epic

> 当前执行面已在 2026-08-30 转入
> [`simd-v3 multi-provider routing`](simd-v3-provider-routing.md)：Highway、xsimd 与 HYLE 是并存候选，
> baseline 的 387 个 operation 必须逐项有账、比较和终态。SIMD-00～11 及下文原 vendor-first 计划保存已发生
> 的工作；SIMD-12 已被新范围取代，不再授权“抽样后整体删除”。
>
> 当前进度：SIMD-13、SIMD-14 已完成，SIMD-15 正在执行。下文 00～12 的 vendor-first 目标形状是历史方案，
> 不能覆盖 v3 的多 provider、逐项资格化与 fail-closed route 合同。

## 目标

在不把 API 和实现扭曲成 benchmark 形状的前提下，把约 3.8 万行 CPU SIMD 产品迁移为“成熟 vendor
机制优先、HYLE consumer contract 与差异化 kernel 为主”的可维护基础加速层。目标不是继续复制上游
源码形成私有 fork，而是直接锁定并组合 Highway 等成熟实现，自动取得其 ISA/能力演进；HYLE 自实现只
保留真实语义、交付或 consumer gap。最终只按具名 capability、consumer contract、ISA、交付形状和
精度合同签有限性能结论，不声明无法证实的“整个 SIMD 全面 SOTA”。

顶层 `math` 不并入本 epic 的产品施工。它只接受一次只读边界复核；若出现满足既有 reopen trigger 的
新事实，另建 `math-v2`，不得借 SIMD ticket 顺手修改已关闭模块。

## 为什么必须使用 epic

- 当前 SIMD 约 3.86 万行 tracked 实现；其中 `vec.h`、`native/*` 与七套 `backend/*` 约 24,990 行，
  是本轮必须清算的平行机制，不得因普通 family 已迁移就继续冒充终态产品；
- tests/simd 约 11,475 行，bench/simd 约 5,846 行；全量操作、类型、size、ISA、peer 的笛卡尔积不可作为
  日常收口门；
- 48 份相关文档中 37 份是历史 task，architecture 又混入阶段流水账和旧终态叙事；
- 现有 `simd-v1` 把五组产品修复和聚合 P1 矩阵绑成一个批次，超过普通一日工作包边界；
- 产品路径包含 Highway/xsimd 的 integrated-source port，升级不是简单改 FetchContent tag。

## 产品边界

| 层 | owner |
|---|---|
| C++23 已有的普通 scalar primitive | 标准库 |
| 非负整数 shape 的 ceiling division / representable round-up | 已关闭的 `hyle::math` |
| dtype、连续 row/buffer、private runtime、exact 与 source-fat dispatch | `hyle::simd` |
| stride/layout/workspace/threading、tensor operator、模型融合、quant block/GEMM | `compute` 或消费者 |
| GPU/accelerator backend | 对应 accelerator owner |

SIMD 可以提供模型无关且重复出现的连续 row primitive；不得变成第二套 tensor runtime，也不得因为 ggml
存在一个名字就机械扩张公共 API。

## Vendor-first 目标形状

1. 普通用户继续使用稳定、自然的 `hyle::simd` pointer/span/buffer API，不接触 vendor 宏和内部类型。
2. expert register/kernel 用户优先直接使用锁定 Highway 能力或极薄 HYLE adapter，不再维护 387-name
   平行 DSL；现有 D-tag surface 是否保留必须由迁移 proof 决定。
3. compiled Runtime/exact/fat kernels 优先直接在 Highway 上实现；Platform 仍为全库提供 CPU facts，但
   SIMD 不为“统一”而复制 Highway 已正确拥有的 ISA 选择。
4. xsimd/SLEEF 只在 Highway 无等价能力或资格化 accuracy/performance 不满足时补位，不形成第二套完整
   portable backend。
5. checked-in Highway/xsimd ports 是待消减的过渡债务。能 wrap 就删除，不能 wrap 才保留最小 local delta。

## 执行合同

1. ticket 严格按依赖顺序执行，同时最多一个 `ACTIVE`；canonical Windows worktree 不并行写同一产品面。
2. 每张产品票必须已被当前 closure card 的冻结 ledger/FIX 集纳入；否则停止并重新裁决状态。
3. 普通票最多一个工程日；性能调查最多 60 分钟、一次反序复现、一次机制检查、一个产品修改和一次重跑。
4. 正确性广覆盖，性能只测该 capability 的少量正常 consumer contract。`n=31` 只测 tail/correctness；
   `n=4096` 与适用 streaming size 才进入计时。
5. source-primary 与 source-primary peer 同 TU/同 ISA/同语义；runtime 路线必须与含等价 dispatch 成本的路线
   比较。精度、稳定性、NaN、alias、workspace 或异常合同不同即 `NOT_COMPARABLE`。
6. 小于 5% 的差异不驱动产品，除非机制明确且可传播到真实 consumer；无真硅的 ISA 只签 correctness。
7. ticket 不得修改 closure checker、中央统计器、机器 profile 或 material margin。仪器失格登记独立 blocker。
8. 每票留下 input/output commit、受影响路径、correctness/delivery/performance verdict 与后续影响图。最终候选
   运行全 family correctness smoke；性能只重放受后续改动影响的 family 和预冻结 sentinel。
9. 每票同步其架构或用户合同，不在末尾一次性重写所有文档。历史 task 保留为事实，但不进入当前导航。
10. 到预算必须 `DONE/BLOCKED` 或退回当前 closure 阶段；“继续研究”不是 ticket 状态。

## 顺序 ticket

| 顺序 | ticket | 类别 | 依赖 | 产品修改 | 预算 |
|---:|---|---|---|---|---|
| 00 | [控制面与草稿归属](tickets/simd-00-control-plane.md) | program | 无 | 否 | 90 分钟 |
| 01 | [Math 边界与文档权威](tickets/simd-01-math-doc-authority.md) | audit | 00 | 默认否 | 2 小时 |
| 02 | [vendor ownership 与升级决策](tickets/simd-02-vendor-ownership.md) | audit | 01 | 否 | 4 小时 |
| 03 | [Highway 直接承载可行性](tickets/simd-03-highway-adoption.md) | architecture spike | 02 | 隔离 proof | 1 日 |
| 04 | [数学 vendor 合并与 port 消减](tickets/simd-04-xsimd-port.md) | product | 03 | 条件是 | 1 日 |
| 05 | [register API 迁移边界与定宽 source seam](tickets/simd-05-register-fixed.md) | product | 04 | 是 | 1 日 |
| 06 | [scalable backend、dispatch 与交付](tickets/simd-06-scalable-dispatch.md) | product | 05 | 是 | 1 日 |
| 07 | [dtype、narrow 与普通 row ops](tickets/simd-07-row-dtype.md) | product | 06 | 是 | 1 日 |
| 08 | [transcendental 与 activation](tickets/simd-08-activation.md) | product | 07 | 是 | 1 日 |
| 09 | [search 与 scan](tickets/simd-09-search-scan.md) | product | 08 | 是 | 1 日 |
| 10 | [bitpack 与 random](tickets/simd-10-bitpack-random.md) | product | 09 | 是 | 1 日 |
| 11 | [ordering、radix 与 top-k](tickets/simd-11-ordering.md) | product | 10 | 是 | 1 日 |
| 12 | [机制清算、集成、文档、六问与 closure](tickets/simd-12-integration-closure.md) | superseded historical work | 11 | 停止；不得继续删除 | terminal |
| 13 | [387-operation 全量 inventory 与审计](tickets/simd-13-all-op-audit.md) | scope audit / machine contract | v3 card | 否 | 4 工程时 |
| 14 | [多 provider 静态路由底座](tickets/simd-14-provider-substrate.md) | product architecture | 13 | 是 | 1 日 / P1 30 分钟 |
| 15 | [全部 operation 路由资格化](tickets/simd-15-route-qualification.md) | correctness / P1 route audit | 14 | 只提取赢家 | 2 日 / 串行机器 8 小时 |
| 16 | [consumer 迁移与旧机制清算](tickets/simd-16-migration-cleanup.md) | product migration / deletion | 15 | 是，受 ledger 控制 | 1 日 / P1 60 分钟 |
| 17 | [交付与收口](tickets/simd-17-delivery-closure.md) | candidate / verification | 16 | 只修集成 blocker | 1 日 / 串行机器 4 小时 |

条件产品票允许得出 `DIRECT WRAP / DELETE PORT / KEEP MINIMAL PORT`；“升级”不预设一定复制更多上游代码。

SIMD-00～11 的 `DONE` 是其能力、语义和当时有限 FIX 的历史事实，不等于旧 register mechanism 已经删除。
SIMD-12 原计划已被 owner 的 multi-provider/387 全量要求击穿。当前每项终态细化为
`KEEP_HYLE/ROUTE_HIGHWAY/ROUTE_XSIMD/REMOVE_REDUNDANT`；只有 387/387 ledger 零漏项、零 `PENDING`，
且获胜实现先提取、引用和覆盖迁移后，SIMD-16 才有删除权限。

## 控制面建立时的草稿归属裁决

| 控制面快照中的工作树改动 | ticket | 裁决/当前结果 | 原因 |
|---|---|---|---|
| inference row seam 的 audit revision 2 | 00/07 | `KEEP` → SIMD-07 `DONE` | 修正了把基础 row arithmetic 误判为完整的事实错误，最终由 07 的完整 row 合同吸收 |
| `index_of_min/index_of_max` 实现与 Runtime v9 草稿 | 09 | `REWORK` → SIMD-09 `DONE` | 以 direct Highway + 最小 NaN gap 重写并完成跨 ISA、交付与 P1 证据，没有原样接受草稿 |
| stable radix/argsort workspace/API 草稿 | 11 | `REWORK` → SIMD-11 `DONE` | 以完整 typed-scratch type family、overlap/OOM/index/float-order 合同重写并完成交付与有界 P1 证据，草稿未原样继承 |
| allocation-free activation kernel 两个未跟踪头 | 08 | `REWORK` → SIMD-08 `DONE` | 算法意图被重写进 direct-Highway math seam，旧草稿没有原样进入候选 |
| integrated Highway content hash 改动 | 03/11 | 03/11 均已按最终内容机械锁定 | hash 由最终接受的最小 port 内容生成，控制面字面值未单独继承 |
| core/util/methodology 等其他脏文件 | 不属于本 epic | `PRESERVE` | 不得顺手提交或覆盖 |

本表保存 SIMD-00 建立控制面时的归属事实，并补记 terminal 结果；它不是当前工作树清单。`REWORK`
表示草稿只作为所属 ticket 的输入，不代表 API、算法或代码行已经接受。任何草稿若不能满足冻结合同，
会记录理由后重写或从产品树删除。

## 文档终态

- architecture：只保留长期边界、vendor ownership、dispatch/accuracy/resource 决策；
- user：只保留普通、fixed-source、exact/source-fat kernel 的选择方法、合同和陷阱；
- developer：仪器、dispatch 维护与已复现疑难问题；
- implementation：本 epic、tickets、历史 task 和按 family 保存的证据。

`simd-design.md` 中按日期追加的 stage chronology 不属于 architecture；旧性能表不得继续充当当前证书。

## Epic 完成条件

- 每张 ticket 都为 `DONE` 或有不会被掩盖的 terminal blocker；
- 所有准入 `MUST` 为 `COMPLETE/ROUTED`，不存在 `PARTIAL/MISSING + KEEP`；
- 第三方 integrated source 的版本、修改面、许可证、semantic/mechanism/delivery owner 可追踪；
- source/static/shared、fixed/exact/fat/runtime、直接 consumer 与三平台适用 correctness 通过；
- 每个 P1 family 只有具名、合格、可重放 verdict，未测 ISA/类型/操作不被外推；
- C/Native FFI 保持清洁负空间，C++ 普通入口无需理解内部 selector；
- architecture/user/developer/implementation 四类文档无互相冲突的当前结论；
- 当前 closure round 形成不可变 candidate、独立红队、六问直答、四项 LOC 和 terminal receipt。
