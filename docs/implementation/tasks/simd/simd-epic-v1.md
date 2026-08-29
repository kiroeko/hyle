# SIMD/Math 重构与收口 epic

## 目标

在不把 API 和实现扭曲成 benchmark 形状的前提下，把约 3.8 万行 CPU SIMD 产品做成可维护、可升级、
可被通用 C++ 与未来推理引擎自然消费的基础加速层。最终只按具名 capability、consumer contract、ISA、
交付形状和精度合同签有限性能结论，不声明无法证实的“整个 SIMD 全面 SOTA”。

顶层 `math` 不并入本 epic 的产品施工。它只接受一次只读边界复核；若出现满足既有 reopen trigger 的
新事实，另建 `math-v2`，不得借 SIMD ticket 顺手修改已关闭模块。

## 为什么必须使用 epic

- 当前 SIMD 约 38,118 行 tracked 实现、108 个实现文件；约占当前自实现非测试代码的 45.5%；
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
| register/D-tag、dtype、连续 row/buffer、runtime/exact/native dispatch | `hyle::simd` |
| stride/layout/workspace/threading、tensor operator、模型融合、quant block/GEMM | `compute` 或消费者 |
| GPU/accelerator backend | 对应 accelerator owner |

SIMD 可以提供模型无关且重复出现的连续 row primitive；不得变成第二套 tensor runtime，也不得因为 ggml
存在一个名字就机械扩张公共 API。

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
| 03 | [Highway integrated port 刷新](tickets/simd-03-highway-port.md) | product | 02 | 条件是 | 1 日 |
| 04 | [xsimd/transcendental port 刷新](tickets/simd-04-xsimd-port.md) | product | 03 | 条件是 | 1 日 |
| 05 | [register API 与定宽 backend](tickets/simd-05-register-fixed.md) | product | 04 | 是 | 1 日 |
| 06 | [scalable backend、dispatch 与交付](tickets/simd-06-scalable-dispatch.md) | product | 05 | 是 | 1 日 |
| 07 | [dtype、narrow 与普通 row ops](tickets/simd-07-row-dtype.md) | product | 06 | 是 | 1 日 |
| 08 | [transcendental 与 activation](tickets/simd-08-activation.md) | product | 07 | 是 | 1 日 |
| 09 | [search 与 scan](tickets/simd-09-search-scan.md) | product | 08 | 是 | 1 日 |
| 10 | [bitpack 与 random](tickets/simd-10-bitpack-random.md) | product | 09 | 是 | 1 日 |
| 11 | [ordering、radix 与 top-k](tickets/simd-11-ordering.md) | product | 10 | 是 | 1 日 |
| 12 | [集成、文档、六问与 closure](tickets/simd-12-integration-closure.md) | verification | 11 | 仅修复阻断项 | 1 日 |

条件产品票允许得出 `KEEP CURRENT` 或 `DROP PORT`；“升级”不预设一定复制更多上游代码。

## 当前未提交产品改动的归属裁决

| 工作树改动 | ticket | 当前裁决 | 原因 |
|---|---|---|---|
| inference row seam 的 audit revision 2 | 00/后续审计输入 | `KEEP` | 修正了把基础 row arithmetic 误判为完整的事实错误 |
| `index_of_min/index_of_max` 实现与 Runtime v8 草稿 | 09 | `REWORK` | 有 Qwen/ggml/Highway 需求，但尚未绑定完整边界、跨 ISA 与 P1 证据 |
| stable radix/argsort workspace/API 草稿 | 11 | `REWORK` | 修复真实 alias/OOM/API 缺陷，但 overlap、异常说明、float order 与交付仍须独立审计 |
| allocation-free activation kernel 两个未跟踪头 | 08 | `REWORK` | 对应已准入需求，但尚未接入 build/runtime/public API/tests，不能作为候选 |
| integrated Highway content hash 改动 | 03/11 | `REGENERATE` | hash 必须由最终接受的 port 内容机械生成，当前字面值不单独继承 |
| core/util/methodology 等其他脏文件 | 不属于本 epic | `PRESERVE` | 不得顺手提交或覆盖 |

`REWORK` 表示保留为该 ticket 的输入草稿，不代表 API、算法或代码行已经接受。任何草稿若在所属 ticket
内无法满足合同，会记录理由后从产品树删除。

## 文档终态

- architecture：只保留长期边界、vendor ownership、dispatch/accuracy/resource 决策；
- user：只保留普通、expert native、exact/fat-kernel 的选择方法、合同和陷阱；
- developer：仪器、dispatch 维护与已复现疑难问题；
- implementation：本 epic、tickets、历史 task 和按 family 保存的证据。

`simd-design.md` 中按日期追加的 stage chronology 不属于 architecture；旧性能表不得继续充当当前证书。

## Epic 完成条件

- 每张 ticket 都为 `DONE` 或有不会被掩盖的 terminal blocker；
- 所有准入 `MUST` 为 `COMPLETE/ROUTED`，不存在 `PARTIAL/MISSING + KEEP`；
- 第三方 integrated source 的版本、修改面、许可证、semantic/mechanism/delivery owner 可追踪；
- source/static/shared、native/exact/fat/runtime、直接 consumer 与三平台适用 correctness 通过；
- 每个 P1 family 只有具名、合格、可重放 verdict，未测 ISA/类型/操作不被外推；
- C/Native FFI 保持清洁负空间，C++ 普通入口无需理解内部 selector；
- architecture/user/developer/implementation 四类文档无互相冲突的当前结论；
- 当前 closure round 形成不可变 candidate、独立红队、六问直答、四项 LOC 和 terminal receipt。
