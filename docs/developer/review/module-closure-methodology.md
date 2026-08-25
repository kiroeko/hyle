# HYLE 模块审查与收口方法 v2

> 本方法适用于每个模块及 build/package system。目标不是把 checklist 打绿，而是建立可追溯、
> 可复现、能停止也能正确重开的模块证书。当前施工状态见
> [implementation/status](../../implementation/status.md)。

## 1. “收口”意味着什么

一个 `SIGNED` 模块必须同时满足：

1. 常见且属于本层的需求都有明确去向，没有被 benchmark 形状代替；
2. 类型、ownership、lifetime、并发、失败、复杂度和依赖方向适合真实消费者；
3. 正确性与边界经过独立反证，构建/交付闭包真实可用；
4. 性能仪器先被证明能分辨真实差异，再对正常 workload 与强 peer 给出有限范围的结论；
5. C++ 用户能自然完成任务，文档职责清楚且与代码一致；
6. 没有未知的 material 风险；剩余非目标、限制和重新开启条件已经登记。

收口不表示功能永远不增加，也不表示每个纳秒点值都领先。它表示在已登记需求与证据范围内，
继续重构没有足够理由，应停止扰动。

## 2. 状态机

| 状态 | 含义 | 进入条件 |
|---|---|---|
| `OPEN` | 范围与旧结论均可质疑 | 开始新审查或证书失效 |
| `AUDITED` | 需求、能力、消费者、依赖和差距已盘清 | 账本可追溯，红队问题有去向 |
| `CANDIDATE` | 本轮产品形状冻结 | 实现、公开面、合同和测试已冻结；只修证据发现的缺陷 |
| `EVIDENCE_COMPLETE` | 所有适用门已有可复现 verdict | 正确性、性能、交付、文档和红队报告齐全 |
| `SIGNED` | 本轮停止关注 | 无未知 material 问题；限制与 reopen trigger 明确 |
| `REOPENED` | 已签证书被新事实击穿 | 真实缺口、缺陷、依赖变化或同合同实质落后 |

不得从 `OPEN` 直接写“已完成”。旧里程碑的完成状态不继承为 v2 证书。

## 3. 本轮必须产生的最小证据

每个模块只有一份当前工作项，至少包含：

- 起始 commit、owner、范围与非目标；
- capability ledger 与 demand ID；
- public/internal/borrow/higher-layer 的归属裁决；
- 语法消费者、语义重复和依赖边扫描；
- 类型与复杂度表；
- correctness、performance、delivery、usability 和 docs verdict；
- candidate commit、依赖/工具链 manifest、原始证据位置；
- 独立红队报告、剩余限制、结束 commit 与 reopen trigger。

实施页保存本轮事实；architecture、developer、user 只接收仍然有效的沉淀。原始 CSV/JSON/log 放
根级 `evidence/`，且必须由实施页精确引用。没有 candidate/合同/环境绑定的原始文件只是历史档案。

## 4. 需求和能力完整性

### 4.1 两轴分类

每个候选需求先给稳定 ID，并同时记录来源优先级和处置：

| 处置 | 含义 |
|---|---|
| `MUST` | 当前真实消费者或 HYLE 正确运行所必需 |
| `GENERAL` | 成熟库普遍提供、基础用户经常需要，且有清楚的通用形状 |
| `INTERNAL` | 只服务 HYLE 内部，公开会泄露策略或不稳定实现 |
| `BORROW` | 成熟依赖已正确解决，HYLE 只需自然组合或 adapter |
| `DEFER` | 有可能但无当前证据，先不增加实现或常态成本 |
| `REJECT` | 属于消费方、上层模块或明确不在范围 |

推理引擎的 `ENGINE_BLOCKER/COMPETITIVE/GENERALIZATION_GATE/...` 是需求优先级；本表是基础模块的
处置。两者不可互相替代。

### 4.2 Capability ledger

不要按现有类和 benchmark 名称盘点。先按用户任务列能力，例如“有界 arena 批量释放”“获取
单调 deadline”“保存错误上下文”。每项至少记录：

```text
demand_id | consumer task | normal scale/distribution | module owner
disposition | public/internal route | strong peers | failure/lifetime/concurrency
complexity target | verification | negative space
```

基础库的“功能尽量多”不等于每项都自造类型。一个常见需求只要有以下一种自然路径即可算覆盖：

- 模块直接提供；
- 现有原语可以无陷阱地组合；
- 成熟依赖已提供且 HYLE 正确暴露/适配；
- 明确归属于更高层或消费方；
- 有理由地 `DEFER/REJECT`。

成熟库有某能力是重要查漏信号，但不是复制许可。若多个成熟库反复出现同一种通用形状，即使
HYLE 暂无消费者，也必须认真裁决 `GENERAL`，不能简单用“没人用”跳过。

### 4.3 反向追踪

- 每个公开类型/函数必须反向指向至少一个 demand ID；只有 benchmark 使用即删除或私有化。
- 每个 `MUST/GENERAL` 必须正向指向实现、组合路线、依赖或明确裁决。
- 搜索不仅看 include 和符号，还看语义重复：手写 error、allocator、clock、CPU probe、queue、
  cache hint 等都可能是绕过模块。
- 检查上下游两边：本模块是否错误依赖上层；下游是否该用本模块却各自发明。

## 5. 类型与架构审查

每个保留形状都回答：

1. 谁拥有资源，谁借用，move/copy/destroy 后状态是什么？
2. lifetime 能否从类型和命名看出；callback、异步与晚到完成如何保活？
3. 单线程、thread-compatible、thread-safe 或 lock-free 的合同是什么？
4. 容量、OOM、partial progress、取消、重试和程序不变量如何表达？
5. 常态对象大小、常态 metadata、allocation、indirection 和 branch 是多少？
6. 平均/摊还/最坏时间与空间复杂度是什么；进展保证是什么？
7. 一次一个和 batch 两种正常调用是否自然；默认值是否代表多数用户？
8. 可选优化是否显式 opt-in，而不是让所有用户承担税？

优先参考成熟库已被真实用户验证的形状与算法，再根据 HYLE 的统一 error/platform/mem/time 等
底座调整。不能为了让基准更短而暴露 unnatural handle、拆散一次用户操作、隐藏必要语义或增加
benchmark-only metadata；也不能以“不是为了 benchmark”为由拒绝业界常用、能改善复杂度档的
索引、cache 或批处理结构。

公开 C++ API 以任务为中心，采用惯用 RAII、view/owner、range/span、Result、strong type 与
合理 overload；热路径内部可特化。HYLE 自身源码直接使用 C++ API，不绕经产品 FFI。

Native FFI 默认 `N/A`。source/static/shared 是 C++ 交付。只有通过
[FFI 准入策略](../../architecture/native-ffi-policy.md)的真实外语任务才建立窄的运行时类型擦除面，
不要求 C++ parity。

## 6. 实现质量与正确性

### 6.1 算法与实现

- 对每个非平凡操作列当前算法、强 peer、复杂度、常态成本和平台 fast/fallback path。
- 先寻找复杂度档、数据搬运、syscall、allocation、contention 与向量化差距，再做指令级优化。
- 检查最终 release codegen 和链接产物，不以源代码“看起来零成本”代替证据。
- 平台差异通过 platform 能力和编译期/运行期 dispatch 表达，避免模块各自 probe。
- 可恢复失败走 error；程序不变量才 fail-fast。不得在每个模块发明另一套 status/string code。
- 依赖组件既登记在中央 component graph，也由真实 target link 表达，并用机器合同对账。

### 6.2 Correctness matrix

至少覆盖：正常、空/零、边界容量、溢出、OOM/系统失败、错误输入、move/destroy、并发/重入、
取消/关闭、平台 fallback。按风险加入 fault injection、property/differential/fuzz、stress、ASan、
UBSan、TSan 和真实 ISA/backend。测试不得只调用 benchmark 方便面。

公开头必须 self-contained；所选组件必须能独立构建。source、installed static、installed shared
三种 C++ 消费者都验证最小正常 journey 和依赖闭包。平台不适用必须写理由，不能用未运行冒充通过。

## 7. 性能证据

先阅读 [性能证据标准](../performance/benchmark-evidence-standard.md)。顺序固定：

1. 预声明能力合同、正常 workload、强 peer、计费边界、资源指标与 material margin；
2. 冻结 candidate 和输入 manifest；
3. 用 A/A、known-delta positive control、timer floor、order/placement 等证明仪器有资格；
4. 先跑真实消费者/trace replay，再用 microbench 归因；
5. 保存原始数据、codegen、环境和五态 verdict；
6. 在留出 workload 上验证没有为开发集过拟合。

产品改动需要“两把钥匙”：存在可解释的机制/算法/codegen 差异，并且在真实消费者中观察到或
可防御地放大为 material 影响。仅有稳定的 0.5 ns、换顺序输赢或地址布局敏感结果，不足以改变
API、对象布局或常态 metadata。

性能结论必须限定为 `(capability, contract, hardware, delivery, candidate)`。复杂度同档、
non-inferior、某 workload 胜出和全局 SOTA 是不同命题；禁止写没有边界的“模块 SOTA”。

## 8. 用户易用性与文档

在只看用户页和 public headers 的条件下，用一个未参与实现的人完成至少：创建、正常操作、失败、
组合、并发/取消（适用时）和销毁。记录必须阅读源码、容易选错、默认慢、生命周期含糊或错误难
处理的点；这些是产品缺陷，不是“用户水平问题”。

文档分工固定：

- architecture：需求、长期边界、决策及为什么；
- implementation：本轮任务、实验、结果、commit 和历史事实；
- developer：可重复的规则、工具、review、debug 与 troubleshooting；
- user：怎么选、怎么用、ownership/concurrency/failure 和坑；
- `evidence/`：原始机器数据，不承担解释。

文档不得互相复制当前状态或 benchmark 数字。修改后运行 `python scripts/check-docs.py`。

## 9. 独立红队与签署

进入 `CANDIDATE` 后，由未主导实现的人或独立 agent 从反方向审查：

- 找缺失的常见能力和更强 peer；
- 找 benchmark-only shape、隐藏常态税和复杂度退化；
- 找 ownership、OOM、取消、并发和平台失败；
- 找绕过 error/platform/mem/time 等底座的重复实现；
- 找 component graph、真实链接、安装包和文档不一致；
- 尝试使性能仪器在 A/A、order、placement、positive control 上失败。

红队问题必须逐项 `fixed / disproved / accepted limitation / out of scope`。修复导致公开合同、算法、
对象布局或正常 workload 改变时，candidate 失效并回到相应门。

只有所有适用门有 verdict、没有未知 material 问题、限制与 reopen trigger 写明时才能 `SIGNED`。
签署后停止无证据重构。

## 10. 重开条件与回归预算

只有以下事实重开受影响部分：真实消费者暴露缺口或难用形状；correctness/security 缺陷；直接
依赖改变算法/语义；公开边界或交付形状变化；新平台/ISA/backend；强 peer 在同合同正常 workload
证明复杂度更优或可复现 material advantage。LOC、feature 数、命名偏好、新 benchmark、单次跑分
或亚纳秒差异不是重开理由。

每轮结束运行受影响模块测试、直接上下游 smoke、三种交付闭包和文档检查，并记录四个 LOC：

1. 全库代码；
2. 自实现包含测试；
3. 自实现不包含测试；
4. 当前模块实现代码。

使用 `python scripts/loc-report.py --module <name> --self-check`，把结果写入实施任务而不是架构页。
