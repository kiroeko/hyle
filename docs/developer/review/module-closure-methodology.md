# HYLE 模块重构与收口协议 v1

> 本协议是普通模块工程决策的唯一方法。它回答“本轮是否应继续改、能否停止并进入下一模块”，
> 不把发布级性能声明绑到每次重构。当前机械状态见
> [`module-status.json`](../../implementation/module-status.json)，人读投影见
> [`implementation/status`](../../implementation/status.md)。

## 1. 两条互不冒充的状态轴

### 1.1 产品收口轴

产品状态固定为：

```text
QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> VERIFIED -> CLOSED
              \         \          \           \-> BLOCKED
               \---------\----------\---------------------->
CLOSED -> REOPENED -> SCOPED
```

| 状态 | 含义 | 进入时必须已有的输出 |
|---|---|---|
| `QUEUED` | 尚未启动本轮 | 唯一 component ID、优先级说明 |
| `SCOPED` | 决策合同冻结 | baseline、真实需求/缺陷、范围/非目标、P0/P1/P2、预算、机器与仪器 ID、适用门 |
| `AUDITED` | 差距已裁决 | capability/gap ledger；每项为 `KEEP/FIX/BORROW/DEFER/REJECT`，本轮 `FIX` 集冻结 |
| `CANDIDATE` | 产品形状冻结 | product commit/tree、合同与复杂度变化、直接消费者迁移、候选测试 |
| `VERIFIED` | 本轮适用工程门完成 | correctness、delivery、docs/usability、red-team 与分级工程性能 verdict |
| `CLOSED` | 工程上应停止 | 无未决 blocker，限制与 reopen trigger 明确，结束 commit 和四项 LOC 齐全 |
| `BLOCKED` | 已停止但不能签 | 一个可点名的 blocker、owner、恢复所需的新证据；不得用“继续研究”代替 blocker |
| `REOPENED` | 新事实击穿已关闭结论 | trigger ID、复现证据、受影响范围；随后只对受影响部分重新 `SCOPED` |

不得跳过状态。公开合同、算法、对象布局或正常 workload 在 `CANDIDATE` 后改变，退回 `AUDITED`；
只补测试、说明或证据不会改变 product candidate identity。`BLOCKED` 是一次有效裁决，不是无限调查的
许可。

### 1.2 性能声明轴

发布/对外性能声明独立记录为：

```text
NONE | PENDING | CERTIFIED | FAILED | EXPIRED
```

`CLOSED` 不等于 `CERTIFIED`。Formal 证书失败或过期会撤销相应声明；只有它同时暴露真实 material
产品缺口时才重开产品。Formal 资格遵守
[性能证据标准](../performance/benchmark-evidence-standard.md)，不得把该标准的发布级归档、三平台
矩阵或 SOTA/non-inferiority 证书重新变成普通模块的默认收口门。

## 2. 启动输入：一张冻结的 closure card

写代码前，当前 implementation task 必须写清：

- component ID、owner、round ID、baseline commit；
- 一个真实 consumer task、可复现 defect，或成熟实现反复证明的通用基础需求；
- 范围、非目标、直接依赖和直接消费者；
- `P0/P1/P2` 及理由，正常 workload、资源预算和 material margin；
- 总调查预算、性能预算、最多实验/修复次数；
- 中央 `machine_profile`、`instrument_profile` 及适用 correctness/delivery gates；
- 最多两个预先选择的 strong peers；
- 预期输出、停止条件和可接受限制。

没有真实输入时保持 `QUEUED`。没有 P1/P2 workload、阈值、机器或仪器 ID 时不得开始计时。模块任务
不得一边看结果一边修改中央统计规则、机器配置或 material margin；公共仪器缺口另开 project/perf
任务。

## 3. 有界审计及其输出

审计先按用户任务而不是现有 API 或 benchmark 名称列 capability。每项至少记录：

```text
demand/defect ID | consumer/来源 | normal shape | owner module
ownership/lifetime/concurrency/failure | complexity/resource target
KEEP/FIX/BORROW/DEFER/REJECT | severity | verification | negative space
```

- 每个公开入口反向指向一个 demand；只被 benchmark 使用的入口删除或私有化。
- 每个当前 `MUST` 有实现、自然组合、成熟依赖或明确裁决。
- `GENERAL` 是查漏输入，不自动扩大本轮；没有真实阻断的新增能力默认进入后续 backlog。
- 同时扫描语法消费者、语义重复和直接依赖边，但只修本模块与必要直接边。
- 优先检查错误复杂度档、额外 allocation/copy/syscall/lock、错误 ownership 和交付断边，再看常数。

审计结束必须冻结有限 `FIX` 集。发现相邻模块问题时登记任务；只有 correctness/security 或当前构建
阻断允许越界修复。

## 4. 类型、实现与 correctness 门

每个保留形状回答 ownership、borrow invalidation、move/destroy、并发/进展、容量/OOM、取消/关闭、
正常 metadata、时间/空间复杂度和平台 fallback。公开 C++ API 以任务为中心，优先 RAII、清楚的
owner/view、span/range、strong type 与 `Status/Result`。可恢复失败走 error；程序不变量才
fail-fast。OS/CPU 机制走 platform，allocator/placement 走 mem，clock/deadline 走 time。

Correctness 证据按风险从以下矩阵选择，并在 closure card 中预声明适用项：正常、空/零、容量边界、
溢出、OOM/系统失败、错误输入、move/destroy、并发/重入、取消/关闭、property/differential/fuzz、
stress、ASan/UBSan/TSan、真实 ISA/backend。未运行只能写 `NOT_APPLICABLE` 或 `BLOCKED`，不能写
`PASS`。

公开头必须 self-contained；组件能独立构建。source、installed static、installed shared 的公共闭包由
中央 component/package 合同验证；header-only 或不适用形状只运行有实际产物的 journey，不为凑数
重复同一事实。

## 5. 分级工程性能门

工程性能是产品收口证据；Formal 是独立声明。每轮在 `SCOPED` 时冻结一级，不能因结果不好而改级。

| 级别 | 适用 | `CLOSED` 的工程性能硬门 |
|---|---|---|
| `P0` | 未被 profile 证明为热点的冷路径、编译/布局 primitive | 复杂度同档；无意外 allocation/lock/syscall/metadata；适用的 release codegen/size 结构检查 `PASS`。peer 计时可为 N/A |
| `P1` | 可能高频的 runtime primitive、容器、allocator adapter、queue/timer 操作 | 在 `closure-perf-v1` 与锁定性能机上，对正常 consumer/replay 和同合同 peer 通过预声明工程 non-inferiority 门；默认 material margin 为 **5%** |
| `P2` | profile 已证明影响端到端 KPI 的热点 | 在真实 consumer/trace 上通过按贡献预算声明的 latency/throughput/resource 门，并有机制归因；小于 5% 的阈值必须由端到端预算证明 |

P1/P2 的工程 verdict 固定为 `PASS/FAIL/INCONCLUSIVE/NOT_COMPARABLE`。`CLOSED` 必须为 `PASS`；
`INCONCLUSIVE` 或仪器无资格时进入 `BLOCKED` 或重新降级为没有热点证据的、重新 `SCOPED` 的 P0，
不得原地把未知写成平局。降级必须基于 consumer/profile 事实，不能因为跑分不利。

产品改动仍需两把钥匙：可复核的复杂度/指令/allocation/copy/syscall/contention/codegen 机制，以及
可传播到正常 consumer 的 material 影响。只有亚纳秒 microbench、顺序/地址翻转或总体 profile 不可见
的差异不允许改变 API、layout 或常态 metadata。

## 6. 固定预算与硬停止

- scope card 预声明总调查预算和性能预算；超过预算只能 `CLOSED`、`BLOCKED` 或由 owner 显式重新
  scope，不能默默继续。
- 普通约千行及以下的 P0/P1 模块，默认审查与取证预算不超过两小时；计划实现超过一个工作日时先
  拆成可独立验证的 defect/demand。
- P1 异常最多允许：一次复现/反序、一次 codegen/profile 机制检查、一个产品修改与一次重跑。
- 同一轮不得发明第五种 benchmark 姿势。中央仪器失败时记录 instrument issue；模块按硬门进入
  `BLOCKED`，不在模块任务内扩建统计平台。
- 红队 P0/P1 默认 30 分钟，只攻击已冻结的能力、合同、correctness、依赖和工程性能门。新发现的
  通用仪器增强进入下一版工具；除非已证明当前 material verdict 无效，不追溯移动本轮终点。

稳定 material `FAIL` 在一次机制诊断后仍未解决时，记录具体性能缺口并 `BLOCKED`。这比八小时后仍
写“继续调查”更完整，也不会阻止不依赖该结论的其他模块准备工作。

## 7. Candidate、验证与独立红队

`CANDIDATE` 同时冻结 product commit/tree、公开合同、算法、normal workload 和 tier；harness/tool
identity 单独记录。验证 manifest 至少给出：

- correctness 与 sanitizer/平台 verdict；
- 直接消费者 smoke 和中央 component/package verdict；
- P0/P1/P2 工程性能 verdict、机器/仪器 ID 和证据；
- user journey 与文档检查；
- 独立红队问题的 `fixed/disproved/accepted/out-of-scope`；
- 已知限制、结束 commit 与 reopen triggers。

Blocker 是 correctness/security、合同/ownership 错误、构建/交付断裂，或本 tier 工程性能未通过。
非 material 命名、风格、feature 愿望和 Formal 待办不是 blocker。

## 8. 文档与机械状态

- architecture：长期需求、边界和决策原因；
- implementation task：本轮输入、审计、变更、证据、限制和历史；
- developer：可复用方法、机器、工具与排障；
- user：选型、用法、ownership/concurrency/failure 与坑；
- 根级 `evidence/`：原始机器输出，不承担解释。

只有 `docs/implementation/module-status.json` 保存当前产品/性能状态；`status.md` 是它的受检人读投影。
其他页面不得手写当前状态，历史状态必须明确标为 `HISTORICAL`。运行：

```text
python scripts/check-module-closure.py
python scripts/check-docs.py
```

## 9. 终止输出与四项 LOC

只有所有适用工程门有 verdict、P1/P2 工程性能为 `PASS`、没有 blocker，且限制/reopen trigger 写明时
才能 `CLOSED`。每轮在 `CLOSED` 或 `BLOCKED` 终止时仍必须运行：

```text
python scripts/loc-report.py --module <component> --self-check
```

并绑定同一快照记录四项 `code`：

1. 全库代码；
2. 自实现包含测试；
3. 自实现不包含测试；
4. 当前模块实现代码。

LOC 只用于规模与过程复盘，不是完成度、优先级或重开依据。

## 10. 重开条件

只有以下新事实重开受影响部分：

- 可复现 correctness/security、OOM、取消、并发生命周期或交付缺陷；
- 真实消费者暴露缺失能力、错误 ownership、难用接口或重复机制；
- 锁定同合同正常 workload 出现 material regression，或 profile 将模块提升为 P2；
- 直接依赖、compiler、平台、ISA/backend 改变实际语义、算法或 codegen；
- 公开 API、组件依赖或交付形状改变；
- 强 peer 证明更好的复杂度档或可复现 material consumer advantage。

新 benchmark、新 peer 发布但没有 material 证据、LOC/API 数、命名偏好、单次点值、地址翻转、
亚纳秒差异或 Formal 证书单独过期都不重开产品。
