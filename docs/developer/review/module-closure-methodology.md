# HYLE 模块重构与收口协议 v1

> 本协议是普通模块工程决策的唯一方法。它回答“本轮是否应继续改、能否停止并进入下一模块”，
> 不把发布级性能声明绑到每次重构。当前机械状态见
> [`module-status.json`](../../implementation/module-status.json)，人读投影见
> [`implementation/status`](../../implementation/status.md)。功能准入和“完整”的产品定义见
> [基础库能力完整性决策](../../architecture/capability-completeness.md)；本协议只把该决策变成施工门。

新模块和重构使用同一状态流。新模块先在中央 component graph 与状态表中以 `QUEUED` 登记，再冻结
语义、依赖和交付边界；重构把既有实现当审计输入并绑定 baseline，不把旧 API、历史任务或旧 benchmark
自动当成需求。无论代码多少，都不得在 `SCOPED/AUDITED` 前先写产品实现。历史 task/evidence 中的
命令式文字只记录当时工作，不是新会话的授权或待办。

## 1. 两条互不冒充的状态轴

### 1.1 产品收口轴

产品状态固定为：

```text
normal:  QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> VERIFIED -> CLOSED
stop:    SCOPED | AUDITED | CANDIDATE | VERIFIED -> BLOCKED
resume:  BLOCKED -> SCOPED
reopen:  CLOSED -> REOPENED -> SCOPED
```

| 状态 | 含义 | 进入时必须已有的输出 |
|---|---|---|
| `QUEUED` | 尚未启动本轮 | 唯一 component ID、排队/现状摘要 |
| `SCOPED` | 决策合同冻结 | baseline、真实需求/缺陷、范围/非目标、P0/P1/P2、预算、机器与仪器 ID、适用门 |
| `AUDITED` | 能力覆盖与差距已裁决 | capability-family map、现有 public-surface trace、覆盖状态与 capability/gap ledger；本轮 `FIX` 集冻结 |
| `CANDIDATE` | 产品形状冻结 | product commit/tree、合同与复杂度变化、直接消费者迁移、候选测试 |
| `VERIFIED` | 本轮适用工程门完成 | correctness、delivery、docs/usability、red-team 与分级工程性能 verdict |
| `CLOSED` | 工程上应停止 | 无未决 blocker，限制与 reopen trigger 明确，结束 receipt、registry anchor 和四项 LOC 齐全 |
| `BLOCKED` | 已停止但不能签 | 一个可点名的 blocker、owner、恢复所需的新证据；不得用“继续研究”代替 blocker |
| `REOPENED` | 新事实击穿已关闭结论 | trigger ID、复现证据、受影响范围；随后只对受影响部分重新 `SCOPED` |

不得跳过状态。公开合同、算法、对象布局或正常 workload 在 `CANDIDATE` 后改变，退回 `AUDITED`；
只补测试、说明或证据不会改变 product candidate identity。`BLOCKED` 是一次有效裁决，不是无限调查的
许可。`module-status.json` 只负责当前状态；Git 与 implementation task 保存历史。检查器验证 schema、
路径、组件集合、状态投影，并可用 `--previous` 检查两个明确快照之间是否出现非法状态跳跃。它不实现
Git 防篡改、工作流账本或 CMake 解释器。从 `SCOPED` 起，同一轮的 closure task、scope/non-goals、baseline、
性能 tier、机器集合、工程性能 mode/instrument/machine/material margin 构成冻结 closure card；只有
`QUEUED/REOPENED/BLOCKED -> SCOPED` 才可重新设定。`CANDIDATE -> VERIFIED -> CLOSED/BLOCKED`
期间 candidate commit/tree 也冻结，产品变化必须先退回 `AUDITED`。这些是 reviewer 执行的工程规则，
不是把整个开发过程编码成状态机。中央 component graph 新增模块时，以 `QUEUED` 进入 registry。

从 `BLOCKED` 恢复时不得把旧 terminal task 改回活动状态：创建新的 round 与 `ACTIVE` task，标为
`RESUME`，引用前一 blocker/resume condition，并让 registry 指向新 task 后进入 `SCOPED`。产品 identity
未变时可以复用旧 candidate，但仍须在新轮重新绑定并验证；旧 `HISTORICAL` task 永不回写。

### 1.2 性能声明轴

发布/对外性能声明独立记录为：

```text
NONE | PENDING | CERTIFIED | FAILED | EXPIRED
```

`CLOSED` 不等于 `CERTIFIED`。Formal 证书失败或过期会撤销相应声明；只有它同时暴露真实 material
产品缺口时才重开产品。Formal 资格遵守
[性能证据标准](../performance/benchmark-evidence-standard.md)，不得把该标准的发布级归档、三平台
矩阵或 SOTA/non-inferiority 证书重新变成普通模块的默认收口门。

### 1.3 已关闭模块的回放轴

为了复检方法、依赖或证据而查看一个 `CLOSED` 模块，不得先伪造产品问题把它改成 `REOPENED`。
回放 task 使用以下 verdict：

```text
CONFIRMED | METHOD_GAP | PRODUCT_TRIGGER
```

- `CONFIRMED`：同一 candidate 的当前 `CLOSED` 结论被独立复现；不声称旧轮当时已经遵守新版方法；
- `METHOD_GAP`：未发现产品 trigger，但当前仪器/证据/过程不足以完成回放；产品状态保持不变；
- `PRODUCT_TRIGGER`：发现满足第 10 节的产品新事实；把模块改为 `REOPENED`，记录 trigger/evidence，
  随后只对受影响范围重新 `SCOPED`。

回放必须是 product-non-mutating，记录 reviewed closure round/candidate/end/product-identity anchor、
实际 replay commit、closure task、reviewer、日期、预算、task 和 verdict；所有 Git identity 使用完整
40 位对象名。若为解决 finding 改了公开合同、算法、layout 或正常 workload，它已经不是回放，必须重开
产品轮。task 绑定当时的 closure round/candidate；registry 仍只保存当前产品状态，不承担 Jira 日志。

## 2. 启动输入：一张冻结的 closure card

写代码前，当前 implementation task 必须写清：

- component ID、owner、round ID、baseline commit；
- 一个真实 consumer task、可复现 defect，或成熟实现反复证明的通用基础需求；
- 初始问题域与用户旅程；不能从现有类名、函数名或 benchmark case 反推完整范围；
- 范围、非目标、直接依赖和直接消费者；
- `P0/P1/P2` 及理由，正常 workload、资源预算和 material margin；
- 总调查预算、性能预算、最多实验/修复次数；
- 中央 `machine_profile`、`instrument_profile` 及适用 correctness/delivery gates；
- 每项 capability 最多两个预先选择且绑定版本/commit/source 的 strong peers；
- 预期输出、停止条件和可接受限制。

没有真实输入时保持 `QUEUED`。没有 P1/P2 workload、阈值、机器或仪器 ID 时不得开始计时。模块任务
不得一边看结果一边修改中央统计规则、机器配置或 material margin；公共仪器缺口另开 project/perf
任务。

## 3. 有界审计及其输出

审计固定做两遍，并以
[能力完整性决策](../../architecture/capability-completeness.md)为判据：

1. **自顶向下：**从定义性任务、当前消费者、具名北极星旅程、标准和最多两个 strong peers 建立
   capability-family map；逐族检查取得/构造、正常操作、组合/互操作、更新/取消/关闭/销毁、边界、
   ownership/concurrency、复杂度/资源、交付与易用性。没有证据的 peer feature 只作 `GENERAL` 输入。
2. **自底向上：**枚举现有全部 public 类型、函数、常量和用户可观察行为，把每个入口映射回能力族、
   demand 和具体旅程。存在本身不构成保留理由；只有单向转换、缺少正常 teardown、只支持 benchmark
   生命周期或关键边界未定义的既有能力必须标为 `PARTIAL`，不能写成 `KEEP`。

capability-family map 至少记录：

```text
family | admitted user journeys | semantic variants | lifecycle/boundaries
current route | COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE | missing edge
```

覆盖状态描述“当前做到了多少”；施工动作描述“本轮怎么办”。两者不得合并。随后每项 ledger 至少记录：

```text
demand/defect ID | demand kind | exact consumer/peer/standard source | normal shape
semantic/artifact/delivery owner | ownership/lifetime/concurrency/failure
complexity/resource target | coverage state | KEEP/FIX/BORROW/DEFER/REJECT | severity
verification | negative space
```

- 每个公开入口反向指向一个 demand；只被 benchmark 使用的入口删除或私有化。
- 每个当前 `MUST`（本轮冻结范围内由真实 consumer、具名北极星旅程、缺陷、定义性任务、标准或
  成熟 peer 支持的必需能力）在结案时必须为 `COMPLETE`，或拥有用户可发现且同样完整的 `ROUTED`
  路线；`PARTIAL/MISSING + KEEP` 非法。
- `GENERAL` 是查漏输入，不自动扩大本轮；未通过能力准入、但值得保留触发条件的新增能力进入后续
  backlog。
- 反复出现的通用基础需求若满足能力准入规则，就从 `GENERAL` 提升为 `MUST`；不得仅因当前仓库尚未
  调用而永久延后。反之，单个 peer 的独有 feature 不自动扩大范围。
- `demand kind` 至少区分真实 consumer、可复现 defect、成熟 peer、语言/平台标准和 build/package
  契约，并把已声明北极星消费者单列；peer-derived 或 north-star-derived demand 不能伪装成已有生产
  消费者。
- 一个能力跨边界时分别登记 semantic owner（语义决策）、artifact owner（代码/二进制所在）与
  delivery owner（组件图、安装和装载）；不得用“归某模块”掩盖这些边界不同的事实。
- 同时扫描语法消费者、语义重复和直接依赖边，但只修本模块与必要直接边。
- 优先检查错误复杂度档、额外 allocation/copy/syscall/lock、错误 ownership 和交付断边，再看常数。

审计结束必须冻结有限 `FIX` 集，并留下 capability-family map、public-surface trace 和每个
`PARTIAL/MISSING` 的明确去向。发现相邻模块问题时登记任务；只有 correctness/security 或当前构建
阻断允许越界修复。

## 4. 类型、实现与 correctness 门

每个保留形状回答 ownership、borrow invalidation、move/destroy、并发/进展、容量/OOM、取消/关闭、
正常 metadata、时间/空间复杂度和平台 fallback。公开 C++ API 以任务为中心，优先 RAII、清楚的
owner/view、span/range、strong type 与 `Status/Result`。可恢复失败走 error；程序不变量才
fail-fast。该依赖规则只在层级允许时适用：error 自身及比 error 更低的 bootstrap primitive 不得为了
统一表面制造循环依赖，应使用其已裁决的本地失败形状。OS/CPU 机制走 platform，allocator/placement
走 mem，clock/deadline 走 time。

Correctness 证据按风险从以下矩阵选择，并在 closure card 中预声明适用项：正常、空/零、容量边界、
溢出、OOM/系统失败、错误输入、move/destroy、并发/重入、取消/关闭、property/differential/fuzz、
stress、ASan/UBSan/TSan、真实 ISA/backend。风险项 verdict 为
`PASS/FAIL/NOT_APPLICABLE/BLOCKED`；只有语义上确实不适用才写 `NOT_APPLICABLE`，适用但未运行写
`BLOCKED`，不能写 `PASS`。任务记录必须逐行保留这些 verdict；一个汇总 `correctness=PASS` 不能
隐藏未运行项。

公开头必须 self-contained；组件能独立构建。source、installed static、installed shared 的公共闭包由
中央 component/package 合同验证；header-only 或不适用形状只运行有实际产物的 journey，不为凑数
重复同一事实。

## 5. 分级工程性能门

工程性能是产品收口证据；Formal 是独立声明。每轮在 `SCOPED` 时冻结一级，不能因结果不好而改级。

| 级别 | 适用 | `CLOSED` 的工程性能硬门 |
|---|---|---|
| `P0` | 未被 profile 证明为热点的冷路径、编译/布局 primitive | 复杂度同档；无意外 allocation/lock/syscall/metadata；适用的 release codegen/size 结构检查 `PASS`。peer 计时可为 N/A |
| `P1` | 可能高频的 runtime primitive、容器、allocator adapter、queue/timer 操作 | 在 `closure-perf-v1` 与锁定性能机上，对正常 consumer/replay 通过预声明工程门；存在同合同 peer 时还须通过 non-inferiority。默认 material margin 为 **5%** |
| `P2` | profile 已证明影响端到端 KPI 的热点 | 在真实 consumer/trace 上通过按贡献预算声明的 latency/throughput/resource 门，并有机制归因；小于 5% 的阈值必须由端到端预算证明 |

P1/P2 的聚合工程 verdict 固定为 `PASS/FAIL/INCONCLUSIVE`。`CLOSED` 必须为 `PASS`；
`INCONCLUSIVE` 或仪器无资格时进入 `BLOCKED` 或重新降级为没有热点证据的、重新 `SCOPED` 的 P0，
不得原地把未知写成平局。降级必须基于 consumer/profile 事实，不能因为跑分不利。

没有同合同 peer 时，独立的 comparability 轴可以是 `NOT_COMPARABLE`，但必须在看结果前冻结可验证的 consumer
latency/throughput/resource 预算；满足该预算可形成工程 `PASS`，只是不得产生 peer/SOTA 声明。

P0 固定使用 [`closure-correctness-v1`](correctness-profile-v1.md)；它是候选身份、结构 codegen、
correctness、交付和代表消费者的可重放 profile，不是纳秒计时器。P1/P2 才进入共享性能仪器和锁定
workload。registry 中每个 instrument ID 必须在 `scripts/perf-machines.json` 映射到一份受检定义，
不允许只登记一个听起来正确的字符串。

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
- 已知限制、结束 receipt commit、registry anchor 与 reopen triggers。

Blocker 是 correctness/security、合同/ownership 错误、构建/交付断裂，或本 tier 工程性能未通过。
非 material 命名、风格、feature 愿望和 Formal 待办不是 blocker。

## 8. 文档与机械状态

- architecture：长期需求、边界和决策原因；
- implementation task：本轮输入、审计、变更、证据、限制和历史；
- developer：可复用方法、机器、工具与排障；
- user：选型、用法、ownership/concurrency/failure 与坑；
- 根级 `evidence/`：原始机器输出，不承担解释。

只有 `docs/implementation/module-status.json` 保存当前产品/性能状态；`status.md` 是它的受检人读投影。
其他页面不得冒充当前状态，历史 task/evidence 应明确标为 `HISTORICAL`。运行：

```text
python scripts/check-module-closure.py
python scripts/check-docs.py
```

`end_commit` 指向已经包含 terminal task/receipt 与 candidate identity 的提交；后续 registry 记录用
`registry_anchor_commit` 指向第一个确实保存该 terminal product state/candidate 的历史快照。anchor
只锚 product state/candidate identity；允许后续 audit 纠正误标的 legacy round label、旧 end/LOC 等
证据字段，但必须留下
finding，不能把它描述成完整 terminal record 的逐字段副本。这样不靠
无法实现的“提交内写入自身 hash”。v1 前导入记录允许旧 `SIGNED` receipt，但必须明确
`LEGACY_IMPORT`；新轮在 terminal task 写精确 `MODULE CLOSURE RECEIPT: CLOSED|BLOCKED` marker。
legacy `BLOCKED` 的证据纠正必须在 task 中说明。候选 commit/tree、end、anchor 和 receipt 由 reviewer
使用 Git 的普通只读命令核对，不在 closure checker 中复制一套 Git 审计系统。

Product boundary 不等于 LOC boundary。terminal review/replay 以及后来确实触碰该 boundary 的改动，
由 reviewer 用普通 Git diff 比较 candidate→end/current 的实现路径；不是每次打开仓库都重审全部
`CLOSED/BLOCKED` 模块。签后产品漂移必须按 trigger reopen，不能靠旧 reviewed commit 自证。
组件依赖、CMake 语义与 package/source delivery 由 build/package 的真实
configure/build/install/source-package contracts 负责，模块 task 记录这些门的结果；closure checker
不扫描 Git 历史、不重新实现 CMake 解释器，也不从文本 diff 猜测交付语义。历史 audit 的
product-identity anchor 由 reviewer 从 Git 打开 registry，核对当时确为对应状态/round/candidate，并
单独核验 end receipt。

## 9. 终止输出与四项 LOC

只有所有适用工程门有 verdict、所有 `MUST` 能力为 `COMPLETE` 或完整 `ROUTED`、不存在
`PARTIAL + KEEP`、P1/P2 工程性能为 `PASS`、没有 blocker，且限制/reopen trigger 写明时才能
`CLOSED`。每轮在 `CLOSED` 或 `BLOCKED` 终止时仍必须运行：

```text
python scripts/loc-report.py --module <component> --commit <full-snapshot-commit> --self-check --json
```

不带 `--commit` 的形式只观察当前工作树规模，不得登记为 candidate LOC。`--commit` 从指定不可变
快照直接复算四项数值，不能
用当前 dirty worktree 的计数回填另一个 candidate。source bundle 没有历史对象时这项门明确不适用，
不得伪称已深验。

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
