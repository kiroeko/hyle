# HYLE 模块重构与收口协议 v2

> 本协议是普通模块工程决策的唯一方法。它回答“本轮是否应继续改、能否停止并进入下一模块”，
> 不把发布级性能声明绑到每次重构。当前机械状态见
> [`module-status.json`](../../implementation/module-status.json)，人读投影见
> [`implementation/status`](../../implementation/status.md)。功能准入和“完整”的产品定义见
> [基础库能力完整性决策](../../architecture/capability-completeness.md)；本协议只把该决策变成施工门。

新模块和重构使用同一状态流。新模块先在中央 component graph 与状态表中以 `QUEUED` 登记，再冻结
语义、依赖和交付边界；重构把既有实现当审计输入并绑定 baseline，不把旧 API、历史任务或旧 benchmark
自动当成需求。无论代码多少，都不得在 `SCOPED/AUDITED` 前先写产品实现。历史 task/evidence 中的
命令式文字只记录当时工作，不是新会话的授权或待办。

若要审的不是一个已确定边界的组件，而是全库功能归属和组件图本身，先执行
[能力展开与组件质因数分解](../../architecture/capability-factorization.md)：完整统计所有产品问题域的
功能超集和自然形状以后，才确定 owner、组件和依赖。这个只读项目阶段不沿现有拓扑逐模块施工，也不
改变模块状态；新因子图成立后，才对拓扑序首项使用本文状态流。

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
| `VERIFIED` | 本轮适用工程门完成 | correctness、delivery、docs/usability、内部复用/加速适用性、red-team 与分级工程性能 verdict |
| `CLOSED` | 工程上应停止 | 无未决 blocker，限制与 reopen trigger、结束 receipt 和三项 LOC 齐全 |
| `BLOCKED` | 已停止但不能签 | 一个可点名的 blocker、owner、恢复所需的新证据；不得用“继续研究”代替 blocker |
| `REOPENED` | 新事实击穿已关闭结论 | trigger ID、复现证据、受影响范围；随后只对受影响部分重新 `SCOPED` |

不得跳过状态。公开合同、算法、对象布局或正常 workload 在 `CANDIDATE` 后改变，退回 `AUDITED`；
只补测试、说明或证据不会改变 product candidate identity。`BLOCKED` 是一次有效裁决，不是无限调查的
许可。`module-status.json` 只负责当前状态；Git 与 implementation task 保存历史。检查器验证 schema、
路径、组件集合、状态投影，并可用 `--previous-ref` 从 Git 读取改动前 registry，检查两版之间是否
出现非法状态跳跃。它不实现
Git 防篡改、工作流账本或 CMake 解释器。从 `SCOPED` 起，同一轮的 closure task、scope/non-goals、baseline、
性能 tier、机器集合、工程性能 mode/instrument/machine/material margin 构成冻结 closure card；只有
`QUEUED/REOPENED/BLOCKED -> SCOPED` 才可重新设定。`CANDIDATE -> VERIFIED -> CLOSED/BLOCKED`
期间 candidate commit/tree 也冻结，产品变化必须先退回 `AUDITED`。这些是 reviewer 执行的工程规则，
不是把整个开发过程编码成状态机。中央 component graph 新增模块时，以 `QUEUED` 进入 registry。

预发布阶段若 owner 明确纠正一个错误的内部 component 名称，不得靠删除旧记录、伪造新 `QUEUED` 记录或
跳过 `--previous-ref` 实现。当前 registry 必须保存一对一 `component_identity_migrations` 及 repository 内
证据；新 ID 在原 ledger 位置继续旧 ID 的合法状态迁移，中央 manifest、target、直接边和当前文档原子
迁移，不保留未经需求证明的 compatibility alias。检查器仍拒绝未登记删除/重命名、链式/循环迁移和借
迁移重排组件。历史 task 保留当时名称并加迁移注记，而不是改写已经发生的事实。

从 `BLOCKED` 恢复时不得把旧 terminal task 改回活动状态：创建新的 round 与 `ACTIVE` task，标为
`RESUME`，引用前一 blocker/resume condition，并让 registry 指向新 task 后进入 `SCOPED`。产品 identity
未变时可以复用旧 candidate，但仍须在新轮重新绑定并验证；旧 `HISTORICAL` task 永不回写。

#### `FROZEN` confirmation 不是第二套状态机

`CLOSED` 只证明 registry 指向的那一轮按其冻结范围结束。项目进行过方法升级后，只有完成当前能力
完整性、真实消费者、内部复用、性能比较、C++ 易用性、文档、构建交付和红队回放的 `CLOSED` 模块，
才可在 registry 附加 `freeze_confirmation.marker=FROZEN`。该 confirmation 必须绑定唯一 ID、日期、
完整 reviewed commit/tree 和 repository 内证据文件；状态表的 `Freeze` 列只是这些字段的机器投影。

`FROZEN` 不表示永久 ABI、Formal SOTA 或未来永远没有新能力，也不替代 `CLOSED`。没有 marker 的旧
`CLOSED` 仍是合法历史收口，只是尚未取得这一轮方法的复核确认；文档不得靠 summary 中写
“frozen”冒充 marker。模块出现已登记 reopen trigger 后必须先移除 confirmation，并按
`CLOSED -> REOPENED -> SCOPED` 开新轮；新轮重新关闭并完成所需回放后才能签发新的 confirmation。
API/usability freeze、candidate freeze 和这里的产品复核 confirmation 是三个不同时间点，不能混称。

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
- 总调查预算、性能预算，以及 P1/P2 的有效采样与机制诊断上限；
- 中央 `machine_profile`、`instrument_profile` 及适用 correctness/delivery gates；
- 每项 capability 最多两个预先选择且绑定版本/commit/source 的 strong peers；
- 预期输出、停止条件和可接受限制。

没有真实输入时保持 `QUEUED`。没有 P1/P2 workload、阈值、机器或仪器 ID 时不得开始计时。模块任务
不得一边看结果一边修改中央统计规则、机器配置或 material margin；公共仪器缺口另开 project/perf
任务。

模板是问题清单，不是篇幅合同。task 只保留本轮适用的表格行和风险项；语义上不适用的整节可用一句
具体理由裁决后删除空表。七问、全部 `MUST`、适用风险和最终 verdict 不能省略，但不得为了填满模板
制造虚构 peer、binary journey、性能样本、消费者或重复粘贴原始日志。小型 P0 因子的记录应随风险
缩小，而不是随模板长度膨胀。

### 2.1 从用户任务到 API 的固定设计顺序

基础库没有当前仓库消费者时，不能从已有类名、peer symbol 或 benchmark case 反推产品。先建立一个
可追溯的**使用原型**，证据优先级依次是：当前真实消费者、北极星消费者中的具体操作序列、至少两个
独立成熟实现共同提供的稳定任务、语言/平台标准或定义性算法，最后才是明确标为 diagnostic 的 synthetic
shape。没有真实消费者、但由后两类证据准入的形状必须标为 `PROVISIONAL_ARCHETYPE`；它能驱动基础
能力设计和 P1 replay，不能冒充已经存在的生产 trace。

一个能力族按以下顺序设计，不能先写实现再从跑分解释 API：

1. 用对象、操作序列和完成条件描述用户任务；
2. 枚举真正不同的语义档位，而不是先命名“fast/light/heavy”入口；
3. 为每档写清 result 粒度、ownership/lifetime、failure/OOM、cancel/close、blocking/suspension、
   concurrency/progress 和 teardown；
4. 写常态成本合同：per-item/per-batch metadata、allocation ownership、queue admission、copy、atomic、
   lock、wake/context-switch、syscall 和工作集的目标档位；
5. 用最多两个 strong peers 验证该任务和语义档位确实成熟、常用，并记录 peer 选择了什么取舍；
6. 给出普通 C++ 调用草图、适用前提、误用路径和相邻档位的选择规则；
7. 冻结有限 capability/`FIX` ledger 后才实现 correctness；
8. 性能先做同语义产品比较，再做自然路线竞争，最后用机制实验归因；
9. 只有机制差异与 normal task 的 material 影响同时成立，才继续改变产品；
10. 用一个 normal primary shape 和至多一个不同粒度/生命周期的 holdout 防止过拟合，不建立无界矩阵。

其中第 6～8 步之间有一个不可省略的 **API / usability freeze**。在同一轮全部 `MUST` 能力都已经有
普通 C++ 调用草图、选型规则、ownership/lifetime/failure/concurrency 合同和常态成本合同以前，不得
开始 P1/P2 计时，也不得按已有 benchmark 逐项补入口。冻结时同时锁定 public 名称与类型关系、默认值、
错误形状、依赖暴露、source/static/shared 发现方式和用户指南中的正常 recipe；compile-only journey 与
误用诊断是冻结证据的一部分。随后先在这套固定接口后完成 correctness、边界和直接消费者迁移，再形成
candidate，最后才运行性能门。

接口冻结后，性能工作默认只能改变内部算法、provider 路由、布局中不属于公开合同的部分和构建时选择，
不能因为某个 benchmark 点值重新命名类型、拆出 `fast/light` 入口或给所有对象增加常态 metadata。只有
同时证明“接口合同本身造成可复现的机制差异”以及“该差异在 normal consumer 中达到 material 影响”，
才允许退回 `AUDITED` 重新裁决接口；该退回必须更新有限 `FIX` 集和 usability 证据，不能在性能循环中
静默漂移 public shape。

两个入口若只在预期“快慢”上不同，却没有 material 的 result、ownership、failure、cancel、progress 或
lifecycle 差异，应合并为同一入口并由实现选择最佳机制。反之，要求逐任务结果/控制与只要求批级完成，
或可挂起任务与 worker-stack 非挂起任务，会产生不同的语义与资源账单，可以形成独立、按任务命名的
public shape。用户指南必须直接说明选择前提，不能要求用户理解内部 benchmark 名称。

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
current route | existing-facility/acceleration route
COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE | missing edge
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
- 同时扫描语法消费者、语义重复、直接依赖边，以及当前实现是否绕开了已有 HYLE/标准库/compiler/OS/
  已锁定成熟依赖的同合同机制。每个本地机制必须裁决为 `REUSE/BORROW/LOCAL/NOT_APPLICABLE`；选择
  `LOCAL` 必须留下 ownership、语义、复杂度、常态成本、依赖方向或交付方面的具体理由，不能以“更快”
  或“更简单”代替证据。
- 对 SIMD、allocator/pool、Container、reclamation、CPU/topology/NUMA、clock、wait/synchronization、
  hash/text codec 等加速设施逐项检查适用性。存在某个设施不等于必须引用：不连续的 pointer chasing、
  冷路径、用户定义 callback、依赖倒置/成环或更重常态 metadata 都可以形成
  `NOT_APPLICABLE/LOCAL`，但必须写清原因。反之，同合同且依赖方向合法的已有设施不得重复实现。
- 复用审计只修本模块与必要直接边；发现 owner/因子图错误时按相邻任务或全局 factorization 流程处理，
  不在模块轮内顺手重排全图。
- 优先检查错误复杂度档、额外 allocation/copy/syscall/lock、错误 ownership 和交付断边，再看常数。

审计结束必须冻结有限 `FIX` 集，并留下 capability-family map、public-surface trace 和每个
`PARTIAL/MISSING` 的明确去向。发现相邻模块问题时登记任务；只有 correctness/security 或当前构建
阻断允许越界修复。

### 3.1 Owner 逐模块七问

每个模块在 `AUDITED` 和终止时都必须给 owner 留下一份直答，不得只给聚合绿色状态：

1. **第三方：**产品路径是否直接或经组件闭包间接使用第三方实现？逐项列出锁定身份、直接/间接、
   `product/test/benchmark/peer` 类别，以及 semantic、mechanism/artifact、decision 和 delivery owner。
   标准库、compiler builtin 与 OS API 不冒充第三方模块，但仍记录其平台合同。vendored、ported 或
   integrated source 仍算第三方实现，不能因复制进仓库而记作 HYLE 自实现。
2. **功能：**对于已声明的量化推理引擎旅程，必须使用四组当前锁定证据作差异输入：**独立 ggml
   底座能力及 llama.cpp 真实消费形状**、**ds4**、**vLLM**、**SGLang**。每轮先更新或确认四组本地
   reference 的远端状态，再记录 clean commit；llama.cpp 随仓 ggml 与独立 ggml 不同时分别记录，
   不把漂移的 `master`/`main` 当证据身份。它们是 consumer/demand evidence，不自动成为同合同性能
   peer；Python/Triton/GPU/tensor/operator-fusion 表面不得机械复制到低层模块。每个差异必须落到本模块、
   具名相邻模块、推理引擎策略或 `MISSING/BLOCKED`，不能以 API 数量对齐代替 owner 裁决。所有准入
   能力必须为 `COMPLETE` 或完整 `ROUTED`。“没有可再加的”只允许解释为当前证据下没有未处置的
   `MUST`；不承诺未来不存在新能力，也不要求复制任一引擎的 tensor/backend/模型策略。
3. **内部复用与加速设施：**模块需要的 error、allocation、container、reclamation、clock、OS/CPU/
   topology、synchronization、SIMD/hash/text 等机制，是否优先复用了依赖方向合法且同合同的现有能力？
   逐项记录候选设施、实际路线和 `REUSE/BORROW/LOCAL/NOT_APPLICABLE`。本地保留必须说明为什么已有路线
   在语义、ownership、复杂度、常态 metadata、codegen、平台或交付上不等价；不得用模块引用数量作为
   KPI，也不得以“避免依赖”为由复制本应抽成更低因子的通用能力。运行时 dispatch/ISA/topology 等
   决策只能有一个 authority。存在未裁决的 material 重复机制或漏用加速设施时，本项为
   `FAIL/BLOCKED`，不能冻结。
4. **性能：**该模块的正常用户任务属于哪一级，是否在冻结 candidate、相同语义/资源计费和真实交付
   形状下得到有效 verdict？`P0` 必须直答“timing peer comparison `NOT_APPLICABLE`，结构成本
   `PASS/FAIL`”，不得改写成“平或更快/SOTA”；`P1/P2` 只有在合格仪器、拟真 consumer/replay、同合同
   strong peer（若存在）和预声明 margin 下才能回答 non-inferior 或 better。没有同合同 peer 时只能
   回答 consumer budget，不得产生 SOTA 声明。
5. **旧 C ABI / Native FFI：**当前全局策略为 `N/A` 时，确认产品树没有公开 C schema/header、bridge
   source、未装饰 C export、C/managed sample、专用 option/target/package metadata 或只为 FFI 支付的
   layout/indirection。内部 OS/第三方/汇编边界的 C 或 `extern "C"` 必须标为 private，不能仅凭存在就
   判成产品 FFI；shared library 导出的 C++ decorated symbols 也不是 C ABI。若未来某个窄 FFI slice
   已通过准入，则改为验证其具名语言 journey、符号白名单、ownership/version/unload，而不是要求清空。
6. **构建与导入：**真实验证组件根的 source embedding、installed static 和 installed shared consumer；
   核对中央组件闭包、selected headers、static compile definitions、shared export/import、Windows runtime
   staging 或 Unix loader-relative lookup、relocation 和 exact package discovery。header-only 或平台上
   确实无该 artifact 时写 `NOT_APPLICABLE`，不能制造虚构 binary，也不能把 producer build 成功当作
   consumer 导入成功。
7. **C++ API 与文档：**从正常用户任务而不是 symbol 清单验证构造、操作、组合、失败、ownership、
   lifetime、concurrency 和 teardown 是否自然；合理默认值与 opt-in 成本是否清楚，编译错误和类型
   形状是否不会迫使用户接触 internal/benchmark seam。architecture 只解释长期边界和决策，user guide
   必须按“什么情况选什么”给出可复制 journey 与陷阱，implementation/performance 只保存事实和证据；
   当前 user 文档不得宣传历史 API、已删除 C ABI 或尚未准入的能力。

七问分别裁决依赖风险、能力完整性、内部复用/加速适用性、性能证据、遗留产品面、交付闭包与实际使用
质量，不能互相代替。
依赖为零不证明功能完整；功能路线完整不证明性能；shared library 可运行不证明存在或需要 C ABI；
producer build 成功不证明安装后的 consumer 能导入；API 能编译不证明用户能自然发现和正确使用；
`CLOSED` 也不把 `P0` 变成计时冠军。owner 可以据此逐模块亲自签署，而不必从 task 全文猜测结论。

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

`PARITY/PASS` 也不能让稳定同方向点值从报告中消失。若全部必需 strata 的 HYLE
`advantage_ratio < 1`，本轮必须按[性能证据标准 §6.1](../performance/benchmark-evidence-standard.md#61-固定方向点值复核)
完成一次有界复核并留下分类；该复核本身不是 reopen trigger，仍须满足上述两把钥匙和第 10 节的真实
产品事实。这样既不会用 5% margin 遮住可解释的系统性问题，也不会为无机制的 0.1% 点值重写产品。

### 5.1 比较类型与语义等价前置门

证据来源的 `consumer/replay/mechanism` 与比较类型是两个正交维度。每个有 peer 的 workload 必须在
采样前只选择一种比较类型：

| Comparison class | 两侧路线 | 可以回答 | 禁止外推 |
|---|---|---|---|
| `NATURAL_ROUTE` | 完成同一用户任务，各库使用其自然、公开且正确的最佳路线 | 用户采用哪条产品路线更合适；HYLE 是否缺少自然档位 | 不能把差距归因于底层算法，也不能签 same-contract SOTA/non-inferiority |
| `SAME_CONTRACT` | minimum semantics、生命周期、进展和计费边界相同；缺失项由具名 adapter 对齐 | 实现级 non-inferiority/better，以及资源/机制取舍 | 不能隐藏 adapter 成本，不能让一侧承担未被任务要求的额外产品语义 |
| `MECHANISM_DIAGNOSTIC` | 刻意隔离 allocation、queue、wake、atomic、dispatch 等单一机制 | 差距归因与复杂度/codegen 检查 | 不能单独形成产品 gate 或公共 API 需求 |

`SAME_CONTRACT` 在运行前必须逐项填写语义等价矩阵：工作/结果、result 粒度、ownership/lifetime、
failure/OOM、cancel/close、ordering/capacity/backpressure、thread-safety/progress、caller participation、
blocking/suspension、affinity/stealing、setup/steady-state/teardown 计费、memory-resource 与交付/codegen
合同。每格只允许 `EQUAL`、`ADAPTER_EQUALIZED` 或 `INTENTIONALLY_DIFFERENT`。任何 material
`INTENTIONALLY_DIFFERENT` 都使 comparability 成为 `NOT_COMPARABLE`，该行只能降为
`NATURAL_ROUTE` 或 diagnostic。实际 allocation 次数、bytes、atomic、wake 和 context-switch 是应当
报告的结果，不要求数值相等；必须相等的是谁拥有资源、何时创建/销毁以及是否计费。

相同用户任务、输出条数或 checksum 不足以证明同合同。任一侧实际承担 material 的额外工作或保证——
包括线程安全、精确一次、失败/取消、排序、生命周期、安全检查、allocation、同步、clock、metadata 或
teardown——都必须写成有方向的 `INTENTIONALLY_DIFFERENT`；候选更强也不能例外。这样的“苹果对橘子”
对比只能签自然路线的用户预算，不能签底层实现 `PARITY`、peer non-inferiority 或 SOTA。完整机械规则见
[性能证据标准 §5](../performance/benchmark-evidence-standard.md#5-同合同对拍)。

`NATURAL_ROUTE` 的额外语义成本仍由选择它的产品承担：如果 HYLE 只提供明显更重的自然路线而输给
peer，这可以形成“缺少合适产品档位”的 capability/performance finding，但不能写成 scheduler、queue
或 allocator 实现本身慢同样幅度。若 peer 没有可防御的同合同路线，写 `NOT_COMPARABLE`，使用预冻结
consumer resource budget 和 HYLE 内部语义成本阶梯；不得为得到绿色 verdict 随意拼一个弱语义 peer。

独立红队必须先攻击 comparison class 与等价矩阵，再看统计结果。发现 material 语义不等价时，即使
A/A、known-delta、timer floor 和统计门全部通过，也要撤销该 same-contract verdict；精密仪器不能修复
错误命题。

## 6. 固定预算与硬停止

- scope card 预声明总调查预算和性能预算；超过预算只能 `CLOSED`、`BLOCKED` 或由 owner 显式重新
  scope，不能默默继续。
- 普通约千行及以下的 P0/P1 模块，默认审查与取证预算不超过两小时；计划实现超过一个工作日时先
  拆成可独立验证的 defect/demand。
- 每个 immutable candidate 只保留一份有效完整 P1 bundle，必需反序在同一冻结 bundle 中完成。有效
  material FAIL 后最多一次 codegen/profile 机制检查；需要产品修改时先退回 `AUDITED` 并形成新
  candidate。旧候选不得靠追加样本、换顺序或 metadata 提交跑到绿。
- 纠正必须分成三类，不能共用一个任意的“总修正次数”：**产品机制纠正**改变 API、算法、layout 或
  normal workload，必须退回审计并生成新 candidate；**验证 fixture/tool 纠正**只使受影响 gate 的
  execution identity 失效，修好后只重跑该 gate；**环境/采样失格**不产生产品 PASS/FAIL，只能在重新
  取得预声明合格环境后重做该 bundle。后二者不消耗产品修改名额，但仍受本轮 wall-clock 与性能预算
  约束，并必须保留失败事实和纠正理由。
- 同一轮不得发明第五种 benchmark 姿势。活动收口轮需要的中央仪器无法给出 verdict 时，记录独立
  instrument issue，并让该轮按硬门进入 `BLOCKED`；不得在模块任务内扩建统计平台。回放一个已经
  `CLOSED` 的模块时，后来发生的仪器实现或兼容策略漂移只产生 `METHOD_GAP`，产品状态保持不变；
  只有新的产品事实满足 reopen trigger，或 owner 明确要求用当前身份续签声明，才进入重开/阻塞流程。
- 历史 engineering evidence 的 consumer provenance 整文件 hash 漂移时，先比较冻结 workload 的
  ownership、charged phases、material contribution、算法和 delivery。全部不变才可由独立 instrument
  task 登记精确旧 hash→新 hash 的 `NON_MATERIAL` renewal；未知变化继续 `METHOD_GAP`，material 变化
  使用新 contract/qualification。Formal 不继承这种 renewal，旧 manifest 永远不改写。
- 红队 P0/P1 默认 30 分钟，只攻击已冻结的能力、合同、correctness、依赖和工程性能门。新发现的
  通用仪器增强进入下一版工具；除非已证明当前 material verdict 无效，不追溯移动本轮终点。

稳定 material `FAIL` 在一次机制诊断后仍未解决时，记录具体性能缺口并 `BLOCKED`。`BLOCKED` 必须
指向仍然存在的 material 问题、缺少的外部输入或无法取得的适用环境及其恢复条件；不得仅因为 fixture
修正、环境失格或任意计数器耗尽而阻塞一个已有确定性、范围内修复的问题。

## 7. Candidate、验证与独立红队

普通迭代先用[change-impact selector](change-impact-validation.md)生成机器计划，保留
selected/skipped gates 与理由。文档、状态和已证明格式变化不触发产品构建或 P1；模块实现只验证
对应 factor 与真实一跳消费者，公共头/API、组件图和 build/package seam 才扩大。额外 sanitizer/
ISA/平台风险仍以冻结卡为准。P1 在 API/算法冻结的 immutable candidate 上只启动一次完整 bundle；
tree/contract/toolchain/machine/tool 身份一致时复用证据。有效 FAIL、无效 fixture/tool、环境失格和
操作者中断按第 6 节分别记账；换日志目录或追加文档提交不能把有效旧结果冒充为新产品证据。

`CANDIDATE` 同时冻结 product commit/tree、公开合同、算法、normal workload 和 tier；harness/tool
identity 单独记录。验证 manifest 至少给出：

- correctness 与 sanitizer/平台 verdict；
- 直接消费者 smoke 和中央 component/package verdict；
- P0/P1/P2 工程性能 verdict、机器/仪器 ID 和证据；
- user journey 与文档检查；
- 独立红队问题的 `fixed/disproved/accepted/out-of-scope`；
- 已知限制、terminal receipt 与 reopen triggers。

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

v2 的 terminal task 与 registry 状态可以在同一提交结束：task 从 `ACTIVE` 改为 `HISTORICAL` 并在正文
记录逐组件结论，registry 绑定每个组件的 candidate、closure task、gates、限制、reopen triggers 与 LOC。
联合 task 可以让不同组件得到不同 verdict，因此不再用单个 task-level `CLOSED/BLOCKED` marker 复制或
覆盖逐组件状态。Git 已经为这个提交提供身份和历史，不再要求先提交 receipt、再提交状态、再
回填一个无法在提交内自引用的 `end_commit/registry_anchor_commit`。v2 当前 registry 不接受这两个字段、
与 `closure_task` 重复的 `closure_evidence` 或已退役的 `self_with_tests`；旧 Git revision 和历史 task
仍保存当时事实，不把兼容字段继续带入当前合同。新记录只用 `closure_task` 指向该轮冻结卡与终态
receipt。旧 task 中的 `SIGNED` 或 receipt marker 只按历史原文解释，不是 v2 兼容入口或当前状态。

Product boundary 不等于 LOC boundary。terminal review/replay 以及后来确实触碰该 boundary 的改动，
由 reviewer 用普通 Git diff 比较 candidate→terminal/current 的实现路径；不是每次打开仓库都重审全部
`CLOSED/BLOCKED` 模块。签后产品漂移必须按 trigger reopen，不能靠旧 reviewed commit 自证。
组件依赖、CMake 语义与 package/source delivery 由 build/package 的真实
configure/build/install/source-package contracts 负责，模块 task 记录这些门的结果；closure checker
不扫描 Git 历史、不重新实现 CMake 解释器，也不从文本 diff 猜测交付语义。历史 audit 的
product identity 由 reviewer 从 Git 打开 terminal registry/task，核对当时确为对应状态、round、
candidate 与 receipt。

## 9. 终止输出与三项 LOC

只有所有适用工程门有 verdict、所有 `MUST` 能力为 `COMPLETE` 或完整 `ROUTED`、不存在
`PARTIAL + KEEP`、内部复用/加速设施审计为 `PASS`、P1/P2 工程性能为 `PASS`、没有 blocker，且
限制/reopen trigger 写明时才能
`CLOSED`。每轮在 `CLOSED` 或 `BLOCKED` 终止时仍必须运行：

```text
python scripts/loc-report.py --module <component> --commit <full-snapshot-commit> --self-check --json
```

不带 `--commit` 的形式只观察当前工作树规模，不得登记为 candidate LOC。`--commit` 从指定不可变
快照直接复算三项数值，不能
用当前 dirty worktree 的计数回填另一个 candidate。source bundle 没有历史对象时这项门明确不适用，
不得伪称已深验。

并绑定同一快照记录三项 `code`：

1. 全库代码；
2. 自实现不包含测试；
3. 当前模块实现代码。

历史 registry/task 的“自实现包含测试”数字仍是当时事实，但 v2 不再计算、要求或常规报告。LOC 只
用于规模与过程复盘，不是完成度、优先级或重开依据。

## 10. 重开条件

只有以下新事实重开受影响部分：

- 可复现 correctness/security、OOM、取消、并发生命周期或交付缺陷；
- 真实消费者暴露缺失能力、错误 ownership、难用接口或重复机制；
- 新的或原先漏审的低层 HYLE/标准/成熟依赖路线成为同合同可用设施，证明当前存在 material 重复机制、
  重复 runtime decision authority 或漏用加速；
- 锁定同合同正常 workload 出现 material regression，或 profile 将模块提升为 P2；
- 直接依赖、compiler、平台、ISA/backend 改变实际语义、算法或 codegen；
- 公开 API、组件依赖或交付形状改变；
- 强 peer 证明更好的复杂度档或可复现 material consumer advantage。

第三方版本、vendored/ported 内容或 adapter 改变时，先比较锁定身份与实际产品输入。若变化会影响
源码、语义、算法、默认策略、ABI/codegen、运行时 decision authority、许可证或交付，即形成受影响
能力族的定向重开输入，并回放直接消费者；旧性能证据不得跨产品依赖 identity 自动继承。只有下载
位置或等价元数据变化、且机器合同证明产品输入完全相同时，才可只走 build/package 维护任务。

新 benchmark、新 peer 发布但没有 material 证据、LOC/API 数、命名偏好、单次点值、地址翻转、
亚纳秒差异或 Formal 证书单独过期都不重开产品。
