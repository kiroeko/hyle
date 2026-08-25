# 模块严格收口方法论

> 本文规定 HYLE 已有模块如何从“代码存在”走到“按当前实际使用标准严格收口”。它是
> 所有模块复审的稳定方法信源；具体施工状态归 `milestones/`，性能点值归 `performance/`，
> 用户写法归 `guide/`。严格收口不是把测试跑绿，也不是把成熟库的 feature 数量抄齐，而是
> 证明模块的职责、产品形状、复杂度、接口、依赖和证据已经形成一致闭环。

## 一、三个不能混用的状态

| 状态 | 只说明什么 | 不能推出什么 |
|---|---|---|
| 实现存在 | 仓库中有代码，可能已有测试和 benchmark | 功能完整、形状合理、性能 SOTA |
| 历史任务完成 | 某次 milestone 的目标和当时验收已完成 | 符合现在的产品标准，或未被后续依赖改动影响 |
| 当前严格收口 | 本文全部门槛在一个明确实现绑定上闭环 | 永远不再修改；真实新证据仍可触发局部重开 |

一个模块只有同时交付稳定设计、正常使用 API、复杂度/性能证据、平台正确性、依赖收束和
正确分层的文档，才能取得当前严格证书。任何一项未知就写未知或黄色，不能拿其他项代偿。

## 二、先冻结审查单元，而不是先看目录

收口第一步是写清“这次到底审什么”。审查单元至少包含：

- 本模块公开 C++ 头、实现、纯 C 头、导出清单和 build target；
- 根目录或其他目录中逻辑上由本模块拥有的能力，例如 core 的根级 compiler macros；
- 本模块使用的直接上游及其 ownership/error/platform 合同；
- 直接下游消费者、adapter、ported/vendor-derived kernel 和公开 sample；
- 对应 architecture、guide、C guide、performance 与 milestone 页面。

目录不是产品边界。能力放错目录、根级头、header-only seam、PRIVATE target 和复制到下游的
局部实现都必须纳入。开始时建立一张边界表：

| 项 | 当前事实 | 本轮是否纳入 | 理由 |
|---|---|---|---|
| 公开 C++ surface | 头/target | 是/否 | 用户或下游合同 |
| C/FFI surface | 头/导出 | 是/否 | 是否语言无关 |
| 私有热核 | 文件/消费者 | 是/否 | 是否决定公开承诺或性能 |
| 上游依赖 | target/header | 是/否 | 是否要回验调用方式 |
| 下游消费者 | module/file | 是/否 | 是否采用或复制了本能力 |

同时按依赖图判断模块角色。越靠近根，越应是零状态、窄职责、稳定的编译期/机制合同；越靠
近上层，越可以承载 ownership、调度和应用级组合。若能力进入本模块必须反向依赖更高层，先
修归属，不能用 forward declaration、C ABI 或复制代码遮蔽架构倒置。

## 三、从真实场景重建功能面

### 3.1 外部用户场景

不能以现有类名或 benchmark case 作为需求清单。先按用户任务写场景，每项至少回答：

| 场景 | 数据/操作形状 | ownership/lifetime | concurrency | failure | 规模/平台 |
|---|---|---|---|---|---|
| 用户要完成什么 | 固定/可变、批量/单项 | 谁创建、借用、释放 | 单线程/共享/分片 | 可恢复/空值/程序错误 | 正常 N、硬件、OS |

场景来自 HYLE 的北极星用户、公开 sample、现有 issue/guide 和成熟基础库的常见消费方式。每个
公开类型或函数都要能反向指向一个场景；无法指向的入口是删除、私有化或重新归属候选。

### 3.2 HYLE 内部消费者：必须做两遍扫描

第一遍是**语法消费者扫描**：

- include、namespace、类型名和函数调用；
- CMake PUBLIC/PRIVATE/INTERFACE 链接边；
- C ABI 调用、导出、sample 和 benchmark；
- std/OS/第三方 API 的直接调用，判断是否绕过已存在的 HYLE 能力。

第二遍是**语义重复扫描**，不能依赖“谁 include 了本模块”：

- 下游自建同义 wrapper、helper、macro、allocator、clock、status 或 platform branch；
- ported 代码为适配而复制的 compiler/cache/error/ownership primitive；
- 被置空、降级或 test-only 的上游宏和算法合同；
- 名称不同但解决同一问题的局部类型；
- benchmark 为方便比较而独立造出的元数据、入口和状态。

core 的旧审查曾只找到显式引用者，因而看不到 SIMD/vqsort 内部复制的 prefetch/compiler
primitive。这个案例确立一条硬规则：**“没有 include 本模块”不能证明“没有消费本模块的
职责”**。后续每个模块都要同时交付语法与语义两份消费者账本。

### 3.3 功能账本与负空间

将结果收成能力账本，而不是散落 TODO：

| 能力/场景 | 外部用户 | 内部消费者 | 当前实现 | 决策 | 证据 |
|---|---|---|---|---|---|
| 能力 A | 谁/为什么 | module/path | 完整/部分/无 | 保留/补齐/迁移/删除/不做 | API/test/ref |

“不做”也是产品结论，但必须写明：属于哪个更高/更低模块、标准库或成熟依赖已经更合适，还是
缺少真实消费证据。不能用“以后可能有”增长公共 surface，也不能因当前 benchmark 没覆盖就
把常见能力写成 non-goal。

## 四、成熟库对账：参考问题解法，不比功能数量

每个模块选择承担不同角色的强参考：通常包括一个产品 API 参考、一个算法/复杂度参考、一个
跨平台或 ABI 参考。直接看其公开头、实现、测试和真实消费者，不只读 README 或 benchmark。

| 成熟能力 | 解决的真实问题 | API/ownership 形状 | 复杂度/算法 | HYLE 决策与差异 |
|---|---|---|---|---|
| peer feature | 使用场景 | 谁拥有/并发/失败 | 时间与空间档 | 借形状、借算法、已有等价、明确不做 |

采用一项能力需要同时满足至少一个 HYLE 用户或内部消费者场景，并且不破坏模块归属。成熟库
存在某 feature 只是查漏信号，不是实现授权；反过来，若强 peer 用不同数据结构把复杂度从
线性降到对数、摊还常数或无锁快路，不能把差距解释成“benchmark 误差”，必须研究其方案。

参考实现可以带来四类结果：

1. HYLE 已有同档能力，只补证据或命名/易用性；
2. HYLE 缺常用能力，按真实场景补齐；
3. HYLE 形状错误，迁移、组合或删除旧类型；
4. 能力不属于 HYLE/本模块，记录负空间和替代路径。

## 五、先裁决类型和模块归属，再扩 API

对每个保留能力依次裁决：

- **ownership**：owned、borrowed、view、intrusive、process-lifetime 还是 external resource；
- **lifetime**：作用域 RAII、显式 close、bulk release、异步 completion 后释放；
- **concurrency**：单 owner、跨线程 free、MPMC、shard-per-core、外部同步；
- **capacity**：固定、有界、可增长、调用方 backing、系统 backing；
- **failure**：`Status/Result`、空值合同、partial result 或 fail-fast；
- **composition**：默认路径、注入 allocator/platform/scheduler 的方式，以及与相邻模块怎么拼；
- **specialization**：真正公共产品、HYLE 内部 seam，还是某个 kernel 的特化实现。

公开类型应贴近用户描述问题的方式，而不是 benchmark harness 的参数方式。只为测量方便存在的
入口、常态元数据、全局开关、隐式 TLS cache 或第二套错误类型必须删除或内收。内部特化可以
保留，但不能伪装成通用用户能力。

此时还要裁决运行时/编译期和动态/header-inline 边界：布局常量、模板、指针原语和编译器 hint
通常应 header-inline；跨 DLL ownership、runtime identity 和 opaque resource 才进入动态 ABI。

## 六、复杂度先于纳秒优化

为所有非平凡操作写复杂度表：

| 操作 | 正常/最坏复杂度 | 空间/常态元数据 | 并发进展保证 | 强 peer | 结论 |
|---|---|---|---|---|---|
| find/free/wake 等 | O(1)/O(log n)/O(n) | 每对象/每线程/全局 | wait-free/lock-free/blocking | 算法 | 同档/待修 |

顺序必须是：

1. 先找更好的算法、索引、批处理或 ownership 模型；
2. 再消除分配、锁竞争、cache miss、间接调用和错误 dispatch；
3. 最后才调指令、prefetch、unroll 和 layout。

如果成熟库通过 TLSF、树索引、分片、批处理、generation、radix wheel 等结构改变了复杂度档，
HYLE 必须严肃尝试或用真实约束否决。不能用一个只覆盖小 N 的 benchmark 证明 O(n) 足够。

同时检查优化是否把永久成本强加给普通对象：常态字段、TLS、owner map、refcount、虚表或统计
都要有生产消费者和收益证据。可选诊断/缓存应模块化，而不是污染默认类型。

## 七、依赖收束是双向的

### 向下：本模块正确使用地基

检查本模块是否绕过或错误使用已经冻结的：

- error：是否另造状态码、TLS last-error、异常或静默失败；
- platform：是否直接散落 OS syscall、CPU 探测、线程/VM/NUMA 分支；
- mem：是否跨模块混淆 host/device ownership，或私自选择 allocator；
- time：是否混用 wall/monotonic、单位或溢出语义；
- core/util/obs：是否复制 compiler primitive、锁、日志或 trace seam。

### 向上：下游正确采用本模块

扫描直接消费者是否：

- 复制本模块已有机制；
- 绕过公开/内部自然入口直调 OS、std 或第三方；
- 错用 ownership、线程或失败合同；
- 迫使本模块暴露只适合该消费者的特化类型；
- 因旧 API 仍保留 adapter、兼容分支或空实现。

改动后只重测真正触碰的依赖边和消费者，不把所有已冻结模块全部重开；但如果发现的是模块
职责或公共合同改变，就必须把受影响的已认证直接邻居转黄并局部复证。

## 八、性能证书必须来自同合同正常用法

性能问题按三层验证：

1. **算法/复杂度**：代码审查、数据结构不变量和大规模增长曲线；
2. **机制/codegen**：指令、内联、分配、锁、syscall、cache layout、dispatch；
3. **产品 workload**：真实公开入口或正常内部消费者，对强 peer 做同合同对拍。

“同合同”至少要求相同：语义、ownership、错误处理、线程数、容量策略、数据分布、ISA、链接
形状和 QoS。不能让 HYLE 承担检查/分配/dispatch，而 peer 使用 unchecked direct kernel；也不能
反过来靠删掉用户需要的语义赢。

选择 benchmark 的顺序是场景 → workload → peer → 统计方法，绝不能从现有 benchmark 反推
API。微基准可以解释一次原子、hint 或函数边界，但不能独立证明产品 SOTA。没有独立 workload
的 compiler/cache primitive 应通过 codegen 和真实消费者验证，而不是制造无意义的单指令表。

每个结果在进入性能表前先分级，不能让一个数值自动取得胜负资格：

| 级别 | 必须满足 | 允许的结论 |
|---|---|---|
| 产品对拍 | 同语义/ownership/QoS，正常用户或内部消费者 workload，setup 与必要检查同等计费 | 通过下述统计和稳定性门后才可写 WIN/TIE/LOSE |
| 机制诊断 | 隔离一段原子、链表、dispatch 或 codegen，用于定位产品结果 | 只解释根因，不单独签 SOTA |
| 不可比 | 任一侧改变分配、检查、批量、容量、数据布局、并发语义或失败合同 | 只能写“不可比”，不得计算胜负 |
| 不稳定 | 同合同但执行顺序、等价 harness 写法、函数/堆布局或微小 code placement 扰动使方向翻转 | 写“未知/placement-sensitive”，不得挑有利变体，也不得用 TIE 暗示已经证明同档 |

尤其是 sub-cache、小批次和百纳秒级 aggregate：即使单一二进制的 paired sign test 显著，若
合同等价的 runtime bound/value capture、variant 排列或 code placement 能使方向翻转，该格仍
没有产品胜负资格。只有机械确认关键热核同等计费，并且真实消费者没有回归时，才可把它降为
“机制 TIE（placement-sensitive）”；缺少任一层就继续保持 unknown/黄色。禁止为了得到稳定
有利的地址或指令排列而修改产品字段、API、常态元数据或挑选 benchmark 写法。

真实消费者语料不能只来自 HYLE 已有测试。与模块职责相关时，应从本地成熟项目的生产路径
抽取 workload，例如 llama.cpp/ggml 的模型映射与权重驻留、量化块遍历、KV 生命周期、
prefill/decode 批量、线程/NUMA 和 backend handoff。抽取的是数据、生命周期和并发合同；模型
专用类型与调参不得倒灌到基础模块公开 API。

默认性能闸包括：

- 代表性小/中/大规模和正常数据分布；
- 适用时单线程、共享争用和分片路径；
- 至少 15 个独立进程样本；同进程 peer/candidate 比值做 paired sign test，只有 `p < 0.01`
  且中位差大于 1% 才判 WIN/LOSE，否则 TIE；
- 默认顺序与 peer-first 正反顺序必须给出相同方向；翻转就记 TIE/未知，不挑有利顺序；
- 对 sub-cache 或短 aggregate，至少再做一种合同等价的 code-shape/placement 扰动；方向翻转
  即降为 unknown/placement-sensitive，并转由真实消费者守门；
- Windows/Linux/macOS 真机；ISA/后端不可得时明确写“未认证”，不用模拟器时序；
- 每个 LOSE 都要归因：算法缺口、实现缺口、必要语义税、硬件墙或测量无效。

单边绝对 spread 是机器噪声诊断，不代替同进程 paired ratio：共同噪声可能让两边绝对值都漂，
却不破坏配对方向。反之，paired ratio 自身大幅漂移、正反顺序不一致或样本不足，只能使结论
变成未知/TIE，不能证明没有问题。先用隔离 workload、反汇编、计数器或不同机器降低不确定性；
确实无法判定时保留黄色，而不是把怀疑写成 SOTA。

## 九、C++ API 与 C ABI 是两个消费产品

API 易用性是独立硬门，不能被“功能完整”“符号齐全”或“性能很快”代偿。评审时至少走三条
用户旅程：第一次使用的默认路径、需要注入/调优的高级路径、与相邻 HYLE 模块组合的路径。
每条都实际写一段最小调用代码，并记录用户是否被迫知道实现细节。

| 易用性问题 | 通过标准 | 常见失败形状 |
|---|---|---|
| 可发现性 | 按任务能找到一个自然头、类型和动词 | 必须从 benchmark/test 猜入口 |
| 首次正确调用 | 最短 recipe 含完整 ownership/error/lifetime | 先拼 vtable、内部状态或神秘常量 |
| 默认值 | 常用场景少参数且语义明确 | 每次都填 options；默认依赖隐式 TLS/全局状态 |
| 类型引导 | owned/view/handle、单位、clock、device 等不易混用 | 裸整数/`void*` 承担多个语义 |
| 状态与失效 | valid/moved-from/closed/cancelled 规则可见 | 调用顺序只藏在实现或 death test |
| 高级控制 | 注入 allocator/cache/backend 等 opt-in，普通对象零税 | 所有人承担调优元数据和生命周期 |
| 失败处理 | 正常失败自然传播，程序错误明确 fail-fast | bool/NULL/TLS/异常混用，错误信息丢失 |
| 组合 | 与上下游类型直接拼，不需要格式往返或 adapter 链 | 每个调用点手写转换/复制/桥接 |

可复制 guide recipe 是易用性测试的一部分，但文档不能替坏 API 辩护：如果正确调用需要长篇
解释内部状态机，应先改类型、默认构造、options builder 或 convenience，再写 guide。

### C++ source API

必须提供：

- 一个不读内部实现也能选对的默认路径；
- RAII/typed ownership、明确 move/copy 与失效规则；
- 正常组合 recipe，而不要求用户拼 vtable 或 benchmark 参数；
- 与共享 error、allocator、platform/scheduler 的自然集成；
- 高级控制是 opt-in，不让普通调用承担其状态和成本。

至少保留一段“默认路径”和一段“高级/组合路径”的编译测试或 sample；只在单测里直接构造
private-ish primitive，不能证明 C++ 用户 API 易用。

HYLE 自己的 C++ 实现直接使用 C++ API/内部 seam，不绕经 C ABI。

### C/FFI 产品面

C ABI 按**语言无关能力**覆盖，不机械翻译 C++ 模板和内部类。每项 C++ 能力都要登记：C 可
直接表达、需要 opaque handle/callback/visitor、应 header-inline，还是有明确语言相关理由不
提供。没有“无法覆盖但未解释”的洞。

C 易用性另查：是否有 `*_INIT`/默认 options、create-use-destroy 对称、NULL/长度查询约定、
typed/header-inline 便利层、callback+ctx 和 buffer-fill/visitor。能靠公开 POD 表达不代表应该
让用户手填内部字段；只有 intrusive link 等明确以 transparent layout 为产品合同的原语例外。

机械验证至少包括纯 C 编译、独立头、导出 manifest、create/destroy 与跨 DLL allocator 归属、
callback 生命周期、错误字符串所有权、共享/static consumer。C ABI 易用性不能只靠“符号数
相等”判断；一个需要用户手工管理内部状态机的完整镜像仍然是坏接口。

## 十、正确性与平台门

测试从合同推导，不从实现分支凑覆盖率：

- ownership/lifetime：move、self-move、释放顺序、bulk/individual release、迟到 completion；
- 边界：zero、overflow、OOM、invalid handle、重复调用、部分失败；
- concurrency：竞态、进展保证、跨线程/跨核、shutdown/cancel；
- platform：Windows/Linux/macOS 的真实后端与降级路径；
- toolchain：MSVC、GCC、AppleClang，适用时 baseline/特殊 ISA 和交叉编译；
- sanitizer：ASan/UBSan/TSan 按能力适用性执行；模拟器只证明正确性，不证明性能。

测试绿只证明已写断言的合同，没有替代功能查漏、依赖扫描和成熟库对账。

## 十一、文档和证据归位

一次收口必须同步交付：

| 信源 | 必须回答 |
|---|---|
| `architecture/<module>-design.md` | 当前职责、类型/所有权、依赖、负空间和长期理由 |
| `guide/<module>.md` | 用户在什么场景选什么，默认 C++ 写法、组合方式和坑 |
| `guide/c-<module>.md` | C 能力映射、opaque/inline/导出、所有权和 FFI 坑 |
| `performance/benchmarks-<module>.md` | workload、peer、合同、平台、commit、统计、LOSE/未知 |
| `milestones/<module>-*.md` | 本轮改动、命令、验收、剩余门、LOC、commit/push |

architecture 不写每日状态和 benchmark 点值；milestone 不承担当前设计真相；guide 不复制性能
表。公共 API 改了但 guide 没变，或者性能结论没有绑定实现，都不能标绿。

每轮结束固定报告四项 `cloc`：全库、HYLE 自实现含测试、自实现不含测试、当前审查模块实现。
代码量不是完成度，但能暴露漏算边界、测试挤占和异常膨胀；显著变化必须解释。

## 十二、红队问题：在宣布完成前必须逐条回答

1. 如果某个下游没有 include 本模块，它是否仍复制了本模块职责？
2. 每个公开类型能否指向真实用户/内部场景，还是只为 benchmark 好测？
3. 强 peer 是否用另一种算法或 ownership 形状取得更好的复杂度，而我们只解释成噪声？
4. 本模块是否绕过 error/platform/mem/time/core 等已冻结地基？
5. C ABI 是用户可用的能力产品，还是对 C++ 内部形状的机械镜像？
6. 默认对象是否为可选统计、cache、owner tracking 或调试承担永久元数据？
7. 文档、导出、build target 与代码是否描述同一个产品边界？
8. 三平台缺一时，状态是否诚实保持黄色？
9. 本轮改动影响了哪些已认证邻居，是否只回验了正确的边？
10. 如果删掉全部 benchmark，这个 API 和类型形状是否仍然合理？

红队不能只找“代码 bug”，还要主动攻击需求来源、模块归属、复杂度、依赖和证据有效性。

## 十三、证书、停止和重开

严格证书绑定一个明确 commit、公共 surface、参考账本、三平台证据和性能文档。全部门完成前
保持黄色；不允许“先写完成、以后补机器”。标绿后停止无证据的重复全审，只有以下事件重开：

- correctness/security/ABI 缺陷；
- 新平台、ISA、后端或编译器使合同失效；
- 直接或传递依赖升级，改变了本模块实际编译进来的 API/ABI、默认 feature、算法、线程模型、
  ownership、错误语义、平台选择或性能基线；
- 同合同强 peer 证明复杂度落档或存在可复现 material LOSE；
- 真实消费者暴露缺失能力、难用接口、复制机制或错误依赖；
- 发布/授权/构建形状改变模块边界。

重开默认是**受影响项复证**，不是推翻所有旧证据。旧证书绑定的未修改算法仍可保留；新能力
及其消费者转黄，完成相应平台、性能、API 和文档门后重新绑定。没有 workload 的猜测、单次
亚纳秒波动、新库出现、命名偏好或为了增加 LOC，都不是重开理由。

依赖升级不能只做“改 pin + 全量测试”。升级提交必须先产出影响账本：依赖的旧/新 revision、
实际启用 feature、HYLE adapter/直接调用点、install/static closure、license/offline 来源，以及
可能变化的语义和性能路径。然后按影响账本重开相应门：编译通过只复证了 source compatibility，
测试通过只复证了已有断言；默认行为、错误映射、跨 backend identity、allocator ownership 和
同合同性能仍须分别检查。若升级后发现旧 HYLE 实现本来就错误，按产品缺陷修实现并补回归，
不能为了保持旧证书而在 adapter 中模拟错误行为。

## 十四、每个模块的最小交付清单

- [ ] 审查单元与依赖位置明确；
- [ ] 外部场景矩阵、语法消费者账本、语义重复账本完成；
- [ ] 成熟库能力/算法/形状对账与负空间完成；
- [ ] ownership/lifetime/concurrency/failure/组合与 public/internal 边界裁决完成；
- [ ] 非平凡操作复杂度表完成，算法档无未解释落后；
- [ ] 向下采用与向上消费者收束完成；
- [ ] C++ 默认路径和高级路径可用；
- [ ] C++ 默认/高级/组合三条用户旅程有可复制 recipe 或 sample，易用性表无未解释摩擦；
- [ ] C ABI 能力映射无未解释缺口，C 便利层、纯 C/导出/FFI 机械门通过；
- [ ] Windows/Linux/macOS、适用 sanitizer/ISA 正确性完成；
- [ ] 同合同正常 workload 性能完成，LOSE/未知诚实登记；
- [ ] architecture/guide/C guide/performance/milestone 归位；
- [ ] 四项 LOC、实现 commit、提交与推送完成。

任何未勾项都必须显示为剩余门；“功能面/性能/API/文档大致没问题”不再作为收口结论。
