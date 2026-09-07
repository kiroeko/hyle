# util-runtime v4：推理 factory 生命周期与新标准收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen intake 与有限 closure card

- component_id：`util`，语义名称 **util-runtime**；owner：runtime utility owner；round：
  `util-runtime-v4-inference-lifecycle`；kind：`REOPEN`。`util_core` 是另一个已 FROZEN 因子；
  `hyle::util` 是聚合两因子的用户产品 façade，三者不能混称。
- canonical checkout：`C:\public\project-hyle\hyle`。开始时 clean，main、origin/main 和只读查询到的
  远端 main 均为 `ded05c9e36fffe8cdc54515d7413097cfdec230e`；本轮 baseline 固定为该提交。
- 权威输入：完整 AGENTS、README、registry、closure methodology、capability-completeness、任务模板、
  engineering-charter、coding-conventions、roadmap、build-system、util-design，以及已 CLOSED 的
  frozen inference/performance audits。另读取 registry 当前指向的 util-runtime-v3 terminal 和
  change-impact selector v2 使用合同；不重做三个已完成项目，不继承其微性能问题为实施任务。
- trigger `UTIL-V4-DEPENDENCY-AND-DELIVERY-2026-09-07`：旧 candidate
  `b2b7bd679293d748d996662414c01c4cbcefcbda` 后，`796b654d2791a0472565f19a2ca2fdd311347728`
  已把 Registry 的公开 allocator/storage 改为 `AllocatorRef/Arena`；中央图又引入
  `container_text` 直接边和产品 façade。当前 `samples/package/util/cpp/CMakeLists.txt` 仍断言
  不得出现 `_factor_container_intrusive`，与 `util -> container -> container_intrusive` 的真实
  闭包相反。满足旧 registry 第 2 条 dependency/delivery 及第 3 条 correctness/usability trigger。
  标准升级、旧卡缺 FROZEN 字样本身不是产品 trigger。
- 真实任务：owner 在启动时选择并登记少量已编译或已装载 backend，校验版本/能力，按名解析 factory，
  构造可失败的对象；服务前 freeze，随后缓存解析结果。不同 backend 可有自己的 descriptor/image
  状态；注册失败必须回滚，创建的对象及 callback 不得越过定义它们的 image 生命周期。诊断线程
  把 runtime C++ symbol 写入 owning string 或 caller buffer。
- 范围：两份 runtime public headers、`src/util`、`tests/util`、util package consumer、相关四类文档与
  仅 `module=util` 的状态/投影。审计自然注册/查询/优先级/发布、错误、所有权、复杂度、依赖与交付。
- 非目标：修改或重开 util_core/其他 FROZEN 依赖；Sched/IO/Compute/Graph 产品；通用 ModuleManager、
  unload/unregister/drain 服务、版本求解器、设备选择、Native FFI/C API、稳定跨工具链 C++ ABI、
  benchmark-only API、公共 checker/构建基础设施/性能仪器修改。
- 依赖：`PUBLIC core,error,mem,container,container_text`；没有当前产品直接依赖 util-runtime。
  util façade 与 `samples/package/util` 为本轮真实 C++ 交付消费者；后续 owner 只接收输入。
- tier / instrument：`P0 / closure-correctness-v1`，Formal `NONE`。五个参考的相关操作均为初始化、
  发现或冷诊断；没有 Registry token/kernel 热路径 profile 或计时主张。P1/P2、计时样本预算 **0**；
  timing peer、5% margin、固定输项评测均 `NOT_APPLICABLE`，不能写成数值 PASS。
- 成本预算：register 事务式、owned 字节线性复制、索引 expected-amortized O(1)；freeze 为 O(1)、
  零分配、发布同一索引；frozen lookup 为 O(key bytes)+expected O(1)，无查询隐藏分配/锁；creator/
  visitor 在锁外。仅真实 ownership 需求允许登记必要 metadata，不以旧 benchmark 布局为先验。
- 预算：审查 45 分钟、实现一个有限批次 60 分钟、取证 75 分钟（含候选后 20 分钟独立第二遍红队）；
  审查与取证合计不超过两小时，总计三小时。最多两遍审计、一个冻结 FIX 批次、一次具名纠正与
  受影响重跑；不反复跑已通过门。计时从本轮入口 2026-09-07 08:50 UTC 计，11:50 UTC 硬停止。
- 机器：W1-WIN MSVC 19.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21；只读工具探测三处均可达。
  WSL/macOS 使用 Windows candidate 的独占验证副本；不回传产品，不改既有 checkout。工具链、stdlib
  文件/include roots、flags、machine 与 candidate tree 由 selector context/receipt 绑定。
- 适用门：selector 选择的 util factor correctness、util runtime 两头 self-contained、util 产品
  source/static/shared（搬迁后安装消费）；没有 util-runtime 反向产品消费者，不跑其他 factor 测试。
  另预声明同一 util 测试集的 WSL ASan/UBSan/TSan，三平台 compiled demangle symbol 检查，以及
  P0 release codegen/成本检查。默认 shipping layout；若需 fault seam，只留 util 测试的私有 adapter。
- correctness：正常/empty/invalid、长度与 overflow、每分配点失败、重复/priority、owned/borrowed
  生命周期、capture 与 image 保活、concurrent register/read/freeze、release/acquire、callback 重入、
  锁外执行、typed Result/nullable、demangle required size/截断/双向 overlap/异常与 ABI failure。
  任一适用风险未运行记 BLOCKED；同步名字表的异步取消/排空服务语义不适用。
- 验收：所有准入 MUST COMPLETE/ROUTED；有限 FIX 全部落实；适用门真实通过；文档命名/ownership/
  依赖校准；没有其他 factor 产品/状态变化；immutable candidate、终态 receipt、提交与 push 可追溯。
- 停止：满足上述条件才 CLOSED 并登记新 FROZEN；否则 BLOCKED，列出 owner、已知失败与精确
  resume_condition。不为取得 FROZEN 把未知门改为 PASS 或继续扩大任务。
- LOC：按本轮用户要求只统计全库、自实现不含测试、util-runtime 不含测试。旧 schema 强制保留的
  `self_with_tests` 字段只携带旧值与独立 provenance，不重新统计、不作为本轮候选规模、不对外报告；
  不在模块任务中改 checker 或 LOC 工具。

每次 registry 状态变化之前把原字节保存到 checkout 外
`C:\public\project-hyle\_validation\util-runtime-v4\state-before-<phase>.json`，运行
`check-module-closure.py --previous <snapshot>`，并更新受检 status 投影。状态只以
[`module-status.json`](../../module-status.json) 为真；本卡的 audit/candidate/receipt 是轮次证据。

## 审计进入条件

本卡先登记实际依赖/交付 trigger，再经 `REOPENED -> SCOPED`。在 capability-family map、public trace、
owner 七问与 API/有限 FIX 集完成并进入 AUDITED 前，不修改产品或测试实现。
