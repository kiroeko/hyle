# M2-B1 Apple：原版 MLX 张量、归一化与状态批量覆盖

## Closure card

- component / owner：`compute_route_apple`；`/root`。普通Graph是直接消费者，当前没有已证实需要
  修改`graph_route_apple`产品的事实，不自动重开它或ggml Metal/runtime。
- baseline：`50189a2b`，Windows canonical，工作树干净。当前阶段SCOPED，先审计后固定有限FIX。
- trigger：M0/M2的B1要求F01.1–3、F02.1–4、F03.1–4、F04.1–6、F19.1–4共21行由成熟
  实现承载；M1-A terminal明确允许新数值、shape或region需求重开。旧有限M1验收继续有效。
- 主纵切：embedding→norm/gate→residual/index/state，两个请求、动态索引、旧结果保活。
  正交纵切：无模型注解的broadcast/strided/scalar/empty张量和多输出Graph。
- 来源：已锁MLX0.32.2 `1f8e74e3`（MIT），必要时仅用已准入原版ggml Metal组合。
  不从库名推断完整语义，不重新下载已有锁定来源。
- scope：21行需求到原版MLX operation/compiled region的成批映射与资格；复用现有native
  array、Context线程/stream、Program/Invocation、checked结果和普通Direct/Graph。
  现有纯MLX调用可按身份复用；冻结的自有Metal/scatter/check数学不算成熟覆盖。
- non-goals：研究/修改/扩张/比较/删除任何冻结自研算法；新Metal kernel或同义改写；模型loader；
  新万能执行器；公共构建/性能/closure仪器重构；解除CPU oneDNN Q2；替其它路线签覆盖。
- API/依赖：保留自然Apple typed入口与MLX原生array/region，普通用户不写Provider glue；
  Compute不依赖Graph，不为逐节点管理拆坏原版融合。当前无新增公开API/依赖factor要求。
- 预算：本M2 Apple批源码/设计总120分钟，前置合同/接口定位计6分钟，余114分钟。
  独立于已经关闭的M1资源范围；同批后续子卡共享额度，不按算子续杯。每具名根因最多两次
  源码驱动修正；正常构建/预声明验证单列，到预算固定具体结论或阻塞，不无限研究。
- 性能：先P0账单；API/算法稳定的不可变候选才启动一次必要P1，对照同一原版成熟完整调用，
  包括转换、同步、融合、checked终态与结果寿命。5%非劣门；A/A和双顺序先资格，15对起，
  合法总上限31对。不测试冻结实现的输赢，Mini不代签Air Formal profile。
- stop：21行及全部适用门完成后关闭；不兼容域保留原版反例与具名成熟后备责任，不缩小分母。
  不把接口未接通改名为算法缺口。完整goal继续；M5开始前暂停等待用户。

## 待审有限改动集

| ID | 允许范围（审计冻结后才实施） | 必须证明 |
|---|---|---|
| A1 | 21行能力表：复用原版MLX tensor/arithmetic/norm/region；按需要接原版indexed/reduction/sort及合法组合 | dtype/rank/stride/empty/axis、特殊值/舍入、ties/NaN、重复索引和checked溢出，不能进入自研数学 |
| A2 | 既有冷lowering与原版MLX检查结果接线；只作格式/布局/端口、错误语义与成熟原子的组合 | 原版计算在GPU；只读回必要验证摘要；坏索引/溢出不发布成功结果或推进后段state |
| A3 | 普通Direct/Graph默认成熟选择、compiled region与每请求原生owner | 两slot、旧结果、变化输入、取消/关闭/失败、无热Provider搜索或计时；Graph变化若必需先另定owner |
| A4 | 对应测试/真实组合consumer、所在factor CMake、用户文档和成本记录 | 两Mac、host插桩、Windows/WSL OFF、同份source/static/shared组合消费，不逐算子重跑交付 |

## 审计需回答

1. 已有原版MLX绑定与冻结数学的边界：仅以调用/来源身份确认，不调查冻结实现的算法。
2. 原版Gather/Scan/稳定TopK、Select/min/max、stable activation、mask/LogSoftmax与norm舍入
   是否满足C0精确域；索引/溢出检查也必须由原版成熟运算承担，不新增检查kernel。
3. 原版compiled图如何保留动态数据依赖、检查失败和两输出；既有checked元数据只作资源接线。
4. 原版Scatter重复add是否满足逐更新舍入与顺序；不相容时记录反例，不用atomic同名算子代签。
5. 所有准备/运行分界、原版隐式materialization/分配/host wait以及公开输入ready前提准确记账。

## 预声明验证

- 一组原版资格表，覆盖本批新增成熟调用的正常/空/尾/strided、rank0..6/8、各axis、F32/F16/BF16、
  index宽度及特殊值；独立oracle，不调用或计时冻结实现。失败最多两次具名纠正。
- 产品普通Direct与Graph的主/正交组合、两slot/旧结果/变化输入、动态非法索引与溢出、OOM/失败
  与close；不变资源设施只复用身份匹配的证据，新增调用必须实跑。
- Air和Mini的Apple原生正确性；Air适用host ASan/UBSan不代签Metal device。Windows/WSL
  按selector执行Apple-OFF。重型跨机器串行、parallel2，不修改电源/权限或弹UAC。
- 一份真实组合consumer在两Mac完成source/static/shared、prefix搬移和producer隐藏，检查原版
  MLX资源/许可/导出；复用未变交付接缝，新增域仍要实际运行。
- selector选中门、docs/closure（registry带明确previous-ref）、一次最终候选复核与必要P1；
  文档/状态不构建、不采样。统一Compute+Graph实现LOC，不包含测试/vendor。

当前仅SCOPED；未冻结产品FIX，无新增完整覆盖PASS。CPU/CUDA原卡、预算、未决域与验收保持。
oneDNN新能力按用户2026-09-21决定进入[独立scope](llm-moe-m2-b1-onednn.md)，旧Q2资格FAIL保留。

## 首组原版资格检查点

Air Mac16,12、macOS27.0/26A428上，锁定原版MLX的GPU compiled程序已执行127组资格：
19组F32/I64提升Scan及溢出摘要、48组stable argsort→take（F32/F16/BF16/I32、17/129/4097列、
两方向/变化输入）、60组原版take（六种storage、I32/I64、重复/负数/轴宽/最大索引）。
原版算子组合及独立host oracle通过，未调用冻结HYLE计算或计时。

原始源、构建、API调用首轮编译错误与修正、库与源码hash见
[资格收据](../../../../evidence/correctness/compute-llm-moe-m2-b1-apple/indexed-qualification.json)。
这只证明所列原版组合；产品lowering、普通入口、资源/失败、Mini与三交付/P1尚未签。
来源/组合与checked资源接缝核对累计26/120分钟，余94分钟；独立构建与资格执行单列。
下一步提交有限AUDITED接法后方可实施，不据此填完整需求格。
