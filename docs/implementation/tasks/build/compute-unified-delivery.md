# C1–C3 / R1：统一 Compute/Graph 配置与交付

<!-- MODULE REVIEW: CANDIDATE -->

- baseline：`0a167159e08bf2df021c0b7f7fc8942a4dc87c3e`；Windows canonical，启动工作树干净。
- 用户已明确启动实施；[已定设计](../../../architecture/compute-next-stage.md)与
  [入口审计](compute-entry-design.md)是本轮输入，不恢复已关闭旧任务。
- owner：compute、graph、compute_backend_cpu、compute_backend_cuda、compute_provider_native、
  compute_provider_onednn、compute_ggml_runtime；compute_route_cpu/cuda/apple/vulkan、
  graph_route_cpu/cuda/apple/vulkan。共享 build/package seam 没有独立 registry 身份。
- reopen trigger：用户新增唯一路线集合/完整自动 Provider、统一运行资产、预编译 recipe 和
  后续跨路线支持的公开交付需求；E1/E2/E3/E4 的具名入口反例已在上轮源码审计确认。

## 正常旅程与边界

普通 C++ consumer 只选产品和路线集合，源码/静态/动态取得同一完整能力。
主纵切：目标模型 Direct/Graph 的 CPU/GPU 单路线及 mixed SDK；正交纵切：模型无关 tensor 批处理。
依赖继续使用中央 component DAG、依赖 lock/cache、package registry 与 native adapters。
保留已验数学、原生 Context/Program/Queue/资源/完成 owner；源码引导和 runtime 资产不增第二套实现。

C1 先统一构建输入，C2 统一 SDK/资产，R1 校准 identity/冷诊断，接线结束后 C3 一个稳定候选批验。
H1/H2/H3 联合计算须另卡绑定实际 owner；本卡不删除 same-Device guard，不签 mixed 执行。
2026-10-09 用户已接受 scope 调整：H1–H3 追加有界异步传输/依赖/真实可重叠执行、引擎驱动
的 expert 权重片段迁移机制；同步 run 仍保留，不把异步整体 DEFER。expert cache/预测/淘汰/
模型放置策略和多 GPU 不做。新 H 卡实施前重新冻结 native 进展/槽/取消关闭的有限 FIX 与验收。
增量依赖图继续 DEFER。SIMD、模型 loader/engine、算子扩张、新后台 runtime、公共治理/计时核心都不改。

## C1 AUDITED：有限改动集

| ID | 源码问题/真实旅程 | 允许 FIX | 验收 |
|---|---|---|---|
| C1-01 | 六公开能力开关重复且可裁掉完整 CPU | 一个 route-set 解析器；平台合法性；ordinary 旧参数明确迁移错误；必要 Provider 正常变量派生 | 默认/单项/组合/空/未知/重复/平台错配/旧 cache/父工程输入 |
| C1-02 | 未选 GPU 仍在静态产品根，靠 OPTIONAL_ABSENT 收尾；CPU 默认独立 | 从中央声明过滤实际所选 route 根并取闭包；维护 exact factor 与 TEST_DEPS 不混入用户产品 | 单路线不 probe/fetch/link 未选 GPU/CPU 数学；语义 metadata projection 不受配置依赖 |
| C1-03 | Apple/Vulkan 无对称普通 preset | 四 preset 同一规则，旧 Provider AUTO/六 route knobs 撤出普通 preset | preset metadata、同 parser，低并行，无普通 lite mode |
| C1-04 | 现行 build fixture/selector 参数仍使用旧公开选项 | 仅迁移消费者配置输入/对应合同断言；内部隔离用既有 DEV_FACTORS；不改仪器/selector 算法 | 必要消费 fixture、路由关闭/缺件/未知诊断与工具合同 |

API/usability freeze：`HYLE_COMPUTE_ROUTES=cpu|cuda|apple|vulkan` 或显式集合，顺序无 preference；
省略 Compute/Graph 产品时不探计算依赖，省略路线时完整 CPU；显式空/未知/AUTO 拒绝，大小写归一并去重。
normal full CPU 的 Torch 必须满足已锁 OS/ABI；oneDNN 仅在已准入 x86 平台取得，ARM 不强开。
旧 CMake 开关拒绝，target 派生的 presence 宏保留；cache/root/archive 仅为获取方式。
硬件 CUDA 编译目标/Toolkit/driver 分开；选择新路线不继承不同软件 recipe 的 exact WIN。

## C2/R1 范围（尚待各自有限审计）

C2：复用 generic/Apple helpers，提供一个应用 staging 入口；现 package registry 投影路线/recipe；
installed 请求只验证实际 SDK 子集，不能重编或伪宏；DLL exe 同路径、metallib/Unix loader 及搬迁。
R1：保留 matcher/profile/all-port 语义、现有 mature defaults；检查实际 recipe 与 native facts/冷诊断，
submit 无新增搜索/设备查询/计时；未知环境保持合法默认，缺真实能力拒绝，不偷偷 CPU fallback。
每子包审计后才能固定其产品 FIX；C1 的 AUDITED 不自动授权 C2/R1 新源码/API。

### C2 AUDITED：一个应用资产入口和真实 SDK metadata

baseline 为 C1 产品 `f01b349a5`。源码反例已核实：generic helper 只 stage DLL/loader path；
Apple helper 另行获取 metallib，并不检查 app 是否真正链接 MLX；安装 metadata 只有 features、
没有所选路线/实际软件 recipe 可查询值。现有 runtime graph 已处理 alias/条件边/diamond/cycle，复用它。

| FIX | 有限产品动作 | 验收 |
|---|---|---|
| C2-01 | 唯一公开 `hyle_stage_runtime(app)`；原 helpers 私有化；同一 target walk 收集有 owner 的 MLX 资产，保留条件表达式/producer | CPU/Core 不 stage 未链接资产；MLX source/install；重复调用、条件边/多 app、DLL 更新与 exe 邻置 |
| C2-02 | 从实际 features/已登记锁/编译 recipe 投影 installed 路线与 fingerprint；source 采用相同值；显式路线需求只验子集，不修改 SDK | source/static/shared 查询一致；unknown/缺路线/旧参数拒绝；metadata 不被称 ABI/SOTA 证书 |
| C2-03 | 迁移现行 CMake/用户 helper 调用；原 source/install asset graph、下载器和许可证不复制 | selected source/SDK consumer、搬迁/缺件/离线仍在 C3 批验 |

正常三交付仍只链接公开 products；多路线聚合 target 会需要其实际全闭包，不能从代码只调用 CPU
推断 Apple runtime 可删除。target-owned resource property 是既有资产闭包的投影，不新建依赖 registry。
本包改变 cold CMake/API 名，不改数学、运行资源或 Provider preference；P0，无 P1。

工具接缝 TT1（独立有界维护项）：新增 route metadata test 和旧 non-package sample/correctness
CMake 调用迁移后，selector 对九个此前无映射的路径拒绝 verdict。只在 policy 登记该测试及八个
实际 shared build 输入，保留全 package/授权门；不改 selector/checker 算法，不将样例当工具以跳过交付。
WORKTREE 累积基线的已提交 owner attribution 不能代表最终 candidate，正式 C3 使用不可变 commit。

### R1 AUDITED：保留已有冷绑定，校准新构建入口

有限源码扫描结论：没有证据要求改写 matcher、policy 表或 native 执行器。

| 项 | 实际接缝/依据 | 裁决 |
|---|---|---|
| Direct/Graph 默认一致 | CPU 两 create_context 共享 default_policy；CUDA Graph 调同一 internal factory，仅 scope/region carrier 不同 | KEEP 完整 region 优先，不让普通用户排序 |
| 编译 recipe 失配 | Native default_rule_build 查 compiler/flags/linkage/IPO/CRT/parent options；CUDA CMake 与 best_of 检查实际硬件/driver/toolkit/进程条件 | KEEP 精确 WIN 域；C2 fingerprint 只诊断，不成为新证书 |
| mixed ggml 来源 | runtime 一个选定 core；CPU-only/含 GPU pin 和 port identity 已区分 | KEEP 合法成熟默认，不继承另一个 recipe 的 WIN |
| model/格式/环境匹配 | 公共 route_policy/workload；Apple route_input、Vulkan matches_cold_rule 从实际 Program descriptors/environment/profile 构造 | KEEP 实际 ports/native admission；不增加全模型 Q4 枚举 |
| 冷诊断 | PreparedOperation implementation/route_policy/workload；Graph route reports；native Program environment/workload | KEEP 原查询，补统一构建与 recipe 解释；不为统一报告强制改变 native 对象 |
| hot/失败边界 | 本批 include/C++/shader 原文零 diff，原 prepared/run state 与完成路径不改 | KEEP submit 固定、不计时/探设备；任何 accepted error 不换设备重算 |

R1 有限 FIX：当前使用文档将新 SDK route/recipe 查询与已有 implementation/policy/native reports 对接，
解释 maturity default 与精确 WIN、存放与执行分离；C3 在新闭包运行 ordinary Direct/Graph 接缝，
验证未选 CPU 无隐式 fallback。无需为“必须重构”制造没有反例的 C++ 改动。
原旧性能例外/INCONCLUSIVE 不重写，旧规则与源身份相同时不重跑 P1。

### C3 入口与候选冻结

实际三交付 consumer 使用现 `samples/package/compute/cpp`、`graph/cpp` 的数学/FFN journey，
不新增 copy-only 演示。旧 selected-component smoke 动态拼 CPU/CUDA/oneDNN 旧参数，本次仅迁移
为一个 `HYLE_SMOKE_COMPUTE_ROUTES`，保留原三交付、缺头/许可/运行资产/搬迁断言及低并行行为。
不新写另一套发布工具，不把当前仪器无映射/旧参数失败当产品数学 FAIL。
归档基线 superset；执行直接影响的 Compute/Graph 和 Core runtime staging/包接缝，
未改其他模块数学/算法而不适用的完整 correctness 逐项保留原因。

本机实际环境：Windows MSVC/Toolkit 13.3 和 RTX4070；WSL Ubuntu-24.04 x86_64/GCC14，
可见 RTX4070/driver617.14；MacBook Air Mac16,12/arm64/macOS27.0.1/AppleClang21/CMake4.3.4，
Metal 使用 setup 中既有签名组件/shim，默认 CLT 的 xcrun 不代表组件不存在。
三验证机尚未为本候选签验收；Air 约40GiB可用，每 producer 串行、有界清理，不积累重复源码/SDK。

## 预算、验证和停止

- 审计一遍当前入口/源码和现有 fixture；一个配置边界最多一次定向修正，不翻历史性能队列。
- C1 接线只跑无重型 build 的解析/配置合同、docs/closure/diff；数学未改不跑 P1。
- C1/C2/R1 完成后冻结不可变候选；C3 批量四单路线与三 CPU+GPU SDK 的适用平台 source/static/shared、
  public headers、runtime assets、selected consumers、迁移/缺 SDK/离线/OFF 隔离。
- 现有数学/资源证据仅在 recipe/tree/合同身份相同时复用；变化域才跑直接消费者，不能称新数学 PASS。
- 所选 shared-seam selector superset 保留；非相关模块无产品/边界影响项逐项说明，不扩成全库数学重验。
- build 2 / test 1；重型验证串行；WSL/MacBook Air 精确验证副本，不反向同步；只维护本轮有界产物。
- 完成 C1/C2/R1 和 C3 适用门后 CANDIDATE→CLOSED；预算耗尽或实际条件不足按具体恢复条件 BLOCKED。
  适用未运行保持未验，聚合门 INCONCLUSIVE；不能用文档/空 target 关闭。

## 七问与当前裁决

第三方身份/许可证与算法不升级，沿现 lock/registry；完整性是公开配置/交付的新需求；
复用现有图/依赖/资产 owner；P0 只在冷 CMake/准备路径，既有 hot 成本不变；没有 C/FFI 新入口；
三交付在 C3 批量实测；user/developer 当前用法随实际产品迁移，不提前称可用。
当前 C1-01–04 已 AUDITED；C2/R1 仍须各自子包审计，产品改动/新验收为零。

### 审计固定的接法

- 元数据 projection 的 CMake script 不 probe 平台/依赖；实际 project 才解析产品路线。
  保留中央 factor/product 声明和静态 graph adapter，不为新 helper 改 selector 核心。
- 普通产品过滤未选 route/backend/CPU Provider 根后按原 DAG 取闭包；无 Compute 产品不探 SDK。
  tests 的 CPU oracle 是 TEST_DEPS，不成为 shipping GPU fallback。
- DEV_FACTORS 的能力由最终选中 factor/test 闭包和宿主适用性派生；其 optional OFF header fixture
  不冒充正常已选路线。selector 不再传重复 compute feature cache，保留原平台验收规则；
  只迁移 policy 和对应断言，不新增列表 merge/状态规则/仪器行为。
- 普通完整 CPU 当前真实可承载 Windows/Linux x86_64、Darwin aarch64；其它 CPU ABI 若缺
  必要 original ATen SDK 明确拒绝，不 silently lite。CUDA/Apple/Vulkan 保持各 native SDK admission。
- 必要 Provider ON/OFF 仅为内部正常变量，由唯一解析路径派生；不再声明公开 cache option。
- 现行 sample/合同 fixture 的参数随实际语义迁移；历史 performance contract 仍绑定原不可变候选，
  不迁移其 recipe、改历史证书或重跑 P1；新性能任务须新版本合同。
- 有界解析测试和既有 configure fixture 对正/反入口取证；C1/C2/R1 不逐项启动完整 native run。

### C1 接线与部分验证

已实现唯一 route-set、所选产品根过滤、完整必要依赖私有派生、四 preset、旧参数迁移错误；
维护 factor 自动原生适用性，selector policy 撤掉重复 Compute cache，核心算法不变；现行 source
samples/Apple parity/MASM fixture 和用户配置说明同步迁移。历史性能合同不修改。
`python tests/tooling/compute_routes_test.py` 实际 8 tests 通过，含边界/父工程与 cache 迁移、
canonical set、平台错误、维护 OFF headers、三条实际 manifest configure 闭包及四 preset。
这是配置证据，未运行数学或 SDK；不能将 LANGUAGES NONE metadata fixture 当 native 路线完成。
`python tests/tooling/validation_selector_test.py` 实际 60 tests 通过（99.857 秒）；
docs/closure/diff 检查通过。原 component-contract 与平台三交付仍留 C3 批验，不写 PASS。

## 固定候选与 C3

产品 candidate：`321bb5bd8ff5743a439111c49e7d75d4e8dae394`。
C1/C2/R1 接线完成；C3 一条 Graph producer journey 同时签实际 Direct/Graph 消费者，
每平台/路线/交付共享 producer，不分别构建两套 SDK。当前只运行配置与工具检查，三交付尚未签。
Compute+Graph 统一 LOC 实际仍为 62,486 code lines / 268 files（`.scratch/compute-unified/loc.json`）；
include/C++/shader 零 diff，没有新增自研数学。

### C3 定向修正 B1：发行默认测试

独立 review 的 B1 是真实正常使用缺陷：裸设置路线/安装仍继承顶层 tests ON。
候选按协议退回 AUDITED；有限 FIX 仅为：显式产品路线或 install 请求的默认 tests OFF，
维护 preset/用户明确 tests ON 继续有效，显式 tests ON + install 的冲突仍明确拒绝。
不改 test instrumentation ABI、组件边、数学/flags/Provider 规则。增加真实裸配置取证。
已运行 source/共享批次均明确 tests OFF，其产物/数学没有这个默认差异；保留已完成格，不因
文档或此默认改动从头重跑全部旧 gate，最终新候选对受影响配置及既有格做明确身份/适用性记录。

B1产品 `79137d31` 仅改变未显式给tests时的默认值。实际root Core配置三格 PASS：
裸 install默认OFF、显式路线但无Compute产品默认OFF、明确维护tests ON保留ON。
当前候选另有 route/SDK元数据9 tests、runtime资产图测试、docs/closure/diff PASS；
独立路由/构建/全文档review的有限发现已处理，详见
[review卡](../project/compute-routing-build-doc-review.md)。C3继续原已固定显式OFF批次，没有P1。
最终产品候选更新为 `79137d31`（registry保存完整SHA）；只有默认参数行为变化，显式OFF批次
继续保留自己的实际recipe/产品身份，最终归档区分原候选产物与新增配置取证，不伪造同tree复用。

Windows CPU C3已实际完成：source/shared/static × Direct/Graph 六项正常数学消费者 PASS，
共享/静态 SDK 搬迁、DLL EXE邻置及 shared export 检查通过。证据见
[C3部分收据](../../../../evidence/correctness/compute-unified-c3/README.md)。
CUDA进入下一个串行批次；其余平台/路线/mixed验收与H1–H3仍未签，模块继续CANDIDATE。

同一 Windows CPU relocated shared/static SDK 另完成六项路线需求/recipe配置检查，以及八项离线/
缺件配置检查：离线缓存命中接纳；离线缓存缺失、显式SDK根无效、原版ggml运行库缺失均按具体诊断拒绝。
这些只签配置/交付边界，不签其它路线数学或新性能，原source/shared/static产物身份继续按收据分别保留。

Windows CUDA 源码嵌入的真实 Direct/Graph FFN 两项已 PASS，实际 RTX4070、Toolkit13.3；
所需DLL均在EXE邻侧，未构建HYLE CPU backend或oneDNN。原批次继续shared/static，不重跑source。
最终不可变候选79137d31的selector计划READY，逐commit产品授权无issue；全共享接缝superset保留，
不把计划或跳过项写成执行PASS。新增适用Torch CPU平台/版本元数据测试已实际通过，不签模型数学。

C3已约定平台/路线展开为[15组交付矩阵](../../../../evidence/correctness/compute-unified-c3/delivery-matrix.json)：
Windows/WSL各cpu、cuda、vulkan、cpu+cuda、cpu+vulkan；Air为cpu、apple、vulkan、cpu+apple、cpu+vulkan。
每组source/shared/static共45格，每格普通Direct/Graph两消费者。当前12项消费者PASS、78项UNRUN；
mixed SDK仅验证其两条native路线分别可用，不代签H1–H3联合、异步或专家搬运。

### C3 发现 C2-01 条件 producer 接缝缺陷（待定向修正）

已在真实CMake构建图夹具复现：普通application的`BOOL:0`链接边正确不复制metallib，
但统一helper把未求值的source owner收集成无条件producer，`add_dependencies(stage, producers)`
仍生成源资产。见[反例收据](../../../../evidence/correctness/compute-unified-c3/apple-conditional-producer/receipt.json)。
夹具只签构建图事实，不签Metal、Provider或数学；此前LANGUAGES NONE/imported-executable夹具
被CMake拒绝后改为真实CXX application，本结论来自成功配置和实际构建，不来自失败夹具。

有限修正范围仍在C2-01：source producer依赖保留真实条件，正分支/跨目录/原生generator顺序仍正确；
不改数学、路由、flags、公共staging入口或治理核心。当前Windows CUDA不可变批次继续，结束后
先按协议将本卡实际owner退回AUDITED再改helper、冻结新candidate。保留不受Apple条件影响的
已完成CPU/CUDA格及其原身份；不重启已通过数学或P1。Apple实际三交付在修正后首次验证。

Windows CUDA relocated shared SDK 的Direct/Graph FFN两项也已实际PASS，DLL邻EXE；
shared export hygiene检查18个HYLE库/985个导出通过。共四项CUDA消费者已归档；
原批次进入static，不重跑source/shared。数学/路由/原生进展未改变，没有P1。

C2-01有限修正接法已在[可丢弃helper副本](../../../../evidence/correctness/compute-unified-c3/apple-conditional-producer/design-experiment/receipt.json)
通过Windows Ninja和Air Unix Makefiles正/反分支、跨目录library/资产producer构建图检查：
collector保留producer条件，stage使用求值后的`DEPENDS`，撤掉无条件producer边。
仅修改副本，当前产品/候选未变；不签native MLX数学。当前static原批次完成后按上述范围正式修正。

现有`runtime_staging_graph_test.py`已补producer条件断言；当前helper在Debug conditional分支
实际失败：producer为`n10`而应为空，[失败记录](../../../../evidence/correctness/compute-unified-c3/apple-conditional-producer/metadata-regression-before-fix.json)
明确保留，不称PASS。此测试补缺复现既有C2-01反例，修正后只重跑本接缝适用回归和实际Apple交付。

### C2-01 定向返修 AUDITED

Windows CUDA原不可变79137d31批次已exit0：source/shared/static × Direct/Graph六项真实FFN
消费者全PASS，日志/实际SDK元数据已归档。此时没有本批native进程；CPU/CUDA完成格保留实际身份。

返修baseline为79137d31，仍绑定本卡15个当前owner，仅恢复已声明C2-01的条件producer合同。
固定FIX两项：collector为非imported资产owner保留原链接条件；stage将条件producer放进`DEPENDS`，
撤去无条件`add_dependencies(stage, producers)`。不改公共API、数学、路由/flags或仪器。
验收只选已有元数据回归、已经固定的两generator正反/跨目录构建图，以及后续实际Apple三交付；
非Apple已通过格因不触及此路径保留，不重建、不跑P1。实现后立即冻结新candidate并继续C3剩余矩阵。

AUDITED有限FIX已应用：仅修改两helper的producer条件和stage依赖声明，条件正确传递；
现有Debug/Release条件、alias/diamond/cycle/reset元数据回归实际PASS；Windows新目录中
直接复制产品helper的真实library正/反分支及跨目录资产生成/复制实际PASS。
未改include/C++/shader、Provider或路由；Air同产品helper的Makefiles复核及Apple native交付继续C3。

新产品candidate为`2614676ee64bceddadcf791c218d6eea1f051c64`；前一79137d31的CPU/CUDA已完成格
保留其实际产物身份。新diff只影响Apple source producer的条件边，CPU/CUDA所用无Apple资产路径
未变；不把旧产物改标新tree，也不重跑其数学。剩余矩阵从新candidate开始，source archive重新绑定。

Air新目录直接复制产品helper的Unix Makefiles正/反分支和跨目录producer实际PASS，
[应用后收据](../../../../evidence/correctness/compute-unified-c3/apple-conditional-producer/product-fix-receipt.json)保存真实产品SHA。
修正后的完整source-only archive已在WSL/Air新`source-2614676e`目录展开，两端SHA256匹配。
Windows Vulkan-only普通三交付进入下一个串行批次；没有启动P1或并行重型平台构建。

Windows Vulkan source实际构建完成但两消费者FAIL，[原始失败和有界诊断](../../../../evidence/correctness/compute-unified-c3/windows-vulkan-failure/receipt.json)
已保留。Graph报告为5节点/2native/1managed，样例仍固定旧4native；数值失败仅在rows16，
首输出GPU=-24.757810592651367、原F32 oracle=-24.76485824584961。只把down激活舍入F16的
独立模拟恰好得到GPU值；禁用coopmat2或全部coopmat也不改善，不能归结为设备波动。
下一步只审计原成熟F32完整精度路径的接法及旧样例断言，不改误差门、不新增自研算子/用户开关。
若需要改已CLOSED的ggml Vulkan Provider，先以此具体correctness反例绑定新owner卡SCOPED/AUDITED。
现有677个source构建产物保留，定向修正后恢复对应阶段，不能重跑破坏root的整批driver。

定向F32返修的Windows普通三交付/范围/30项资源和dtype回归均已实际通过，见当前
[F32返修卡](../compute/vulkan-f32-matmul-admission-repair.md)，其跨平台尚待C3；不回写旧FAIL。
下一串行批次已进入Air Apple-only：实际Mac16,12/Apple M4/macOS27.0.1/AppleClang21，
完整validation源码c8874688（产品c96c7c6e加test/docs后续）hash核对后部署；source Direct/Graph
两个真实消费者已PASS，shared/static仍未签，[部分收据](../../../../evidence/correctness/compute-unified-c3/air-apple-partial.json)。
没有使用Mac mini或并行重型平台build。C3当前20消费者PASS、70项UNRUN，不签H1–H3。

Air Apple-only原批次已exit0：source/shared/static × Direct/Graph六项消费者全PASS，
真实SDK的mlx.metallib与runtime闭包、搬迁和原生MLX/ggml-Metal数学旅程通过；四项新增SDK日志
和实际HyleConfig已归档。C3当前24消费者PASS、66项UNRUN。
下一串行Air Vulkan-only三交付已启动，使用同一精确源码snapshot，未增用户额外route/Provider开关。

### C3 当前批次（2026-10-10）

Windows CPU/CUDA/Vulkan、Air Apple/Vulkan/CPU+Apple 六组普通三交付已通过，当前消费者矩阵为
36 PASS / 54 UNRUN。记录保留各自候选身份，不把后续文档提交当新数学资格。
Air CPU+Apple 首次配置发现 DS4 共享 lock 重复登记；已按
[有限登记修正卡](compute-mixed-shared-license-registration.md)修正并冻结产品 `530e230bb`。
[原失败](../../../../evidence/correctness/compute-unified-c3/air-mixed-license-failure/receipt.json)
保留，精确完整源码 snapshot `7a7b4c3b` 在原目录恢复配置后实际 PASS，
[当前收据](../../../../evidence/correctness/compute-unified-c3/air-cpu-apple-partial.json)已签 source/shared/static × Direct/Graph 六项真实计算 PASS，SDK 原循环 exit0。
本组搬迁及普通资产闭包已通过，build2/test1；不重跑单路线、P1 或删除原 build root。
H1–H3 包括已接受的异步传输/重叠/专家片段机制，仍未实施；mixed SDK 不代签这些能力。
统一脚本在不可变 `6e7af97eb` 测得 Compute+Graph 62,508 实现行（不含 tests/third_party），
self-check PASS；本次登记修正只有 CMake，不改变该实现行数。

Windows CPU+CUDA 普通 source Direct/Graph 两项真实计算现已 PASS（0.89s/0.72s），
[收据](../../../../evidence/correctness/compute-unified-c3/windows-cpu-cuda-partial.json)绑定启动时的
不可变 validation source `b7115b114`。C3 当前38 PASS / 52 UNRUN；原 driver 串行继续 shared/static，
保留 source 对象，不重启、不跑 P1。组合包仍不代签 H1 联合/异步/专家传输。

### C3 当前资格纠正：组合 SDK 工具缺陷

上述 Air CPU+Apple 六项中的四项 SDK 资格判断错误：实际 metadata 只有 CPU，原数学日志
确实 PASS，但只签 CPU 数学。四项组合资格撤回为 UNRUN；两个 source 消费者仍真实通过。
Windows CPU+CUDA shared 也在此处正确报缺 CUDA 后停止，原产物/CPU objects 保留。
当前有效 C3 为34 PASS / 56 UNRUN；历史计数不代表当前资格。
具名反例与有限修正见[独立工具卡](compute-delivery-route-argv-repair.md)和
[原判断/失败收据](../../../../evidence/correctness/compute-unified-c3/mixed-sdk-route-failure/receipt.json)。
只修参数边界和全路线 SDK guard，原目录恢复 shared/static，不重跑 source/单路线/P1。

工具修正 ed2dd3c3 后，Windows 原 shared SDK 完整 CPU+CUDA 两项真实数学已 PASS，
配置和安装 route metadata 均为 `cpu;cuda`，DLL 邻 EXE；C3 当前36 PASS / 54 UNRUN。
同一原循环继续 static，Air 两个错误组合 SDK 尚待串行恢复，四项资格继续未验。

Windows CPU+CUDA 原 static Direct/Graph 也已真实 PASS，完整路线与 DLL 邻置核对通过，
原 SDK 循环 exit0。该组 source/shared/static 六项完成；C3 当前38 PASS / 52 UNRUN。
下一串行 Air CPU+Apple shared/static 已从原目录恢复，新的配置/启动参数均完整 `cpu;apple`，
其四项新数学尚未签。source 两项保留，不以新工具 fixture 或配置通过替代数学。

Air 完整 CPU+Apple shared 两项现已真实 PASS（Graph20.28s/Direct14.78s），精确 route-set 已核对；
新产物证据独立保存，原错误 CPU-only SDK metadata/数学/判断不覆盖。C3 当前40 PASS / 50 UNRUN。
Air static 原批次继续，H1–H3仍未实施。统一 LOC 在 `5e94f4cd` 实测62,508 code lines，self-check PASS。

Air 完整 CPU+Apple static 两项也已真实 PASS，原 SDK 循环 exit0；C3当前42 PASS / 48 UNRUN。
参数转发与完整 SDK guard 的独立工具卡已 CLOSED，两组合 SDK 原失败全部按实际完整路线修正。
先前四份 CPU-only consumer 记录仍是真实 CPU 数学，不能继续冒充组合；后续只在实际完整CPU
软件/recipe/合同身份匹配时归入 CPU 证据，避免重建同语义 SDK，不将旧判断当混合资格。

Air CPU 单路线现在已完成三交付：新 source Direct0.40s/Graph4.23s 实际 PASS；shared/static
只做当前完整配置，route/feature/recipe 与原 CPU-only SDK 各自逐项精确一致。7a7b4c3b→ed2dd3c3
全产品 include/src/lock/build 输入零变化（仅验证 driver/test/docs变），实际机器/编译器/SDK/数学合同
相同，因此复用原四项 CPU 数学，不将原产物改标成新tree，也不重建相同 SDK。
[CPU收据](../../../../evidence/correctness/compute-unified-c3/air-cpu-partial.json)保存新source、
旧实际SDK身份与新配置匹配证明；C3当前48 PASS / 42 UNRUN。下一串行 Air CPU+Vulkan 新批次。

Air CPU+Vulkan source 普通 Direct0.63s/Graph6.93s 两项真实数学现已 PASS，
[组合收据](../../../../evidence/correctness/compute-unified-c3/air-cpu-vulkan-partial.json)保存实际完整
源码/路线身份；shared配置的route-set也精确 `cpu;vulkan`，原循环继续shared/static，尚未签。
C3当前50 PASS / 40 UNRUN，不签H3联合/异步。Windows已完成CPU+CUDA两个SDK producer object
目录清理4,350,873,870 bytes（4.05GiB），两relocated SDK、source、所有consumer EXE/assets、
machine cache和已核对SHA256的日志均保留，未删除产品源码或重启任何native批次。

Air CPU+Vulkan 原 shared SDK 两项现也真实 PASS（Graph7.26s/Direct1.39s），精确 `cpu;vulkan`
metadata、搬迁和原生资产闭包核对通过。C3 当前52 PASS / 38 UNRUN；原 static 循环继续，
H3联合/异步/专家传输未实施。不因本轮文档或状态记录重编已通过阶段。

Air CPU+Vulkan static 两消费者也已真实 PASS，精确完整路线/原生资产/搬迁检查通过，原循环exit0。
Air五组cpu/apple/vulkan/cpu+apple/cpu+vulkan三交付全部完成；C3当前54 PASS / 36 UNRUN。
下一串行 Windows CPU+Vulkan 新批次，然后WSL剩余组；其Vulkan实际GPU未资格仍按机器证据区分，
不能用软件设备签GPU。H1–H3仍未开始；完成平台SDK不代签联合执行。

Windows CPU+Vulkan 新 source Direct0.52s/Graph0.76s 两项真实数学已 PASS，
[收据](../../../../evidence/correctness/compute-unified-c3/windows-cpu-vulkan-partial.json)保留启动时的
不可变 source `2b60fa15` 与工具/产品身份。C3 当前56 PASS / 34 UNRUN；原 shared/static
循环继续，不重启 source root、不跑 P1，仍不签 H3 联合/异步/专家传输。

Windows CPU+Vulkan relocated shared Direct1.17s/Graph1.20s 两项真实数学也已 PASS，
metadata 精确 `cpu;vulkan`、DLL 邻 EXE、搬迁与闭包 guard 通过。C3当前58 PASS / 32 UNRUN；
同一原循环继续 static，源码/shared 不重跑，H3仍未实施。

Windows CPU+Vulkan static Direct1.15s/Graph1.03s 也已完成真实数学及搬移后资产验证，原批次退出0。
C3 当前60 PASS / 30 UNRUN；Windows与Air声明的五组均完成三交付。下一批为WSL，
H1–H3联合/异步/专家迁移仍未实施；不重跑已通过source/shared或P1。

C3 的组件合同发现最小源码夹具漏带路线解析文件，按
[独立工具卡](compute-component-fixture-repair.md)只迁移两个夹具；原失败保留。
本工具修改不改变 native 产品、recipe 或已通过的交付，WSL 原批次继续。

WSL CPU source Direct 已 PASS；Graph 在创建 invocation 时无法装载原生 CPU 程序，
按[Linux loader 修正卡](compute-ggml-cpu-linux-loader-path.md)定向退回实际 owner。
原 objects/失败日志保留，不重启 source root；不将 loader 失败写成数值或性能失败。

限定 Linux loader 修正已冻结 `a2476b62`；原 WSL CPU source 两消费者无 LD 覆盖均 PASS，
MODULE 缺失/恢复也已验证。C3 当前62 PASS / 28 UNRUN；原 shared/static 剩余阶段继续。
组件合同的两个原夹具断言已通过；后半段因启动时未传本机 Vulkan SDK 环境而失败，
原失败另存，补正确环境后只恢复尚未通过的后半段，不重跑前段或全库数学。

组件合同尾段又发现一个 metadata child 缺真实 target identity，独立工具卡按具体反例
追加唯一夹具纠正；原 Sched/32-bit/目标身份断言保留。最终全部原合同已分段 PASS/exit0，
失败与成功段独立归档，工具卡 CLOSED。C3 平台矩阵仍62/90，下一串行 WSL CPU SDK剩余循环。

已安装 Windows CPU+Vulkan 与 CUDA shared SDK 的全部 Compute/Graph 普通、backend、Provider
目录逐头检查，复用原 `scripts/header-self-contained.py`；实际10组/63份头均 PASS，
[收据](../../../../evidence/correctness/compute-unified-c3/windows-installed-headers/receipt.json)
绑定实际 SDK config 与 compiler。此项只签 bare-header self-containment，不代签其它平台、
静态宏档位或未运行的数学；SDK未重建，无P1，C3消费者计数仍62/90。

实际 MacBook Air 的 CPU+Apple/CPU+Vulkan 安装头检查共8组/60份 bare headers PASS；
又用已安装 `hyle::graph` 的真实 usage requirements 独立编译60份头，compile command 确认
Apple/Vulkan 路线宏实际启用，MLX/Vulkan 公开声明与依赖可见。只编译头，不重建 SDK/运行GPU/P1。
[Air 证据](../../../../evidence/correctness/compute-unified-c3/air-installed-headers/active-receipt.json)
保留两种检查的边界，不能以未启用宏的空声明替代 active 资格；三交付样例继续保留原独立证据。

C3 [验收范围对账](../../../../evidence/correctness/compute-unified-c3/gate-scope.md)汇集当前实际证据
与剩余域，保留 selector superset/未运行区别；不再因文档/工具局部修正重跑无影响数学。
WSL Vulkan 的 SDK 编译仍适用，真实 GPU 数学保持未资格，不能用 llvmpipe 代签。

Windows 安装 target 的实际路线宏逐头资格也已完成：CPU+Vulkan32份、CUDA31份均 PASS，
compile commands 确认真实 GPU/CPU presence 宏启用，原 export target 自动带入 SDK依赖。
[激活头收据](../../../../evidence/correctness/compute-unified-c3/windows-installed-headers/active-receipt.json)
不代签静态宏/WSL/数学。已完成 Windows CPU+Vulkan 的两个 SDK producer object目录经
六PASS/exit0、日志SHA、路径/链接及无活动进程检查后清理，释放4,464,806,780字节；
两个 relocated SDK、全部消费者、source、cache和证据保留。没有触碰活动WSL目录。

WSL CPU relocated shared Graph0.17s/Direct0.15s 均实际 PASS，原程序 MODULE 在 SDK lib下，
共享 runtime 自身 `$ORIGIN` 查找与17库/1354导出检查通过，无 LD覆盖。
[资源收据](../../../../evidence/correctness/compute-unified-c3/wsl-cpu/shared-resource-receipt.json)
与各 CTest/config 绑定；C3 当前64 PASS / 26 UNRUN。原静态循环继续，source/shared不重跑。

WSL 已安装 CPU SDK 的30份公开头以原脚本 bare 编译 PASS；另以真实 `hyle::graph` usage
requirements/GCC14.2.0 单头 OBJECT 编译30份均 PASS，实际CPU/oneDNN宏启用。
[逐头证据](../../../../evidence/correctness/compute-unified-c3/wsl-cpu-installed-headers/active-gcc14-receipt.json)
保留第一次误用系统GCC13的额外诊断，未用其签GCC14门；只校正头 consumer，不重建SDK/数学/P1。
其它WSL路线的安装头门须待对应包真实完成，静态宏档位不由此虚签。实现LOC无变化。

原 WSL CPU static Graph/Direct 也已 PASS，原 SDK 循环exit0；static 主程序的实际 ELF
查找含 relocated SDK/lib，原程序 MODULE 保留，未设置LD覆盖。
[静态资源证据](../../../../evidence/correctness/compute-unified-c3/wsl-cpu/static-resource-receipt.json)
与原 source/shared 合成六项三交付资格。C3 当前66 PASS / 24 UNRUN；下一串行 WSL CUDA
已使用冻结a2476b62来源启动，CPU/source/shared/static不再重跑；H1–H3仍未实施。
