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
