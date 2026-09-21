# M2-B1 Vulkan：成熟张量、归一化与状态的批量接入

## Closure card

- component / owner：`compute_provider_ggml_vulkan`、`compute_route_vulkan`、
  `graph_route_vulkan`；`/root`。当前阶段 AUDITED；`b9478142`已独立完成SCOPED。
  精度增量另绑定`compute_provider_iree_vulkan`为AUDITED；`1a4ac820`已独立完成SCOPED，
  原M1关闭事实保留。
- baseline：`09d2cf99f1298e988ccda65ca76ec1b4907bd0e7`，Windows main，工作树干净。
- trigger：[总计划](llm-moe-coverage-completion-plan.md) M2-Vulkan 和
  [M0 清单](llm-moe-provider-integration-ledger.md) B1 的真实新消费域，符合
  [M1-V terminal](llm-moe-m1-vulkan-admission.md)的 M2 新 shape/数值/片段重开条件。
  当前 `region_supports` 仅接受有限 dense rank≤4、主要 F32 域；不能承载完整 B1。
- 需求：矩阵 F01.1–3、F02.1–4、F03.1–4、F04.1–6、F19.1–4，共21行。
  首批实现检查点不是缩小分母；尚未接通、语义不相容和未验收分别记录。
- 主纵切：模型 embedding→norm/gate→residual/index/state；正交纵切：无模型标签的
  多维广播、非连续输入、指定轴归约与发布中间结果。两请求独立、输入更新后重放、旧结果保活。
- scope：原版 ggml Vulkan 算子/合法组合的批量 schema 映射、普通 Direct/Graph、
  原生资源与失败、冷绑定和成本、适用三交付。复用已验收 managed region/native bridge。
- non-goals：冻结自研 CPU/CUDA/Metal/Vulkan 数学的调查、扩张、优化、比较和删除；
  SIMD 基础；新 shader/算子算法；模型 loader/engine；公共 selector/性能仪器改写。
- 依赖/消费者：Compute 公共 schema、既有 ggml native device/graph/work、Mem/Error；
  Graph 单向消费 Compute 的同一 region。共同runtime等其他CLOSED owner保持不变；
  只有具名新缺陷才另行定向审计。IREE原Scan/FP8和runtime补丁不重做；本节后续的精度增量
  按新SCOPED/AUDITED绑定，其它CLOSED owner不变。
- 第三方：ggml `456172ec733a135778adcd32d00e576a58232e45`，MIT；沿用锁和资源 overlay。
  原始缓存只读。核对实际 Vulkan backend 与 shader 域，不从 CPU/CUDA 或同名 API 推断资格。
- 预算：本批源码/设计合计120分钟，自2026-09-21 05:50 UTC起，前置核对计入；
  每具名根因至多两次源码驱动纠正。实现细节调查合并计费，构建和预声明验证单列。
  不按子卡重置预算；遇到成熟覆盖未证明的域先具名裁决，不转换为自研授权。
- 性能：P0 查原版分配/编码、转换/cont/repeat、边界copy与同步；固定候选后一次必要 P1，
  对照同一成熟完整调用，相同输入输出/资源/完成语义，非劣门5%，A/A与双顺序先资格化，
  15对起、合法总上限31对。候选前冻结真实工作负载和成本账单，不比较冻结自研。
- 机器：Windows RTX4070、Air/MoltenVK；WSL/Mini 按实际公共/OFF/host风险。
  parallel2、重型串行、Windows DLL邻置、无UAC/电源改动；同一批四路线推进，
  Vulkan 不再等其它三路线全部完成后启动。
- stop：有限 FIX、21行成熟承载及适用门完成才 CLOSED；预算到期须裁决，未运行不签 PASS。
  M5开始前按用户要求暂停；旧C1/M3-01专门暂停已由用户取消。

## SCOPED 审计边界

| ID | 要裁决的有限改动 | 验收边界 |
|---|---|---|
| V1 | Copy/Cast/Concat、浮点 Binary/Unary、norm/归约、indexed/mask/stable gate 的原版承载；高rank折叠、广播/布局与精度翻译 | 与原版真实后端能力及公共语义一致；失败域明确，不能只以名称接线 |
| V2 | 新片段的metadata、外来buffer/view、对齐/范围/dispatch、暂存和输出root | 准备期不执行用户payload；发布中间值、旧结果、两slot和accepted失败/close；成本透明 |
| V3 | 普通 Direct/Graph 共用能力判断和原版region；默认冷规则与报告 | 用户只选Vulkan，无模型标签同样可用；完整region保留成熟融合，热期不搜索/计时 |
| V4 | 具名测试、真实consumer、指南、source/static/shared与最终P0/P1 | 每批复用相同接缝证据，不逐枚举重复全库或三交付；不造第二套构建机制 |

AUDITED前冻结公开形状、精确首批承载域、资源/错误/成本合同与有限 FIX；本表尚非实施授权。

## AUDITED：首批原版映射与资源形状冻结

2026-09-21 05:56 UTC，源码/设计保守累计10/120分钟，余110分钟。以下为V1–V4的
首批实施边界；其它需求继续留在21行账本中，不能以首批通过关闭整卡。

| 能力族 | 实际源码与覆盖状态 | 有限裁决 |
|---|---|---|
| F01 Copy/Concat | 原pin Vulkan `supports_op` 的CONT/CPY/DUP与`ggml_vk_concat_supported`具备F32原版路径；HYLE现有region未接 | PARTIAL/BORROW：接dense非空F32、rank0..8的Copy及合法轴Concat；Concat首批沿既有最多6输入region边界。stride/empty/其它storage和精确Cast仍未完成 |
| F03 广播/逐元素/门控 | 原版Binary只广播RHS，repeat与cont已有Vulkan实现；现有HYLE只接某个输入已等于输出shape的rank≤4 | PARTIAL/BORROW：双向广播用原版reshape/repeat，保留普通低rank已有直接路径；高rank冷折叠。Unary与SwiGLU只扩大已接原版F32的布局域，特殊值用独立oracle回归 |
| F04 norm/归约 | 原版RMSNorm、GroupNorm中心方差、SumRows/Mean已接；原版row归约无padded-row guard，现有精确tile组合必须保留 | PARTIAL/BORROW：dense高rank折叠；任意选轴用冷stride/shape重排和原版cont，Sum/Mean仍用原版精确tile；keepdim两档恢复公开shape。全mask/其它归约和半精度仍未签 |
| F02/F19索引/状态 | M1原版TopK/checked Scan和资源桥已有有限证据；Gather/Scatter/mask还有原生旧实现和未覆盖域 | PARTIAL；不以旧自研充当新成熟覆盖。已有IREE owner不重开；后续成熟承载须具名核对并依法绑定owner |

### 有限 FIX 与公开形状

- **V1-F**：仅修改ggml region的描述符/原版调用翻译；高rank公开描述保持不变，内部按
  行/列或连续维组折叠到ggml四维，选轴归约使用原版view/cont。借鉴已接CUDA adapter的
  冷布局翻译，不复制CUDA数学或强制共享执行器。不得改shader、上游数学、依赖pin。
- **V2-F**：按新增布局节点数量有界预留metadata；检查native计数、buffer与实际dispatch域。
  新恢复shape的输出可能是ggml view，必须绑定并标记真正producer root，防止allocator/fusion
  吞掉发布中间值。原版图、work、fence、import与每slot资源仍用现有owner。
- **V3-F**：普通Direct/Graph共用region判定；新增F32域默认走原版ggml，报告如实更新。
  既有非适用域保持原行为，不改冻结数学；prepare不运行用户payload，submit不重新搜索。
- **V4-F**：直接入口覆盖高rank双向广播、非相邻轴Sum/Mean、Copy/Concat；Graph用
  L2组合和归一化/残差片段发布中间值，两个slot、变更输入和保留旧结果；加类型/容量拒绝。
  完整卡的跨平台/三交付/P1仍按下节执行，开发测试不代签。

公开API/对象layout/组件边保持；模型与硬件/driver事实仍由既有冷绑定合同拥有。
新增原版repeat/cont可能有GPU暂存和dispatch；初始对齐copy、原版managed编码与分配完整计费，
不承诺新零分配/零copy或SOTA。只复用现有资源适配，不扩张冻结tensor shader的支持域。

## 预声明验证

- 全21行能力账本与主/正交 Direct/Graph 旅程；独立 scalar/bit oracle、特殊值、
  scalar/rank6/8、正stride、广播、tail/empty、非相邻归约轴、低方差归一化。
- 按实际新增域验证 dtype/layout/overflow/alias拒绝，动态index/scan失败不发布结果；
  两slot、变更payload、旧结果、部分构造/OOM/accepted失败/drain/close。
- Windows/Air 真GPU的 owning factor与直接消费者；适用host ASan/UBSan及WSL/Mini/OFF。
  host插桩不代签driver；不因未装可选validation layer伪造PASS。
- 稳定批次的一份组合consumer覆盖Windows/Air source/static/shared、prefix搬移、
  producer隐藏、原生资源/许可与DLL邻置。M1已有资源/交付事实只按实际未变输入复用。
- 运行change-impact selector选中门并保留skip原因；文档/状态只运行轻量门，
  closure明确previous-ref、docs和diff检查。候选后单次与实施分离的复核。
- 当前七门均未最终裁决；没有新增完整矩阵PASS。启动LOC为56,908行/249实现文件，
  使用统一脚本，排除tests/vendor；该数字只描述规模。

## V1-F–V4-F：Windows/Air 开发检查点

2026-09-21，源码/设计保守累计30/120分钟；构建和预声明验证单列。首批有限FIX已落地：
dense非空F32 Copy/Concat、rank0..8双向广播和既有norm/unary/gate布局扩展、任意选轴Sum/
Mean及keepdim两档，均映射原版ggml操作。高rank冷折叠与必要repeat/cont不改变公开描述；
保留归约精确tile，新增metadata有界预留与实际buffer/grid检查。输出reshape view绑定并保护
真实producer，发布中间结果不再依赖最后一个view是否被原厂allocator保留。

普通Direct/Graph共用绑定判断；测试显式确认走成熟region，独立scalar/FP64 oracle覆盖
双侧广播、非相邻轴、rank6/8和513行尾部，Copy检查特殊值原始位，Concat覆盖六个轴。
Graph的Copy→平方→归约→eps→Rsqrt→Mul组成一个原版region，发布中间归约和最终结果；
两slot、变输入重放、旧结果保活、owner关闭后下载及quota归零通过。没有新增shader或自研数学。

产品`9a49fad478a1181d068a6a7ec8d924c82ca8bbf9`，tree
`85b950623e2a60c03c14d970a1f5d7132e0b8e46`。Windows RTX4070 Release/shared/WERROR
的9/9 CTest通过（166项C++通过、1项可选validation-layer跳过）。Air首次Clang/WERROR在新
测试的Operation按值循环处失败，已在Windows改成const引用并提交`4a7058979d1fe7593de4a8bf8eaae6421d4a6a01`；
没有产品改动。Windows对应109项复验通过，其余原有制品hash保持；Air/MoltenVK Release/
static/WERROR的8/8 CTest通过（167项C++通过、同一可选项跳过）。原始失败、两机身份与
完整输出见[开发证据](../../../../evidence/correctness/compute-llm-moe-m2-b1-vulkan/README.md)。

本检查点不关闭21行B1。其余dtype、非连续/empty、mask/index/state等成熟承载仍待接线；
适用WSL/Mini/OFF、host插桩、新增域source/static/shared和最终selected复核仍待批次稳定。
原版repeat/cont、managed分配编码和边界copy/sync完整计费；未运行P1，不比较冻结自研。
完整聚合INCONCLUSIVE，三个owner保持AUDITED；新完整覆盖0。统一LOC为**57,086行/249实现
文件**，不含tests/vendor，self-check通过。每机保留一个活动build复用，没有另建重复验证树。

## V1-P：精度增量 SCOPED

baseline `88c61331`；owner仍为`/root`，额外将`compute_provider_iree_vulkan`从CLOSED绑定
至SCOPED。触发为F01.3和F03/F04的本地2B BF16普通计算需求，以及锁定ggml原版转换的
具体反例，不改旧M1验收结论。源码/设计保守累计50/120分钟；本增量合用余70分钟，不另开预算。

原版Windows GPU资格：F16/BF16分别65,537项原始位Copy和转F32全通过；451,837个包含全部
相邻有限值中点/邻值、随机F32及NaN/Inf的输入中，F32→F16全通过，F32→BF16有4项NaN
分类错误（包括`0x7f800001`）。原因对应原pin `types.glsl:fp32_to_bf16`缺少NaN分支；不得
把原厂名字或有限模型数据当成公共Cast全域资格。相同输入交给原版IREE `arith.truncf`→
`arith.bitcast`，451,837项全部通过。原版Vulkan标准转换需要Int16与storage16；首次遗漏Int16
导致SPIR-V store类型不匹配，声明实际所需capability后编译和执行通过，不修改compiler数学。

有限审计项：

- P1：ggml合格的dense非空F16/BF16 Copy、F32/F16/BF16浮点Cast分工；BF16 narrowing交给
  原版IREE，F16/BF16互转保留显式F32精度语义；不扩旧shader，不接未资格化整数转换。
- P2：复用ggml F32 region计算F16/BF16输入，按公开输出精度舍入；BF16公开输出以成熟
  Provider边界保留舍入，不绕过NaN检查或在host计算模型内容。先覆盖当前已接的dense张量族。
- P3：IREE私有程序增加具名转换入口，复用原VM/HAL/queue/完成；单输入转换不得误走Scan的
  overflow flag流程。只新增离线标准算子制品和in-tree身份，不改公共构建或原版runtime。
- P4：实际启用Int16/storage16事实、原生dispatch/bytes域、暂存、metadata和独立请求；
  ordinary Direct/Graph成批验证位模式、中点/NaN、两slot/输入更新/结果保活与混合Provider。

AUDITED前固定具体程序、资源与映射；原始探针仅作来源资格，不计覆盖。Air原版/普通入口及
既定三交付、插桩和必要P1仍按全卡完成，不能以Windows探针或本小节缩小21行分母。

### V1-P AUDITED：有限 FIX

源码/设计累计55/120分钟，余65分钟；P1–P4审计完成。IREE的F32→BF16（451,837样本）与
F16→BF16（全部65,536位模式）已通过Windows原版GPU资格；async-external制品已离线编译。
原版输出是`arith.truncf`的BF16，经标准`arith.bitcast`以i16端口承载原始位，不写舍入算法。

- **P1-F**：ggml新增dense非空rank0..8的F16/BF16 Copy；浮点Cast按合格方向接线，
  BF16→F16显式经过F32。F32/F16→BF16固定使用IREE标准程序，禁止绑定已失格的ggml方向。
- **P2-F**：当前ggml Unary/Binary/SwiGLU/RMSNorm/LayerNorm/Sum/Mean/已接Softmax域，
  按公共F32中间精度用原版Cast提升F16/BF16输入。F16输出在ggml region内舍入；BF16公开
  输出以ggml F32 region→IREE BF16转换的两Provider边界实现，保留每个公开节点的舍入点。
  本增量不扩min/max/Select/index、stride/empty或新融合算法；既定21行需求仍保留。
- **P3-F**：IREE私有program增加两种转换入口和第三个小型离线module，沿用独立VM/请求、
  input/output copy、异步retirement和quota；overflow flag仅对Scan创建，不能按单输入推断。
  不改原版runtime补丁、Provider版本、CMake机制或公共接口；只更新vendored身份与制品。
- **P4-F**：私有ggml native事实补充实际启用的shaderInt16；该pin将查询到的完整core
  features传入DeviceCreateInfo，冷期事实记录不改变启用策略。IREE转换要求Int16/storage16，
  F16入口另要求Float16；已核对生成程序LocalSize64和每workgroup至多64项，按ceil(N/64)、
  F32暂存/descriptor范围及32位索引裁决。其余ggml转换计入现有有界metadata和layout限制。

预声明新增验收：普通Direct/Graph全部16位Copy/扩大精度、F32中点与NaN/Inf/溢出/负零，
高rank低精度广播/norm/门控和混合ggml/IREE，F32 gamma，奇数尾部，两slot和改变payload重放、
结果保活/关闭、资源失败。Windows/Air串行复用活动build；新域完整三交付/插桩/P1仍按整卡。
原版probe通过不代签普通入口；完整矩阵覆盖仍未新增。

### V1-P Air 定向纠正：浮点执行合同

2026-09-21，源码/设计保守累计85/120分钟，余35分钟。Windows候选`eccbd046`相关8门通过；
Air同候选仅新BF16转换的4个NaN边界失败，其余普通精度/Graph/资源用例通过。生成SPIR-V的
NaN判断为原版`FUnordNotEqual(x,x)`，却未声明`SignedZeroInfNanPreserve`。仅在诊断进程关闭
MoltenVK fast-math后，同一个失败用例通过；该环境实验不作为产品PASS。

P3-F/P4-F内固定一次具名纠正：离线使用原版IREE的executable-configurations阶段，保留原版
64-lane调度、全部算术和VM/HAL结构，只为两个转换入口声明标准SPIR-V ABI的32位特殊值保留。
原版IREE/MLIR已生成`SignedZeroInfNanPreserve 32`执行模式。冷期额外核对物理设备的
`shaderSignedZeroInfNanPreserveFloat32`；不修改compiler数学、runtime补丁、ggml shader，
不改变整进程/其它shader的fast-math设置。新增离线复现脚本及依赖身份完整留档；两机按相同
普通入口回归，资源/交付/P1总边界不变。该根因纠正上限两次，仍合用原120分钟预算。

依据为锁定IREE `ConvertToSPIRV.cpp`的既有entry-point ABI保留逻辑、
[MLIR官方lowering](https://mlir.llvm.org/doxygen/LowerABIAttributesPass_8cpp_source.html)、
[Vulkan浮点环境](https://github.khronos.org/Vulkan-Site/spec/latest/appendices/spirvenv.html)及
[MoltenVK配置](https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Configuration_Parameters.md)。
这是同一数值合同的编译元数据纠正，不是另开算子算法或性能调查。

### V1-P：两机开发检查点

产品候选`2191c6160`，源码/设计保守累计95/120分钟，余25分钟。P1-F–P4-F本精度增量
已实施：原版ggml合格Copy/Cast和F32中间计算，原版IREE BF16转换、公开节点舍入边界，
冷期启用feature/实际FloatControls/资源域检查及既有异步完成、配额和资源回收。

Windows Release/shared/WERROR 9/9 CTest通过、171项C++通过；Air/MoltenVK Release/static/
WERROR 8/8通过、172项C++通过。两机同一可选validation-layer测试跳过。完整16位模式、
舍入中点与特殊值、rank6/8和奇数尾部、F32 gamma、两slot/重放/中间结果与close后保活通过。
公开压缩存储和half累加声明的拒绝边界已回归；资源失败按真实首次prepare分配点验收。

Air具名NaN根因用标准执行元数据一次纠正后通过，未全局修改fast-math。首轮编译/接纳边界/
资源预期失败全部保留，数值门没有降低；见[精度证据](../../../../evidence/correctness/compute-llm-moe-m2-b1-vulkan/README.md#v1-p半精度与同设备-provider-组合)。
本检查点仍不关闭21行B1，不新增完整矩阵PASS。其它dtype/stride/empty/index/mask/state成熟
承载、适用WSL/Mini/OFF与host插桩、新域三交付、最终selected综合门和一次P1仍按本卡继续。
`HyleDependencyLock.json`变化导致selector扩展至121个门，开发子集通过不代签其余门。
统一Compute+Graph实现为57,333行/249文件；清理本批多余资格制品约72.5MiB，活动build复用。

### V1-S：存储族接线的有限审计

baseline `6bc3a439`，原四owner保持AUDITED；源码/设计累计102/120分钟，余18分钟。
本增量仍属于V1/V2的F01.2存储翻译，未增加数学、公开API或资源owner。原pin
`ggml-vulkan.cpp:19396`明确支持同类型I32 DUP/CONT/CPY，复用原版按存储位复制；
`ggml_vk_concat_supported:2642`明确接受1/2/4/8字节同类型非量化存储，原版Concat承担合并。
当前HYLE只给浮点Copy和F32 Concat接了这些入口。以下FIX现已冻结：

- S1：在已具备的dense非空、rank0..8和原生资源限制内接I32 Copy，以及F16/BF16/I32 Concat；
  仍使用原版ggml调用，不以float数值Cast搬运整数或half位模式。Concat保留现有最多6输入边界。
- S2：输出非空的Concat在冷翻译时略去轴长为0的输入，不为没有payload的端口创建原厂import。
  输出全空沿既有无GPU工作路径；不以空路径作为成熟计算证明。单个有效输入用原版Dup保证独立
  输出，多个有效输入直接组成Concat链，移除首项无必要的Dup。高rank折叠和producer保护复用。
- S3：Direct四dtype逐位oracle覆盖NaN/负零/整数极值、所有六个轴、混合空输入与不等尾部；
  Graph验证空输入、公开中间值、两slot/重放和保留结果。Windows/Air复用同一活动build串行。

普通入口的新域仍须实测资格；未扩展的stride、其它storage、mask/index和完整交付/P1保留待办，
不缩小21行分母、不扩大冻结数学。最后剩余预算耗尽时必须按本卡停止规则裁决。

### V1-S：两机开发检查点

产品候选`5b45fb764d098712ec59bfd621387b1a303baee6`，tree
`a42ab53fb8d5a225fb3cc135e54cedc1a9520ca4`。S1–S3已实施并通过两机开发门：dense非空
I32 Copy及F16/BF16/I32 Concat复用原版ggml。Concat的空输入只在路线冷翻译时省略，
原版region仍只接受非空native端口；公开bind仍要求全部输入。单个有效输入使用Dup，
多输入直接组成Concat链，移除不必要的首项Dup；没有新增数学或shader。

Direct逐位oracle覆盖四dtype、六个高rank轴、六输入中的混合空端口/单有效输入/全空输出、
NaN载荷/负零/I32极值与513项尾部。Graph的Copy→Concat→单有效输入Concat发布全部三个
结果，验证空input和constant、两slot/重放、旧结果和独立输出、close后下载及quota归零。

Windows Release/shared/WERROR与Air/MoltenVK Release/static/WERROR均8/8 CTest通过，
分别174/175项C++通过；同一个可选validation-layer测试跳过。首次测试编译遗漏span入口的
显式Operation包装，测试纠正后通过，产品候选未因该问题改变。相关selector七门已执行，
125项保留skip/reuse原因；Windows重配前后CMakeCache一致。原生MASM桥本批未变，复用
前检查点验证，不另跑重型组合。见[存储证据](../../../../evidence/correctness/compute-llm-moe-m2-b1-vulkan/README.md#v1-s存储类型与空输入拼接)。

源码/设计保守累计110/120分钟，余10分钟；本批不另开预算。完整21行的其它dtype/stride/
index/mask/state成熟承载、新域三交付/适用插桩及最终综合门/P1仍未完成，聚合INCONCLUSIVE，
四owner保持AUDITED，完整矩阵新增0。未与任何冻结自研比较。统一Compute+Graph为
**57,343实现行/249文件**，排除tests/vendor，self-check通过。两机均复用唯一活动build，
Windows约0.73GiB，无新增build或模型副本。
