# F1 CPU构建、软件资格与预编译合同

<!-- MODULE REVIEW: SCOPED -->

- 当前owner：compute_provider_native、compute_ggml_runtime、compute_route_cpu、graph_route_cpu；baseline `167cb5ff6f9edac8a9aaa77c41e6720bd3c377d2`。
- 总合同：[CPU最终收口](cpu-final-closure-plan.md)；用户新增三平台/全重叠域与构建一致性要求构成新的公开交付trigger。
- 已确认源码反例：Native默认WIN宏只查peer，`qualified_default_environment()`只查编译器/Release/硬件，
  不查有效flags/LTO；自定义优化recipe仍能继承旧证书。CPU-only新ggml与mixed旧ggml能力不同；Mac ARM64不提供现ATen接入。
- 主consumer：普通CPU模型计算Direct/Graph及installed SDK；正交consumer：同语义tensor批处理与定长预编译replay。
- 非目标：不改数学、公共Graph动态模型、GPU资格、SIMD、性能统计或closure工具。

## 本包有限审计项

| ID | 问题 | 审计后允许的有限动作 | 验收 |
|---|---|---|---|
| B1 | flags/LTO/相关feature变化仍可能启用旧Native WIN | owning构建投影精确性能recipe；不匹配只撤资格，保持合法成熟默认 | matching/flags/IPO/Debug/多config及peer失配规则投影 |
| B2 | 标准CPU及mixed SDK实际core/格式能力不一致 | 固定同进程runtime/core ABI与能力合同，优先同一CPU能力；不可保持时具体决策，不静默降级 | 单/混合路线配置、能力说明、实际新格式负/正消费 |
| B3 | 预编译SDK与Graph变体容易被误称全部best-of | 区分构建期存在集/冷规则/Invocation尺寸选择，投影真实已编入程序与身份 | safe baseline/ISA、规则失配、变体命中/回退、source/static/shared |
| B4 | 无特化规则时只用全局Provider顺序，默认依据不完整 | 已测优选→逐op/format/precision/resource/region成熟默认→合法兜底；规则低成本冷绑定 | 未知硬件/版本/线程、feature裁剪、完整region与普通叶、错误不伪fallback |
| B5 | 用户要求oneDNN升级后再比较，当前锁v3.13.2且identity硬编码 | 核对官方最新稳定版/实际API与overlay，固定commit/archive/hash；两adapter身份随实际依赖投影，旧WIN不继承 | owning/真实默认两入口、资源/线程/错误、source/static/shared及许可/offline资产 |

scope只授权审计；B2的ABI/源码反证与F2/F3能力冻结后才写该部分产品。
标准`compute-cpu`必须交付固定目标全部合同，未匹配性能特化仍可执行成熟默认；必需依赖缺失configure失败，
Mac的AUTO关闭不是本goal可接受的最终标准能力。高级显式裁剪与完整标准配置分别记录。
构建获取/安装复用现有中央锁与package registry，不另造第二构建系统。普通用户仍只选CPU，不给timing或Provider顺序。
公开对象/ownership/失败/并发/完成不变；提交前owner必须AUDITED且有限FIX固定。

## 预算与验证

一轮本地源码/已归档SDK身份审计、一次实际机器工具链盘点；每ABI疑问最多两个具名现有接缝候选，
不能借本包重签全部GPU。B1一次参数化配置投影检查、匹配旧成本的普通consumer/owning测试；
B2/B3稳定后集中三交付/selected package/相应CPU直接消费者。P0，无本包赢家P1。
docs/state只closure/docs；未运行不签PASS。涉及公共package seam再定向选择，不能扩大治理工具。
完成后CANDIDATE→七门CLOSED；若B2依赖未接齐，保留实际未完成，不能以B1已修关闭本包。

启动时：SCOPED，B1–B4待AUDITED；产品改动0，新验收0。后续子批实际进展见下节。

## B1子批AUDITED：性能recipe不匹配撤资格

Native owner单独完成B1审计，其它owner继续SCOPED；本卡整体仍未冻结B2–B4产品方案。
现两份实际P1都为MSVC19.50、Release `/O2 /Ob2 /DNDEBUG`、portable ISA、static、LTO off；
macro和`qualified_default_environment()`未核对flags/IPO/linkage，构成明确证书误投影。
有限FIX只在Native owning CMake增加recipe条件与Release配置投影，原peer/CPU/线程/端口条件继续；
非matching flags、IPO、shared、sanitizer、exact ISA/trace、相关feature变化撤优选资格，保持原正常数学与成熟默认。
无新API、layout或热字段；冷metadata同条件匹配，不给未测shared标签。当前17/187是历史有证格而非通用冠军。
低成本私有CMake合同检查覆盖正例/失配/多config，正常shared默认two-entry consumer按F1选定验证；
匹配static旧数学与P1复用，不因资格保守缩小重测。B1不算B2–B4已完成。

## B5升级子批SCOPED

用户明确要求升级oneDNN，真实owner compute_provider_onednn、graph_provider_onednn绑定本卡；
原v3.13.2及硬编码identity不能给新候选签资格。只审计一个官方最新稳定候选及当前原threadpool/BRGEMM接缝，
最多一批API/overlay纠正，不转为自研优化或全库升级；新版本先完成接线/正确性/资源/三交付再进入F4。
Native B1资格门继续只按已测peer，升级后不自动携带旧17/187 WIN。此为B5启动SCOPED事实，后续审计见下节。

## B5子批AUDITED：v3.13.4

固定官方稳定版`c531fe1a5cd631b7f60277cdf6675c588cadbc28`、Git tree `65c08c66fb85c5c68f4f04807b906514af420b59`；
codeload archive SHA `ac75d4a771492ff8e5996f0daace71de7d40a23c2527705bb8dbca60a17f7ba9`，
中央算法expanded tree SHA `5c0c833df87c71a31ac82489e56650f0197fa38f3dec85a9988a404ec532fdba`。
11个具名seam文件逐SHA比较：C API、BRGEMM ukernel、threadpool/stream、primitive runtime与LICENSE完全相同；
C++ header只增加verbose查询，options只新增默认OFF的SYCL profiling。原runtime overlay SHA匹配，不修改overlay数学/线程合同。
有限FIX：更新oneDNN唯一锁；CPU身份与version()由实际生成version header投影，Graph引用同一version owner；
BRGEMM编译断言和identity测试更新；Native旧/新core两组WIN都必须对应原oneDNN commit，升级后默认成熟。
owning/onetime默认两入口/失败并发资源、三交付与锁/许可/offline差异集中验证；P0，本包不比较赢家。
compute_provider_onednn、graph_provider_onednn合法AUDITED；runtime/routes的B2–B4仍SCOPED，不借本批升级整个GPU或改公共仪器。

## B1已执行（子批，非F1完成）

产品`adc794aa3384499d23b6c9b054570b017b236fae`，私有CMake合同20格通过；matching static只Release配置启用，
flags/IPO/target IPO/shared/peer feature/sanitizer/ISA/trace/编译器/平台/CRT/parent options失配撤资格。
Windows shared source/public-facade实际24格成熟默认、两次变输入replay、24成熟region、9错配、独立并发与close通过；
声明动态view物化通过。configure35.828秒、build48.516秒，未重跑P1。B2–B5及最终SDK仍待本包验收。
Mac Air已可达：Mac16,12/arm64/macOS27.0.1/AppleClang21/CMake4.3.4，只盘点无远端修改；不代签CPU覆盖。

B5 selector把中央lock文件变化展开为全库superset；原计划保留，逐依赖对账只有oneDNN条目变化。
本包执行oneDNN/Native/CPU Direct/Graph消费者、component/lock与Compute/Graph package；
没有oneDNN链接边、没有实际产品/依赖变化的IO/Container等非CPU目标保留语义不适用原因，
不重编无关模块、不改selector核心、不把未运行superset写PASS。

## B5 Windows子批已执行（非整包关闭）

产品`688d28f15361ed33de1c528fd865377c1742749c`，oneDNN3.13.4集中低并行build454.672秒，
原四组owning有一项旧Native MatMul后备断言失败，原记录保留；未发现新版数学失败。
测试`d2ffda17793cfef84eb9be2a29652aee8eb198f5`改为实际ATen转置MatMul两次变输入；
显式裁剪时要求拒绝，不能复活自研。修正build1.875秒，四组owning、24格普通Direct/Graph、
动态物化/并发/close均通过。没有P1；Linux/macOS与最终安装SDK仍待验。
归档见[F1部分证据](../../../../evidence/correctness/cpu-final-f1/README.md)，不能解释为F1完成。
统一Compute+Graph实现61,084行/249文件，不含tests/vendor/bench，新增自研数学0。

## B4子批AUDITED：具名成熟默认

有限FIX：CPU route拥有静态工程默认表，普通Direct/Graph的成熟leaf在成功冷proposal后投影policy/preference；
规则不绕过原provider的dtype/layout/precision/resource准入，不因Unknown硬件失去可用实现。
plain MatMul与norm以oneDNN为首选，原packed/原生tensor程序以ggml为首选；
稳定激活、布尔/LogSoftmax等以现有已验ATen数学为首选；RoPE与checked scatter保留已验ggml程序与prepared重放。
替代候选仍按真实资格可用，
唯一承载不制造赢家标签，完整Graph region仍先于leaf。F4允许按实际比较修订表，当前全部为工程默认、非性能WIN。

只装饰成功的cold propose，保留原ProviderIdentity/Options/create/destroy/prepare/drain/extension与原PreparedVTable，
submit不经过装饰器。静态同build vtable副本共享原trivial/owned配置语义；Core私有ProviderAccess仅增加保留配置的重绑定操作，
公开Provider/Prepared对象与合同version14不变，无额外Context allocation、hot字段或逐kernel回调。
compute、compute_route_cpu、graph_route_cpu按此有限FIX AUDITED；其它owner不因本批进入实施。
验收：normal/未知模型、三种mature角色、feature缺席、失败/OOM/close、expert仍无默认policy、Direct/Graph一致、
完整region与原执行/资源沿用；原24格批量consumer更新为工程默认policy有效且Native旧WIN=0。
两入口/公开头与source/static/shared在本包集中验证，P0，无新赢家P1。

## B2子批AUDITED：同core原IQ1端口，不引入双core

有界反证：旧456172ec core已有前八种新增GGUF原类型，但adapter受CPU-only宏整组拒绝；
三个IQ1_XS/XXS/XXXS原port的19处新接缝全部能在旧core唯一匹配。探针保留旧资源patch、
真实MatMul/Gather运行三种原格式且不同row内容/scale通过；旧/port后tensor=336、plan=56，
既有F32/F16/BF16/IQ1_S/M类型数值不变，COUNT43→67。

有限FIX：混合SDK保持已锁旧core/GPU源码，精确合并原IQ1 CPU扩展与旧资源manifest；
不复制数学body，派生manifest由已有两个已登记输入生成并逐before/after SHA核验。
CPU-only仍用现0.26；有CPU的两类SDK均投影完整原格式映射，GPU新格式拒绝不解除。
独立CPU程序文件名统一含core/实际patch身份，避免旧ABI size相同而误装旧无IQ1 DLL；实际CPU版本标签含port身份。
许可证/原source身份/程序/derived manifest进入安装与offline闭包，CPU OFF不引入CPU port。
prepare_source工具/公共治理/Provider接口/CPU数学/旧GPU算法不改；差异是合法CPU承载与新枚举CPU类型。
F32 flash优化仍只新版core，旧core用已验同精度数学，不把它冒称新版融合。F4分别裁决实际成本。

compute_ggml_runtime、compute_provider_ggml绑定此有限B2；普通CPU+实际CUDA/Vulkan/Apple组合的配置、
CPU新格式/模型分支、GPU原格式和新格式拒绝、三交付适用验收待运行，探针不算完整路线资格。
保留一个core/原GPU能力，旧Graph Vulkan性能blocker不解除，不启动GPU性能或自研优化。

## B4 Windows已验／B2组合待验

B4候选`6216e6d0651ad2684950abec036d8749ccc3c3a1`：Direct16/Graph15全部通过、fixture已绑定无skip；
原24格两次重放、24成熟region、9错配、动态物化/并发/close通过。首次不必要的managed Scatter/RoPE选择
已按真实prepared资源合同纠正，失败记录保留，未放宽数值/资源门。静态cold表无新增Context allocation，submit沿原表。
证据见[F1部分证据](../../../../evidence/correctness/cpu-final-f1/README.md)，不签性能WIN或整包完成。
B2产品与派生19处原port已固定、源码/CPU MatMul/Gather探针通过，真实CPU+Vulkan标准产品组合正在验证；
首两次configure的maintenance-root缓存及notice分发类型错误保留，分别清除互斥旧root与改为SDK说明后重试。
没有修改公共package checker来绕过锁，原实际许可证仍按中央锁安装。

标准mixed配置暴露运行库collector按路径指数遍历，另开[有限工具修复](../build/runtime-staging-diamond-walk-v1.md)。
只对同target/condition去重，条件/循环/独立collect保留；原本轮运行与中断不改PASS，不扩性能统计/治理。

## B2 Windows真实组合已验（部分）

CPU+Vulkan标准shared/source候选`afba74122c26d49489bee895c8b14dc5381426d2`集中build611.562秒，
原CPU格式owning、默认Direct16项、24格Direct/Graph与成熟region、变输入重放、动态物化/并发/close通过。
原GPU九项回归通过，新CPU格式仍不取得GPU资格；新增负空间测试待运行。
[混合子批证据](../../../../evidence/correctness/cpu-final-f1/README.md)保留原配置错误与工具缺陷中断。
其它混合路线/平台和installed static/shared未验，F1不关闭。

### B2具名负空间纠正已审计

候选6359804a新增测试实际发现：Vulkan的native_quantized_type已拒绝十一CPU新格式，
但quantized_weight_domain/grouped只核对公共canonical布局，绕过原格式映射并在proposal错误准入。
真实owner compute_provider_ggml_vulkan仍AUDITED，当前C3卡已授权这项原GPU格式不扩张的有限拒绝修复。
有限FIX仅在两个domain检查同一native_quantized门；扩现具名负测试为普通/分组两种矩阵。
原算法、支持旧格式与state/API不改变；一批compile、十一格式拒绝与原GPU九项回归，不跑P1、不重开GPU队列。
原负测试FAIL完整保留；改后成功不能回写该旧结果。

纠正候选`b2ad8084b4fcc996301e45426413817e62c27b8f`单目标build4.375秒，具名格式拒绝与原GPU回归通过。
全fixture25项：24 PASS、0 FAIL、1显式validation-layer实例skip保留未验；不签该skip或完整GPU资格。
证据归入F1部分页，混合SDK其它平台/交付仍未签完成。

## F1剩余B2/B3有限验收批

baseline `1207968c5`，Windows工作树干净，F3已关闭；本批不借此重开F3产品或恢复GPU性能队列。
源码对账CUDA/Metal原tensor-domain先native格式映射后要求plain/合法native packed，
新增CPU格式不能绕过，当前没有产品FIX证据。增加两组负空间test而非mock provider：
十一新增CPU格式的MatMul/GroupedMatMul必须在GPU原边界明确拒绝，已有原GPU数学/完成/错误回归同批保留。

| 剩余项 | 本批动作/证据 |
|---|---|
| CPU+CUDA | 复用现C:/hv/m2b1-cuda，清互斥旧DEV_FACTORS，标准CPU+CUDA配置；集中编译CPU普通两入口/格式、原ggml CUDA，原格式数学与新格式拒绝 |
| CPU+Apple | Windows不可变source前向Air；启用CPU+Apple原MLX/Metal，CPU新格式两入口/原Metal计算/新格式拒绝；不以设备枚举签路线 |
| mixed交付 | 源码及同SDK的static/shared两CPU消费者/40格；GPU本批原能力只做适用delta/资源装载，不重测P1或全算子 |
| B3预编译 | 复用F3现CPU三交付与native CPUprogram实际身份/安全ISA gate；对旧mixed core重复40格，绑定actual software/core/patch，unknown规则不带旧WIN |
| 清理/停止 | 每共享producer每形态一次，j2/tests j1；原失败/skip保持，不顺手重写工具。完整适用格齐后候选/七门关闭F1 |

不新增自研数学、不比较赢家、不升级GPU源码/驱动/Toolkit或任意模型。若实际混合暴露具体产品缺陷，
先记录精确owner并重新AUDITED有限FIX，现GPU CLOSED/BLOCKED不自动解除。

## B2/B3 Windows CPU+CUDA源码批已验

源码候选 `e0388fbda7fec39e98bee0757f5ab4f91f1cc625`，测试纠正候选
`b041e792194ec1d1fd3367855d58d14562e6b11e`：初次1253.719秒构建仅因新增GPU拒绝测试
漏传描述符prepare的显式resource而失败，原失败保留；以真实CUDA resource纠正后，依赖不重编，
增量29步18.703秒通过。

CPU owning fixture、Direct20/Graph19、普通两入口40个数值格、原GPU数学与十一格式的两种矩阵拒绝4项、
CPU原程序/安全ISA/工作区/独立槽/取消与原IQ1编码6项、Graph有限变体/回退/并发及模型冷资格8项全部通过，
均无skip。原资源与原数学保留，没有GPU新格式资格或性能WIN。
[收据与原失败/纠正日志](../../../../evidence/correctness/cpu-final-f1/README.md)已归档；
这只是shared源码组合与B3定向检查，installed static/shared和Apple组合仍待，不关闭F1。

Windows CPU+CUDA installed三交付子批也已完成：同producer的shared/static源码消费者各40格，
搬移后安装消费者各同40格均通过；执行PATH仅系统目录，Torch/c10与具名旧core原CPU程序DLL邻exe。
15个configure/build/install/consumer/export步骤均成功，shared实际出口检查通过；
shared增量124.594秒、static增量192.500秒，原ggml GPU和oneDNN依赖复用。
[安装收据/日志与实际runtime布局](../../../../evidence/correctness/cpu-final-f1/README.md)已归档。
CPU+Apple的不可变Windows archive `e0388fbda`已在Air配置通过并开始j2集中编译；
Apple三交付及Vulkan安装形态仍未签PASS，本包不关闭。

## B2/B3 Apple三交付已验及Linux混合loader尾项

Air候选 `e0388fbda7fec39e98bee0757f5ab4f91f1cc625` 完成27步：CPU owning fixture、Direct16/Graph15、
原Metal数学/十一CPU格式拒绝5项、CPU原程序与ISA/资源6项、Graph变体/冷资格8项无fail/skip。
源码、shared/static源码及两搬移安装消费者各40格通过；实际loader与shared出口通过。
初编130.598秒、shared增量13.927秒、static增量73.534秒，没有重跑P1。
[原始收据与完整日志](../../../../evidence/correctness/cpu-final-f1/README.md)绑定完整Windows archive/SHA。
当前HEAD与该archive的include/src/cmake/third_party无产品差异；后续CUDA测试调用修正不影响Apple。

F1终审还需具名验证Linux混合旧core的实际loader，而不是用CPU-only新版core的安装结果代签。
沿原Linux CUDA工具链/同机GPU与锁定cache，只增加一批CPU+CUDA的源码/搬移static/shared40格和
原CPU程序资源/ISA检查；GPU原算法不升级，不做GPU P1、不重新签全模型GPU覆盖。
这是B2/B3原交付风险的具体平台尾项，产品FIX=0；由不可变Windows完整archive前向ext4验证，
j2、重型串行，成功producer复用。未知特化不带旧WIN，未验不能关闭F1。
