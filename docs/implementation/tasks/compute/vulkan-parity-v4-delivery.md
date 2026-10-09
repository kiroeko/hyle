# Vulkan V4：当前候选的平台与三交付

<!-- MODULE REVIEW: AUDITED -->

baseline `d04978bd`；协调owner为当前Vulkan/Graph，build/package是横向交付工作。
[父轮](vulkan-cpu-cuda-apple-parity-closure.md)全部需求保持。V3性能资格条件已获
[明确工程取舍](vulkan-v3-performance-engineering-decision.md#用户明确决定2026-10-09)，本包独立验证，不代签P1。
当前source/环境/旧交付接缝审计已固定以下有限批次，route owner依法进入AUDITED。

有限结果：Windows与Air真实可用设备的source/static/shared与搬迁、DLL/dylib邻置、
内嵌SPIR-V/VMFB、许可证、公开头、offline、OFF不探测/下载其它路线；WSL实际GPU
缺席时只签build/header/拒绝行为，不能以software device冒充GPU数值资格。
两个普通入口执行已审固定模型/量化/状态/失败资源范围。只有新平台具体产品缺陷
才依法另定有限FIX和owner，旧CPU/CUDA/Apple及公共工具不顺手重开。

当前身份（2026-10-09）：Windows canonical/RTX4070资格继承本轮；Air Mac16,12/AppleM4/
16GiB、macOS27.0.1/26A434、AppleClang21.0.0、CMake4.3.4、剩余41GiB；实际MoltenVK1.4.1，
loader1.4.350、device API1.4.334、vendor106b/device1b000209。Air验证clone当前7365f0fb，
树干净；仅从Windows当前完整候选同步，不反向取产品。SDK工具/loader已在Homebrew。
WSL Ubuntu24.04/6.18.40.1/原ext4，尚未确认真实Vulkan GPU；旧llvmpipe不继承为本轮PASS。

预算：交付source审计20分钟，批量三交付每机一producer/两配置，build2/test1，重型串行。
复用锁定sourcecache与有效producer，精确清理本轮拥有者临时包，不删模型/共享cache。
每具名配置/fixture/ABI错误一次纠正；未适用与未运行分别记录，不以旧R6三交付代签。
selector根据真实API/package seam选择门；文档不重建。closure/docs/diff及统一LOC必需。
全部平台/交付及父轮性能取舍未结束前不CLOSED/不声称基础完成。

## AUDITED：一批交付，不恢复算子或性能队列

| FIX/gate | 固定动作及边界 |
|---|---|
| D1 | Windows复用`C:/hv/vp1`作shared producer；一个短路径source-embedding/static producer，compute+graph同批，公开入口与独立host oracle、DLL邻置、SDK搬迁/offline、许可证及内嵌原程序一起验证 |
| D2 | Air只从Windows不可变候选同步，先读实际numeric features；同样source/static/shared一批，用实际MoltenVK执行具名完整consumer。WSL无真实GPU则只签适用编译/公开头/拒绝，软件device不代签数值 |
| D3 | route OFF和独立公开头、默认条件字段/版本及API交付闭包；CPU/CUDA/Apple产品代码和公共统计工具保持各自terminal |
| D4 | 只有此批精确编译/导出/loader/fixture或platform语义反例才记录一个具名纠正，不顺手重写build工具；新成熟owner缺陷须另绑定合法FIX。失败原收据保留 |

已有D5外部consumer源码与其公开test helper闭包可复用，更新原改名的full-MoE filter，
逐实际GTest执行数量检查，不让零case或skip通过。旧D5硬编码producer/候选脚本不续跑。
Windows当前公开产品接缝需移除开发factor选择、关闭test instrumentation，实际重配置一次，
不能凭旧developer测试库签install；相同dependency sourcecache不重复下载。
本包V2数值证明及V3冷规则证明按未变源码复用，三交付consumer仍实际运行；无新的AA/n15。

### W1：真实Werror交付编译残留

`707f769a`shared配置成功，原开发cache为WERROR=OFF；正式WERROR=ON在
`compute_provider_iree_vulkan/native_program.cc:607/609/610`四处IQ1 shape initializer
报C4838/C2220。未运行install或consumer，原日志/失败收据保留。
有限W1只把已经通过positive/INT32 packed/workspace检查的维度显式转换成
`iree_hal_dim_t`，与相邻official/MLX分支相同；不放宽warning，不改索引域、
数学/VMFB/公开合同或资源。该IREE owner保持AUDITED并将本轮record绑定本卡，
协议独立提交后才修改。续编当前producer，三交付差异验证；不重跑V2全矩阵/P1。

W1产品`2f78714c8959c0eeac859381f7b991c75b8bb0a2`：Windows source/static/shared
各14实际测试全PASS、无skip；两个SDK各3个独立ON头和3个独立OFF分支头编译通过。
36份carrier/基础许可各SDK均到位，DLL邻exe、搬迁/隐藏producer与精简PATH通过。
首轮Werror/源码SDK环境FAIL保持，不重跑Windows共同数值全集。

### W2：Air Clang的两处未使用常量捕获

Air本批当前source配置通过，82秒增量前置build在同一IREE私有文件485/497处
报告`-Wunused-lambda-capture`：局部constexpr `limit`被按值捕获，但只作常量表达式，
并未ODR-use。有限W2只删除这两个捕获，保留lambda及原INT32范围方程、状态/资源与
VMFB；不用关闭Werror或改变数值域。owner已是本卡AUDITED，追加FIX独立提交后
实施，复用原producer续编。所有原步骤/FAIL和预算保留；唯一此错误的定向纠正，
不是换目录重新开始。Windows旧数学/冷规则不重跑，受影响SDK按等价冷代码差异复核。

W2续编发现Mac的`size_t`与`uint64_t`虽同宽却是不同C++类型，
`vulkan_iree.cc:84 std::min`无法推导同一类型。一次`-k0`汇集本producer全部剩余
编译，只有此处产品错误，无其它新增错误。有限W3只把已限制到INT32/8的elements
显式存为uint64，与网格上限同型，值/公式/数值程序不变；route owner已AUDITED。
协议先独立提交再改此一行。所有原失败/纠正次数累计保留，不重配置成功的source
producer；收齐错误的既有对象和最终三交付继续复用。

## Windows/Air当前批次完成，V4仍未终局

[当前真实收据及剩余项](../../../../evidence/correctness/vulkan-parity-v4-delivery/current-scope-verdict.json)
绑定产品`b4f21f1c1`：Windows两SDK增量build/install/link和Direct/Graph IQ1差异4项
全通过，复用不变的三交付14项数学/资源证明，不重跑完整V2/P1。Air一个producer/
两配置实际完成source/static/shared各14测试，无skip；两SDK独立ON/OFF分支头各3项
编译通过。Air原14文件fixture逐SHA与Windows相同，真实M4/MoltenVK数值执行不拿
设备枚举或host oracle代替GPU资格。原Werror、调用身份、resume目录和type错误均保留。

本批旧WindowsSDK不是当前输入的无条件证书：W2/W3的冷期常量capture/同宽计数
值保持等价，新的增量SDK实际IQ1重放和坏索引已运行。VMFB、shader数学、冷优选规则、
公开接口均不变；旧CPU/CUDA/Apple产品未改。代码统一统计**62,486行／268文件**，
排除tests/vendor/bench；比本轮前只多3行类型处理，没有新算子。

尚待WSL当前编译/公开头与software-only默认拒绝、OFF配置/artifact隔离差异、
最终source/resource复核、临时产物清理及合法terminal状态；本卡保持AUDITED，
不签V4或整个goal完成。当前WSL只发现llvmpipe CPU设备，真实GPU数值资格缺席，
不能代签PASS。四路线single-primary有限审计已经独立结束，不再启动旧性能队列。

### W4：WSL GCC的两个HAL枚举条件表达式

当前Windows来源快照`6d7beb36`在GCC14.2/WERROR下的Linux配置通过；原build及
一次`-k0`剩余错误收集只发现`native_program.cc:642/681`的`-Wextra`：
`IREE_HAL_ELEMENT_TYPE_SINT_8`枚举与`iree_hal_element_type_t`整数混合条件表达式。
有限W4仅将这两个原常量显式转换为HAL定义类型，值/shape/资源/VMFB保持；不关闭
warning，不改Shader或数学。IREE owner已绑定本卡AUDITED，协议独立提交后修改。
续编同producer；原失败不丢弃。WSL只有CPU软件device，后续只验公开构建与默认
拒绝，不把软件可创建/枚举当真实GPU数值PASS；Win/Air相同值的旧完整数学按source
等价复用，Linux实际编译和拒绝仍必须完成。

W4续编后GCC到达native pipeline，剩余三处相同enum/uint32条件表达式在
`vulkan_pipeline.cc:51/67/339`：descriptor binding/layout/pool的update-after-bind flag。
同producer一次`-k0`收齐其余对象，仅此三处。有限W5只用Vulkan各flag typedef显式
构造原bit常量，rebindable与0分支、原能力门和资源生命周期不变；route owner
保持本卡AUDITED，先提交该精确FIX再改产品。无新handler/queue/kernel或性能采样。
