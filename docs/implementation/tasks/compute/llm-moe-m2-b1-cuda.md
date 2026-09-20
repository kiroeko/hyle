# M2-B1 CUDA：成熟 Provider 的张量、归一化与状态接入

## Closure card

- component / owner：`compute_provider_ggml_cuda`、`graph_provider_ggml_cuda`；`/root`。
- baseline：`c9744a45b9f6aa2fab27d857af70410f04685518`；Windows canonical，工作树干净。
- 当前阶段：AUDITED；`be44efa5` 已独立完成 SCOPED，以下 C1–C4 及本节精确域冻结后实施。
- trigger：[M0 接入清单](llm-moe-provider-integration-ledger.md)的 B1 新批量片段需求，符合
  [M1-C terminal](llm-moe-m1-cuda-admission.md)的 reopen trigger。现有原版 CUDA adapter
  只接有限 F32 rank2 RMSNorm/Add 及 attention，不能承载张量、门控、索引和状态的完整需求。
- 需求真相：[64 行矩阵](llm-moe-coverage-matrix.json)的 F01.1–3、F02.1–4、F03.1–4、
  F04.1–6、F19.1–4，共21行；本卡不缩小需求，不把部分映射或旧 CLOSED 计为完成。
- 主纵切：目标模型 embedding→norm/gate→residual/index/state；同 GPU 原版成熟程序，
  两请求独立、只读常量共享、状态与旧结果保活，payload 改变后重复执行。
- 正交纵切：无模型注解的广播/归约/转换图，带非连续布局、动态错误、关闭与多输出。
- scope：成熟原版 operation/组合的批量映射、普通 CUDA Direct/Graph、原生资源与失败、
  固定冷绑定、适用交付和完整接入成本。Graph 只译 region 边界，Compute 不依赖 Graph。
- non-goals：所有冻结自研算子的扩张、调查、优化、比较或删除；新数学 kernel/shader；
  模型 loader/服务器；公共 selector/性能仪器或全库资源系统改写；oneDNN 旧 Q2 恢复。
- 直接依赖：复用已验收的 Compute/schema、CUDA backend、Graph region、Mem/Error 及
  `compute_ggml_runtime` 的原版 ggml CUDA vendor/资源桥。既有普通 CUDA route 为直接消费者。
  这些 CLOSED owner 不因本卡自动重开；具名新缺陷才另行定向审计。
- 来源：沿用 M0/M1 锁定 ggml `456172ec733a135778adcd32d00e576a58232e45`、MIT、
  已验收资源补丁；每个新增映射查原版后端真实域，不能只看同名公开函数。
- 预算：本批 CUDA 源码/设计总120分钟，自2026-09-20 20:10 UTC起，前置核对保守计10分钟，
  余110分钟。每具名根因最多两次源码驱动纠正；实现细节调查合并计费，不按子卡续杯。
  构建及预声明验证单独记录。CPU 的原120分钟预算和 oneDNN 待决项保持原状。
- 性能：先 P0；API/算法固定的不可变候选才运行一次必要 P1，与同一原版成熟完整调用相比，
  非劣门5%，含转换、同步、融合和实际完成。A/A/双顺序先资格化，15对起，合法总上限31对。
  不比较冻结自研；具体 workload/账单在候选前固定，不在实现阶段计时找答案。
- 机器：Windows/WSL RTX4070 SM89，重型串行、parallel2、原生编译 `89-real`；Air/Mini CUDA-OFF。
  不声称其它架构已有资格。Windows DLL 邻置，不出现 UAC、可见 helper 或电源策略修改。
- stop：全部21行与适用门完成后关闭；未支持/数值不相容先记录原版反例和成熟后备责任，
  不能变为自研授权。预算耗尽须作明确裁决。M5 前按用户指示暂停。

## 待审有限改动集

| ID | 允许边界（AUDITED 前不得实施） | 必须验证 |
|---|---|---|
| C1 | 复用既有私有 Recipe，成批接原版 Copy/Cast/Concat、浮点广播/Unary、norm/归约及合格 indexed 域；只作描述符/端口与原版组合翻译 | 原版真实 dtype/layout/容量/数值域，独立 oracle，拒绝上游不接受或语义不等价的域 |
| C2 | 每 invocation 的既有原生图/stream/buffer/event、必要中间值、绑定与检查 | 不执行准备期用户状态，不漏检查，失败不伪造完成；两 slot、旧结果、OOM/部分构造、回收 |
| C3 | Graph 共用 C1 的能力判断和同一程序；成熟融合、发布中间值及错误/effect 边界 | 普通 Graph 真实组合，资源/生命周期与 Direct 一致；无模型标签也可用 |
| C4 | 测试、普通消费者、指南、三交付与成本记录；仅所属 factor 的必要源文件描述 | 不新造 CMake 依赖机制；Windows/WSL 及适用插桩、OFF、交付、候选复核/P1 |

## 冻结前必须回答

1. rank0..6、正 stride/空值、广播、动态索引和两输出如何映射；ggml rank4 不能当作公共语义上限。
2. F32/F16/BF16 转换舍入与特殊值、stable unary/gate、L2 epsilon、norm-before-gate、全遮罩
   Softmax、重复 Scatter、checked I32 Scan、TopK ties/NaN 的确切成熟承载与拒绝域。
3. 原生操作能否冷验证；动态索引/溢出检查不能以性能为由省略，也不能退回隐藏 CPU 模型计算。
4. 同一 invocation 的外来 buffer/view 与原版内部 storage 所有权、异步失败/销毁、常量上传及
   managed replay 成本；只复用已验收 bridge，不擅自改变 stream/capture 或 backend 合同。
5. 已有默认工厂能否直接消费新增域；模型/硬件/版本条件准备时绑定，执行不搜索或竞速。

## 预声明验证与状态

- 全21行能力账本、主/正交 public journey、独立数值 oracle；Direct 与 Graph 使用同合同输入。
- 边界/类型/shape/stride/alias/索引/溢出拒绝、两 slot、旧结果、OOM/部分构造、失败/关闭；
  不变资源桥的既有身份匹配证据可复用，新操作风险须补实际验证。
- Windows/WSL 真 GPU 的受影响 factor/消费者；host ASan/UBSan 不代签 GPU device 安全；
  Air/Mini 执行 selector 适用 CUDA-OFF，不制造无 CUDA 的 native 旅程。
- 最终稳定程序用一份真实组合 consumer 覆盖 Windows/WSL source/static/shared、prefix 搬移、
  producer 隐藏、依赖/许可证/导出与 DLL 邻置；不逐算子重跑同一 package 接缝。
- selector 选中门及 docs/closure，七门与一次独立候选复核；文档/状态不构建或跑 P1。
- 当前 scope/architecture/correctness/delivery/usability_docs/red_team/engineering_performance
  均未最终裁决；不存在新增完整覆盖 PASS，矩阵仍0/256。

本轮沿用已有公开 CUDA 工厂；oneDNN新能力已按用户2026-09-21决定进入独立scope，
旧Q2资格FAIL保留，不由本卡代签其资格或验收。

## AUDITED：批量承载与精确域

2026-09-20 UTC 核对原 pin 的 `ggml-cuda/{ggml-cuda,unary,binbcast,cpy,concat,norm,sumrows,softmax}.cu`
及已验收 HYLE bridge/Recipe。源码预算累计25分钟、余95分钟；这是同一120分钟预算。

| 需求族 | 原版事实与当前覆盖 | 有限裁决 |
|---|---|---|
| F01 布局/复制/转换 | CUDA CPY 支持 F32/F16/BF16 相互转换；Concat 按 storage 位宽复制；公开 rank 高于原生4维 | PARTIAL/BORROW：dense 高 rank 冷折叠，原生可表示的 aligned stride 用 cont；最终 dense 输出。非浮点与 checked index Cast 仍未接 |
| F03 广播/Unary | binbcast 原版 Add/Sub/Mul/Div/Repeat；Unary F32/F16，BF16 不直接进入该 kernel | PARTIAL/BORROW：半精度先原版转F32、最终一次cast；用原版 reshape/repeat 译完整合法广播；首批 Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh |
| F04 norm/归约 | RMSNorm/F32 sum_rows/mean、reshape/view/cont 已有后端；CUDA norm 使用二阶矩相减 | PARTIAL/BORROW：RMS 原版+affine；LayerNorm 用 mean→sub→sqr→mean→epsilon→sqrt/div 原版两遍组合，再affine和最终cast；任意归约轴先冷布局翻译 |
| F02/部分 F03–F04 | 动态索引、重复Scatter、checked Scan、stable TopK、Select/min/max、stable gate、masked Softmax尚无本轮同合同资格 | PARTIAL；不冒用CPU原版结论，也不以已有自研填新覆盖；下一成熟承载必须具名核对，任何新owner先合法定界 |
| F19 生命周期 | 已有不可变Recipe、每slot原生backend/stream、非owning boundary buffer、两event、真实Completion | PARTIAL/FIX：新增scalar constants在冷期上传；输出reshape的实际root也要绑定外来buffer并标记output，防止view重绑覆盖或融合吞掉发布中间值 |

首批输出/输入须 plain F32/F16/BF16，dense 支持公共 rank0..8；非dense先限原生rank≤4且stride
按元素对齐。空张量、不可折叠高rank stride、奇数字节stride、其它storage与剩余算子仍属于21行
的未完成部分，不削分母。公开 schema 先推导并核对输出，检查元素/byte/原版索引容量；原生每个
中间节点再次核对后端和实际launch维度，不能把 `supports_op` 泛化为所有内部数值域正确。

普通 Direct/Graph 工厂已有 ggml 优先绑定，不需修改其 CLOSED owner。保持 SM89 资格与 M1-R
冷模型/环境事实；未知条件不扩大证书。新算子没有自己的公开factory、runtime或queue。

资源/API freeze：沿用现有public工厂和私有Recipe；每个slot创建一次图/metadata/backend/scratch，
提交只更新外来地址、view依赖及既有两event桥。新增常量由slot独立持有并冷上传，不在prepare执行
用户数据；FP32转换、repeat/cont及原版managed维护如实收费，无法确知的copy/repack/私有字节不写0。
Graph合并合法pure节点，发布的view及其root都标output；attention仍保留已验收完整leaf，动态检查
节点未证明可融合前保留边界。没有新增数学kernel、CPU计算fallback、依赖pin或公共owner变化。

C1–C4 的首批实现由上述精确域约束；未绑定域和全部适用验证继续留在本卡，不按一批提交签收B1。
预声明差分增加：rank6/8双向广播、非相邻轴归约、scalar/strided输入、三种浮点的转换/特殊值，
LayerNorm大偏移常量/低方差行，以及一张发布中间值的多节点图在两个slot重放。

### C2 定向细化：真实 slice 边界（实施前冻结）

首轮 Windows 真机的新数学/资源/Graph 测试通过，普通 Graph 回归暴露旧 capture 消费者没有在
编译期声明 prepared-only。迁移到既有 `ProviderReplayPolicy::kRequirePrepared` 后保留其原
capture 验收；普通无限制入口另验 managed ggml，既有 owner/API 无需改动。

接着两个旧普通 Direct 消费者提供了真实反例：Copy/Concat 接受元素对齐的 slice，却被继承的
256字节端口要求在 submit 拒绝。原版 `cpy.cu:180–245` 和 `concat.cu` 使用各storage类型的
scalar load/store，转置tile只对shared tile要求更高对齐；无需256字节外来地址。C2 因此固定：

- Copy/Cast/Concat 的输入和输出最小对齐取相应storage类型；其余数学/attention仍保持256。
- Graph 对每个boundary value合并所有原版读/写节点要求，取最严格值；发布的复制结果若随后
  被数学节点使用，仍须数学对齐，不能只检查最后一个writer。Compute Recipe应用完全相同的规则。
- 不改资源bridge，不热搜索或添加copy；不可变Program保存每value已算好的对齐要求，submit
  保留原生指针/offset检查。新增两个slice Direct回归及一张slice→Copy→Unary的原版region。
- 这是一项具名资源根因的第一次修正；不降低数值/capture门，不解冻任何旧数学。源码/设计
  累计56分钟、余64分钟；普通编译和正确性单列。全部21行与后续平台/交付/P1仍未完成。

## Windows 首批开发检查点（AUDITED，非整卡验收）

[开发 receipt](../../../../evidence/correctness/compute-llm-moe-m2-b1-cuda/development-windows.json)
与压缩原始日志保存本批实现、所有失败及修正。Windows RTX4070 SM89、MSVC Release shared，
parallel2 构建、串行测试；最终 **8/8 CTest、42 次 GoogleTest 执行**通过，含4项逐头检查、
两个 Provider factor、普通 Direct/Graph 及既有 capture 消费者。DLL 均邻置，无 UAC/电源策略改动。

- 已实现：本卡首批 F32/F16/BF16 Copy/Cast/Concat、双向广播、7种 Unary、affine RMSNorm、
  原版两遍组合 LayerNorm、任意选轴 Sum/Mean；普通入口和共享原版 region 程序已接通。
- 已验证：rank6/8 广播、scalar/strided 及非相邻轴归约、半精度转换、LayerNorm 偏移/低方差，
  发布 reshape 中间值、两 slot/旧结果/重放、70端口 Concat 与 slice 对齐拒绝。
- 保留原失败：一次 `copy_bytes` 非 optional 字段的编译错误；旧 capture 消费者缺少既有
  prepared-only 编译约束；Copy/Concat 的真实 slice 对齐缺陷及旧 Direct Provider 名称断言。
  后者按 C2 冻结方案修复，capture 门没有放宽；不同成熟归约次序采用既有数值容差，独立 oracle 保留。
- 成本：proposal 不虚报原版内部 copy/repack 为0。Graph artifact 的 `copy_bytes=0` 只表示
  adapter 未额外插入 boundary copy；原版 cont/cast/repeat 和 managed 维护仍有实际成本。
- 尚未完成：本卡已列的其它 dtype/空值/布局及 indexed、stable gate、完整 masked Softmax 域；
  新代码的 WSL、Air/Mini OFF、适用插桩、source/static/shared、最终复核与 P1 均未运行。
  selector 的 delivery 门保留待验，不能把这次开发测试当作整卡 PASS，完整需求覆盖仍0/256。
- 统一 LOC：Compute + Graph 实现 **55,462行 / 241文件**，排除测试和 vendor。只保留当前
  CUDA 开发构建（358,862,831 bytes）供后续接线复用，没有新增复制的构建树。

源码/设计预算仍累计56/120分钟；当前保持 AUDITED，继续同一有限改动集。没有展开、调查、
优化、比较或删除冻结的自研数学；oneDNN 旧 Q2 条件和用户的 M5 前暂停要求均保持。

## C1/C2 第二批定向审计：原版 Softmax（实施前冻结）

沿同一 pin 核对 `ggml.c` 的 `ggml_soft_max_ext` 与 CUDA `softmax.cu`：原版接收 F32 行和
F32/F16 additive mask，scale 在 mask 相加之前应用；无有效值行的指数和为 NaN，与公开合同
一致。原版 CUDA `l2_norm` 使用 `max(sum,epsilon²)`，不满足 F04.3；已接原版 Multiply→Sum→
Add→Sqrt→Divide 组合继续承载 L2，不能替换成同名原子。

本次有限增量仍属 C1–C4，固定如下：

- `SoftmaxKind::kSoftmax`，F32/F16/BF16、F32 accumulation、任意合法 axis、scale；无 mask
  或同类型可广播 additive mask。沿首批非空/容量/布局范围，半精度原版转F32、末端一次cast。
- 以原版 reshape/permute/cont 把 `[outer,axis,inner]` 转为行矩阵；mask 按已审广播展开后作
  同样轴变换，调用原版 `ggml_soft_max_ext`，再逆变换。保留原版 fused scale/mask/Softmax；
  原版 materialization/managed 成本仍未知，不冒称零copy或capture。
- Direct 只为单节点 Softmax 的 input0/output0、相同 dense 描述符声明精确原位别名；
  additive mask 与输出不允许别名，部分重叠仍拒绝。Graph region 保持原有不别名边界。
- 独立 FP64 log-sum-exp oracle：三dtype、所有轴、rank6/8、strided、广播mask、全遮罩、
  NaN/Inf、长行；另外检查精确原位、部分重叠拒绝、普通 Direct/Graph、多slot和重复执行。
- `LogSoftmax`、Bool mask 和其余尚未绑定域继续明确未完成；不以 `log(softmax)` 冒充稳定
  LogSoftmax，不接入未资格化的激活，不调查或修改冻结自研数学。

本次合同/源码核对计10分钟，累计 **66/120分钟、余54分钟**。本次只推进有限成熟接入；完整
21行、跨平台、交付、插桩与不可变候选性能仍未验收，状态保持 AUDITED。

### Softmax 开发结果

[第二批 receipt](../../../../evidence/correctness/compute-llm-moe-m2-b1-cuda/softmax-windows.json)
保存原始 configure/build/test、两项失败及修正、selector 和实现 hashes。Windows 原配置通过
**8/8 CTest、47次 GoogleTest 执行**；Softmax 独立参数表在三dtype下执行78组，含32769列
原版长行路径，普通 Direct/Graph、精确原位和发布中间结果的两slot重放均通过。

第一次构建缺少测试头的 `<algorithm>` 已补齐。首轮运行唯一失败是旧 TopK 消费者要求 Direct
默认 managed ggml 与 prepared capture Graph 的 Softmax 数值逐位相等；原来两边共用实现。
两边现分别对照同一个既有 FP64 公式和原有 `4e-6 relative + 2e-7 absolute` 容差，TopK 索引
仍严格相等，capture/资源验收未放宽。没有改动或比较旧数学的性能。

补充接缝/别名与失败归因计6分钟，源码/设计累计 **72/120分钟、余48分钟**。完整 B1 验收和
剩余成熟域继续未完成；没有新增完整需求 PASS。统一 Compute + Graph 实现 LOC 为
**55,510行 / 241文件**（排除测试/vendor），仍只保留当前CPU/CUDA开发构建。

### 原版 CCCL 索引 collectives 开发结果

[CUDA-I有限卡](llm-moe-m2-b1-cuda-collectives.md)已按`c1923e1a` SCOPED→`59cc78fb` AUDITED
接入原版DeviceScan/DeviceRadixSort/DeviceReduce/DeviceTransform，承载F02.4/F04.6的Scan/TopK。
普通Direct/Graph自动选择具名成熟Provider；Windows 10/10 CTest与100次GoogleTest执行通过。
原版语义投影、工作区和失败成本见该卡，旧自研math保持不变。完整21行及跨平台/交付/P1仍未完成。

该子卡与本卡共享同一120分钟源码/设计额度；当前累计 **96分钟、余24分钟**，不另开研究预算。
统一Compute+Graph实现 **56,172行/244文件**（不含测试/vendor），完整需求覆盖仍为0/256。

### 原版 Gather 开发结果

同一[CUDA-I卡](llm-moe-m2-b1-cuda-collectives.md)的I5在`fa65e0dc`独立定界、`4a2b4b32`
冻结原版资格后接入。CCCL的DeviceReduce Min/Max与Thrust Gather承载F02.1，保留全部plain
storage位、I32/I64索引、任意轴/stride和checked输出不变。Windows10/10 CTest、105次
GoogleTest执行通过；普通TopK→Gather两slot与失败恢复也实跑通过，不改旧自研数学。

预算与所有子卡共享，累计 **110/120分钟、余10分钟**。统一Compute+Graph实现
**56,374行/244文件**（不含测试/vendor）。Scatter等未绑定域与完整跨平台/交付/插桩/P1
仍未完成，状态AUDITED；本检查点不签完整需求覆盖格。
