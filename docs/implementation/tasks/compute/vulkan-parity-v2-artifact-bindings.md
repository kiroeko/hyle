# Vulkan V2：原 NF4／AWQ／compressed-tensors 批量接入

<!-- MODULE REVIEW: AUDITED -->

父轮：[共同目录收口](vulkan-cpu-cuda-apple-parity-closure.md)。baseline
`487b474072529a11612963add826bff40e4f5a6f`；owner
`compute_provider_iree_vulkan`、`compute_route_vulkan`、`graph_route_vulkan`。
IREE 原 CANDIDATE 因新增原程序退回 AUDITED；两路线已由独立 paired A/A 恢复 SCOPED，
本次完成具名审计后进入 AUDITED。其他 owner 的 CLOSED/有限 FIX 不变。

## 真实 consumer 与来源

[共同端口](cpu-final-demand-carriers.json)中 6 个 NF4、7 个 AWQ、5 个 CT packed
合同需要原字节解码和后续投影；不换成近似 GGUF 格式。主纵切是混合权重的显式准备、
decoded Tensor→MatMul，正交纵切是 U8/U32 payload 的精确 Copy/Fill/Select/View/Concat。
普通调用沿现有 `artifact_weight_decode`、Vulkan `prepare`、GraphBuilder/compile，
不增加模型 loader、provider 排序或用户 timing。

已核对公共 `weight_decode.cc` 的全部端口、group、shape 和输出 dtype；既有 Apple 借用
代码和登记原源码互相对应：bitsandbytes `833649043474794b8fe7a4136e0c40faf077b2e0`，
AutoAWQ `88e4c76b20755db275574e6a03c83c84ba3bece5`，compressed-tensors
`ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4`。复制原 tensor 序列为 IREE 标准
tensor/arith/linalg recipe，保留许可/来源，不自写 GLSL、量化算法或调度器。
锁定 compiler/runtime 均为 `6ef606b6fa3e12bd2c12d36a07463069eb71185f`；编译器已本地可用。

## 有限 FIX（产品前冻结）

| ID | 缺陷／允许改动 | 验收 |
|---|---|---|
| A1 | IREE 原标准程序承载 NF4 high-first nibble、nested absmax 256+offset、AWQ 原 nibble 顺序及零点、CT symmetric4；保留源 scale dtype 的乘法舍入及最终 F32/F16/BF16 输出 | 独立原 CPU 借用参考，全部格式/scale-output dtype 组合、group、非方阵和变 payload |
| A2 | 私有 ProgramSpec 原程序类型、group/source-output dtype；最多六个 native inputs 和 group scalar；原 module/VM/HAL/fence/quota/async retirement 复用 | 普通两入口、独立 slot、replay/保活、close/cancel/资源失败，无 host 数学 fallback |
| A3 | 路线冷匹配 ArtifactWeightDecode、逐真实 port 拷贝，Graph typed hook；原六输入不得误走 FP8 的 port%2/scale offset 或 Scan flag | nested 六个不同输入、普通 Graph 未提供 custom callback、decoded weight 后接原 MatMul |
| A4 | 原 U32 存储 shader 仅新增 U8/U32 storage ID/width；不改数学 body；公共 schema 与 float-only guard 保持 | bytes 精确、非四字节尾、stride/view、禁止未资格 unsigned 数学 |
| A5 | 新标准 recipe/离线 VMFB、来源许可及 vendor digest 纳入既有生成机制 | 同一 producer 编译、现有 native IREE/precision/FP8 回归；父轮三交付统一运行 |

34 packed 中其余 16 MLX、三种扩展 IQ1、七 typed hook 中其它六个及完整目录/官方 FP4/FP8
仍由父轮具名后续批次处理，不把本批静态支持当整个 Vulkan runtime PASS。

## 冻结语义与成本

NF4 scale/book 是 F32；nested 首先以原 F32 顺序还原 absmax。
AWQ 原顺序 `{0,4,1,5,2,6,3,7}`，差值先 cast 到 scale dtype，再乘并保留该 dtype 的舍入；
CT signed4 同样先 cast 再乘。最终显式 output dtype 在此后转换，不能先提升 F32 避掉
原半精度舍入。BF16 端口以 i16 原位承载，内部用标准 bitcast/truncf；F16 使用原类型。

decoded tensor 是公开、显式的准备结果；动态 decode 每次执行和 GPU copy/完成均计成本。
不暗中常驻整模型 F32 bank。不可变权重可由 caller 先 decode，再作为已有 MatMul 常量
绑定；expert selected-only 与 MLX fused region 属后续独立 batch。所有 feature、shape、
descriptor/workspace/dispatch 范围在冷准备前拒绝，不靠 OOM 试探能力。
每 Invocation 自己的 VM/context/input/private-output、原 completion 和结果 lease 不变；
失败/取消禁止发布，退役后才能释放或复用。六输入只扩大本 provider 原有的有限参数数组。

## 七问与验证预算

1. 三原成熟源及 IREE 已登记；新 recipe 标注各来源及许可证，compiler 不成为消费依赖。
2. 本批 18 packed 合同仍须实际 geometry 验证；样本与静态 source 不代签完整目录。
3. 复用现有 schema、Vulkan owner、ggml 后续 MatMul、IREE VM/HAL 与 quota；不重开下层全库。
4. P0：每输出一次原 scalar decode，原标准编译器调度；显式输出及边界成本如实记录。
   本批只有一家完整 carrier，不制造 provider 竞赛；父轮稳定候选再做适用完整成本 P1。
5. 原 VM 私有 FFI 不进入公开 C++，无新增公共 vtable/executor 或兼容旧 API。
6. 原 CMake 嵌入所有 bytecode；无需运行时 compiler/Python。三交付/搬迁/DLL/许可由 V4 一批验证。
7. 现有 Direct/Graph schema 自然入口，不要求外部 custom shader callback；错误说明具体 feature/domain。

批末一个不可变 candidate，复用 `C:/hv/vp1`，build2/test1、重型串行；源码/离线 recipe
实现预算 90 分钟，资格与纠正预算 45 分钟。只允许一次具名 source/compiler/接线纠正，
不逐格式重配、不重复已通过整矩阵、不跑本批 P1。新 source 事实改变算法或资源范围时
重新定界，不写自研替代。Windows RTX4070 原 GPU 先验；Air capability 与三交付仍未验。

预声明：格式/原源 dtype/最终 dtype、group64/128、tail/nested跨256、两入口投影、原字段
更新、两 slot/lease/close、schema错误/容量/OOM/cancel，以及现有 IREE/存储影响子集。
selector、docs/closure 必跑；适用未运行写 UNPROVEN，不能 PASS。budget 到达留下明确
剩余项及原因，本卡保持 AUDITED 或依法转阶段，不能用无限调查续命。

当前：审计完成；产品与 runtime 尚未实现/运行，全部新增验证 UNPROVEN。

## 首批 candidate 与具名交付纠正

`1ea7e3074` 完成 A1–A5 产品接线和 24 个原格式/scale-output dtype 入口、六 native input。
锁定编译器离线原标准 recipe 生成 24 个 LocalSize64/SignedZeroInfNanPreserve 入口成功；
这不是 runtime PASS。首次集中构建在 package registry 拒绝新增三个原源 license
未列入安装资源，尚未编译新 C++。一次 A5 纠正仅补 owning factor 的三项
`RUNTIME_FILE_SPECS`；不修改公共 package checker，不删除许可证或放宽 gate。
复用同一 producer，未启动测试或 P1，旧通过域不重跑。

首次 GPU `156ab3c7`：原 IREE7/7、U8/U32两个存储及旧 FP8/Scan 域通过；新增三 Direct/
两 Graph ID 失败，不能签新增 PASS。一次 `2585bc6a` 原四格式捕获证明 NF4/AWQ/CT
解码样本匹配；nested NF4 高位 scale code 148/185 变成负 index，原生成 SPIR-V 在
`ShiftRightArithmetic` 后直接 AccessChain，丢失 i8→unsigned index 的零扩展，robust
读取为0。唯一 A1 纠正把原 U8 lookup 的 unsigned 语义写成标准 index cast +255 mask；
不改原 scale 数学、格式、compiler pass/调度或数值门。其它21原入口不改。

同次捕获明确请求创建拒绝是 `Vulkan device memory quota`，不是 ABI 或六端口数值失败。
原 IREE `vulkan/allocator.c` 的池 slab 为64MiB，完整配额包含保留池和 caller/native output；
128MiB 新 fixture 无法覆盖已保留 native 池及输出。恢复既有 FP8 fixture 的256MiB正常
预算，同时仍以 reserve-all 注入真实 OOM、检查释放到0，不放宽产品 quota 或绕过计费。
仅重跑失败的新增五 ID（与首次未能到达的新增组合）；原 Provider/FP8/Scan/U8 PASS 复用。
一次 test 显式 Operation 构造纠正已通过增量编译，未更改产品接口。

## A1-R：源精度舍入边界定向审计（AUDITED）

`61d0585b` 唯一纠正复验：Direct 冷解码→MatMul 和取消/OOM/关闭/释放到0通过，Graph
冷解码→MatMul通过，nested U8 lookup 原问题已消失；两批量 ID 仍 FAIL。首次真正走到
全部 scale/output 组合发现新事实：AWQ/CT F16/BF16 source→其它 output 的中间舍入被
compiler canonicalization 消掉。例如原 BF16 source→F16 的生成 SPIR-V 是 FMul F32→
直接 FConvert F16，缺少源 BF16 舍入；不能放宽精度或把不同输出层级当成通过。

新增有限 A1-R 只允许在已有原 recipe 的 source dtype 值上使用 IREE 已有
`util.optimization_barrier`，保持原 tensor materialization 的舍入，再做最终转换。
锁定原 `UtilOps.td` 603–610 明确该原操作在最后编译阶段删除、无 runtime 开销；不修改
编译器、shader 算法、source dtype、最终 dtype 或数值门。cold ABI/六输入/VM/资源不变。
离线编译与两失败批量 ID 纠正预算10分钟、一次；已通过的三新增 ID 和旧域复用。
若原标准 barrier 仍不能保持正确舍入，具名拒绝这些 cross-dtype 域并退回范围裁决，
不继续改仪器、随机换 compiler flags 或写自研舍入算法。

原标准 barrier 的标量位置会产生非法无限动态 workgroup scratch；tensor 位置缺失
Flow shape propagation，两个离线结果均未产生可交付制品，未启动新 GPU run。A1-R
依据这些具体反例重新固定为已有成熟分段组合：原解码只生成源 F16/BF16 typed tensor，
最终 Cast 复用已有合格 ggml 浮点 Cast／IREE F16→BF16；不继续 compiler 调参。
两个位置的失败保存证据；未编译候选源恢复到已验原 recipe 后移除八个失格 cross-dtype
导出，不保留未使用错误程序。NativeProgram 只接受合格单段域；路线冷期追加内部 typed
value 和成熟 Cast，原 liveness/完成/失败/结果 lease 负责两个片段。新增中间 bytes、
GPU copy／Cast／完成均真实计费；不写舍入算法，不改公开入口。
source 仍16个离线 entry，跨源精度的公开组合由第二成熟片段承载；生成器/manifest/native
resolver同步裁剪。仅两失败批量 ID 验证，既有三新增成功 ID 仍复用；最多一次真实验收。

## Windows batch 结果（非父轮终局）

产品 `9e00073bf`；`1710f874` 两失败 ID 唯一复验 Direct1/1、Graph1/1 PASS，48个
格式/scale-output/group组合各自经过独立slot/完整payload更新/replay/旧结果，另验
nested9/258组边界。合并 `61d0585b` 的两入口 cold decode→MatMul、取消/OOM/close/释放
PASS 和首轮原Provider/FP8/Scan/U8/U32影响子集，Windows微型原链资格完成。
[完整证据与限制](../../../../evidence/correctness/vulkan-parity-v2-artifacts/README.md)。
原失败保留，约106MiB失败raw无损压成1.54MiB，验证逐字节相等后清理重复raw；未扩公共仪器。

这不是18个全模型geometry、559/34/45目录或Air/三交付/default/P1完成。
父轮仍active；下一批按原MLX16 packed/剩余typed hook/真实官方精度与状态审计定界。
当前统一Compute+Graph非测试实现61,663行/261文件；vendor recipe/bytecode不算自研实现。
