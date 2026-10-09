# Vulkan V2：共同实际端口与完整状态链

<!-- MODULE REVIEW: AUDITED -->

baseline `9881ce1cf`；owners `compute_route_vulkan`、`graph_route_vulkan`、
`compute_provider_ggml_vulkan`。属于[父轮](vulkan-cpu-cuda-apple-parity-closure.md)。
共享 Compute/Graph、CPU/CUDA/Apple、ggml core 与公共验证仪器不重开。

## 审计与有限 FIX

需求只来自[共同目录](../../../architecture/model-compute-catalogue.md)、其 JSON 与现有
`cpu_c5_*`、`cpu_c4_*`、D5 主干/状态 consumer。559/80/34/45/104/八族不缩小；
沿其它路线已声明的完整 K/N、全词表 row 分块和显式 expert window 验收合同，报告
原 E 与执行 E；不把 window、冷准备或微型程序冒充完整巨大银行 GPU 驻留。
另保留三 IQ1 原 E 冷域与可用银行实跑任务，不以编译或静态 source 代签。

| ID | 已定位问题 | 有限修改 |
|---|---|---|
| M1 | 45 plain 中4个 F16/BF16 conv；公开 CausalConv1d 允许三 dtype，现 ggml hook 只 F32 | 沿既有 native Cast→原 ggml ssm_conv→原 typed Cast，F32 累加/输出舍入保留；无新数学 |
| M2 | 原 C5 common fixture guarded CPU/Apple，Vulkan 的旧 runner 无 cold preparation/invalid-index/replay phase | 复用共同数据和程序，补 Vulkan 测试承载与独立原 GPU/codec oracle；两普通入口、prepared constants、反馈/reset与坏 IDs |
| M3 | grouped 原 ggml_mul_mat_id 未绑定公共 F32 accumulation；原 ggml_prec_set_acc 明确支持此 op，Vulkan 原 map 读取 op_params[0] | 与现 MatMul 同样调用成熟公共 precision setter，拒绝设置失败；实际宽度/多token selected 差分资格 |
| M4 | 两旧官方 MoE 测试仍要求缺 hook 拒绝 | 改成原完整片段运行验收；新版 Pro0813/Flash0731/DSpark/MTP独立接入验收，不改模型源或数值门 |
| M5 | 新目录普通形状尚未实际运行 | 同 producer 一次集中矩阵，预审容量/feature/stride；具名额外接线缺陷先在本卡补明确有限 FIX，不无界实现 |

原 `ggml.c::ggml_prec_set_acc` 两 MAT op 同源接口；原 Vulkan grouped selector
`ggml_vk_get_mul_mat_mat_pipeline_map(...dst->op_params[0],true)`证明源参数相关。
M1 公共 schema 的 F32 accumulation 与 matching output dtype 已固定，复用原 casts 与
资源机制；所有转换、临时、managed/native 同步计完整成本。只改有限三 owner，
不更新依赖、不添加 HYLE shader、不恢复 self 数学研发/优化或扩张目录。

主纵切为 104 制品实际权重与完整 R02–R13、主干及适用 MTP/DSpark；正交纵切为
非GGUF plain/packed真实 dtype、存储/布局、changed inputs/weights/IDs、坏索引与资源。
Provider 保留内部融合；Graph 管全局边界；普通 C++ prepare/compile、不增加用户 flags。
PUBLIC API 不变，源码/static/shared仍同产品和原依赖闭包；三交付由父轮 V4 集中验证。

P0 只原算子/转换成本与实际资源账；本包不跑 P1、不写 WIN。审计/批接预算90分钟、
一次集中资格45分钟，每具名源/接线/fixture错误一次定向纠正，未完成明确标记。
一个 producer、build2/test1、重型串行；失败域分类后只定向 delta，文档不构建。

预声明验证：两入口559实际权重合同/80参数类/34 packed/45 plain/104身份、旧已验
完整8族分支及当前官方两个main/DSpark和命名sidecars；完整cold准备→hot重放，
原输入/中间dtype、finite/tie/尾部、state反馈/reset；容量拒绝/索引失败不发布、
资源/lease/cancel/OOM对具名变化执行必要delta。native已过微型包不重跑。
机器 Windows RTX4070真实 GPU；Air/WSL/三交付/default/P1仍父轮未验范围。

七问：原实现身份/许可保持；真实主/正交完整程序固定；下层和 checked/storage复用；
成本披露与原语义不变；无FFI/新provider vtable；原交付闭包；用户自然入口和具名限制。
本卡 AUDITED 授权 M1–M5，尚无本轮 runtime PASS，不签完整路线完成。

### M5-G：实际宽矩阵的原 grid 单位

共同plain `p033ae23672c4671a` N=17408/K=5120，现冷检查将8 split-K
乘到原始列数，误拒绝已审成熟矩阵。原 Vulkan MAT pipeline 的最小列 tile32，
实际 X=8*ceil(N/tile)，split reduce tile1024；原源码4419–4600/9208/9213/5804
可复现。只修这两个冷grid单位，保持8片F32 workspace和buffer/index上限；
matvec仍按不大于N的保守grid约束。不放宽数值，不改变原pipeline选择。

第一集中批次接入34 packed/45 plain及原官方四MoE完整片段；[原始证据](../../../../evidence/correctness/vulkan-parity-v2-catalogue/README.md)。
同版本共同fixture/误差门保持，native cold结果保留GPU常量；Graph同slot重放，
Direct复用不可变Program、按旧结果lease合同更新可变请求。原559/80/104及新增
完整branch余项仍属于本卡，未实跑不能由此批代签。

M2 下一矩阵已批接共同559 GGUF/104身份：只给测试增加外部Bank hook，现CPU/
CUDA/Apple分支不变；无CPU backend宏/链接/fallback。锁定原Vulkan 8 prototype
codec/dot为集成oracle，plain与三IQ1扩展用独立原解码/舍入方程，原K/N与全部
head行分块/显式expert window保持。旧80项两入口通过域复用，新矩阵尚未运行。

Windows 第一批 `6ed2cdefa` Direct80/80、Graph80/80 PASS，无validation layer错；
34 packed/45 plain实际合同及原四官方MoE完整程序已资格。M1/冷宽grid与cold
preparation普通入口通过；M3普通GGUF grouped精度绑定随后在559矩阵签运行，
不借上述IREE grouped PASS冒充。剩余559/80/104及新版状态/sidecars继续本卡。
当前整体约38%；统一非测试Compute+Graph62,412行/267文件。

GGUF首矩阵 `a3c3567f` 在原GPU oracle初始化处停止，尚无case PASS。
锁定Vulkan运行时有原instance隔离补丁，直接vendor全局init没有激活owner而assert。
只修测试oracle：创建既有NativeDevice、使用其backend和作用域包围原native调用/
释放；不改runtime/产品/数学/门。定向原559/104矩阵一次，原80项不重跑。

原GPU oracle作用域修正后 `08764fc4` Direct560/560、Graph560/560 PASS，
每入口559实际GGUF合同与104精确身份，原head全行分块和expert window明示；
无validation layer错，M3原grouped F32绑定取得实际宽度/IDs/重放资格。
首失败不删；下一批只是80参数承载及最新main/state/MTP/DSpark，以上矩阵不重跑。

新版状态/参数批已集中build：原C4 Pro0813/Flash0731 main、target/confidence/
Markov、attention/cache、完整三stage DSpark；Qwen命名Q4 sidecar与Flash两
sidecars；共同C5 convolution/Qwen实际attention/GDN/MTP/newGGUF主干消费者。
只新增测试原Vulkan reference作用域与具名test宏分支；CPU/CUDA/Apple方程、
普通产品和数学不变。原IREE dot→scale→add原序列的非融合定义预先固定；
GPU实跑尚未完成，原560/80两入口结果不重复、不由原model版本代签最新版本。

### M5-S：最新sidecar的两个真实接线反例

`575b5d80` 状态批次未完成，Qwen命名MTP的 SDPA causal=false被冷拒绝；
原原生scores→softmax→value支持非因果，当前adapter固定拒绝/始终mask。只放开
已承载同dtype/shape档的causal=false，并仅causal=true生成原diag mask。
Flash三stage BF16 activation T=5触发原matvec断言；原源码10534的选择阈值为8，
且native ne1=1适用于有batch的调用，现adapter只判rank2的1–4行。只按同原
native ne1/ne2/ne3 predicate widening BF16 activation，保留BF16权重/原F32acc，
所有转换成本计账。M5-S具名有限纠正，不改库/Shader/精度门，原80/560已过域
不重测，状态批尚未取得资格，之后一次定向状态验。

### M5-N1：Qwen单query SDPA的错误原程序参考

状态Direct已运行24项16 PASS/8 FAIL。Qwen实际attention消费者为Qseq=1，
当前Provider明确lower为F32 scores MatMul→softmax→F32 value MatMul；原Vulkan
10534选择matvec而非coopmat2，两个乘法均F32。共同fixture.native_outputs
却固定假设coopmat2两乘数F16，48个小误差断言仅对这个错误参考。
Vulkan测试wrapper只对固定region0x040c0501、Qseq=1清除这条不适用的F16
参考，保留更严格的原F32 8e-5/4e-5门、RoPE/norm/state方程；原primitive/
算法/API不改，不以更宽门取PASS。两入口这一个ID一次定向验证。
其它DSpark/attention/新版GGUF数值FAIL仍保留，须逐首偏差有界查因，不推定环境或
绕过原完整片段门；本条不能替那些域签PASS。

## 当前资格与具名剩余数值域

产品 `9ec5c4ee` 修复两真实接线边界；状态Direct16/24、Graph15/24；
`d0c6138c` Qseq1原F32参考纠正两入口单ID PASS，当前17/24、16/24。
原非GGUF80/80与GGUF560/560两入口矩阵保持，未重测或宣称全state通过。

M5-N2 有限follow-up：剩余7/8项按首分歧分成多query sink attention/FP32或F16
乘数、DSpark全词表复合误差、新官方/两GGUF main的状态舍入，不先推定为产品
或oracle问题。一次原native leaf与独立源方程的同输入捕获，最多30分钟；只
允许具体冷绑定/转换/原程序参考修正，精度门和原模型程序不变。需要新public
合同或自研算法时停下具名决策；其它已过域不反复跑。其后只受影响ID一次
集中delta，不重测80/560/24全集。该scope先作为诊断，不授权盲调实现。

完整八族剩余历史branch映射/80参数归属、三IQ1巨大bank能力、平台/三交付/default/
正式P1仍未完成，父轮45%不因此加成。所有未验或失败范围维持UNPROVEN/FAIL。

### M5-N2-S：同输入原程序精度已证

2026-10-09 14:30：一次Flash0731首sink原边界捕获，T5/H64/D512/L133；
HYLE与锁定原ggml直接图native最大差0；独立F32方程最大差6.92009926e-5。
源9537–9640在coopmat2下明确F32→F16两乘数，F32累加；独立方程加原F16
乘数/概率存储边界后最大差1.34110451e-7。证据`sink-rounded-source-cause.log`。
首次diagnostic误给空初cache导致fixture越界；已按完整window512修正，原记录保留。

只把当前Model.attention_reference绑定原Vulkan sink计算，照原mask/sinks/shape
输入，在取得HYLE输出前构造参考；原GPU错误/类型/完整片段门与F32/BF16
状态方程保持。先两入口以外只Direct两个Prefill ID一次定向资格；原生公式
与独立方程证据并列，不把native一致性冒充完整主干资格，仍无产品/Shader改动。
