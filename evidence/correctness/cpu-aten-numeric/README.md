# CPU ATen：稳定数值程序接线资格

MN1–4 **CLOSED**，不可变产品候选 `0763fd3e3163d16eb0b43e684f5881f8314bc1e9`。
[当前卡](../../../docs/implementation/tasks/compute/cpu-aten-numeric-admission.md)；Windows产品真相，
验证副本单向同步。已锁LibTorch2.8.0 CPU SDK限定Windows/Linux x86_64；本批不签其它CPU平台。

## 实现与语义

plain F32/F16/BF16，exact shape、rank0..8、任意合法byte stride与inferred dense output：四Unary
SiLU/GELU/GELU-tanh/Softplus、三GatedActivation的packed/separate、number Binary Min/Max、
Reduce Sum/Mean/Min/Max。Reduce任意合法axes/keepdim/empty，显式F32 accumulation；F32还允许Unknown。
gate在F32 activation与乘法之后一次输出舍入。普通入口与Graph下游不需要用户注册或排序Provider。

原Byte copy先将输入打包到冷私有对齐typed native Tensor，不能将odd byte stride截断成element stride。
SiLU借原Double silu，erf-GELU借原Double erfc等价程序，tanh-GELU借原Double sigmoid/pow/add/mul；
原where仅修复−Inf极限。显式F32 activation后乘value，再一次storage cast。Softplus借原Float稳定kernel。
Min/Max借原fmin/fmax，单NaN取另一数，双零用原signbit/where修复；Reduce借原sum/mean/amin/amax，
传播NaN，空集合写约定identity，零号用原kernel mask组合修复。没有HYLE新payload数学/近似/SIMD循环。

原Double转换、多节点temporary、input/output packing及F32中间materialization均属于真实完整程序成本。
普通规则仍先现有oneDNN/ggml，再ATen，再具名Native残余；这不是SOTA/WIN裁决。ATen是managed
同步leaf，host阻塞、原库私有线程/allocator、unknown内部内存；不承诺capture、零热分配或严格内存上界。
仅Unary或Binary匹配equal dense description的端口支持精确alias；所有Gate/Reduce、broadcast/strided
alias拒绝。prepared program不可变，native packing槽每invocation独立，borrowed Tensor先于maps关闭。

## 实际验证

| 环境 | 受影响测试 | 同一普通C++消费者 |
|---|---|---|
| Windows | Provider120/120；CPU12通过/1跳过；Graph15/15；公开头1/1；feature关闭adapter/程序集 | source shared/static、installed shared/static通过 |
| Linux GCC14 | Provider120/120；CPU12通过/1跳过；Graph15/15；公开头1/1 | source shared/static、installed shared/static通过 |

6项新增批量测试包含三dtype、稠密/odd strides、tail/±Inf/NaN/±0、large-value gate、广播、rank0/8、
empty/all/single/multi axes、keepdim、零号/NaN Reduce、允许alias/拒绝其它alias、真实wait/close，
两线程独立invocation/queue12轮并发、tracked冷prepare/invocation逐点OOM恢复与资源归零。
独立scalar oracle使用稳定erfc/branch exp和独立低精度RNE encoder；未以消减后的0弱化门。
ordinary consumer只链接hyle::compute/graph，三dtype四组能力变化输入重放，核对实际成熟identity与
op→Copy Graph最终结果；odd-byte域实际绑定ATen。没有新增整ATen region或拆除已有成熟融合。

旧D5RealQwenProjectionUsesExactProfileAndReplays因缺fixture两平台跳过，不计PASS；本卡未改该画像，
新的普通主/正交旅程实际运行。native物理OOM/noncoherent mapping fault未注入，仅源码审计传播，
不写成故障注入PASS。Windows feature-OFF是当前ATen/CPU/Graph TU与assembler的编译/消费资格，
复用已安装依赖，不宣称完整SDK-OFF重建或去除Torch DLL。Windows DLL按既有helper放exe邻目录。

## 失败、不可变产品与证据复用

首批Windows119/120来自qualification对非法Axes::from(empty)取value；改为合法axes，保留原XML，
产品代码未改，最终120/120。Linux先被-Werror=range-loop-construct拦住，仅把测试vector循环变量
改const引用；保留失败命令/编译日志。测试维护commit `91f04856270880e9cdc4548c116d92f10a839e57` 不改变产品candidate，
725项产品/CMake/普通消费者输入hash与Windows已验收版本一致，Linux1333项输入已单向校验。
同一测试数值/形状保持，Windows证据按原实际commit复用，不伪写成新commit运行记录。

原版diagnostic **48,780断言**仍绑定[原manifest](original-manifest.json)，只证明原程序可运行，
不是算子/模型数量。直接raw GELU与消减oracle的失效日志保留；负尾部−8.36250019073原double结果
与消减0不一致，最后采用稳定原程序和独立参考。原−Inf和signed-zero边界均明确，未放宽数值门。

## 终审、成本与剩余

root于2026-10-06独立一遍MN1–4源码/证据检查，核对exact准入、native对齐packing、alias顺序、
borrowed生命周期、异常/OOM、私有lane、数值中间阶段、ordinary identity、交付和fixture复用边界。
P0为O(input+output)多节点原程序；所有cast/packing/scratch/hot metadata显式managed收费。
七门通过，但无新P1/WIN、模型级推理或其它路线资格；Native/SIMD自研优化/性能清退保持冻结。

ggml现有Sigmoid/SiLU有限负尾部仍有具名源码疑点，须一次完整consumer核对，不由ATen后备掩盖。
其余RoPE/I64 positions、Bool/log-softmax、FP8 numeric Cast、shape/资源残余及最终自研取舍仍未完成。
本卡关闭仅本批域，CPU整体没有完成。只有本批语义/资源/组合/交付反例或依赖升级可重开。
统一Compute+Graph实现LOC **61,950/249文件**，不可变产品candidate统计，排除测试/vendor。
本批消费者临时目录已删 **914,243,314 logical bytes**；producer/SDK/cache保留，probe DLL硬链接
不计为实际回收空间，不复制巨型二进制到证据。

[收据](receipt.json)和[工件](qualification.zip)保存命令/所有失败/XML、同一consumer/stub、两份
manifest/fixture-only复用证明、selector selected/skipped、驱动/cleanup/LOC。工件SHA256
`fc0dc342778796e9b5bcc2be26f5c707af27eaf596e0dd407d82c78a56cad5c1`。终审状态/文档变化不重建/P1。
