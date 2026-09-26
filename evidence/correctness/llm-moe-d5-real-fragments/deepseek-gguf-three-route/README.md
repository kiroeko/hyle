# DeepSeek Flash GGUF：CPU / CUDA / Apple 普通两入口

[收据](receipt.json)按真实权重角色与必需 region，签署 Flash UD-Q4_K_XL、UD-Q8_K_XL
的 CPU、CUDA、Apple Direct/Graph，共新增 **12格**。D5 当前 **140/176 RUN_PASS、36格未资格化**。
Vulkan 的4格仍有数值失败，四个官方 FP8 画像的32格、D5三交付/P1及D6仍未完成。
这不是巨型 checkpoint 的整模型推理或吞吐验收。

## 覆盖和证据复用

- R01：完整129280词表的 embedding、checked Gather、head；learned mHC head 的实际输出
  经过 norm 后接全词表 head，独立 oracle 保留整链误差和 head 局部门。
- R02/R03/R07–R09：真实 H4096、Q-rank1024、64×512 heads，raw窗口128，CSA/HCA的
  ratio4/128投影、位置、压缩cache、indexer→TopK→Gather→带sink attention→分组输出投影。
  保留Q8_0或BF16逐角色权重，六次变化输入、真实状态回送及reset，单独压缩器含65536长位置。
- R10/R11/R14：前三层token查表与其余score+bias路由分别接完整MXFP4 routed/shared MoE；
  shared保持对应Q8_0/BF16。权重是header对应的合法非零合成payload，不将标签Q8解释为全模型Q8。
- R12/R15：learned mHC混合、四路状态、head gate、准备/重放/失败保留原测试；新Metal中间buffer
  的分配失败、回滚及同owner的recurrent/完成失败，在静态故障注入中通过。
- R13不适用于这两份固定GGUF：完整header只有43层且无NextN权重。官方DeepSeek的MTP仍待验。

CPU完整owner `b111e3b7` 已通过404个GTest及其它合同；此后CPU产品未变。
CUDA首次批次中通过的用例逐项复用，长位置失败由原版unary取消fast-math的不可变候选修复，
两入口attention/压缩及38项Provider通过；失败Invocation恢复按公开合同修正后单独通过。
见[CUDA完整前后证据](../explicit-rope-phase/cuda/README.md)。失败批次本身没有改写为PASS。

Apple首次完整批次Direct168/170、Graph141/143通过，失败只在新attention/压缩。
Bool上传修正后的失败暴露了原版Metal没有`bf16_bf16`矩阵核的接线问题；候选`91af7569`
用原版cast精确扩宽激活、保持BF16权重，进入已有`bf16_f32`矩阵核。Direct19、Graph13、
Provider7项定向通过；独立静态Provider12项（含5项故障测试）全部通过，无跳过。
新增workspace由prepared program持有并复用，不展开BF16权重；成本尚待A6。
静态测试树已被清理导致的首次启动失败也保留，没有当作产品FAIL或PASS。

## Vulkan 精确未决域

`6c27e592`完整批次：Direct253/255、Graph162/165通过，ggml/IREE Provider通过。
`91af7569`修正Graph漏掉SinkAttention的冷分类和Direct既有边界断言后，定向Graph的sink及
两个custom拒绝测试通过；两入口attention/压缩仍数值失败（4项）。没有改误差门或删长位置。

长位置压缩的首个旋转对频率为1，排除了此前倒数舍入；其误差和CUDA原fast-math路径一致。
原版ggml Vulkan `unary.comp`及`rope_funcs.glsl`直接调用GLSL sin/cos。
[Khronos精度合同](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#precision-of-glsl-std-450-instructions)
只给出了[-π,π]内不超过2^-11的单精度绝对误差；不能据此承诺65536处的CPU库精度。
这是已定位的高可信原因，尚未用替代成熟实现完成原片段复验。

现有IREE编译器的[有界原版探针](vulkan-trig/receipt.json)也未解决该域：在65536处，
sin误差约0.001315、cos约0.001263。固定compiler `6ef606b6` 原生lowering生成F32
`x-floor(x*(2/pi))*(pi/2)`缩角再做原版多项式，生成IR随证据保留；没有HYLE替代数学。
该探针不是HYLE程序接入，也不是所有成熟实现都不支持的证据。普通IREE切换不再重复尝试；
下一步需要限定精度域的成熟实现资格，或对原生完整片段另行固定数值验收，不能悄悄放宽本门。

CPU/CUDA/Apple已经完成的格独立签署，不等待Vulkan；三交付和性能仍单独验收。
[统一实现统计](loc.json)排除测试和第三方源；本批没有新增数学kernel。
