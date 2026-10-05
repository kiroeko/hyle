# CPU自研算子：逐精确域与成熟Provider裁决

<!-- MODULE REVIEW: SCOPED -->

- baseline：`155d1385c2b9a50eaa66a6c069b569f11daf4b03`，Windows canonical。用户要求：有成熟替代的所有自研算子都要裁决，
  不能因为低优先级后备就永久保留；只有唯一实现的精确域才直接保留。SIMD基础不删除。
- owner：`compute_provider_native`、`compute_route_cpu`、`graph_route_cpu`先绑定本卡SCOPED。
  [标准CPU构建](../build/cpu-standard-provider-build-v1.md)与ggml优化程序接入已独立关闭，不能代替本卡。涉及oneDNN/ggml的具体接入修复，
  先据反例重新绑定相应owner并固定有限FIX，不借旧CLOSED/AUDITED记录授权。
- scope：现存Native全部payload数学域（MatMul、norm、activation/gate、reduce/softmax、
  attention/RoPE、index/transform/cast），相对于oneDNN、ggml、已接Torch成熟域逐项核对。
  TorchAO/ds4原版数学、ATen调用和checked生命周期adapter与自研数学分开。
- 证据：D4的宽泛KEEP_RESIDUAL不是新裁决；当前默认先oneDNN100、ggml200、Native1000，
  优先级本身不证明胜负或保留依据。旧原子/完整片段性能只能在身份与合同匹配时复用。

## 顺序与停止

1. 从实际prepare/propose条件批量生成精确域表：op、dtype/storage/block、shape/layout、
   数值/alias/failure/资源/线程、平台/ISA/软件、模型画像/端口角色；列出具名成熟候选和缺口。
2. 没有成熟同合同候选的域记录UNIQUE_IMPLEMENTATION及源码证据；其余全部进入DECISION_REQUIRED。
   无证据不是WIN，测试通过也不是性能领先，不将宽泛fallback记录当免审理由。
3. 有有效旧同合同证据则直接复用；其余按相同机制批量固定有限代表/holdout消费、原数值门和
   必要转换/完成/资源成本，在不可变候选上一次适用P1；预检失败先查源码，不无限测量或优化。
4. 按严格方向和工程门固定KEEP_WIN/RETIRE_OVERLAP/KEEP_UNIQUE/明确测量阻塞；
   路由规则表达实际dtype/量化、模型/硬件/软件/shape条件，准备时绑定，执行不比赛。
5. 先固定有限删除/路由FIX，再修改产品；一批构建/正确性/必要三交付后终审。SIMD基础保持，
   只有不再有独立用途的算子专用扩张另有具名证据才随域裁决。

本卡目前只是SCOPED审计，尚未测新比较、删算子或签WIN。具体有限样本/成本与FIX须AUDITED
后才开始，禁止生成逐算子无界benchmark目录；在用户确认的既有矩阵上批量处理。

## 当前源码域表（尚未裁决）

按实际已接同合同候选划界；不同 dtype、mask、alias、bounded shape、预算档位分别审计。
下表的 `DECISION_REQUIRED` 仍是未完成，不把 vendor 尚未接入的能力虚报为可调用路线。

| Native 能力族 | 成熟重叠候选 | 不能用重叠域结果直接删除的边界 | 动作 |
|---|---|---|---|
| MatMul | ggml 非空 rank2 plain F32、contiguous-K 的转置权重；oneDNN 非空 exact dense row-major F32，已删除 serial M<16 过滤 | Native F32 NumPy batch 广播、空K、非vendor可表达批次布局；ggml 不声明一般 row-major RHS | DECISION_REQUIRED；两种物理布局分别匹配合格候选，旧M1 Native-vs-Native不是胜出 |
| RMSNorm/LayerNorm | oneDNN dense浮点、input0精确原位；ggml可表达浮点、F32 accumulation | bounded RMS整符号域、受控预算、非vendor布局；原位不能一概归唯一 | DECISION_REQUIRED；普通out-of-place与oneDNN原位均须比较，旧Graph不能代签新融合 |
| Unary | ggml Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Sigmoid；oneDNN dense F32稳定Softplus | 未接ReLU/SiLU/GELU/GELU-tanh、narrow Softplus、layout/alias/非有限值 | DECISION_REQUIRED，按kind/dtype分开 |
| Binary/Select | ggml Add/Subtract/Multiply/Divide；oneDNN Bool Select广播/plain分支 | number Min/Max的NaN/signed-zero、特殊alias/layout/scalar-width | DECISION_REQUIRED |
| Reduce | ggml Sum/Mean、Unknown/F32 accumulation | Min/Max语义、部分axes/rank/empty/checked strides | DECISION_REQUIRED |
| GatedActivation | ggml SiLU的packed/separate浮点端口；原位-Inf数值边界已完成两平台三交付资格 | 原版数学核不改；GELU/GELU-tanh原函数未接，alias/layout单列 | SiLU DECISION_REQUIRED；其它kind继续成熟准入核对，边界检查和必要cast成本全部计入 |
| RoPE | ggml dense rank4、sequence-axis1、i32 positions、F32 frequencies | i64 positions、其它axis/rank/layout与checked strides | DECISION_REQUIRED，不偷换公共位置公式 |
| Softmax | ggml dense rank2 F32、axis−1、scale1、unmasked普通Softmax | mask/log-softmax、其它axis/scale/narrow/strides与非有限值 | DECISION_REQUIRED |
| SDPA | ggml causal BSHD、unmasked、F32 accumulation，满足具体Q/K/V shape | non-causal/mask/BHSD/不同depth；Native仅相同Q/K/V heads | DECISION_REQUIRED，只测共同shape |
| Gather/Scatter | ggml i32 indices浮点Gather、浮点Scatter Add | i64、其它scalar/rank/layout；Assign最后逻辑写入获胜不与Add比较 | DECISION_REQUIRED，包含越界和失败不写输出 |
| Concat/Cast/Copy | ggml可表达dtype/rank/layout；oneDNN plain reorder | FP8 Cast、bounded Copy整域、更宽scalar/rank与预算；需实际准入区分 | DECISION_REQUIRED |
| View/Fill | 当前无相同已接替代；View无数值payload，Fill为基础资源写入 | schema/offset/alias/close/empty/OOM仍须已有证据 | KEEP_UNIQUE候选，不虚构性能胜负 |
| ds4/Torch官方lowp | 原版ds4/TorchAO数学与ATen CPU程序 | HYLE只拥有描述、checked mapping、生命周期与组合 | BORROW；官方44/44不重开数学 |
| Scan/TopK接缝 | `standard_index.cc`采用C++标准库inclusive_scan/partial_sort | 明确并非ggml数学；checked overflow、stable tie/NaN/storage-bits为语义适配 | BORROW/STANDARD；不重添旧Native实现 |

源码锚点为Native `build_plan`/各`prepare_*`、ggml CPU `internal::supports`与
`standard_index::supports`、oneDNN `validate_route`/`validate_primitive`/`validate_tensor`。
矩阵只是源码审计，下一批还需实际prepare与数值检查，不能仅凭源码谓词写支持PASS。

比较的共同边界固定为冷Context/prepare、resident端口、每次submit到真实wait；计入成熟实现
必需的转换、复制、分配与同步。Graph另验完整融合片段，叶算子赢不授权拆更快的whole-region。
硬件/OS/编译器/Provider版本/线程预算绑定证据，未知域不套用WIN。
SIMD基础与通用primitive保留，只清理随已退场算子失去独立用途的专属调用/kernel。
目前未删除新的payload或宣布新WIN。

## 成熟CPU ISA准入前置

旧M1只交付安全基线profile，不能与Native SIMD比较后把配置优势误写成算法WIN。
[独立有限接入卡](cpu-ggml-isa-admission-v1.md)已CLOSED：原版优化程序、两平台三交付及资源/隔离
验收已完成。现在才固定本卡批次；不能继承旧scalar ggml计时或旧宽泛KEEP_RESIDUAL。

## 有限准入核对（先于自研计时）

先用一批无计时资格化检查实际普通prepare和必要的原版carrier，区分三类：已接同合同、
成熟库存在但adapter挡住、真实数值/资源语义差异。只在第三类有源码与实际反例时，才允许记录唯一域。
当前具体疑点：oneDNN的serial M<16冷拒绝来自旧“Native更便宜”判断，非成熟kernel能力声明；
Unary/Gated激活必须核对NaN、±Inf与尾部，不能仅以同名算子或有限随机数签等价。
一次batch保存正向/拒绝/数值结果，不计时、不优化自研；之后固定接入/清退/规则FIX及一次P1批次。

第一批已在同一Windows安装SDK上实际复现两项，见
[准入证据](../../../../evidence/correctness/cpu-native-domain-admission/README.md)：

- oneDNN普通prepare拒绝M1，而同版已接原BRGEMM carrier对M1/K64/N64正确产生64个结果。
  这证明旧adapter条件挡住真实成熟能力，不能据该拒绝给Native签唯一实现或WIN。
- 相同F32 separate SiLU gate的`-Inf`输入、value=1，Native输出0，原版优化ggml输出NaN；
  其余此次列出的有限/NaN/+Inf输入符合预期。该差异须在固定普通数值合同和成熟接缝后处理，
  不能拿有限随机数的通过外推整个域，不能直接启动性能比赛或因同名而删除自研。

这次是准入/正确性诊断，不是性能比较、完整目录资格或产品完成。Native仍冻结，未增加任何数学。

### 已落实的有限准入修正

[oneDNN 小矩阵卡](cpu-onednn-small-matmul-admission.md)的 OM1/OM2 已实现：普通 M1/M4/M15/M16
可使用原 BRGEMM，Windows 定向 7/7、Linux 批量 8/8 与两平台三交付已通过，本接入卡已 CLOSED。
[ggml 门控卡](cpu-ggml-gated-numeric-admission.md)的 GN1/GN2 已实现：保留原版 SwiGLU 与整程序，
在中间值进入下游之前处理 -Inf 极限，数值/资源/普通完整 Graph 与两平台三交付已通过，该卡已 CLOSED。
这些是成熟实现的准入修正，不是 Native 性能裁决；不复用未经适配的 ggml 原函数成本作为公平基线。

MatMul 的 Native 比较应保持调用方真实 RHS 物理布局：row-major 对 oneDNN；contiguous-K 转置布局
对 ggml。若完整消费者需要变换布局，必须把变换成本列入该完整消费形状，不能零成本替换输入布局。
