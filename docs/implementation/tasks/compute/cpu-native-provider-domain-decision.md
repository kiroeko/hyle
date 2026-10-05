# CPU自研算子：逐精确域与成熟Provider裁决

<!-- MODULE REVIEW: SCOPED -->

- baseline：`59f68e326`，Windows canonical。用户要求：有成熟替代的所有自研算子都要裁决，
  不能因为低优先级后备就永久保留；只有唯一实现的精确域才直接保留。SIMD基础不删除。
- owner：后续涉及实际产品修改的Native/CPU route按当前registry绑定；先完成CPU标准构建卡。
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
| MatMul | ggml 非空 rank2 plain F32，row-major/转置权重；oneDNN 非空 dense F32、serial M≥16 | Native F32 NumPy batch 广播、空K、非vendor可表达批次布局 | DECISION_REQUIRED；旧M1 Native-vs-Native不是胜出 |
| RMSNorm/LayerNorm | oneDNN dense浮点；ggml可表达浮点、F32 accumulation | bounded RMS整符号域、受控预算、in-place；普通托管out-of-place仍须比较 | DECISION_REQUIRED；旧Graph结果不能代签新融合 |
| Unary | ggml Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Sigmoid；oneDNN dense F32稳定Softplus | 未接ReLU/SiLU/GELU/GELU-tanh、narrow Softplus、layout/alias/非有限值 | DECISION_REQUIRED，按kind/dtype分开 |
| Binary/Select | ggml Add/Subtract/Multiply/Divide；oneDNN Bool Select广播/plain分支 | number Min/Max的NaN/signed-zero、特殊alias/layout/scalar-width | DECISION_REQUIRED |
| Reduce | ggml Sum/Mean、Unknown/F32 accumulation | Min/Max语义、部分axes/rank/empty/checked strides | DECISION_REQUIRED |
| GatedActivation | ggml SiLU、合格packed/separate浮点端口 | GELU/GELU-tanh当前无相同已接片段；alias/layout单列 | DECISION_REQUIRED |
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

默认ggml当前是旧M1的安全基线profile，禁用AVX2/FMA/F16C/BMI2；不能与Native SIMD比较后把配置
优势误写成算法WIN。[独立有限接入卡](cpu-ggml-isa-admission-v1.md)先保留基线并接通原版优化程序，
此后才固定本卡不可变批次。该前置不授权新数学、GPU施工或无界ISA/benchmark扩张。
