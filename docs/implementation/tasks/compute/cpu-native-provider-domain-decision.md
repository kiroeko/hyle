# CPU自研算子：逐精确域与成熟Provider裁决

<!-- MODULE REVIEW: SCOPED -->

- baseline：`155d1385c2b9a50eaa66a6c069b569f11daf4b03`，Windows canonical。用户要求：有成熟替代的所有自研算子都要裁决，
  不能因为低优先级后备就永久保留；只有唯一实现的精确域才直接保留。SIMD基础不删除。
- owner：本卡保留SCOPED的冻结自研裁决ledger，不是当前产品owner授权。`compute_provider_native`
  当前绑定[TC卡](cpu-aten-cast-stable-route.md) AUDITED；ggml同卡AUDITED；`compute_route_cpu`、`graph_route_cpu`
  当前绑定[SI卡](cpu-aten-storage-index-admission.md)并独立CLOSED；
  后续自研逐域裁决或新成熟接线需按新事实重新绑定owner并固定有限FIX。
  [标准CPU构建](../build/cpu-standard-provider-build-v1.md)与ggml优化程序接入已独立关闭，不能代替本卡。涉及oneDNN/ggml的具体接入修复，
  先据反例重新绑定相应owner并固定有限FIX，不借旧CLOSED/AUDITED记录授权。
- scope：现存Native全部payload数学域（MatMul、norm、activation/gate、reduce/softmax、
  attention/RoPE、index/transform/cast），相对于oneDNN、ggml、已接Torch成熟域逐项核对。
  TorchAO/ds4原版数学、ATen调用和checked生命周期adapter与自研数学分开。
- 证据：D4的宽泛KEEP_RESIDUAL不是新裁决；当前Direct默认先oneDNN100、ggml200、可用ATen350、Native1000，
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

## 接线遗漏专门排查（先于性能裁决）

2026-10-05 用户要求：连续发现准入遗漏后，先专门批量排查，不再逐个尝试接线或启动自研比较。
本卡原本已覆盖全部 Native payload；现固定这轮有限审计 A1–A3，不扩张公开能力或重审全库。
已提交的 GB1–3 验证完成后，优先完成本轮；其它新接线与 Native 性能/清退继续冻结。

| ID | 有限工作 | 必需输出与停止 |
|---|---|---|
| A1 | 将当前 `build_plan` 每个数学分支、dtype/量化/布局/mask/index-width/资源档对照已锁 ggml 与 oneDNN 的原版 API/CPU kernel；官方 Torch/ds4 和标准库借用域单列 | 一张覆盖表，区分已接、adapter 限制、真实语义差异、原库未提供；有函数名不等于合格，未接不等于 UNIQUE |
| A2 | 同一批普通/expert 冷准备 witness，覆盖全部现存泛型能力族和具名限制；三个 provider 返回事实保持，不计时 | 保存每行完整端口与 prepare 状态；成功只证明冷准入，不代签执行、精度、生命周期或性能；无效 public descriptor 不误计 provider 缺口 |
| A3 | 对表中有源码支持的关键限制，最多一批原版程序 witness：已验 rank4 MatMul/激活，加 scaled rank4/narrow Softmax、非因果 attention、Fill 与 i64 row-write 的具名调查 | 实际数值或源码拒绝证据；不能因 witness 未写而判唯一。最终批量 FIX/DEFER ledger 一次冻结，最多一轮具体失败纠正，不逐算子扩建工具 |

范围是成熟实现接线审计，不是增加自研目录、性能赛跑或完整模型再研究。
专门表与收据落在本卡引用的 evidence，完成后才确定下一批成熟接线及 Native 逐域裁决样本。

本轮 A1–A3 已完成[批量表与证据边界](../../../../evidence/correctness/cpu-ggml-mature-batch/adapter-audit.md)：
188 个合法案例、564 条实际冷准备；原程序的 scaled/masked/narrow rank4 Softmax、非因果整 attention、
Fill 与 unique I64 row-write 有实际 witness。未资格的特殊值/alias/duplicate/其它原 carrier 不冒充通过。
下一批成熟接缝按普通 tensor、transformer、index/layout 三族处理；本卡仍 SCOPED，Native 继续冻结。

## 当前源码域表（尚未裁决）

按实际已接同合同候选划界；不同 dtype、mask、alias、bounded shape、预算档位分别审计。
下表的 `DECISION_REQUIRED` 仍是未完成，不把 vendor 尚未接入的能力虚报为可调用路线。

| Native 能力族 | 成熟重叠候选 | 不能用重叠域结果直接删除的边界 | 动作 |
|---|---|---|---|
| MatMul | ggml rank2..4 plain F32 row-major/contiguous-K转置RHS、双向leading广播与K0；oneDNN非空exact dense row-major F32含小M | rank>4、非vendor可表达strides/空输出region/预算档；转换成本不得抹掉 | DECISION_REQUIRED；两种物理布局按实际消费者匹配，旧Native-vs-Native不签WIN |
| RMSNorm/LayerNorm | oneDNN dense浮点、input0精确原位；ggml可表达浮点、F32 accumulation | bounded RMS整符号域、受控预算、非vendor布局；原位不能一概归唯一 | DECISION_REQUIRED；普通out-of-place与oneDNN原位均须比较，旧Graph不能代签新融合 |
| Unary | ggml Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Sigmoid/ReLU三floating dtype；oneDNN dense F32稳定Softplus；ATen三dtype稳定SiLU/GELU/GELU-tanh/Softplus，rank<=8/odd-byte与限定exact alias | bounded/严格预算及原ggml有限尾部仍待具名核对 | DECISION_REQUIRED，按kind/dtype/精度与资源分开；MN成熟接线已验 |
| Binary/Select | ggml Add/Subtract/Multiply/Divide；oneDNN Bool Select广播/plain分支；ATen全15scalar exact Select及三floating number Min/Max/odd-byte/broadcast/NaN-signed-zero与限定dense exact alias | 其它alias/严格预算/bounded域 | DECISION_REQUIRED；SI/MN成熟接线已验 |
| Reduce | ggml Sum/Mean、Unknown/F32 accumulation；ATen三floating四kind、任意合法axes/keepdim/rank0..8/empty/byte stride，显式F32 accumulation，NaN与零号资格 | bounded/严格预算；native内部内存unknown | DECISION_REQUIRED；MN成熟接线已验 |
| GatedActivation | ggml SiLU packed/separate与-Inf边界；ATen三kind/三dtype packed-separate、rank<=8/odd-byte，稳定F32 activation+乘法后一次舍入，普通Graph下游已验 | 原ggml SiLU有限负尾部另核对；bounded/严格预算；公共gate禁止alias | DECISION_REQUIRED；MN成熟接线已验，完整Double/F32 cast/packing/原节点成本包含 |
| RoPE | ggml dense rank4、sequence-axis1、i32 positions、F32 frequencies | i64 positions、其它axis/rank/layout与checked strides | DECISION_REQUIRED，不偷换公共位置公式 |
| Softmax | ggml dense三floating dtype任意axis、finite scale、同类型dense additive广播，安全特殊行 | Bool mask/log-softmax、非dense/alias/bounded预算等剩余域 | DECISION_REQUIRED；包含安全score扫描/scale-add和原布局materialization成本 |
| SDPA | ggml unmasked BSHD/BHSD、三floating dtype、causal/noncausal、独立Dk/Dv与GQA | mask/alias/预算与具体非有限合同；不把同head Native与GQA完整程序混比 | DECISION_REQUIRED，只测同合同共同shape及真实资源/完成 |
| Gather/Scatter | ggml i32 indices浮点Gather、浮点Scatter Add；ATen全15scalar/I32-I64/各axis/多维indices/odd-byte stride Gather与最后逻辑写入Assign，exact dense input0 Assign alias | bounded/严格预算/其它alias；Assign不与Add比较 | DECISION_REQUIRED；SI成熟接线已验，late越界不写输出与失败槽重建 |
| Concat/Cast/Copy | ggml全15plain scalar位Copy/Concat、三floating dtype Cast；oneDNN plain reorder；ATen全15scalar Copy/Concat含odd/unaligned stride | FP8 numeric Cast、bounded整域与预算；宽载体不等于宽math | DECISION_REQUIRED；宽存储成熟接线已验，数值转换另资格 |
| View/Fill | ggml三floating dtype原精确scalar bits→repeat Fill；ATen全15scalar exact bits Fill；View为无数值payload的schema操作 | alias/bounded/严格预算仍单列，不能以adapter拒绝签唯一 | Fill DECISION_REQUIRED；全宽Fill成熟接线已验，View非payload不虚构胜负 |
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

### 成熟接线的当前资格增量

[GB批](cpu-ggml-mature-admission-batch.md)、[TF批](cpu-ggml-transformer-fill-admission.md)、[LB批](cpu-ggml-layout-broadcast-admission.md)、[SI批](cpu-aten-storage-index-admission.md)与[MN批](cpu-aten-numeric-admission.md)已分别真实关闭；上表更新为当前已接域。原188/564冷审计是不可变历史，不重写为新执行证据。RoPE/I64 positions、Bool/log-softmax、FP8 numeric Cast与其它具名shape/资源域候选仍需批量资格；I64 Gather/Assign已由SI接入，不再混列未接。oneDNN原REDUCTION API不等于当前primitive whitelist已启用。Native/SIMD算子优化、比较和清退继续冻结，拒绝不签UNIQUE，优先级不签WIN。


MN数值批已完成Windows/Linux120/120、三dtype普通Direct/Graph与三交付资格，上表更新为本批
已接域，不是性能WIN或CPU整体完成。已锁ggml的Sigmoid与SiLU scalar尾部仍用exp(−x)，可能在有限
负尾部溢出归零，特别是大value的gate；此具体源码疑点在CPU最终收口前必须做一次完整消费反例
核对，不由当前ATen后备或旧−Inf边界通过掩盖，也不借Native owner权限修改ggml。
