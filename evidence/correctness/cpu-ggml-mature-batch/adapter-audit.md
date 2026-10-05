# CPU 成熟库接线遗漏：批量审计

审计产品 `ef9defea07bbc576a4c6bff1ce0d07e4e53acd19`，Windows canonical；由
[逐域卡 A1–A3](../../../docs/implementation/tasks/compute/cpu-native-provider-domain-decision.md)持有。
本表审计现存 Native payload 的全部能力族，不签性能胜负，不增加自研数学，不将已接域等同完整功能。

## 实际冷准备矩阵

同一组 **188 个合法 schema/端口案例 × ggml/oneDNN/Native = 564 条**实际准备记录，
schema 拒绝 0、其它错误 0。每行记录 operation 档位、dtype、shape、byte strides、plain/dense、
推导输出与 Provider 结果/错误。上下文使用已资格化普通 CPU device，专家绑定只用于识别承载方。

| Provider | 冷准入 | Unimplemented |
|---|---:|---:|
| ggml | 91 | 97 |
| oneDNN | 14 | 174 |
| Native | 164 | 24 |

这些数量是 witness 数，不是算子数量、覆盖百分比或正确性 PASS。尤其不能从 Native 准入更多推导
Native 更好，也不能从其它 Provider 拒绝推导库没有能力。公开 dtype enum 与具体矩阵源码一起归档。
冷准备未运行 payload；执行/数值/alias/OOM/资源/三交付的资格只由各接入卡实际验收。

## 全部现存自研能力族回映

`ADAPTER_LIMIT` 是接线不足；`SEMANTIC_CHECK` 是尚需同合同准入；`BORROW` 是原版/标准库，
不是待比较自研数学。未知域不写 UNIQUE。以下覆盖的是现存能力族，未声称列举每个任意 shape。

| Native 族/精确边界 | 原库能力与当前接线事实 | 本次裁决 |
|---|---|---|
| MatMul F32 row-major rank2 | oneDNN 原 BRGEMM，M1 已修；冷准备准入 | 已接，后续同布局比较 |
| MatMul F32 contiguous-K rank2..4 | 原 ggml `mul_mat` 支持；GB2 修原 rank2 条件，变化输入及混合广播已验 | 接线遗漏已修；本批完整资格见当前卡 |
| MatMul F32 row-major batch、左小右大 NumPy 广播、rank>4、空K | 当前两 mature adapter 拒绝；ggml transpose/cont、oneDNN `matmul` 的原始批次描述另有能力，不能拿 BRGEMM adapter 限制代替库限制 | ADAPTER_LIMIT/SEMANTIC_CHECK；转换/重复/空 identity 成本单列，暂不签唯一 |
| Unary 已接 Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Sigmoid | 原 ggml；三 dtype witness 准入，scalar/rank/stride/alias 域另划 | 已接；准入不代签所有非有限/alias |
| ReLU | ggml 原核 NaN→0；GB1 仅恢复 NaN，普通 Direct/整 region 已验 | 接缝遗漏已修；外部 alias 仍拒绝，不能当成 Native 原位档 |
| SiLU/GELU/GELU-tanh、narrow Softplus | ggml 有函数，SiLU 有负尾部 underflow，GELU 默认查表/erf cancellation；oneDNN 有 `eltwise_swish/gelu_erf/gelu_tanh/soft_relu`，当前仅 F32 Softplus；锁定 ATen 也有原函数，当前仅 lowp 程序接入 | SEMANTIC_CHECK + ADAPTER_LIMIT；先查原精度配置/具名成熟 carrier，不写自研、不开性能比赛 |
| Gated SiLU/两个 GELU、packed/separate | 原 `swiglu/geglu_erf/geglu{,_split}` 都存在；只 SiLU 已接；finite gate × 大/Inf value 会放大尾部差异 | SiLU 的 -Inf 接缝不代签新增尾部域；两个 GELU 是未接原能力，需按 F32 中间语义资格 |
| Binary 4种算术与 Min/Max | 4种已接 ggml；该 pin 无通用 binary min/max，oneDNN `binary_min/max` 与 ATen 有成熟能力，但 NaN/signed-zero 不等于公共 number min/max | 已接算术；Min/Max SEMANTIC_CHECK，不能因为 ggml 没函数直接保留自研 |
| Bool Select、宽 scalar 原位/广播 | oneDNN 已接 Bool 条件和 plain branch；ggml 当前没有同名通用选择，ATen `where` 有候选 | 现存成熟域已接；更宽类型/alias/位保留的精确域单列 |
| Reduce Sum/Mean、Min/Max、任意 axes/empty | ggml Sum/Mean 已映射 axes/empty；该 pin 无公共 min/max reduction，oneDNN `reduction` 有对应算法；ATen `amin/amax` 等有候选 | Min/Max ADAPTER_LIMIT/SEMANTIC_CHECK，含空 identity 和传播 NaN；不是已证唯一 |
| RMSNorm/LayerNorm | oneDNN/ggml 三 dtype 原能力已接；Native caller workspace、原位、受控预算/全符号域与成熟 Managed/Prepared 档需真实匹配 | 已接重叠；资源差异独立，不能用 out-of-place 耗时删除原位能力，也不能把预算约束泛化为唯一数学 |
| Softmax/log-softmax、scale/mask、rank/axis/narrow | 当前 ggml 只 rank2 F32 axis−1/scale1/unmasked；`soft_max_ext` 原版 rank4 scale/additive mask、F32/F16/BF16 cast 程序实际通过；oneDNN `softmax_forward` 和 ATen 有其它原档 | 已证 ADAPTER_LIMIT；Bool mask、log-softmax、其它 axis、全 mask/NaN/Inf 仍 SEMANTIC_CHECK |
| SDPA causal/noncausal、BSHD/BHSD、mask/depth/heads | 当前只 causal BSHD；原 mul_mat→soft_max_ext→mul_mat 非因果整 region 与 double oracle 实际通过；F16 flash 原 API 也接受可空 mask | 已证非因果接线限制；layout/mask/不同depth/GQA 与失败成本分别资格，不以现有谓词签唯一 |
| RoPE rank/sequence-axis、i32/i64 position | 原 rope 支持自己的 n-dimensional tensor 与 I32 position；当前仅 rank4 axis1。其它轴可调查 view/cont；i64→i32 要 checked payload boundary，不得截断 | ADAPTER_LIMIT/SEMANTIC_CHECK；原频率公式、长位置、layout、alias 验收保持，未证唯一 |
| Gather scalar/rank/layout/i32/i64 | 已接 ggml I32 index/floating 及具名 GGUF table；原 get_rows CPU 索引要求 I32，ATen index_select/gather 有 I64 候选 | index width 是原档差异，原库候选仍存在；checked 越界不写输出与纯 storage bits 必须匹配 |
| Scatter Add/Assign、duplicate index | 原 get_rows_back 已接浮点 Add/I32；原 set_rows 支持 I32/I64，unique I64 实测通过，但原 header 明确重复 destination 行为未定义 | Assign 不可直接冒充公共 last-wins；候选需受控去重/原语义证据，不能为性能删除该合同 |
| Concat/Copy、宽 scalar/blocked storage | ggml 已接可表示 dtype/layout，原 concat 任意 byte-width 有代码；oneDNN 原 byte reorder 已接 Bool/F64/I64 Copy；Native plain-storage/预算域单列 | 已接域与 adapter type/layout 限制分开，不能把表示限制当数学能力缺失 |
| Cast float/FP8、bits/alias | ggml float casts、oneDNN byte Copy 已接；oneDNN/ATen 原 dtype conversion 的数值承诺另查，byte reorder 不等于数值 Cast；Native FP8 Cast 调 SIMD primitive | 原能力与数值合同资格不足；SIMD primitive 独立用途保留，不据未接成熟 Cast 签 KEEP_WIN |
| Fill、View、空/原位/预算 | 原 ggml Fill F32/F16 实际通过，BF16 可原 F32 Fill→Cast；目前未接。View 是 HYLE 无 payload 的资源/描述边界 | Fill 有成熟候选，不能称全域唯一；View 基础保留，Fill 的宽 scalar/位保持/零成本资源路径独立裁决 |
| Scan/TopK 接缝 | `standard_index.cc` 复用 inclusive_scan/partial_sort；原 ggml top_k 的 tie/index 档不自动等于已登记稳定语义 | BORROW/STANDARD，不重添旧 Native 数学 |
| ds4/TorchAO/ATen lowp 模型程序 | `ds4_quantization.cc`/`torch_cpu_program.cc` 原版数学；GGUF/Grouped/Hadamard/attention/model region 由已登记原程序承载，已有能力不是自研 payload | BORROW；本审计不重开已关闭官方数学或伪造模型覆盖 PASS |

## 原版程序 witness 与证据边界

`original-broad-program.log`：6 个 scaled rank4 Softmax（3 dtype × 2 mask），最大绝对误差依次约
1.7e-8、6.1e-5、4.8e-4；F32 非因果整 attention 最大约2.65e-8；3 dtype Fill 和 unique I64 row-write
通过。F16/BF16 Softmax 明确使用原 F32 cast/softmax/output cast，是原成熟组合，不是原半精度 kernel。
这些有限/有界 witness 证明下一步接线调查有具体依据，不证明特殊值、所有掩码、索引重复或普通 Graph 已验。
源码另证 set_rows duplicate destination 为 undefined；因此没有运行该 UB 来追求偶然 PASS。

源码锚点与 hash 在 `source-ledger.json`；原 ggml pin `456172ec733a135778adcd32d00e576a58232e45`，
oneDNN 3.13.2，ATen 原 SDK 2.8.0+cpu，由 dependency lock 指认。缓存路径只是来源定位，不是产品源码来源。
审计没有修改原库、ISA/config、Native payload 或性能仪器。

## 下一批冻结顺序

1. 先补齐成熟 carrier 的数值/资源准入：普通 tensor（activation、Min/Max/reduce、Cast/Fill）、
   transformer（Softmax/SDPA/RoPE）、index/layout（I64/Assign/宽 scalar/alias）。按以上三族批量处理，
   不再为每个新函数开一轮全量 SDK 验证。
2. 每批使用已有原 kernel/region；所有 caller/workspace/alias/non-finite/checked-failure 差异列入有限 FIX 或
   具名拒绝，不能静默放宽。oneDNN/ATen 尚未实测的通用候选先实际准入，再写 BORROW 支持。
3. 在全部成熟候选 ledger 已处理后，再固定 Native P1/清退批次与精确域规则；SIMD 基础保持。

本轮审计 A1 的能力族回映、A2 的实际冷准备与 A3 的有限原版 witness 已完成；
成熟能力接线本身尚未完成，Native card 仍 SCOPED，不签 AUDITED 性能样本或 CPU 完成。
