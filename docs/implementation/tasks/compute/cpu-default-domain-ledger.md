# CPU 数学域最终去留与默认绑定

2026-10-07补充：[三项新peer接线与清退](cpu-three-operator-route-closure.md)已替代本文中SDPA的保留决定：
Native SDPA整项源级删除，普通strided Copy转成熟接缝、真实declared-domain view物化保留；
CPU-only新core只应用17格新资格Copy/RMS规则。下文187/274及资源决定是原轮事实，不得作为新版当前优选。

这些比较是Native对当时默认选中的成熟完整调用，不是oneDNN/ggml/ATen所有重叠域的全组合比较。
当前Direct成熟叶顺序为oneDNN→ggml→ATen；Graph先匹配ggml/oneDNN完整region，再匹配成熟叶与有资格的Native叶。
17格规则只证明对已测成熟peer的优势，不证明全Provider最优；其它剩余自研族的历史依据也不自动取得新版资格。
CPU全重叠域best-of仍未完成，不能将本轮三项CLOSED解释为该项完成。

本账本归属 [当前 CPU 施工卡](cpu-default-domain-routing-final.md) 的 F4。
性能输入是 [已结束的 ND3 比较](../../../../evidence/performance/cpu-native-domain-verdict/README.md)，
不增加测量、算法优化或新的算子。原样本、置信区间和 AA 失败记录保留。

## 逐域裁决

原 278 场景中四个 Native MatMul 场景已随整项源级退役撤回；当前剩余 274 场景。

| 集合 | 数量 | 默认入口与源码裁决 |
|---|---:|---|
| 通过原 5% 完整调用收益门 | 187 | 只在已测硬件/软件、串行和精确端口/参数条件下 Native 优选 |
| 有效比较中的 Native LOSS | 13 | 同资格条件下 Native exact propose/prepare 冷拒绝；默认成熟实现 |
| 严格方向 WIN、未达 5% 完整收益的 Cast | 3 | 成熟优先；不宣称性能胜出或平局 |
| 未通过 AA 资格 | 71 | 成熟优先；不按未经资格的数据签 WIN/LOSS |

完整 ID 和既有统计在原性能 receipt.json；批量资格消费者按该输入逐个核对 274 场景，
不是重新比较耗时。当前规则的输入只含语义域，不携带 timing profile。

## 13 项冷拒绝

| 数学与端口 | 参数 | 场景数 |
|---|---|---:|
| 加权 RMSNorm F32 `[32,2048]`、权重 `[2048]` | epsilon=1e-5、F32 accumulation | 1 |
| Scatter Assign，base/output `[64,2048]`，updates `[32,2048]` | axis=0、checked；F32/I32、F16/I64、BF16/I32 | 3 |
| Bool Softmax 与 Bool LogSoftmax，F32/F16/BF16 `[32,2048]`，mask 同形 | axis=-1、scale=1、F32 accumulation | 6 |
| 等头 SDPA，Q/output `[1,32,4,64]`，K/V `[1,128,4,64]` F32 | BSHD、none mask、非 causal、scale=.125、F32 accumulation | 1 |
| Softplus F16/BF16 `[32,2048]` | 普通 Unary Softplus | 2 |

Softmax 原测试 variant=kind+2*BoolMask；variant 2 是 Bool Softmax，3 是 Bool LogSoftmax。
原人读表已纠正，原始实验不改。各项代码与其它已证 WIN 共享，不能删除共同 kernel 后
再假称保留了其优势域。拒绝不误伤 whole-domain Copy/RMSNorm 或原借用模型程序。

## 保留依据与已经删除的能力

| 源码族 | 明确保留依据 |
|---|---|
| Unary、Binary、Reduce、Norm、Softmax、SiLU Gate | 共享 dtype/kind kernel 在 187 集中确有对应完整调用优势；劣势 exact 域冷拒绝 |
| Select、Concat、Fill、Copy、Cast | 对应存储/转换族有已测优势；Fill 只优选数值 1.25，不泛化到零或其它值 |
| Gather、Assign、ScatterAdd | 对应 dtype/index-width/rows 有 WIN；Assign 的三项劣势不进入该优选 |
| F32/I32-position RoPE | 原 F32 phase 语义资格与两种 rows 的完整调用优势；不是旧错误 Double oracle |
| SDPA | rows32 已测 LOSS 冷拒绝；rows1 未通过 AA、成熟优先。已有 expert 正确性/明确 caller-workspace 资源合同保留，不签性能优势 |
| whole-domain Copy、固定 hidden 的 F32 RMSNorm | 已验派生 dense 形状/Caller workspace/严格资源和 prepared replay 能力；exact 胜负不撤回 whole-domain 资格 |
| 原 TorchAO/ATen、ds4 与 Scan/TopK 借用程序 | 成熟原程序或非 SIMD 数学实现；不冒称 HYLE 自研 kernel |
| SIMD 基础 | 独立通用产品，用户要求保留；本轮不按 Compute 输赢删除 |

Native MatMul 已物理删除（691 实现行与两文件），严格内存/高 rank prepared-only 独有能力按
用户决定撤回；普通数学由成熟实现接替。早前 18 个错误数值域已经退役，本轮不恢复。
本轮没有增加自研数学实现。保留源码不等于默认优选，不用 1000 后备本身证明价值。

## 非泛化条件

当前优选限 i9-13900K、vendor/family/model/stepping/brand/AVX2、Windows MSVC 195035728、Release（NDEBUG），
native_threads=1 且 executor concurrency<=1。身份在 Context 创建时缓存；端口和参数在 prepare
匹配，submit 和 Invocation layout 不变。Linux、新编译器/其它硬件不承接这批 WIN 标签。
其余模型、量化制品、dtype/资源/精度合同仍通过已有成熟规则和实际端口表达，不由此表外推。
Direct 优选 preference=50；Graph Native 叶为 175，原 ggml region=100、oneDNN region=150。
整片段资格始终先于叶，普通用户无需提供排序或计时资料。

验收完成情况和不可变候选以当前施工卡及其证据为准，本文不把尚未运行的门写为 PASS。
