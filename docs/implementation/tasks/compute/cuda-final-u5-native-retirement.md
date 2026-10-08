# CUDA U5：冗余自研域清退与独有资源边界

<!-- MODULE REVIEW: AUDITED -->

- baseline `fcad07ae0`；Windows唯一产品真相，工作树干净。用户已授权继续goal/删除有完整成熟替代且无完整优势的自研；不把CPU严格能力撤回决定移植CUDA。
- owner：compute_provider_cuda；必要时compute_route_cuda仅默认/诊断迁移，不改公共Compute/Graph算法。
- trigger：U3成熟覆盖已接齐、U4逐operation成熟默认已验证；有限AA失败不能产生native优势证书。当前native整族仍含与成熟重复域和真实独有需求，须明确分开。
- consumer：当前固定模型Direct/Graph、FP32中间与一次storage rounding、原resource/capture/失败旅程；正交为高rank/odd-stride/Bool mask/empty与独立invocation资源限制。
- budget：一遍6源码族的有限对账；固定可删子域后批量改产品/一次对应双平台及三交付增量。不得追加性能采样、扩新算子、自研优化或新治理；发现完整独有能力需撤回才向用户给具体方案。
- non-goals：不改CPU/SIMD/Mem、Apple/Vulkan，SGLang/DeepSeek/CCCL/ggml/ATen/FlashInfer原数学与资源胶水不算自研；不为了删文件扩大第三方数学。

## 六族原源码与成熟替代对账

| 源码族 | 当前成熟重复域 | 真实未替代域／审计裁决 |
|---|---|---|
| native_kernel.cu：gated | SGLang精确packed向量三种activation，原Prepared/capture、零private一致 | separate、非完整向量宽度、scalar alignment仍unique；原一个kernel实例同时接完整向量和低alignment，不能只按shape删资格 |
| norm_kernel.cu：RMS/Layer | SGLang matching half weighted RMS；ggml普通F32/F16/BF16 affine norm | 混合F32参数、empty、prepared/capture/no-private与低alignment仍未完全替代；若同族共享模板不能无损切走数学，明确KEEP unique而非WIN |
| tensor_kernel.cu | ggml Copy/Cast/算术，SGLang vector activation、Torch stable Softplus | View为通用alias机制；Fill/Select/number Min/Max、odd byte-stride、empty、rank8不可折叠及strict/capture仍unique |
| reduction_kernel.cu | ggml普通Sum/Mean/Softmax | Bool mask/LogSoftmax/NaN min/max/empty identities/strict prepared及odd stride未替代 |
| concat_kernel.cu | ggml非空、可表示stride storage concat | empty group、rank8复杂stride、scalar subviews/strict capture仍有具体能力差异 |
| scatter_kernel.cu | 当前ggml未接公共Scatter，原CCCL只借排序并非整个Scatter语义 | 当前唯一完整checked Scatter overwrite/add，多次重复ID与失败输出语义；KEEP，非性能优势 |

以上来自cuda_provider.cc各family分派、六个select/prepare、ggml tensor_program.cc以及各public provider合同；不按文件名把所有CUDA内容判自研。
本阶段不能声称“落后已删”：U4没有合格LOSS，也无WIN。按授权成熟ENGINEERING_DEFAULT可裁重复源码域；具名独有数学/资源先保留，后续真实完整替代才删。

## 待固定有限 FIX

- G1：核对gated同shape而不同绑定alignment的事实；不是每个shape都具备完整成熟替代。普通合格vector已优先SGLang，不为减少文件而重写cold/hot dispatch或新造scalar/vector双轨。
- G2：norm/tensor/reduction/concat/scatter逐族核对上述unique事实，移除任何仅为旧benchmark保留的独立入口/分支；没有完整替代时显式KEEP并列具体缺口，不扩自研或签WIN。
- G3：ordinary default与expert capability说明、对账/红队、Windows/WSL source/static/shared受影响小纵切；复用U23数学和U4 defaults不变域，重型串行j2/testj1，统一LOC/清理。

本卡SCOPED；有限FIX最终AUDITED及registry改绑之前不改产品。U4原性能FAIL/INCONCLUSIVE保留，公开默认是工程取舍，未测硬件不PASS。

## 有限审计结果：不能把部分替代写成完整替代

六个源码族的每个独有域已沿select/prepare/submit核对。关键具名反例为
`tests/compute_provider_cuda/cuda_provider_test.cc:VectorRejectionPreservesSlotAndScalarAlignmentCapability`：
H128相同packed request在offset=sizeof(T) subview上SGLang拒绝而native实际可运行。
ProviderOperationRequest只有descriptor、无actual pointer；同一个native Prepared必须仍可绑定
这些低alignment指针，不能在propose按shape伪称全替代，或在每次submit以指针改变赢家。
native_kernel.cu一套同shape scalar程序承载这项能力；高alignment能力不是独立可删math TU。

本有限审计没有发现可**整族**无损删除且有完整成熟替代的自研族。六族KEEP均由上表明确
未替代数学/资源需要支撑，**不是WIN/SOTA**。原默认已把成熟合格重叠优先、native1000 fallback；
不因本结论继续挽留优化，也不将user的CPU MatMul strict撤回决定视作CUDA能力撤回授权。
是否撤回这些公开prepared/capture/stride/empty等域属于具体产品决定；先交付完整清单，
不为“清退”顺手扩大ATen/新写adapter数学来跨过当前范围。

首次整族审计只有G3的文档/能力清单和残留别名误述纠正；G2未发现仅旧
benchmark使用的独立public entry。普通Compute用户不选择内部self provider，路线控制/完成和
borrowed库保留。本轮不建、不重跑P1；原U23/U4对应correctness/source/static/shared实际
证据逐域复用，适用strict/alias/empty已在固定完整源批次运行。不把本次审计写成新的性能验收。

## AUDITED：算术与索引分开清退

baseline `6513a899c70ad62c148b216b2625afd1a32bccaa`。上面的整族结论不能反过来保护族内
冗余数学：native_kernel.cu的三种gate activation公式与原SGLang apply_activation_f32逐表达式
相同；tensor_kernel.cu的三种stable Unary公式与已明确适配的apply_unary_activation_f32相同。
低alignment/stride/尾宽的独有需求来自索引及binding，不要求HYLE复制这六个公式。

固定G4：物理删除六份重复公式，两个原scalar/index kernel直接复用已交付、锁定SGLang
inline arithmetic；保留原索引、F32中间、一次RNE、alias/empty/资源/Prepared/capture/失败。
不改vendor原文件，不新增runtime dispatch/状态或性能规则，provider身份的HYLE版本更新为
本次math借入实现；原SGLang版本不变。这是BORROW，不是自研优化或性能挽留。
其它五族具名unique KEEP，暂无可整族无损删除的完整成熟替代；不声称所有CUDA self已删除。

G4验收固定：原gate三种storage/activation、packed/separate/tail/special值、scalar subview、
独立slot/close/metadata失败，以及通用Unary strided/special/alias原测试；普通Direct/Graph
FFN/capture对应消费者。Windows/WSL各source与relocated/offline static/shared小纵切，
source math通过后交付只重验本改动的借入算术和实际link/load，不重复559合同或全部模型。
复用现producer/cache，build2/test1串行；一个实现批次、至多一次具名纠正、不跑P1。
公开合同/编译依赖图/产品select不变，原license随provider SDK已有交付。先独立提交本AUDITED
记录并将compute_provider_cuda现AUDITED owner改绑后才实施。

首次Win shared源码link检出U4新增expert-factory测试漏声明cuBLAS test依赖；static通过
来自传递archive，shared不得继承。仅补测试DEPS，不改产品/公开依赖图/数学；原FAIL保留，
同build ID一次具名纠正，configure proof复用，增量CMake自更新。其它无关模块不重验。
