# CUDA U5：冗余自研域清退与独有资源边界

<!-- MODULE REVIEW: SCOPED -->

- baseline `fcad07ae0`；Windows唯一产品真相，工作树干净。用户已授权继续goal/删除有完整成熟替代且无完整优势的自研；不把CPU严格能力撤回决定移植CUDA。
- owner：compute_provider_cuda；必要时compute_route_cuda仅默认/诊断迁移，不改公共Compute/Graph算法。
- trigger：U3成熟覆盖已接齐、U4逐operation成熟默认已验证；有限AA失败不能产生native优势证书。当前native整族仍含与成熟重复域和真实独有需求，须明确分开。
- consumer：当前固定模型Direct/Graph、FP32中间与一次storage rounding、原resource/capture/失败旅程；正交为高rank/odd-stride/Bool mask/empty与独立invocation资源限制。
- budget：一遍6源码族的有限对账；固定可删子域后批量改产品/一次对应双平台及三交付增量。不得追加性能采样、扩新算子、自研优化或新治理；发现完整独有能力需撤回才向用户给具体方案。
- non-goals：不改CPU/SIMD/Mem、Apple/Vulkan，SGLang/DeepSeek/CCCL/ggml/ATen/FlashInfer原数学与资源胶水不算自研；不为了删文件扩大第三方数学。

## 六族原源码与成熟替代对账

| 源码族 | 当前成熟重复域 | 真实未替代域／审计裁决 |
|---|---|---|
| native_kernel.cu：gated | SGLang精确packed向量三种activation，原Prepared/capture、零private一致 | separate、非完整向量宽度、scalar alignment仍unique；允许物理裁去完整向量高alignment子域，不能拿该域独有要求挽留重复实现 |
| norm_kernel.cu：RMS/Layer | SGLang matching half weighted RMS；ggml普通F32/F16/BF16 affine norm | 混合F32参数、empty、prepared/capture/no-private与低alignment仍未完全替代；若同族共享模板不能无损切走数学，明确KEEP unique而非WIN |
| tensor_kernel.cu | ggml Copy/Cast/算术，SGLang vector activation、Torch stable Softplus | View为通用alias机制；Fill/Select/number Min/Max、odd byte-stride、empty、rank8不可折叠及strict/capture仍unique |
| reduction_kernel.cu | ggml普通Sum/Mean/Softmax | Bool mask/LogSoftmax/NaN min/max/empty identities/strict prepared及odd stride未替代 |
| concat_kernel.cu | ggml非空、可表示stride storage concat | empty group、rank8复杂stride、scalar subviews/strict capture仍有具体能力差异 |
| scatter_kernel.cu | 当前ggml未接公共Scatter，原CCCL只借排序并非整个Scatter语义 | 当前唯一完整checked Scatter overwrite/add，多次重复ID与失败输出语义；KEEP，非性能优势 |

以上来自cuda_provider.cc各family分派、六个select/prepare、ggml tensor_program.cc以及各public provider合同；不按文件名把所有CUDA内容判自研。
本阶段不能声称“落后已删”：U4没有合格LOSS，也无WIN。按授权成熟ENGINEERING_DEFAULT可裁重复源码域；具名独有数学/资源先保留，后续真实完整替代才删。

## 待固定有限 FIX

- G1：gated族把packed且完整16-byte vector/alignment的同合同域改由原SGLang承载，去掉该域自研kernel实例/接入；scalar-alignment、tail、separate能力和真实失败诊断保持。未变原SGLang程序无需重新性能比较。
- G2：norm/tensor/reduction/concat/scatter逐族核对上述unique事实，移除任何仅为旧benchmark保留的独立入口/分支；没有完整替代时显式KEEP并列具体缺口，不扩自研或签WIN。
- G3：ordinary default与expert capability说明、对账/红队、Windows/WSL source/static/shared受影响小纵切；复用U23数学和U4 defaults不变域，重型串行j2/testj1，统一LOC/清理。

本卡SCOPED；有限FIX最终AUDITED及registry改绑之前不改产品。U4原性能FAIL/INCONCLUSIVE保留，公开默认是工程取舍，未测硬件不PASS。
