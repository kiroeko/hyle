# F38：原版 CUDA 动态量化与完整 GLM 共享 FFN

候选 `896200a9d88b3cad858a179a0f087a42679bb28a`，Windows RTX 4070 / SM89 / Driver API13040。
本页签署局部动态投影与两种 GLM 完整共享 FFN；完整画像 ledger 仍为 **140/176**。

F40 适用域补注：本批 Flash 输入未达到 `swiglu_limit=10`，不能外推截断分支。F40 已补接原模型
截断，并通过跨阈值激活与共享 FFN 回归；新增混合注意力仍未通过，见
[当前检查点](../glm-official-mixed-checkpoint/README.md)。本页原始测试事实保留。

## 有限纠正与原门

[原失败及原因](../sglang-fp8-dynamic-checkpoint/README.md)保留。
普通scale量化改接同 SGLang pin 的原版 CUDA `per_token_group_quant_8bit_kernel`，直接使用
CUDA FP8 nearest-even转换；原算法和16/8/4/2/1整组launch规则不变。SM89原Triton量化程序的
中间F16截断不再进入产品；两个被替代的cubin数组已删除。BF16幂次量化和GEMM保持原身份。
没有新增HYLE数学、改测试输入或扩大误差门。

- [BF16/幂次72案例](bf16-receipt.json)、[F32的36案例](f32-receipt.json)通过原精度门。
- [原失败激活4组](failure-input-receipt.json)重新量化后，FP8 codes与scale均和原独立参考一致。
  该诊断文件中的最终down输出仍是旧候选的保留观测，不能当成新候选结果；新的整段结果见下项。
- [普通入口收据](receipt.json)：17个GoogleTest、4个CTest全部通过、零跳过。
  公共语义7项、Provider4项、Direct/Graph各3项；包含216次动态投影、72次预编码投影回归，
  GLM-5.3与GLM-5.3-Flash真实宽度共享FFN两入口共16次执行，均使用两轮变化输入。
- FFN实际连接动态量化→gate/up GEMM→成熟packed SiLU乘法→动态量化→down GEMM；
  三个可见中间输出仍分别检查原1 BF16 ULP门及原完整链门。不是孤立投影相加代签整段。
- 新增F32/BF16错位输入拒绝与1/2/3/4/8/16组launch覆盖。原生向量读取的16字节对齐要求
  在准备资源约束和提交检查中一致表达。既有OOM、并发独占workspace、重复执行、无热metadata
  分配和Completion保活继续通过。

当前候选增量构建8.83秒、所选测试3.01秒。公共模型输入事实13项通过。
常态成本仍是两次native launch、caller workspace `M*K+4*M*(K/128)` 字节，零host同步、
零权重展开；Driver内部持久内存仍报告unknown，没有填虚构零值。

## 来源与可重建性

[编译收据](compile-receipt.json)记录SGLang原始源、向量头和cubin身份。
[上游身份](flashinfer-identity.json)固定FlashInfer原始头与许可证；Windows原生编译只将
GNU inline属性写成CUDA `__forceinline__`，原vector加载/转换正文不变。
先前临时探针的模板实例化错误发生在运行前，修正显式指针类型后完成资格；不是数值调参。

产品只携带原生程序；普通构建和运行无需Python/Triton。维护者使用vendor内
`rebuild_cuda_quant.py`和CUDA13.3重新生成的头与当前已验头逐字节一致，见
[重建收据](regeneration-receipt.json)。FlashInfer许可证已接入同一CMake package声明。
独立source/static/shared消费仍由D5-A5签署，重建工具检查不代替三交付。

## 剩余

官方FP8 grouped专家、其它路线的动态FP8完整链、完整官方画像、A5三交付、A6性能及D6仍待完成。
[selector计划](validation-plan.json)列出的扩展门未全部运行，不以局部4个CTest代签。
未重跑P1；旧失败原样保留。统一[实现统计](loc.json)为 **59,312行/235文件**，不含测试与第三方。
