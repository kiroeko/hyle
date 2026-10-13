# 单机推理基础能力审计证据

2026-10-13，[审计结论](../../../docs/implementation/tasks/project/inference-single-machine-foundation-audit-2026-10.md)
以单GPU、一个CPU与一个GPU联合计算、CPU-only为目标，覆盖单/多请求及CUDA、Apple、Vulkan
实际资格。HYLE baseline为67192441c70e8e07b1cae1b09d221114b738c9ca。

本轮是源码与公开合同审计。18个旅程族、7个实际路线和10组候选缺口不代表整模型数值、性能
或所有新域已通过；原CLOSED状态保持其原有限范围。没有修改产品源码/API、模块状态、构建
系统、第三方锁或验证工具，没有运行全库构建或P1。

## 固定输入

- [source-manifest.json](source-manifest.json)：三peer精确Git对象、归档SHA256、原checkout
  HEAD/status前后身份。使用fetch精确对象和git archive，不移动原checkout分支或覆盖源码。
- [public-api-inventory.json](public-api-inventory.json)：97个Compute/Graph头的SHA256及导航，
  68个公开头全部映射到旅程族，29个detail头单列为内部导航。声明正则只作索引，不能证明
  已逐个运行或完整审查每个方法。
- [source-anchors.json](source-anchors.json)：报告所引实现与补充反证文件的SHA256、行锚点和
  原源码行。仅描述列出的调用链与合同，不把未引用源码都声明为完整语义审计。

三peer固定对象为SGLang7224fce1b5e488c9916f20ec0556d696e595d512、
vLLM98f721116227f0f48690deb05c815c057bac0668、
llama.cpp cbe28bb3cb0758f0c566a9a3561766821aeb0748。公开出处使用固定commit URL；
获取方式是实际本地Git对象。GitHub页面读取cache miss没有转成缺口或外部测试结论。

## 裁决依据

G02是公开组合形状的API_GAP：graph_batch保存单Queue并调用单Queue submit，而原
LeasedInvocation在多lane graph上明确要求完整bindings；native hybrid也没有PagedArenaRead
绑定。G01/G04/G05/G06有明确参考成本或缺席域；其严重程度依目标模型/机器和消费者裁决。
G03/G07–G10需最小消费者把已有正确路径、通用glue和实际高性能成本进一步分开。

G01的示例计算为4×32×32×4096×128=2147483648 bytes（2 GiB），仅包括参考recipe的F16/BF16
K/V gathered storage；不是进程/设备峰值resident测量，也不包括之后F32 casts或其他scratch。
CPU ggml fused SDPA、Apple MLX fast SDPA、原生affine quantized matmul、F1 exclusive输入、
F2 producer及底层multi-queue replay均作为反例保留，避免把已有机制判成全域缺失。

## 验证范围

只选文档/closure机械检查，保留native build、correctness、headers、delivery、performance
的跳过原因。验证记录见[validation-result.json](validation-result.json)。源码定位及成本计算
是研究证据，不签整模型性能PASS。模块registry未变化，不需要状态转换previous-ref证明。
selector记录中的candidate_tree是记录生成前实际被检查的工作树；随后保存该记录自身、补
来源限定和族到缺口映射后，再直接执行docs/closure/diff检查。不得把前者树身份冒充最终提交。
