# Compute v3：provider-neutral block storage format

## Closure card

- component / owner：`compute` / Compute；round=`compute-v3-block-storage-format`；
- baseline commit：`360ac5b2c02801abdc02d82dad015d9a978244a6`；
- trigger 与证据：[`compute-provider-ggml-v1`](compute-provider-ggml-v1.md) 对 ggml v0.23.0、llama.cpp、
  ds4、vLLM 与 SGLang 的源码审计证明，真实量化权重同时具有逻辑 element type、编码 identity、block
  elements、block bytes、alignment 和 byte strides；当前 `TensorDesc` 只能表达 scalar-size/element-stride，
  是阻止已接受量化 provider/Graph journey 的公开合同缺口；
- 正常 consumer 任务：计算用户用 provider-specific typed factory 取得一个自描述 block storage format，
  为逻辑 Tensor 建立 dense 或合法 permutation 描述并绑定精确 byte span；Compute/Graph 可以做通用大小、
  越界、alias 和 identity 检查，而只有理解该 encoding 的 provider 才接受执行；
- 主产品纵切 / 具名来源：ggml/llama.cpp 与 ds4 的 Q4_0、Q8_0、Q4_K、Q6_K/IQ 类矩阵权重，后续由
  ggml CPU provider 消费；同时对照 vLLM/SGLang 将 packed weight 与 scale/zero tensor 分离的路线；
- 正交反偏置纵切：普通自定义工程 block codec（逻辑 f32、每 16 elements/12 bytes）经过 dense、permuted、
  required-bytes 和 TensorView 校验；不使用 ggml 枚举、模型名或 MatMul benchmark；
- 跨模块 public journey 与每段 owner：provider extension 拥有 typed encoding factory；Compute 只拥有稳定的
  format identity/geometry、TensorDesc 和 view 验证；Mem 继续拥有 opaque bytes；Graph 后续只复制描述并规划
  byte span；operation/provider 决定是否支持该 encoding；
- scope：重新设计 Tensor 的逻辑 dtype 与物理 storage format 分离；提供无字符串、可比较的 format identity，
  scalar 与 one-block-axis dense/byte-strided 描述，安全 permutation/required-bytes/alignment/TensorView；迁移
  Compute、Graph、oneDNN 和测试到明确 byte-layout 合同，并保持现有 scalar journey 自然；
- non-goals：列举任何 ggml/FP8/AWQ/GPTQ 格式、量化/反量化算法、scale/zero 策略、constant binding、Graph
  region/partition、provider 实现、负 stride、稀疏格式、模型文件、Native FFI；
- 直接依赖 / 直接消费者：保持 `compute -> core,error,mem` public 与 `math,platform,simd,util_core` private；
  直接消费者为 Graph、oneDNN provider、SCOPED ggml provider 和 typed custom-kernel 用户；不新增下层依赖；
- 性能 tier 与原因：`P0`；rank 上限仍为 8，全部验证/大小计算 O(rank)，不在 steady kernel loop。审计对象是
  TensorDesc 大小、scalar 快路径、整数运算、分配/锁/原子/虚调度账单；不为冷 metadata 制造纳秒 P1；
- 适用机器 / 仪器 / material margin：Windows、WSL Linux、两台 Apple 做 correctness/header/交付；P0 无
  material margin。若源码发现描述进入 steady inner loop，退回审计而不是补 synthetic benchmark；
- 调查与纠正预算：ggml/oneDNN/MLX/PyTorch packed-layout 各一遍；API 最多两轮；整数溢出/布局实现最多两轮；
- 预声明验证：plain/block dense、byte-strided、permutation/transposition、zero/one/max dimensions、block-axis
  divisibility、format identity、alignment、overflow、undersized/misaligned/wrong-resource view、native/oneDNN
  unsupported behavior、Graph metadata replay、逐头、三平台、sanitizer、source/static/shared 与 P0；
- stop / blocker 条件：若一个通用 block-axis + byte-stride 合同不能覆盖 ggml 与独立 codec，或需要在普通
  TensorView 增加 owner/heap/vtable/global registry，停止并重新 scope；不得把 ggml enum 放进基础 Compute，
  不得用 `UInt8`、magic integer、字符串或 benchmark-only constructor 逃避类型设计。

## SCOPED 待审计项

1. public vocabulary 应是 `StorageFormat` / `StorageFormatId` 还是更能区分 logical `DataType` 的名字；typed
   provider factory 如何创建 identity 而普通用户不手填 magic ID；
2. block 只允许一个逻辑 axis 是否覆盖 ggml/独立 codec，并为未来 tile/sparse 留出不破坏当前语义的边界；
3. public stride 是否统一为 bytes；plain convenience 如何避免用户在 element/byte 之间静默误用；
4. dense storage 的 block axis、permutation 后 axis identity、row padding和总 byte span如何在 O(rank) 安全计算；
5. zero-size tensor、block-axis 非整除、broadcast zero stride、overlap 与 alignment 的确切合法性；
6. `TensorDesc` 大小、trivial copy、共享库 ABI 内构造/比较与 provider extension identity 生命周期；
7. native/oneDNN 只接受 plain format 的错误应在 prepare 冷路径返回 `Unimplemented` 还是
   `InvalidArgument`；fallback 只能对前者继续；
8. 现有 Graph candidate 和 oneDNN candidate 是否仅需 targeted replay，还是实现改动形成独立 reopen trigger。

AUDITED 前不得修改 `include/hyle/compute`、`src/compute`、Graph、provider 或测试产品合同。本卡先冻结
coverage map、API sketch 与有限改动集，再进入实现。
