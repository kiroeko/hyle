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

## AUDITED：能力、API 与有限施工集（2026-09-12）

### Peer 与 consumer 事实

- ggml v0.23.0 把 `type`、`ne[]` 与 byte stride `nb[]` 分开；对 quantized type，`nb[0]` 是一个 block 的
  bytes，`nb[1]` 按 `ne[0] / block_elements` 推进。这证明“element stride × scalar size”不是通用 storage
  模型；
- ds4 `f62ca29` 独立保存 GGUF-compatible block type、block bytes 和 block elements，并对推理权重逐 row
  校验；Q4_0/Q8_0/K/IQ 等不是 ggml benchmark 私有姿势；
- oneDNN `6b7f1fbb` 的 memory descriptor 同时提供 plain strides、inner blocks、packed、grouped、CSR/COO
  encoding。HYLE 本轮不复制这些格式，但必须有不泄漏 oneDNN 类型的 opaque physical-format 退路；
- MLX 0.32.2 的公开 `quantized_matmul` 把 packed weight、scales、optional biases、group size、bits 与 mode
  分开；vLLM/SGLang 也大量使用 packed weight + 独立 scale/zero tensor。因此 storage format 只描述一块
  tensor storage，量化公式和关联 scale 仍属于 typed operation/provider，不能塞进一个万能 dtype；
- 当前 HYLE 只有 `tensor.cc` 和一个 native layout predicate 使用 element stride；迁移到 byte stride 不会
  把转换带进 kernel inner loop。Graph 只复制 TensorDesc/调用 `required_bytes()`，oneDNN 只应接受 plain
  canonical dense。

### 能力覆盖与承载裁决

| ID | 能力族 | 层级 | 语义 / owner | 当前状态 | 裁决 |
|---|---|---|---|---|---|
| CT30 | logical scalar type | 原子机制 | `DataType` 只描述数学 element；plain storage 才由 `data_type_size()` 给出 bytes | `PARTIAL`：当前同时充当物理格式 | `FIX` |
| CT31 | plain byte-strided layout | 原子机制 | dense convenience + 显式 non-negative byte stride；zero stride 仍只是描述，operation 决定 broadcast | `MISSING` | `FIX` |
| CT32 | open physical format identity | 原子机制 | 两个稳定 `uint64_t` 组成 value identity；无 pointer/string/global registry；provider typed factory 拥有具体常量 | `MISSING` | `FIX` |
| CT33 | one-axis block layout | 原子机制 | `StorageFormat::blocked()` 固定 identity、elements/block、bytes/block、power-of-two alignment；Tensor 固定 block axis 与 byte strides | `MISSING` | `FIX` |
| CT34 | opaque packed layout | 原子机制 | `StorageFormat::opaque()` + 每 Tensor 精确 byte extent；不可由 Compute 索引或任意 permutation | `MISSING` | `FIX` |
| CT35 | safe shape/storage arithmetic | 原子机制 | rank≤8，zero、divisibility、stride/span/alignment/size 全部 checked O(rank)；无 heap/vtable | `PARTIAL` | `FIX` |
| CT36 | zero-copy logical permutation | 复合功能 | plain/block byte strides 与 block axis 一起 permutation；opaque 拒绝假装 zero-copy | `PARTIAL` | `FIX` |
| CT37 | borrowed TensorView bridge | 复合功能 | view 精确收窄到 physical bytes；明显错位 offset 拒绝，base/native alignment 由能解释 handle 的 provider 终检 | `PARTIAL` | `FIX` |
| CT38 | provider support/fallback | 复合功能 | native/oneDNN 对有效但未知 encoded format 返回 `Unimplemented`，使有序 provider 可继续；畸形 descriptor 在构造时 `InvalidArgument` | `MISSING` | `FIX` |
| CT39 | sparse/multi-buffer quant metadata | 应用策略/后续机制 | indices/scales/zeros 是独立 typed tensors/operation inputs；不伪装成单 Buffer block | `OUT_OF_SCOPE` | `ROUTED` 到未来具体 op/provider |

全部现有公开 Tensor 入口映射到 CT30/31/35/36/37；本轮新增入口只映射 CT32--CT34。没有 benchmark-only
constructor，也不增加格式枚举大全。

### API / usability freeze

冻结后的普通 scalar 路线仍是：

```cpp
auto desc = hyle::compute::TensorDesc::dense(
    hyle::compute::DataType::kFloat32, shape).value();
auto view = hyle::compute::ConstTensorView::create(desc, buffer.view()).value();
```

provider author 的高级路线是：

```cpp
constexpr hyle::compute::StorageFormatId codec_id{vendor_namespace, format_value};
auto format = hyle::compute::StorageFormat::blocked(
    codec_id, 16, 12, 4).value();
auto physical = hyle::compute::TensorDesc::blocked_dense(
    hyle::compute::DataType::kFloat32, std::array{out, in}, format).value();
auto logical_weight = physical.transposed().value();
```

- `StorageFormatKind={kPlain,kBlocked,kOpaque}`；plain 的 ID 为空，encoded ID 必须非空；
- `TensorDesc::strided_bytes()` 取代含混的 `strided()` / `element_strides()`；不存在保留旧名字但偷偷换单位
  的兼容层。`byte_strides()` 对 opaque 返回空 span；
- `blocked_dense()` 的 block axis 是最后一轴；`blocked_strided()` 显式给 block axis 和每 storage unit 的
  byte stride。该轴 stride 的单位是一个 block，其他轴单位是一个逻辑 element；getter 明确返回
  `block_axis()`；
- `opaque()` 接受 logical dtype/shape、opaque format 与精确 bytes。opaque permutation 返回
  `Unimplemented`；具体 provider factory 可用其规则创建另一个描述，不能宣称通用 zero-copy；
- `required_alignment()` 是 allocation/provider 合同。TensorView 能检查 subview offset；资源 base/native
  handle 的最终地址或设备对齐只有 typed provider 可解释，不能为此扩大四字 BufferView；
- `StorageFormat`、`StorageFormatId` 与 TensorDesc 都是 self-contained、trivially copyable；没有 owner、动态
  分配、虚调用、注册表或 DLL 内静态地址。目标 `sizeof(TensorDesc) <= 192`；
- malformed schema/format 是 `InvalidArgument`，算术溢出是 `OutOfRange`，合法但当前 route 不懂的 encoding
  是 `Unimplemented`。只有最后一种允许 provider fallback；
- scale/zero/bias、quantization mode、packing/copy/prepack policy 不在 StorageFormat；它们保持 typed inputs/
  provider options，避免“一个 opaque 格式对象解释整个模型”。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| CV3-01 | logical/physical dtype 混合 | 在 `tensor.h/.cc` 增加 StorageFormat value 与三种 layout；换成 byte stride | plain/block/opaque 属性、边界与溢出测试 |
| CV3-02 | permutation 不携带 block axis | 重写 permutation/transpose/required-bytes/is-dense | plain 与 transposed ggml-like row 的 byte extent 相同；opaque 明确拒绝 |
| CV3-03 | TensorView 不认识 physical bytes/alignment | 精确收窄 physical span并拒绝显然错位 offset | short/offset/empty/resource 测试；不改变 Mem layout |
| CV3-04 | native layout predicate 仍按 element | 只迁移 Compute native/operation checks；encoded route返回 Unimplemented | v2 differential 与 fallback semantics 不变 |
| CV3-05 | oneDNN 可误认 encoded dense | provider 显式要求 plain + canonical dense；定向 replay | oneDNN 原 journey 通过，encoded 不到 vendor API |
| CV3-06 | Graph 复制/alias 用旧 byte 算术 | 只做编译迁移与 metadata replay，不增加 constant/region API | Graph v2 journey/测试通过；是否重开由实际产品变化裁决 |
| CV3-07 | 用户与架构文档仍描述 element stride | 更新 Compute guide/架构和 provider-author 限制 | docs checker、逐头、自然调用样例 |

这七项是全部产品改动；任何 ggml type factory、constant、region、量化 kernel、Graph partition、Mem layout 或
性能仪器改动都必须属于后续卡。

### P0 与七问预裁决

1. 第三方只提供需求证据，不进入本 factor；无新 dependency/license/fallback；
2. plain、one-axis block 与 opaque 三档覆盖当前真实 embedded-block 和成熟 packed seam；quant 公式由 typed
   op/额外 tensor 承载，不遗漏本轮 MUST；
3. 继续复用 Math checked arithmetic、Mem BufferView、Error Result，不复制 allocator/format registry；
4. 构造/比较/required-bytes 最坏 O(rank≤8)，无 allocation/lock/atomic/syscall/vtable；descriptor 只在冷
   metadata/binding 边界复制，不在 element loop；
5. 只有 C++ value；旧 C/Native FFI 不恢复，numeric identity 只为 shared/static 一致而不是 FFI；
6. 所有实现仍在 `compute` factor；没有新 target/option/dependency，三种 C++ 交付同头同语义；
7. scalar 默认不要求理解 format；高级用户通过 provider typed factory，provider author 才使用通用 constructor。

审计结论：范围、API 和有限 ledger 已冻结，进入 `AUDITED`。CV3-01--07 全部完成并通过预声明验证后才形成
candidate；实现若证明需要 owner/vtable/多 block axis，退回 `SCOPED`，不能顺手扩张。
