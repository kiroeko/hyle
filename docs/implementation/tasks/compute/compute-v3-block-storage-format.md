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

## Implementation checkpoint（2026-09-12）

- CV3-01--03：`StorageFormatId`、plain/block/opaque `StorageFormat`、统一 byte-stride `TensorDesc`、
  checked required-bytes/permutation/alignment 与精确收窄的 borrowed TensorView 已完成；旧
  `TensorDesc::strided()` / `element_strides()` 没有兼容残留；
- CV3-04--06：native route 只接收它实际理解的 plain layout；合法 encoded weight 保持逻辑 shape 并以
  `Unimplemented` 继续 provider fallback；oneDNN 在调用 vendor API 前拒绝 non-plain layout；Graph 只复制
  descriptor，并把内部 workspace alignment 提升到 descriptor requirement，没有加入 constant/region 能力；
- CV3-07：Compute/Graph user guide 与 Compute architecture 已改成 logical dtype + physical byte-layout
  词汇，明确 block divisibility、opaque 限制、typed-provider owner、无隐藏 padding/copy 与当前没有量化
  kernel；新入口在 `bench/` 中没有引用；
- 红队把 submit 时最多三次 alignment `%` 改为 power-of-two bit mask。StorageFormat factory 已保证该前提，
  因而这只是消除真实调用边界整数除法，不增加 metadata、分支、公开 knob 或 benchmark shape；
- immutable product candidate 为 `ada35161916d4e7b89ccabe846897cbea9a43808`。从 baseline 到该提交的产品
  改动只落在有限 ledger 指定的 Compute、oneDNN predicate 与 Graph metadata replay；没有新增 target、
  option、第三方依赖或 FFI。

## Candidate validation checkpoint（2026-09-12）

候选形成前完成的验证如下；远程与 WSL checkout 只接收 exact archive，没有反向覆盖 Windows canonical
checkout：

| 风险 | 候选绑定结果 |
|---|---|
| correctness / boundary | Windows 与 Linux Compute `36/36`、Graph `23/23`；Windows oneDNN provider `10/10`；两台 Apple Silicon Compute `36/36`、Graph `23/23` |
| sanitizer | WSL GCC 14 ASan 与 UBSan 的 Compute `36/36`、Graph `23/23`，零 report；本轮没有共享可变状态或并发算法变化，TSan 对 CV3 ledger `NOT_APPLICABLE` |
| public headers | MSVC 逐头编译 Compute `7/7`、oneDNN provider `1/1`；Linux/Apple 对应 header test 随定向套件通过 |
| source/shared/static | Windows Compute direct-source、relocated shared、relocated static consumer 全部运行通过；shared 扫描 6 个 HYLE libraries / 700 exports，未泄漏 HYLE C、oneDNN、mimalloc 或私有 provider SPI；static 带 ukernel headers/library/license，shared 不泄漏 vendor headers |
| new installed API | 独立 consumer 在 relocated shared/static SDK 上分别构造 block 与 opaque format，验证 required bytes、alignment、transpose 后 block-axis identity并运行通过；不可整除 block extent 正确返回 `InvalidArgument`，不隐式 padding |
| Graph replay | Windows 官方 selected-Graph source/shared/static relocated consumers 全部运行通过；没有形成 Graph 产品 reopen trigger |
| P0 | MSVC x64 `sizeof(StorageFormat)=40`、`sizeof(TensorDesc)=184 <= 192`，三种 value 均 trivially copyable；构造、比较、permutation、required-bytes 最坏 `O(rank<=8)`；无 heap、lock、atomic、syscall、vtable、registry；HYLE Compute Release 命令为 `/O2 /Ob2 /DNDEBUG` |

CV3-01--07 已全部处理，公开 API、对象布局、算法和常态成本在候选提交停止变化。候选阶段仍需完成 Linux
与两台 Apple 的 source/shared/static relocated delivery 回放、终局七问/红队、治理 checker 与 terminal
签发；这些验证不得据点值继续调整候选产品。

## 跨平台交付收尾（2026-09-12）

不可变候选 `ada35161916d4e7b89ccabe846897cbea9a43808` 随后以 exact Git archive 在 Linux 和两台
Apple 机器上完成预声明回放；远程临时目录没有反向覆盖 Windows canonical checkout：

- Linux GCC 14.2 的 Compute source embedding、relocated installed shared、relocated installed static
  consumer 全部运行通过；shared export hygiene 检查 6 个 HYLE library、756 个 export；x64 static SDK
  携带 oneDNN 开发头、静态库与许可证，shared SDK 不泄漏 provider 私有 SPI；
- MacBook Air（AppleClang 21.0.0.21000334）与 Mac mini（AppleClang 21.0.0.21000101）的 Compute
  source/shared/static relocated consumer 全部运行通过；ARM 上 oneDNN provider 按能力合同关闭且没有下载；
- 因本候选迁移了 Graph 保存的 Tensor metadata，Windows、Linux 与两台 Apple 又分别运行同候选的
  selected-Graph source/shared/static relocated consumer，全部通过；oneDNN 保持关闭，未形成 Graph 产品
  reopen trigger；
- oneDNN 的 `ONEDNN_ENABLE_PRIMITIVE=MATMUL` 限制最终可用 primitive，但其上游 CMake 仍以单体 source
  glob 编译较宽的 CPU/x64 源集。这解释了 x64 三交付的构建时长，不表示 HYLE 暴露额外算子，也不构成
  storage candidate 的运行时成本或产品缺陷。

## 七问终局答案

1. **第三方路径**：本轮没有新增直接第三方、下载、许可证或公开 vendor 类型。Compute 仍通过已关闭的
   Mem/Platform/SIMD 等 HYLE factor 间接取得其合法实现；可选 oneDNN 继续由独立 factor 拥有，ARM 自动
   关闭，关闭时不下载。
2. **功能完整**：本轮精确关闭 logical dtype 与 physical storage 的分离：plain byte-strided、一个逻辑轴
   fixed-block 和不可通用解释的 opaque exact-extent 三档均可自然构造、比较、置换、计算 bytes 并绑定
   borrowed view。scale/zero、量化公式、多 buffer sparse schema、常量 binding 和 provider region 有明确
   后续 owner。它不冒充完整 Compute 算子目录；当前内建算子仍只有 RMSNorm、MatMul 与 SiluAndMul。
3. **内部复用**：checked arithmetic 复用 Math，storage/view/resource 复用 Mem，失败复用 Error，ISA 与
   whole-row kernel 复用 Platform/SIMD；没有私造 allocator、format registry、string identity 或 Graph
   lifetime planner。
4. **性能**：所有 layout 构造、比较、置换、span 与 alignment 检查最坏 `O(rank<=8)`；descriptor 无 heap、
   lock、atomic、syscall、vtable 或 registry，submit alignment 使用 power-of-two mask，不做整数除法。
   本轮 metadata 不进入 element loop，因而按预声明 `P0` 签发，不制造没有产品意义的纳秒 P1。
5. **接口与易用性**：普通用户继续用 `TensorDesc::dense()`；provider author 才使用 typed
   `StorageFormat::blocked/opaque()` 与明确的 byte-stride 入口。旧含混 element-stride API 已删除，没有
   compatibility alias、magic dtype、隐藏 padding/copy 或 Native FFI。
6. **构建交付**：没有新增 target、feature switch 或内部依赖边。Windows、Linux、MacBook Air、Mac mini
   的 Compute 与定向 Graph source/shared/static relocated consumer 均通过；x64 oneDNN 静态开发闭包与
   shared export hygiene 成立，ARM 负能力成立。
7. **文档与状态**：Compute/Graph 用户指南和 Compute 架构页均使用 logical dtype + physical byte-layout
   词汇，说明 block divisibility、opaque 限制、provider owner 与无隐藏成本。当前状态只由 registry 指向
   本 terminal record；后续算子、异步 backend、constant/region 和 GPU 能力不能由本卡推断为已交付。

## Terminal record（2026-09-12）

不可变 Compute v3 产品候选为 `ada35161916d4e7b89ccabe846897cbea9a43808`。候选到终局的
`include/hyle/compute`、`src/compute`、`include/hyle/graph`、`src/graph` 与相关测试产品差异为空；候选后
只更新状态和证据文字，没有根据验证结果回写 API、layout、算法或常态 metadata。

| Gate | 候选绑定证据 | Verdict |
|---|---|---|
| scope | CT30--CT38 全部完成，CT39 明确 routed；普通 scalar 与独立非 ggml codec 两条 journey 都成立 | `PASS` |
| architecture | logical/physical 分层、typed provider identity owner、Mem storage 与 Graph metadata 边界成立；无新依赖或反向边 | `PASS` |
| correctness | 四机 Compute/Graph 定向套件、block/opaque/permutation/overflow/alignment/view/fallback 边界以及 Linux ASan/UBSan 全过；本 ledger 无共享可变状态，TSan 为 `NOT_APPLICABLE` | `PASS` |
| delivery | 四机 Compute source/shared/static relocated consumer 全过；四机 Graph metadata 三交付回放全过；x64 vendor 开发闭包与 shared export hygiene 成立 | `PASS` |
| usability_docs | scalar 默认路径不增加概念；高级 storage 入口有 typed 示例、精确 owner/限制和可诊断失败；用户/架构文档一致 | `PASS` |
| red_team | 复核旧 element-stride 残留、benchmark seam、格式身份、算术、padding、alias/alignment、provider fallback、vendor 泄漏与 Graph reopen 条件，未发现候选产品反证 | `PASS` |
| engineering_performance | `sizeof(StorageFormat)=40`、`sizeof(TensorDesc)=184<=192`，均 trivially copyable；全部 descriptor 工作 `O(rank<=8)` 且无常态动态成本，预声明 P0 通过 | `PASS` |

代表性验证结果：

```text
Windows/Linux/MacBook Air/Mac mini:
cmake -DHYLE_SMOKE_COMPONENT=compute -P cmake/HyleCiSelectedComponentSmoke.cmake
cmake -DHYLE_SMOKE_COMPONENT=graph -P cmake/HyleCiSelectedComponentSmoke.cmake

Windows/Linux/Apple targeted suites:
Compute 36/36, Graph 23/23; Windows oneDNN 10/10
Linux ASan/UBSan: Compute 36/36, Graph 23/23, zero reports
```

公开限制：这次只签发 provider-neutral tensor storage，不扩张既有同步 CPU backend 或三算子目录。异步
Completion/backend、开放 provider identity、完整推理算子目录、Graph constant/region、ggml 量化路线及
CUDA/Metal/Vulkan/HIP 必须由后续真实纵切重新 scope；它们不是本卡的隐含承诺。只有可复现
correctness/security/lifetime/storage 缺陷、真实 consumer 证明本 owner 下的新 physical-format 能力、相关
依赖升级、公开合同/算法/交付变化，或合格证据显示 material regression 才重开本轮。

七门全部 `PASS`，Compute v3 storage-format 纵切裁决为 **`CLOSED`**。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
