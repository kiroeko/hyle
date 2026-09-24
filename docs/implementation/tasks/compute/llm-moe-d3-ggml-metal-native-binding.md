# M3-D3：ggml Metal 原生压缩权重接缝

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- owners：`compute_route_apple`、`graph_route_apple`、`compute_provider_ggml_metal`；公共
  GGUF storage catalog 仍由 `compute_ops` 的 M3 主卡持有。
- baseline：`f18bf7b2dad9b903bea06bfe8e14b60d1f023d8c`；审计提交前 Windows canonical 工作树干净。
- reopen trigger：固定 ggml `456172ec` 已含目标 Q/K/IQ/Q8/MXFP4 Metal kernel，公共十格式
  descriptor 已冻结到 `08b8e2f1`；但 `NativeProgram` 仍把 blocked payload 当逻辑浮点 tensor，普通
  Apple `InputSpec`/binding 也不能同时表达逻辑 tensor 与物理 packed bytes，真实 GGUF 权重因此从
  Apple Direct/Graph 不可达。
- consumer：已解析 GGUF tensor table 的普通 C++ 用户，用公共 checked `TensorDesc` 和保有 packed
  payload 的 MLX/Metal carrier 准备 Apple MatMul/GroupedMatMul 或 Graph region；准备期固定 ggml Metal，
  同一 invocation 对变化 activation 重复 submit。
- 主纵切：Qwen dense MatMul 的 `[M,K] * Q4_K[K,N]`；反偏置纵切为 MoE
  `IQ4_XS[E,N,K] * [T,S,K] + ids[T,S]`，另以 MXFP4 证明 32/17 block 不被 K-quant 假设吞掉。
- scope：Apple route 的原生 packed input carrier；ggml Metal 精确十格式 import、MatMul/
  GroupedMatMul lowering、能力检查和原生资源；Direct/Graph 普通入口、失败/复用与三交付。
- non-goals：GGUF 文件加载、量化/重编码、HYLE decoder/kernel/shader、把 MLX affine Q4 与 GGUF 格式
  混同、逐 kernel executor、改变 CPU/CUDA candidate、扩张算子目录或模型级推理。
- dependency / consumer：`compute_ops -> compute_provider_ggml_metal -> compute_route_apple ->
  graph_route_apple`；公开消费者是 `docs/user/apple-compute.md` 与 Apple Graph recipe。
- performance：P0。packed payload 不得被 HYLE 常驻解码、复制为浮点或在每次 submit 重排；完整片段
  P1 仍由 D5 在不可变 profile candidate 上统一计算准备、上传、同步和融合成本。
- machines：MacBook Air `Mac16,12` 完成 source/static/shared 与主 correctness；Mac mini `Mac16,10`
  只签独立 correctness/portability，不冒充同一 Formal 性能机。Windows 仅保留产品真相、selector、文档和
  closure 检查。
- budget：一个公开 carrier 字段/mapper、两种 operation lowering、一个 Graph 分区判定、一个批量平台
  测试目标和现有交付 smoke；若实机否定公共 API 或需要格式转换，立即退回 AUDITED，不扩建仪器。
- stop：十种 catalog 格式逐一取得静态 kernel 与两台实机 capability/run verdict；至少 Q4_K、IQ4_XS、
  MXFP4 与 grouped expert 通过普通 Direct/Graph、两次 submit、资源/失败和 source/static/shared，或记录
  固定上游 rejection。不得用 MLX affine Q4、F32 展开或 CPU 路线代签。

## 能力与形状审计

| 能力 | 当前事实 | 裁决 |
|---|---|---|
| 公共十格式 ABI | `low_precision` 已固定 block bytes、block axis、逻辑 `[K,N]` / `[E,N,K]` | `KEEP`，Apple 只消费 catalog |
| Metal kernel | 固定上游的 `ggml-metal-device.cpp` 为十格式登记 matvec pipeline；`mul_mv.metal` 具备 Q/K/IQ/MXFP4 kernel，`MUL_MAT_ID` 明确实例化 MXFP4、Q3/4/5/6_K、IQ4_NL/XS 等 | `BORROW`；注册不等于特定芯片/形状已运行，保留逐格式实机 verdict |
| 物理 carrier | `InputSpec` 只有 MLX shape/dtype；Graph `native_spec()` 只认 plain 与 MLX affine Q4 | `FIX`：一个 mapper 把 GGUF logical desc 映射到 `uint8[required_bytes]`，同时保存 logical desc |
| Direct MatMul | `ggml_model_operation()` 不含 MatMul；spec overload 会重建 dense desc | `FIX`：只在 RHS 是公共 GGUF 格式时固定 ggml Metal，调用 `ggml_mul_mat(weight, activation)` |
| GroupedMatMul | 已走 `ggml_mul_mat_id`，但 blocked type 未映射，carrier 不可达 | `FIX`：复用公共 grouped descriptor，保持 `[E,N,K]`/block-axis-2 与上游 `[K,N,E]` 一致 |
| Graph region | 只按 operation 名分区；MLX region 可能错误接收 packed bytes | `FIX`：按 operation 加输入描述判定；GGUF 只准进入对应 ggml MatMul/GroupedMatMul region |
| 生命周期/失败 | 每 invocation 独占 backend/graph，导入原始 `MTLBuffer`，checked sync，失败后 terminal | `KEEP`；packed carrier 仍由 MLX array owner 保活，重复 submit 只换 activation/绑定 |
| MLX affine Q4 | `uint32[N,K/8] + scale/bias`，与 GGUF block ABI 不同 | `KEEP` 且保持独立，不参与本卡 ggml 路由 |

上游 `supports_op` 仅排除 NVFP4/TQ1_0，并按 activation/芯片决定 matrix 或 vector path；它不能替代实机
kernel 创建与运行。十格式因此统一接线、逐格裁决，任何 `UNIMPLEMENTED` 都保留为该格式/形状/机器的
精确 capability 结果，不回退到隐式 F32 解码。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| A1 | public spec 无法表达 logical GGUF + physical bytes | `InputSpec` 增加可选 logical description；公开 checked `input_spec(TensorDesc)` 生成 carrier | 旧 `{shape,dtype}` 调用不变；非法/未知 storage 拒绝；十格式 byte 数逐一吻合 |
| A2 | provider 未映射 blocked ggml type/MatMul | 在 `NativeProgram` 复用公共 type catalog、block-axis shape 和 MatMul/GroupedMatMul lowering；按端口校验 offset/capacity | `ggml_backend_supports_op` 冷检查；无 HYLE 解码/重排；错误 carrier/offset 明确失败 |
| A3 | Direct/Graph 只按 operation 名分区 | 根据 operation + 输入 description 选择 ggml region，阻止 packed value 流入 MLX region | dense MatMul 保持 MLX；GGUF MatMul/Grouped 走 ggml；不支持组合在 prepare 时失败 |
| A4 | 缺少真实格式与复用证据 | 一个批量测试生成/使用合法 packed block，覆盖 Direct、Graph、两次 submit、错 schema、失败 terminal 与两台 Mac | 十格式 capability 表；Q4_K/IQ4_XS/MXFP4/grouped 有数值 oracle |
| A5 | 交付与用法未声明 | 更新 Apple 用户文档；运行现有 source/static/shared 安装、搬迁、外部消费与导出检查 | 三交付均从 installed package 编译运行普通入口 |

本表之外不新增数学、格式、路由维度或通用执行器；新事实要求改变 carrier owner/热路径成本时退回本卡
AUDITED。

### API / usability freeze

普通 Direct 调用先构造公共 logical descriptor，再由 Apple mapper 得到物理 spec：

```cpp
auto weight_desc = gguf_weight_description(GgufQuantizedType::kQ4K,
                                           DataType::kFloat32, k, n);
auto weight_spec = apple::input_spec(*weight_desc);
std::array specs{apple::InputSpec{{m, k}, mlx::core::float32}, *weight_spec};
auto program = context.prepare(MatMul{}, specs, workload);
```

`InputSpec::shape/dtype` 始终描述实际 MLX carrier；新增 logical description 只在 provider-native storage
需要时有效。GGUF carrier 固定为一维 `uint8[required_bytes]`，不得把 packed bytes 伪装成逻辑 MLX
float array。examples overload 无法从 `uint8` 猜 GGUF 类型，压缩权重必须用 spec overload；Graph 从
`ValueSpec/TensorDesc` 自动产生同一 spec。packed carrier 必须 row-contiguous 且从原 buffer 起点开始，
由 invocation 持有 array owner 到 checked completion。准备期创建 ggml metadata/pipeline；submit 不搜索
provider、不解码权重、不建立逐 kernel vtable。dense MatMul 和 MLX affine Q4 保持现有 MLX 路线。

OOM/Metal import/command-buffer failure沿现有 `Status`/terminal invocation 传播；独立 invocation 可并行，
单 Context 仍 FIFO。source/static/shared 均通过 `hyle::compute::apple` 公开相同 helper 和对象。

## 七问

1. 第三方：锁定 ggml `456172ec` 与当前 MLX dependency；只调用原始 Metal backend/kernel，许可证与升级
   仍由现有 build manifest 管理。未知格式无 fallback。
2. 完整性：dense Qwen、MoE grouped、MXFP4 三个纵切覆盖 axis、格式族和运行形状；完整模型加载/编排
   属于上层。
3. 复用：复用 `low_precision`、`TensorDesc::required_bytes()`、现有 Apple invocation 与 native import；
   不复制 codec、allocator 或调度器。
4. 性能：P0 要求一次 metadata/pipeline 准备和零常驻解码；非连续 dense 输入现有一次 contiguous copy，
   packed 输入直接拒绝非连续 carrier。
5. FFI：没有新增 C ABI；ggml/Objective-C seam 保持私有，公开面仍为 C++/MLX 类型。
6. 交付：Apple route 已有 source/static/shared factor；A5 复用现有安装/搬迁 consumer，不造空 binary。
7. 易用：一个 public mapper 从同一 TensorDesc 产生 carrier spec；文档明确 byte owner、spec overload、
   affine-Q4 区别和 capability 错误。

## 预声明验证

- Windows：`validation-select.py plan --base f18bf7b... --candidate WORKTREE`；selected 静态/文档 gate；
  `check-module-closure.py --previous-ref <registry-before>`、`check-docs.py`、`git diff --check`。
- MacBook Air：Apple provider/route/graph 批量测试，十格式表，Q4_K/IQ4_XS/MXFP4/grouped 数值与两次
  submit、错误 schema/capacity/terminal failure；source/static/shared 安装、搬迁及外部 consumer。
- Mac mini：相同批量 correctness/portability；不签 Formal 性能。
- P0：源码账单确认 packed bytes 直接 import，prepare 后 submit 不含 decode/repack/provider search；本卡
  不运行 P1。

## Candidate / 验证 / 红队

尚未形成 candidate。有限改动完成并在两台 Mac 取得预声明 verdict 后填写 immutable commit、逐 gate
证据、LOC、限制和最终状态；未运行项保持 `UNPROVEN`，不借 CPU/CUDA 或静态 kernel 名称代签。
