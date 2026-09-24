# M3-D3：ggml Vulkan 原生压缩权重接缝

<!-- MODULE REVIEW: CANDIDATE -->

## Closure card

- owners：`compute_route_vulkan`、`graph_route_vulkan`、`compute_provider_ggml_vulkan`、
  `compute_provider_iree_vulkan`；公共 GGUF storage catalog 仍由 `compute_ops` 的 M3 主卡持有。IREE
  owner 本批只有限定域复核，没有预定产品改动。
- baseline：`92b08826168a93c04f6868de9896c0bdbb262060`；Windows canonical 工作树干净并与
  `origin/main` 一致。
- reopen trigger：固定 ggml `456172ec` 的 Vulkan `supports_op` 与 shader 生成器已含公共十格式的
  `MUL_MAT`/`MUL_MAT_ID`；但 HYLE Vulkan managed region 只识别 plain storage 与一个 MXFP4 特例，
  其余真实 GGUF K/IQ/Q8 权重从普通 Direct/Graph 入口不可达。
- 正常 consumer：已解析 GGUF tensor table 的普通 C++ 用户，把公共 checked `TensorDesc` 与原始 packed
  bytes 上传或借入同一 Vulkan device，准备 MatMul/GroupedMatMul 或 Graph region，并在 prepare 固定
  Provider 后对变化 activation 重复 submit。
- 主产品纵切：Qwen dense `[M,K] * Q4_K[K,N]`；正交反偏置纵切为 MoE
  `IQ4_XS[E,N,K] * [T,S,K] + ids[T,S]`，另用 MXFP4 检查 32/17 block 与已有路径兼容。
- scope：复用公共十格式 catalog，批量补全 ggml Vulkan native type/shape/import、MatMul 与
  GroupedMatMul、Direct/Graph 选择、资源/失败/复用和三交付；核对限定 IREE 标准/FP8/scan 域，不把它
  伪装成 GGUF Provider。
- non-goals：GGUF 文件加载、量化/重编码、HYLE decoder/kernel/shader、CPU fallback、逐 kernel vtable、
  新 Provider、自研数学、模型级推理或改变 CPU/CUDA/Apple candidate。
- 依赖与 consumer：`compute_ops -> compute_provider_ggml_vulkan -> compute_route_vulkan ->
  graph_route_vulkan`；IREE 只保留精确合格域。公开 consumer 是 Vulkan Compute/Graph 用户文档。
- performance：P0。packed bytes 必须直接进入原生 ggml Vulkan tensor，不得常驻解码、复制为浮点或在
  submit 重新选路/重排。完整 profile 的端到端 P1 留给 D5。
- machines：Windows RTX 4070 是产品、主 correctness 与交付真相；MacBook Air 的 MoltenVK 只签
  portability/capability，不替代 Windows verdict。构建低并行、重型验证串行。
- budget：一个批量 type mapper、两类 operation lowering、现有 route/Graph 判定、一个批量测试目标与
  现有交付 smoke；若上游实际不支持某格式/形状，记录精确 rejection，不增加自研 shader。
- stop：十种格式逐一取得静态及 RTX 4070 capability verdict；至少 Q4_K、IQ4_XS、MXFP4 与 grouped
  expert 通过普通 Direct/Graph、两次 submit、资源/失败和 source/static/shared，或形成上游具名拒绝。

## 能力与形状审计

| 能力 | 当前事实 | 裁决 |
|---|---|---|
| 公共十格式 ABI | `low_precision` 已固定 block bytes、block axis、逻辑 `[K,N]` / `[E,N,K]`；Vulkan `Tensor` 已保存完整 `TensorDesc`，`upload`/`borrow_ready` 按 `required_bytes()` 携带原始 bytes | `KEEP`；不增加公开 carrier 或 Vulkan 特有格式描述 |
| Vulkan 上游 kernel | 固定 ggml `456172ec` 的 `ggml_backend_vk_device_supports_op` 明确接受 Q3_K、Q4_K、Q5_K、Q6_K、Q8_0、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、MXFP4；shader loader 同时为 `MUL_MAT`/`MUL_MAT_ID` 建立这些类型的 matrix/vector pipeline | `BORROW`；实际 device/shape 仍由冷期 `supports_op` 和实机运行裁决 |
| Provider type/shape | `region.cc` 只有 `mxfp4_domain()`，其余 blocked desc 被 `tensor_domain()` 拒绝；input tensor 也只为 MXFP4 改成 ggml `[K,N]` | `FIX`：复用公共 catalog 建一个十格式 mapper；block axis 移到 ggml `ne[0]`，其余轴逆序，保持 `[K,N]` 与 `[K,N,E]` |
| Direct MatMul | MXFP4 特例跳过 RHS transpose；其余 packed type 不可达，普通 dense 路线仍需要 transpose/contiguous | `FIX`：十格式 weight 都直接调用 `ggml_mul_mat(weight, activation)`；dense 行为不变 |
| GroupedMatMul | lowering 已调用 `ggml_mul_mat_id(weight, activation, ids)`，但通用 non-MatMul plain gate 与 dense-only准入先拒绝 blocked weight | `FIX`：只允许 port 0 是精确十格式 `[E,N,K]`，activation/ids/output 继续 plain F32/I32；不放宽其它 custom op |
| 冷能力与资源 | `create_storage()` 已对每个最终 ggml node 调用 Vulkan device `supports_op`，并由 gallocr 建一次 workspace；每 invocation 持有 imported buffer owner、work/fence 和 terminal failure | `KEEP`；不另建 capability registry，不在 submit 搜索或计时 |
| buffer 安全 | route allocation/upload 已按 logical `required_bytes()`；Provider validate 只对 MXFP4 要求原生 storage alignment | `FIX`：所有十格式 weight 使用相同原生 storage alignment、capacity 与 overflow 校验；错误在 prepare/bind 明确失败 |
| Graph | Graph 复用同一 `Plan`、constant `Tensor` 与 managed-region matcher；当前报告只写 MXFP4 | `FIX`：扩展 matcher 后自然走同一 region，并更新精确报告/文档；不建第二套 Graph provider |
| IREE | 当前只执行 checked I32 Scan、F32/F16→BF16 和两种具名 FP8 MatMul；`iree_operation()` 不接受 GGUF/MXFP4，必要 boundary copy 与设备 feature 检查已显式计入 | `KEEP_CONDITIONAL`；不把 IREE type legalization 或 `named_compressed_storage()` 当 GGUF 执行能力，本批不改其程序 |

`supports_op` 是每台 Vulkan device、最终 graph 与形状的冷期判定，不是静态符号即通过。十格式会统一接线，
但任何 device-specific rejection 都保留为格式/形状/设备的精确结果；不得解码为 F32、转交 CPU 或由 IREE
相近 FP8 路线代签。RTX 4070 是主 verdict，MoltenVK 只给独立 portability 结论。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| A1 | ggml Vulkan 只识别 MXFP4 storage ID | 在 `region.cc` 复用公共 `gguf_quantized_type()`，批量映射十种 `ggml_type`、block axis 与 native shape | 十格式 byte size 与 `ggml_nbytes()` 一致；未知/畸形 storage 拒绝 |
| A2 | MatMul/GroupedMatMul 准入仍把其它 packed weight 排除 | 仅在 MatMul RHS 或 GroupedMatMul weight port 放行 exact GGUF descriptor；对应 lowering 直接使用原生 tensor | Q4_K dense、IQ4_XS grouped、MXFP4 都经过原版 `MUL_MAT`/`MUL_MAT_ID`；dense 路线不变 |
| A3 | 通用 route/Graph 与 native binding 尚未覆盖新 packed 域 | 复用 managed region matcher，补充 packed device-limit/alignment/capacity 检查和报告文本 | prepare 固定 Provider；两次 submit 无 decode/repack/reselect；错端口、offset、capacity 明确失败 |
| A4 | 现有实机测试只有 MXFP4 | 建一个 Vulkan 批量 fixture，静态覆盖十格式，实际创建/运行 capability；Q4_K、IQ4_XS、MXFP4 与 grouped 有独立数值 oracle | RTX 4070 Direct/Graph、常量/运行输入、两次 submit、资源与 terminal failure；MoltenVK 独立 portability |
| A5 | 用户文档与交付仍宣称只有 MXFP4 | 更新 Vulkan guide；复用 source/static/shared 安装、搬迁、外部 consumer、导出/许可证/DLL 检查 | 三交付从 installed package 编译运行普通 Q4_K Direct 与 IQ4_XS Graph，不依赖 producer tree |

本表之外不修改 IREE program、公共 Tensor API、路由维度、shader、codec 或自研数学。若真实上游要求改变
公开 carrier、发生热路径格式转换或需要新增 Provider，退回本卡 `AUDITED` 重新定界。

### API / usability freeze

Vulkan 已有自然 API 足够表达 packed storage，不增加平台特有 wrapper：

```cpp
HYLE_ASSIGN_OR_RETURN(auto weight_desc,
                      gguf_weight_description(GgufQuantizedType::kQ4K,
                                              DataType::kFloat32, k, n));
HYLE_ASSIGN_OR_RETURN(auto weight,
                      context.upload(weight_desc, weight_bytes));
HYLE_ASSIGN_OR_RETURN(auto program,
                      prepare(context, MatMul{}, activation_desc, weight));
```

`weight` 的 `TensorDesc` 是逻辑 `[K,N]`，native buffer 中仍是 GGUF `[N,K]` 连续 block bytes；
`gguf_grouped_weight_description()` 同理表达 `[E,N,K]`。常量 `Tensor` 由 `Argument` 捕获并保活；运行时
packed input 也必须来自同一 device、已完成可见性移交。Graph constant 直接使用同一个 `Tensor`。准备期
建立 ggml metadata、检查 device `supports_op`、分配每 invocation workspace；submit 只刷新 native
binding 并提交固定 graph，不解码、不重排、不重新选择 Provider。

packed storage 只允许作为 MatMul RHS 或 GroupedMatMul weight；activation、ids 与输出继续使用原有
plain F32/I32 合同。unknown storage、错误 block axis/bytes、跨 device buffer、未完成 producer、错
offset/capacity 在 prepare/bind 失败。独立 invocation 保持并行，单 invocation pending 时不可重入；失败
不发布结果，device lost 仍按既有 route terminal 语义处理。source/static/shared 使用相同公开 API。

## 七问

1. 第三方：锁定 ggml `456172ec`，直接消费原版 Vulkan backend、`MUL_MAT`/`MUL_MAT_ID` 与设备
   `supports_op`；现有 runtime patch 只承载 HYLE device/resource seam。许可证与升级锁继续由构建清单管理。
2. 完整性：Q4_K dense 是主纵切，IQ4_XS grouped 防止只为单矩阵特化，MXFP4 防止错误复用 256 元素
   block 假设；十格式批量 capability 表防止只接测试格式。模型加载与整模编排仍属上层。
3. 复用：复用公共 catalog、Vulkan `Tensor`/upload/borrow、managed region、native buffer import、gallocr、
   work/fence 与 Graph plan；不复制量化器、allocator 或调度器。
4. 性能：P0 要求 packed bytes 零常驻解码、零 submit repack、一次冷 graph/workspace；D5 才在真实 profile
   统一计算上传、同步、边界和融合成本。本卡不以 microbenchmark 选择格式或 Provider。
5. FFI：不新增 C ABI；ggml bridge 继续私有，公开面只有 C++ `TensorDesc`/`Tensor`/`Program`。vendor patch
   没有新增模型格式语义。
6. 交付：现有 Vulkan factor 已支持 source/static/shared 与 staged Vulkan/DLL 依赖；A5 复用同一安装消费
   驱动并加入 packed 纵切，不创建空 target。
7. 易用：普通用户只需公共 descriptor 与 `Context::upload`/`borrow_ready`；guide 明确 bytes、block axis、
   owner/visibility、IREE FP8 与 GGUF 的区别及 capability rejection。

## 预声明验证

- Windows：以产品 candidate 为 base/candidate 运行 selector；selected Vulkan provider/route/graph、公开头、
  source/static/shared、文档和 closure gates；RTX 4070 批量十格式 capability 与 Q4_K/IQ4_XS/MXFP4/grouped
  correctness、两次 submit、错误路径和资源生命周期。低并行，重型目标串行。
- MacBook Air：从 Windows 精确 source archive 部署，运行同一 provider/route/graph fixture；MoltenVK 的
  不支持项保留具名 device verdict，不冒充 Windows 主机。
- P0：源码账单确认 prepare 后没有 HYLE decode/repack/provider search；本卡不运行 P1，D5 负责不可变
  profile candidate 的完整成本。
- 通用：`check-module-closure.py --previous-ref <registry-before>`、`check-docs.py`、`git diff --check`；
  module-status 改动只生成状态，不触发无关构建。

## Candidate / 验证 / 红队

产品 candidate 为 `2757ecc96815fd12c3050faab176cc2ddad02ccd`。A1-A5 已全部完成：公共十格式
批量映射到原版 ggml Vulkan type/shape/import，MatMul 和 GroupedMatMul 分别进入原生 `MUL_MAT` 与
`MUL_MAT_ID`；旧 route gate 已从“保留全部 rank-2 F32 MatMul”收窄为只保留 plain/plain F32，packed
descriptor 因而能从普通 Direct/Graph 入口到达成熟 region。prepare 固定 Provider、device
`supports_op` 与 workspace，submit 不解码、重排、计时或重新选路。

Windows RTX 4070 的 candidate-bound 测试逐一运行十格式非零量化数值 oracle 与两次 submit；Q4_K、
IQ4_XS、MXFP4、IQ4_XS grouped experts、Graph captured constants、错误端口/格式/accumulation、资源和
terminal failure 均通过。正式 selector verdict 为 `PASS`，selected gates 为 docs、closure、configure、
`compute_provider_ggml_vulkan`、限定域未改动的 `compute_provider_iree_vulkan`、
`compute_route_vulkan` 与 `graph_route_vulkan`。P1 按合同未选择，完整 profile 成本仍由 D5 负责。

同一 candidate 的精确 Git archive 已通过 source 嵌入、迁移后的 installed static SDK 和迁移后的
installed shared SDK。三个 consumer 都从公开 API 执行 Q4_K Direct 与 IQ4_XS Graph 各两次；installed
消费前已隐藏 source 与三个 producer build。shared 所需 HYLE DLL 全部在 EXE 同目录，consumer 使用
非交互 Windows error mode；static EXE 无 HYLE DLL dependency。安装头、CMake relocation 和 ggml/IREE/
Vulkan 许可证均通过。

[完整收据](../../../../evidence/correctness/llm-moe-d3-ggml-vulkan-binding/README.md)保存 selector、三交付、
hash、链接边界和限制。统一脚本在 immutable candidate 上统计 Compute + Graph 全实现为 **60,923 行 /
252 文件**，排除 tests/bench/docs/vendor/build/cache/scratch，self-check `PASS`。本批不宣称模型加载、
整模推理、D5 真实 profile P1 或未运行的 MoltenVK portability；IREE 仍只保留精确限定域，不代签 GGUF。
