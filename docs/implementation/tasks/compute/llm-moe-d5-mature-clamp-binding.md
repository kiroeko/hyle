# D5-F24：成熟模型限幅接线

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- baseline：`bc21360418a310baa3d0bea3b1e1e44635d907b5`；Windows canonical，保留正在进行的 MoE 测试修改。
- owner：`compute_ops` 从 CANDIDATE 退回 AUDITED；四个 ggml Provider、Apple Direct/Graph、
  Vulkan Direct/Graph 的当前 AUDITED 范围仅追加本卡 F24。旧范围和未完成交付/P1 继续有效。
- trigger：十三画像 router 及 DS/GLM Flash 激活的普通 CPU Direct/Graph 测试确认，
  Binary minimum/maximum 落到冻结的 HYLE Native；ggml 的相同模型程序已有原生 Clamp。
- consumer：router 分母下限 `6.103515625e-5`；DS/GLM Flash gate 上限10、up 区间[-10,10]。
  正交消费为一般模型张量限幅。实际宽度、dtype、expert/shared 角色来自本批固定 header 表。
- scope：仅具名 `model_ops::Clamp` schema 与四路线原生 ggml 调用和必要分类；不写数学实现，
  不更改通用 Binary/Softplus，不扩充 Provider、不重写路由或验证仪器。
- 依赖：已固定 ggml runtime；CPU/CUDA 复用现有 custom prepare，Apple/Vulkan 复用原生 program。
  不增加 allocator/queue/线程/完成路径，也不改变并发 invocation 的 workspace 所有权。
- 验证：schema 正/负例、四路线 Direct/Graph 真实 router/激活/MoE 批次、失败时最小中间量诊断；
  API 稳定后 selector 选择受影响 owner/直接消费者。D5-A5 三交付、A6 P1 仍在批末统一执行。
- P0：单输入单输出、每元素 O(1)、一个原生节点；无额外阈值张量、热路径扫描或路由重选。
- 预算：一次完整接线批次与一次有具体失败的修正；数据差异先查接线和 oracle，不放宽数值门。
  若原版模型域确实不支持，记录精确拒绝；不以自实现补洞，不无限研究。
- stop：本批接线和适用门有明确裁决；局部通过不增加整画像 ledger。

## 源码审计与固定合同

固定 llama.cpp `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a` 的 `src/llama-graph.cpp`
在 MoE 权重归一化和 clamped SwiGLU 中使用 `ggml_clamp`；固定 SGLang 的
`python/sglang/srt/models/glm5_next.py::swiglu_clamped` 独立确认 gate/up 边界。
已有 runtime 的 CPU `ggml-cpu/ops.cpp`、CUDA `ggml-cuda/clamp.cu`、Metal unary 和
Vulkan `vulkan-shaders/unary.comp` 均提供原生限幅，不需要 HYLE kernel。

公共调用 `model_ops::clamp(minimum, maximum)`：plain dense、非空、rank 1–4、F32，输出同描述。
prepare 拒绝 NaN bounds 和 minimum > maximum；允许无穷 bounds。有限输入与 ±Inf 按闭区间限幅，
NaN 输入和 signed-zero 的位模式遵循原生 backend，不承诺跨路线一致；不冒充通用 Binary 的 NaN 合同。
这项限定在文档中明确，测试不使用 NaN 数据签数学等价。调用前照常检查 description/bytes/device/alias，
不新增热路径内容检查或隐藏转换。Custom descriptor 只有两个标量，冷绑定原生节点，复用现有完成/失败状态。

| FIX | owner / 范围 | 决定 |
|---|---|---|
| F24.1 | compute_ops | BORROW：模型 Clamp schema、元数据拒绝测试和公开说明 |
| F24.2 | 四个 ggml Provider | BORROW：`ggml_clamp` 的准入与 native emit，无 vendor 修改 |
| F24.3 | Apple Direct/Graph、Vulkan Direct/Graph | FIX：具名成熟 custom 分类；Vulkan 纳入已有 elementwise 设备容量检查 |
| F24.4 | D5 共用测试 | FIX：替换旧 Binary 限幅；真实激活宽度、router ID 精确比较、完整 MoE 的原门不变 |

## 当前 verdict

候选 `a11aac6a88a9a32265ce230e094f12d09299d061` 已完成 F24.1–F24.4 接线。
Clamp 四路线 Direct/Graph 边界/变化输入通过；公共 schema 和独立头2项 ctest 通过。
CPU/Apple/Vulkan 的整个新批次各入口5/5；CUDA数值通过，但 DeepSeek router 的旧 HYLE Softplus
触发成熟 Provider 断言，因此该路线整个批次仍 FAIL。该问题不在 Clamp 内扩大修复范围。
完整日志和初始 CPU 夹具量化反例见
[部分收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/moe-execution-batch/README.md)。
当前 owner 仍在 D5 AUDITED；完整 selected owner gates、三交付、P1 未完成，不能关闭或增加整格。
