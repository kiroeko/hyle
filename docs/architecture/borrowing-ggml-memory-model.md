# 借 ggml 算子的显存归属边界：host 归 HYLE，device 归 ggml

> **一句话**：用 ggml 算子算 AI 时，**host 内存归 HYLE（用 mem 喂入/取出），device 显存归 ggml（它自己 gallocr 管理）**——就像 llama.cpp。想要显存完全自管 + 零拷贝，那是 compute-04/06 自造 kernel 的领域，不是借 ggml 的领域。
>
> 这条边界不是设计偏好，是 **ggml CUDA 后端的结构性能力约束**，经本仓库 compute-05 实现期的 gdb 逐行调试坐实（下有铁证）。写这篇专文，是因为它直接决定"HYLE 显存自管抽象"的适用范围，值得单独讲透、供后续所有借 ggml 的里程碑（compute-05/07/08）遵循。

## 0. 结论先行（要遵守的规则）

| 数据 | 归谁管 | 怎么做 |
|---|---|---|
| **喂给 ggml 算子的 host 输入**（权重加载源、激活输入） | **HYLE host RAM** | `mem::Allocator` 分配，或 `PinnedStaging` 在其直接拥有的 pinned host block 中 bump → `ggml_backend_tensor_set(tensor, hyle_host_ptr, …)` 拷入 |
| **ggml 算子计算用的 device 显存**（输入/中间/结果 tensor 的 buffer） | **ggml**（`ggml_gallocr` / backend buffer） | ggml 自己 cudaMalloc + 图内复用；HYLE 不碰 |
| **ggml 算子的 host 输出** | **HYLE mem** | `ggml_backend_tensor_get(tensor, hyle_host_ptr, …)` 拷出 |
| **HYLE 自造 kernel（compute-04 NVRTC / 06 SPIR-V）的显存** | **HYLE**（compute-02 Buffer / DevicePool / VMM） | HYLE cudaMalloc，kernel 直接在 HYLE 显存上算——**真零拷贝**（已三平台验证） |
| **非 ggml 的 device 数据**（KV-cache、CAD 网格、BVH、CAE 刚度矩阵） | **HYLE** | HYLE 显存管理，与 ggml 无关 |

**要点**：借 ggml，就接受 ggml 管它算子用的那段显存。你的 mem 抽象在 **host 侧全程有效**（喂入/取出都是 HYLE mem 的地盘），在 **device 侧退让给 ggml**——但仅限"ggml 算子正在用的那段"。HYLE 显存自管在 compute-04/06（**一切非 AI 的通用 GPU 计算**：普通 CUDA vecadd/reduce/scan + ② CAD/CAE 几何/网格/SpMV）上完整成立——ggml 只是"AI 推理算子多"，不是通用 GPU 计算框架。

## 1. 为什么 device 侧不能自管——ggml CUDA 结构性否决外部显存

最初 compute-05 的愿景（compute-design 早期发现 2）是：**HYLE 自己 cudaMalloc 的显存，直接当 ggml tensor 的 `->data`，让 ggml 算子零拷贝地在上面算**。这在 CPU 后端成立，但在 **CUDA 后端被 ggml 结构性否决**：

**铁证 1 —— CUDA 后端明确不支持外部指针建 buffer**（`ggml-cuda.cu:5032`）：
```cpp
/* .buffer_from_host_ptr = */ false,   // CUDA device caps
```
对比 CPU 后端有 `ggml_backend_cpu_buffer_from_ptr(ptr, size)`（`ggml-backend.h:430`）能零拷贝包外部内存——**CUDA 没有等价物**。ggml 官方 device 能力表里，CUDA 的 `buffer_from_host_ptr` 就是 false。想把一块 HYLE 显存"注册"成 ggml buffer，无公开 API。

**铁证 2 —— 硬凿"注入 `->data` + 造假 buffer 元数据"在多算子图必崩**（gdb 逐行坐实）：
试过绕过——给 tensor 挂一个 size-0 dummy buffer（`ggml_backend_buft_alloc_buffer(buft, 0)`）拿元数据、`->data` 注入 HYLE 显存。单算子侥幸能跑，但**多算子合批 + 不同形状必段错误**：
- 崩溃点 `ggml-cuda.cu:1924` `const bool src1_on_device = id == src1_ctx->device;`
- 根因：`src1->buffer->context == NULL`。`cudaMalloc(&p, 0)` 返回 null，size-0 dummy 的 base 是 nil；多个输入共享这个 dummy 时，`ggml_gallocr_alloc_graph` 处理把共享 dummy 的 context 弄丢；varied 形状走通用 `ggml_cuda_op_mul_mat` 路径读 `src1_ctx->device` → 空解引用。
- （uniform 形状走 cuBLAS batched 路径不读该字段，所以"单发/同形状能跑"是假象，掩盖了结构问题。）
- 全过程见 [`../debug/ggml-zerocopy-null-buffer-hunt.md`](../debug/ggml-zerocopy-null-buffer-hunt.md)。

**结论**：ggml 的显存模型是它算子正确性的一部分——tensor 的 buffer 必须由 ggml 自己分配管理（gallocr），算子内核依赖 buffer 的 context/device 元数据、图内显存复用、padding 等一整套。**"外部显存注入"在 CUDA 上不是"没找对 API"，是 ggml 明确不支持。** llama.cpp 在 GPU 上也从不注入外部显存——权重加载时拷进 ggml 显存一次、常驻复用（`llama-model.cpp:1570` `ggml_backend_alloc_ctx_tensors_from_buft` = 真 buffer），激活走 gallocr。

## 2. 为什么 host 侧完全归 HYLE——ggml 对 host 源指针零约束

好消息：**数据进出 ggml 显存的入口，对 host 指针没有任何要求**。

**铁证 3 ——`tensor_set` 就是一次裸 `cudaMemcpyHostToDevice`**（`ggml-cuda.cu:688`）：
```cpp
static void ggml_backend_cuda_buffer_set_tensor(..., const void * data, ...) {
    cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, ...);
}
```
那个 `data`（host 源）**不要求 pinned、不要求 ggml 分配、不要求任何特殊来源**——任意 host 指针都行。所以它可以来自 HYLE `mem::Allocator`，也可以来自 `PinnedStaging` 直接拥有并 bump 的 pinned host block。`tensor_get`（取出）同理。

**数据流（每段谁管，清清楚楚）**：
```
HYLE mem 管的 host 数据                    ← HYLE mem 抽象，我们说了算
   │  ggml_backend_tensor_set(data = HYLE host 指针)
   │  = cudaMemcpyHostToDevice(任意 host 指针 → 显存)
   ▼
ggml gallocr 分配的 device 显存             ← ggml 管（CUDA 上必须）
   │  ggml 算子在这上面算（cuBLAS/自有 kernel）
   ▼
ggml 结果显存
   │  ggml_backend_tensor_get(data = HYLE host 指针)
   ▼
HYLE mem 管的 host 结果                     ← 回到 HYLE mem 抽象
```

**性能接口点**：`tensor_set` 的 host 源若是 **pinned host RAM**，H2D 可走更适合 DMA/异步重叠的路径。当前 `PinnedStaging` 直接拥有一个 pinned host block 并在其中 bump，不经过 mem 池；它可把连续准备区交给 `tensor_set`。`pinned_allocator` 也可选择与 `SharedPool` 组合，但目前没有生产消费者，不能写成现役数据路径。

## 3. 边界不模糊——两种 kernel 来路对应两种显存归属

这条边界和 compute-design「发现 5：三层 × 两条 kernel 来路矩阵」严丝合缝：

| kernel 来路 | 显存归属 | 零拷贝? |
|---|---|---|
| **借 ggml 算子**（AI：matmul/attn/…） | device 归 ggml（gallocr），host 归 HYLE mem | device 侧否；host 侧是 HYLE mem |
| **HYLE 自造 kernel**（compute-04 NVRTC / 06 SPIR-V：普通 CUDA 计算 + 几何核/网格/SpMV） | 全归 HYLE（Buffer/DevicePool/VMM） | **是**，kernel 直接在 HYLE 显存上算（已验证） |

**所以 HYLE 显存自管抽象没有名存实亡**——它在自造 kernel 路径完整成立（服务**一切非 AI 的通用 GPU 计算**：普通 CUDA 数值 kernel + 北极星② CAD/CAE 的几何/网格/求解核），只是在"借 ggml 算 AI"这条路上，device 侧让位给 ggml。这是**借 vs 造**的本质分工：借人家的 AI 算子，就用人家的显存规矩；要跑 ggml 给不了的任何 GPU 计算（哪怕只是 vecadd），就自造 kernel、显存全 HYLE 自管。

## 4. 对 compute-05 及后续的落地约束

- **compute-05 ggml_bridge**：matmul（及后续算子）走 `ggml_gallocr` 分配 device buffer + `tensor_set`(host in) / `tensor_get`(host out)。host 数据是 HYLE mem（优先 pinned）。合批同理（一图多算子，gallocr 一次分配、一次提交）。**不注入外部显存**。
- **compute-07 CPU 算子后端**：CPU 后端 ggml 支持 `buffer_from_ptr`，host 侧可真零拷贝（HYLE mem 直接当 tensor buffer）——CPU 路径比 CUDA 宽松，届时可用。
- **compute-08 HIP/Metal**：HIP 大概率同 CUDA（不支持外部指针）；Metal 待查。默认按"device 归 ggml"设计。
- **权重常驻**：若某消费者要反复用同一批权重（推理），权重拷进 ggml 显存**一次**、常驻复用（llama.cpp 模式），不是每次推理都拷。这是初始化成本，非热路径成本。

## 5. 参考

- ggml 铁证：`ggml-cuda.cu:5032`（CUDA `buffer_from_host_ptr=false`）、`:688`（tensor_set = 裸 cudaMemcpyHostToDevice，host 源无约束）、`:1924`（注入 dummy 的崩溃点）、`ggml-backend.h:430`（CPU `buffer_from_ptr` 存在、CUDA 无）。
- llama.cpp 铁证：`llama-model.cpp:1555`（CPU/mmap 权重 `buffer_from_host_ptr` 零拷贝）、`:1570`（GPU 权重 `alloc_ctx_tensors_from_buft` 真 buffer + 拷入）。
- 调试血泪：[`../debug/ggml-zerocopy-null-buffer-hunt.md`](../debug/ggml-zerocopy-null-buffer-hunt.md)。
- 关联：[`compute-design.md`](./compute-design.md) 发现 2/5/7/9 + compute-05 定案；[`../milestones/compute-05-ggml-bridge.md`](../milestones/compute-05-ggml-bridge.md)。
