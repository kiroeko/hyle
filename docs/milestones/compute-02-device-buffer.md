# compute-02：device buffer —— 显存所有权 + host↔device 传输原语

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-01（Device/Stream/Event）|
| **状态** | 📋 规划（接口开工前定形，2026-07-06 深调研 ggml/llama.cpp/kompute 后）|

> **范围**：compute-design「模块形状」的 `buffer.h`——**一块 device 内存的所有权 + host↔device 拷贝 + 带 offset 的部分读写**。镜像 ggml 的 `buffer_type`/`buffer` 模型（枚举/能力查询薄包，内存所有权 + stream 绑定自造）。取代 compute-01 里 `Stream::device_alloc`/`memcpy_async` 那两个「脊验证最小辅助」。
>
> **借 vs 造**：buffer_type 的 trait（alignment/max_size/is_host）照镜 ggml；**内存所有权 + 生命周期 + pinned 池化自造**（ggml pinned 不池化，是可超越的弱点）。CUDA 后端直调 `cudaMalloc`/`cudaMemcpy`/`cudaMallocHost`。
>
> **一句话职责**：Buffer 给上层「一块可 base+offset 读写的 device 内存」这个**原语**；「怎么用它拼推理引擎/流式灌显存」是 L1 的事（见下「L0/L1 边界」）。

## 为什么现在做 / 接口为何能一次定对

compute-01 的 `device_alloc`/`memcpy_async` 是脊验证的临时辅助，正式的 device 内存抽象是 compute-02。**动手前用三路调研把接口定死**（避免返工——下游 compute-03/04/05 全依赖 Buffer）：
- **ggml buffer_type 模型**（`ggml-backend-impl.h:17-62`）：`set/get_tensor` 签名**带 offset+size**，pinned host 建模成独立 buffer_type（`ggml_backend_cuda_host_buffer_type`，`ggml-cuda.cu:1530`）。
- **下游怎么消费**（agent 反推）：kernel launch **只要 `device_ptr()` 一个裸首指针**（offset 由调用方 `base+off` 自算，ggml 在 alloc 期就把 `base+offset` 算进 `tensor->data`，launch 时 buffer 已不参与）；bridge 只要 `{base,size,alignment,is_host}` 四件套交给 ggml 当 tensor backing。**Buffer 不需要子视图/tensor 概念**（那是 L1）。
- **北极星真实用法**（llama.cpp）：KV-cache 是**同一持久大 buffer 运行期按动态 token 行 offset 反复写**（`llama-kv-cache.cpp:1343`）——这是 `set(offset,size)` 的决定性验收用例；权重加载用 pinned staging ring（4×64MB + event 流水，`llama-model-loader.cpp:1580-1633`）。

## Buffer 接口（定形——下游反推 + 北极星验证，不多不少）

```
// BufferType：一类内存的工厂 + trait（镜像 ggml buffer_type_i；一种内存模型=一个实例）
struct BufferType {
  alloc_buffer(size) -> Result<Buffer>   // 唯一分配入口
  alignment() -> size_t                  // CUDA 恒 128（ggml-cuda.cu:799）
  max_size()  -> size_t                  // 单 buffer 上限（默认 SIZE_MAX；驱动上层切多块）
  is_host()   -> bool                    // true=host 内存、可直接解引用（决定 copy 分派）
};
// device 暴露：device.buffer_type()（device-local 显存）+ device.host_buffer_type()（pinned staging）

class Buffer {                            // 一块具体 device 内存（move-only，析构 free）
  void* device_ptr() const;              // 裸首指针 = ggml get_base;compute-04 kernel launch 唯一所需
  size_t size() const;
  size_t alignment() const;
  bool is_host() const;                   // = 所属 buffer_type 的 is_host
  // 同步拷贝（compute-02 落地）：内部 cudaMemcpy((char*)base+offset,...);host 后端 memcpy
  Status set(size_t offset, const void* src, size_t size);   // H2D 部分写入（KV-cache 刚需）
  Status get(size_t offset, void* dst, size_t size) const;   // D2H 部分读出
  Status copy_from(const Buffer& src, ...);                  // D2D/D2H/H2D（= ggml cpy_tensor）
  Status clear(uint8_t value);                               // = ggml clear（KV 初始化清零）
  // async 变体：签名预留、compute-02 先同步实现（发射+立即 sync,同 ggml buffer 层），
  // compute-03 completion 落地后接 co_await（见下「与 compute-03 的耦合」）
  // Status set_async(offset, src, size, Stream&);  get_async(...);
};
```

**手动虚表分派**（同 compute-01）：`BufferType`/`Buffer` 内联 `{const ComputeBackend* vt, void* native, ...}`，vtable 加 buffer 相关 fn-ptr（`buffer_alloc/free/set/get/copy/clear`）。每后端一个组合单元，非上帝类 switch。

## pinned host staging = 一个 BufferType 实例（结合 mem 加速，超越 ggml）

- **显存 device buffer**：`cudaMalloc`，**compute 自有、不走 mem**（host 不能解引用显存，mem 的 CPU allocator 语义不适用——用户明确边界）。
- **pinned host staging**：`cudaMallocHost`（page-locked，DMA 直传 + 异步拷贝重叠）分配的是 **host RAM**，走 `mem::Allocator`。**ggml 每次 alloc/free 不池化（弱点）**；HYLE 用 `mem::MemoryArena(host_pinned_alloc, block_size, block_count, alignment)`（`memory_arena.h:35`，固定大小块池 + 地址路由分片归还）做 **pinned 块池**——同大小 staging buffer churn 复用，超越 ggml。失败/无 CUDA fallback 到普通 host buffer（同 ggml `GGML_CUDA_NO_PINNED`）。

## 与 compute-03 的耦合（为何接口预留 async、实现分两步）

调研证明 **async 拷贝（compute-02 的核心价值之一）离不开 completion（compute-03）**：ggml 的 `set_tensor_async`（`ggml-cuda.cu:3151`）是 `cudaMemcpyAsync` 进 stream **不 sync**，靠 event 观察完成。所以：
- **compute-02 落**：Buffer 全套 **同步语义**（`set/get` = 发射 + 立即 `cudaStreamSynchronize`，对调用者呈现同步，同 ggml buffer 层 `ggml-cuda.cu:684-690`）。**接口里预留 `set_async/get_async(Stream&)` 签名但先同步实现**。
- **compute-03 落后**：async 变体接上 `co_await`（拷贝进 stream → record event → 轮询 → 唤醒 fiber），**不改 Buffer 接口**（签名已预留）——一次定对、分两步实现，不返工。

## L0/L1 边界（关键：流式灌大显存 / 合批是 L1，不是 compute-02）

用户场景「显存>内存时分块流式写入 + 合批提交」拆成两层，**compute-02 只做底层原语**：
- **L0（compute-02）= 原语**：`set(offset,size)` 部分写入 + async 拷贝 + pinned staging 池 + `max_size`/multi-buffer + buffer 复用。**铁证：ggml/llama.cpp 的"分块流式"完全靠调用方循环调 set_tensor + pinned ring 拼**，库本身不做单 tensor 的 chunk memcpy。
- **L1（上层，不在 compute-02）= 策略**：tile 循环、双缓冲 pinned ring、按显存容量决定切几块、合批换吞吐。这是推理引擎/渲染器的调度策略（llama.cpp 的 4×64MB ring 就在它的 model-loader 里，不在 ggml buffer 层）。按 scope 铁律「底座永远是底座」，**这层在 L1**。
- compute-02 的义务：**把原语备齐**（offset 写 + pinned 池 + async + max_size），让 L1 能拼出流式灌显存——不把 tile 调度写进 compute-02。

## 明确不做（防蔓延）

| 不做 | 去向 | 为什么 |
|---|---|---|
| tensor / 子视图 / shape | L1 应用层 | Buffer 只给 `base+size`；offset 切分是调用方的事（ggml 也在 alloc 期算进 tensor->data）|
| 流式灌显存 tile 调度 / 合批策略 | L1（推理引擎）| compute-02 只备原语（见 L0/L1 边界）|
| async co_await 拷贝的完整实现 | compute-03 | 依赖 completion；compute-02 预留签名先同步实现 |
| unified memory（DGX/Mac 统一内存）| 按需 | 统一内存无 host↔device 互传，is_host 语义覆盖；具体后端实例后续 |
| 2D/strided 拷贝 | 按需补 | 合批/transposed-V 用；先留待真实消费者（KV transposed V）驱动 |

## 交付物（预期）
- `include/hyle/compute/buffer.h`（`BufferType` + `Buffer`，async 签名预留）
- `src/compute/backend.h` 扩展 vtable（buffer_alloc/free/set/get/copy/clear + host_buffer_type）
- `src/compute/cuda_backend.cu`（cudaMalloc/Memcpy/MallocHost + pinned MemoryArena 池）
- `src/compute/host_backend.cc`（malloc + memcpy，is_host=true）
- `tests/compute/buffer_test.cc`（**验收用例 = KV-cache 式大 buffer 反复 offset 写** + 往返校验 + pinned 池复用；三平台，GPU 真机 + host fallback）

## 验收门槛
- 三平台真机（Win MSVC+nvcc / WSL GCC14+nvcc / mac host）：device buffer 分配 + set/get(offset) 往返数据校验 + copy_from + clear。
- **KV-cache 验收**：一块大 buffer，模拟每 token 按动态行 offset 反复写 + 读回校验（命中 `set(offset,size)` 反复写语义）。
- pinned staging BufferType：`cudaMallocHost` 分配 + MemoryArena 池复用 + 无 CUDA fallback 到普通 host。
- 显存 `cudaMalloc` 不走 mem、pinned host 走 mem —— 边界断言。
- WSL TSan（host-only）零 race；cloc 验真回填；对抗自审对照 ggml buffer_type 覆盖。

## 参考
- compute-design「模块形状」buffer.h + §「device 内存归 compute」+「compute-06 kompute 映射」（Vulkan buffer 用 memory-type 选择 + staging 双缓冲）。
- ggml `ggml-backend-impl.h:17-62`（buffer_type_i/buffer_i）、`ggml-cuda.cu:684/1497/1530`（set_tensor/host_malloc/host_buffer_type）、`ggml-alloc.c:1172`（max_size 切多块）。
- llama.cpp `llama-kv-cache.cpp:247/1343`（KV 一次性分配 + offset 反复写）、`llama-model-loader.cpp:1580-1633`（pinned staging ring）。
- `mem/memory_arena.h:35`（MemoryArena 固定块池，pinned 池底子）；compute-01（Device/Stream/Event，前置）。
