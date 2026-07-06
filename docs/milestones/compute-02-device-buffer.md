# compute-02：device buffer —— 显存所有权 + host↔device 传输原语

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-01（Device/Stream/Event）|
| **状态** | ✅ 完成（2026-07-06，三平台真机：Win/WSL GPU 各 19/19 + mac host 18/19 + WSL TSan 零 race）|

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

## pinned host staging = 一个 BufferType 实例（+ 可选 pinned 块池）

- **显存 device buffer**：`cudaMalloc`，**compute 自有、不走 mem**（host 不能解引用显存，mem 的 CPU allocator 语义不适用——用户明确边界）。
- **pinned host staging**：`cudaMallocHost`（page-locked，DMA 直传 + 异步拷贝重叠）分配的是 **host RAM**，是独立 BufferType（`kHostPinned`，`is_host()=true`）。compute-02 的 `buffer_alloc` 给**通用任意 size 的 `cudaMallocHost` 原语**（每次 alloc/free，同 ggml）。pinned 分配失败**返 Status 不 silent fallback**（free 路径由 kind 固定，malloc fallback 会与 `cudaFreeHost` 失配）。
- **pinned 块池化 = compute 分配器（mem 正确用法，非 L1）——纠正先前判断**：先前写"池化归 L1"是**过度保守**。区分两层：
  - **pinned 块池（定长块复用）= 分配器，是 compute 基础设施**。「预分配一大块、内部复用」正是 arena/pool 换性能的本质，也是 `mem::MemoryArena` 的正确用法（`memory_arena.h`：定长块、无锁分片、地址路由归还）。做法：把 `cudaMallocHost`/`cudaFreeHost` 包成 `mem::Allocator` 的 3 个 fn-ptr → 喂 `mem::MemoryArena(pinned_alloc, block_size, block_count, align)` → 得一个 pinned 定长块池（如 4×64MB 轮转，llama.cpp `llama-model-loader.cpp:1580` 证明是刚需）。这**摊薄每次 `cudaMallocHost` 的开销**（cudaMallocHost 是重操作，注册 page-lock）+ 块复用。**mem 只管 host RAM，pinned 恰是 host RAM，完全适配**——这是 mem 结合 compute 加速的正确落点（不同于显存不走 mem）。
  - **staging ring 调度（tile 循环 + event 流水线）= L1 策略**。用 N 个 pinned 块 + `cudaEvent` 轮转做大数据分块上传的**编排**依赖 completion（compute-03 的 event），是推理引擎/渲染器的调度策略。**池是分配器（这层归 compute），ring 是用池 + event 的调度（这层归 L1/compute-03 之后）。**
- **范围**：pinned 块池作为 compute 基础设施本轮补上（小、示范 mem 正确用法）；ring 的 event 流水线编排待 compute-03。

## 抄全 ggml 分配器 SOTA（用户指示"ggml 很 SOTA 抄全一点"，2026-07-06 深挖）

> ggml 的内存管理是"预分配大块 + 内部复用换性能"的教科书。之前只抄了 buffer_type 接口，现把它的分配器层抄全（确切机制见调研）。分两类：**通用显存/host 手法（抄）** vs **张量图特化（不抄，HYLE 不做 tensor）**。

**① VRAM sub-allocator（显存"大块换性能"的核心，登记 compute-02b/后续，kernel 中间张量驱动）**：
- **VMM 版 `ggml_cuda_pool_vmm`（ggml-cuda.cu:478-585）= 用户"预留大块地址、按需提交物理"的教科书**：`cuMemAddressReserve` 一次预留 **32GB 连续虚拟地址**（指针稳定）+ 按需 `cuMemCreate`/`cuMemMap` 映射物理页到 pool 末尾（物理内存按需增长、永不缩=缓存复用）+ 分配是 `pool_addr+pool_used` **O(1) bump**。约束：**LIFO 逆序释放**（栈式，配 RAII 作用域），非通用堆。
- **leg fallback `ggml_cuda_pool_leg`（361-474）**：无 VMM 时用定长数组缓存已 cudaFree 的块 + **best-fit 复用** + **look-ahead 超额分配 `1.05*size` 向上取整 256B**（这正是"分配大于所需换性能"）+ OOM 时刷缓存重试。
- **RAII `ggml_cuda_pool_alloc<T>`（common.cuh:1150）**：中间缓冲 alloc/free 作用域化，直接抄。
- **解决的问题**：每个 kernel 中间张量都 cudaMalloc 太慢（同步、重）→ pool 复用。**这是 compute-04 kernel 落地时的刚需**——现在无 kernel 消费者，故**登记为 compute 显存 sub-allocator（compute-02b 或随 compute-04）**，不投机预造，但机制已挖清、届时抄全。**显存池不走 mem**（mem 只管 host RAM）。

**② host 数据准备区 + 合批提交（用户新洞见：mem::Arena over pinned，连续则一次提交）——本轮做**：
- **想法**：用 `mem::Arena`（`memory_arena.h:257`，bump 分配器）在一个 pinned 块上**连续 bump 出多份待上传数据**；因为它们内存连续，**整段用一次 `set(offset, arena_base, total)` 提交显存**，而非 N 次小拷贝——**arena 的连续性天然实现合批**（用户说的"合批换性能"）。
- **落地**：`mem::MemoryArena(pinned_alloc, block_size, block_count)` 做 pinned 定长块池（摊薄 cudaMallocHost 重开销 + 块复用）；每个块上开 `mem::Arena` bump 准备数据；准备好的连续段一次 H2D。**这是 mem 结合 compute 的正确落点**——pinned 是 host RAM，完全适配 mem（区别于显存不走 mem）。ggml **没做 pinned 池化**（每次 cudaMallocHost 直分，ggml-cuda.cu:1503）——**这是 HYLE 可超越 ggml 的点**。
- **pinned 块池（分配器）是 compute 基础设施**；**用池 + event 做 ring 流水线编排**是 L1/compute-03（见 pinned 节的两层区分）。本轮补池 + Arena 准备区 + 合批 set；event ring 待 compute-03。

**③ 通用小件（抄）**：`aligned_offset`/`GGML_PAD` 对齐（compute-02 buffer_alignment 已含）、multi-buffer 按 max_size 切块（max_size 已暴露，切分逻辑上层用）。

**④ 张量图特化（不抄）**：`ggml_dyn_tallocr`/graph allocator 的生命周期复用、inplace、张量图遍历（ggml-alloc.c）——绑定静态计算图的张量生命周期，HYLE 不做 tensor 则不适用。但其 **free-list + 相邻块合并（coalesce）算法（ggml-alloc.c:311-348）** 是通用碎片管理精华，若显存 sub-allocator 要支持**乱序释放**（VMM 版只 LIFO）可移植该数据结构——登记备查。

调研证明 **async 拷贝（compute-02 的核心价值之一）离不开 completion（compute-03）**：ggml 的 `set_tensor_async`（`ggml-cuda.cu:3151`）是 `cudaMemcpyAsync` 进 stream **不 sync**，靠 event 观察完成。所以：
- **compute-02 落**：Buffer 全套 **同步语义**（`set/get` = 发射 + 立即 `cudaStreamSynchronize`，对调用者呈现同步，同 ggml buffer 层 `ggml-cuda.cu:684-690`）。**接口里预留 `set_async/get_async(Stream&)` 签名但先同步实现**。
- **compute-03 落后**：async 变体接上 `co_await`（拷贝进 stream → record event → 轮询 → 唤醒 fiber），**不改 Buffer 接口**（签名已预留）——一次定对、分两步实现，不返工。

## L0/L1 边界（关键：流式灌大显存 / 合批是 L1，不是 compute-02）

用户场景「显存>内存时分块流式写入 + 合批提交」拆成两层，**compute-02 只做底层原语**：
- **L0（compute-02）= 原语 + 分配器**：`set(offset,size)` 部分写入 + async 拷贝 + pinned staging alloc 原语 + **pinned 块池（compute 分配器，mem::MemoryArena over cudaMallocHost）** + `max_size`/multi-buffer + buffer 复用。**铁证：ggml/llama.cpp 的"分块流式"靠调用方循环调 set_tensor + pinned ring 拼**，库本身不做单 tensor 的 chunk memcpy——但**池化（分配器）是 compute 基础设施，不是 L1**。
- **L1（上层，不在 compute-02）= 调度策略**：tile 循环、**pinned ring 的 event 流水线编排**、按显存容量决定切几块、合批换吞吐。这是推理引擎/渲染器的调度（llama.cpp 的 4×64MB ring 轮转 + event 在 model-loader 里）——**用池 + event 做编排**是 L1，**池本身**是 compute 分配器。按 scope 铁律「底座永远是底座」，编排在 L1，分配器在 compute。
- compute-02 的义务：**把原语 + 分配器备齐**（offset 写 + pinned alloc + pinned 块池 + async 签名 + max_size），让 L1 用它们拼出流式灌显存 ring——不把 tile/event 调度写进 compute-02。

## 明确不做（防蔓延）

| 不做 | 去向 | 为什么 |
|---|---|---|
| tensor / 子视图 / shape | L1 应用层 | Buffer 只给 `base+size`；offset 切分是调用方的事（ggml 也在 alloc 期算进 tensor->data）|
| 流式灌显存 tile 调度 / 合批策略 | L1（推理引擎）| compute-02 只备原语（见 L0/L1 边界）|
| async co_await 拷贝的完整实现 | compute-03 | 依赖 completion；compute-02 预留签名先同步实现 |
| unified memory（DGX/Mac 统一内存）| 按需 | 统一内存无 host↔device 互传，is_host 语义覆盖；具体后端实例后续 |
| 2D/strided 拷贝 | 按需补 | 合批/transposed-V 用；先留待真实消费者（KV transposed V）驱动 |

## 交付物（实际落地）
- `include/hyle/compute/types.h`（+`BufferKind{kDevice,kHostPinned}`）
- `include/hyle/compute/buffer.h`（`BufferType` + `Buffer`，内联 `{vt,ptr,size,ordinal,kind,is_host}` 无 pimpl；async 签名注释预留）+ 自由函数 `device_buffer_type`/`host_buffer_type`
- `src/compute/backend.h`（vtable +9 fn-ptr：buffer_alignment/max_size/is_host/alloc/free/set/get/copy/clear）
- `src/compute/compute.cc`（BufferType/Buffer 分派 + offset 越界/跨后端 fatal-assert）
- `src/compute/cuda_backend.cu`（cudaMalloc/cudaMallocHost + cudaMemcpyAsync+sync 的 set/get/copy + cudaMemsetAsync clear；pinned **不池化**=通用原语，见上纠正）
- `src/compute/host_backend.cc`（malloc + memcpy/memset，两 kind 均 is_host=true）
- `tests/compute/buffer_test.cc`（9 测试，验收用例 = KV-cache 式大 buffer 反复 offset 写；GPU 真机 + host fallback）

**cloc 实测**：净增 **~473 生产**（types 15 + backend.h 40 + compute.cc 108 + cuda 101 + host 64 + buffer.h 145）+ 测试 142。473«2-3k 预测=buffer 是薄抽象转发层（alloc/copy 转发 cudaMalloc/Memcpy），重量在 compute-04 kernel/05 bridge；非 stub（三平台真跑 + KV-cache + TSan 为证）。

## 实测结果（2026-07-06，三平台真机）
- **WSL（GCC14+nvcc）**：全量 compute 19/19（compute-01 的 10 + buffer 9）；buffer 9/9 在 RTX 4070 真跑——**KvCachePattern**（512 token 逐行 offset 写 + 全量读回校验）+ **PinnedStagingToDevice**（pinned host→VRAM→回读）+ AllocSetGetRoundTrip/Clear/CopyFrom(D2D)/MoveSemantics。
- **WSL TSan（host-only）**：buffer 9/9 零 data race。
- **Windows（MSVC19.50+nvcc）**：全量 19/19，GPU 真跑。
- **macOS（Apple clang arm64，无 CUDA）**：18/19 + 1 skip（GPU event 测试）——host fallback 两 kind 均普通 RAM，KV-cache/staging 模式在第三平台跑通。

## 交付物（原预期，保留对照）
- ~~`src/compute/cuda_backend.cu`（... + pinned MemoryArena 池）~~ → **纠正**：pinned 不池化，通用 `cudaMallocHost` 原语（池化归 L1，见 pinned 节纠正）。

## 验收门槛（✅ 达成）
- 三平台真机（Win MSVC+nvcc / WSL GCC14+nvcc / mac host）：device buffer 分配 + set/get(offset) 往返数据校验 + copy_from + clear。✅
- **KV-cache 验收**：一块大 buffer，模拟每 token 按动态行 offset 反复写 + 读回校验（命中 `set(offset,size)` 反复写语义）。✅ 双 GPU 真机
- pinned staging BufferType：`cudaMallocHost` 分配 + 无 CUDA fallback（mac host 两 kind 普通 RAM）。✅（池化归 L1，非本里程碑）
- 显存 `cudaMalloc` 不走 mem、pinned host `cudaMallocHost` 也不走 mem（无池化）—— 边界成立。✅
- WSL TSan（host-only）零 race ✅；cloc 验真回填 ✅；对抗自审对照 ggml buffer_type 覆盖（含 pinned 池化纠正）✅。

## 参考
- compute-design「模块形状」buffer.h + §「device 内存归 compute」+「compute-06 kompute 映射」（Vulkan buffer 用 memory-type 选择 + staging 双缓冲）。
- ggml `ggml-backend-impl.h:17-62`（buffer_type_i/buffer_i）、`ggml-cuda.cu:684/1497/1530`（set_tensor/host_malloc/host_buffer_type）、`ggml-alloc.c:1172`（max_size 切多块）。
- llama.cpp `llama-kv-cache.cpp:247/1343`（KV 一次性分配 + offset 反复写）、`llama-model-loader.cpp:1580-1633`（pinned staging ring）。
- `mem/memory_arena.h:35`（MemoryArena 固定块池，pinned 池底子）；compute-01（Device/Stream/Event，前置）。
