# compute-02：device buffer —— 显存所有权 + host↔device 传输原语

> **当前边界（2026-07-22）：**Buffer 分别持有 opaque `native_handle()` 与可选 `host_data()`；Vulkan 分别返回 `VkMem*` 与 `VkMem::mapped`。非-coherent host memory 通过按 `nonCoherentAtomSize` 对齐的范围 `flush_host()` / `invalidate_host()` 维护 cache visibility；它们不等待 GPU completion。零消费者且建立在错误指针合同上的 pinned allocator / 单块 staging API 已删除。未来 staging 只能由 HostBuffer + Event + backpressure ring 组成。

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-01（Device/Stream/Event）|
| **状态** | ✅ ownership + offset copy + native/mapped/coherence + async lifetime 合同已闭合；错误 staging 实验已删除（三平台与 sanitizer 证据见验收节，non-coherent 真机能力缺失明确标注） |

> **范围**：compute-design「模块形状」的 `buffer.h`——**一块 device 内存的所有权 + host↔device 拷贝 + 带 offset 的部分读写**。镜像 ggml 的 `buffer_type`/`buffer` 模型（枚举/能力查询薄包，内存所有权 + stream 绑定自造）。取代 compute-01 里 `Stream::device_alloc`/`memcpy_async` 那两个「脊验证最小辅助」。
>
> **借 vs 造**：alignment/max_size/kind trait 照镜 ggml；backend allocation ownership、同步 copy 与 native/mapped accessor 由 HYLE compute 自己定义。pinned pooling 不属于本里程碑。
>
> **一句话职责**：Buffer 给上层「一块 owned backend allocation + offset copy」原语；tensor/shape 与流式编排在上层。

## 需求与当前合同

compute-01 的 `device_alloc`/`memcpy_async` 是脊验证辅助，compute-02 保留下来的正式原语是 owned Buffer + offset-aware `set/get/copy/clear`：

- ggml `set/get_tensor` 与 llama.cpp KV-cache 都证明 offset+size 部分读写是稳定 L0 需求。
- Buffer 不做 tensor/shape/subview；这些属于上层数据模型。
- backend allocation result 是 **native handle**，不是跨后端统一的裸 device pointer：CUDA 可恰好是 device address，Vulkan 是持有 `VkBuffer/VkDeviceMemory/mapped` 的 `VkMem*`。
- `native_handle()` 只回传同一 backend 给 kernel launch；`host_data()` 只对 host-visible kind 有效。直接 CPU 写后调用 `flush_host()`，读取 device 写入后在 completion 之后调用 `invalidate_host()`。`copy_from()` 自动维护 mapped-cache visibility，但不跨 Stream 等待 producer；`set/get/copy/clear` 与直接 host 访问前，相关 Buffer 必须没有未完成 GPU 使用。
- `launch/launch_batch/replay` 引用的 Kernel 与 Buffer allocation 必须存活到对应 Stream/Event 完成；提前析构或 move-assign 覆盖 owning Buffer 会释放仍在途的资源。
- 合法 `size==0` 的 set/get/copy/clear/flush/invalidate 在公共层统一成功 no-op，避免 Vulkan 发出 size=0 的非法 copy/fill 命令。

`BufferType` / `Buffer` 以内联 `{ComputeBackend*, native, host, size, ordinal, kind}` + 手动 vtable 分派，不泄漏 backend 类型。

## pinned host buffer 与 staging 边界

- device VRAM、pinned/unified/mapped host memory 都归 compute；普通 mem allocator 不拥有 backend coherence 或 completion 合同。
- `kHostPinned` 表示 backend host-visible allocation kind。Buffer 同时保留 native handle、CPU mapped pointer 与 Device identity；分配失败返回 `Result`，不 silent fallback。
- 不提供 pinned mem allocator 或同步单块 staging convenience：它们无法表达 native-handle ownership、coherence 与 completion。
- pinned block pool 也不等于 staging ring。真正 ring slot 还必须拥有 Event 与 `free→filling→in-flight→free` 状态；无空 slot 时有明确 backpressure。只有出现至少两个异步传输在途的真实消费者才实现。
- Vulkan 非 ReBAR 的同步 `set/get` 内部允许持有一个懒创建、按最大单次传输增长的 staging slice；它随 Buffer 所有、每次提交后立即等待，只解决同步可达性，不对外伪装成 staging ring，也不按 device buffer 全尺寸预留 host memory。
- VRAM VMM/legacy/cached-native-block sub-allocator 已独立落在 compute-02c `DevicePool`，与 host staging、mem::SharedPool 无关；Vulkan 侧缓存并 best-fit 复用完整的 `VkBuffer+VkDeviceMemory` block，不再是逐次 allocate/free 的假 pool。

同步 `set/get/copy/clear` 是当前已实现合同。真正 async copy 必须绑定 Stream，并由 Event 保持 host slot 生命周期直到 completion；不能把“enqueue 后立即 sync”称为 async，也不能只预留注释签名就称接口已定。

## L0/L1 边界

- **L0 Buffer**：owned backend allocation、offset copy、clear、size/kind/max-size trait、分离的 native handle / mapped host pointer 与显式 coherence。
- **compute staging 基础设施**：HostBuffer + Event + backpressure slot ring；只在 async overlap workload 出现后实现。
- **L1 策略**：tile 大小、按显存容量切块、权重/激活编排、连续批处理。llama.cpp 的 4×64MB upload ring 证明完整 HostBuffer+Event 形状，不证明裸 pinned allocator 或固定块池可独立替代它。

## 明确不做（防蔓延）

| 不做 | 去向 | 为什么 |
|---|---|---|
| tensor / 子视图 / shape | L1 应用层 | Buffer 只给 owned allocation + size + offset copy；数据模型由调用方定义 |
| 流式灌显存 tile 调度 / 合批策略 | L1（推理引擎）| compute-02 只备原语（见 L0/L1 边界）|
| async copy / StagingRing | compute 后续 | completion 原语已有，但缺至少两个在途传输的真实 overlap 消费者；不能以上游 comment 预留冒充能力 |
| unified memory（DGX/Mac 统一内存）| compute 按需 | coherence/device visibility 是独立合同，不能由 `is_host` 一个布尔值覆盖 |
| 2D/strided 拷贝 | 按需补 | 合批/transposed-V 用；先留待真实消费者（KV transposed V）驱动 |

## 交付物（实际落地）
- `include/hyle/compute/types.h`（+`BufferKind{kDevice,kHostPinned}`）
- `include/hyle/compute/buffer.h`（`BufferType` + move-only `Buffer`，内联 `{vt,native,host,size,ordinal,kind}`；`native_handle/host_data/flush_host/invalidate_host`）
- `src/compute/backend.h`（buffer vtable：trait、alloc/free、flush/invalidate、set/get/copy/clear）
- `src/compute/compute.cc`（BufferType/Buffer 分派 + offset 越界/跨后端 fatal-assert）
- `src/compute/cuda_backend.cu`（cudaMalloc/cudaMallocHost + cudaMemcpyAsync+sync 的 set/get/copy + cudaMemsetAsync clear；pinned **不池化**=通用原语，见上纠正）
- `src/compute/host_backend.cc`（malloc + memcpy/memset，两 kind 均 is_host=true）
- `tests/compute/buffer_test.cc`（14 测试：KV-cache offset、HostBuffer direct mapping/coherence、copy、move）+ `vulkan_test.cc` 的 native-handle/mapped-pointer 分离真机测试

**cloc 实测（当前 P0 交付相对基线）**：生产代码 +188/-120（净 +68，另 70 行修改），测试/CMake +257/-121（净 +136，另 23 行修改）。净增长小是因为删除了 103 行错误 pinned 公开头和 173 行对应测试，同时补齐 native/mapped/coherence 合同与回归测试；不是以新增类型堆体量。

## 实测结果（当前实现）
- **Windows MSVC + RTX 4070（2026-07-22）**：mem 126/126；compute 78 pass / 2 预期 skip；Vulkan 14 pass / 2 skip（pipeline-cache opt-in、物理设备无 non-coherent host-visible type）；强制 non-ReBAR staging 11 pass / 同 2 skip；GPU await 2/2。
- **Windows ASan**：mem 126/126；Buffer + Buffer contract focused 22/22，零 sanitizer 报告。
- **WSL GCC14 TSan（host-only）**：mem 126/126；compute 71 pass / 9 GPU/cross-backend skip，零 ThreadSanitizer 报告。
- **macOS Apple M4**：mem 126/126；host compute 71 pass / 9 GPU skip，GPU await 2/2；显式配置 Homebrew Vulkan loader + MoltenVK ICD 后，Vulkan 14 pass / 2 skip，强制 staging 11 pass / 2 skip。真机还抓出旧测试把 512 B 误当所有设备 push-constant 上限的假设，现改用 `UINT32_MAX` 契约输入。
- **non-coherent 边界**：test-only 选择器穷举全部 compatible memory type；Win RTX 4070 与 Apple M4 都没有 non-coherent host-visible type，driver round-trip 明确 skip。独立 synthetic 测试按设备真实 `nonCoherentAtomSize` 验证 interior/cross-atom 范围归一化；transfer-chain 测试覆盖 `set/clear/copy/get` 的 transfer-write→transfer-read barrier，并在两平台强制 staging 真跑。硬件未提供的 flush/invalidate driver 调用仍不写成真机通过。

staging ring 尚未实现；同步 HostBuffer/staging fallback 测试不计为 async ring 验收。

## 参考
- compute-design「模块形状」buffer.h + §「device 内存归 compute」+「compute-06 kompute 映射」（Vulkan buffer 用 memory-type 选择 + 同步 transfer-sized lazy staging）。
- ggml `ggml-backend-impl.h:17-62`（buffer_type_i/buffer_i）、`ggml-cuda.cu:684/1497/1530`（set_tensor/host_malloc/host_buffer_type）、`ggml-alloc.c:1172`（max_size 切多块）。
- llama.cpp `llama-kv-cache.cpp:247/1343`（KV 一次性分配 + offset 反复写）、`llama-model-loader.cpp:1580-1633`（pinned staging ring）。
- compute-01（Device/Stream/Event，前置）、compute-03（Event completion 原语）、compute-design 的 HostBuffer/StagingRing 边界。
