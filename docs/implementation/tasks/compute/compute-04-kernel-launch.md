# compute-04：kernel launch —— 任意 kernel 提交抽象

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-01（Stream/Event）、compute-02（Buffer.device_ptr）、compute-02c（DevicePool，kernel 中间张量）、compute-03（co_await 完成）|
| **状态** | ✅ 完成（2026-07-06，B 档三平台真机：Win/WSL GPU NVRTC vecadd 端到端真编真跑 + mac host fn-ptr + WSL TSan）|

> **范围（B 档，用户定"基础实现完整"）**：compute-design「模块形状」的 `kernel.h`——把**任意 kernel** 提交到 GPU stream 的完整抽象：**源码字符串 → NVRTC 运行时编译(PTX) → 模块加载 → launch** 端到端 + 给定 PTX/cubin 直接加载入口 + in-process 编译缓存。统一 CUDA raw kernel 与后续 SPIR-V compute shader（compute-06），是 05（ggml_bridge 借算子）与 06（Vulkan）的上层公共接口。
>
> **两个参考来源（各补一半）**：**kernel 启动工程**参考 ggml（grid/block、launch_bounds、错误检查、参数平铺——ggml 全用编译期 `<<<>>>`）；**运行时编译链**参考 **CCCL**（`c/parallel/src/` 的 nvrtc→模块→launch 生产链——ggml 全仓零 nvrtc/cuLaunchKernel，这块无可参考，改抄 CCCL）。

## 核心抉择：driver `cuLaunchKernel` + NVRTC 运行时编译（B 档完整版，深调研定案）

**① 底座 = driver `cuLaunchKernel`，非编译期 `<<<>>>`**（ggml 净新缺口）：

| | 编译期 `<<<>>>` | driver `cuLaunchKernel` + `cuModuleLoad` |
|---|---|---|
| kernel 来源 | 编译期已知（nvcc 静态编进库）| **运行时**（NVRTC 编 / PTX/cubin 加载 → `CUfunction`）|
| 任意性 | ✗ 必须编译期类型化已知 | ✓ `void** args` 容纳任意签名 |
| 与 SPIR-V/Vulkan 同构 | ✗ | ✓（module+entry+args 同 `vkCmdDispatch`）|
| ggml/CCCL | ggml 用 `<<<>>>` | **CCCL 生产链用这个**（HYLE 照抄）|

**② B 档做完整运行时编译链**（源码字符串 → 可 launch 的 kernel）：
- **NVRTC 编译**：`nvrtcCreateProgram → nvrtcCompileProgram → nvrtcGetPTX`（经典 PTX 路线，CCCL 教学版 nvrtcc_build.h:80-115）。编译选项 `-arch=sm_XX`（HYLE 探测 device compute capability 拼）+ `-std=c++20`。**先用 PTX 不上 nvJitLink/LTO-IR**（那是 CCCL 链接多 TU 的加档；单 kernel PTX + 驱动 JIT 一次足够，登记后续）。
- **name mangling**：`nvrtcAddNameExpression`(编译前) + `nvrtcGetLoweredName`(编译后) 拿 mangled 名当 launch key（CCCL command_list.h:165/234，通用于 C++ 模板 kernel，优于强制 `extern "C"`）；简单 kernel 也支持 `extern "C"` 直接用名。
- **模块加载**：`cuModuleLoadData(PTX) → cuModuleGetFunction(lowered_name)`（经典 API；CCCL 生产用新 Library API `cuLibraryLoadData` 是加档，基础版用 cuModule 足够）。
- **launch**：`cuLaunchKernel(CUfunction, grid xyz, block xyz, shmem, stream, void** args, extra=0)`，args = 指向各参数的指针数组（CCCL for.cu:51/63）。
- **compile/load 分离**（照 CCCL 分层）：`Kernel::compile(source, entry)` → payload(PTX) → `load` → CUfunction；payload 可缓存。
- **arch 探测**：`cuDeviceGetAttribute` compute capability → `cc=major*10+minor` → `-arch=sm_{cc}`（HYLE 给定 device 自己探测，不像 CCCL 由调用方传）。
- **in-process 编译缓存**：按 `hash(source)+arch` 缓存 PTX（用 HYLE SwissTable + XXH3）——避免同 kernel 重复 NVRTC 编译。CCCL 生产链无 in-process 缓存（靠 AOT 序列化），HYLE in-process 场景值得自建（参考 CCCL test 层 build_cache_t 的 map 模型）。
- **错误日志 fail-fast**：编译后无条件 `nvrtcGetProgramLog`（有内容即打 stderr）+ `check()` 转 Status（CCCL command_list.h:259-267）。
- **给定 PTX/cubin 直接加载**：`Kernel::load_ptx(ptx, entry)` 入口（跳过 NVRTC，直接 cuModuleLoadData），供已有 PTX 的消费者。

## Kernel launch 抽象（定形）

```
// A loaded, launchable kernel. Backend-opaque handle (CUmodule+CUfunction / VkPipeline).
// Move-only (owns the module/pipeline). B 档三条加载路径:
class Kernel {
  // ① 运行时编译 CUDA C 源码字符串 -> PTX -> 加载 (NVRTC 完整链). `entry` 是 kernel
  //    函数名 (extern "C") 或 C++ name expression. arch 自动探测 device compute cap.
  static Result<Kernel> compile(const Device&, const char* source, const char* entry);
  // ② 给定已编译 PTX 直接加载 (跳过 NVRTC, 供已有 PTX 的消费者).
  static Result<Kernel> load_ptx(const Device&, const char* ptx, const char* entry);
  //    (cubin 变体 + Vulkan SPIR-V 变体 -> compute-06)
  bool valid() const;
};

struct Dim3 { unsigned x = 1, y = 1, z = 1; };
struct LaunchConfig {
  Dim3 grid;               // block 数
  Dim3 block;              // 每 block 线程数
  unsigned shared_mem = 0; // 动态 shared memory 字节
};

// 提交 kernel 到 stream。KernelArgs 保存 ordered Buffer native handles + scalar blob；
// CUDA 降为 driver void**，Vulkan 降为 descriptor + push constants。ASYNC 提交即返回，
// completion 经 compute-03 co_await StreamComplete 观察。
Status launch(Stream&, const Kernel&, const LaunchConfig&, const KernelArgs&) noexcept;

// grid/block helper (ggml 到处手写 (n+bs-1)/bs, HYLE 提一次).
constexpr unsigned blocks_for(std::size_t n, unsigned block_size) noexcept
    { return static_cast<unsigned>((n + block_size - 1) / block_size); }
```

**in-process 编译缓存**：`compile()` 内部按 `XXH3(source) ^ arch` 查 SwissTable 缓存 PTX，命中跳过 NVRTC（同 kernel 源码重复编译是热点）。缓存 payload(PTX 字符串)，load 仍每 Kernel 独立（CUmodule 不跨 Kernel 共享，简单）。

**手动虚表分派**（同全 compute）：vtable 加 `kernel_load_ptx / kernel_destroy / kernel_launch`。CUDA 后端 driver API 实现；host 后端 kernel = 一个 host 函数指针，launch = 直接调用（CPU fallback，验证抽象跨后端 + 无 GPU 可测）。

## 参考 ggml 的启动工程细节（用户"参考 ggml 调研"）

ggml 的 kernel 是编译期 `<<<>>>`，但它的**启动工程**是 SOTA，全盘参考进 compute-04 的 CUDA 后端：
- **grid/block 计算**：`num_blocks = (n + BS - 1) / BS`（clamp.cu:20 等到处手写）→ HYLE 提为公开 `blocks_for()` helper（ggml 重复手写是缺陷，HYLE 修正）。
- **超大 n 兜底**：grid 维度有上限（`MAX_GRIDDIM_*`），超限时 kernel 内 grid-stride loop 兜住（im2col.cu:22/56）——HYLE launch 抽象文档化此约束，helper 可选 clamp 到上限。
- **`__launch_bounds__`**：写 `__global__ void __launch_bounds__(BS) k(...)`（concat.cu:7）——这是 kernel 源码侧的事（HYLE 不写 kernel，但文档提示写 kernel 的消费者用）；运行时无源码时用 `cuOccupancyMaxPotentialBlockSize` 动态求 block（compute-04 可选提供）。
- **launch 后强制 `cudaGetLastError`**（common.cuh:1638 `ggml_cuda_kernel_launch` 封装）→ HYLE launch 后必查，fail-fast（`HYLE_FATAL_ASSERT`/Status），别像 ggml clamp/fill 漏查。
- **参数平铺**：ggml kernel 收裸 device address + 标量，不 struct 打包（clamp.cu:8）。HYLE typed `KernelArgs` 平铺 `Buffer::native_handle()` + 标量 blob：CUDA backend 解读为 driver ABI device address，Vulkan backend 解读为 `VkMem*` 并提取 descriptor；CPU mapped pointer 另由 `host_data()` 暴露，不进入 kernel ABI。
- **DevicePool 消费者到位**（compute-02c）：kernel 的中间/临时 device buffer（scratch）用 `DevicePool` 借还（ggml `ggml_cuda_pool_alloc` 同款）——**compute-02c 的真正消费者就是 compute-04 的 kernel scratch**。

## 明确不做（防蔓延，登记后续；B 档已含 nvrtc 运行时编译）

| 不做 | 去向 | 为什么 |
|---|---|---|
| **nvJitLink / LTO-IR → CUBIN** | 后续加档 | CCCL 用它链接多 TU + 出 SASS 省驱动 JIT；HYLE 单 kernel PTX + 驱动 JIT 一次足够，nvJitLink 是优化加档（登记，for.cu/nvjitlink.h 已挖清） |
| **Library API**（cuLibraryLoadData/GetKernel）| 后续 | CCCL 生产用的新 API；基础版经典 `cuModuleLoadData`/`cuModuleGetFunction` 足够 |
| **AOT 编译结果序列化落盘** | 后续/按需 | CCCL 的 `_serialize`/`_deserialize` 跨进程缓存；HYLE 先做 in-process 缓存（同进程重复编译才是热点） |
| **occupancy 动态 block**（cuOccupancyMaxPotentialBlockSize）| 可选 | 调用方给 LaunchConfig；无源码 launch_bounds 时才需 |
| **SPIR-V / Vulkan kernel** | compute-06 | 同一 `launch`/`Kernel::compile` 抽象的 Vulkan 实现（SPIR-V module + entry + dispatch）；compute-04 立抽象 + CUDA 完整实现 |
| **HYLE 自写算子 kernel** | 不做（scope）| HYLE 不写算子（借 ggml，compute-05）；compute-04 立 launch+编译抽象，kernel 源码由消费者/测试提供 |
| **多 stream fork/join 编排 API** | 后续 | compute-03 已定 GPU 依赖下沉 cudaStreamWaitEvent；编排 API 待消费者 |

## 交付物（实际落地）
- `include/hyle/compute/kernel.h`（`Kernel{compile/load_ptx/load_host/HostFn}` + `launch()` + `blocks_for()`）；`Dim3`/`LaunchConfig` 落 types.h（POD，让 CUDA TU 不碰 std::expected）
- `src/compute/backend.h`（vtable +5：kernel_compile/load_ptx/load_host/destroy/launch）
- `src/compute/compute.cc`（Kernel 分派 + `launch` 自由函数〔friend Stream 取 native〕）
- `src/compute/cuda_backend.cu`（NVRTC 链：CreateProgram/AddNameExpression/CompileProgram/GetProgramLog/GetLoweredName/GetPTX + arch 探测 sm_XX + FNV-1a 编译缓存〔mutex+unordered_map，冷路径〕+ cuModuleLoadData/GetFunction + cuLaunchKernel + cudaGetLastError；**cudaFree(0) 强制 primary context** 见纪要）
- `src/compute/host_backend.cc`（compile/load_ptx→Unimplemented，load_host 存 HostFn，launch 直调）
- CMake：CUDA 时链 `CUDA::nvrtc`
- `tests/compute/kernel_test.cc`（7 测试：NVRTC vecadd 端到端 + load_ptx + 缓存 + 编译错误 + DevicePool scratch + blocks_for + host fallback + backend mismatch）

**cloc 实测**：~464 生产（kernel.h 113 + backend.h 24 + compute.cc 82 + cuda 184 + host 40 + stream/types 微改）+ 175 测试。cuda 184 = NVRTC 链 + module load + launch 分量。

## 实测结果（2026-07-06，三平台真机）
- **WSL（GCC14+nvcc）**：compute 全量 37/37 + 1 skip；kernel 6/6——**NvrtcVecAddEndToEnd 在 RTX 4070 真跑**：NVRTC 编译 vecadd CUDA C 源码→PTX→cuModuleLoadData→cuLaunchKernel→显存校验（compute 01~04 device/buffer/pool/kernel + 运行时编译合体）；CompileCacheHit / CompileErrorReturnsStatus（NVRTC 日志转 Status）/ DevicePoolScratch / blocks_for 全绿。gpu_await 另 2/2。
- **WSL TSan（host-only）**：kernel host 路径 3/3 + 4 GPU 自跳过，零 race。
- **Windows（MSVC19.50+nvcc）**：全量 37/37 + kernel NVRTC 真编真跑。
- **macOS（Apple clang arm64,无 CUDA）**：33/38（GPU 全 skip）；**HostFallbackLaunch 真跑**（load_host 注册 host vecadd fn，launch 直调，无 GPU 验证抽象）+ blocks_for + BackendMismatch。

## 实现纪要 / 对抗自审
- **driver context 初始化坑（真机揪出）**：首跑 `cuModuleLoadData` 返 `CUDA_ERROR_INVALID_CONTEXT`——runtime `cudaSetDevice` 惰性建 primary context，未必绑到当前线程，driver API 拿不到 current context。修：load_module 里 `cudaFree(nullptr)`（CUDA 惯用"强制 primary context 实例化"）后再 driver 调用。ggml/CCCL 在已有 current context 下跑，HYLE kernel load 是首个 driver-API 入口需此 nudge。
- **types.h 放 Dim3/LaunchConfig**（非 kernel.h）：CUDA backend TU 要 cast vtable 的 config 但不能碰 kernel.h（std::expected）——同 EventStatus/BufferKind 的 result.h-free 隔离。
- **编译缓存用 FNV-1a + std::unordered_map**（非 XXH3/SwissTable）：避免 SIMD 头进 nvcc TU；内部冷路径（仅 compile 时查），非热路径生产容器，可接受 std。
- **name mangling**：`nvrtcAddNameExpression`+`GetLoweredName`（照 CCCL）解 C++ kernel；extern "C" 走 verbatim 名。
- **launch 走 driver `cuLaunchKernel`（真"任意"）**：`void** args` 容纳任意签名，非编译期 `<<<>>>`；与 compute-06 SPIR-V dispatch 同构。launch 后 `cudaGetLastError` fail-fast（ggml 契约）。
- **异步生命周期**：`KernelArgs` 数组只需活过 submit 调用；Kernel 与所有 Buffer allocation 必须活到 Stream/Event completion。同步 Buffer copy/host access 不是 completion primitive，不能与未完成 producer kernel 竞态。
- **DevicePool 消费者到位**：kernel scratch 用 compute-02c 的 PoolAlloc（LIFO），02c 的真实消费者兑现。
- **NVRTC 编译端到端**（三平台 GPU）：CUDA C 源码字符串（`extern "C" __global__ void vecadd(float* a, float* b, float* c, int n)`）→ `Kernel::compile` NVRTC 编 PTX → 加载 → compute-02 buffer 喂输入 → `launch` → `co_await StreamComplete`（compute-03）→ 读回校验。**device/buffer/pool/completion/kernel + 运行时编译全链路真跑**（compute 01~04 合体）。
- **给定 PTX 路径**：`load_ptx` 加载一段手写/预编 PTX → launch → 校验（不走 NVRTC 的入口）。
- **编译缓存**：同 source 第二次 `compile` 命中缓存不重编（计时/计数验证）。
- **name mangling**：一个 C++ 模板/非 extern-C kernel 经 name expression 正确 launch。
- **编译错误**：坏源码 → `nvrtcGetProgramLog` 有内容 + 返 Status 错误（不崩）。
- **DevicePool scratch**：kernel 用 DevicePool 借中间 buffer（compute-02c 消费者验证）。
- **blocks_for** 单元验证；launch 后错误检查（坏 config → Status/fail-fast）。
- **host fallback**：kernel=host fn-ptr（compile/load_ptx 在 host 后端映射为注册 fn），launch 直调，无 GPU 可测抽象。
- 三平台（Win/WSL GPU NVRTC 真编真跑 + mac host fn-ptr）；WSL TSan（编译缓存并发 + launch 路径）；cloc 验真；对抗自审对照 ggml 启动工程 + CCCL nvrtc 链。

## 交付物（预期，B 档）
- `include/hyle/compute/kernel.h`（`Kernel{compile/load_ptx}` + `LaunchConfig` + `Dim3` + `launch()` + `blocks_for()`）
- backend vtable + fn-ptr（kernel_compile / kernel_load_ptx / kernel_destroy / kernel_launch）
- `src/compute/cuda_backend.cu`（NVRTC 链：`nvrtcCreateProgram`/`AddNameExpression`/`CompileProgram`/`GetProgramLog`/`GetLoweredName`/`GetPTX` + arch 探测 + XXH3 编译缓存 + driver `cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel` + launch 后 cudaGetLastError）
- `src/compute/host_backend.cc`（kernel = 注册的 host fn-ptr，launch 直调）
- CMake：CUDA 时链 `CUDA::nvrtc`（新增，双平台库已确认可用）
- `tests/compute/kernel_test.cc`（NVRTC vecadd 端到端 + load_ptx + 缓存命中 + name mangling + 编译错误 + DevicePool scratch + blocks_for + host fallback）

## 参考
- compute-design §2（kernel = ggml 净新缺口）、「模块形状」kernel.h、§待深挖 #1（kernel 表示统一 CUDA+SPIR-V）。
- **kernel 启动工程（ggml）**：clamp.cu/fill.cu/scale.cu（`<<<>>>` grid/block）、im2col.cu:22/56（多维 grid + stride loop）、concat.cu:7（`__launch_bounds__`）、common.cuh:1638（`ggml_cuda_kernel_launch` + cudaGetLastError）。
- **运行时编译链（CCCL）**：`c/parallel/src/for.cu`（最简完整链 compile/load/launch/serialize）、`nvrtc/command_list.h`（fluent 编译 + name expression `nvrtcAddNameExpression`/`GetLoweredName`）、`libcudacxx/test/utils/nvidia/nvrtc/nvrtcc_build.h`/`nvrtcc_run.h`（经典 NVRTC→PTX→cuModuleLoadData→launch 教学版）、`util/errors.cpp`（check 转异常/Status）。
- CUDA driver：`cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel`；NVRTC：`nvrtc.h`（双平台库确认可用 libnvrtc.so.13.3 / nvrtc.lib）。compute-01/02/02c/03 前置。compute-06（同抽象的 Vulkan SPIR-V 实现）。
