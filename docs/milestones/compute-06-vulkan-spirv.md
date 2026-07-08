# compute-06：Vulkan compute —— HYLE 自造 SPIR-V kernel（跨厂商通用 GPU 计算）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-01（Device/Stream/Event 抽象 + event-poll completion 原语）、compute-02（Buffer）、compute-03（GPU completion→fiber，GpuPoller `co_await stream`）、compute-04（kernel launch 抽象：`Kernel::compile`/`launch`，CUDA NVRTC 侧已立）、sched（GpuPoller 挂 Waker） |
| **状态** | ✅ **完成（2026-07-08，三平台真机 + 红队闭环）**。HYLE 自己的跨厂商 GPU 计算路径落地：自造 SPIR-V kernel 经 Vulkan compute 跑上 GPU。`src/compute/vulkan_backend.cc`（volk + 手动虚表，vk 类型全封 .cc）+ 统一 launch ABI（KernelArgs：buffer 列表 + 标量 blob，CUDA/Vulkan 各降级）+ Device::backend()。**测试 3 个 · 三平台真机全绿**：Win RTX4070 3/3 + WSL RTX4070 3/3 + WSL TSan 干净 + **mac Apple M4 via MoltenVK 3/3**：SaxpyEndToEnd（buffer+push 端到端）+ BatchedSaxpy（#15 合批：3 dispatch 一 cmdbuf 一 submit）+ CoAwaitCompletion（#21 异步：fiber co_await stream，marker-fence + GpuPoller 唤醒）。**⚠️ mac 真跑抓出真 bug（探针实测坐实，非猜）**：现状 instance 创建在 MoltenVK 上 `vkCreateInstance = VK_ERROR_INCOMPATIBLE_DRIVER(-9)` → 直接 SKIP。根因 = MoltenVK 是 portability 驱动，须开 `VK_KHR_portability_enumeration`(instance 扩展+flag) + `VK_KHR_portability_subset`(device 扩展，spec 强制)。**修：运行时探测扩展、有才启用**——一份代码通吃 mac(advertise→启用) + Win/WSL discrete GPU(不 advertise→不请求，Win 回归 3/3 确认无影响)。**核对表落实**：#1 共享 pipeline cache、#6 ReBAR + #9 持久映射、#11 barrier、#14/#17/#18 descriptor、#15 合批、#20 workgroup=grid、#21 fence→GpuPoller 异步 completion。**⚠️ 实现期两个 Windows-only 崩溃（WSL 不崩、Windows 0xC0000005，跨平台测试才抓到）**：①合批多 InFlight 共享 fence 双重释放；②co_await 的 marker cb 挂 event 借用 fence 致 event 析构后 UAF——均修（一 InFlight 拥全部 / marker cb 靠 pool 销毁回收）。**红队审计修 3 项**：VkPipelineCache 无锁并发（spec external-sync 违规）+ lazy pipeline build 竞争 → build_mtx 锁 + 双检；合批 descriptor pool 失败静默丢 dispatch → 改 fail-fast（项目纪律）；KernelArgs 文档"synchronous"口径错 → 改"buffer 须活到 completion 非 launch 返回"。**登记待办**（红队 E/G，非阻塞）：event marker cb 长命 stream 累积、build 部分失败无回滚。（红队 D WAW barrier 未覆盖 SHADER_WRITE 已 2026-07-08 修复=收尾②-b，见下）**多队列/async compute 的 transfer 重叠/小 kernel 工况仍待驱动实测**（核对表 #23）。**收尾①（2026-07-08，补原语正确性硬伤）**：初版 buffer set/get/copy 遇非 host-mapped（非 ReBAR）内存直接返 Unimplemented——"通用 Vulkan 后端"实际只 ReBAR 卡可跑。补 **staging 双缓冲**（kompute #8/#13）：非 ReBAR device 内存自建 host-visible+coherent staging + 持久映射，set=host→staging→copyBuffer→primary、get=copyBuffer primary→staging（shaderWrite→transferRead / transferWrite→hostRead barrier 链）→host、copy=D2D on-device、clear=vkCmdFillBuffer；同步经一次性 cmdbuf staged_submit。**普通独显/集显/老卡现可用**。TEST-ONLY env `HYLE_VK_FORCE_STAGING`（HYLE_TESTING 门控）在 ReBAR 开发机强制走 staging，CTest 变体 `hyle_compute_vulkan_staging_test` 自动 CI 覆盖。双平台常规6+staging3全绿 + WSL TSan(staging)净。**收尾②-a（compute-06b，2026-07-08，#25 GPU timing）**：核对表 #25 timestamp query 落地——GPU 侧计时是"证 SOTA"前置（benchmark 精确耗时、自适应 kernel 选择、性能回归监控），本身是一等 profiling 能力（非 debug-only、不进 HYLE_TESTING 门）。手动虚表两层：新增 6 个 timing 槽（`event_create_timing`/`event_destroy_timing`/`event_record_timing`/`event_query_timing`/`event_sync_timing`/`event_elapsed_ns`），与 plain 完成事件**分开**（CUDA=timing cudaEvent 复用同 impl；Vulkan plain=裸 VkFence、timing=`{VkFence,VkQueryPool}` 结构，布局不同不可共用 destroy/query/sync 槽，公开层按 `Event::timing_` 路由）。公开 API：`Event::create_timing(device)` + 自由函数 `elapsed_ns(start, stop)`→ns（CUDA cudaEventElapsedTime ms→ns、Vulkan `getQueryPoolResults × timestampPeriod`、host inline=0）。测试 +2（`HostTimingInlineIsZero` 任意平台跑 inline=0 契约；`GpuTimingMeasuresElapsed` 真机 RTX4070 spin kernel 夹两 timing event 读回 ns>0）。**收尾②-b（compute-06b，2026-07-08，#11/#12 barrier 精确化）**：批量 launch 原对每 dispatch 无条件插 barrier（kompute 低效点=过度串行化，独立 kernel 无法重叠）。改依赖驱动——`pending`=上次 barrier 以来触碰的 buffer 集，仅当某 item 触碰前序未 barrier item 的 buffer（真 RAW/WAW/WAR 冒险）才在其前插 barrier 并清 pending（一次 barrier 解所有前序冒险），不相交 dispatch 跳过让 GPU 重叠；终止 barrier 提出循环外只插一次（仅最终可见性需 host/transfer scope）。**正确性证明**：buffer 入 pending 后仅 barrier 能移除，故两 dispatch 共享 buffer 必有 barrier 介于其间（无漏 barrier）；barrier 为 global VkMemoryBarrier 覆盖全内存故保守但正确。**红队 D 修复**：单发+批量 barrier 的 dstAccessMask 原漏 `SHADER_WRITE`，后续 submit 对同 buffer 的 WAW 未排序，均补上。测试 +1 `BatchedDependentChain`（job j 读 job j-1 输出=共享冒险，barrier 必触发排序链 b0=10→b4=20 golden 校验；与 BatchedSaxpy 不相交跳过路径互补），双平台 Vulkan 5/5。**剩收尾②**（#2 spec constant/#1 cache 落盘/#14+#16 rerecord+replay/#6 HOST_CACHED，见 backlog A 类，不阻塞）。 |

> **范围**：给 HYLE 一条**自己的跨厂商通用 GPU 计算路径**——用 Vulkan compute（SPIR-V）把**任意** kernel 跑上 GPU，是 compute-04（CUDA NVRTC「任意 kernel」）的 Vulkan 孪生。**参考库 = kompute**（`C:\public\project-hyle\kompute`，纯 Vulkan 零隐藏运行时），整体照抄其 Vulkan 样板，**唯一实质改动是把阻塞 fence 等待换成非阻塞轮询接 HYLE fiber 调度器**。
>
> **一句话定位**（compute-design 发现 5/7/9 固化）：**ggml 只是"AI 推理算子多",不是通用 GPU 计算框架**——普通 CUDA/Vulkan 计算（vecadd/reduce/scan/逐元素/自定义数值核）+ ② CAD/CAE 专用核（几何/网格 stencil/BVH/SpMV）ggml 一概不给，这些**HYLE 自造**。CUDA 侧走 compute-04 NVRTC，跨厂商侧走 compute-06 SPIR-V。

## 为什么做这个 / 消费者

- **跨厂商 GPU 计算刚需**：只押 CUDA = 锁死 NVIDIA。Vulkan compute（SPIR-V）一份 shader 跨 AMD/Intel/Apple(MoltenVK)/NVIDIA 全平台跑，是 HYLE"通用、不特化"定位的落地。
- **ggml 结构上给不了（发现 9，grep 实证）**：ggml GPU 后端只有编译期固定的 ~90 个 AI 算子枚举，`GGML_OP_CUSTOM` 在 CUDA/Vulkan 后端零实现——想在 GPU 上跑一个 ggml 没有的算子（哪怕只是 vecadd），结构上做不到。compute-06 填的是"一切非 AI 的 GPU 计算"这个大类。
- **主消费者 = 北极星② CAD/BIM/CAE**：几何核/网格 stencil/BVH 遍历/CAE 稀疏求解 SpMV，全是 ggml 没有、必须自造的 kernel。北极星① 推理的 AI 算子 ggml 全有、不需 compute-06。
- **与 ggml-vulkan 不重复（发现 5）**：ggml-vulkan 给"AI 算子的跨厂商实现"（借，走 compute-05 桥）；compute-06 给"HYLE 自定义 SPIR-V kernel 的跨厂商实现"（造）。两者并存。

## 调研实测（2026-07-07，RTX 4070 真跑，是本 spec 所有决策的依据）

> **纪律**：compute-05 的血泪是"没实测就把'零拷贝注入'写进 spec、gdb 才发现方向错"。compute-06 的每条设计都先真跑坐实再写。判据不是"能跑",是"够不够快"（HYLE = 超高性能库）。

### 实测 1：volk 是正确的 loader 方案（不用 SDK / 不用 find_package(Vulkan)）

`find_package(Vulkan)` 在本机**失败**——只有驱动带的运行时 `vulkan-1.dll`，无 `vulkan-1.lib` 导入库（那随 LunarG SDK，未装）。**volk**（zeux/volk，业界标准 Vulkan meta-loader，**非 glad 式代码生成器，是固定的 2 文件库**）`volkInitialize()` 运行时 `LoadLibraryA("vulkan-1.dll")` + `vkGetInstanceProcAddr` 解析全部 vk 函数——**零 .lib / 零 SDK / driver 解耦**，且 device-level 函数解析跳过 loader trampoline（每调用少一跳，更快）。基线已真跑坐实。

### 实测 2：getFenceStatus = HYLE Event::query() 的非阻塞语义（难点3清）

- 未提交的 fence → `VK_NOT_READY`（非 UB，可安全轮询）；提交后非阻塞轮询到 `VK_SUCCESS`。**与 compute-03 GpuPoller 现用的 `cudaEventQuery` 语义同构**——`vk::Fence` 就是 Vulkan 后端 Event 的 native handle，可直接填 `Event::query()` 虚表。
- ⚠️ 但**裸 busy-poll 烧 CPU**（1M vecadd 死等 4355 次轮询）。GpuPoller **必须退让**（后台线程轮询 + fiber 让出，同 io poller），不能单线程 busy-spin。

### 实测 3：性能杠杆 = 合批，不是并发提交/多队列（难点1，颠覆直觉，分层结论）

重 kernel（256K×20000 iter，单 dispatch 纯 GPU 0.916ms，timestamp query 隔离 GPU 时间），8 job 四种提交模式：

| 模式 | 总时间 | speedup | 结论 |
|---|---|---|---|
| 串行 submit+wait×8 | 7.62ms | 1.00x | 基准；提交开销很小 |
| 并发单队列 submit-all→poll all | 7.42ms | 1.03x | 几乎无提升 |
| 并发 8 队列 round-robin | 8.97ms | **0.85x（更慢）** | 队列调度开销 > 收益 |
| **合批 1-cmdbuf 1-submit** | 7.10ms | **1.07x（最快）** | 省 N 次 submit+fence |

- **⚠️ 结论有适用范围（用户提醒 + 权威资料校正，勿犯"单工况当普世"）**：上表只测了**一种工况——同构 + 各自占满 GPU 的纯 compute job**，这恰是多队列/async compute **最不利**的工况（GPU 的 SM 已被单个大 dispatch 占满，多提交只增调度开销）。**不能据此否定多队列本身。**
- **已坐实（此工况）**：① 单 GPU 一次只真跑一个占满型 dispatch，8 job 总时间≈8×单个；② 合批是正收益（1.07x），**与 compute-05 发现 4（llama.cpp 真并发 = 合批）在 Vulkan 上再次坐实**——确立为**主性能杠杆**。
- **未测、不能否定（多队列/async compute 真有用的工况，资料坐实，留驱动）**：① **compute + transfer 重叠**（一边 H2D 拷下一批、一边算当前批，NVIDIA async compute 核心用例，对 HYLE 流式推理/CAE 数据很可能有用）；② **多个填不满 GPU 的小 kernel 并排**（提 SM 利用率）；③ 异步优先级。能否真并行还高度依赖硬件（kompute 文档 NVIDIA 1650 例：仅 1 graphics+1 compute family 并发）。

### 实测 4：descriptor update 便宜到可忽略（难点2清）

`vkUpdateDescriptorSets`(3 buffer) **0.035µs/次**。kompute 的 maxSets=1 限制不是性能问题，**HYLE 每次 dispatch 重新 update descriptor set 即可**，不用为复用做复杂设计。

### 硬件观察

RTX 4070 暴露 family 0（16 queue，graphics+compute+transfer）+ family 2（8 queue，**dedicated async compute**，无 graphics）。async compute 的物理基础在（留给 compute+transfer 重叠工况调研）。

## 核心抉择（基于上述实测）

### ① loader = volk（FetchContent，运行时 dlopen，无 SDK）

- Vulkan **loader** 靠 volk 运行时 `LoadLibraryA/dlopen` 系统驱动带的 `vulkan-1.dll`/`libvulkan.so`——HYLE 不 vendor 驱动、不要求装 SDK、不碰 slate 旧头。
- Vulkan **headers** = FetchContent `Vulkan-Headers` pin（版本 HYLE 自己钉，非"机器上碰巧有的 SDK"）。
- 全 `#define VK_NO_PROTOTYPES` + volk 动态 dispatch（同 kompute 的 `VULKAN_HPP_DEFAULT_DISPATCHER`）。

### ② 性能杠杆 = 合批（实测 3 坐实），多队列 = 待工况驱动的可选优化

- **主路径 = 合批**：多个 dispatch 录进一个 command buffer、一次 submit、一个 fence。后端 API 该像 ggml_bridge 提供**合批入口**（N 个 kernel 一次提交），而非"N 个独立 stream"。默认单 compute 队列。
- **多队列 = opt-in 高级路径,不锁死不否决**（§0.2）：照 kompute 模型（调用方决定用几个队列，默认单队列）。**登记为开放项**——等真遇到 compute+transfer 重叠 / 小 kernel 填不满 GPU 的工况,再补测再定（附实测 3 的已测数据 + 未测清单为据）。**首版不做多队列。**

### ③ completion = 非阻塞 fence 轮询接 GpuPoller（实测 2，与 compute-03 同构）

- `co_await stream` 让 fiber 挂起，GpuPoller 后台线程 `vkGetFenceStatus` 非阻塞轮询，`VK_SUCCESS` 时 `Waker::wake()`——**与 compute-03/04 的 CUDA `cudaEventQuery` 路径同构**（`vk::Fence` = Vulkan Event 的 native handle 填 `Event::query()` 虚表）。
- **GpuPoller 必须退让**（实测 2：裸 busy-poll 烧 CPU）——复用 compute-03 的 poller 机制，不单线程 busy-spin。
- **对比 compute-05 借 ggml 走 spawn_blocking**：那是因 ggml stream 私有、只能阻塞 synchronize；compute-06 是 **HYLE 自己的 stream/fence**，能走 GpuPoller 非阻塞轮询（§4 愿景在自造 kernel 路径成立，同 compute-03/04）。

### ④ shader 编译 = 预编 SPIR-V + 提供工具文件夹（用户定，性能好）

- **预编**：构建期用 glslang（FetchContent 产出的 `glslang` validator）把 `.comp`→`.spv`，产物 embed/放 kernel 目录。运行时零编译开销。
- **工具文件夹**：HYLE 构建产出 `glslang` validator 工具 + 文档，给 HYLE 开发者和客户编自己的 `.comp` 用（是构建产物,非手放路径）。
- **运行时编译**（glslang 的 `glslang_c_interface.h`）留作后续（CAD/CAE 动态生成 kernel 参数时），同一 glslang FetchContent 依赖需要时开 ENABLE 即可,不额外加库。

### ⑤ 后端填 compute-01/04 的手动虚表（不引入 kompute 类层次）

- Vulkan 后端填 compute-01 `ComputeBackend` 的手动函数指针虚表（device/stream/event/buffer/kernel），**不引入 kompute 的 Manager/Sequence/Algorithm/Tensor 类层次、不引入 OpBase 虚基类**（发现:HYLE 是手动虚表非虚继承）。
- Vulkan 类型（`vk::*`）全封 `.cc`（同 ggml_bridge.cc），公开头零 Vulkan 类型泄漏。
- kompute 深读的抄/改/丢清单见下「kompute 映射」节。

### ⑥ CMake 集成（已写进根 CMakeLists，实测配置通过）

- `HYLE_COMPUTE_VULKAN` 门控（仿 `HYLE_COMPUTE_CUDA`/`HYLE_COMPUTE_GGML`），三个 FetchContent 全 pin `vulkan-sdk-1.4.341.0`（= 本机 loader 版本，零 skew）：Vulkan-Headers（`b5c8f996`）、volk（`d979819f`）、glslang（`f0bd0257`）。
- glslang 关 `ENABLE_OPT`（免 SPIRV-Tools）/`ENABLE_HLSL`/`GLSLANG_TESTS`/`GLSLANG_ENABLE_INSTALL`/`BUILD_EXTERNAL`，开 `ENABLE_SPIRV`/`ENABLE_GLSLANG_BINARIES`（产 validator 工具）。
- 继承全局 `FETCHCONTENT_BASE_DIR=.deps`（cil2cpp 技法）→ 删 build 不重拉、每机独立树，同 ggml/gtest。
- **不用 find_package(Vulkan)**（本机无 .lib）、**不用 LunarG SDK**（每机手动装步骤,非自动可重建）、**不用 slate 旧头**（v270 陈旧）。

## kompute 映射：抄什么 / 改什么 / 丢什么（深读源码,2026-07-07）

> 铁证:全 `src/` 唯一阻塞点是 `Sequence.cpp:172` 一行 `waitForFences`，零 `std::thread`/`condition_variable`。kompute 是理想参考,唯一实质改动就那一行。

**生命周期骨架**（谁 own 谁）：
```
Manager  ─ vk::Instance / PhysicalDevice / Device / Queue
  ├ Sequence  ─ vk::Fence + CommandPool + CommandBuffer   ← completion 核心
  ├ Algorithm ─ ShaderModule / DescriptorSet / Pipeline    ← kernel 本体(→ compute-04 Kernel)
  └ Tensor    ─ primary(device-local) + staging(host-visible) buffer  ← (→ compute-02 Buffer)
```

**该抄**（file:符号 → HYLE 对应物）：
- `Manager.cpp:174-311/336-499`（instance/device 创建，含 `__APPLE__` portability 分支——MoltenVK 保留）→ Vulkan 后端 device vtable。
- `Sequence.cpp:317-356`（command pool/buffer，`eResetCommandBuffer`）+ `:130-152`（submit）→ Stream submit。
- `Tensor.cpp:369-438`（buffer 创建 + memory-type 手动扫描）+ `Memory.cpp:224-258`（property flags）→ Buffer 后端虚表。
- `Algorithm.cpp:128-380`（DescriptorPool/Set + ShaderModule + Pipeline + push-constant + dispatch）→ compute-04 Kernel 的 Vulkan 实现。
- barrier 编排：`OpAlgoDispatch.cpp:22-45`、`OpSyncLocal.cpp:26-52`、`Tensor.cpp:226-251`（barrier 模板）→ 传输/dispatch 的固定 barrier 序列（借语义,不抄 Op 类）。

**该改**（kompute 做法 → 违反哪条 HYLE 铁律 → 改成）：
- `waitForFences`（`Sequence.cpp:172` 阻塞）→ 不引入第二调度器 → `vkGetFenceStatus` 非阻塞轮询接 GpuPoller。
- 全体 `shared_ptr` + Manager 的 weak_ptr 列表 → 不用 shared_ptr 跨边界 → 侵入式 `Ref<T>`/明确所有权。
- 公开头暴露 `vk::*` → 公开头零第三方类型 → vk 类型全封 `.cc`（同 ggml_bridge）。
- 到处 `throw std::runtime_error`（含调用方契约违反）→ 契约优先 fail-fast → 调用方 bug 用 `HYLE_FATAL_ASSERT`；仅真外部失败（device 创建失败/扩展缺失）返错误。
- `mIsRunning` begin-end 录制状态机 → 简化为按需一次录制。

**该丢**（独立库包袱）：`OpBase` 虚基类 + `operations/*` op 对象化（HYLE 手动虚表）、spdlog/fmt logger + debug callback、`Image` 类（compute 只需 buffer）、timestampQueryPool（除非要 profiling）、Python 绑定、可选 submitMutex（fiber 调度器保证串行）。

## kompute 逐优化点抄写核对表（深读源码 2026-07-07，写代码时逐行打勾，防漏抄/防抄进低效）

> **用法（用户定「边写边抄，抄需要抄全」+ io 半途而废血泪）**：实现 Vulkan 后端时对着此表逐条做，完成打 ✅。**关键认知：kompute 本身有明确低效点（空 pipeline cache / 无条件 barrier / 逐个 update / workgroup 不 clamp）——「抄全」= 抄全语义 + 修掉这些低效，不是照抄。** 收益判据 = HYLE 超高性能目标。

| # | 优化点 | kompute file:line | HYLE 决策 | 收益/风险 |
|---|---|---|---|---|
| 1 | **PipelineCache** | `Algorithm.cpp:302-307` | **改**：kompute 每 algorithm 建**空** cache = 没优化。HYLE 建 Manager 级共享 cache + 落盘（`getPipelineCacheData` 序列化，带 device UUID 校验，换卡作废） | 省冷启动重编 SPIR-V→ISA（几十 ms/kernel）|
| 2 | **Specialization constants** | `Algorithm.cpp:268-294` | **抄全**：每常量 `SpecializationMapEntry(id,offset,size)`→`SpecializationInfo`→stage。把 local_size/循环上界/tile 编进 shader | 驱动编译期死代码消除+循环展开,比 push constant 快 |
| 3 | spec constant 改需 rebuild | `Algorithm.hpp:88-142` | 抄语义（spec 定死不可运行期改）,配 #1 cache 降 rebuild 成本 | 误当运行期可变会踩坑 |
| 4 | Push constants 布局 | `Algorithm.cpp:252-261` | **抄全**。**本机 `maxPushConstantsSize=256`**（非 kompute 注释的 128）→ 上限 `HYLE_FATAL_ASSERT` | 传标量零开销免 descriptor |
| 5 | push 只在 record 时写 | `OpAlgoDispatch.cpp:47-52` | 抄；知道改 push 值=要重录 cb（影响 #16 replay）| 循环改参数时权衡重录 |
| 6 | **Memory 4 分支 + ReBAR** | `Memory.cpp:224-258` | **抄全 4 类**。**本机确认有 DEVICE_LOCAL+HOST_VISIBLE+COHERENT（ReBAR）memory type**（vulkaninfo 实证）→ `eDeviceAndHost` 免 staging 拷贝直写显存 | ReBAR 大红利;缺 HOST_CACHED(回读密集)可补 |
| 7 | memory type 首拟合 + **fallback** | `Tensor.cpp:409-420` | 抄首拟合,**补降级**：eDeviceAndHost 找不到→退 eDevice（kompute 直接 throw 无降级）| ReBAR 不可用的卡 |
| 8 | staging 双缓冲**仅 eDevice** | `Tensor.cpp:352-364` | 抄：primary 本身 host-visible 时 staging 纯浪费 | 正确的"何时需 staging"判据 |
| 9 | **持久映射(map once)** | `Memory.cpp:157-183` | **抄**：映射保持到 destroy,不每次 map/unmap | 省每次 map 开销 |
| 10 | coherent 免 flush | `Memory.cpp:177-179` vs `:208` | **统一**：coherent 内存删冗余 flush（kompute unmap 处仍 flush,自相矛盾/防御）;non-coherent(HOST_CACHED) 才显式 flush/invalidate | 存疑点,按 coherent 语义清理 |
| 11 | **dispatch 前 barrier 无条件** | `OpAlgoDispatch.cpp:22-45` | **改**：kompute 对每个 mem obj 无条件插 barrier=过度串行化。HYLE 按依赖精确插（前驱确有 transfer 写才插）| 热路径固定开销,SOTA 关键 |
| 12 | buffer barrier(非 global) + **合并** | `Tensor.cpp:226-251` | 抄 buffer barrier（比 global 精确）,**多 buffer 合并成一次 `pipelineBarrier` 多 barrier**（非逐个）| 减 barrier 调用 |
| 13 | OpSyncLocal 三 barrier 链 | `OpSyncLocal.cpp:31-51` | **抄全**（回读正确性:shaderWrite→transferRead→copy→transferWrite→hostRead）| 漏 host barrier→读脏数据 |
| 14 | command pool eResetCommandBuffer | `Sequence.cpp:328-329` | 抄:非 ONE_TIME_SUBMIT→cb 可 replay | 见 #16 |
| 15 | **一 cb 一次 submit(合批核心)** | `Sequence.cpp:294-315,126-138` | **抄全——主性能杠杆**（实测 1.07x）:多 dispatch 录进单 cb 一次 submit | 省 N 次 submit+fence |
| 16 | rerecord/replay | `Sequence.cpp:209-217` | 抄:固定 pipeline 反复跑=record 一次多次 submit（除非 #5 改 push）| 省重录 CPU |
| 17 | descriptor pool sizing maxSets | `Algorithm.cpp:160-170` | 抄 sizing;知道无 FREE_DESCRIPTOR_SET_BIT=整池销毁 | 简化生命周期 |
| 18 | **updateDescriptorSets 逐个** | `Algorithm.cpp:205-216` | **改**：kompute for 循环逐个 update=低效。HYLE 收集所有 write **一次性 update** | 省驱动调用 |
| 19 | push descriptor(VK_KHR) | 全库无 | **可选**：`VK_KHR_push_descriptor`（本机支持,rev2）免 set 分配/update,录制时直接推。实测 update 便宜故收益有限,登记备选 | 每 dispatch 换 buffer 时 |
| 20 | **workgroup = group count 非 local_size** | `setWorkgroup:382-404` | **改（最易踩）**：kompute `Workgroup`=dispatch group 数,默认取 tensor.size() **不 clamp**。HYLE:local_size 在 shader/spec constant,`group=ceil(N/local_size)`,受 **`maxComputeWorkGroupInvocations=1024`**（本机）约束 | 照抄=local_size=1 灾难占用/超限 |
| 21 | **阻塞 waitForFences → 非阻塞轮询** | `Sequence.cpp:171` | **改（HYLE 硬约束）**：换 `getFenceStatus` 轮询接 GpuPoller,勿引第二调度器。实测 getFenceStatus 语义已坐实 | 阻塞占死 fiber worker |
| 22 | binary fence → 可升级 timeline semaphore | `Sequence.cpp:20,130-138` | 起步抄 binary fence（够用）;**登记** timeline semaphore 升级（免每序列一 fence + 支持序列间依赖计数）| 多序列依赖时 |
| 23 | EXCLUSIVE + queue family IGNORED | `Tensor.cpp:387-390` | 抄（单 compute 队列无需 ownership transfer）;**若上多队列（transfer 重叠工况）要补真 ownership transfer** | 多队列 DMA 才需 |
| 24 | 批量 createComputePipelines | `Algorithm.cpp:309-333`(TODO) | **可选**：多 kernel 一次 `createComputePipelines` 批量创建（kompute 留了 TODO 没做）| 批量建 kernel 省驱动往返 |
| 25 | timestamp query pool | `Sequence.cpp:358-404` | **抄**（验 SOTA 必需,GPU 侧计时）:writeTimestamp 前后 + `getQueryPoolResults`,`timestampPeriod` 换算 ns。本机 `timestampComputeAndGraphics=true` | benchmark 精确耗时 |

**5 个漏了后面补最痛的（day-1 就要定对）**：
1. **#20 workgroup 语义**——kompute `Workgroup`=group count 非 local_size 且不 clamp,照抄=灾难占用率/超限。起手定清"local_size 在 shader,group=ceil(N/local_size)≤limit"。
2. **#21+#22 fence 模型**——照抄阻塞 waitForFences 后期要改整条 eval/await 链接 GpuPoller,牵动 Sequence 全生命周期,极痛。起手就设计成 poll。
3. **#6 ReBAR + #9 持久映射**——免 staging 拷贝的核心红利,kompute 埋好易被当边角忽略。漏了默认走 eDevice 双缓冲,平白多一趟 copy。
4. **#11+#18 barrier/descriptor 逐个化**——kompute 明确低效点,连低效抄进热路径会拖垮 SOTA benchmark,回头拆烦。抄语义但合并调用/精确插。
5. **#1 PipelineCache 落盘/共享**——kompute 空 cache=没用,漏了每次重编 SPIR-V→ISA。做 Manager 级共享+磁盘持久化。

**存疑/待自测**：#10 coherent unmap 仍 flush（像冗余/防御,未在文档确认 kompute 意图）;#19 push descriptor 收益需 HYLE 自测;Image 路径（layout 状态机 `Image.cpp:257-389`）本表未覆盖,若做 image kernel 需另核。

## 内存模型（对接 compute-02 Buffer，别重复造）

- kompute Tensor 双缓冲仅在 `eDevice` 时建（primary=device-local storage + staging=host-visible）；host 访问靠 `mapMemory`，coherent 免 flush。
- **HYLE 已有 compute-02 Buffer（含 pinned staging pool）**。Vulkan 后端 Buffer 只需填手动虚表：`create/destroy`（抄 `createBuffer`+`allocateBindMemory`+memory-type 扫描）、H2D/D2H 录制（抄 barrier→copy→barrier 三段）。**不引入 kompute 的 Tensor/Memory 类层次。**
- **HYLE 显存自管在此完整成立**（对比 compute-05 借 ggml 让位）：自造 kernel 直接在 HYLE Buffer 上算,真零拷贝(同 compute-04 已验证)。

## SPIR-V → pipeline（对接 compute-04 Kernel）

- `Kernel::from_spirv(words)`：`ShaderModuleCreateInfo(codeSize, spirv.data())` → DescriptorSetLayout（N storage buffer binding）→ PipelineLayout（+ push-constant range）→ `createComputePipeline`。
- `launch(workgroup, buffers, push)`：barrier → `bindPipeline` → `bindDescriptorSets` → `pushConstants` → `dispatch(x,y,z)`。descriptor binding index = kernel 参数序号，push-constant = 标量参数，workgroup = grid dim。
- 与 compute-04 CUDA `Kernel::compile→PTX→cuLaunchKernel` 平级——同一 `Kernel` 抽象两后端实现（CUDA / Vulkan）。

## 线程安全 / 生命周期契约

- **backend/device 建一次持久复用**（同 compute-01/ggml_bridge），析构 free。
- **⚠️ Stream 单执行流契约（同 CUDA stream / io "一连接一 fiber"）**：一个 Stream 归**一个** fiber/线程,不可多 fiber 并发在**同一** Stream 上 launch/record——per-stream 的 command pool + 在途追踪**无内部锁**,并发用是数据竞争 + VkCommandPool external-sync 违规。并行提交请**每 fiber 一个 Stream**(不同 Stream 的 submit 在共享 VkQueue 处经 submit_mtx 串行化)。契约写进 `stream.h` 的 Stream 类 usage 注释。
- **每提交独立 command buffer + fence**（同一 Stream 内**顺序**发起的多个在途提交各自独立 cmdbuf/fence,不复用——kompute 单 Sequence 单 fence 是串行假设,HYLE 放开为多个在途）。GpuPoller 管在途 fence（首版合批 = 在途 fence 少,复杂度低）。
- Vulkan 对象创建/销毁顺序：pipeline→layout→descriptor pool→shader module→command pool→fence→buffer→device→instance（逆序毁）。

## 验收（B 档最小可信闭环，三平台）

- **自造 SPIR-V kernel 端到端**：HYLE Buffer 装输入 → `Kernel::from_spirv` + `launch` → `co_await stream`（GpuPoller 非阻塞唤醒，fiber 挂起）→ Buffer 读回 → 对拍 CPU 参考。**vecadd 首消费者**（基线已裸跑通,搬进 HYLE 后端）。
- **合批**：N 个 dispatch 一 cmdbuf 一 submit（实测的主杠杆）vs 分次,验证正确 + 性能对比登记 bench。
- **co_await 不阻塞 worker**：kernel 在途时另一 fiber 在同 worker 继续跑（同 compute-03）。
- **三平台**（全绿）：Win（RTX4070，volk+glslang）3/3 + WSL（`libvulkan-dev`，GPU 透传）3/3 + WSL TSan（GpuPoller 跨线程 wake 零 race）+ **mac（Apple M4 via MoltenVK）3/3**。MoltenVK 依赖 portability enumeration/subset 扩展——运行时探测启用（见状态行 mac 真 bug）。
- **cloc 验真** + 对照 clay coverage-map（clay 无 Vulkan compute,登记净新）。

## 明确不做（防蔓延，登记后续）

| 不做 | 去向 | 为什么 |
|---|---|---|
| **多队列 / async compute 并发** | 📋 待工况驱动调研 | 实测:同构占满型 compute 无收益(0.85x);但 compute+transfer 重叠 / 小 kernel 工况未测、不否定。首版单队列合批,遇驱动工况再补测（§0.2） |
| **运行时 GLSL→SPIR-V 编译** | 后续（glslang_c_interface）| 首版预编（性能好);CAD/CAE 动态生成 kernel 时再开,依赖已 clone |
| **ggml-vulkan 接入** | compute-05 桥（借）| 那是"借 AI 算子的 Vulkan 实现",与自造并存不重复（发现 5）|
| **Image / 纹理** | 按需 | compute 只需 storage buffer;Image 是图形/采样路径 |
| **HIP/Metal 原生后端** | compute-08 | 优先借 ggml-hip/metal;Metal 原生待需要 raw kernel 时 |
| **SPIR-V 优化器（spirv-opt）** | 后续 | ENABLE_OPT=OFF 免 SPIRV-Tools 依赖;预编阶段够用,要优化再开 |

## 参考

- **compute-design**「compute-06 kompute 映射」节 + 发现 5/7/9（自造 kernel 正当性 = 通用 GPU 计算,ggml 给不了）。
- **探路 + 三难点实测**：memory `project_compute06_vulkan_scouting`（volk 定案 / getFenceStatus 坐实 / 合批 vs 并发 vs 多队列真数据 / descriptor 便宜 / 多队列适用范围校正）。
- kompute 源码：`C:\public\project-hyle\kompute\src`（Manager/Sequence/Algorithm/Tensor/Op*，深读清单见上「kompute 映射」）。
- volk：`github.com/zeux/volk`（meta-loader，非 glad 生成器）；Vulkan-Headers/glslang：KhronosGroup。
- 前置里程碑：compute-01（device/stream/event + event-poll）、03（GpuPoller `co_await stream`）、04（Kernel launch 抽象，CUDA 侧）。compute-08（HIP/Metal 借 ggml）。
- 基线 + 实验代码：session scratchpad `vkbaseline.cc`（基线）/ `vkexp.cc`（轻 kernel 并发/descriptor/poll）/ `vkheavy.cc`（重 kernel timestamp 合批对比）——探路实验,不入 HYLE 构建。
