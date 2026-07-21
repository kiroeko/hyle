# compute-05：ggml_bridge —— 借 ggml 算子（不借运行时）

> **当前名称边界：**本文中的 host `MemoryArena` 旧称现对应 `SharedPool` / `SharedPoolCached`；旧头与旧 C ABI 已删除。当前 `PinnedStaging` 直接拥有一个 pinned host block 并 bump，不经 mem 池。

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-01（Device/Stream/Event）、compute-02（Buffer；本里程碑首消费者用 host 数据，Buffer 用于 ② CAD/CAE）、compute-02b（PinnedStaging，host 数据入口）、compute-03（spawn_blocking）、sched-11（spawn_blocking / BlockingPool / JoinHandle）、mem（Allocator） |
| **状态** | ✅ **完成（2026-07-07，三平台真机 + 5 测试全绿）**。matmul 走 llama.cpp 忠实形态（gallocr 分配 device buffer + `tensor_set`/`tensor_get` 进出 host 数据）在 RTX 4070 真跑（Win MSVC+nvcc、WSL gcc14+nvcc）、mac CPU-only（Apple clang），三平台各 **5/5 PASSED**：MatmulDefaultDevice（host→ggml→host 端到端对拍）+ CpuThreadModelSplit（n_threads 二分）+ ConcurrentIndependentBackends（6 fiber 各自 backend）+ **BatchedMatmul（合批：3 不同形状 matmul 一图一提交，finding 4）** + **BackendReuseAcrossShapes（持久 gallocr create-once/reuse-many + move 语义 + 跨调用 reserve/realloc，红队补）**。**⚠️ 实现期重大设计修正**：**"HYLE 显存零拷贝注入 ggml 算子"愿景（发现 2 初版）被两阶段真跑 + WSL gdb 彻底证伪**——CUDA 后端 `buffer_from_host_ptr=false`（`ggml-cuda.cu:5032`）结构性否决外部显存指针；dummy-buffer 注入在合批 varied 形状崩于 `ggml-cuda.cu:1924`（共享 size-0 dummy 的 context 被 gallocr 弄丢，gdb 坐实）。**定案边界：host 归 HYLE（mem 喂入/取出）、device 归 ggml（gallocr），就像 llama.cpp**——专文 [`../architecture/borrowing-ggml-memory-model.md`](../architecture/borrowing-ggml-memory-model.md)，血泪 [`../debug/ggml-zerocopy-null-buffer-hunt.md`](../debug/ggml-zerocopy-null-buffer-hunt.md)。ggml/googletest 走纯 FetchContent（cil2cpp 形式，三机各自 `.deps` 下载，无 override）。 |

> **范围**：把 ggml 当**算子库供应商**接进 HYLE——在**唯一一个 adapter 文件** `ggml_bridge.cc` 里，让 ggml 的 AI 算子（matmul/softmax/rope/conv…约 90 个）在 **HYLE 自管的显存/completion/调度** 之上运行。**借的是"算子 kernel"粒度，不是整库应用（不借 sd.cpp/llama.cpp），不 fork，不注入 ggml 运行时。**
>
> **一句话定位**（motivation 已固化）：**算子借 ggml，底座替 llama.cpp，通用 GPU 计算补 ggml 之不能（compute-04/06）**。compute-05 只做第一句。

## 为什么做这个 / 消费者

- **北极星①推理**：算子几乎全在 ggml。HYLE 对推理的价值 = 统一底座（io/sched/mem/completion），**算子借 ggml 就够**——这是 compute-05 的主消费者。
- **北极星②CAE 能借的部分**：GEMM/SpMV 里 ggml 已有的（如稠密 matmul）可借；ggml 没有的几何核/网格 stencil/BVH/稀疏求解器走 compute-04/06 自造 kernel（发现 9：ggml 结构上给不了）。
- **不是重造轮子**：ggml-backend 已把 CUDA/HIP/Metal/Vulkan/SYCL 的算子跑通（发现 5 矩阵的"借 ggml"那一列），HYLE 不重写这些算子。compute-05 = 把它接进 HYLE 的脊。

## 核心抉择（全部来自 compute-design 发现 1~10，此处只列结论 + 落地形态）

### ① 显存归属边界：host 归 HYLE / device 归 ggml（发现 2，gdb 定论；不 fork）

> **⚠️ 设计修正（2026-07-07，两阶段真跑 + WSL gdb）**：初版愿景"HYLE 显存零拷贝注入 ggml 算子"**被彻底证伪**。CUDA 后端 `buffer_from_host_ptr=false`（`ggml-cuda.cu:5032`）结构性否决外部显存指针；dummy-buffer 注入在合批 varied 形状崩于 `ggml-cuda.cu:1924`。**这不是"没找对 API"，是 ggml CUDA 的能力边界。** 详见 [`../architecture/borrowing-ggml-memory-model.md`](../architecture/borrowing-ggml-memory-model.md) + [`../debug/ggml-zerocopy-null-buffer-hunt.md`](../debug/ggml-zerocopy-null-buffer-hunt.md)。

**借 ggml 算子 = host 内存归 HYLE，device 显存归 ggml，就像 llama.cpp**：ggml 用 `gallocr` 分配它算子用的所有 device buffer（输入/中间/结果）；HYLE 的 host 数据经 `ggml_backend_tensor_set`(拷入，源可 pinned)/`tensor_get`(拷出) 进出（`tensor_set` 就是裸 `cudaMemcpyHostToDevice`，`ggml-cuda.cu:688`，对 host 源零约束）：

```
ggml_context* ctx = ggml_init({.mem_size=…, .mem_buffer=nullptr, .no_alloc=true});
ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);  ggml_set_input(a);
ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);  ggml_set_input(b);
ggml_tensor* c = ggml_mul_mat(ctx, a, b);                       ggml_set_output(c);
ggml_build_forward_expand(gf, c);
ggml_gallocr_alloc_graph(galloc, gf);          // ← device buffer 全归 ggml（CUDA 上必须）
ggml_backend_tensor_set(a, hyle_host_a, 0, ggml_nbytes(a));  // ← HYLE host 数据拷入(H2D)
ggml_backend_tensor_set(b, hyle_host_b, 0, ggml_nbytes(b));
// build cgraph → backend graph_compute（在 HYLE 调度下）
ggml_backend_tensor_get(c, hyle_host_c, 0, ggml_nbytes(c));   // ← 结果拷回 HYLE host(D2H)
```

- **device 归 ggml**：不注入外部 `->data`、不造假 dummy buffer（那在多算子 varied 形状必崩）。ggml 算子内核依赖 buffer 的 `context->device`/`buft` 元数据 + 图内复用 + padding，是算子正确性的一部分。
- **host 归 HYLE**：`tensor_set/get` 的 host 源/目标可以是 HYLE mem（`mem::Allocator`/`MemoryArena`）分配的内存。**性能接口点**：源用 pinned（compute-02b `PinnedStaging`）则 H2D 走 DMA 更快、可 async 重叠——借 ggml 的正确数据入口 = HYLE PinnedStaging → tensor_set → ggml 显存。
- **权重常驻**：反复用同一批权重（推理）时拷进 ggml 显存**一次**、常驻复用（llama.cpp 模式），非每次都拷。
- **HYLE 显存自管的适用范围**：compute-04/06 自造 kernel（真零拷贝跑在 HYLE Buffer 上，已验证）+ ② CAD/CAE 数据（KV-cache/网格/BVH）。借 ggml 算 AI 这条路，device 侧让位给 ggml。
- **quant padding**：F32 首版不涉；登记 quantized 类型须查 `ggml_backend_buft_get_alloc_size` 留 padding。

### ② completion 走 spawn_blocking，不接 GpuPoller（发现 3/4/8）

- **ggml CUDA stream 私有**（`common.cuh` `streams[][]` 懒创建，无注入 API）→ HYLE GpuPoller（compute-03 的 `cudaEventQuery` 轮询）**poll 不到 ggml 私有 stream 上的活**。
- **对外完成语义只有阻塞** `cudaStreamSynchronize`（`ggml_backend_synchronize`）→ **HYLE 把 `ggml_backend_graph_compute` + `synchronize` 整体 `spawn_blocking` 卸到 sched-11 的 blocking pool**，fiber 挂起、完成经 `JoinHandle`/`Waker` 唤醒。
- **这不违背"fiber 统一"**：语义一致（都经 sched Waker 唤醒），只是 completion source 是"阻塞卸载线程"而非"GpuPoller 轮询的 event"（发现 8）。sched-11 的 spawn_blocking 本就是为"借入的阻塞第三方"设计的。
- **对比 §4 自造 kernel 路径**：compute-03/04 的 `co_await stream` 走 GpuPoller 零阻塞轮询；compute-05 借 ggml 走 spawn_blocking 阻塞卸载。**两条路都归一到 fiber Waker，但机制不同**（这是 ggml stream 私有强加的，非缺陷）。

### ③ CPU 线程模型 = 合批性二分，`n_threads` 可配（发现 10 + compute-design「CPU 执行模型专章」，照 llama.cpp 铁证）

> **⚠️ 原计划两度被证伪**：先是"`ggml_backend_cpu_set_threadpool` 注入 HYLE 调度器"（发现 10 铁证 A：threadpool 不可插拔）；再是我一度简化的"n_threads=1 普世铁律"（也错——大模型合批场景该全核）。**正确道路 = 照抄 llama.cpp 的 batched 二分**（没人比它更懂调 ggml）。

- **核仲裁者是 HYLE 的 fiber 调度器，不是 ggml**：ggml 永远是被 HYLE 调度的叶子。llama.cpp 用 `ggml_backend_cpu_set_threadpool` 注入自建 threadpool 是对的（它自己就是顶层唯一仲裁者）；HYLE 底下有 fiber 调度器，让 ggml 起 threadpool 才叫第二调度器打架。**同一 API，llama.cpp 用对、HYLE 用错——差在谁说了算用核。**
- **线程数按"能不能合批"二分**（compute-design 专章决策轴）：

  | 场景 | 打架? | 线程模型 | 出处 |
  |---|---|---|---|
  | 合批大 graph（prefill/多序列）| 否（worker 本该 park）| **ggml 全核** | llama.cpp `llama-context.cpp:2428` |
  | 单请求 decode | 否 | 半核（访存瓶颈）| llama.cpp `common.cpp:133` |
  | **高并发独立小推理**（embedding/多租户/CAE 代理，合不了批）| **是** | **`n_threads=1` + HYLE 调度 N 个** | 发现 10 |
  | 有 GPU | 否 | CPU 侧一根 spawn_blocking sync 线程 | 发现 3/8 |

- **`n_threads==1` 零 spawn 铁证**：ggml `n_threads==1` 时不起任何线程，直接在调用线程 `ggml_graph_compute_thread(workers[0])`（`ggml-cpu.c:3356-3358`）——这是"独立小推理"格的落地机制。
- **借不了 ggml 核内并行**（发现 10 铁证 B）：mul_mat 的核内并行靠自旋 barrier（`ggml-cpu.c:576`），与协作式 fiber 对立，无法"借分块换 HYLE 线程"。要核内并行只能整块放 ggml 全核（合批格）。
- **vendored 必须关 OpenMP**：否则 OpenMP 路径（`ggml-cpu.c:3338`）在 `n_threads>1` 起线程组，"独立小推理"格的 n_threads=1 保证失效。
- **elementwise 自切块留 07**（发现 10 铁证 C）：mul/add/激活等零 barrier 算子 HYLE 可按行切喂多 fiber，边角优化，不影响 compute-05 骨架。

### ④ 性能杠杆 = 合批（发现 4，llama.cpp 实证的"真并发手段"）

- llama.cpp 真并发不是多 stream，而是**连续批处理**：把所有活跃请求的 token 合并进一个 batch 一次 decode。
- HYLE 侧同理：**合并多个算力请求成一个 ggml cgraph 一次 `graph_compute`** > 多 fiber 各自 spawn_blocking 卸载（后者线程切换 + 小 kernel launch 开销）。合批是 compute-05 的主要性能杠杆，不是多 stream。

### ⑤ adapter 单点隔离 + pin 上游版（发现 1 的 iron 反面教材）

- **ggml 类型全封 `ggml_bridge.cc`**：`ggml_context`/`ggml_tensor`/`ggml_backend_t`/`ggml_cgraph` **一个都不出现在 HYLE 公开头**。对外只 HYLE 抽象（Buffer/Stream/Device + 一个薄算子入口）。
- **ggml 升级只动这一文件 + 跑对拍测试**（发现 1：iron 把整个 sd.cpp 吞进来、clay/ggml 完全不通，正是要消除的东拼西凑）。
- **pin 上游 0.15.3**（clone `C:\public\project-hyle\ggml`，git head `eced84c8`），**不绑 iron 的冻结旧快照**。

### ⑥ ggml vendored（构建集成，2026-07-07 实测定形）

- **零重复下载，两层**：①**本地镜像 SOURCE_DIR**——`third_party/ggml`（`git archive` pin commit 的纯净源码，.gitignored）存在则 `FetchContent_Declare(ggml SOURCE_DIR ...)` → 永不联网；不存在则 `GIT_REPOSITORY + GIT_TAG` 拉 pin commit 一次。②**持久缓存 `FETCHCONTENT_BASE_DIR=.deps`**（cil2cpp runtime/CMakeLists.txt 技法）——populate/subbuild 落 in-source `.deps`（.gitignored），`rm -rf build`/重配置/同机第二 build 树都复用、不重 populate。
- **前提：每平台是独立机器、独立树**（这才让 in-source `.deps` 安全）：**Windows 在 `C:\public\project-hyle\hyle`（Windows 专用构建/开发目录）；WSL 在自己的 `/root/hyle`（原生 ext4 clone——绝不用 `/mnt/c`，那是 WSL 的 9p 网络挂载、慢且应视为不可访问）；mac 在 `~/hyle`**。无任何树跨 OS 共享，故 `.deps/ggml-subbuild` 的 CMakeCache 绝对路径在各自机器内永远自洽。（曾出现的"cache dir differs"跨 OS 报错是**误在 /mnt/c 上建 WSL 自找的**，非技法缺陷——已纠正为 WSL 用 /root/hyle。）
- **关**：`GGML_OPENMP`（关键，见 ③ n_threads=1 leaf）/`GGML_BUILD_TESTS`/`GGML_BUILD_EXAMPLES`/`GGML_CUDA_FA`（跳 ~120 FlashAttention kernel 实例——bridge 借 GEMM 不碰 attention，FA 占首编大头）。
- **只开**：CPU + CUDA 后端。**链接类型随 HYLE `BUILD_SHARED_LIBS`（默认 shared）**——原想强制 static，但 HYLE 顶层 `set(BUILD_SHARED_LIBS ON)`（normal var）遮蔽了 FORCE cache var，且孤立 static ggml 混在 shared HYLE 库里易 ODR/符号错，故 ggml 随大流 shared（runtime 需 libggml.so 在路径，归 close-01 install/rpath）。
- 门控：`HYLE_COMPUTE_GGML`（仿 `HYLE_COMPUTE_CUDA`/`HYLE_IO_TLS` 可选能力模式；不 vendor ggml 则不编 bridge、compute 其余照跑）。

## bridge 抽象（定形，待实现时以代码为准）

> **设计张力**：HYLE 不想暴露 ggml 类型，但 ggml 算子有 ~90 个、签名各异。**compute-05 不做"全算子门面"**（那是 L1 推理引擎的活）——只立**最小可验证的算子入口 + 内存/completion 桥接机制**，证明"HYLE host 内存（喂入/取出）+ HYLE 调度 + ggml 算子（跑在 ggml 显存）"三者合体跑通。
>
> **⚠️ 原则纠正（2026-07-09，用户指出"只暴露一个算子 + 消费者驱动让 HYLE 像玩具"）**：compute-05 只暴露 matmul 是"桥接机制证明"的合理起点,但**"算子门面按消费者驱动逐步加"这条被误用**——推理链的标准算子集是确定/低成本/无歧义的(ggml 都定义好),等虚构消费者是循环论证(只 matmul → 当不了推理底座 → 永无消费者)。**推理链本身就是消费者**。已在 **compute-09 step-1**(`compute-09-op-face.md`)立通用图执行器 + 铺 rms_norm/silu/add/mul,证明加算子=一句建图;step-2 铺完整 decoder block + 权重常驻。**"消费者驱动"仅适用于成本高/设计有歧义的件(如序列化 schema 编译器),不适用于确定的标准算子集。**

```
// ggml_bridge.h —— 唯一对外头。不含任何 ggml 类型（全在 .cc 里）。
#include "hyle/sched/blocking.h"   // BlockingPool, spawn_blocking, JoinHandle
namespace hyle::compute {

// ggml 后端句柄（HYLE 侧不透明）。绑一个 Device，internally holds ggml_backend_t。
// 生命周期 = 建一次、持久持有、析构 free（照 ggml 惯例 gpt-2 example: init→复用→free；见
// 「backend/context 生命周期与所有权」节）。Move-only（拥有 ggml_backend_t）。
// ⚠️ 线程安全契约：单实例串行用（ggml 无 backend 级线程安全承诺 + 全局 g_state）——
//    并发提交须每 fiber 一个 GgmlBackend 或调用方串行化（见「线程安全契约」节）。
class GgmlBackend {
 public:
  // n_threads: CPU 后端线程数。合批大 graph 传全核；高并发独立小推理传 1（③ 决策轴）。
  // CUDA 后端忽略之（GPU 侧 stream 调度）。默认 = 1（保守：独立小推理不打架；合批路径显式传全核）。
  static Result<GgmlBackend> create(const Device&, int n_threads = 1);
  bool valid() const noexcept;
};

// 首个消费者 = 稠密 matmul（发现 5 的"借 ggml AI 算子"最小验证；llama.cpp 热点算子）。
// C = A · Bᵀ，全 F32，三块都是 HYLE 管的 HOST 数据（① host 归 HYLE：ggml 用 gallocr 分配
// device buffer，数据经 tensor_set 拷入 / tensor_get 拷出；host 源可以是 HYLE mem/pinned）。
// A=[m×k], B=[n×k] 行主序，C=[m×n]。ASYNC：内部 spawn_blocking 把「建 ggml_context + 建
//   cgraph + gallocr alloc + tensor_set(H2D) + graph_compute + synchronize + tensor_get(D2H)」
//   整段卸到 `pool` 的线程；返回 JoinHandle<Status>（照 io/dns.h 模式）。`alloc` backs
//   JoinState + job。GgmlBackend + host 缓冲 a/b/c 必须活过 join（pool 线程读 a/b 写 c）。
// 这不是"matmul 专用 API"，是"证明桥接机制"的首个真实消费者；后续算子同法在 bridge 内加。
sched::JoinHandle<Status> ggml_matmul_f32(GgmlBackend&, sched::BlockingPool& pool,
                                          mem::Allocator alloc, const float* a,
                                          const float* b, float* c,
                                          int m, int k, int n);

// 合批（finding 4，性能杠杆）：N 个 matmul（可各自不同形状）合进一个 cgraph 一次提交。
// 每个 job 携带 host 指针 + 维度（POD，按值捕获进闭包）；host 缓冲须活过 join。
struct MatmulF32Job { const float* a; const float* b; float* c; int m, k, n; };
sched::JoinHandle<Status> ggml_matmul_f32_batched(GgmlBackend&, sched::BlockingPool&,
                                                  mem::Allocator, const MatmulF32Job*,
                                                  int count);

}  // namespace hyle::compute
```

**实现要点**（`ggml_bridge.cc`，全部 ggml 耦合封在此）：
- `ggml_init({.no_alloc=true})` 建 context → `ggml_new_tensor_2d` + `ggml_set_input`(输入)/`ggml_set_output`(结果) → `ggml_mul_mat` → `ggml_build_forward_expand` → `ggml_gallocr_alloc_graph`（① device buffer 全归 ggml）→ `ggml_backend_tensor_set`(HYLE host 输入拷入) → `ggml_backend_graph_compute` → `ggml_backend_tensor_get`(结果拷回 HYLE host)。合批 = 一个 context 里 N 组三元张量、一次 alloc、一次 compute。
- **completion**：`graph_compute` + `synchronize` 整体丢 `spawn_blocking`（②）。
- **CPU 后端**：`ggml_backend_cpu_init()` + `ggml_backend_cpu_set_n_threads(be, n_threads)`（③ 合批性二分）。
- **CUDA 后端**：`ggml_backend_cuda_init(device_ordinal)`；device buffer 由 gallocr 分配（① 不注入外部显存）。
- **持久 gallocr**：`create()` 时 `ggml_gallocr_new(buft)`，随 backend 一次建、每次算子复用（llama.cpp 模式）。
- **quant padding**：F32 首版不涉；登记 quantized 类型须查 `ggml_backend_buft_get_alloc_size` 留 padding。

## backend / context 生命周期与所有权（照 ggml 源码惯例定死）

ggml 有两个生命周期截然不同的对象，混淆会导致 free 错线程 / 悬垂：

| 对象 | 生命周期 | 谁持有 / 谁 free | 依据 |
|---|---|---|---|
| `ggml_backend_t`（cuda/cpu_init）| **建一次、持久复用、析构 free** | `GgmlBackend` 成员，`create()` 建 / 析构 `ggml_backend_free` | gpt-2 example: `ggml_backend_cuda_init` 一次 → 全程复用 → `ggml_backend_free`（`main-backend.cpp:206/943`）|
| `ggml_context`（no_alloc）+ cgraph | **每次算子调用建、算完即弃** | `ggml_matmul_f32` 内部局部，`spawn_blocking` 闭包里 `ggml_init`…`ggml_free` | simple-backend example: `ggml_init` → build → `ggml_free`（`simple-backend.cpp:79/93`）|

- **关键：context 的建/弃都在 blocking pool 线程内**（spawn_blocking 闭包体），不跨线程传 `ggml_context*`——闭包只捕获 backend 指针 + gallocr 指针 + 三个 host 指针 + 维度（都是 POD/指针，照 dns.h 的 POD Job 模式），在 pool 线程建 context、gallocr alloc、tensor_set 拷入、compute、synchronize、tensor_get 拷出、free context，返回 `Status`。**没有任何 ggml 对象逃逸到 fiber 线程。**
- **GgmlBackend + host 缓冲必须活过 JoinHandle.join()**：backend/gallocr 指针 + host 数据指针被闭包捕获、在 pool 线程用（读输入 a/b、写结果 c）。调用方保证（同任何 spawn_blocking 捕获引用的契约）。
- **no_alloc context 的 `mem_size`**：只存 tensor 元数据（`ggml_tensor` 结构体，非数据本身，device buffer 由 gallocr 分配）——按算子涉及的 tensor 数 × `ggml_tensor_overhead()` + graph overhead 估，小额定值即可。

## 线程安全契约（ggml 无 backend 级承诺，保守串行）

- **ggml 不承诺 backend/graph 线程安全**：`ggml.h:2778` 的 "thread-safe" 只指 `ggml_quantize_init/free`（无关）；backend 计算路径用全局 `g_state`（`ggml-cpu.c` barrier 段）+ 后端私有 stream/内存池，**单 backend 实例并发 `graph_compute` 是 data race**。
- **契约：一个 `GgmlBackend` 实例同一时刻只能有一个在途 `graph_compute`（串行用）**。落到 ③ 的两格：
  - **合批格**：本就单个大 graph 串行提交，天然满足。
  - **高并发独立小推理格（N fiber）**：**每 fiber 持一个自己的 `GgmlBackend`**（各自 `ggml_backend_cpu_init`，n_threads=1）——backend 建栈开销小（gpt-2 惯例是一次性建，但小模型/embedding 场景 per-fiber 建可接受，实测定夺），彼此无共享状态、无锁、真并行。**不共享单 backend 加锁**（那会把"独立小推理"重新串行化，违背该格初衷）。
- **CUDA 后端同理**：单 CUDA backend 的私有 stream 串行；多 fiber 各自 backend（或调用方串行化提交到一个 backend）。首版验收走"单 backend 串行 + 多 backend 并行"两条都测。
- **⚠️ 设备侧 OOM = 进程终止，非 Status，且无优雅降级出路**（红队审计发现，ggml 设计所限，诚实登记为**约束非缺口**）：bridge 只能捕获 `ggml_gallocr_alloc_graph` 返回 false + `ggml_backend_graph_compute` 的 `ggml_status`。但 device 显存 OOM 发生在 ggml 内部的 `GGML_ASSERT`/CUDA `CUDA_CHECK` 时会直接 `abort()` 整个进程——**这是 ggml 无异常/无错误返回的失败模型，借 ggml 就继承它，bridge 和消费者都无解**（除非 fork ggml 改错误处理，已否决）。`ggml_backend_buft_get_alloc_size` 只报"单 tensor 需多少字节"，**不查空闲显存、不保证 alloc 成功**（碎片/并发/其他分配都可能让预估通过但实际仍 abort），故**不构成"预估即可优雅降级"的可靠出路**。**这不是 compute-05 欠的功能，是借 ggml 的固有代价**——真要 OOM 可恢复得靠 compute-04/06 自造 kernel（HYLE 自己管显存分配、失败返 Status）。首版 F32 小矩阵不触发。
- **⚠️ 并发多 CUDA context 的 TSan 假阳（KI-4）**：`ConcurrentIndependentBackends`（N fiber × N 独立 CUDA context）在 WSL TSan 下报 1 条 libcuda 内部 memcpy race——**已查清为 CUDA 驱动内部跨 context 结构 TSan 不可见，非 HYLE 竞争**（决定性反证：同机制走 CPU 后端零 race）。Release 并发稳定。详见 known-issues KI-4。

## 验收（B 档最小可信闭环，三平台真机）

- **matmul 端到端**（CUDA）：HYLE host 数据装 A/B → `ggml_matmul_f32`（内部 gallocr 分配 device buffer + `tensor_set` 拷入）→ spawn_blocking 卸载 ggml graph_compute → fiber 挂起 → 完成唤醒 → `tensor_get` 拷回 host C → **对拍 HYLE 侧 CPU 参考（朴素三重循环）逐元素校验**。这是"HYLE host mem + HYLE fiber 调度 + ggml CUDA 算子（跑在 ggml 显存）"三者合体的决定性验证。
- **CPU 线程模型二分验证**（任何平台，无 GPU 可跑，③）：①`n_threads=1` 走 matmul（"独立小推理"格，`ggml-cpu.c:3356-3358` 零 spawn）；②`n_threads=全核` 走 matmul（"合批"格）+ 两者结果一致。**证明 n_threads 可配 + 两格都对。**
- **spawn_blocking 不阻塞 worker**：matmul 在途时，另一 fiber 在同 worker 上继续跑（发现 8）。
- **多 backend 真并发（线程安全契约验证）**：N 个 fiber 各持一个 `GgmlBackend`(n_threads=1) 并发跑独立 matmul（"高并发独立小推理"格）→ 全部结果正确 + **WSL TSan 零 race**。
- **合批 varied 形状**：3 个不同形状 matmul 合进一个 cgraph 一次提交，每个 golden 对拍——**这正是 dummy-buffer 注入版崩溃的 case（`ggml-cuda.cu:1924`），gallocr 模型下 buffer context 合法、跑通**（回归护栏）。
- **持久 backend 复用（create-once/reuse-many + move 语义）**：一个 `GgmlBackend`（create 后 move 一次）连跑 single(变形状×3) → batched(varied) → single，全部共享同一持久 gallocr——验证 gallocr 跨调用的 reserve/realloc（形状变 + 节点数变 1↔N）正确，且 move 后 galloc_ 指针完好。这是"持久 gallocr"设计卖点的直接测试（红队指出原 4 测试每 backend 只跑 1 次 op、未覆盖复用）。
- **三平台**：Win MSVC+nvcc / WSL GCC14+nvcc（CUDA 真跑）+ mac host（CPU 后端二分）+ WSL TSan（spawn_blocking 跨线程 wake + CPU 后端零 race；CUDA 多 context 的 libcuda 假阳记 KI-4）。
- **cloc 验真** + **对照 clay coverage-map**（ggml 无对应 clay 组件，登记为净新桥接）。

## 明确不做（防蔓延，登记后续）

| 不做 | 去向 | 为什么 |
|---|---|---|
| **全 ~90 算子门面** | L1 / 消费者驱动 | compute-05 只立桥接机制 + matmul 首消费者；算子按 CAE/推理消费者逐个在 bridge 内加（§0.2） |
| **让 ggml 自建 threadpool 当核仲裁者** | ⛔ 否决（发现 10 + ③）| HYLE fiber 调度器是唯一核仲裁者；ggml threadpool 不可插拔且会成第二调度器。合批格是 HYLE"授权"ggml 吃全核（此刻 worker 本该 park），非 ggml 自作主张 |
| **fork ggml** | ⛔ 否决（发现 1，compute-design §borrow） | fork 重写比借还贵；ggml 100k+ 活跃演进，merge 痛苦 |
| **借 sd.cpp/llama.cpp 整库** | ⛔ 否决（发现 1，iron 反面教材） | 借算子 kernel 粒度，不借整个推理应用 |
| **ggml-vulkan 接入** | compute-06 | 06 立 HYLE 自造 SPIR-V；ggml-vulkan（借 AI 算子的跨厂商实现）与之并存不重复（发现 5），按需 |
| **quantized 全类型** | 后续 | F32 首版验证机制；quant 的 padding/dequant 按推理消费者驱动 |
| **ggml sched（多后端算子图自动分派）** | 后续/按需 | ggml_backend_sched 是 ggml 的跨后端调度；HYLE 首版单后端手动建 graph，多后端分派待消费者 |
| **通用 GPU 自定义 kernel** | compute-04（已完成）/ 06 | ggml 结构上给不了（发现 9）；那是自造 kernel 的活，非 bridge |

## 交付物（预期）

- `include/hyle/compute/ggml_bridge.h`（`GgmlBackend` + `ggml_matmul_f32` 首消费者；**零 ggml 类型泄漏**）
- `src/compute/ggml_bridge.cc`（全部 ggml 耦合：no_alloc context + `ggml_set_input/output` + gallocr 分配 device buffer + `tensor_set`(host in)/`tensor_get`(host out) + cgraph 构建 + graph_compute spawn_blocking 卸载 + CPU `set_n_threads(n)` 合批性二分 + CUDA init；单发 + 合批同模型）
- `third_party/ggml`（vendored 上游 0.15.3，或 submodule）
- CMake：`HYLE_COMPUTE_GGML` 门控 + FetchContent pin commit（本地镜像 SOURCE_DIR 优先零下载 / GIT fallback + `FETCHCONTENT_BASE_DIR=.deps` 持久缓存，每机独立树）+ 关 tests/examples/OpenMP/CUDA_FA + 只开 CPU+CUDA；bridge TU 链 ggml + hyle_sched（spawn_blocking）+ hyle_compute
- `tests/compute/ggml_bridge_test.cc`（matmul host→ggml→host 端到端对拍 + CPU 线程二分〔n_threads=1 / 全核〕+ 多 backend 并发 + 合批 varied 形状〔dummy-buffer 崩溃 case 的回归护栏〕+ 持久 backend 复用〔create-once/reuse-many + move + 跨调用 gallocr reserve/realloc〕）

## 参考

- **compute-design**「compute-05 借 ggml 算力：调研发现汇总」发现 1~10 + 定案（**唯一技术信源**，本文档是其里程碑化）。
- **motivation** ggml vs llama.cpp 分层 + HYLE 定位（算子借 ggml/底座替 llama.cpp/通用 GPU 补 ggml 之不能）。
- ggml 上游 0.15.3：`include/ggml.h`（`ggml_init_params.no_alloc`:663、`ggml_new_tensor_2d`:825、`ggml_mul_mat`、`ggml_new_graph`/`ggml_build_forward_expand`）、`include/ggml-backend.h`（`ggml_backend_graph_compute`:104、`ggml_backend_synchronize`:98、`ggml_backend_tensor_set/get`:92）、`include/ggml-cpu.h`（`ggml_backend_cpu_init`:131、`set_n_threads`:134）、`include/ggml-cuda.h`（`ggml_backend_cuda_init`:23）。
- 铁证行号（ggml 0.15.3）：`src/ggml-cpu/ggml-cpu.c:3356-3358`（n_threads=1 零 spawn）、`:471`（threadpool 封闭 struct）、`:576`（自旋 barrier）、`:1355`（mul_mat barrier）、`ggml-cpu-impl.h:18`（compute_params ith/nth）、`ggml.h:2844`（threadpool_params 无 fn-ptr）。
- 铁证行号（llama.cpp，③ 线程模型来源）：`src/llama-context.cpp:2425-2429`（graph_compute 按 batched 二分 n_threads + 双 threadpool）、`common/common.cpp:133`（单序列半核 n/2）、`:275`（batch 全核 = 物理核数）。
- 生命周期惯例出处（ggml examples）：`examples/gpt-2/main-backend.cpp:206/943`（backend 建一次→复用→free）、`examples/simple/simple-backend.cpp:79/93`（context 每次 init→build→free）。thread-safe 澄清：`ggml.h:2778` 仅指 quantize_init/free，非 backend。
- HYLE 前置代码：`include/hyle/sched/blocking.h:192`（spawn_blocking 签名）、`include/hyle/io/dns.h:73-94`（POD Job + spawn_blocking + JoinHandle 模式，bridge 照抄）。
- 前置里程碑：compute-01/02/02c/03 + sched-11（spawn_blocking）。compute-06（ggml-vulkan 并存 + 自造 SPIR-V）、compute-07（CPU 算子后端）、compute-08（HIP/Metal 借 ggml）。
