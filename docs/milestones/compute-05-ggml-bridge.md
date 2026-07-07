# compute-05：ggml_bridge —— 借 ggml 算子（不借运行时）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-01（Device/Stream/Event）、compute-02（Buffer.device_ptr）、compute-02c（DevicePool）、compute-03（co_await 完成 / spawn_blocking）、sched-11（spawn_blocking / BlockingPool / JoinHandle）、mem（Allocator） |
| **状态** | ✅ **完成（2026-07-07，双平台 GPU 真机 + CPU 三格 + TSan 诊断）**。ggml_bridge.h 39 + .cc 220(C++) + test 177 = **259 code / 146 comment**（cloc；精瘦=重活在 ggml）。零拷贝 CUDA matmul 在 RTX 4070 真跑（Win MSVC+nvcc 3/3、WSL gcc14+nvcc 3/3），CPU n_threads 二分（1 与全核）+ 6-backend 并发全绿。**⚠️ 实现期两处真跑纠错**：①**发现 2「NULL buffer 零拷贝」被证伪**——CUDA matmul deref `tensor->buffer->buft/context`（ggml-cuda.cu:1832/1853），NULL 段错误；修=`alloc_ctx_tensors` 拿合法 buffer 元数据 + 覆写 `->data` 注入 HYLE 指针（数据仍零拷贝），详见 `docs/debug/ggml-zerocopy-null-buffer-hunt.md`。②**WSL TSan 报 10 race 全系 mimalloc 跨线程块回收假阳**（`[anon:mimalloc]` 地址，非 HYLE 逻辑；Release 并发 5/5 稳）→ 记为 KI-3。mac host CPU 待 commit 后 clean pull 验。 |

> **范围**：把 ggml 当**算子库供应商**接进 HYLE——在**唯一一个 adapter 文件** `ggml_bridge.cc` 里，让 ggml 的 AI 算子（matmul/softmax/rope/conv…约 90 个）在 **HYLE 自管的显存/completion/调度** 之上运行。**借的是"算子 kernel"粒度，不是整库应用（不借 sd.cpp/llama.cpp），不 fork，不注入 ggml 运行时。**
>
> **一句话定位**（motivation 已固化）：**算子借 ggml，底座替 llama.cpp，通用 GPU 计算补 ggml 之不能（compute-04/06）**。compute-05 只做第一句。

## 为什么做这个 / 消费者

- **北极星①推理**：算子几乎全在 ggml。HYLE 对推理的价值 = 统一底座（io/sched/mem/completion），**算子借 ggml 就够**——这是 compute-05 的主消费者。
- **北极星②CAE 能借的部分**：GEMM/SpMV 里 ggml 已有的（如稠密 matmul）可借；ggml 没有的几何核/网格 stencil/BVH/稀疏求解器走 compute-04/06 自造 kernel（发现 9：ggml 结构上给不了）。
- **不是重造轮子**：ggml-backend 已把 CUDA/HIP/Metal/Vulkan/SYCL 的算子跑通（发现 5 矩阵的"借 ggml"那一列），HYLE 不重写这些算子。compute-05 = 把它接进 HYLE 的脊。

## 核心抉择（全部来自 compute-design 发现 1~10，此处只列结论 + 落地形态）

### ① 内存零拷贝：no_alloc + `tensor->data = HYLE device_ptr`（发现 2，不 fork）

HYLE 自己 `cudaMalloc`（compute-02 Buffer）的 `device_ptr()`，直接喂给 ggml 算子零拷贝地算，**ggml 不拒绝**（铁证：ggml CUDA 普通算子路径从不校验 buffer 归属，只强转 `(char*)src->data`）：

```
ggml_context* ctx = ggml_init({.mem_size=…, .mem_buffer=nullptr, .no_alloc=true});
ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
a->data = hyle_buffer_a.device_ptr();   // ← HYLE 显存直接注入，tensor->buffer 留 NULL
ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
b->data = hyle_buffer_b.device_ptr();
ggml_tensor* c = ggml_mul_mat(ctx, a, b);
c->data = hyle_buffer_c.device_ptr();
// build cgraph → backend graph_compute（在 HYLE 调度下）
```

- **约束**（照 compute-design 发现 2）：F32 最省心；**quantized 类型 `get_alloc_size` 加 padding，HYLE 分配须比 `ggml_nbytes` 多留 padding 否则 quant kernel 越界**；指针须在正确 device；`tensor->buffer` 留 NULL 仅普通 CUDA 算子安全（tensor_get/set、split-buffer、sched 会读 buffer，那些路径不走）。
- **拷贝版 fallback**：`ggml_backend_tensor_set/get`（发现 2）作对拍与兜底路径。**不用** `ggml_backend_tensor_alloc`——它要求 addr 落在 ggml buffer 内、且 ggml CUDA buffer 析构会 `cudaFree` 你的指针 = 所有权冲突（发现 2）。
- **风险管理**：依赖"kernel 只读 data、不反查 buffer"这个**当前为真但非 ggml 公开承诺**的行为 → **pin 上游版 + adapter 单点封装 + 零拷贝 vs 拷贝版对拍测试兜底**（ggml 升级跑对拍即知是否破，破了退拷贝版）。

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

### ⑥ ggml vendored（构建集成）

- `add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)`（同 picohttpparser 的 vendoring 惯例，但 ggml 有自己的 CMake 故用 add_subdirectory 非直接编源）。
- **关**：`GGML_BUILD_TESTS`/`GGML_BUILD_EXAMPLES`/**OpenMP（关键，见 ③）**。
- **只开**：CPU + CUDA 后端，静态链（Vulkan 后端等 compute-06 再评）。
- 门控：`HYLE_COMPUTE_GGML`（仿 `HYLE_COMPUTE_CUDA`/`HYLE_IO_TLS` 可选能力模式；不 vendor ggml 则不编 bridge、compute 其余照跑）。

## bridge 抽象（定形，待实现时以代码为准）

> **设计张力**：HYLE 不想暴露 ggml 类型，但 ggml 算子有 ~90 个、签名各异。**compute-05 不做"全算子门面"**（那是 L1 推理引擎的活）——只立**最小可验证的算子入口 + 内存/completion 桥接机制**，证明"HYLE 显存 + HYLE 调度 + ggml 算子"三者合体跑通。算子门面按消费者驱动逐步加（§0.2 消费者驱动原则）。

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
// C = A(m×k) · B(k×n)，全 F32，三块都是 HYLE 显存 Buffer（device_ptr 零拷贝注入 tensor->data）。
// ASYNC：内部 spawn_blocking 把「建 ggml_context + 建 cgraph + graph_compute + synchronize」
//   整段卸到 `pool` 的线程；返回 JoinHandle<Status>（照 io/dns.h 模式，非 Task——spawn_blocking
//   的天然产物，从 fiber `.join()` 即 fiber-aware 挂起）。`alloc` backs JoinState + job（同
//   dns::resolve：用你 join 的那个 allocator）。GgmlBackend 必须活过 join（backend 持久）。
// 这不是"matmul 专用 API"，是"证明桥接机制"的首个真实消费者；后续算子同法在 bridge 内加。
sched::JoinHandle<Status> ggml_matmul_f32(GgmlBackend&, sched::BlockingPool& pool,
                                          mem::Allocator alloc, const Buffer& a,
                                          const Buffer& b, Buffer& c,
                                          int m, int k, int n);

}  // namespace hyle::compute
```

**实现要点**（`ggml_bridge.cc`，全部 ggml 耦合封在此）：
- `ggml_init({.no_alloc=true})` 建 no-alloc context → `ggml_new_tensor_2d` → `tensor->data = buffer.device_ptr()`（②的零拷贝）→ `ggml_mul_mat` → `ggml_new_graph` + `ggml_build_forward_expand` → `ggml_backend_graph_compute`。
- **completion**：`graph_compute` + `ggml_backend_synchronize` 整体丢 `spawn_blocking`（③）。
- **CPU 后端**：`ggml_backend_cpu_init()` + `ggml_backend_cpu_set_n_threads(be, n_threads)`（③ 合批性二分：合批传全核、独立小推理传 1）。
- **CUDA 后端**：`ggml_backend_cuda_init(device_ordinal)`；HYLE 显存指针直接当 tensor->data（②）。
- **quant padding**：F32 首版不涉；登记 quantized 类型须查 `ggml_backend_buft_get_alloc_size` 留 padding。

## backend / context 生命周期与所有权（照 ggml 源码惯例定死）

ggml 有两个生命周期截然不同的对象，混淆会导致 free 错线程 / 悬垂：

| 对象 | 生命周期 | 谁持有 / 谁 free | 依据 |
|---|---|---|---|
| `ggml_backend_t`（cuda/cpu_init）| **建一次、持久复用、析构 free** | `GgmlBackend` 成员，`create()` 建 / 析构 `ggml_backend_free` | gpt-2 example: `ggml_backend_cuda_init` 一次 → 全程复用 → `ggml_backend_free`（`main-backend.cpp:206/943`）|
| `ggml_context`（no_alloc）+ cgraph | **每次算子调用建、算完即弃** | `ggml_matmul_f32` 内部局部，`spawn_blocking` 闭包里 `ggml_init`…`ggml_free` | simple-backend example: `ggml_init` → build → `ggml_free`（`simple-backend.cpp:79/93`）|

- **关键：context 的建/弃都在 blocking pool 线程内**（spawn_blocking 闭包体），不跨线程传 `ggml_context*`——闭包只捕获 backend 指针 + 三个 `device_ptr()` + 维度（都是 POD/指针，照 dns.h 的 POD Job 模式），在 pool 线程建 context、注入 data、compute、synchronize、free context，返回 `Status`。**没有任何 ggml 对象逃逸到 fiber 线程。**
- **GgmlBackend 必须活过 JoinHandle.join()**：backend 指针被闭包捕获、在 pool 线程用。调用方保证（同任何 spawn_blocking 捕获引用的契约）。
- **no_alloc context 的 `mem_size`**：只存 tensor 元数据（`ggml_tensor` 结构体，非数据本身，data 是 HYLE 显存）——按算子涉及的 tensor 数 × `ggml_tensor_overhead()` + graph overhead 估，小额定值即可。

## 线程安全契约（ggml 无 backend 级承诺，保守串行）

- **ggml 不承诺 backend/graph 线程安全**：`ggml.h:2778` 的 "thread-safe" 只指 `ggml_quantize_init/free`（无关）；backend 计算路径用全局 `g_state`（`ggml-cpu.c` barrier 段）+ 后端私有 stream/内存池，**单 backend 实例并发 `graph_compute` 是 data race**。
- **契约：一个 `GgmlBackend` 实例同一时刻只能有一个在途 `graph_compute`（串行用）**。落到 ③ 的两格：
  - **合批格**：本就单个大 graph 串行提交，天然满足。
  - **高并发独立小推理格（N fiber）**：**每 fiber 持一个自己的 `GgmlBackend`**（各自 `ggml_backend_cpu_init`，n_threads=1）——backend 建栈开销小（gpt-2 惯例是一次性建，但小模型/embedding 场景 per-fiber 建可接受，实测定夺），彼此无共享状态、无锁、真并行。**不共享单 backend 加锁**（那会把"独立小推理"重新串行化，违背该格初衷）。
- **CUDA 后端同理**：单 CUDA backend 的私有 stream 串行；多 fiber 各自 backend（或调用方串行化提交到一个 backend）。首版验收走"单 backend 串行 + 多 backend 并行"两条都测。

## 验收（B 档最小可信闭环，三平台真机）

- **零拷贝 matmul 端到端**（CUDA）：HYLE Buffer 装 A/B（compute-02 `set` H2D）→ `ggml_matmul_f32` 零拷贝注入 → spawn_blocking 卸载 ggml graph_compute → fiber 挂起 → 完成唤醒 → HYLE Buffer `get` C 回来 → **对拍 HYLE 侧 CPU 参考（simd `dot_f32` 或朴素三重循环）逐元素校验**。这是"HYLE 显存 + HYLE fiber 调度 + ggml CUDA 算子"三者合体的决定性验证。
- **CPU 线程模型二分验证**（任何平台，无 GPU 可跑，③）：①`n_threads=1` 走 matmul，**验证不 spawn 额外线程**（TSan / 线程计数探针）——坐实"独立小推理"格零 spawn（`ggml-cpu.c:3356-3358`）；②`n_threads=全核` 走 matmul，验证 ggml 起线程组吃满核（"合批"格）+ 两者结果对拍一致。**证明 n_threads 可配 + 两格都对。**
- **零拷贝 vs 拷贝版对拍**：同输入两条路（no_alloc+data 注入 vs `ggml_backend_tensor_set`）结果一致——兜底版正确性 + 零拷贝版无越界。
- **spawn_blocking 不阻塞 worker**：matmul 在途时，另一 fiber 在同 worker 上继续跑（证明 fiber 挂起不占 worker，发现 8 的"比 llama.cpp 单线程阻塞好"）。
- **多 backend 真并发（线程安全契约验证）**：N 个 fiber 各持一个 `GgmlBackend`(n_threads=1) 并发跑独立 matmul（"高并发独立小推理"格）→ 全部结果正确 + **WSL TSan 零 race**（坐实"每 fiber 一 backend、无共享状态"契约，反证"共享单 backend 会 race"）。
- **合批**：多个 matmul 合进一个 cgraph 一次提交 vs 分次提交，验证正确性（性能对比登记 bench）。
- **三平台**：Win MSVC+nvcc / WSL GCC14+nvcc（CUDA 真跑）+ mac host（CPU 后端二分：n_threads=1 与全核都测，无 CUDA）+ WSL TSan（spawn_blocking 跨线程 wake + CPU 后端零 race）。
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
- `src/compute/ggml_bridge.cc`（全部 ggml 耦合：no_alloc context + tensor->data 注入 + cgraph 构建 + graph_compute spawn_blocking 卸载 + CPU `set_n_threads(n)` 合批性二分 + CUDA init；对拍拷贝版路径）
- `third_party/ggml`（vendored 上游 0.15.3，或 submodule）
- CMake：`HYLE_COMPUTE_GGML` 门控 + `add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)` + 关 tests/examples/OpenMP + 只开 CPU+CUDA；bridge TU 链 ggml + hyle_sched（spawn_blocking）+ hyle_compute
- `tests/compute/ggml_bridge_test.cc`（零拷贝 matmul 端到端 + CPU 线程二分〔n_threads=1 不 spawn / 全核起线程组〕对拍 + 零拷贝/拷贝版对拍 + spawn_blocking 不阻塞 worker + 合批）

## 参考

- **compute-design**「compute-05 借 ggml 算力：调研发现汇总」发现 1~10 + 定案（**唯一技术信源**，本文档是其里程碑化）。
- **motivation** ggml vs llama.cpp 分层 + HYLE 定位（算子借 ggml/底座替 llama.cpp/通用 GPU 补 ggml 之不能）。
- ggml 上游 0.15.3：`include/ggml.h`（`ggml_init_params.no_alloc`:663、`ggml_new_tensor_2d`:825、`ggml_mul_mat`、`ggml_new_graph`/`ggml_build_forward_expand`）、`include/ggml-backend.h`（`ggml_backend_graph_compute`:104、`ggml_backend_synchronize`:98、`ggml_backend_tensor_set/get`:92）、`include/ggml-cpu.h`（`ggml_backend_cpu_init`:131、`set_n_threads`:134）、`include/ggml-cuda.h`（`ggml_backend_cuda_init`:23）。
- 铁证行号（ggml 0.15.3）：`src/ggml-cpu/ggml-cpu.c:3356-3358`（n_threads=1 零 spawn）、`:471`（threadpool 封闭 struct）、`:576`（自旋 barrier）、`:1355`（mul_mat barrier）、`ggml-cpu-impl.h:18`（compute_params ith/nth）、`ggml.h:2844`（threadpool_params 无 fn-ptr）。
- 铁证行号（llama.cpp，③ 线程模型来源）：`src/llama-context.cpp:2425-2429`（graph_compute 按 batched 二分 n_threads + 双 threadpool）、`common/common.cpp:133`（单序列半核 n/2）、`:275`（batch 全核 = 物理核数）。
- 生命周期惯例出处（ggml examples）：`examples/gpt-2/main-backend.cpp:206/943`（backend 建一次→复用→free）、`examples/simple/simple-backend.cpp:79/93`（context 每次 init→build→free）。thread-safe 澄清：`ggml.h:2778` 仅指 quantize_init/free，非 backend。
- HYLE 前置代码：`include/hyle/sched/blocking.h:192`（spawn_blocking 签名）、`include/hyle/io/dns.h:73-94`（POD Job + spawn_blocking + JoinHandle 模式，bridge 照抄）。
- 前置里程碑：compute-01/02/02c/03 + sched-11（spawn_blocking）。compute-06（ggml-vulkan 并存 + 自造 SPIR-V）、compute-07（CPU 算子后端）、compute-08（HIP/Metal 借 ggml）。
