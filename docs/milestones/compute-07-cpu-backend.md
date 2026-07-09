# compute-07：CPU 算子后端 —— compute-04 Kernel 抽象接 hyle simd

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | compute/ |
| **前置** | compute-04（Kernel/Stream/launch 抽象 + host 路径 hook）、simd-04（运行期 dispatch 的 `dot_f32`/`saxpy_f32`/`exp_f32_batch` 等数值核）|
| **状态** | ✅ **完成（2026-07-09，WSL GCC + MSVC 真跑 6/6，full compute 78/78）**。`host_ops.h/.cc` 提供库级 `Kernel::HostFn`(saxpy/vecadd/dot/exp，f32+f64)包 hyle simd 核，经**同一** compute-04 `launch(stream,kernel,config,args)` ABI 驱动。hyle simd 的首个 compute 消费者（含刚铺全的 f64）。link `hyle_simd` PRIVATE。|

> **一句话定位**：compute-04 的"提交任意 kernel"抽象已有 CUDA(NVRTC)/Vulkan(SPIR-V)两个 GPU 后端；**compute-07 是它的 CPU 后端**——host kernel 用 hyle simd 数值核实现，completion 恒 ready（同步内联）。**接 hyle simd，不是借 ggml。**

## 为什么做 / 消费者

- **compute-04 抽象的 CPU 落地**：README:175 定义——`host_backend` 原本只是 device/stream/event 的 inline 兜底(测试用)，**不是真算子后端**。07 把 simd 核包成真 host kernel，让抽象在无 GPU 时也能真算。
- **双用途**：
  1. **无 GPU 功能基线**：mac/CI/边缘设备无显卡时，compute-04 抽象仍可跑（host 后端 event 恒 kReady）。
  2. **GPU 对拍 oracle**：`host_vecadd_f32` 与 CUDA vecadd 测试核同语义、`host_saxpy_f32` 与 Vulkan saxpy 测试核同语义——同输入喂 CPU/GPU 逐元素比对，是 compute 的正确性回归护栏。
- **hyle simd 的首个 compute 消费入口**：simd 支柱(dot/saxpy + 全 14 f32/f64 超越)一直缺 compute 侧下家，07 正是它。CAE/科学计算的 CPU 数值路从这里长出(simd 有 f64 全套，ggml 没有)。

## 与 compute-05 的正交关系（纠错定案）

> **⚠️ 一次方向纠错**：设计中一度误判"07 也借 ggml CPU 后端"——**错**，那和 05 重复。翻 README:175 + backlog + scope 定论：07 是另一条正交线。

| | compute-05（借 ggml） | compute-07（接 hyle simd） |
|---|---|---|
| 借谁 | ggml AI 算子（matmul/softmax…，自带 SIMD+量化 GEMM）| hyle simd 数值核（dot/saxpy/exp_batch…）|
| 属哪套抽象 | `ggml_bridge`（算子库供应商，独立入口）| compute-04 `Kernel::launch` 的 CPU 后端 |
| completion | spawn_blocking 卸载（ggml stream 私有）| 恒 ready（CPU 同步内联）|
| 定位 | AI 推理算子 | 无 GPU 基线 + GPU 对拍 oracle |

**AI-GEMM 恒借 ggml**（hyle simd 无 matmul，dot 拼 GEMM 慢数十倍）；07 不碰 GEMM，是通用 kernel 抽象的 CPU 后端。两条 CPU 路并存不重复。

## 落地形态（compute-04 hook 已就位，07 是"用起来"非"重建"）

- **已就位的 host 路径**（compute-04）：`Kernel::HostFn = void(*)(const LaunchConfig&, const KernelArgs&)`、`load_host(device, fn)`、`host_kernel_launch` 同步调用、`host_event_query` 恒 kReady。07 零新运行时、零 sched 改动。
- **host kernel 解包 KernelArgs**（照 `host_vecadd` 模式）：buffers[] 按序 = 输入/输出指针；scalar blob 按 `fields[i].{offset,size}` 解包（与 kernel.h typed-shell 打包顺序一致：buffers 先、scalars 后，各标量按自身大小对齐）→ 转调 simd 核。
- **交付的 host kernel**（`host_ops.h`）：
  - `host_saxpy_f32/f64`：buffers[0]=x,[1]=y；scalars n(int),a(float/double)。`y=a*x+y`（simd::saxpy）。
  - `host_vecadd_f32`：buffers a,b,c；scalar n。`c=a+b`（seed c=b + saxpy(1,a,c)；对齐 CUDA vecadd 核做 oracle）。
  - `host_dot_f32/f64`：buffers x,y,out；scalar n。`out[0]=sum(x*y)`（simd::dot，归约）。
  - `host_exp_f32`：buffers src,dst；scalar n。`dst=exp(src)`（simd::exp_f32_batch，逐元素超越）。
- **不引第二运行时**（scope 铁律）：host kernel 单线程内联；数据并行铺多核是 graph-02 的 `parallel_for`(sched fork-join)，不是后端自起线程。

## 验收

- **抽象一致性**：每个 host kernel 经 `Kernel::load_host` + `launch(stream,kernel,config,args)` 驱动（与 GPU 后端完全同 ABI），host Stream completion 恒 ready 故无需 co_await。
- **正确性对拍**：saxpy/dot/exp 对拍标量参考(f32 ~1e-4/1e-5 相对、f64 dot ~1e-12、saxpy DOUBLE_EQ)；vecadd 用 CUDA 测试核同公式。simd tail 覆盖(n=1000/777/513 非 SIMD 宽度倍数)。
- **双平台真跑**：WSL GCC14 6/6 + MSVC 6/6，full compute 78/78 无回归。**MSVC 抓一个可移植性 bug**：`__builtin_memcpy`(GCC/Clang builtin)MSVC 无 → 改 `std::memcpy`。双平台真编真测才暴露。

## 对抗式审查（2026-07-09 红队,发现 6 项全修）

首版只做了测试、跳过对抗式自审(CHECKLIST 第 4 步)。补审(独立红队 subagent + 自审)揪出 6 项、全修:

| # | 类别 | 问题 | 修法 |
|---|---|---|---|
| 1 | 契约 | host kernel 对 buffer_count/scalar_count/null 零校验,违背库 fail-fast 纪律(vulkan_backend:1098 有先例)| `require()`+`scalar_at` 加 `HYLE_FATAL_ASSERT`(args 是调用方契约非外部输入)|
| 2 | **严重 bug** | `int n<0` → `static_cast<size_t>` 变 ~2^63 → simd 核越界扫内存 | `HYLE_FATAL_ASSERT(n>=0)` |
| 3 | 性能 | `host_vecadd` 标量拷贝+saxpy 两趟内存(三趟 c 流量)| **simd 层补 `add_f32/f64` 逐元素加原语**(本就该有),vecadd 单趟直调 |
| 4 | 规范 | `scalar_at` 假设 `fields[i].size==sizeof(T)` 无校验,类型错配静默错读 | 加 `HYLE_FATAL_ASSERT(fields[i].size==sizeof(T))` |
| 5 | 测试缺口 | 无契约/负面测试;未经真实 typed-shell 驱动(测试自复刻打包逻辑)| 加 3 death test(null/负n/buffer不足)+ `TypedShellSaxpyF32`(真 `launch(s,k,cfg,Buffer...,scalar...)`)|
| 6 | 规范 | f64 缺 exp(f32 有);对称破缺 | 补 `host_exp_f64`(simd `exp_f64_batch` 已有)|

**方法学教训**:"跑了 6 个测试通过" ≠ "做了对抗式审查"。测试证功能,红队挑正确性/性能/契约/规范偏离——两者不可互替(#2 负 n 越界 + #3 性能偏离靠测试绿发现不了)。**#3 根因是 simd 能力面缺 elementwise add——红队反向暴露了 simd 一个真缺口,顺带补全 `add_f32/f64`(dispatch 五后端 + ops_test 对拍 + op_parity 三方 bench 与 Highway/xsimd 统计打平,带宽受限 op 预期但实测坐实非假设)。** 修后 WSL 83/83 + simd 234 + avx512 SDE add 验 + MSVC。

## 明确不做（防蔓延）

| 不做 | 去向 | 为什么 |
|---|---|---|
| 重造 GEMM/matmul | compute-05 借 ggml | hyle simd 无 matmul；ggml CPU 后端自带高性能量化 GEMM |
| 给 07 堆 90 个 host 算子门面 | 消费者驱动 | 只提供已有 GPU 对拍需要的 + simd 现成核；按 CAE/推理消费者增量加 |
| CPU 后端自起线程池铺数据并行 | graph-02 parallel_for | scope 铁律"不引第二运行时"；铺多核归 HYLE sched fork-join |
| f64 全超越都包成 host kernel | 消费者驱动 | 首版只 exp(证逐元素超越路径通)；其余按 CAE 需求加，simd 核已全在 |

## 交付物

- `include/hyle/compute/host_ops.h`（库级 HostFn 声明 + 每 kernel 的 KernelArgs 解包契约；零 simd 类型泄漏，只声明 HostFn 签名）
- `src/compute/host_ops.cc`（解包 KernelArgs + 转调 `simd::saxpy_f32/f64`、`dot_f32/f64`、`exp_f32_batch`）
- `tests/compute/host_ops_test.cc`（6 测试：经 compute-04 launch ABI 驱动 + 对拍标量参考）
- CMake：`host_ops.cc` 入 compute 源 + `target_link_libraries(hyle_compute PRIVATE hyle_simd)`（simd 首个 compute 消费者）

## 参考

- **compute-design**「compute-07 CPU 算子后端：接 hyle simd，不是借 ggml」（唯一技术信源）+ README:175 缺口登记 + scope 后端矩阵 CPU fallback 行。
- compute-04 kernel.h（`HostFn`/`load_host`/typed-shell 打包）、`tests/compute/kernel_test.cc:44`（host_vecadd 解包范式）。
- simd `include/hyle/simd/ops.h`(dot/saxpy f32/f64)、`math.h`(exp_f32_batch)。
- 前置里程碑：compute-04（kernel 抽象）、simd-04（运行期 dispatch）。相关：compute-05（借 ggml，正交）、graph-02（parallel_for 数据并行）。
