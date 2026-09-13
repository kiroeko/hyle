# Compute Backend CPU v1：主机执行因子审计

## Closure card

- component / owner：`compute_backend_cpu` / Compute；round=`compute-backend-cpu-v1-factor`；
- baseline commit：`ec83514fe9d082698eb05018680cfaeda1822a5a`；
- trigger 与证据：当前 `compute` 同时拥有通用 runtime、CPU device/queue 和 HYLE SIMD/native kernels，导致
  后续 CUDA/Metal-only product 无法只取得 runtime，并阻碍 backend 与 provider 并行施工；`Device::cpu()`、
  全局单项 `enumerate_devices()` 和同步 queue 已提供可迁移的 CPU 边界；
- 正常 consumer 任务：普通 C++ 用户枚举/选择 host CPU，建立 context 和 ordered queue，取得 host
  `MemoryResourceRef`，执行 direct/prepared work，显式 close/drain；空 CPU completion 仍与异步 backend
  使用同一 runtime 合同；
- 主产品纵切：HYLE-native/oneDNN 的 transformer CPU prepared operation；正交纵切为不经过 Graph 的 typed
  host engineering kernel；
- scope：逐入口裁决并迁移 CPU device factory/enumeration/properties、backend context/queue、host resource 与
  CPU native interop；验证它只实现执行轴，不拥有 Operation schema 或 kernel；
- non-goals：任何算子/reference/SIMD kernel、oneDNN/ggml、GPU backend、线程池、Graph、Native FFI；
- 直接依赖 / 消费者：候选公开依赖 Compute runtime 与 Mem；CPU 拓扑、NUMA 和 worker 规划继续由
  Platform/Sched 各自拥有，本 factor 不复制也不私有依赖它们；直接消费者为
  `compute_provider_native`、oneDNN、direct CPU Compute 和 Graph；
- 性能：`P0` 优先，CPU create/submit/close 不隐藏线程、allocation、syscall 或 provider search；若 runtime
  seam 比直接 caller-thread function 有无法由合同解释的 material 成本，候选后只跑一次同合同 `P1`；
- 验证：无设备/非法 ordinal、properties、resource identity、queue ordering、并发 queue、close/drain、
  context/queue/completion 销毁顺序、错误注入、逐头、三平台、sanitizers、source/static/shared、factor-only、
  P0/必要 P1 和红队；
- 预算 / stop：API 与 owner 最多两轮，FIX 各最多两轮；若通用 runtime 尚不能承载 backend state/resource
  lifetime，返回 Compute v4 而不是在 CPU factor 私造另一套 Context/Completion。

## SCOPED 待审计项

1. `Device::cpu()`、全局 enumeration/query、Context/Queue 当前实现分别 `MOVE/REPLACE/DELETE` 到哪里；
2. CPU backend 是否只需 caller-thread ordered queue，怎样使用通用异步 terminal 而不增加 heap/atomic；
3. `mem::system_resource()` 是公开默认 resource 还是由 typed CPU adapter 显式返回；
4. source/static/shared 与 Compute product façade 的唯一投影。

AUDITED 前不移动 CPU 产品代码或添加依赖。

## AUDITED 结论（2026-09-12）

| ID | 当前入口 | 裁决 | v1 形状 |
|---|---|---|---|
| B10 | `Device::cpu()` | `REPLACE/MOVE` | `compute::cpu::backend()` 返回显式 Backend；`backend.device(0)` 与 `cpu::device()` 是零分配 convenience |
| B11 | 全局 `enumerate_devices()` | `DELETE` | 无静态注册；`Backend::device_count/device(index)` 只枚举该显式 backend |
| B12 | `query_device(Device)` | `MOVE` | `Device::properties()` 经其 backend vtable 查询，非法/过期 identity 返回 typed error |
| B13 | Context 内没有 backend state | `FIX` | CPU create/destroy hooks 无额外 runtime allocation；通用 Context state 保留 Device 与 provider slots |
| B14 | Queue 只是 Context borrow | `FIX` | 通用 QueueState 独立 owner；CPU queue handle 为空，caller-thread ordered submit，drain 为 O(1) |
| B15 | 无 backend resource factory | `FIX` | `cpu::memory_resource(Device)` 明确返回 `mem::system_resource()`；domain/properties 不复制实现 |
| B16 | `submit_host()` 在 core | `MOVE` | 迁入 `hyle/compute/backends/cpu.h`；类型化 callable 在返回前执行，不保留、不分配 |

CPU backend 不拥有 ParallelExecutor：那是 native/oneDNN provider 的算子内并行策略。它也不把 Platform
CPU topology 复制成第二份 device 探测；`DeviceProperties` 只报告 backend/class/name、统一内存与 queue
是否异步，详细 host topology 由 Platform 单独查询。公开依赖冻结为 `compute` 与 `mem`，无私有依赖；
产品 façade 最终包含本 factor，
但 exact `compute` runtime factor 可在 GPU-only 维护构建中独立存在。

P0 冻结成本：backend()/device() O(1) 无分配；device 枚举 O(N_cpu_devices)=O(1)；Context 通用一次分配，
CPU hook 不再分配；Queue 创建一次冷 metadata 分配；host submit O(1)，只支付通用 admission 的一次 atomic
读取，immediate completion 不分配/加锁/syscall。没有独立 P1：若通用 Queue/Completion 无法得到相同
codegen，问题归 Compute runtime 卡。

## Implementation checkpoint（2026-09-12）

- B10--B16 已按审计边界物理迁移：`cpu::backend()` / `cpu::device()` / `cpu::memory_resource()`、自然的
  `cpu::create_context()` 与 typed `cpu::submit_host()` 均由本 factor 拥有；通用 Compute 不再私藏 CPU
  factory 或 host callable；
- CPU backend 复用 Mem system resource，只实现 caller-thread ordered queue；通用执行属性由自身静态事实给出，
  不复制 Platform 的 host topology；create、
  close、drain 和 immediate completion 没有另造线程池、storage 或错误系统；`cpu::device()` 直接构造已知
  有效的小值，不在 `noexcept` convenience 中通过 `Result::value()` 引入无意义的失败/异常路径；
- Windows factor header/test、Compute 聚合、oneDNN/native 直接 consumer，以及 source/static/shared package
  consumer 已通过。静态链接还发现并修复 benchmark/test 对 CPU/native factor 的遗漏真实依赖；CPU
  backend 由真实组合者声明，不再反向挂在可接受任意 CPU-class Device 的 oneDNN Provider 上。

## Candidate（2026-09-14）

- 前置 `compute` v6 已在 `7718b163946104989c9a000f19fd01f454211d36` 通过异步生命周期、三平台、
  sanitizer、三交付、P0 与终局红队并关闭；原有前置阻塞已经解除；
- `compute_backend_cpu` 的公开头、实现和组件边在该不可变提交后没有变化，因此本 factor 与 Compute
  runtime 使用同一产品候选 `7718b163946104989c9a000f19fd01f454211d36`，后续只允许补证据和记录；
- Windows 已有 factor 定向测试、逐头和 Compute source/static/shared 消费；Linux GCC 14 的
  ASan/UBSan/TSan 与两台 Apple Silicon 的 portable Compute/Graph 回放均实际包含
  `hyle_compute_backend_cpu_test`，不以只编译 runtime 代替 factor 验证；
- P0 继续冻结为：CPU backend/device convenience 零分配；Context/Queue 只支付通用 runtime 的冷 metadata；
  caller-thread submit 不创建线程、不分配、不加锁、不做 syscall/provider search，并在返回前完成 callable。

本卡进入 `CANDIDATE`。终局只回放已声明的 factor correctness、交付和红队，不重新运行 Compute/CUDA/Graph
已经对同一产品候选签发的昂贵 P1/P2。

## 终局记录（2026-09-14）

### 验证与红队

- W1-WIN/MSVC 19.50 factor-only 配置以 `HYLE_DEV_FACTORS=compute_backend_cpu` 精确解析
  `core,util_core,error,math,platform,mem,compute_ops,compute,compute_backend_cpu`；WERROR 和 test-only
  instrumentation 开启。`hyle_compute_backend_cpu_header_self_contained_test` 与
  `hyle_compute_backend_cpu_test` 2/2 通过，后者覆盖 device/enumeration/properties、显式 allocator、
  resource identity、copy/fill、caller-thread submit、close/drain、并发 queue、provider rollback、
  Completion ownership/abandon 与异步 endpoint 拒绝；
- 同一产品候选的 Compute 终局证据已经在 Ubuntu 24.04/GCC 14 通过该 factor 的 ASan/UBSan/TSan，在
  MacBook Air 与 Mac mini（macOS 26.6.2、AppleClang 21、ARM64）通过该 factor 定向测试；Windows、
  两台 Apple 的 Compute/Graph 产品 source embedding、relocated static/shared consumer 又实际取得本 CPU
  backend。内部 factor 按构建合同不可单独安装，不虚构第四种交付；
- public/header/source/component red team 未发现全局 backend registry、静态初始化、隐式默认 Context、
  第二套 allocator/error/threadpool/topology、Graph 反向边、C/Native FFI 或 benchmark-only 入口。
  `submit_host` 只接受 nothrow callable，并在调用线程、返回前执行；CPU vtable 不做 provider search、JIT、
  syscall、隐藏线程或 backend allocation；
- P0 `PASS`：backend/device/property/resource factory 均 O(1)；CPU hook state 为空；一次 Context/Queue 冷
  metadata 由通用 runtime 拥有；每次 sequence 只有通用 admission atomic 与立即完成值。没有未决机制差异，
  因而不升级为 P1，也不拿同步 CPU queue 与异步 GPU queue做苹果对橘子计时。

### 七问

1. 第三方：本 factor 无第三方；Mem 的 mimalloc 路线不影响 `system_resource()` 身份合同。
2. 功能：显式单 CPU device、Context/ordered Queue、host resource、copy/fill、Completion 与 typed host
   escape hatch 完整；多核算子执行属于 provider，拓扑属于 Platform，均有具名 owner。
3. 复用：直接复用 Compute runtime、Error 结果和 Mem resource；没有为了避开依赖重写 Platform/Sched。
4. 性能：caller-thread 路线是 O(1) admission、零 warm allocation/lock/syscall/search；P0 通过。
5. 交付：纯 C++ factor；历史 C/Native FFI 为 N/A。
6. 构建：中央图与真实 target 都公开链接 `compute,mem`；产品 façade、factor-only 开发构建和三种产品交付
   使用同一 target，没有 CPU implementation 藏在 Compute runtime。
7. 易用：普通用户用 `cpu::device()`、`cpu::create_context()` 和 `cpu::memory_resource()`；高级 host callable
   可用 `submit_host()`，其 nothrow、立即执行和不保留参数限制在头文件与 Compute guide 中可见。

### 限制与重开条件

- 限制：v1 只有一个逻辑 host CPU device、caller-thread ordered queue，不承诺后台 CPU worker、NUMA/cluster
  device 切片或异步 CPU endpoint import；这些由 Platform/Sched/provider/Graph 的显式组合承担；
- 重开：可复现 device/context/queue/resource lifetime 或 close/drain 错误；真实 provider 证明 caller-thread
  backend seam 无法承载；公开 Compute backend 合同变化；或同合同 P0/P1 证明 material wrapper 税；
- candidate：`7718b163946104989c9a000f19fd01f454211d36`；LOC：`compute_backend_cpu` implementation
  `150` 行，自实现不含测试全库 `116,292` 行。

`compute_backend_cpu` v1 本轮 `CLOSED`。
