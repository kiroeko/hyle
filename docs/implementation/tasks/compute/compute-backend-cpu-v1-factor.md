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
- 直接依赖 / 消费者：候选公开依赖 Compute runtime，私有复用 Platform/Mem；直接消费者为
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
CPU topology 复制成第二份 device 探测；properties 复用 Platform 已关闭事实。公开依赖冻结为 `compute`，
私有依赖为 Platform；Mem 通过 Compute 的公开 storage contract 到达。产品 façade 最终包含本 factor，
但 exact `compute` runtime factor 可在 GPU-only 维护构建中独立存在。

P0 冻结成本：backend()/device() O(1) 无分配；device 枚举 O(N_cpu_devices)=O(1)；Context 通用一次分配，
CPU hook 不再分配；Queue 创建一次冷 metadata 分配；host submit O(1)，只支付通用 admission 的一次 atomic
读取，immediate completion 不分配/加锁/syscall。没有独立 P1：若通用 Queue/Completion 无法得到相同
codegen，问题归 Compute runtime 卡。

## Implementation checkpoint（2026-09-12）

- B10--B16 已按审计边界物理迁移：`cpu::backend()` / `cpu::device()` / `cpu::memory_resource()`、自然的
  `cpu::create_context()` 与 typed `cpu::submit_host()` 均由本 factor 拥有；通用 Compute 不再私藏 CPU
  factory 或 host callable；
- CPU backend 复用 Platform properties 与 Mem system resource，只实现 caller-thread ordered queue；create、
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
