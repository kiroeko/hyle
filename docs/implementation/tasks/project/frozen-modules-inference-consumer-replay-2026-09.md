# 已冻结模块的真实推理消费者回放（2026-09）

<!-- READ-ONLY CROSS-MODULE CONSUMER REPLAY: ACTIVE -->

> 本任务把最新真实推理引擎的消费形状回放到已经 `CLOSED + FROZEN` 的 HYLE 单元。它是发现任务，
> 不是第二套模块状态，也不因“参考库有更多 API”自动授权改产品。只有具名事实满足既有 reopen trigger，
> 对应组件才另开自己的 `REOPENED -> SCOPED -> AUDITED` 产品轮。

## 目标、身份与边界

- HYLE baseline：`840ab249a2f4257ebcff5bafe144e7bc7832bb40`；本任务建立时的 canonical checkout 为
  `C:\public\project-hyle\hyle`；
- owner / start date：Codex / 2026-09-06；
- 目标：逐项确认真实模型加载、prefill、decode、KV 生命周期、continuous batching、取消/关闭和
  serving 观测是否暴露已有冻结单元的能力、自然接口、内部复用、性能 workload 或交付缺口；
- 本 goal 的唯一产品范围是 15 个真正冻结 factor：`core`、`error`、`container_intrusive`、
  `util_core`、`time`、`timer_queue`、`mem_reclaim`、`obs`、`math`、`platform`、`platform_thread`、
  `platform_fs`、`mem`、`mem_vm`、`simd`；
- 有新能力时，先通过语义擦除、owner、生命周期和准入门，再尽量在正确模块实现；不能复制模型、tensor、
  token、backend 或服务策略；
- 性能方向由真实热路径和成本结构确定。推理仓库用于确定 workload/provenance，不自动成为同合同 peer；
  计时前必须先冻结接口、语义、工作量、ownership、并发、计费边界和 delivery shape；
- `simd` 已由 v6 对同一四组 consumer 完成逐能力回放，本任务只做 reference drift 差异和直接消费者回放，
  不重跑 422-scope 全矩阵；
- `container`、`util`、`sched` 尚未按本轮标准冻结，明确排除；`io` 保持现有 `SCOPED` 且暂停，
  `compute/graph` 保持 `QUEUED`。这些模块只作为需求和 owner 反例，不在本项目回放中修改；
- Build/package 作为每一波的横向交付合同复查，不伪造一个可链接模块状态。

建立任务时存在的 Container/Mem/Util performance replay、`roadmap.md` 与 `status.md` 未提交改动均为
外部既有工作；它们不属于本任务 baseline，不得被覆盖、吸收或提交。

## 锁定推理证据

2026-09-06 对五个 checkout 执行了 clean-tree 检查和远端 `HEAD` 核对；落后的干净仓库只做
`git pull --ff-only`。本轮发现使用以下完整 identity：

| evidence group | clean commit | role |
|---|---|---|
| standalone ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | CPU/backend 基础机制、model-independent reference |
| llama.cpp | `73a43d1f69345aee8bb186ef4b3172cef892f2e5` | 模型加载、CPU/统一内存推理、采样与单机服务 journey |
| ds4 | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` | 紧凑量化 CPU/统一内存引擎及直接热循环 |
| vLLM | `569adb5a9780f9c02d22a6b29826acf711512356` | batching、KV block/offload、runtime 与 custom-op journey |
| SGLang | `5bebe7a033ab57751fac3c9d8db219da62ca8c72` | serving、MoE/kernel、调度、KV/cache 与观测 journey |

Python、Rust、CUDA、Triton 或 vendor kernel 中的业务对象不会按名字下沉为 C++ 基础 API。它们仍可证明
某个模型无关的 ownership、storage、synchronization、mapping、error 或 observability 机制在真实系统中
重复出现。

## 消费路径抽取合同

每个 finding 必须记录以下字段，缺一项就不能成为产品 trigger：

| field | required evidence |
|---|---|
| exact source | repository commit、文件、symbol/调用链，而不是搜索命中数 |
| task shape | 创建、正常操作、组合、关闭/取消和销毁全过程 |
| contract | ownership、lifetime、failure、concurrency/progress、normal size 和 worst case |
| frequency | 下列 hotness 等级及其证据；不得凭直觉写“热” |
| semantic owner | HYLE factor、standard/mature dependency、Compute/Graph/engine 或明确不做 |
| current route | `COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE` |
| cost/performance | 算法档、allocation/copy/lock/atomic/syscall/dispatch/working set 与自然计费边界 |
| generalization | 去掉模型/token/backend 名称后仍自然，并有第二消费者或成熟 peer 重复 |

热度只使用三档：

- `OBSERVED_HOT`：仓库自带 profile、benchmark、trace 或明确性能注释把它列为关键路径；
- `STRUCTURAL_HOT`：从代码可证明它按 token/layer/block/batch/request 重复，但不伪造耗时百分比；
- `CONTROL_PATH`：启动、配置、模型加载一次性步骤、错误或关闭路径；它仍可因延迟、RSS、可靠性重要，
  但不能用纳秒 primitive 榜替代完整 journey。

四个引擎对同一通用机制的独立实现或绕行优先级高于单个仓库的 feature；代码拓扑可以证明重复频率，
不能替代 profile 证明 materiality。

## 性能裁决与防止苹果对橘子

1. 先做 P0：复杂度、遍历次数、临时存储、allocation/copy、锁/原子、syscall、动态分发和 codegen。
2. 只有 `OBSERVED_HOT/STRUCTURAL_HOT` 且 HYLE route 可能产生 material 差异时才建立 P1；先从真实调用
   抽取版本化输入和自然公开/内部 seam，不围绕内部阈值造 benchmark。
3. P1 每格都必须核对工作量、结果、ownership、failure、thread-safety/progress、冷热状态、线程/亲和、
   ISA、delivery、setup/teardown 和 charged phase。同合同强 peer 最多两个；不等价就写
   `NOT_COMPARABLE`，不得用总平均或 5% margin 把差异抹平。
4. 真实 trace/profile 足以稳定重放完整阶段时才升 P2。P2 不能拿不同模型、量化、backend、batch、KV、
   输出质量或计费边界直接比较 tokens/s。
5. 固定 material loss 只做一次有界机制诊断：能指向算法/实现/API 成本则形成 trigger；只能随地址、顺序、
   code placement 翻转则记录仪器边界，不改公共 API、layout 或常态 metadata。
6. 已冻结 workload 不得因结果难看而改形状。consumer repository 不是 peer 时，只用它锁定 workload；
   peer 必须另行锁定版本和等价合同。

## 分波执行与依赖顺序

每波先完成只读 differential；只有 `PRODUCT_TRIGGER` 才按依赖顺序重开单个组件。上游产品改动完成后，
只回放直接消费者，不批量重开全库。

| wave | factors | inference emphasis | build/package replay |
|---|---|---|---|
| A | core, error, container_intrusive, util_core, time | 热循环编译合同、错误传播、stable-node/queue 链、RNG/resource、clock/deadline | header ownership、source/static/shared 与零/最小依赖 |
| B | timer_queue, platform, platform_thread, platform_fs, mem_reclaim | timeout/cancel、topology/ISA、placement、mapping/read、read-mostly reclamation | factor façade、OS 条件源码、传递系统库 |
| C | mem, mem_vm, obs, math | allocation/pool/quota、large mapping/load-to-ready、structured hot observation、scalar sizing | mimalloc/Tracy/OS/vendor lock、三交付 |
| S | simd delta only | ggml/ds4/vLLM/SGLang 新增 dtype、row/kernel primitive 与 fixed-ISA consumer | Highway/xsimd lock、target flags、无成本路由 |

波内每个 factor 的合法结果只能是：

- `CONFIRMED`：当前 candidate 能完成本轮准入 journey，且性能/接口/交付没有新 trigger；
- `METHOD_GAP`：未发现产品问题，但当前证据或工具不足；产品状态保持不变，另记有限 tool/evidence task；
- `PRODUCT_TRIGGER`：记录精确 consumer、缺陷、owner 和重开条款，随后才更新 registry 并建立独立 closure
  card；不能在本项目页直接施工。

## 预算、交付物与停止条件

- 小 factor differential 默认 45 分钟，中型 90 分钟；一个 finding 最多一次源码机制诊断。超时只允许
  `METHOD_GAP`，不允许无界调参；
- 每波交付：reference identity、consumer-path ledger、能力差异、hot-path map、性能 comparability、
  dependency/reuse/build 复核和逐 factor verdict；
- 每个合法 trigger 单独拥有 FIX 集、candidate、验证、红队、三交付和文档，不共享模糊“全库候选”；
- 项目停止条件：上述 15 个 frozen factor 均取得当前 reference identity 下的一个合法 verdict，所有
  `PRODUCT_TRIGGER` 已转入各自状态机；SIMD drift 已明确为 confirmed 或独立 trigger；
- 全轮仍执行 `python scripts/check-module-closure.py` 与 `python scripts/check-docs.py`。只有实际改变
  `module-status.json` 的子轮才保存 previous registry 并用 `--previous` 校验。

## Wave A：foundation prime differential

本波只读回放绑定本页锁定的五个 revision。以下 source path 均相对于对应 reference checkout；它们
证明任务形状和重复频率，不把 Python/Rust/Torch API 本身下沉为 HYLE API。

### Core

- exact sources：ggml `include/ggml-cpu.h:ggml_vec_dot_t` 的 restrict contract 与大量
  `GGML_ASSERT`；vLLM `csrc/cpu/cpu_attn_rvv.hpp:gemm_macro_rvv_fma_Mx8_Ku4` 的 restrict/force-inline
  kernel contract、`csrc/attention/dtype_bfloat16.cuh` 的 unreachable route；vLLM
  `csrc/torch_utils.h:TORCH_UTILS_CHECK` 以及 ds4 `cuda/mmq/vendors/hip.h:__trap`；
- task/contract：kernel 编译提示必须零运行时状态；fatal invariant 在失败路径终止；target/device
  dispatch 与 tensor validation 由 SIMD/Compute/backend owner 承担。它们按 element/kernel 调用为
  `STRUCTURAL_HOT`，但 Core 只提供 compile-time lowering；
- route/cost：`HYLE_RESTRICT`、force-inline/noinline/cold/likely/unlikely、assume/unreachable、assert
  field contract、prefetch/cache constants 与 environment facts 已覆盖模型无关部分。所有正常路线均为
  inline/compile-time，零 allocation/lock/syscall/dispatch；GPU/device annotation 不能伪装成 host Core
  缺口；
- generalization/verdict：四组 consumer 重复的是现有编译合同，没有第二个新的 host primitive，也没有
  新 codegen 或交付失败事实，故为 `CONFIRMED`。性能保持 P0 structural；不为单个 kernel 新建宏。

### Error

- exact sources：llama.cpp `src/llama-context.cpp:llama_context` 的 owning diagnostic exceptions；ggml
  `src/gguf.cpp:gguf_init_from_file_impl` 的 errno + operation text；ds4 `ds4_gpu_args.h` 的 status + caller
  buffer 与 `ds4_distributed.c:dist_connect_endpoint_once` 的立即 errno 捕获；vLLM
  `csrc/fs_io.cpp:atomic_copy` 的 errno preservation 和 `csrc/torch_utils.h` 的 boundary check；
- task/contract：模型加载/配置失败是 `CONTROL_PATH`；Result success、registry borrow 和 scheduler first-error
  handoff 是正常传播路径。owner 必须保留 canonical code、owning message、native integer 与逐层 context，
  backend/Torch/Python exception ABI 不进入 Error；
- route/cost：`Status`/`Result<T>`/`Result<T&>`、`WithContext`、errno/Win32/WinSock/HRESULT/backend native
  detail 和 `std::error_code` adapter 已覆盖。HIP/Metal 命中尚未证明现有 canonical + native integer +
  message 无法恢复，因此继续受 `ERROR-WATCH-01` 管理；不按 backend 名单扩 enum。现有 v5 P1 已对
  success、owning first-error 和 non-owning lookup 同合同路径全部通过；冷失败能力保持 P0；
- generalization/verdict：没有发现需要 Error 拥有的 multi-error、exception registry 或第二 native field，
  也没有固定 material regression，故为 `CONFIRMED`。

### Container-intrusive

- exact sources：vLLM `vllm/v1/core/kv_cache_utils.py:KVCacheBlock` 与
  `FreeKVCacheBlockQueue`；`vllm/v1/core/block_pool.py:BlockPool.get_new_blocks/touch/free_blocks`；
- task/contract：每个 KV block 是 caller-owned stable node。自由队列以双哨兵维持 LRU/局部性次序，要求
  两端插入/弹出、任意已知节点 O(1) 删除、O(1) count、零 steady-state allocation；批量取出/归还按
  block 数量为 O(n)。`touch/free/get` 随 block/request 重复，属于 `STRUCTURAL_HOT`；eviction policy 仍归
  engine；
- route/cost：`CountedIntrusiveList` 已提供相同 sentinel/member-hook/size/iterator_to/erase/front/back，
  已链接的批量链可用 O(1) splice；逐节点 detach 必然 O(n)。现有 `counted-membership-4096` P1 对
  preallocated node 的插入、每次 size 和任意删除与 Boost constant-time-size list 同合同，恰好覆盖新增
  consumer 的成本机制；不需要 KV 专名、`popleft_n` 返回 owning vector 或常态新 metadata；
- generalization/verdict：真实推理消费确认既有自然形状，复杂度与证据合同一致，故为 `CONFIRMED`。

### Util-core

- exact sources：ds4 `ds4.c:sample_rng_next/sample_rng_f32` 及 sampling/speculative-accept 调用；llama.cpp
  sampling 中的 `std::mt19937`/standard distributions；HYLE 内部 Mem/Sched 的 `Ref`/`SpinLock` 和资源
  owner 路线作为直接 foundation consumer；
- task/contract：显式 seed/state 的 scalar RNG 在采样决策中按 token 使用，属于 `STRUCTURAL_HOT`；并行
  stream、bulk draw 或 crypto 不是该任务。资源/borrow/callback 路线分别要求明确 lifetime、零隐藏
  ownership 和同步调用；
- route/cost：`Rng` 已是标准 UniformRandomBitGenerator，提供可复现 state、xoshiro256++、24/53-bit
  float、无偏 bounded integer、shuffle、jump/long-jump 和 thread convenience；steady draw 全 inline、
  zero allocation/lock/division（bounded rejection rare），比 ds4 私有 xorshift64* 覆盖更完整的通用合同。
  单次 draw 相对完整 logits/filter scan 没有已证明 material 独立预算，因此保持 P0，不制造 RNG 微榜；
  Ref/SpinLock/FunctionRef 的既有同合同 P1 仍覆盖其当前热消费者；
- generalization/verdict：未出现 bulk RNG、weak ownership 或 parking mutex 的准入证据，故为
  `CONFIRMED`。观察清单已移除早已实现的 conditional scope/resource 假缺口。

### Time

- exact sources：ds4 `ds4.c:now_sec`、`ds4_kvstore.c:kv_now_sec`、`ds4_server.c` 的 monotonic/absolute
  realtime wait；vLLM `vllm/v1/core/sched/scheduler.py:Scheduler.schedule` 与
  `vllm/v1/worker/gpu_worker.py` deadline；SGLang
  `python/sglang/srt/managers/scheduler_components/metrics_reporter.py:_record_scheduler_time` 的
  monotonic/process CPU accounting，以及
  `python/sglang/srt/observability/req_time_stats.py:calibrate_time_diff/convert_time_to_realtime_ns`；
- task/contract：elapsed/deadline 必须 monotonic，pthread absolute wait 可需要 native realtime interop，
  utilization 需要 process/thread CPU domain；这些按 request/stage 为 `STRUCTURAL_HOT`。SGLang 还先记
  monotonic stage，再以定期校准 offset 导出 Unix trace；NTP、跨线程/进程传播和精度是独立合同；
- route/cost：typed monotonic/realtime、Instant、saturated deadline/remaining、native interop、coarse/
  profiling clock、thread/process CPU clock 和 absolute sleep 已覆盖前三类；现有
  `host-phase-deadline-64` P1 正好计费成组 monotonic sample + elapsed/deadline。所有 clock 原语 O(1)、
  zero allocation；OS clock syscall/vDSO/QPC 由选定精度决定。cross-clock correlation 目前只有 SGLang
  一个具名 consumer，Tracy/Seastar/Folly 当前 pinned source 未给出同形状公共基础 API，不能据此重开；
- generalization/verdict：现有产品为 `CONFIRMED`；cross-clock snapshot 记为 `TIME-WATCH-05`，等待第二
  consumer/peer 与误差合同。观察清单同时纠正 ManualClock/ClockRef 已完成的旧描述。

### Wave A dependency / build replay

中央 manifest 与真实 target 都保持 `core -> {container_intrusive, util_core, error, time}` 的单向 prime
边（箭头表示被依赖到依赖者）；Core/header-only factor 无伪 runtime，Error/Time compiled owner 只链接
声明的 Core，两个 split factor 只由 Container/Util façade 聚合。公开 include owner validator、exact-factor
开发入口和 source/static/shared package smoke 已由当前 build contract 承载；本波没有产品/header/edge
变化，不继承 reference engine 的 Torch/Python/Rust 或 backend 依赖。W1-WIN / CMake 4.4.0 / VS 18 的
fresh `HyleCiComponentContract.cmake` 于 2026-09-06 返回 `HYLE component contracts passed`；docs 与
module-closure validators 同轮通过。

## 执行账本

| wave / factor | exact consumer paths | capability/hot-path result | performance method | verdict / next action |
|---|---|---|---|---|
| references | five clean commits above | remote identities locked | N/A | `COMPLETE` |
| A / core | ggml vec-dot contract；vLLM RVV/dtype；ds4 HIP trap | compile/assert/cache facts `COMPLETE` | P0 inline/codegen | `CONFIRMED` |
| A / error | llama load；gguf errno；ds4 buffers；vLLM fs/Torch boundary | value/native/context/borrow `COMPLETE` | v5 success/failure P1 + cold P0 | `CONFIRMED` |
| A / container_intrusive | vLLM KV free queue + BlockPool | counted stable-node free-list `COMPLETE` | existing counted membership P1 same mechanism | `CONFIRMED` |
| A / util_core | ds4/llama scalar sampling RNG；HYLE Mem/Sched resource paths | scalar RNG/resource/borrow `COMPLETE` | RNG P0；existing Ref/Spin/FunctionRef P1 | `CONFIRMED` |
| A / time | ds4/vLLM deadlines；SGLang stage/CPU clocks | clocks/deadlines `COMPLETE`；correlation `WATCH` | existing phase/deadline P1 + clock P0 | `CONFIRMED`；`TIME-WATCH-05` |
| B / timer_queue | vLLM P2P pending lookup；SGLang deferred KV release | absolute monotonic arm/cancel/advance `COMPLETE`；callback/periodic policy stays Sched/IO | existing Seastar same-contract P1 | `CONFIRMED` |
| B / platform | vLLM per-node CPU memory budget/usage；SGLang rank/node binding | node id/placement `COMPLETE`；per-node memory fact `MISSING` | new query P0 + old host-planning P1 holdout | `PRODUCT_TRIGGER`；Platform v10 |
| B / platform_thread | vLLM CPU worker placement；SGLang CPU affinity/name process setup | affinity/name/priority/wait `COMPLETE`；GPU-to-node policy routed Compute/engine | Parker P0 unchanged | `CONFIRMED` |
| B / platform_fs | vLLM direct vectored disk offload；ds4/llama temp publish/model mapping | sync buffered/path primitives `COMPLETE`；direct/async routed IO，mapping routed Mem-VM | current warm-file P1 | `CONFIRMED` |
| B / mem_reclaim | vLLM scheduler fence frees；SGLang GPU-event/deferred release；HYLE Obs snapshots | host read-mostly grace period `COMPLETE`；device/scheduler fences routed owners | current Folly RCU P1 | `CONFIRMED` |
| C | pending | pending | pending | pending |
| excluded / container, util, sched | 不在本 goal | 尚未按本轮标准冻结 | N/A | 后续独立重构，不在此偷渡 |
| S / simd drift | pending | pending | pending | pending |
