# Core--Obs llama.cpp 真实消费者扫描 — 2026-08

<!-- READ-ONLY CROSS-MODULE CONSUMER SCAN: COMPLETE -->

> 本页只回答：固定版本的 llama.cpp 真实推理路径是否为已经收口的 Core--Obs 八个单元带来新的
> 产品事实。它不是新的模块 closure card，不改写历史任务，也不替代
> [`module-status.json`](../../module-status.json)。没有满足重开条件时，`CLOSED` 产品代码和状态保持不变。

## 扫描合同与边界

- HYLE 扫描基线：`867700687dc982a2d452fd6a6f1550c08bf2195b`；
- consumer：本地干净的 llama.cpp
  `7584430716ee229751771ed0d6bbcb780d105eeb`；
- 唯一受审产品单元：`core`、`error`、`container_intrusive`、`util_core`、`time`、
  `platform`、`mem`、`obs`；
- `container`、`util-runtime`、`simd/math`、`sched`、`io`、`compute`、`graph` 在本页一律视为
  **尚未重构/尚未签署**。它们可以作为某个语义的正确 future owner，但这种路由不证明其当前实现完整、
  正确或高性能；
- 方法：从模型加载、prefill/decode、采样、KV 生命周期、取消/关闭和服务观测六条路径抽取 ownership、
  failure、lifetime、concurrency 与 cost shape，再与八个模块的公开能力和既有 watch 条目对账；
- 本轮是 product-non-mutating read-only scan：不新增 benchmark，不把 llama.cpp 的模型/backend 类型
  下沉为基础 API，也不因 peer API 数量扩大模块；
- 工作树中已有与本轮无关的 SIMD、依赖锁和方法文档修改，本页没有覆盖或吸收它们。

关键 consumer 文件身份：

| 文件 | SHA-256 | 本轮用途 |
|---|---|---|
| `src/llama-model-loader.cpp` | `57b011a756890a5659e08139fb17017ae29975dea67b95cd7badf13f90b76eed` | tensor/file mapping、异步 upload、load progress/取消 |
| `src/llama-mmap.cpp` | `0673e5273594690a1a7b81010b18f158f5b153f502fed161a595c941e8e5808a` | mmap advice、partial unmap、增量 mlock |
| `src/llama-context.cpp` | `053bf9298f49c658605df55815839e06ba2a2f2d1ed483e4bf4aace98570e2a8` | reserve、prefill/decode、async graph、错误/回滚 |
| `src/llama-kv-cache.cpp` | `611ebc4389ab1a8c8bc5f100246549de03cc3fc0fedd8127a298e5b6f5d59b28` | KV placement、copy/remove、backend buffer ownership |
| `src/llama-sampler.cpp` | `ade0e3fee8ae823ee1fe0c527e5c3fbe66189b0201ce42327cd6940f83883026` | URBG、distribution、可复制/事务式 RNG state |
| `tools/server/server-context.cpp` | `a79cf9e1d4a8d7c1f0ee608aa781628db403e8f59e25e731f997d0952d230e47` | slot 生命周期、取消、timing、日志、metrics |
| `tools/server/server-common.h` | `c0b744eec3faf9495171f031262c1a8bd863f8d31cff75147ee69e23d9ec6612` | request/slot stats、业务计数与 pipe lifecycle |
| `tools/server/server-queue.cpp` | `0e196113d16f130230d2cc0b232915e1f5b4f1dda184790afbfdc4986dd1fd58` | response wait、poll/cancel、drain |
| `common/log.h` | `cd065a0799ffc9a8adcb96f5188f77aec139e2520cb76316bf091ca289bb5bcb` | 单 host callback 与日志宏 shape |

## 六条真实路径的 owner 对账

| 路径 | llama.cpp 中的正常形状 | Core--Obs 可复用机制 | 不得借本页宣称完成的部分 |
|---|---|---|---|
| 模型加载 | 同步读取或 mmap；sequential/random/will-need；可取消 progress；CPU tensor 指向 mapping，offload tensor 走 pinned staging/event；load 后释放未使用 mapping 区间 | Core invariant；Error canonical/native/context；Platform file/mapping/advice/NUMA；Mem host allocation/pressure；Time elapsed；Obs progress/log | direct I/O 属 IO；pinned/device buffer、event 和 upload 属 Compute；GGUF/model schema 属引擎 |
| Prefill / decode | 启动前 reserve worst-case graph/output；按 ubatch 异步 compute/copy；下一阶段或取结果时同步；失败时删除本 ubatch 新增的 memory positions | Core contract；Error 可表达 aborted/resource-exhausted/backend-failed 并由 adapter 保留 owner detail；Time monotonic timing；Obs trace/log | graph/buffer/stream/event、连续批处理和 backend dispatch 属尚未重构的 Compute/Sched/引擎 |
| 采样 | `std::mt19937` 作为 URBG，配 `uniform_real_distribution`/`discrete_distribution`；复制 main/backend RNG state，rejected output 不提交 draw | Util-Core `Rng` 已是标准 URBG，可复制、可显式保存/恢复 state，并可直接组合标准 distributions；Core 提供 invariant | softmax/top-k/top-p、token candidate layout 和 backend sampling graph 属 SIMD/Compute/引擎；不造 `TokenRng` |
| KV 生命周期 | backend-owned KV buffers；sequence remove/copy/keep；slot placement 失败可回滚；跨 stream copy 可能是真 device copy | Mem 提供通用 host backing/pool/quota/pressure/NUMA span；Error/Time/Obs 提供失败、计时和诊断机制 | KV cell/page/prefix/fork/evict/defrag 是模型状态策略；device buffer/coherence 是 Compute，不进入 Host Mem API |
| 取消 / 关闭 | backend abort callback；queue wait 定期检查 `should_stop`；reader 析构 stop；decode 区分 aborted/alloc/failed；失败 ubatch 回滚 | Error 的 value failure + context；Time 的 monotonic deadline/remaining/timer containers；Core invariant | callback routing、late completion、bounded drain 和跨线程 cancel 归尚未重构的 Sched/IO/Compute；本页不替它们签署 |
| 服务观测 | 一个进程 logger adapter；slot/request-owned interval progress；局部累计 prompt/predict/decode/accept counters，scrape 时渲染 Prometheus | Obs v3 的 one-adapter structured log、call-site/object-owned limiter、trace plot/counter seam；Time 提供 clock domain | 指标命名、推理统计结构、窗口聚合、Prometheus/OTLP、告警和持久化归 host/未来 telemetry owner |

## 八个模块的差异裁决

这里的 `NO PRODUCT TRIGGER` 只表示本次固定 consumer 没有推翻当前产品边界；它不是一次新的 Formal
性能认证，也不表示未来引擎不会产生新需求。

| 模块 | 本次直接反证 | 剩余边界 | consumer-scan verdict |
|---|---|---|---|
| Core | llama 热路径需要 release-surviving invariant、unreachable/optimizer contract、cache layout 与 prefetch intent；现有 Core family 已覆盖。kernel target attribute 或模型常量仍由 owning kernel 证明 | crash reporter、runtime topology、kernel-specific attributes 不进入 Core | `NO PRODUCT TRIGGER`；维持 `CLOSED` |
| Error | loader 的 exception 文本和 decode 的 `aborted / alloc failed / failed` 整数都可在 HYLE 引擎边界自然映射为 `Status/Result`、canonical code、native detail 与 `WithContext`；模型/backend enum 留在 owner | 多错误聚合、第二 typed detail、stack/source correlation 仍需真实无法保真的恢复路径 | `NO PRODUCT TRIGGER`；维持 `CLOSED` |
| Container-Intrusive | 本次 llama 路径没有出现一个必须由默认 HYLE intrusive list 承载的新通用 shape；已有 stable-node timer consumer 仍覆盖当前 primitive | counted list、并发 reclamation、完整 hook/algorithm matrix 继续按既有 route/trigger，不由服务队列的业务计数反推 | `NO PRODUCT TRIGGER`；维持 `CLOSED` |
| Util-Core | llama sampling 明确验证了 URBG + 标准 distribution + 可复制/事务式 state 的真实需求；`Rng` 已满足该通用 shape。抽样算法语义没有下沉 | crypto entropy、跨 DSO 唯一 stream、token sampler 和 bulk/vectorized draw 仍不承诺 | `NO PRODUCT TRIGGER`；维持 `CLOSED` |
| Time | llama 使用 precise monotonic elapsed、steady deadline、wall timestamp、sleep/wait 与大量 timer/cancel shape；现有 clock/deadline/sleep、radix `TimerQueue` 和 `CommonTimeoutQueue` 已给出底层机制 | periodic/cross-thread callback service、request SLO、device timestamp correlation 由 Sched/IO/Compute/Obs owner 决策 | `NO PRODUCT TRIGGER`；维持 `CLOSED` |
| Platform | 一般模型加载需要 subrange mapping、range advice/prefetch、whole-map lock、文件/NUMA/VM 机制，现有公开面可承载；llama 额外使用 incremental range mlock 与 POSIX partial unmap | `range mlock / MLOCK_ONFAULT / partial unmap / prefault-collapse` 保持 `PLATFORM-WATCH-04`。当前只有一个具名 loader，尚无第二独立同语义 consumer，也没有相对 subrange mapping/`kDontNeed` 路线的 load-to-ready、RSS、page-fault material 证据 | `NO PRODUCT TRIGGER + WATCH retained`；维持 `CLOSED` |
| Mem | llama 的普通 host allocation、phase scratch、quota/pressure 与 large-span placement 均有现有机制或标准 PMR 路线；pinned staging 和 KV buffer 的 native handle/completion 证明它们不应伪装成 Host Mem allocator | KV policy、device/pinned/coherent buffer、跨 device copy 与 unified-memory policy 留给 Compute/引擎；多节点物理收益仍需真机 workload | `NO PRODUCT TRIGGER`；维持 `CLOSED` |
| Obs | v3 本来就绑定同一 llama revision 的单 logger adapter 和 slot-owned interval progress；当前源码 hash 与 closure 证据一致。server metrics 是 owner-owned业务值和 export policy，非基础 record/sink 缺口；profiling counter 已由 trace seam 提供 | async owning logger、metrics aggregation/schema/export、OTLP、storage/network 继续在 host/future owner；准入后再重开 | `NO PRODUCT TRIGGER`；维持 `CLOSED` |

### Platform 候选为何没有被藏起来，也没有直接施工

llama.cpp 的 `llama_mlock::grow_to()` 会按加载进度只锁新增 page range，POSIX loader 还会
`unmap_fragment()`；Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的
`MemoryMapping` 则提供 whole-mapping `MLOCK_ONFAULT` 和 opt-in `MADV_POPULATE_READ + MADV_COLLAPSE`。
这些事实证明该 family 不是凭空想象，也说明 HYLE 当前 `MappedFile::lock()` 的 whole/eager shape 并不
等价于全部 specialized-loader 需求。

但本轮仍不能据此改变产品：llama.cpp 是唯一具名消费者；Folly 当前是成熟 mechanism/peer 证据，
不是第二个独立 HYLE consumer；HYLE 已有 subrange map 和 POSIX range `Advice::kDontNeed` 组合路线；
还没有证明新 API 会在正常模型加载上实质改善 load-to-ready、RSS 或 page faults。因而正确结果是保留
`PLATFORM-WATCH-04` 及其精确提升条件，而不是写 benchmark 迎合一个 syscall，或反过来宣称能力已完整。

## 依赖、模块调用与当前产品身份

中央组件边保持无环且与公开 include 方向一致：

```text
core
├─ error
├─ container_intrusive ─ time
├─ util_core
├─ error + time ─ platform
├─ core + error + platform + util_core ─ mem
└─ core + error ─ obs
```

- Platform、Mem 和 Obs 的 recoverable public failure 继续走 Error；Core/Util-Core/Time 的纯
  header/contract path 没有另造 TLS error；
- Time 复用 Container-Intrusive 的 safe stable-node list；Obs 的 limiter 只需标准 `steady_clock`，
  不为一个 clock read 增加 Time component edge；
- Mem 复用 Platform 的 VM/NUMA mechanism，并保留 Mem 的 allocation/lifetime/policy owner；
- Core、Error、Container-Intrusive、Util-Core、Time、Platform 没有产品路径第三方实现；Mem 的
  system allocator mechanism 锁定 mimalloc v3.5.0 / `18b08671c9302247bfb682286e6bf3cc1773f801`；
  Obs 仅在 trace ON 时使用 Tracy v0.14.1 / `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c`。
  llama.cpp、Folly 等本页来源只作 consumer/peer evidence，不链接进产品。

八个 candidate 到扫描基线的产品路径检查结果：Core、Error、Container-Intrusive、Time、Platform 和
Obs 无 committed drift；Util-Core 只有 RNG/Ref/type-name 合同注释校准，Mem 只有 `alignment.h` 的
旧容器名称注释校准，均不改变 API、算法、layout 或 codegen。工作树当前 dependency-lock diff 只涉及
SIMD ported-source hash，不改变 mimalloc/Tracy 身份。

## 结论与合法下一动作

- 跨模块 consumer differential：`CONFIRMED`；八个模块均无 `PRODUCT_TRIGGER`，不改变
  `module-status.json`，产品实现改动为 0；
- 这次补强了此前不均匀的 llama.cpp 直接证据：Util-Core 得到真实 sampling state shape；Time 得到
  serving/decode clock/cancel shape；Core/Error/Intrusive 得到明确的无新增基础形状结论；Platform/Mem/Obs
  的 owner 边界被逐路径复核；
- 既有工程性能结论只保留各 closure task 原本限定的 candidate、机器、contract 和 margin。本页没有
  运行新性能 workload，因此不续签 Formal、跨平台固定点值或“所有操作 SOTA”；
- 下一次合法动作不是继续无界回看这八个模块。只有真实 HYLE inference consumer、第二独立 consumer、
  correctness/ownership/交付缺陷，或同合同 material performance evidence 命中某个 reopen trigger，
  才定向重开相应 family；
- 其他模块依用户指定仍按“尚未重构”处理。本文出现的 IO/Compute/Sched/SIMD/Container/Graph route
  只是 owner 边界，不能在它们未来 closure card 中当成完成证据。
