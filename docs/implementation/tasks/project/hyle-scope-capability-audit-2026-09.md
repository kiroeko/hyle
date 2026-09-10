# HYLE 范围重定标与全组件能力扫描（2026-09-10）

> 终态：`PROJECT AUDIT: CLOSED`。本记录冻结新的产品准入边界，并对 Compute/Graph 之外 27 个已实现
> factor 与 build/package 做一次只读能力扫描。它不修改
> [`module-status.json`](../../module-status.json)，不授权产品施工，也不续签既有 correctness、性能或
> 三交付证据。

## 1. 范围与基线

- HYLE baseline：`beea347170158519deb648ca8f587e4a839df944`，Windows canonical checkout 干净；
- 自动范围：registry 中除 `compute`、`graph` 外全部 27 个 factor；build/package 作为横向工程单元另列；
- 输入：公开头、architecture/user 负空间、当前 terminal record、既有跨产品纵切审计，以及下列本地
  consumer/peer 快照；
- 非目标：不设计 Compute/Graph，不写产品代码，不测性能，不把 `WATCH` 变成 backlog。

本轮使用的主要 clean Git 身份如下；`slate-render/cycles` 是无 Git 身份的本地源码快照，只作递归资产
观察的辅助证据。

| 来源 | commit |
|---|---|
| llama.cpp | `e71b80510c848c00175924ecf3c40333ccae8eb5` |
| ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` |
| ds4 | `f62ca29a308724cde5bc99134ede19104b2a3260` |
| vLLM | `51da0ca66c8065619c79e35dff97aa99aeaf5644` |
| SGLang | `dcebe8c4733a5cb802fb679e0c8fbb886e2801b0` |
| Folly | `65749da412d92fe27d87ae45a8ef670720712858` |
| Seastar | `78f64e2317b19c7e34ba61f5e8ea85f083eca1da` |
| Tokio | `060cc4e46aedc40b0a44dcbc31deb0df39abdb4e` |
| oneTBB | `e00c68dcac0e83947c9b54f6d5304c0616c9639f` |
| libuv | `3e606bc30a842701f8d12007d3ade662d258b76b` |
| nginx | `5c812c034e6cb10368f9e98cf05b6d0a01f27f1f` |

## 2. 本轮边界

权威长期定义已经写入[项目范围](../../../architecture/hyle-scope.md#可执行的产品边界)。本轮实际执行的
判据是：语义通用、需求可复核、HYLE 有集成价值、owner/无环依赖成立、承载可持续、成本可隔离、公开
旅程自然，七项必须同时成立。

因此 HYLE 的停止线不是“只收最底层函数”，也不是“让用户自行组合所有东西”：

- 高性能原子机制属于 HYLE；
- 拆开后会让每个 consumer 重写易错 lifetime/cancel/backpressure/teardown 状态机的通用机制闭包也
  属于 HYLE；
- domain policy、schema、routing、auth、model/KV/sampling、material 与部署策略属于上层；
- 成熟库能自然完成任务时使用 `BORROW` 或可执行的 `ROUTED`，不靠薄包装增加表面能力数。

下文 verdict 含义固定：`NO_TRIGGER` 表示本轮没有新的已准入缺口，不等于重新验证模块；
`PRODUCT_TRIGGER` 表示出现足以建立后续 closure card 的新事实；`WATCH` 仍缺晋级证据；`ROUTED` 表示
具名标准库/成熟库或上层 owner 是完整路线。

## 3. 27 个 factor 的扫描结果

### 最小地基、时间与平台

| factor | 当前可用能力与自然路线 | 本轮 verdict |
|---|---|---|
| `core` | 编译/平台宏、布局/cache primitive、assert/assume/restrict/unroll 与最小版本合同 | `NO_TRIGGER`；crash dump/minidump 是 Obs/宿主路线 |
| `container_intrusive` | caller-owned list/count/forward/hlist 与 sequence algorithm | `NO_TRIGGER`；多索引/tree/hash/policy matrix `ROUTED` 到 Boost.Intrusive |
| `util_core` | non-owning callable、scope guard、resource owner、RNG、低争用 spin primitive | `NO_TRIGGER`；公平/parking mutex 归 Sched/Platform |
| `error` | Status/Result、context、OS/native adapter、引用结果与有界格式化 | `NO_TRIGGER`；聚合策略、backtrace/crash collection 归 owner/Obs |
| `math` | 标量 checked sizing、整数/bit/浮点补位 | `NO_TRIGGER`；lane、tensor、quant/fused math 归 SIMD/Compute |
| `time` | monotonic/realtime/profiling clock、deadline、Unix/native interchange | `WATCH`：paired monotonic/realtime correlation 仍只有有限 consumer 证据 |
| `timer_queue` | caller-owned arbitrary/common-timeout queue，运行时 callback 由 Sched/IO 拥有 | `NO_TRIGGER` |
| `platform` | CPU/cache/topology/frequency/NUMA/host-memory/process facts、VM/env/entropy/dynlib | `NO_TRIGGER`；managed subprocess 不是该 prime factor，见 G04 |
| `platform_thread` | thread owner、name/affinity/priority/ISA permission、wait/Parker | `NO_TRIGGER` |
| `platform_fs` | 同步 file/directory/link/lock/random-access/atomic filesystem mechanism | `NO_TRIGGER`；async/recursive stream 归 IO |

### 内存、观测、SIMD 与容器

| factor | 当前可用能力与自然路线 | 本轮 verdict |
|---|---|---|
| `mem_reclaim` | read-mostly epoch/QSBR 回收 | `NO_TRIGGER`；需要 stalled-reader progress 时 hazard-pointer family `ROUTED` 到 Folly 等成熟实现 |
| `mem` | allocator/PMR/arena/pool/heap/resource、Buffer/CompositeBuffer、range/lifetime planning、quota/stats | `NO_TRIGGER`；device/pinned/unified storage 归 Compute provider |
| `mem_vm` | anonymous/file/shared/large mapping、residency/prefix/range control | `NO_TRIGGER`；共享内存通信协议不由 mapping owner 冒充 |
| `obs` | 同步 structured log seam、owner/concurrent metric values、Tracy trace | `NO_TRIGGER`；async file writer、rotation、registry/exporter `ROUTED` 到宿主 logger/telemetry 或未来 `hyle-service` |
| `simd` | HYLE/Highway 路由的模型无关 lane/buffer primitive 与 vendor escape hatch | `NO_TRIGGER`；quantized tensor operator/fusion 归 Compute |
| `container` | sequence/hash/tree/cache/bitset/sparse storage 与 allocator-aware ownership | `WATCH`：generation-safe slot map 尚缺第二个稳定 stale-handle contract |
| `container_concurrent` | 并发 map/set 与 reclamation 组合 | `NO_TRIGGER` |
| `container_queue` | bounded/unbounded queue 与明确 progress/backpressure storage | `NO_TRIGGER`；task fairness 不能由 queue 猜测 |
| `container_text` | byte/ASCII/hex/escape、UTF validate/count/convert 与 lexical path | `PRODUCT_TRIGGER`：缺 G01 Base64/Base64URL；Unicode normalization/casefold/grapheme `ROUTED` 到 ICU/utf8proc |
| `util_runtime` | startup/freeze registry、owning creator、demangle | `WATCH`：plugin unload/quiesce 仍缺两个 native owner 的完整旅程 |

### 调度与 I/O

| factor | 当前可用能力与自然路线 | 本轮 verdict |
|---|---|---|
| `sched` | Fiber/M:N、work stealing、typed task/join、bulk/blocking、cancel、timer/sync/channel/shutdown | `PRODUCT_TRIGGER`：缺 G02 dynamic task scope；task service class 与 task-local context 留 `WATCH` |
| `io_core` | per-worker proactor、file/TCP/UDP/local/DNS/signal、deadline/cancel、borrowed scatter/gather | `PRODUCT_TRIGGER` 的依赖 owner：G06 需要窄 file-to-stream primitive；owning segmented chain 留 `WATCH` |
| `io_tls` | provider-neutral client/server TLS、session、ALPN、正常 close | `NO_TRIGGER` |
| `io_http` | HTTP/1 framing、head/body/streaming/SSE 与 connection lifecycle | `PRODUCT_TRIGGER`：G04 需要显式 101/残留字节/transport hand-off，router/auth/schema 仍属 `hyle-service` |
| `io_http2` | nghttp2-backed session/stream/HPACK/flow control/GOAWAY | `NO_TRIGGER`；gRPC schema/policy 在上层 |
| `io_transfer` | local/HTTP(S) transfer、resume/verify/temp/atomic publication | `PRODUCT_TRIGGER`：缺 G06 file-to-socket zero-copy fast path 与诚实 fallback |
| `io_watch` | exact file/direct-child bounded change stream、overflow/rescan/cancel/close | `PRODUCT_TRIGGER`：缺 G05 recursive tree watch；debounce/reload policy 仍在上层 |

### build/package

build/package 已能表达 factor DAG、产品 façade、source/static/shared、锁定依赖和共享源码 cache。本轮没有
发现新的交付能力缺口；新 factor 只有在各自 closure card SCOPED 后才能登记。发现并当场修正了一项
文档事实错误：`simdjson`、FlatBuffers、Cap'n Proto 当前都不是 HYLE 构建依赖，不能写成“已 vendored/
借核”；它们现在明确为消费方 `ROUTED` 路线。

## 4. 已确认的六项产品缺口

| ID | 能力与最小自然形状 | 证据与承载裁决 | 预计 owner / 依赖影响 |
|---|---|---|---|
| G01 | caller-buffer Base64/Base64URL encode/decode；strict padding/alphabet、size query、typed error offset | llama.cpp、ds4、vLLM、SGLang 均有真实多模态/协议消费；现有 simdutf v9.1 已含 SIMD algorithm core。`BORROW`，HYLE 拥有窄语义/API | 重开 `container_text`；默认 UTF 路径零成本 |
| G02 | 动态 structured task scope：add/spawn、cancel、join/drain，析构不得静默留下超出 scope 的 child | HYLE `JoinHandle` drop 明确 detach；Folly `AsyncScope`/`CancellableAsyncScope` 与 Tokio `JoinSet` 证明成熟形状。`SELF` | 重开 `sched`；opt-in owner，不给每个 task 增加 scope metadata |
| G03 | managed subprocess：argv/env/cwd、stdio ownership、async wait、deadline/cancel、process-tree terminate、reliable reap | llama.cpp MCP、ds4 agent、vLLM 与 SGLang 均重复实现 process lifecycle。`SELF` OS adapter，不复制某语言的全部 `ProcessStartInfo` | 新独立 `process` 产品/factor，向下依赖 Platform/IO-core/Sched/Time/Error；不塞进 Platform prime |
| G04 | WebSocket client/server upgrade、frame/message、fragment/control/close、limits、deadline/cancel/backpressure | vLLM 与 SGLang realtime serving、ds4 WebSocket client 是具名旅程。协议机制属于 HYLE，route/auth/message schema 不属于 | 新 `io_websocket` factor；重开 `io_http` 提供安全 upgrade/handoff；复用 G01/TLS/IO/Sched |
| G05 | recursive directory tree watch：相对路径、动态子目录、rename、overflow→rescan、cancel/close；不承诺无损 event log | slate-render 的 Linux watcher 有递归资产树需求；libuv 在支持平台提供 recursive mode，HYLE macOS 当前还主动过滤 descendant。`SELF` 跨平台机制 | 重开 `io_watch`；显式 opt-in，不改变 direct-child 默认成本 |
| G06 | async file-to-stream send：offset/length/progress、short send、cancel/drain；支持时 sendfile/TransmitFile，其他情况明确 fallback | nginx 与 libuv 都有成熟跨平台 sendfile/fallback；资源流送纵切需要避免无谓 user-space copy。`SELF` OS fast path | 重开 `io_core` 与 `io_transfer`；TLS 只有 provider 真支持时走 zero-copy，否则可见 fallback |

这些是**能力 trigger**，不是已经审定的 API 名或 FIX 集。每项后续都必须单独创建 record，按依赖顺序
完成 `SCOPED -> AUDITED`，冻结真实 consumer journey、负空间、三平台语义和有限实现集后才能改代码。
G03/G04 在登记新 component ID 前同样不能施工。

## 5. 未晋级项与停止线

| 观察项 | 当前路线 / 为什么不施工 | 晋级所需新事实 |
|---|---|---|
| task service classes / priority lanes | llama.cpp/vLLM/SGLang 的 request priority 仍可能是应用 queue policy；Seastar/oneTBB 的 runtime 语义也不相同 | 一个 C++ runtime consumer 的 queueing trace，并冻结 class 数、shares/strict priority、公平、starvation、继承和默认零 metadata 合同 |
| task-local execution context | CancelScope 已有专用 context；一般 request/trace/allocator context 可借显式参数传递 | 两个 C++ consumer 证明显式传递造成重复错误，且能给出 fiber migration、nesting、destruction 与零默认成本合同 |
| owning segmented byte chain | 当前可用 `CompositeBuffer` + borrowed IO buffer sequence；尚未证明 append/prepend/trim/shallow-clone 是共同 MUST | WebSocket/HTTP/resource consumer 的真实 glue trace，证明现有组合有 material copy/allocation 或生命周期缺陷 |
| async log sink / metric exporter | Obs sink 可接宿主 logger，metric values 可由 Prometheus/OpenTelemetry owner 导出 | 一个不引入第二 runtime 的窄 adapter，以及 producer 零额外成本和 drain/flush/backpressure 旅程 |
| generation-safe slot map | 可由现有 container + owner generation 组合 | 两个独立 owner 的 create/get/erase/reuse/stale failure 与 storage-stability 合同 |
| dynamic plugin unload | Util Registry 有意 startup/freeze；DynLib 只提供装载机制 | 两个 native plugin owner 的 library/object/callback quiescence、失败与 unload 旅程 |
| paired host clock correlation | 直接读 realtime 或 owner calibration 可用 | 第二个 consumer/peer，加上 NTP jump、采样误差、重校准和跨进程边界 |
| recursive watch 之上的 debounce/reload | G06 只产出事实 stream；内容 diff、证书/模型/shader 验证与发布是 policy | 仍归应用或 `hyle-service`，不因 G06 下沉 |
| JSON/UUID/schema/auth/router | JSON `ROUTED` 到 simdjson 等；UUID/业务 schema/auth/router 由上层拥有 | 只有新的通用机制证据，不因名称常见而进入 HYLE |

Compute 的 device/pinned/unified storage、量化 tensor/operator/fusion、GPU event，以及 Graph 的 DAG、动态
shape/alias/in-place legality 仍为 `BLOCKED_BY_OWNER`。本轮只确认相邻基础路线，没有借机设计或修补这两个
`QUEUED` 产品。

## 6. 状态与下一步

- registry 改动：0；现有 27 个 `CLOSED` receipt 不被本项目审计伪造为新证书；
- confirmed product gaps：6；`WATCH` 不构成施工授权；
- product/CMake/test/performance 改动：0；
- 后续建议依赖顺序：G01 Base64 → G02 structured scope → G03 process → G04 WebSocket → G05
  recursive watch → G06 zero-copy send。task service class 仍为 `WATCH`，不得捎带进入 G02；各项都不得与
  IO/Process 一起变成无限任务。

本轮只需文档与 closure checker。任一产品项开工时，以本页的 source identity 和能力描述作为 trigger
输入，重新绑定当时的 code/reference baseline；不能把本页当作未来实现已经通过验证的证据。

实际 validation selector 只选择 `docs` 与 `closure`，跳过 75 个无产品影响 gate；结果为：

```text
documentation contract passed: 202 Markdown files, 784 relative links checked
module closure contract passed: 29 components, schema 4
```

`git diff --check` 同样通过。没有运行 native correctness、delivery 或 P1，因为本轮没有产品、公开头、
CMake、测试或性能合同改动；这项跳过是影响选择，不是复用或伪造旧证据。
