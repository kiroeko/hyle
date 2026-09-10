# HYLE 能力族深度复审（2026-09-10）

> 终态：`PROJECT AUDIT: CLOSED`。本轮把此前“每个 factor 一行”的扫描提高为能力族、已交付头反向追溯和
> 四条完整消费者纵切三重校验。它只校准需求事实和文档，不修改
> [`module-status.json`](../../module-status.json)，不授权产品施工，也不续签任何模块的 correctness、性能或
> source/static/shared 证据。
>
> 后续[范围稳定任务](foundation-capability-expansion-scope-2026-09.md)继续对照成熟实现，并形成当前
> [基础能力扩展决策](../../../architecture/foundation-capability-expansion.md)。本页第 7 节保存初次发现，
> 不再冒充当前需求范围；特别是 G06 owner、G07 credential 范围和 G08 priority 以架构决策为准。

## 1. 为什么需要第二遍

[上一轮范围扫描](hyle-scope-capability-audit-2026-09.md)找到了六项产品缺口，但其 factor 摘要不足以证明：

- 一个模块内所有不同能力族都被检查过；
- 每个已交付 C++ 头都能回到一个真实用户任务；
- 单项 API 拼在一起后，完整推理、渲染、批处理和服务旅程没有缺胶水；
- `ROUTED` 真有可执行路线，而不是“上层处理”四个字。

旧的[推理能力观察清单](../../../architecture/inference-engine-capability-watchlist.md)还保留了已经实现的
Util owning creator，并遗漏后来收口的 Obs、SIMD、Sched 与 IO，证明那份模块式清单已经不能承担当前
能力真相。本轮因此不再用 `NO_TRIGGER` 概括整个 factor。

## 2. 范围、基线与停止线

- HYLE baseline：`49ca22212fab230f01e40e49b99205d37a252bd9`，Windows canonical checkout 干净；
- 产品范围：registry 中除 `compute`、`graph` 外全部 27 个已实现 factor，加 build/package 横向单元；
- consumer/peer 身份沿用上一轮同日 clean snapshots：llama.cpp `e71b8051`、ggml `e91ded11`、ds4
  `f62ca29a`、vLLM `51da0ca6`、SGLang `dcebe8c4`、Folly `65749da4`、Seastar `78f64e23`、Tokio
  `060cc4e4`、oneTBB `e00c68dc`、libuv `3e606bc3`、nginx `5c812c03`、Highway `18e449b6`、xsimd
  `3b3151ff`、mimalloc `c683b7c6`；Cycles/Slate 仍只作无 Git 身份的辅助渲染证据；
- 方法：能力族自顶向下展开、已交付头自底向上追溯、四条纵切逐阶段走通；
- 非目标：不设计 Compute/Graph，不写产品代码，不测性能，不把成熟生态复制进 HYLE，也不把观察项
  伪装成 backlog。

本页结果术语固定：

| verdict | 含义 |
|---|---|
| `COMPLETE` | 当前准入旅程已有自然 HYLE 路线 |
| `BORROWED` | HYLE 拥有语义/API，算法或协议核由已审依赖承载 |
| `ROUTED` | 消费方直接使用具名标准/成熟库；HYLE 不造薄包装 |
| `PRODUCT_TRIGGER` | 已有可复核缺失旅程，可建立后续 closure card |
| `WATCH` | 有信号但 owner、共同语义或成本合同尚未冻结，不授权施工 |
| `OWNER_BLOCKED` | 自然 owner 尚未施工；不能把相邻原子相加后写成 complete |
| `APPLICATION` | 领域 schema、identity 或 policy 明确留给最终产品 |

## 3. 已交付头反向追溯

按 `cmake/HyleComponents.cmake::_hyle_component_public_headers()` 的实际安装规则，当前仓库有 193 个
`include/hyle` 头：174 个属于本轮已实现 SDK，7 个是只供源码构建的 SIMD provider/Sched detail，12 个
属于本轮排除的 Compute/Graph。下表的 174 个头没有 orphan；每组都落入第 4 节能力族。

| owner factor | shipped headers | 能力族 |
|---|---:|---|
| core | 8 | F01–F02；含根级 version/visibility |
| error | 4 | F03 |
| math | 1 | F04 |
| time / timer_queue | 3 / 3 | F05–F06 |
| container_intrusive | 4 | C07 |
| util_core / util_runtime | 8 / 2 | C13–C14 |
| platform / platform_thread / platform_fs | 11 / 4 / 3 | P01–P07 |
| mem / mem_vm / mem_reclaim | 25 / 3 / 1 | M01–M08 |
| container / container_concurrent / container_queue / container_text | 23 / 6 / 7 / 4 | C01–C12、C15–C16 |
| simd | 15 | S01–S05；6 个 provider 头不安装 |
| sched | 10 | Q01–Q07；worker-loop detail 不安装 |
| io_core / io_tls / io_http / io_http2 / io_transfer / io_watch | 20 / 1 / 1 / 1 / 1 / 1 | I01–I12 |
| obs | 4 | O01–O04 |

`io_core` 的 20 个中有 8 个 `hyle/io/detail` layout/support 头随 inline awaitable 交付，但文档明确禁止把它们
当 callable public surface。它们仍由 I01–I04 的同一 owner 构建和交付，不形成隐藏的第 28 项产品。

## 4. 能力族账本

### 4.1 Foundation、时间与平台

| ID | 用户任务 | 当前路线 | verdict |
|---|---|---|---|
| F01 | 编译器/OS/arch 合同、assume/restrict/unroll、版本与可见性 | Core 编译期原语 | `COMPLETE` |
| F02 | cache layout、alignment/interference、fatal/debug invariant 与失败 observer seam | Core；进程级 stack/minidump 交给宿主 crash tooling | `COMPLETE`；crash collection `ROUTED` |
| F03 | Status/Result、引用结果、context、native/system error、无异常有界格式化 | Error | `COMPLETE` |
| F04 | checked/saturating integer、cast、product/sum、alignment/rounding 与 bit/scalar 补位 | Math + `<numeric>/<bit>/<cmath>` | `COMPLETE/ROUTED` |
| F05 | monotonic/realtime/profiling/CPU clock、deadline、sleep、Unix/native checked interop | Time | `COMPLETE`；paired clock correlation `WATCH` |
| F06 | caller-owned arbitrary/common timeout storage；runtime callback/cancel | TimerQueue；执行由 Sched/IO | `COMPLETE` |
| P01 | CPU/OS 可安全执行的 feature facts | Platform 提供 facts；SIMD/kernel owner 是唯一 dispatch authority | `COMPLETE` |
| P02 | 当前进程可用 CPU、P/E/SMT、cache/topology/frequency/cgroup quota | Platform 快照 | `COMPLETE`；更深 distance/bandwidth `WATCH` |
| P03 | NUMA node facts、host/process memory limit/pressure、per-span placement | Platform + Mem owner；模型 eviction 是应用 policy | `COMPLETE/APPLICATION` |
| P04 | process startup priority，以及 thread owner/name/affinity/priority/QoS/per-thread ISA permission | Platform + PlatformThread | `COMPLETE` |
| P05 | address wait/Parker 与 deadline loop | PlatformThread；fiber waiter 由 Sched | `COMPLETE` |
| P06 | VM reserve/commit/protect、environment、entropy、dynamic library | Platform；JIT executable/W^X 留 Compute/JIT owner | `COMPLETE`；JIT `OWNER_BLOCKED` |
| P07 | 同步 file/path/dir/link/temp/lock/random access、递归目录遍历 | PlatformFS | `COMPLETE` |
| P08 | child argv/env/cwd、stdio、async wait、timeout/cancel、process tree terminate/reap | 当前只有原子 Thread/IO/Platform，缺统一 owner | `PRODUCT_TRIGGER` G03 |

### 4.2 Memory 与 lifetime

| ID | 用户任务 | 当前路线 | verdict |
|---|---|---|---|
| M01 | 对齐 system allocator、allocator ref、STL/PMR adapter | Mem；system backend 借 mimalloc | `BORROWED` |
| M02 | fixed/growable arena、stack、fixed/object/variable pool、caller-backed heap | Mem，显式选择 individual free 或 bulk reset | `COMPLETE` |
| M03 | quota、allocation stats、leak diagnostics、system/RSS snapshot | Mem + Platform facts | `COMPLETE` |
| M04 | host/foreign resource、Buffer/view/map/flush/invalidate、scatter logical owner | Mem Buffer/CompositeBuffer | `COMPLETE`；动态 owning segment chain `WATCH` |
| M05 | reusable BufferPool、loan/cache/shared pool 与 bounded ownership | Mem | `COMPLETE` |
| M06 | opaque logical range allocation、generation reset、interval lifetime plan | RangeAllocator/LifetimePlanner | `COMPLETE` |
| M07 | anonymous/file/shared/large mappings、warm/residency/partial release、guarded fiber stack | MemVM + PlatformFS/VM | `COMPLETE` |
| M08 | read-mostly epoch/QSBR retirement | MemReclaim；stalled-reader-independent hazard pointer 直接用 Folly | `COMPLETE/ROUTED` |
| M09 | pinned/registered/coherent/unified/device storage 与 completion lifetime | Compute provider 消费 Mem resource seam | `OWNER_BLOCKED` |
| M10 | 跨进程 shared bytes | SharedMapping + offset/POD | `COMPLETE`（仅 storage） |
| M11 | shared-memory queue/doorbell/ack/backpressure/crash-safe shutdown | Boost.Interprocess/ZeroMQ 可完成一般 IPC；scheduler-native zero-copy 共同形状未冻结 | `ROUTED/WATCH` |

### 4.3 Container、Text 与 Util

| ID | 用户任务 | 当前路线 | verdict |
|---|---|---|---|
| C01 | small/static/in-place/POD/普通动态 sequence | Container + `std::vector/deque/array/span` | `COMPLETE/ROUTED` |
| C02 | circular/ring/byte stream storage | CircularBuffer/RingDeque/ByteRingBuffer | `COMPLETE` |
| C03 | flat ordered/hash map/set 与 hardened/process-local hashing | Container；长输入借 xxHash/wyhash | `BORROWED` |
| C04 | ordered node tree 与稳定 iterator | BTree map/set；其它 policy matrix 直接用成熟库 | `COMPLETE/ROUTED` |
| C05 | entry/weight bounded O(1) LRU | LruCache；pin/eviction policy 由 owner 组合 | `COMPLETE` |
| C06 | dense dynamic bits 与 bounded sparse ID membership | DynamicBitSet/SparseSet；compressed bitmap/rank-select `WATCH` | `COMPLETE` |
| C07 | caller-owned double/forward/hlist、safe/auto/count、sequence algorithms | ContainerIntrusive；复杂 multi-index 用 Boost.Intrusive | `COMPLETE/ROUTED` |
| C08 | guarded concurrent flat/node map/set 与 append vector | ContainerConcurrent + MemReclaim | `COMPLETE` |
| C09 | SPSC/MPMC/unbounded/sharded/work-stealing queue storage | ContainerQueue；wait/cancel/close 属 Sched owner | `COMPLETE` |
| C10 | StringView、lexical Path、ASCII/hex/escape、UTF validate/count/convert | ContainerText；完整 normalization/casefold/grapheme 用 ICU4C/utf8proc | `BORROWED/ROUTED` |
| C11 | Base64/Base64URL size/encode/decode/strict error | simdutf 已有算法但 HYLE 无公共路线 | `PRODUCT_TRIGGER` G01 |
| C12 | interned string identity | StringInterner | `COMPLETE` |
| C13 | scope cleanup、unique resource、intrusive ref、borrow/not-null、FunctionRef、deterministic RNG | UtilCore；owning general callable 用 `std::move_only_function` | `COMPLETE/ROUTED` |
| C14 | demangle 与 startup/freeze owning Registry | UtilRuntime；dynamic unload/quiesce/module graph 仍属独立 owner | `COMPLETE/WATCH` |
| C15 | generic trie/Aho-Corasick/radix storage | llama.cpp/SGLang 的 alphabet、output、eviction 与 persistence 语义不同 | `WATCH` |
| C16 | generation-safe dense slot/handle table | 现有 owner 的 storage、address stability 与 stale-handle 合同仍不同 | `WATCH` |

### 4.4 SIMD、Scheduler、IO、Obs 与交付

| ID | 用户任务 | 当前路线 | verdict |
|---|---|---|---|
| S01 | runtime-safe target、fat/fixed kernel 与 expert vendor escape | SIMD；Highway 是唯一 target dispatch authority | `BORROWED` |
| S02 | dtype conversion/narrow/quantization-independent row conversion | SIMD + Highway/HYLE winner route | `BORROWED` |
| S03 | elementwise math、activation、norm、reduction | SIMD + Highway/HYLE winner route | `BORROWED` |
| S04 | search/scan/sort/top-k/radix/argsort | SIMD + HYLE/Highway；custom comparator 用 std | `COMPLETE/ROUTED` |
| S05 | bitpack/unpack 与可复现 bulk RNG | SIMD | `COMPLETE` |
| S06 | tensor stride/batch、quant block、fused GEMM/attention/MoE/KV operator | Compute/engine | `OWNER_BLOCKED` |
| Q01 | M:N CPU scheduler、Fiber/coroutine、work stealing、worker hooks | Sched，context switch 借 Boost.Context | `BORROWED` |
| Q02 | typed task/join、pinned、blocking、bulk、when-all/any 与 shutdown | Sched | `COMPLETE` |
| Q03 | runtime-sized child set 的 add/cancel/join/drain structured scope | 当前 JoinHandle drop 会 detach | `PRODUCT_TRIGGER` G02 |
| Q04 | cancel scope/token、deadline/timer/waker | Sched + Time/TimerQueue | `COMPLETE` |
| Q05 | mutex/rwlock/semaphore/event/latch/wait-group/barrier | Sched | `COMPLETE` |
| Q06 | bounded channel、oneshot/watch/broadcast | Sched + ContainerQueue | `COMPLETE` |
| Q07 | request service class/priority 与 task-local execution context | 应由真实 scheduling trace 定义，不给普通 task 添 metadata | `WATCH` |
| I01 | per-worker completion context、IOCP/io_uring/epoll/kqueue、cancel/deadline | IOCore | `COMPLETE` |
| I02 | async positional/vectored/direct file read/write 与 blocking fallback | IOCore File | `COMPLETE` |
| I03 | TCP listener/stream、UDP、local stream、endpoint 与 socket options | IOCore | `COMPLETE` |
| I04 | async DNS 与 process signal | IOCore | `COMPLETE` |
| I05 | provider-neutral client/server TLS、verification/mTLS/ALPN/session/close | IOTLS 借 OpenSSL | `BORROWED` |
| I06 | encrypted PEM private-key password、不保留 secret 的 context construction | SGLang 有真实入口，HYLE provider 被封装后无绕行路线 | `PRODUCT_TRIGGER` G07 |
| I07 | HTTP/1 client/server framing、streaming body、SSE、limits/reuse/close | IOHTTP | `COMPLETE` |
| I08 | HTTP/2 stream/session/flow-control/GOAWAY | IOHTTP2 借 nghttp2 | `BORROWED` |
| I09 | WebSocket upgrade、frame/message/control/close/backpressure | 当前 HTTP 明确拒绝 101/handoff | `PRODUCT_TRIGGER` G04 |
| I10 | local/HTTP(S) transfer、resume、verifier、temp/atomic publish | IOTransfer | `COMPLETE` |
| I11 | async file-to-stream kernel fast path + explicit portable fallback | 当前只能用户态 copy | `PRODUCT_TRIGGER` G06 |
| I12 | exact/direct-child file change stream；recursive tree change stream | 当前前者完整、后者缺失 | `COMPLETE` + `PRODUCT_TRIGGER` G05 |
| I13 | URL/UUID、JSON/schema、digest、compression/archive、multipart | 具名成熟库或上层 service；见第 6 节 | `ROUTED/OWNER_BLOCKED` |
| O01 | synchronous typed structured log、filter/rate gate、quiescent sink | Obs | `COMPLETE` |
| O02 | owner/concurrent counter/gauge/fixed histogram/merge | Obs；schema/labels/exporter 归 host telemetry | `COMPLETE/ROUTED` |
| O03 | compile-time trace seam、zone/fiber/frame/memory/thread | Obs 借 Tracy | `BORROWED` |
| O04 | async writer/rotation/export/OTel/Prometheus bridge | host logger/telemetry；不建第二 runtime | `ROUTED` |
| B01 | factor DAG、产品 façade、独立/并行开发与选择性构建 | Build/package manifest | `COMPLETE` |
| B02 | source embed、installed static/shared 与 relocatable package | Build/package | `COMPLETE` |
| B03 | dependency lock、共享 source cache、offline、license/notice | Build/package | `COMPLETE` |
| B04 | Native FFI 与其它语言 binding | 无具名 consumer，当前明确 N/A | `APPLICATION/WATCH` |

## 5. 四条完整纵切

这里不把“原子存在”相加成产品可用。每条旅程逐阶段走到终点；任一 trigger 或 owner block 都使整条纵切
保持未完成。

### 5.1 量化推理与 serving

| 阶段 | 路线 | 当前结论 |
|---|---|---|
| host/NUMA/worker 规划 | Platform + Mem + Sched | complete |
| 模型取得、校验、映射与 warm-up | IOTransfer/MemVM；digest 由 OpenSSL EVP/BLAKE3 owner 注入 | complete with route；G06 可消除额外 copy |
| tensor/device/operator/KV 执行 | Compute/engine | `OWNER_BLOCKED(compute)` |
| request task、取消与动态 child drain | Sched | G02 |
| worker/process 隔离与 IPC | G03；一般 IPC 可走 Boost.Interprocess/ZeroMQ | product trigger + routed |
| HTTP(S)/H2/SSE/WebSocket | IO；WebSocket 缺 G04，encrypted key 缺 G07 | product triggers |
| router/JSON/auth/graceful listener lifecycle | 计划中的 `hyle-service` + simdjson/Boost.JSON | `OWNER_BLOCKED(hyle-service)` |
| metrics/log/trace/export | Obs + host telemetry | complete with route |

### 5.2 异构渲染与资源流送

| 阶段 | 路线 | 当前结论 |
|---|---|---|
| 资产遍历、变化与热重载事实 | PlatformFS + IOWatch | G05 recursive watch |
| archive/image/scene/volume decode | libarchive/zstd/OpenImageIO/OpenVDB 等领域 owner | routed |
| host map、staging、range/lifetime | Mem/MemVM | complete |
| device resource/copy/fence | Compute/provider | `OWNER_BLOCKED(compute)` |
| 并行任务与动态 subtree | Sched | G02 |
| shader/compiler helper process | G03 | product trigger |
| asset/file network stream | IO/Transfer | G06 |
| stable resource identity/plugin unload | owner composition；generic slot/plugin unload 仍 WATCH | watch, not blocker |

### 5.3 批处理数值与数据管线

| 阶段 | 路线 | 当前结论 |
|---|---|---|
| 文件/网络 input 与 bounded transfer | PlatformFS/IO/MemVM | complete；G06 fast path |
| format/schema/compression | FlatBuffers/Cap'n Proto/Arrow 类 owner + zstd/zlib/libarchive | routed |
| checked sizing/scratch/parallel partition | Math/Mem/Sched | complete；动态 scope G02 |
| scalar/SIMD transform、search/sort/scan | Math/SIMD/Container | complete |
| tensor/domain kernel 与通用 DAG | Compute/Graph 或 oneDNN/OpenVDB owner | `OWNER_BLOCKED` 或 routed |
| checkpoint verify/publish/recovery | IOTransfer + injected digest; domain manifest belongs application | complete with route |
| external stage/process | G03 | product trigger |
| telemetry | Obs + exporter | complete with route |

### 5.4 低延迟异步服务

| 阶段 | 路线 | 当前结论 |
|---|---|---|
| listener/socket/DNS/signal | IOCore | complete |
| TLS credential/handshake | IOTLS | G07 for encrypted key |
| HTTP/1、HTTP/2、SSE | IOHTTP/IOHTTP2 | complete |
| WebSocket/realtime | G04 | product trigger |
| handler child lifetime/cancel | Sched | G02 |
| static/file response | IO/Transfer | G06 |
| router/middleware/auth/request context/graceful drain | planned `hyle-service` | `OWNER_BLOCKED(hyle-service)` |
| JSON/multipart/compression/URL | named ecosystem routes in section 6 | routed; multipart integration waits for service owner |
| metrics/log/trace/export | Obs + host telemetry | complete with route |

结论很直接：HYLE 的原子地基已经很宽，但今天不能宣称“只用 HYLE 就能轻松写完整推理 server 或 renderer”。
Compute、Graph 和尚未实现的 `hyle-service` 是公开 owner block；G01–G07 是基础机制缺口。

## 6. 路由不是一句“用户自己做”

下表冻结当前不自造的路线。它们不是 HYLE 依赖锁中的 vendored 能力；消费方拥有版本、构建、schema、
threading 与安全更新，除非未来 adapter closure card 证明 HYLE 有额外价值。

| 任务 | 当前可执行路线 | HYLE 边界 |
|---|---|---|
| JSON hot read / owning build-write | simdjson ondemand/DOM；需要 owning mutation/write 时用 Boost.JSON 等完整 JSON owner | HYLE 只提供 bytes/UTF/allocator/IO |
| schema 与 mmap/IPC serialization | FlatBuffers 或 Cap'n Proto builder/reader | schema/version/wire compatibility 属 consumer |
| URL 与 UUID | Boost.URL parse/resolve；Boost.UUID generator/format | transfer 内部窄 URL parser 不冒充公共 URL 产品 |
| stable/cryptographic digest | OpenSSL EVP 的 SHA-2/3，或官方 BLAKE3 incremental hasher | `hash_bytes` 只作进程内 table hash；TransferVerifier 是注入 seam |
| HTTP/content 与 artifact compression | zlib、Zstandard、Brotli；archive 用 libarchive 等 | IOTransfer 保持 identity encoding；codec/format owner 管 limits 与安全更新 |
| Unicode normalization/casefold/grapheme | ICU4C 或 utf8proc | HYLE 保留 byte/UTF validate/count/convert 快路径 |
| generic local IPC | Boost.Interprocess message_queue/named sync/shared storage；复杂 topology 用 ZeroMQ/nng | HYLE SharedMapping 只承诺 bytes；scheduler-native zero-copy channel 留 WATCH |
| multipart/form-data | 当前由完整 HTTP framework/parser 或未来 `hyle-service` 流式 adapter 承载 | raw HTTP body 已有；边界/临时文件/limits/schema 不能让每个 handler 手写 |
| safe regex / many-pattern scan | RE2 承载避免灾难性回溯的通用 regex；Hyperscan 承载大规则集 bulk scan | SIMD search 只承诺 literal/value search，不伪装 regex engine |
| CLI 与配置格式 | CLI11/cxxopts；toml++、yaml-cpp 等 | option/schema/default/reload 是 application policy；managed subprocess 只接收已经决定的 argv/env |
| formatting 与 localization | C++23 `std::format`/fmt；ICU message/localization | Error 只拥有失败热路可控的 bounded/no-throw formatting，不复制通用 formatter |
| 通用密码学 | OpenSSL EVP 或 libsodium 等成熟 provider | Platform entropy 只提供 OS CSPRNG bytes，不拥有 key storage、AEAD、KDF 或协议 policy |
| 线性代数与几何原语 | Eigen/GLM | Math/SIMD 提供标量与批量原语，不复制 matrix expression、坐标系或变换语义 |
| columnar/table data | Apache Arrow 等 | HYLE 提供 buffer/map/SIMD/scheduler substrate；schema、nullability、wire 与 query semantics 属数据 owner |
| image/scene/volume/spatial formats | OpenImageIO/OpenVDB/Embree 等领域库 | HYLE 提供 file/map/mem/sched/SIMD substrate，不复制格式和几何语义 |

`hyle-service` 当前尚不存在，因此它能拥有的 listener/router/request-context/multipart adapter/graceful-drain
路线统一记为 `OWNER_BLOCKED`，不能用一份 architecture 设想把当前旅程写成 complete。

## 7. 最终差距与停止规则

本轮初次确认的基础产品 trigger 为 7 项。G01–G06 的版本化 consumer 行号和初次准入裁决保存在
[第一遍扫描](hyle-scope-capability-audit-2026-09.md)。本表保存当时发现；当前语义和 owner 已由上方所指
架构决策取代：

| ID | 最小缺口 | 证据与 owner |
|---|---|---|
| G01 | caller-buffer Base64/Base64URL size/encode/decode，严格 alphabet/padding 与 typed error offset | 四个推理 consumer 都有协议/多模态旅程；重开 `container_text`，借现有 simdutf algorithm core，不污染默认 UTF 路线 |
| G02 | runtime-sized structured task scope：add/spawn、cancel、join/drain，析构不遗留 child | HYLE `JoinHandle` drop 会 detach，Folly/Tokio 有成熟形状；重开 `sched`，opt-in owner 不给普通 task 添 metadata |
| G03 | managed subprocess：argv/env/cwd、stdio ownership、async wait、timeout/cancel、process-tree terminate/reap | 四个推理 consumer 都重复生命周期；建立独立 `process` factor，依赖 Platform/IOCore/Sched/Time/Error |
| G04 | WebSocket client/server upgrade、残留字节 handoff、frame/message/control/close、limits 与 backpressure | vLLM/SGLang realtime serving 和 ds4 client；建立 `io_websocket` 并定向重开 `io_http` |
| G05 | recursive directory-tree watch：relative path、动态子目录、rename、overflow→rescan、cancel/close | 渲染资产树与 libuv 成熟机制；重开 `io_watch`，debounce/reload 仍是上层 policy |
| G06 | async file-to-stream：offset/length/progress、short send、cancel/drain；可用时 `sendfile`/`TransmitFile`，否则显式 fallback | nginx/libuv 与资源流送旅程；重开 `io_core`、`io_transfer`，TLS 不伪造 zero-copy |
| G07 | client/server encrypted PEM private key 的 borrowed password/callback；只在 context construction 使用，不记录、不保留、不进入连接热路 | SGLang [`serving.py:137`](https://github.com/sgl-project/sglang/blob/dcebe8c4733a5cb802fb679e0c8fbb886e2801b0/python/sglang/srt/arg_groups/fields/serving.py#L137) 有 `ssl_keyfile_password`，并在 [`http_server.py:2499`](https://github.com/sgl-project/sglang/blob/dcebe8c4733a5cb802fb679e0c8fbb886e2801b0/python/sglang/srt/entrypoints/http_server.py#L2499) 传入 TLS runtime；OpenSSL 有配置期 password callback。重开 `io_tls`，不新增依赖。vLLM 的 cipher policy 只加强 TLS 配置审计，不在本页预定 exact API |

下列信号被保留但没有伪造为第 8 个缺口：

- shared-memory IPC：vLLM broadcast ring 与 SGLang collective 都证明需求真实，但 reader membership、
  one-to-many 与 collective、crash、notification、payload ownership 并非同一 API；先走具名路线，只有
  scheduler-native 共同 trace 能冻结时才提升；
- trie/automaton：llama.cpp 的 generic symbol trie/Aho-Corasick 与 SGLang 的 token-frequency/LRU trie 是
  第二份证据，但 alphabet、build、failure output、eviction 与 persistence 仍不同；
- generation-safe slot：StringInterner、RangeAllocator、HTTP/2 和 SGLang NodeRef 都有 generation，但前者
  分别绑定 pool/tree/stream owner；只有两个 owner 能被同一 storage/invalidation 合同替换时才提升；
- owning segmented chain、task priority、task-local context、paired clocks、dynamic plugin unload、async
  telemetry、NUMA migration/distance、JIT executable VM 与 QUIC/HTTP3 继续按精简后的
  [观察清单](../../../architecture/inference-engine-capability-watchlist.md)晋级。

这次结论是**对明确边界和上述版本化输入的完备覆盖**，不是“未来永不出现新需求”的承诺。以后不再按
“再扫一遍所有模块”重开：新事实先映射到本页能力族；已有 `COMPLETE/ROUTED/APPLICATION` 路线足够就
停止，只有具体旅程证明路线失效才产生 module reopen trigger。

## 8. 文档校准与验证

本轮同时：

- 修正上一轮 Platform/G03、recursive watch/G05 的编号笔误；
- 把 Platform 表中错误归属的 mapped file 改回 MemVM；
- 删除观察清单中已实现的 Registry callable 等旧状态叙述，并补齐后来出现的 owner；
- 在用户入口写清相邻成熟库路线、`hyle-service` 尚未实现以及 TLS encrypted-key 当前限制。
- 校正安装清单复算结果：193 个源码头中 174 个属于已实现 SDK、7 个是 source-only detail/provider、
  12 个属于本轮排除的 Compute/Graph。

只选择 docs/closure 门；没有产品、公开头、CMake、测试或性能合同改动，因此不运行 native/P1。最终：

```text
python scripts/check-docs.py
python scripts/check-module-closure.py
git diff --check
```

结果：documentation contract 通过（203 个 Markdown、792 条相对链接），module closure contract 通过
（29 个 component、schema 4），`git diff --check` 通过。
