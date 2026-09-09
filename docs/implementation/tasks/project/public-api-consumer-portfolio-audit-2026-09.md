# Public C++ 消费者纵切组合审计（2026-09）

本记录保存一次项目级只读能力审计的事实，不是模块 closure card，也不创建第二套生命周期状态。
当前状态仍只看 [`module-status.json`](../../module-status.json)。发现只有建立具名 trigger、为 owner 创建新
card 并完成 `CLOSED -> SCOPED` 后，才授权修改产品代码。

## 范围与证据身份

- HYLE baseline：`0ba27393`；本轮只修改需求/方法文档和消费样例，不修改模块产品代码或 registry。
- 推理：llama.cpp `e71b80510c84`、ggml `e91ded11bdcd`、ds4 `f62ca29a3087`、vLLM
  `51da0ca66c80`、SGLang `dcebe8c4733a`，均为 clean checkout。
- 渲染/异构资源：OpenVDB/NanoVDB `d9b571ce6e7c`、Kompute `267e019ec419`；本地
  `slate-render/cycles` 没有 Git identity，只作 Cycles 源码快照辅助证据。
- 批处理：oneDNN `6b7f1fbbb18e`、OpenVDB `d9b571ce6e7c`。
- 低延迟服务：libuv `3e606bc30a84`、Seastar `78f64e2317b1`、nginx `5c812c034e6c`，并复用上述
  vLLM/SGLang serving 路径。

旧的[推理能力审计](frozen-inference-capability-audit-2026-09.md)是当时方法下的历史事实。它只覆盖一个
产品域，且允许用抽象的“上层处理”证明 `ROUTED`；不能继续作为多域功能完整性的充分证据。

## 五条实际旅程

### 1. 量化推理服务

[`samples/08_inference_service`](../../../../samples/08_inference_service/) 使用 public C++ product targets
完成：Platform 选择 worker/memory facts，Mem 映射权重，Util 选择 backend，Sched/IO 驱动 TCP 与
HTTP/1，SIMD 执行假模型，Obs 保存指标，Error 传播失败。Windows source-embedding configure、Release
build 和五个独立 HTTP 请求均通过；默认端口为 2048，启动时显式 `--metrics`，并正常关闭。

它同时暴露了组合事实：JSON/schema 与模型格式应借成熟依赖/留给引擎；但 pinned coroutine launch、
metric exposition 和 server lifecycle 仍需用户手写，其中前者已经是精确的通用 Sched 缺口。

### 2. 异构渲染与资源流送

正常旅程是：发现/读取资产 → host staging 或映射 → 从大块 opaque device backing 分配 aligned range →
提交 copy/kernel → 由 completion/fence 保活 → 发布稳定资源 identity → 文件变化时构建替代资源 →
旧资源延迟销毁 → 记录 latency/bytes/error。

- Cycles 快照 `cycles/src/device/memory.h:215-295,302-347,360-569,571-596` 展示 host/device ownership、
  device-only buffer、typed vector、copy 和 sub-pointer；Metal 路径
  `cycles/src/device/metal/device_impl.mm:830-863` 展示 completion 安全点之前的 delayed free。
- NanoVDB `nanovdb/HostBuffer.h:159-303,307-498` 展示内部/外部 pool、buffer owner、resize 与 typed
  offset access。
- Kompute `src/include/kompute/Tensor.hpp`、`Sequence.hpp` 展示 staging/device copy、barrier、record、
  async submit/await 和 teardown。

当前 CLOSED 基础层能自然承担文件、host mapping、byte resource、scratch、queue、error/time/obs 和 CPU
调度；动态 opaque range 管理缺失。device provider、stream/event/fence、staging completion 与资源 barrier
应由尚为 `QUEUED` 的 Compute 承担，因此完整渲染纵切当前是 `BLOCKED_BY_OWNER(compute)`，不能写成
已完成。shader 编译、图像 codec、材质/场景、窗口与 present policy 仍合法路由到专门依赖或渲染产品。

### 3. 批处理数值与数据管线

正常旅程是：枚举/打开大文件 → mapping 或异步 positional read → 显式 arena/lifetime scratch →
bulk partition → SIMD transform/sort/reduction → checksum/sync/atomic publish → 汇总错误与指标。现有
Platform-fs、Mem/Mem-vm、Sched bulk、Container、SIMD、IO transfer 和 Error/Obs 已有自然组合；格式
parser、压缩 codec 与领域 tensor schema 应路由到成熟依赖。尚未发现新的 CLOSED-owner trigger，但真正的
异构 DAG 仍受 Compute/Graph 未完成约束。

### 4. 低延迟异步服务

正常旅程是：listener → connection admission → TLS/HTTP → request deadline/cancel/backpressure →
streaming response → reload → drain/shutdown → metrics。IO 已覆盖 transport/framing、Sched 覆盖 task 与
取消；sample 证明一个小服务可写，但还不是“直接得到 REST server”。router、schema/auth 属框架/应用，
而 accept/session/drain owner 和 metric exporter 是否应成为窄公共组合仍需进一步定界。

vLLM `vllm/entrypoints/launchers/utils/ssl.py:15-78` 与 SGLang
`python/sglang/srt/entrypoints/ssl_utils.py:13-85` 都使用可取消文件 watch 完成证书 reload；libuv
`include/uv.h:1702-1792` 提供跨平台 fs-event 生命周期。HYLE 没有等价 public route，这不是 TLS 业务
策略，而是跨服务和渲染资产 reload 重复出现的通用 IO 机制。

### 5. 三种 C++ 交付

上述 API 必须能由 add_subdirectory/source、installed static 和 installed shared 消费。推理 sample 已
证明 source-embedding 下的真实产品选择和 staged runtime DLL；本轮未改变 binary/API，因而不借该样例
重签全部安装证据。新能力的 closure card 必须分别声明三交付验证。

## 26 个 CLOSED factor 的跨域回放

下表 verdict 只描述本次审计结果，不是 registry 状态。

| factor | 多域能力结论 | 本轮发现 |
|---|---|---|
| `core` | 编译、cache/layout、assert primitive 可直接复用 | `NO_TRIGGER`；领域 feature 不应下沉 |
| `error` | Status/Result/system failure 可贯穿四纵切 | `NO_TRIGGER`；多错误汇总仍由 batch/resource owner 组合 |
| `container_intrusive` | caller-owned stable node 能支撑 timer/queue/resource lists | `NO_TRIGGER` |
| `util_core` | borrow、FunctionRef、scope/resource RAII、RNG、refcount 可复用 | `NO_TRIGGER` |
| `time` | monotonic/realtime、deadline 与 OS interchange 完整 | `NO_TRIGGER` |
| `timer_queue` | owner-local timer scheduling 可用于 IO/render/service | `NO_TRIGGER` |
| `platform` | CPU/topology/NUMA/memory/random/VM facts 与机制可复用 | `NO_TRIGGER`；GPU adapter facts 属 Compute |
| `platform_thread` | affinity、priority、thread/Parker 能支撑 runtime | `NO_TRIGGER`；per-task QoS 是 Sched 候选 |
| `platform_fs` | 同步 handle、metadata、directory、atomic file primitives 可复用 | `NO_TRIGGER`；async watch 由 IO 拥有较自然 |
| `mem_reclaim` | CPU read-mostly reclamation 可复用 | `NO_TRIGGER`；GPU fence-delayed destroy 不是 epoch 的同合同替代 |
| `mem` | host allocator/arena/pool/resource/buffer/lifetime plan 广度较强 | `PRODUCT_TRIGGER`：缺动态 opaque logical-range suballocator，见 F01 |
| `mem_vm` | file/shared/large mapping 与 NUMA placement 可复用 | `NO_TRIGGER` |
| `obs` | log/trace 控制与 metric value primitives 可复用 | `METHOD_GAP`：export/schema 路线需定界，见 F05 |
| `math` | checked scalar sizing 与离散 math 可复用 | `NO_TRIGGER`；vector/matrix/geometry 路由 Eigen/Imath/领域 math |
| `simd` | 连续数据/kernels 和 expert provider route 可复用 | `NO_TRIGGER`；图像/几何融合算子归 Compute/renderer，不能按名称扩 API |
| `container` | sequence/hash/tree/LRU/bitset/sparse-set 覆盖常用 storage | `METHOD_GAP`：通用 generation-safe slot map 待证实，见 F06 |
| `container_concurrent` | shared registry/set/vector 路线可复用 | `NO_TRIGGER` |
| `container_queue` | SPSC/MPMC/sharded/work-stealing 路线可复用 | `NO_TRIGGER` |
| `container_text` | borrowed text、path、interner、Unicode 可复用 | `NO_TRIGGER`；JSON/schema/asset formats 使用成熟依赖 |
| `util_runtime` | startup registry、owning callable、demangle 可复用 | `METHOD_GAP`：动态 plugin unload/quiesce 待真实 owner，见 F08 |
| `sched` | fiber/task/join/bulk/blocking/cancel/sync 基础完整 | `PRODUCT_TRIGGER`：缺带结果且全生命周期 pinned 的 structured task，见 F02；QoS 见 F07 |
| `io_core` | file/socket/DNS/local/signal/operation deadline 基础完整 | `PRODUCT_TRIGGER`：缺可取消跨平台 file-change stream，见 F03；service owner 见 F04 |
| `io_tls` | client/server TLS 与关闭路线可复用 | `NO_TRIGGER`；证书 reload 依赖 F03，不在 TLS 内复制 watcher |
| `io_http` | HTTP/1 connection/framing/body/SSE 基础完整 | `METHOD_GAP`：窄 service host 是否准入，见 F04；router/schema 仍不属于 framing |
| `io_http2` | HTTP/2 stream/session/GOAWAY 路线可复用 | `NO_TRIGGER` |
| `io_transfer` | local/HTTP(S) artifact transfer/resume/publish 可复用 | `NO_TRIGGER` |

## 已确认 trigger 与有限候选

### F01 — Mem 动态逻辑区间子分配（确认）

SGLang 的 `RangePageAllocator`（`embedding_cache_controller.py:59-124`）管理 page run；其
`BuddyAllocator`（`transport/allocator.py:14-93`）管理 pinned backing 的 byte offset。vLLM
`block_pool.py:658-758` 批量取得/归还稳定 block ID。渲染/GPU heap 同样需要在不可直接解引用的 backing
中分配 aligned subrange。HYLE `FixedHeap` 管理真实 caller-owned host pointer，`LifetimePlanner` 只处理
预先已知 interval；二者都不能合法模拟 runtime `[0, capacity)` offset allocator。

语义擦除后，这是“metadata-owned、payload-free 的 aligned range allocate/free/coalesce/stats”，属于 Mem；
KV hash/eviction、texture、descriptor 与 fence policy 不进入该类型。下一步应为 Mem 建立新 closure card，
先比较 address-ordered/best-fit 与 buddy 档位和自然 handle，再决定是否提取 `mem_range` factor；不能从
benchmark 行倒推 API。

### F02 — Sched pinned structured task（确认）

`Scheduler::spawn_detached_on` 可以 pin 无结果 callable，`spawn_task` 可以返回 `JoinHandle<T>`，但没有
二者交集。IO 的 per-worker completion owner、per-worker arena/staging 和 renderer upload queue 都要求
task 从首次运行到每次 wake 保持 owner；当前 sample 被迫实现本地 `launch_pinned`。应审计自然的
`spawn_task_on(scheduler, worker, allocator, Task<T>, stack_size)` 或等价形状，并复用现有 Fiber owner/
JoinState，不创造第二套 task lifecycle。

### F03 — IO 可取消文件变更流（确认）

证书/model reload 与 asset/shader reload 是两个独立产品纵切；libuv 给出成熟跨平台机制。自然合同至少
需要 owner、path/filter、rename/replace 语义、coalescing/overflow、cancel/close、目录消失与平台降级。
它应与 `IoContext`/`OperationOptions` 组合，OS 机制可在实现中复用 Platform，但不宜把异步 stream 强塞
进同步 `platform_fs`。是否建立 `io_watch` factor 由 scope 后的依赖/成本审计决定。

### 尚未形成产品 trigger 的五项

| ID | 候选 | 为什么现在不能直接施工 | 晋级所需事实 |
|---|---|---|---|
| F04 | listener/session/drain 的窄 service owner | Asio/Seastar 普通应用仍常自己拥有 accept loop；不能把 router/auth/schema 偷带进 IO | 两个 C++ consumer 重复同一易错 lifecycle，证明组合 helper 能显著缩短且不隐藏 policy |
| F05 | metric schema/registry/OpenMetrics exporter | 当前 value primitive 有意零名字/标签；完整 telemetry ecosystem 有成熟依赖 | 给出一个窄 exporter adapter 或具名 dependency recipe；证明用户无需手写错误 exposition，且 producer 热路不增成本 |
| F06 | generation-safe generic slot map/handle table | inference block ID 与 Cycles texture slot 尚未共同证明 stale-handle 检测语义；不能只因 StringInterner 内部已有 generation 就泛化 | 两个独立 C++ owner 的 create/get/erase/reuse/stale failure 与 storage-stability 合同 |
| F07 | task priority/deadline/QoS lane | interactive/background 与 prefill/decode 都有优先级，但调度 policy、starvation 和继承尚未擦除 | 可复现 queueing trace、有限优先级语义、取消/公平/NUMA 合同和同合同 peer |
| F08 | dynamic plugin load-register-quiesce-unload | 当前 Registry 明确是 startup/freeze；多数 backend 不要求运行时卸载 | renderer 与 inference 中至少两个真实 native plugin owner 的 library/object/callback quiescence 旅程 |

## 结论与下一步边界

旧的“19 factor、确认缺口 0”不能外推为整个 HYLE 功能已足够。本轮已经确认三个跨域通用缺口，并把五个
容易过度设计的候选留在证据门外。其它 CLOSED factor 没有因本次只读回放自动失效；Compute/Graph 的
`QUEUED` 状态仍是完整推理和渲染纵切的最大显式阻断。

下一施工顺序应按依赖与用户收益定为：F01 Mem range → F02 Sched pinned structured task → F03 IO file
watch；每项各自建立小 closure card、有限 FIX 和独立性能合同。F04–F08 不与前三项打包施工。前三项
完成后重跑推理 service 和一个资源流送纵切，再决定是否有足够事实晋级候选。
