# 基础能力扩展范围稳定（2026-09-10）

> 终态：`PROJECT AUDIT: CLOSED`。本轮只把已发现缺口与相邻候选对照真实 consumer 和成熟 peer，形成
> [长期范围决策](../../../architecture/foundation-capability-expansion.md)。没有修改产品代码、中央组件图、
> `module-status.json`、测试或性能证据，也不授权任何模块施工。

## 1. 基线、范围与停止线

- HYLE baseline：`ef3d43b5861130b571a79f4daa41bed76eaca0f1`，Windows canonical checkout 干净；
- 输入：上一轮 174 个已交付非 Compute/Graph 头的能力族追溯，以及推理、渲染/资源流送、批处理和
  低延迟服务四条纵切；
- consumer snapshots：llama.cpp `e71b8051`、ggml `e91ded11`、ds4 `f62ca29a`、vLLM `51da0ca6`、
  SGLang `dcebe8c4`；
- peer snapshots：Folly `65749da4`、oneTBB `e00c68dc`、libuv `3e606bc3`、nginx `5c812c03`；产品中
  simdutf 为 vendored v9.1.0/`2d4e249d`，OpenSSL 由 IO dependency contract 解析；
- 非目标：不设计 Compute/Graph，不实现能力，不建立 benchmark，不把 WATCH 变成 backlog，不按库 API
  数量求并集；
- 停止线：每个候选拥有真实任务、strong peer、HYLE 当前路线、准入裁决、owner/carrier、最低语义、
  默认成本和负空间后停止。

## 2. 主要对账事实

| 问题 | 真实消费与成熟实现 | HYLE 事实 | 裁决 |
|---|---|---|---|
| Base64 | 四个推理基座均在协议/多模态路径使用；simdutf 和 Folly 都有高性能 bulk API | container_text 已编入 simdutf，却没有 public Base64 | `G01` 准入，BORROW simdutf |
| dynamic task scope | 动态 request/helper child 是真实 shape；Folly AsyncScope 有 add/join/cancel owner | JoinHandle drop detach，只有静态 when-all/any | `G02` 准入 |
| process lifecycle | 四个推理基座均有 helper/worker/process 管理；Boost.Process/libuv 有完整 owner | HYLE 只有线程、signal、pipe 等原子 | `G03` 准入，独立 process owner |
| WebSocket | vLLM/SGLang realtime serving、ds4 client；Boost.Beast 有成熟 handshake/message/control/close | io_http 明确不能移交 101/预读 bytes | `G04` 准入 |
| recursive watch | 资源树和 reload 需要；libuv/OS 有 recursive backend | HYLE 主动限制 exact/direct-child | `G05` 准入，但不承诺无损 rename log |
| file-region send | nginx/libuv 有 kernel fast path/fallback；资源与静态响应需要 | File/TCP 有原子，但用户必须手写 copy loop | `G06` 准入，owner 改为 io_core+io_http |
| TLS credentials | SGLang 用 encrypted-key password，llama.cpp vendored cpp-httplib 与 Boost.Asio 支持 memory PEM；OpenSSL 有 password callback | provider-neutral TlsContext 只有 file path | `G07` 扩为 file/memory/password credential closure |
| task priority | Folly Executor 与 oneTBB 收敛到非抢占 priority；vLLM/SGLang 有 request-level需求 | Scheduler 只有单 runnable class | `G08` 准入三档 opt-in primitive，应用 policy 不下沉 |

公开 peer 语义也由官方资料复核：

- [Folly AsyncScope](https://github.com/facebook/folly/blob/main/folly/coro/AsyncScope.h) 要求显式 join/cleanup，
  cancellable 版本使用 cooperative cancellation；
- [Folly Executor](https://github.com/facebook/folly/blob/main/folly/docs/Executors.md) 用多队列表达 priority，
  [oneTBB](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/design_patterns/Non-Preemptive_Priorities.html)
  明确这是非抢占选择；
- [Boost.Process v2](https://www.boost.org/doc/libs/develop/libs/process/doc/html/index.html) 与
  [libuv](https://docs.libuv.org/en/v1.x/) 都拥有进程生命周期，但各自绑定另一套 async runtime；
- [Boost.Beast WebSocket](https://www.boost.org/latest/libs/beast/doc/html/beast/using_websocket.html) 覆盖
  handshake、partial message、control 和 close，且支持已预读 request/buffer 的 server handoff；
- [libuv file watch](https://docs.libuv.org/en/v1.x/fs_event.html) 显示 recursive 只在原生支持平台直接成立，
  [file API](https://docs.libuv.org/en/v1.x/fs.html) 提供 async sendfile；
- [simdutf Base64](https://github.com/simdutf/simdutf/blob/master/README.md) 区分 standard/URL、padding、
  safe decode 与 detailed result；
- [OpenSSL password callback](https://docs.openssl.org/3.0/man3/SSL_CTX_set_default_passwd_cb/) 只为
  encrypted PEM construction 提供 callback/userdata。

## 3. 对旧审计的修正

上一轮七项 trigger 是发现结果，不是稳定需求合同。本轮作三项实质修正：

1. `G06` 不重开 `io_transfer`。Transfer 的语义是制品取得、验证和原子发布；file-to-stream 属
   `io_core`，HTTP 易用入口属 `io_http`；
2. `G03` 不预写成“借 Boost.Process/libuv”或某个具名 OS primitive。HYLE 必须拥有 scheduler-native
   语义，且不能引入第二 event loop；direct OS carrier 的具体三平台 primitive 在 process card 的
   AUDITED 前冻结；
3. `G07` 不只补一个 password 字段。既然 HYLE 封装 provider，就同时完成 file/borrowed PEM credential
   source，避免 secret-manager 用户绕过整个 TlsContext；cipher/provider policy 仍无稳定中立合同。

此外，Folly 与 oneTBB 已给出足够独立的通用 runtime 证据，因此把 task priority 从 WATCH 提升为 G08；
只准入 low/normal/high 非抢占选择，不准入推理 request/KV/admission policy。

## 4. 没有晋级的候选

- shared-memory channel：vLLM 与 SGLang 证明 fixed SPMC broadcast 有价值，但实现形状存在明显共同来源，
  跨进程 wake、reader crash/lease 和 coordinator shutdown 尚未获得三平台共同 trace；一般 IPC 已有
  Boost.Interprocess/ZeroMQ 路线；
- SlotMap：Clay 是普通 open-addressing HashMap；StringInterner 的 generation 是专用 identity。当前没有
  两个 HYLE owner 能承诺同一 address-stability/stale-handle 合同；
- trie：llama.cpp symbol/Aho 与 SGLang token/LRU radix 在 alphabet、failure output、eviction 和 persistence
  上不是同一容器；
- owning segmented chain：Folly IOBuf 是强信号，但当前 CompositeBuffer、ByteRingBuffer 和 borrowed
  IO sequence 尚未被两条真实 glue trace 证明失败。

这些项目的晋级条件已写入架构决策和观察清单；没有 benchmark、占位 public API 或预留 CMake 开关。

## 5. 施工计划与状态边界

依赖顺序为：Base64 → TLS credentials → TaskScope → task priority → recursive watch → file-region send →
process → WebSocket。两次 sched round 和各 IO round 保持独立；新 `process`/`io_websocket` 先登记 identity
和 registry `QUEUED`，已有 factor 则各自合法 `CLOSED -> SCOPED -> AUDITED`。

本任务不是 closure card，因此 registry 中 29 项状态保持原值。下一步若开始 `G01`，须以届时 HEAD 为
baseline 新建 `container_text` card，并只预声明该能力的 correctness/delivery/performance。

## 6. 验证

本轮只改变文档合同，适用门为：

```text
python scripts/check-module-closure.py
python scripts/check-docs.py
git diff --check
```

不运行 native 或 P1；没有 candidate code，也没有用旧性能结果续签新能力。

结果：documentation contract 通过（205 个 Markdown、805 条相对链接），module closure contract 通过
（29 个 component、schema 4），`git diff --check` 通过。
