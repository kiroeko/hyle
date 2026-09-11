# 基础能力扩展范围

> 决策状态：`ACCEPTED`，2026-09-10。本文冻结本轮基础能力的产品范围、语义下限、owner 与负空间；
> 它不修改模块状态，也不授权产品施工。每项能力仍须建立 closure card，依次完成
> `SCOPED -> AUDITED` 后才能改代码。

## 为什么需要本决策

HYLE 已经有较宽的内存、容器、调度和 I/O 原子面，但“原子存在”不等于真实用户旅程完整。本轮以
ggml/llama.cpp、ds4、vLLM、SGLang 的推理基座为主需求锚，并用渲染/资源流送、批处理管线和低延迟服务
反偏置；实现形状再与 Folly、oneTBB、Boost、libuv、nginx、simdutf 和 OpenSSL 对账。

参考库用于确认已经成熟的问题语义，不用于求 API 并集。准入仍须同时满足
[HYLE 七道产品边界](hyle-scope.md#可执行的产品边界)：真实任务成立、HYLE 有集成价值、owner 与 DAG
自然、实现可持续、默认成本可隔离、C++ 旅程完整。

版本化 consumer、peer 和 HYLE 现状证据保存在
[范围稳定任务](../implementation/tasks/project/foundation-capability-expansion-scope-2026-09.md)。

## 裁决

本轮准入八个能力包。它们是八个独立的产品问题，不是一张必须一次完成的巨型施工卡。

| ID | 已准入用户任务 | semantic owner / 承载 | 执行状态 / 当前缺口 |
|---|---|---|---|
| `G01` | bulk Base64/Base64URL size、caller-buffer encode/decode 与明确校验策略 | `container_text`；`BORROW` 已锁定 simdutf algorithm core | 公开 `<hyle/container/base64.h>` 路线已交付；后续 G04 直接复用，不复制 codec |
| `G02` | runtime-sized structured task scope：动态 spawn、cancel、close、join/drain | `sched`；`SELF`，复用现有 Task/JoinHandle/CancelToken | 已交付 `TaskScope`；普通 spawn 不承担 tracking 成本 |
| `G03` | managed child process：spawn、stdio、async wait、timeout/cancel、tree termination、reap | `process` factor/product；HYLE 拥有语义和 scheduler-native lifecycle，OS mechanism 自有 | 已交付 `ProcessRuntime`/`ChildProcess`、三平台非轮询 carrier、HYLE async pipe、tree termination 与可靠 reap |
| `G04` | WebSocket client/server handshake、HTTP handoff、message/frame/control/close | 新 `io_websocket` factor + 定向重开 `io_http`；协议状态机归 HYLE | 已交付协议无关 HTTP Upgrade owner 与 RFC 6455 client/server、streaming message、control/close、limits 和 plain/TLS 路线 |
| `G05` | recursive directory-tree change stream | `io_watch`；`SELF` 三平台 backend | 已交付显式 `directory_tree`、root-relative batch、rescan/cap 与三平台 backend |
| `G06` | async file-region to byte-stream + HTTP file body，支持时走 kernel fast path | `io_core` + `io_http`；`SELF` OS fast path 和 portable fallback | 已交付 `FileRegion/send_file_region` 与 HTTP file-body；plain 使用三平台 native fast path，TLS/unsupported 使用一次有界 fallback |
| `G07` | TLS credential source closure：file/borrowed PEM、encrypted-key password/callback | `io_tls`；`BORROW` OpenSSL credential mechanism，HYLE 拥有 provider-neutral C++ 形状 | 已交付 file/borrowed PEM、encrypted-key password/callback、client/server identity 与 mTLS |
| `G08` | opt-in low/normal/high non-preemptive execution priority | `sched`；`SELF`，独立于 OS thread priority 和应用 request policy | 已交付 opt-in 三档、继承/覆盖和全 wake/admission 路由；默认 normal-only 成本隔离 |

八项之外没有隐含的“顺手补齐”。发现新的真实缺口时，先回到能力准入，而不是扩大某一张 closure card。

## 各能力的最低合同

### G01：Base64/Base64URL

- 提供 exact encoded size、checked/max decoded size、caller-owned input/output span；不足容量是 typed failure；
- standard 与 URL alphabet 分开选择，padding required/allowed/forbidden 和 ASCII whitespace accept/reject
  必须显式，不能用一个含糊的 `strict` bool；
- 缺省是 standard alphabet、canonical padding required、whitespace rejected；URL/unpadded/forgiving 都由调用方
  显式选择，协议不能偶然接受另一种编码；
- 快速 decode 返回 status、已消费 input 和已写 output。失败时允许 output 含未提交的 partial bytes，但合同
  必须要求调用方丢弃；需要 first-invalid offset 的 detailed validation 是独立冷诊断路线；
- 默认不分配，不提供只为短样例隐藏 allocation 的 owning string 入口；用户可用 Container/Mem 先按 size
  分配。MIME line wrapping、流式 codec 和自动混合 alphabet 本轮不准入。

simdutf 已提供标准/URL alphabet、padding/last-chunk 选项、safe caller-buffer decode 和详细结果；HYLE 只
收敛常用语义，不复制其全部 option bitmask，也不再写一套标量/SIMD 编解码器。

### G02：动态 structured task scope

- scope 可以在运行期加入任意数量 child；`close` 后拒绝 admission；`join` 必须等待全部 child terminal；
- scope 绑定一个 Scheduler；其任务可以并发 spawn，spawn 与 close 的竞态必须线性化，另一个 Scheduler
  或任意外部 runtime 不能把任务塞入该 scope；
- `cancel_and_join` 只发 cooperative cancellation，但无论首个错误或取消何时出现，都必须 drain 全部 child；
- 只需 scope ownership 的 `void/Status` child 用 `add`，不制造无用 JoinHandle；typed/per-child ownership 仍由
  `spawn` 返回的明确 JoinHandle 承担；scope 聚合首个失败并保留 drain，且 cancellation token 可显式传给
  cancellable IO/等待；
- 析构不能静默 detach。异步析构不可实现，因此 live child 下析构是明确 misuse；指南要求显式 join/drain；
- 只有加入 scope 的任务支付 tracking，普通 spawn/schedule 的 record、queue 和 hot path 不增加 metadata。

Folly `AsyncScope/CancellableAsyncScope` 证明动态 task owner 和显式 join 是成熟形状；HYLE 不照搬其异常
模型，而是复用自己的 Status、CancelToken 和 Scheduler 生命周期。

### G03：managed process

- argv 逐项传递，不默认经过 shell；支持 cwd、环境继承/覆盖和 stdio inherit/null/file/pipe；
- spawn 返回 move-only child owner；wait 同时有同步控制面和 scheduler-native async/deadline/cancel 路线；
- exit status 区分正常退出、signal/forced termination 和 spawn/wait 失败；
- cooperative shutdown 由 caller 通过 stdin/pipe/application protocol 发起；OS termination request 在平台
  能表达时可显式使用，但不得冒充跨平台“优雅退出”。另提供 force kill 和 process-tree ownership；
  任何 abandonment 必须选择
  `terminate-and-reap` 或 `detach-to-runtime-reaper`，不能留 zombie 或在析构里偷偷阻塞；
- runtime shutdown 必须停止 admission、处理仍存 child、关闭 pipe 并 drain completion。

Boost.Process v2 和 libuv 都证明 portable process owner 的语义，但分别绑定 Asio/libuv event loop。HYLE
首席消费者直接使用 HYLE Scheduler/IO，因此不引入第二 runtime；实现以三平台 OS primitive 为 mechanism，
参考成熟库处理 quoting、handle inheritance、group/job 和 reap 竞态。shell pipeline、PTY、sandbox、容器和
完整 `ProcessStartInfo` 不在 v1。

### G04：WebSocket

- client/server opening handshake，且 server 能从已经解析的 HTTP request 和预读残留 bytes 接管连接；
- text/binary message、fragment、streaming read/write、mask、ping/pong、close handshake 与 typed close code；
- 每连接允许一个 read 与一个 write 同时在途，禁止同方向并发；control frame 由同一状态机安全处理；
- frame/message/buffer limits、deadline/cancel、bounded backpressure 和 UTF-8 text validation 是合同部分；
- plain TCP 与 TLS 使用同一协议 core。route、auth、JSON schema、reconnect policy 和房间/订阅语义属于
  `hyle-service` 或应用；`permessage-deflate` 本轮不准入。

Boost.Beast 的 handshake-buffer、message/partial operation 和 control/close 模型作为强 peer；它依赖
Boost.Asio，不能直接成为 HYLE 的第二 I/O runtime。协议需要的 Base64 复用 G01；G04 的 AUDITED card
必须选择不带第二 runtime 的窄 SHA-1 mechanism（可借成熟小实现），不为 WebSocket 新建通用 crypto 产品。

### G05：recursive watch

- recursive 是显式 target/option，不改变 direct-child 默认路线；事件路径相对 root；
- 新建/删除子目录后继续覆盖其后代；backend overflow、watch limit 或路径信息丢失必须产出
  `rescan_required`，不能伪造无损日志；
- rename correlation 只作 best effort，调用方在 correlation 缺失时仍以 rescan/re-read 保持正确；
- Windows/macOS 使用原生 recursive facility；Linux 维护 O(directory count) watch，并暴露容量/资源失败；
- 保留现有 bounded batch、single owner、cancel/close/drain。debounce、内容 diff、证书/模型/shader reload
  和发布策略属于上层。

### G06：file-region streaming

- 核心输入是已有 File、explicit offset/length 和 plain byte stream；结果报告 progress/short send，重试不会
  偷改 caller 的 logical offset；
- 支持 deadline/cancel，并在完成、取消或 stream close 前正确 drain backend operation；
- Linux/macOS/Windows 可用时分别使用适当 kernel file-to-socket primitive；不可用或不适用时走有界
  read/write fallback，且不会宣称 zero-copy；
- HTTP body writer 提供 file-region 入口，拥有 Content-Length/chunking、backpressure 和连接复用语义；
- TLS 默认走显式 fallback，只有 provider/OS 真能证明等价 fast path 时才晋级。

`io_transfer` 拥有的是 source→temporary file→verify→atomic publish 的制品生命周期，不能因为也搬 bytes
就承担静态响应或 socket streaming。

### G07：TLS credential source

- CA、certificate chain 和 private key 至少支持 file path 与 borrowed PEM bytes 两种 construction source；
- encrypted PEM 支持 borrowed password 与 non-owning callback/context；只可在 `make_client/make_server`
  同步构造期间调用，不保留、不记录、不复制到连接对象；
- context 构造后拥有 provider state，既有 stream 保持旧 context 活性；watch/rebuild/atomic swap 是
  `hyle-service` 的 reload 组合，不塞进每条连接；
- provider-native handle/type 不进入 HYLE public API。

自定义 cipher string、provider engine、PKCS#11/HSM 和证书选择回调仍在 WATCH；安全默认和外部 TLS
termination 是当前路线。

### G08：execution priority

- public vocabulary 固定为 low/normal/high；它是 runnable selection hint，不是 deadline、QoS、tenant share、
  prefill/decode 或请求价值；
- priority 在 task yield/wake 后保持；child 默认继承当前 execution priority，也可由 spawn/scope option
  显式覆盖，避免高优先级 parent 产生 normal child 的 priority inversion；
- 非抢占：已经运行的低优先级任务不会被中断；高优先级 ready task 优先选择，持续高流量可以饿死低级，
  用户需要隔离/硬保证时应使用独立 Scheduler/OS thread；
- 默认 Scheduler 只有 normal lane，不分配额外 queue、per-task metadata 或 run-loop lane scan；只有启用
  三档的 Scheduler 才支付 priority context、sidecar queue 与选择成本。统一 Waker/IO spine 和隐式 child
  inheritance 要求 admission/wake 两个边界做一次可预测 mode gate；复制第二套 Scheduler/Task/Waker/IO
  类型宇宙只为字面消除该判断会破坏公共组合面，因而拒绝。默认 public/Fiber layout、循环 codegen 和
  material 性能仍必须由 exact baseline 的 P0/P1 证明；若超过 5% 非劣门，退回独立 opt-in owner，而不是
  让所有任务无界付费。

Folly Executor 的 priority queue 和 oneTBB 的 low/normal/high arena 都采用非抢占语义。vLLM/SGLang 的
request priority、KV/preemption 和 cache-aware policy 只是 consumer evidence，仍由推理引擎映射到这个
窄 primitive，不能反向定义 Sched API。

## 本轮不实现的相邻候选

| 候选 | 当前裁决 | 晋级所需稳定事实 |
|---|---|---|
| fixed-membership shared-memory broadcast channel | `WATCH`；一般 IPC `ROUTED` 到 Boost.Interprocess/ZeroMQ | Process 生命周期已就位；仍须取得单 writer/N reader 的 borrow/ack、slot reuse、跨进程 wake、reader crash/lease、coordinator shutdown 三平台共同 trace |
| generic generational SlotMap | `WATCH` | 至少两个 HYLE owner 能被同一 create/get/erase/reuse/stale/address-stability 合同替换；StringInterner 的专用 generation 和 Clay HashMap 不能单独证明 |
| generic trie/radix/Aho storage | `WATCH` / domain owner | llama.cpp symbol automaton 与 SGLang token/LRU radix 的 alphabet、failure output、eviction、build/persist 尚无共同最小合同 |
| mutable owning segmented byte chain | `WATCH` | WebSocket/HTTP/resource 两条实际 glue trace 证明 CompositeBuffer、ByteRingBuffer 和 borrowed sequences 产生 material copy 或 lifetime 缺陷 |

共享内存 channel 的用途已经明确为推理 worker 间的大块一写多读广播；缺的不是名字，而是跨平台 wake、
死亡 reader 和 coordinator 的可验证合同。Process 生命周期现已成立，但这仍不等于 channel 协议已经被
证明，不能直接照抄一份 vLLM ring。SlotMap 与 trie 同理：专用实现可以继续存在，公共泛型只有在真正消除两个 owner 的重复实现时
才准入。

## 实施顺序

每行建立独立 closure round；同一模块的两项也不合并成无界任务。

1. `G01 container_text/Base64`：先交付 WebSocket 和协议共同依赖；
2. `G07 io_tls/credential source`：配置冷路，独立验证 secret lifetime；
3. `G02 sched/TaskScope`；
4. `G08 sched/execution priority`：只在 G02 candidate 已稳定后单独重开；
5. `G05 io_watch/recursive tree`；
6. `G06 io_core + io_http/file-region`；
7. `G03 process`：先登记新 factor，再在 AUDITED 前裁决三平台 carrier 细节；
8. `G04 io_websocket + io_http handoff`：最后复用 Base64、TLS、IO 与 Sched 稳定合同。

前一项完成不自动授权后一项。新 factor 必须先进入中央 component graph 和 registry `QUEUED`；重开已有
factor 必须保存 baseline、建立 card 并按 registry 合法迁移。

## 验收与性能边界

- correctness 先覆盖成功、partial、invalid、cancel、timeout、peer close、owner shutdown 和三交付；
- P0 先证明复杂度、allocation/copy/syscall/queue/metadata。Base64、WebSocket mask 和 file send 等真实
  bulk 热点才进入 P1；TLS credential、process teardown、watch overflow 以正确性和资源界为主；
- 对比必须同 alphabet/strictness、同 lifecycle、同 buffer ownership、同 completion 和 teardown。raw
  syscall 对完整 façade、plain 对 TLS、detach 对 drain 都不是性能 verdict；
- HYLE adapter 调用已借算法时，目标是 release codegen/吞吐与直接依赖调用没有 material wrapper tax；
- 任一能力在有限预算内无法证明跨平台合同，应裁决 `BLOCKED`，不能降低语义后冒充完成，也不能扩建
  一套无限性能仪器。

## 变更规则

本页是当前范围真相；历史 task 只保存发现过程。新增 consumer、标准变化或强 peer 可以形成具体 trigger，
但不得因为“某库还有一个类型”重扫和重开全库。候选晋级时必须同时更新本页、
[观察清单](inference-engine-capability-watchlist.md)和对应 closure card，避免多份互相冲突的 backlog。
