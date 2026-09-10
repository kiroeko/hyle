# 尚未准入的基础能力观察清单

> 本页只保存仍可能晋级、但当前证据不足的长期候选。它不是 backlog、状态表、施工授权或功能缺口清单。
> 当前模块状态只看 [`module-status.json`](../implementation/module-status.json)；当前能力覆盖、已确认缺口与
> 路由见[能力族深度复审](../implementation/tasks/project/hyle-capability-family-deep-audit-2026-09.md)。

## 使用规则

观察项只有同时完成以下工作，才可以成为模块 reopen trigger：

1. 绑定版本化真实 consumer 的对象、规模、操作序列、失败和关闭路径；
2. 去掉 model/token/route/material 等领域名后，仍是稳定通用机制；
3. 证明标准库、成熟依赖、现有 HYLE 原语或上层 owner 没有自然完整路线；
4. 冻结 semantic、mechanism、decision 与 delivery owner，并保持 factor DAG 无环；
5. 登记默认 metadata/allocation/copy/lock/syscall/dispatch、复杂度和三交付影响；
6. 新建 closure card，完成 `SCOPED -> AUDITED` 后才改产品代码。

仅出现新库、新 API 名、更多 LOC、一次 benchmark 或“以后可能用”都不能晋级。候选若已有成熟完整路线，
直接从本页删除或改为 architecture/user 中的 `ROUTED`，不长期囤积。

## 当前候选

| ID | 候选通用旅程 | 当前 owner/路线 | 晋级所缺事实 |
|---|---|---|---|
| `ERROR-W01` | backend/domain 失败需要第二个 typed detail，或 shutdown 同时保留多个独立错误 | backend adapter / caller-owned aggregate；crash detail 归 Obs/host | 两个 consumer 证明现有 native detail、首错 + context 无法支持恢复，并冻结跨 DSO、OOM、顺序和截断语义 |
| `TIME-W01` | monotonic stage time 在导出时与 Unix epoch 精确关联 | owner calibration；Time 可拥有 paired snapshot | 第二个 consumer、采样误差、NTP jump、重校准和跨进程边界 |
| `PLATFORM-W01` | 多 socket/die 的 topology distance、带宽、页迁移与动态 pressure | Platform facts + Mem/Sched/engine policy | 真多节点硬件和 end-to-end locality trace；不能把应用 placement policy 下沉 |
| `PLATFORM-W02` | AMX/SME/JIT kernel 的线程许可、executable memory 与 W^X transition | Platform mechanism + SIMD/Compute 唯一 kernel authority | 已准入真 kernel、三平台安全合同、fallback 与 codegen；不能只因 OS 有 API 就开放 |
| `MEM-W01` | worker/request-local arena 的 thread-exit merge 与全局 pressure reclaim | runtime 拥有 worker registry，Mem 提供 backing | 跨线程释放、zombie/teardown 和压力 trace，证明显式 owner 不足 |
| `MEM-W02` | append/prepend/trim/split/shallow-clone 的 owning segmented byte chain | 当前 CompositeBuffer 是创建后固定的 logical owner；IO sequence 是 borrow | WebSocket/HTTP/resource 两条真实 glue trace，证明现有 ring/composite/borrow 组合产生 material copy 或生命周期缺陷 |
| `CONTAINER-W01` | 大型稀疏 active set 需要 compressed bitmap、rank/select 或 fast merge | DynamicBitSet/SparseSet 或成熟 bitmap | 规模、density、mutation/iteration trace 与独立 consumer；不得污染普通 bitset |
| `CONTAINER-W02` | parser/tokenizer/cache 需要 generic trie、Aho-Corasick、radix/patricia storage | llama.cpp 有 symbol trie/Aho；SGLang 有 token-frequency/LRU trie | 两者的 alphabet、failure output、eviction、build/persist policy 仍不同；先冻结共同 storage/lifetime，而非求 API 并集 |
| `CONTAINER-W03` | 多 owner 需要 generation-safe dense slot/handle table | StringInterner/RangeAllocator/HTTP owner各有专用 generation；SGLang NodeRef 也检测 reuse | 两个 owner 能由同一 create/get/erase/reuse/stale 与 address-stability 合同替换，并证明 metadata/性能收益 |
| `UTIL-W01` | native plugin 动态 load/register/quiesce/unload/reload | Platform DynLib + startup/freeze Registry；完整 manager 是独立 owner | 两个 native plugin consumer 的 library/object/callback 排空、并发 reader、失败与 rollback 旅程 |
| `SCHED-W01` | interactive/background、prefill/decode 等 service class/priority lane | 当前由应用 admission/queue policy 表达 | 真实 queueing trace，并冻结 shares/strict priority、公平、starvation、继承和默认零 metadata |
| `SCHED-W02` | request/trace/allocator context 随 fiber migration 自动传播 | 显式参数；CancelScope 只有专用 context | 两个 C++ consumer 证明显式传递重复且易错，并冻结 nesting/destruction/跨 worker 语义 |
| `IPC-W01` | scheduler-native shared-memory message/broadcast channel | 一般 IPC 走 Boost.Interprocess/ZeroMQ；HYLE 只有 SharedMapping storage 与 LocalStream | vLLM broadcast 和 SGLang collective 的 membership/ack/crash/notification/payload ownership 尚非同一合同；需一个可移植共同 trace |
| `OBS-W01` | 不引入第二 runtime 的 async log/exporter/OTel/Prometheus adapter | 同步 sink/value 接宿主 logger/telemetry | 一个窄 adapter 的 owning queue、copy、flush/drain/backpressure/error 旅程，且 producer 默认成本不变 |
| `TLS-W01` | in-memory cert/key、cipher suite 或 provider policy customization | 安全默认 + 外部 TLS termination；加密 PEM password 已是 G07，不再是 WATCH | 第二个无需泄漏 OpenSSL string/type 的 consumer，及 secret lifetime、reload、兼容和三交付合同 |
| `IO-W01` | QUIC/HTTP/3 或 gRPC transport | 成熟 QUIC/gRPC runtime；HTTP/1/2 与 schema 保持分层 | 具名 C++ consumer、单一 event-loop integration、cancel/backpressure/0-RTT/security 与 package 路线 |
| `BUILD-W01` | backend/ISA 生成物、交叉编译、签名/SBOM/reproducible release | CMake toolchain + 当前 lock/offline/license contract | 具名 toolchain、artifact identity 和验收方；Compute 尚未冻结时不预造全局开关 |
| `BUILD-W02` | Java/C# 等 foreign runtime 的 stable native entry | 当前 Native FFI 为 N/A | 具名 consumer 的 create/call/error/cancel/destroy/unload 完整旅程；不能机械导出 C++ template surface |

## 不在观察清单重复的内容

- 已确认的 Base64、dynamic task scope、managed subprocess、WebSocket、recursive watch、zero-copy file send
  和 encrypted TLS private key 是当前审计中的 `PRODUCT_TRIGGER`，不是 WATCH；
- JSON/schema、URL/UUID、摘要、压缩/archive、完整 Unicode、一般 IPC 与领域格式已有具名成熟路线；
- model/KV/sampling/material/auth/router 等应用策略，以及 Compute/Graph 尚未设计的 tensor/device/DAG
  能力，分别属于 `APPLICATION` 或 `OWNER_BLOCKED`；
- 已经落地的 Util owning creator、Result reference、checked scalar math、Obs metrics、SIMD route 等不再保留
  “曾经待办”的兼容叙述，历史由 Git 和对应 terminal record 追溯。

## 维护边界

Architecture 只保存候选语义和晋级门，不复制 commit、LOC、点值或当前 module state。一次项目审计发现
新信号时先在 implementation task 中保存版本化事实；只有它仍未准入且值得后续观察，才在这里新增或
更新一行。候选晋级、被路由或被否决时，应在同轮移除旧表述，避免旧文档冒充第二套状态真相。
