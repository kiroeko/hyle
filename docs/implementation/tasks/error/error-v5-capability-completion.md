# Error v5 消费者能力补全

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮不是无新事实复查。Error v4 关闭后，全局能力展开与真实 IOCP 消费路径共同暴露了一个已存在的
> 原生错误类别缺口；成熟 peer 与普通系统集成 journey 还暴露了三项需要有限裁决的 API 缺口。本轮只
> 重开这些受影响能力，不重做已经签署的 Status 表示、传播宏或全部性能研究。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`error` / Error / `error-v5-capability-completion` / `REOPEN`；
- baseline：`afdb6b81ab4e8e3ad9d790be87a80dc9242c1cec`；前轮：`error-v4`；
- trigger：`ERROR-CAPABILITY-2026-09-01`；
- `ERR5-R1`：Windows IOCP 实现反复读取 `WSAGetLastError()`，但 Error 只有 POSIX errno、Win32、
  HRESULT 与若干计算后端 domain。现有 IO 代码只能把 WinSock 数值压成无类别负整数；这满足“真实
  consumer 暴露 missing-capability defect”的重开条件；
- `ERR5-R2`：普通 registry/cache/parser journey 需要“成功时非空借用、失败时传播 Error”的形状。
  当前 guide 只能要求用户写 `Result<std::reference_wrapper<T>>`，公开类型没有直接表达 borrow 的
  ownership 与 `value()/operator*()/operator->()` 语义；
- `ERR5-R3`：边界适配器需要把运行时诊断参数附到 Status，同时维持 HYLE 默认不抛异常。当前只有
  literal/string 路径，没有统一的 no-throw 动态格式化入口；
- `ERR5-R4`：标准库与第三方 adapter 普遍交付 `std::error_code`。HYLE 文档要求每个调用者自己比较
  category 并映射，形成重复、容易丢失类别的冷边界代码；
- capability provenance：[`capability-factorization-v1`](../project/capability-factorization-v1.md)
  的 `ERROR-A-04/06/07/08`；真实 WinSock 消费位于 `src/io/proactor.cc`，其下游迁移仍由 IO owner
  在自己的轮次完成；
- 初步受影响面：Error 的 borrowed-result、native-domain/translator、冷路径动态诊断与标准错误适配，
  以及 architecture/user guide、correctness、三种 C++ 交付和定向 P1 证据；
- 初步依赖结论：Error 保持只依赖 Core，不引入 platform、obs、fmt 或其他第三方产品依赖；OS 文本
  查询与 C++ 标准库 adapter 是 Error 自己的边界机制；
- 暂定 tier：`P1`。现有 Status/Result 热路径仍复放 Error v4 的正常 consumer 合同；新的 borrowed
  result 属可能高频 primitive，WinSock、`std::error_code` 与动态诊断属于冷边界并以复杂度、allocation
  和异常遏制为主；
- 暂定 non-goals：异常框架、任意 payload/cause/source-location、MultiError、运行时 domain registry、
  日志、Native FFI/C API、稳定 C++ ABI、IO 模块迁移、相邻模块收口、公共性能仪器改造和 Formal 认证。

当前仅登记 `REOPENED` 新事实；closure card、API freeze 与有限 `FIX` ledger 尚未冻结，不授权产品修改。

## Closure card（`SCOPED`）

- component / owner / round / kind：`error` / Error / `error-v5-capability-completion` / `REOPEN`；
- baseline / trigger：`afdb6b81ab4e8e3ad9d790be87a80dc9242c1cec` /
  `ERROR-CAPABILITY-2026-09-01`；
- real journeys：IOCP 在失败点立即捕获并分类 `WSAGetLastError()`；registry/backend lookup 返回非空
  mutable/const borrow 或 Status；模型/缓存/设备诊断把运行时路径、尺寸和设备号放进 failure text；
  filesystem/第三方边界把 `std::error_code` 单向降为共享 Status；既有 graph/sched Status/Result owning
  handoff 保持原语义和代码形状；
- scope：补齐固定 WinSock domain、纯 O(1) classifier 与 no-throw factory；让 `Result<T&>` 直接表达
  非空 borrow；提供无日志/无 source-policy 的有界 `format_status`；提供 generic/system/custom
  `std::error_code` 冷适配与 caller-owned canonical override；更新 Error architecture/user/performance 与
  package journey；
- non-goals：异常框架、任意 payload/cause/source-location、MultiError、运行时 domain registry、反向
  `Status -> std::error_code` 伪 round-trip、日志、IO 产品迁移、相邻模块收口、Native FFI/C API、稳定
  C++ ABI、公共仪器重写和 Formal；
- dependencies / consumers：产品依赖仍为 `PUBLIC core` 且无第三方产品依赖。Abseil/Folly 只作 peer，
  `<format>`/`<system_error>` 是 opt-in C++ 标准头；直接消费者是 Platform/Mem/Sched/IO 与未来 registry/
  backend map，当前只回放 Error public journey 和已冻结的 source-consumer owning handoff；
- tier / margin：`P1` / 5%。`Result<T&>` 是可能高频 primitive；既有 owning carrier 重放当前四个正常
  consumer workload。WinSock、动态诊断和 standard-error adapter 是冷边界，分别要求 O(1) classifier、
  有界栈工作集、至多一次 Status backing allocation、无隐式日志/锁/registry；
- normal metadata：owning `Result<T>` 的类型、layout 与 codegen 必须不变；borrow carrier 只为选择它的
  返回类型支付一个 pointer、一个 Status error arm 和 discriminator，不分配、不加全局状态；code-only/
  OK Status 不增加字段；格式化固定使用 1 KiB caller-side stack scratch，溢出带显式 truncation marker；
- budgets：能力/API/实现 4 小时，正确性/交付/文档 2 小时，性能资格 2 小时，红队 30 分钟；最多一次
  frozen workload 复现、一次 codegen/profile 机制检查、一次性能导向产品修正和一次重跑；仪器若不能
  给出预声明 verdict，记录独立 blocker，不在 Error 轮扩建统计系统；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 `closure-perf-v1`，P0 结构检查沿用
  `closure-correctness-v1`；
- strong peers：Abseil `5e12b490a2009cb223dd75a70fd0213847b82f9c` 的 `Status/StatusOr<T&>` 与 errno
  taxonomy；Folly `176c088840724715b2c3c6211b2b347611676002` 的 `result<T&>` deep-const/lifetime
  设计。性能同合同 peer 只用 Abseil；Folly 不因 rich-error 语义差异进入同合同 timing；
- inference provenance：llama.cpp `9d817213a0975020775efe6c458822616826f376`、standalone ggml
  `d4716378882593333721eb33f153144b6885caf2`，均为 2026-09-01 clean fast-forward reference；
- prototype provenance：WinSock 为 `REAL_CONSUMER`；borrow 为跨 Container/Util/Platform 后确认的
  `NORTH_STAR_JOURNEY + PEER_CONSENSUS`；动态诊断和 standard-error 为 llama/ggml 与普通 C++ 系统集成
  `REAL_CONSUMER`，不是 benchmark 需求；
- performance workloads：更新 Abseil identity 后重放 v4 四个 source-primary workload；新增一个
  fixed-table registry lookup（命中返回非空 borrow、缺失返回 code-only Status）的
  `SAME_CONTRACT` consumer，双方共享 lookup 算法、命中分布、lifetime 与 observable，5% 非劣；固定
  backing copy 只由既有 owning handoff 的 material 结果解释，仍不得单独改变产品；
- correctness gates：正常/错误、零码/未知码、WinSock taxonomy、UTF-8/numeric fallback、format 成功/
  truncation/custom formatter failure/OOM、borrow mutable/const/deep-const/temporary rejection/转换/copy/move/
  monadic/传播宏、custom error category/throwing message、溢出与 Result 非 OK invariant；并发只验证共享
  Status immutable handoff，cancel/close 与 real ISA 不属于本轮；
- delivery gates：公开头 self-contained，Error component-only source test；source embedding、installed
  static/shared C++ consumer 与 relocation/runtime staging；中央 factor/façade/include-owner 合同；
- output：有限 capability ledger、冻结 API、实现/测试/guide/architecture/performance evidence、候选与
  manifest、独立红队、四项 LOC 和 terminal receipt；
- stop：四项 `MUST` 均 `COMPLETE` 或完整 `ROUTED`，owning Result 无漂移，正确性/三交付/docs/red-team
  全通过且所有 P1 workload 通过 5% 门即 `CLOSED`；任何 correctness、交付或稳定 material P1 failure
  在一次机制检查后仍存在即 `BLOCKED`，不新增第五种测试姿势。

当前为 `SCOPED`。上述范围、预算、peer、成本与验证合同冻结；产品代码仍须等待 public-surface 反向
追踪、capability ledger 和 API freeze 进入 `AUDITED`。
