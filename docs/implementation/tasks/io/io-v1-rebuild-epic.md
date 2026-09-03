# IO v1 重构 Epic：能力超集、因子化与逐轮收口

<!-- PLANNING RECORD: CURRENT -->

> 本页是多轮施工导航，不是模块状态或产品修改授权。每个 factor 只有在
> [`module-status.json`](../../module-status.json) 指向自己的 closure card 并依法进入相应阶段后才能
> 修改；历史 `io-01` 至 `io-11` 记录只作输入，不能代替本轮审计。

## 为什么拆成多轮

当前约 7,926 行 IO 实现把五类生命周期和可选成本聚在一个单元。把它一次性作为一个模块审查，会让
TLS/HTTP 可选依赖、底层 completion 性能和高层传输 workflow 相互遮蔽，也无法让维护者并行工作。
本轮保留一个用户产品 `hyle::io`，内部按最小独立 ownership 拆成五个因子：

| Order | Machine ID | Capability boundary | Initial state |
|---|---|---|---|
| 1 | `io` | IO core：completion、file、network、local IPC 与统一 cancellation/error seam | existing `QUEUED` input |
| 2 | `io_tls` | TLS config、transport、session、deadline/cancellation | reserved `QUEUED` |
| 3 | `io_http` | HTTP/1 framing、bounded bodies、client/server lifecycle | reserved `QUEUED` |
| 4 | `io_http2` | HTTP/2 connection/stream/flow-control lifecycle | reserved `QUEUED` |
| 5 | `io_transfer` | resumable/bounded artifact and model transfer workflow | reserved `QUEUED` |

现有 `io` ID 承担语义上的 `io_core`，避免无收益的状态身份改名。内部因子对 HYLE 开放精确依赖，对
普通用户封闭；用户仍从同一个 `hyle::io` façade 获得已经实现并收口的完整产品根集合。

## 每轮必须回答的合同

每个 factor 都按同一顺序施工：

1. 由 llama.cpp 真实消费、ggml 基础能力、成熟 peer 和通用工程需求建立 capability-family 超集；
2. 冻结 ownership/lifetime/failure/cancellation/concurrency/default 和自然 C++ journey，再允许实现；
3. 只依据公开类型与源码真实消费声明最小直接 factor 边；不得为了避免依赖而复制已存在能力，也不得
   因历史候选图而重开已关闭底层模块；
4. correctness 与 source/static/shared 通过后，P0 查成本结构，P1/P2 才在同语义 normal workload 上
   测量；不能用协议层与裸机制、同步与异步、无取消与有取消做苹果/橘子比较；
5. 回答第三方路径、推理能力、性能、FFI 清洁、三种交付、C++ 易用性/文档六问，并执行独立红队；
6. 当前 factor `CLOSED` 后才推进下一项；底层 API/依赖变更只对已关闭直接消费者做有限回放。

## 当前 reference 与 capability 输入

2026-09-03 干净 reference：llama.cpp
`0df974d777c904dda1da3b00faa7769c6310ae74`，其 ggml
`59fbdbe5b1d989f9226ca9189256ed4ebcfa01dc`，standalone ggml
`d4716378882593333721eb33f153144b6885caf2`，Asio
`8806a6803cde7054c3049d3666d3ec36786568c5`，Seastar
`9b95b250cc64af11b84fa5ea3d677f76b37fdef0`。开始每个 factor 的 `AUDITED` 阶段时重新检查远端并绑定
实际使用的 commit；本页身份不是永久锁。

既有 Stage C 账本列出的重点缺口包括 typed endpoint/address、统一 Error、socket 配置诊断、resolver
取消与 Happy Eyeballs、vectored/bulk/direct IO、local stream、异步 body/request context、HTTP host、
可靠模型传输、HTTP/2 streaming，以及 TLS config/session/deadline/cancel。它们是审计输入，不是未经
逐项 consumer/ownership 裁决就自动进入 FIX 的清单。

近期 peer 变化进一步提示：Asio 的 io_uring ring/count/submit batching/iowait 和 SSL buffer/cancel
改进属于 IO core/TLS 的机制审计项；Seastar 的 AIO fallback syscall thread、iovec lifetime、Direct-I/O
alignment 与 socket 修复属于正确性/成本审计项。llama.cpp 的 remote media 和 server lifecycle 没有
引入新的 factor 家族，但继续证明 remote transfer/server 是实际 consumer journey。

## 阶段与停止点

| Milestone | Product work allowed | Exit condition |
|---|---|---|
| Build v11 identity registration | no | four reservations are unreachable and existing product builds unchanged |
| `io` SCOPED → AUDITED | audit only until FIX freeze | IO-core capability/API/dependency ledger frozen |
| `io` candidate → CLOSED/BLOCKED | bounded IO-core fixes | all applicable gates adjudicated |
| TLS → HTTP → HTTP2 → transfer rounds | only current factor | each factor independently reaches terminal state |
| IO façade integration | build/package projection only after roots close | source/static/shared product reports exact built features |

Compute 与 Graph 保持 `QUEUED`，不会作为本 epic 的消费者验证或修改对象。任何一轮达到预算、需要新
公共仪器、缺真实设备/peer 或语义等价无法裁决时，按方法写 `BLOCKED/INCONCLUSIVE`，不把下一 factor
一起拖入，也不以空 target、空测试或历史性能表伪造完成。
