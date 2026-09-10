# Container Text v3：Base64 / Base64URL 批量编解码

## Closure card（`SCOPED`）

- component / owner / round：`container_text` / Container /
  `container-text-v3-base64`；kind=`REOPEN`；
- baseline commit：`f3f20528ab3deea1dcf3bed5389226dcee75d1c4`；前轮 terminal record 为
  [`container-v6-identity-p1-resume`](container-v6-identity-p1-resume.md)；
- trigger 与证据：已接受的基础能力扩展决策
  [`G01`](../../../architecture/foundation-capability-expansion.md#g01base64base64url) 证明四个推理基座的
  协议/多模态路径需要 Base64，而 `container_text` 已编入 simdutf v9.1.0 却没有可发现的 HYLE 公共入口；
- 正常 consumer 任务：协议、模型元数据或多模态输入在 caller-owned contiguous buffer 间执行标准
  Base64/Base64URL 的 bulk encode/decode；调用方先查询容量，再复用 Container/Mem storage，不为短样例
  隐藏分配；
- 主产品纵切 / 具名来源：llama.cpp/ggml、vLLM、SGLang、ds4 的模型/请求二进制载荷文本化路线；成熟机制
  采用当前 vendored simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b`；
- 正交反偏置纵切：低延迟网络协议和未来 WebSocket handshake 使用短、中等 token；它防止 API 只适合
  大型模型 blob，也不授权把 WebSocket 或 owning service façade 塞入本轮；
- 跨模块 public journey 与每段 owner：`container_text` 拥有 alphabet/padding/whitespace、size、typed
  progress/error 与 provider adapter；Container/Mem 拥有 caller buffer；后续 `io_websocket` 只消费该公开
  路线，不复制 codec；
- scope：exact encoded size、checked/max decoded size、caller-owned input/output span；standard/URL
  alphabet；required/allowed/forbidden padding；accept/reject ASCII whitespace；fast decode 的 status/input
  consumed/output written；独立 cold detailed validation 的 first-invalid offset；不足容量 typed failure；
- non-goals：streaming codec、MIME line wrapping、自动 mixed alphabet、owning string convenience、
  WebSocket/TLS/JSON、C 或 Native FFI、稳定 C++ ABI、另一套标量/SIMD codec、公共性能仪器修改；
- 直接依赖 / 直接消费者：依赖保持 `PUBLIC core,mem,container / PRIVATE platform`，simdutf 仍是
  `container_text` 的 vendored package dependency；直接消费者保持 Container façade、Util/IO，未来
  `io_websocket` 经公开头消费；本轮不新增 HYLE component edge；
- 性能 tier 与原因：`P1`。Base64 是可观 bulk 热点；目标不是赢一个人为小循环，而是证明 HYLE adapter
  在同 alphabet、padding、whitespace、buffer ownership、validation 与 result 合同下没有 material wrapper
  tax，并保留 simdutf 的线性 SIMD core；
- 适用机器 / 仪器 / material margin：W1-WIN、Release/static source consumer、`closure-perf-v1`、5%；
  W1-WSL 与 M1-MAC 用于 correctness/portability/delivery，不把同物理机或未绑定 Mac 冒充独立 Formal
  性能证据；
- 调查与纠正预算：一次 API/provider 审计、一个产品批次、一次 correctness/red-team correction、一个
  immutable candidate 和一次正式 P1 qualification（只允许合同自带的精度升级）；不扩建公共仪器；
- 预声明验证：header self-containment；RFC 4648 vectors、empty/short/tail/all byte values、alphabet 交叉
  拒绝、三档 padding、whitespace 两档、invalid character/extra bits/remainder、null/empty、overlap 与
  output-too-small；fast/detailed progress；direct simdutf semantic oracle；W1-WIN/W1-WSL/M1-MAC 定向
  correctness；Container source/static/shared installed journey；API/符号/依赖检查；P0 code/cost 审计；
  W1-WIN P1；终局独立红队；
- stop / blocker 条件：若 simdutf 无法承载显式 whitespace/padding 合同且需要复制 codec、三平台错误语义
  无法统一，或唯一 P1 在一次合法纠正后仍 material fail/inconclusive，则诚实裁决 `BLOCKED`；不得降低
  默认严格语义、改变 workload 或增加第二 provider 来伪造完成。

## SCOPED 不变量

1. 默认必须是 standard alphabet、canonical padding required、ASCII whitespace rejected；URL、unpadded
   或 forgiving 都需调用方显式选择，不能用含糊的 `strict` bool。
2. encode/decode 不分配；公开类型不暴露 simdutf enum、result、implementation pointer 或 option bitmask。
3. 快速失败允许 output 含未提交 partial bytes，调用方必须整体丢弃；first-invalid offset 属于独立冷诊断
   路线，不能强迫正常成功路径多跑一遍。
4. size helper 对溢出、非法输入和策略不匹配必须可判别；不得用 `SIZE_MAX` 同时表达多个公开失败语义。
5. P1 只比较固定 API/correctness 后的真实调用；同 alphabet/strictness/buffer/result 之前不产生胜负结论。

## 进入 `AUDITED` 前的有限审计队列

- 逐项映射 simdutf `base64_options`、last-chunk、safe/full result 与 HYLE 的 alphabet/padding/whitespace
  正交语义，特别确认 simdutf 默认忽略 whitespace、padding allowed 与 HYLE 默认 required 的差异；
- 冻结最小自然 C++ API：enum/options、size result、encode result、fast decode result、cold detailed result，
  以及 `StringView`/`std::span<std::byte>` 的 ownership、null、overlap 和 partial-output 合同；
- 证明 exact/max decoded sizing 是否需要扫描、如何报告 invalid，并避免正常 decode 被迫双遍历；
- 扫描四个推理基座、Folly/simdutf 与未来 IO consumer，确认没有遗漏本轮必须的 contiguous bulk shape；
- 冻结有限 `KEEP/FIX/BORROW` ledger、测试矩阵、P0 成本和唯一 P1 workload。完成这些之前不得修改产品代码。

