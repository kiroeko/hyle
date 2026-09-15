# Mem v10：已有 byte quota 的外部存储预留

## Closure card

- component / owner：`mem`；baseline `a3e67bb2`，前轮 [v9](mem-v9-lifetime-plan-allocator.md)。
- trigger：Graph G106 同一 Executable 的并发 Invocation 同时拥有 allocator metadata 与 MemoryResource
  workspace。现有 QuotaAllocator 只能接管真实 allocator 请求，不能为后者先预留相同额度；伪分配或在
  Graph 重写第二套 atomic quota 都不正确。
- 主旅程：先预留独立 resource 请求，再分配真实 Buffer；失败/晚到完成后按真实 owner 释放额度。
  正交旅程：普通 allocator/reallocate 的既有严格 requested-byte 合同保持。
- scope：只给 QuotaAllocator 增加零分配、move-only、借用 owner 的 Reservation；不增加通用资源池、
  动态 limit、physical/RSS 探测或新的 factor。复用现有 reserve/commit/shrink 算法。
- API：`try_reserve_bytes(size) -> optional<Reservation>`；成功零字节也有值，超限无值；Reservation
  提供 size/reset，析构归还。owner 必须活到全部 allocation 和 Reservation 销毁。
- 账目：used/peak 包含 live allocation 请求和显式 reservation；外部 bytes 不再走 get()，禁止双计。
  reservation 是保守预留，不假称实际物理分配；调用方先销毁 resource，再归还 token。
- 性能：P0；既有 allocator 路径算法/对象布局不变；新入口 O(1) 无堆分配，争用沿用现有 CAS。
- 验证：混合 allocation/reservation、超限/overflow、move/reset/零字节、失败回滚、并发与 owner 生命周期；
  Mem/Graph 直接消费、逐头、Windows source/static/shared；完整候选再做 WSL/两台 Apple 和适用
  ASan/UBSan/TSan。未运行的格不签 PASS，不启动 P1/全库/CI。
- 预算：一轮 API/实现及一轮纠正；若必须改变普通 allocator 成本或复制 quota 算法，回到审计。

## SCOPED

真实缺口与自然调用已固定；此阶段不修改 Mem 产品代码。Graph 的所有阶段预算仍归其 G106，
本卡不是完整 Graph 内存验收，也不撤销其它 Mem v9 能力的既有证据。

## AUDITED：有限 FIX

| ID | 动作 | 验收 |
|---|---|---|
| M100 | SELF：Reservation 复用现有 reserve/commit/shrink；保留所有旧路径 | 不调用 backing、不增 QuotaAllocator 字段；空/失败/move/析构账目平衡 |
| M101 | FIX：Graph G106 直接用该 token 预留独立 workspace/private upper bound | 不用虚假 allocation、不重复扣通过 get() 分配的 metadata；异步释放后才归还 |

准入由具名 Graph consumer 证明，机制 owner 是 Mem；Graph 只拥有 family/Invocation 的复合生命周期。
既有 allocator 的代码路径、对象布局和 backing 合同无需变化。以上 API/失败/并发与成本在实现前固定。

## 2026-09-15 实现检查点

M100/M101 已接入：Reservation 只持 borrowed owner/bytes，move/reset/destruction 复用既有 shrink；
零字节成功、超限/overflow 无值，不调用 backing，也不修改原 QuotaAllocator 字段和旧 allocation
算法。新增 4 个测试；Windows Mem/逐头、Graph 并发与失败/异步消费、CPU/CUDA 三交付已回放，详见
[Graph 共享运行存储检查点](../compute/graph-v10-reviewed-foundation-contracts.md#2026-09-15-g106共享运行存储检查点)。
新入口无堆分配、固定空间；无争用 O(1)，竞争重试沿用原 CAS，不承诺 wait-free。未跑 P1，原 allocator
实现未改变，不宣称新增性能胜负。Windows checkpoint 不代替 WSL/两台 Apple/适用 sanitizer；保持
AUDITED，完整候选再签本轮剩余风险，不用旧 v9 CLOSED 冒充新能力已验收。

## Candidate：定向尾项

冻结候选 `4e9f74a09b7218f6053038cc5c5b091e73ff0bc5`；与本轮架构决策 `ec142a92` 相比，Mem
实现和测试没有变化。Windows 复用上述 G106 对应的 Mem/逐头与三交付记录；当前 R2 Graph 的并发预算、
失败/异步回收消费者也通过。只补预声明 WSL、两台 Apple 和 ASan/UBSan/TSan 尾项，当前 CANDIDATE，
未运行部分仍为 UNPROVEN，不触发新的 P1 或全库复检。
