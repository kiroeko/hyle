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
