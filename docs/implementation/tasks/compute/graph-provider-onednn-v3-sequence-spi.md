# graph_provider_onednn v3：queue-sequence enqueue SPI 迁移

## Closure card

- component / owner：`graph_provider_onednn` / Graph provider；round=`graph-provider-onednn-v3-sequence-spi`；
- baseline commit：`2820ff72f9cb24fa3ca59fcf2796eef9e7a55ebe`；上一轮精确 blocker 与 P1 证据保存在
  [`graph-provider-onednn-v2`](graph-provider-onednn-v2-ukernel-region.md)；
- trigger 与证据：上一轮 resume condition 只允许真实 Runtime/Graph 主干自然改变 common submit path 后
  恢复。Compute v5 C55--C56 与 Graph v6 G65 已冻结 provider/region 从每-step Completion 改为
  `EnqueueResult`，同 queue segment 只 finalizes 一次；该变化直接移除了上一轮 P0 账单中的逐 step terminal
  路线，构成精确恢复事实；
- 正常 consumer 任务：同一 oneDNN gated-MLP compiled region 在 Graph queue sequence 中录入一次，返回
  accepted/status，由 Graph segment 的唯一 final Completion 承担生命周期和终态；
- 主产品/反偏置纵切：原四节点 decode/prefill region；反偏置为 mock async region 的 partial accepted +
  synchronous tail error，验证 SPI 一般语义而不修改 oneDNN kernel；
- scope：只迁移 region vtable/implementation/tests/comment 至 EnqueueResult；证明 region 不创建 Completion/
  event、partial failure 保真、workspace owner 仍由 Graph 最终 Completion 保活；
- non-goals：修改 oneDNN BRGEMM kernel、layout/precision/threading、region proposal/admission、扩大算子、重跑
  旧 per-step P1 或为 5% 门特判；
- 依赖/消费者：依赖不变；直接消费者为 Graph v6。Compute v5/Graph v6 必须先处于 AUDITED；
- 性能：P0，region enqueue 本身不分配、不 record/wait event；P1 只在完整 Runtime/Graph candidate 后按新的
  segment completion 合同一次执行；
- 验证/预算：Windows compile/unit/shared-static-source、mock partial acceptance、P0 diff；最多一轮 SPI
  迁移和一轮纠正；其它平台随 Graph candidate 回放；
- stop：若 oneDNN 同步调用不能无损表达 accepted/status，精确阻塞；不得保留兼容旧 vtable 双路线。

本卡只授权审计；AUDITED 前不改 provider 产品。
