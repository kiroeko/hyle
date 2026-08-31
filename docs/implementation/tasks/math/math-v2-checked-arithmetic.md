# math-v2 checked arithmetic 收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> **当前状态：`REOPENED`。** `math-v1` 仍是上一轮有效历史候选；当前机械状态只看
> [`module-status.json`](../../module-status.json)。本轮由
> [`math capability replay`](math-capability-replay-v2.md) 的真实 `PRODUCT_TRIGGER` 启动。在进入
> `SCOPED` 前不得修改产品实现。

## Reopen intake

- component / owner / round：`math` / scalar numeric primitive owner / `math-v2-checked-arithmetic`；
- trigger：`MATH-REPLAY-CHECKED-ARITH-01`；
- evidence：CCCL 与 Folly 都提供 checked integral arithmetic，CCCL mdspan 与 llama.cpp/ggml 推理
  backend 又实际用它保护 extent、stride、element count 和 byte size；Math 当前没有公共 route；
- affected scope：same-type non-bool integral checked add/sub/mul，及其 tests、component/package journey、
  architecture/user/performance documentation；
- unaffected conclusions：现有 ceil/round 四个入口、`math -> PUBLIC core`、header-only delivery 与
  scalar-vs-SIMD 边界保持原结论；
- provisional baseline：`57ea0ba90c8472c938c4e3c6360ae4e0a6c69250`；
- tier / profiles / instrument：`P0`；`W1-WIN`、`W1-WSL`、`M1-MAC`；
  `closure-correctness-v1`；Formal `NONE`；
- immediate next action：冻结 closure card 后进入 `SCOPED`，再完成双向能力审计并冻结有限 FIX 集。

本状态只确认新事实击穿旧能力完整度；它不预先决定函数名、返回形状、compiler lowering 或最终
`FIX`。审计超出 checked add/sub/mul 时必须重新 scope，不得把 Math 扩成泛用算法集合。
