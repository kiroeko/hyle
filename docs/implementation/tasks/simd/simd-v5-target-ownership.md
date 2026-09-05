# SIMD v5：Kernel target 所有权迁移

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮是 Platform CPU fact/target 权威缺陷的直接消费者迁移。当前仅登记 `REOPENED`；在 closure
> card 与有限 FIX 集冻结前不修改 SIMD 产品源码，也不重开 387-operation provider 胜负裁决。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`simd` / SIMD /
  `simd-v5-target-ownership` / `REOPEN`；
- baseline：`66dc97620d34dd05b566fdbc359f2a28ad981447`；前轮：`simd-v4-expert-sdk`；
- trigger：`SIMD-KERNEL-TARGET-AUTHORITY-2026-09-05`；上游证据见
  [`Platform v4`](../platform/platform-v4-cpu-fact-boundary.md)；
- `SIMD5-R1`：`include/hyle/simd/dispatch.h` 声称不重新导出 Platform facts，却公开
  `using Isa = platform::Isa`。`SelectedTarget::isa`、exact target 常量和用户文档因而把 host capability
  与某 family 实际 compiled target 表述为同一类型；
- `SIMD5-R2`：真实 counterexample 已存在：主机可支持 AVX-512，但 classic cl.exe 的锁定 Highway
  AVX3 body 被 blocklist，或某 family 因已认证 lower-tier route 选择较低 target。host facts 与 selected
  kernel target 不能共享一个 authority；
- 初步范围：SIMD 拥有明确的公开 `KernelTarget` 与诊断命名；ordinary runtime、exact target、source-fat
  helper、测试和指南迁移到该类型；provider-specific compile target 仍为内部实现概念；
- 初步级别：`P0`。不得修改任何 operation 语义、provider winner 或 kernel loop；unknown-CPU ordinary
  路径仍每次 whole-buffer 最多一次缓存函数入口，fixed/exact 路径仍为零运行时选择；
- 非目标：新增 SIMD 能力、重测全部 operation、改变 Highway/xsimd lock、VQSort、公共 register DSL、
  Native FFI/C API、稳定 C++ ABI、公共仪器与 Formal 认证。

只读 replay verdict 为 `PRODUCT_TRIGGER`。本轮只迁移 target ownership 和相应 direct-consumer/build
合同；完成 `SCOPED/AUDITED` 前没有产品修改授权。

