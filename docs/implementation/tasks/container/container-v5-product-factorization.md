# Container v5：真实推理能力与内部因子化

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由当前全产品只读回放触发。它不继承 v4 的窄 CPU-selector 结论，也不把历史能力候选直接当作
> 实现授权。机械状态只以 [`module-status.json`](../../module-status.json) 为准。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`container` / Container /
  `container-v5-product-factorization` / `REOPEN`；
- baseline：`a9b5b6523867601dd2c708a2bedc8fc4d81469f2` / tree
  `fadf478497ce9a873a31ea984ed4ba165765530f`；前轮：
  `container-v4-cpu-feature-migration`；
- trigger：`CONTAINER-INFERENCE-SUPERSET-AND-DELIVERY-2026-09-07`；完整只读证据见
  [`Container 当前全产品能力回放`](container-current-product-replay-2026-09.md)，terminal verdict 为
  `PRODUCT_TRIGGER`；
- `CONT5-R1`：public Swiss probe 宣称 SSE2 baseline，但 MSVC x64 无条件编译 SSSE3 指令，target 又将
  SSSE3 option 以 `PUBLIC` 传播给 consumer TU；该形状存在合法旧 x86-64 上的 illegal-instruction 风险，
  并污染任意消费者的 compile ISA floor；
- `CONT5-R2`：vendored simdutf 直接编入 shared Container，Unix 没有 hidden visibility，私有 provider
  namespace 可能进入动态导出表；
- `CONT5-R3`：ggml/llama.cpp、ds4、vLLM、SGLang 的当前消费与成熟 peer 共同证明 generic eviction
  storage 等能力候选必须重新裁决；v4 明确排除了能力/API，不能代表当前完整产品；
- `CONT5-R4`：真实 target 仍把 sequential、concurrent、queue 与 text 编为单一 Container factor；owner
  已要求内部独立/并行开发和最小依赖闭包，因此长期四因子边界需要在本轮正式 admit/reject 并实施，而
  不能继续只留在历史架构记录；
- 初步受影响面：非 intrusive Container public surface、compiled algorithms/providers、内部 factor 与
  product façade、直接消费者依赖、source/static/shared 交付、用户/架构/性能文档；
- 初步 tier：`P1`。容器 lookup、queue handoff、text codec 与 cache touch 属可能高频 primitive；能力和
  API 未冻结前不开始计时，任何 peer 行先完成 comparison-class 与语义等价；
- 只读依赖：`core`、`error`、`container_intrusive`、`util_core`、`time`、`platform`、`mem`、
  `mem_reclaim` 等已 FROZEN factor。只有具体事实击中其自身 trigger 才另开相邻轮；本轮不得顺手修改；
- 非目标：Compute/Graph、tokenizer 或 KV-cache policy、scheduler wait/cancel/service lifecycle、IO protocol、
  Native FFI/C API、稳定跨工具链 C++ ABI、复制标准库/peer 全表面、公共 closure/performance 仪器重写与
  Formal SOTA；
- 下一状态：先冻结产品 suite 的 scope、预算、机器、factor reservation 和 gate 后进入 `SCOPED`；再完成
  top-down/bottom-up audit、普通 C++ 调用草图与有限 FIX ledger 后进入 `AUDITED`。此前不修改产品代码。

