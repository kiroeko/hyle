# Container v4：CPU feature 路由迁移

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只处理 Platform 删除 provider-shaped `Isa` 后的直接消费者迁移。当前仅登记 `REOPENED`；不重开
> Container 能力超集、容器 API、算法或既有 P1 比较。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`container` / Container /
  `container-v4-cpu-feature-migration` / `REOPEN`；
- baseline：`66dc97620d34dd05b566fdbc359f2a28ad981447`；前轮：
  `container-v3-node-table-reclamation`；
- trigger：`CONTAINER-CPU-FEATURE-ROUTING-2026-09-05`；上游证据见
  [`Platform v4`](../platform/platform-v4-cpu-fact-boundary.md)；
- `CONT4-R1`：StringView、hash 与 DynamicBitset 的私有 variant selector 直接使用
  `platform::best_isa/supports(Isa)`；该接口把 Highway group policy 伪装为主机事实，必须随 owner 修复
  而迁移，不能保留第二套兼容 authority；
- 初步范围：只把这些私有、一次缓存的 selector 改为读取 Platform raw feature facts，并用与各自真实
  编译 flags/指令相符的 requirements；公开容器类型、layout、算法 body 与复杂度保持不变；
- 初步级别：`P0`。通过 synthetic requirement tests、release code shape 和现有 Container correctness
  证明 selector 没有 allocation、lock、syscall、循环内探测或额外 dispatch；
- 非目标：任何新容器、node-table lifetime、hash/string/bitset kernel 优化、旧 P1 证据重设计、
  Native FFI/C API、公共仪器或 Formal 认证。

只读 replay verdict 为 `PRODUCT_TRIGGER`。完成 `SCOPED/AUDITED` 前不修改 Container 产品源码。

