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

## Closure card（`SCOPED`）

- 冻结实现形状：Container 不公开 target enum；StringView/DynamicBitset 的 direct-Highway variants 用
  Container 私有、constexpr 的 SSE4/AVX2/AVX512/NEON requirement matcher，XXH3 AVX2 用与自身
  `hyle_set_isa(AVX2)` flags 相符的较窄 matcher；都读取同一 `platform::cpu_features()` cache；
- 语义与 API：所有公开 Container header、类型、layout、异常/alias/lifetime 和 kernel body保持不变；
  不把 SIMD `KernelTarget` 或 Highway 类型引入 Container public/private dependency graph；
- tier / P0 gate：selector 只在函数局部静态初始化时运行一次，set membership 为 O(1)，没有 allocation、
  lock、syscall、第二层 dispatch 或 per-element branch；synthetic masked-feature tests 必须证明每个
  compiled target 缺任一必要 fact 时正确降级；
- machines / delivery：W1-WIN、W1-WSL、M1-MAC 的受影响 StringView/DynamicBitset/hash correctness，
  WSL ASan/UBSan，Container source/static/shared 和公开头；原 node-table P1 路径无 diff，不重计时；
- dependency：仍为 `container -> PUBLIC core,mem,mem_reclaim; PRIVATE platform`；不新增 SIMD 边；
- 预算与停止：与 Platform v4 共享四小时总预算、最多一次 consumer 修正。若真实 compile flags 无法由
  稳定 raw facts 表达则 `BLOCKED`，不能恢复 `best_isa` 兼容层。

进入 `SCOPED` 只授权完成双向审计；有限迁移 FIX 集冻结前不改产品。

## Audit verdict（`AUDITED`）

反向扫描确认唯一产品消费者是 `string_view.cc`、`dynamic_bitset.cc` 与 `hash.cc`；公开 Container header、
layout 和 kernel body 均不出现 Platform tier。前两族的 generated TU 直接实例化 Highway，必须匹配
HYLE 实际 Highway target group；XXH3 TU 使用普通 `hyle_set_isa(AVX2)`，需求集合更窄。把二者继续用
同一“AVX2”排序比较会再次形成苹果对橘子。

| FIX | frozen implementation | P0 evidence |
|---|---|---|
| `CONT4-F01` | 私有 `cpu_dispatch_internal.h` 声明 direct-Highway SSE4/AVX2/AVX512/NEON requirements 与普通 XXH3 AVX2 requirements | 与 `HyleIsa.cmake` flags/Highway group 逐项对照 |
| `CONT4-F02` | StringView/DynamicBitset selector 由最强到最弱显式 `has_all`，每进程缓存一次；Hash 只用自己的 matcher | synthetic complete/missing-one tests；无 per-item branch |
| `CONT4-D01` | 更新注释、variant tests 与旧 `best_isa/Isa/supports` 反证；公开 API 和旧 P1 数字不变 | 三平台 correctness、三交付与 header scan |

API freeze 为 `FROZEN`（2026-09-05）：Container 不公开 target/feature facade，不依赖 SIMD；私有 selector
只对其实际 compiled TU 负责。无功能、layout、算法或 P1 comparison-class 变化，本轮没有未处置
`MUST`，仅 P0 结构与正确性验证。
