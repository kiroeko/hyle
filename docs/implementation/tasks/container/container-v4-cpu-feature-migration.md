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

实现期核对确认 direct-Highway SSE4/AVX2/AVX512 集合与 `hyle_highway_isa_flags()` 一致；XXH3 的普通
AVX2 TU 则只要求 `hyle_set_isa(AVX2)` 允许的 AVX/AVX2/FMA/F16C/BMI2，不错误继承 Highway 的
BMI1/LZCNT requirements。synthetic tests 对每个必要 fact 做 missing-one 降级，并反证 XXH3 不要求 BMI1。

API freeze 为 `FROZEN`（2026-09-05）：Container 不公开 target/feature facade，不依赖 SIMD；私有 selector
只对其实际 compiled TU 负责。无功能、layout、算法或 P1 comparison-class 变化，本轮没有未处置
`MUST`，仅 P0 结构与正确性验证。

## Candidate（`CANDIDATE`）

product commit/tree 为 `d015b0de1cf6d5899da8336fef94104974bcd90e` /
`1456eea7575fa72ce48626a9db13a80f0aa31c09`。W1-WIN MSVC 19.50 shared 的 Platform/Container/Intrusive
实现、公开头、hash seed 与既有 codegen/tooling 共 11/11 PASS；synthetic missing-one tests 证明三种
selector 使用各自的完整 requirement class。公开 Container API、kernel body 与旧 P1 workload 没有 diff。
前态 registry snapshot SHA-256 为
`1A7715C50BC342F5DA90469E3C92A3ACA862AAD91EB7E3FF030259886386AF6F`；三交付和跨平台尚待验证。

## Candidate verification（`VERIFIED`）

产品身份仍为 `d015b0de1cf6d5899da8336fef94104974bcd90e` / tree
`1456eea7575fa72ce48626a9db13a80f0aa31c09`，最终 validation replay 为
`9070a3cdeefdff437451bf5af0003fcc3fd5318c`。Windows affected closure `11/11`；WSL selected Release
`45/45` 且 affected ASan/UBSan 各 `5/5`；macOS arm64 selected Release `45/45`。Container source、
relocated shared、relocated static 三平台九种组合全部 `1/1`，安装后 closure 明确为
`core,container_intrusive,error,mem_reclaim,platform,mem,container`，没有 SIMD 产品依赖。

synthetic missing-one tests 对 direct-Highway target group 的每项 requirement 逐一降级，并反证 XXH3
普通 AVX2 TU 不继承 Highway 的 BMI1/LZCNT group。selector 每进程/函数族只初始化一次，之后不增加
per-element probe、allocation、lock、syscall 或 dispatch layer；公开类型/layout/kernel body、node-table
与旧 P1 workload 均无 diff，所以 P0 structural 为 `PASS`，不重签算法点值。

红队扫描未发现旧 Platform target API、SIMD target 泄漏、公开 feature facade 或苹果/橘子式共用
“AVX2” matcher。三种实际 TU 各自拥有私有 requirement，恰好对应编译 flags；公开 API 和 guide 无需
新增 benchmark-shaped 入口。七项 aggregate gate全部 `PASS`，无未决 medium-or-higher finding。
`CANDIDATE -> VERIFIED` 前态 snapshot SHA-256 为
`AD5A05627E04D1C734CA8F160CAD834FC29565639CFEF3A45A3ABFA7EFBA3A5E`。
