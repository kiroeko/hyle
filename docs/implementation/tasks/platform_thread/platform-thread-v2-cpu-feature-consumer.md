# Platform-thread v2：CPU feature consumer 迁移

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只处理 Platform v4 删除旧 `CpuFeatures` 字段形状后，platform-thread 对 SVE/RVV/AMX
> 主机事实的直接消费迁移；不重开线程、affinity、Parker 或 ISA-state 机制。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform_thread` / Platform /
  `platform-thread-v2-cpu-feature-consumer` / `REOPEN`；
- baseline：`749e67a58bd87efcfb6b1b836dc7cfeaaf0fdfd0`；前轮：
  `platform-thread-v1-factor`；
- trigger：`PLATFORM-THREAD-CPU-FEATURE-CONSUMER-2026-09-05`；
- concrete fact：`src/platform_thread/thread.cc` 直接读取 `CpuFeatures::amx_tile/sve/rvv`。
  Platform v4 已冻结删除字段结构、改用无顺序 `CpuFeatureSet`，所以该 CLOSED factor 的依赖/API
  接入必然发生产品改动，命中“public API/dependency changes materially” reopen trigger；
- 初步范围：只把上述查询迁到 `CpuFeatureSet::has(CpuFeature)`，并回放 thread tests、Platform façade
  三交付和 Sched direct consumer；不改变公开 platform-thread API、layout、错误、同步或控制流；
- 初步性能级别：`P0`。三处缓存 set membership 均为 O(1)，不重跑 Parker P1；
- 非目标：新增 CPU facts、修改 AMX permission/SVE/RVV vector-length 机制、线程或 wait 重构、公开 API、
  Native FFI、通用仪器及 Formal certification。

只读 replay verdict 为 `PRODUCT_TRIGGER`。进入 `SCOPED/AUDITED` 前不修改 platform-thread 产品源码。

## Closure card（`SCOPED`）

- 真实 consumer：`query_current_thread_isa_state()` 与 `request_current_thread_amx_tile()`；
- 冻结 FIX 边界：只把 `amx_tile/sve/rvv` 三种字段读取替换为
  `cpu_features().has(CpuFeature::...)`，必要的局部类型名迁移一并完成；公开头、函数、状态结构、错误和
  OS 调用不变；
- dependency：仍为 `platform_thread -> core,error,time` 并私有消费 Platform host facts；不新增第三方、
  Container、SIMD 或 target-policy 依赖；
- tier / normal cost：`P0 / closure-correctness-v1`；每次检查是固定两次整数运算，无 allocation、lock、
  syscall、分支层或常态 metadata 变化；Parker/Thread 路径没有产品 diff，不重计时；
- gates：三平台 thread correctness、WSL sanitizer、精确 factor 与 Platform source/static/shared、Sched
  direct consumer、旧字段/旧 target symbol scan；
- 预算：共享 Platform v4 四小时预算中的一次 direct-consumer 修正；发现 API、控制流或 OS 机制需要变化
  则 `BLOCKED`，不得顺手重构；
- 停止：三种事实查询迁移且验证通过；不以该轮重新声明 Parker 性能结论。

进入 `SCOPED` 后只允许完成上述字段消费的双向审计；有限 FIX ledger 签署前不改产品。

## Audit verdict（`AUDITED`）

反向扫描确认只有 `thread.cc` 四个 AMX 检查、一个 SVE 检查和一个 RVV 检查读取旧字段；公开 header、
Thread/Parker/wait、affinity、OS state transition 与 Sched 调用均不依赖字段 layout。唯一冻结项是
`PTHR2-F01`：迁到 `CpuFeatureSet::has()` 并更新局部类型名/注释。set 已由 Platform 缓存，查询 O(1)、
无新增状态或控制流；旧 Parker P1 不被本 diff 续签或重跑。API freeze 为 `FROZEN`（2026-09-05），没有
未处置 `MUST`、第三方、FFI、公开 API 或 build graph 变化。

## Candidate（`CANDIDATE`）

product commit/tree 为 `d015b0de1cf6d5899da8336fef94104974bcd90e` /
`1456eea7575fa72ce48626a9db13a80f0aa31c09`。W1-WIN 完整 Platform façade 的 host/thread/fs 实现与
公开头测试 6/6 通过；本 factor 的唯一产品 diff 仍是六处 cached fact membership，不触碰 Parker、thread
state transition 或旧 P1 路径。前态 registry snapshot 与 Platform 主轮共用 SHA-256
`1A7715C50BC342F5DA90469E3C92A3ACA862AAD91EB7E3FF030259886386AF6F`。跨平台与三交付未完成，当前
只允许 `CANDIDATE`。

## Candidate verification（`VERIFIED`）

产品身份仍为 `d015b0de1cf6d5899da8336fef94104974bcd90e` / tree
`1456eea7575fa72ce48626a9db13a80f0aa31c09`；最终 replay 使用 test-only 修正后的
`9070a3cdeefdff437451bf5af0003fcc3fd5318c`。Windows selected Platform façade `6/6`，WSL immutable
archive selected closure `45/45`，macOS arm64 selected closure `45/45`；WSL affected ASan/UBSan 各
`5/5`。Platform source、relocated shared/static 三平台各 `1/1`，Sched selected consumer 也在完整
closure 内通过。

本轮六处读取仍只是 cached `CpuFeatureSet::has()`，固定 O(1)、零 allocation/lock/syscall，未改
Parker、affinity、thread-state transition、SVE/RVV width 或 AMX permission 算法；P0 structural 为
`PASS`，旧 Parker P1 不续签。旧字段扫描为 0，依赖仍是 `platform_thread -> core,error,time` 并私有读取
Platform facts，没有 SIMD target policy、第三方或公开 API 泄漏。七项 aggregate gate 全部 `PASS`，无
未决 red-team finding。前态 snapshot 与 Platform 主轮相同，SHA-256
`AD5A05627E04D1C734CA8F160CAD834FC29565639CFEF3A45A3ABFA7EFBA3A5E`。
