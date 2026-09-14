# Native Provider v2：foundation route 迁移

Compute 后继 [v9](compute-v9-shape-replay-substrate.md) 的 C90/C94 请求签名与 domain 拒绝迁移属于本卡
N20/N21 的直接 SPI 影响；不新增 kernel 或放宽 Native prepared 合同。N22 仍须真实 bounded probe，
不能以新请求能通过编译代替。

C91 的 replay facts 属于 N21：给现有成功 Native replay 明确 prepared 声明，调用方 executor 的内部
成本仍由其 owner 负责；不改变 kernel/线程策略，不把未完成 N22 写成 bounded qualified。

## Closure card

- component / owner：`compute_provider_native` / Native provider；round=`compute-provider-native-v2-foundation-route`；
- baseline commit：`49d32deacf80fbc046e53be64de47042144edc5a`；前轮记录为
  [native-v1](compute-provider-native-v1-factor.md)；
- trigger：Compute v8 的 proposal/shape/cost SPI 是直接依赖变化；Graph foundation 还需要一条真实 CPU
  bounded general route 与 exact tuned route，旧 exact-only candidate 不能冒充；
- 正常任务：19-family 既有 direct exact 路线不变；动态 token CPU Graph 可在一条明确支持的 Native 路线
  绑定 concrete shape 后重复执行；
- scope：迁移新版 proposal/outcome；为所有既有 route 填结构事实；选一个真实 row/MLP shape 实现 bounded
  general + exact variant 探针；保持 HYLE SIMD/Math/Mem/Platform 复用；
- non-goals：新增 operation family、重写全部 kernel 为动态 shape、benchmark 驱动的 route 常数、线程池策略；
- 直接依赖 / 消费者：依赖 Compute/ComputeOps/Math/Mem/Platform/SIMD；direct Compute 与 Graph 消费；
- 性能：SPI 迁移 P0；exact one-op 与 dynamic normal/holdout P1 随 Graph foundation 一次执行；
- 预算/验证：一轮 migration、一条 bounded probe、一轮修正；19-family exact 回归、边界/overflow/OOM、三平台、
  sanitizer、三交付、warm allocation/codegen、P1；
- stop：若 general route 需要 submit-time allocation/JIT/search 或使 exact 路线增加动态分派，阻塞该 probe，
  不横向重写全部 kernel。

## AUDITED 结论与有限改动集

| ID | 允许改动 | 验收 |
|---|---|---|
| N20 | 迁移 C80--C84 proposal/outcome | 19-family exact direct/Graph 行为不变，旧 SPI 删除 |
| N21 | 填 exact/upper-bound/unknown 结构成本与稳定 route identity | 不报告不可资格化的相对 ns，不遗漏 dispatch/workspace/persistent/threading |
| N22 | 一个动态 token normal/holdout route 支持完整 bounded domain，并保留 exact tuned variant | concrete bind 后 warm 无 search/JIT/allocation；非法 domain 点拒绝 |
| N23 | 与 oneDNN MLP 共同参与 selector/copy/workspace 反例 | 完整 Graph cost 而非单 kernel 名称决定胜者 |

公开 API 不变；bounded capability 是 provider 对完整 subdomain 的承诺。当前为 `AUDITED`，只授权 N20--N23。

## 2026-09-14 实现检查点

N20 与 N21 的 exact 路线部分已完成：19-family proposal 明确报告 caller workspace、NativePlan persistent
storage、零 copy/repack/sync 与一次 dispatch；prepare 使用公共四态结果，分配失败为可审计资源拒绝，其他
错误不伪装 fallback。Windows source-tree shared/static 的 Native 逐头与单元测试均通过。N22 bounded
probe 与 N23 完整 MLP selector 尚未开始，状态保持 `AUDITED`。

C93/G103 的共同结构成本迁移已完成：仅把 N21 既有 facts 改为公开共用类型，数值与 kernel/线程策略不变。
本次 Windows shared/static Native 单元与逐头均通过；N22/N23 与终局安装/跨平台/P1 义务仍保留。

C93/N21 后继迁移固定：为既有 exact family proposal 填 `ProviderRouteKey`（family ordinal + 1），
不把临时 prepare token 当稳定身份。完整 key 仍依赖 Provider build/config、语义/shape/layout/device
与 workload/boundary；此字段只由冷 Graph report 消费，不参与当前无 profile 排序，不改任何 kernel。

### N22 表示审计：已编译 RMSNorm 域

按 [Compute v11](compute-v11-bounded-instantiation.md) 的 C110/C111 实施既定 N22：固定 hidden width、
dense f32、输入/输出同轴表达式、可选固定 scale；prefix 可含 bounds/guards 与派生维度。准入先证明
整个域的 schema/布局/字节上界，不能只测试 envelope。每个具体实例只复制既定 NativePlan 的冷参数，
保留同一 SIMD route、alias、资源与完成语义，不重跑 build_plan/propose/prepare。现有 exact family 不
改变 kernel 或热回调。验证包含 small-domain 穷举、派生/guard、错误独立 symbol、zero、源元数据释放、
OOM/销毁/并发与 direct exact 回归；Graph 整体 bounded/variant 仍由其活动卡完成。

2026-09-15：N22 的 Native 域制品与冷实例化已实现，Windows shared/static Native 79/79，含 15 个新增
域/生命周期反例与宽度 8192 的 direct/实例化 executor 对照。完整命令、成本、三交付与未运行项见
[Compute v11 检查点](compute-v11-bounded-instantiation.md#2026-09-15-冷实例化检查点)。Graph general +
exact variant、跨平台/P1 尚未完成，N22/N23 不因此签完。

同日后继：Graph 已消费该域制品，提供公开 general compile 与具体 Invocation；固定尺寸原路线保留，
source/static/shared 的 Native RMSNorm 示例均实际执行。见 [Graph general 检查点](graph-v10-reviewed-foundation-contracts.md#2026-09-15-g101g102-general-路线检查点)。
本段未改 Native 产品；同一 Executable 内的有限 exact variant、完整架构探针与终局跨平台/P1 仍待完成。

后继 Graph 已完成有限 exact 表与 general fallback，直接复用本 Provider 的既有 exact/domain 制品；
见 [G102/G106 检查点](graph-v10-reviewed-foundation-contracts.md#2026-09-15-g102g106-有限变体检查点)。
这不签完完整 Native 架构探针与终局跨平台/P1。

N21 后继 storage 事实：Compute v13 / Graph G106 需要额外 invocation 私有占用上界。Native 的域实例、
run state 都通过传入 allocator 分配，caller workspace 单独声明，replay 不分配；故该新事实为 0，
不能把已有 NativePlan persistent 字节再次计入。只补冷属性，不改变 kernel/线程/热回调。
