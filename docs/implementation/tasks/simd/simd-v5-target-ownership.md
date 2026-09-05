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

## Closure card（`SCOPED`）

- 冻结公开形状：SIMD 自有无跨架构顺序的 `KernelTarget {Scalar,Sse4,Avx2,Avx512,Neon,Sve,Rvv}`；
  `SelectedTarget::target` 报告 family 实际 compiled target，`kernel_target_name()` 供诊断；删除公开
  `simd::Isa` 和只有该 alias/export 宏的旧 `dispatch.h`，export policy 迁独立 `visibility.h`；
- exact/source 形状：不公开跨交付会失真的“当前 TU target”常量；`provider::Target` 不再形成第二个
  枚举，内部 route key 的 `kCurrentTarget` 直接使用 `KernelTarget` 表示正在编译的 provider-coherent
  body；普通运行库只通过 `selected_target()` 报告实际选择；
- target requirements：SIMD 自己声明每个 canonical target 所需的 Platform raw feature set。x86
  Highway-compatible SSE4/AVX2/AVX512 group 必须逐项匹配，NEON/SVE/RVV 匹配各自事实；specialty
  feature 继续由 family 私有 matcher 处理；Platform 不知道这些 group；
- runtime：ordinary unknown-CPU build 初始化一次不可变 Runtime table；每个 whole-buffer 调用仍最多
  一次缓存函数指针；exact/fixed 仍直接绑定；`selected_target` 只诊断、不允许 pin；
- direct/expert consumer：`hyle::simd_expert`、`hyle_target_simd()`、`hyle_add_fat_kernel()` 的三种交付
  拼写和 target flags 不变，用户源码只把类型名迁到 `KernelTarget`；
- tier / P0 gate：不得改变 operation body、provider route、compiled variant 集或循环 codegen；用 synthetic
  target matching、exact/fat fixtures、symbol/reference scan 与 release disassembly/链接图证明没有新增
  runtime layer；无需重跑与本 diff 无关的 387-operation P1 数据；
- machines / delivery：W1-WIN、W1-WSL、M1-MAC；SIMD source/static/shared、ordinary/expert/fixed/fat
  samples，WSL ASan/UBSan，适用的 scalable compile/smoke；
- 预算与停止：与 Platform v4 共享四小时总预算、最多一次 SIMD 迁移修正；任何算法或 route 胜负问题
  立即出本轮，不得用它延长 target-ownership 施工。

进入 `SCOPED` 只冻结以上迁移合同，尚不授权产品实现。

## Audit verdict（`AUDITED`）

反向扫描覆盖 SIMD 公开 headers、runtime table、fat/exact helpers、target manifest、provider route 生成器、
387-operation generated ledger、tests 与 CMake。算法 body 和 winner 没有依赖旧 Platform tier；问题只在
同一概念存在三份类型/名字：公开 `simd::Isa` 是 Platform alias，runtime 用它表示 compiled body，
provider 又有第二个 `Target`。Highway 的 raw feature/group 分层证明新边界与成熟实现一致。

| FIX | frozen implementation | P0 evidence |
|---|---|---|
| `SIMD5-F01` | `target.h` 独占无顺序 `KernelTarget` 与 `kernel_target_name()`；不公开交付语义不一致的 current-TU 常量；export 宏迁 `visibility.h`，删除 `dispatch.h` | public-header/symbol/old-include scan |
| `SIMD5-F02` | SIMD 声明 canonical SSE4/AVX2/AVX512/NEON/SVE/RVV 的完整 raw-feature requirements 与 `kernel_target_supported()` | 每项缺一 fact 必降级；exact 路径不探测 |
| `SIMD5-F03` | Runtime/Module/SelectedTarget/fat helper 全部改用 `.target`；实际 family fallback 如 classic-cl sort 仍诚实报告较低 target | table/fat/exact fixtures + link/disassembly |
| `SIMD5-F04` | 删除 `provider::Target`，route template/generator/generated specializations 直接用 `KernelTarget` | generator `--check`、provider compile probes；无 runtime layer |
| `SIMD5-D01` | target manifest 把 SSE4 token 从 `Sse42` 校正为 `Sse4`，更新 user/architecture/CMake 注释和 tests | source/static/shared、ordinary/expert/fixed/fat journeys |
| `SIMD5-D02` | 公开 specialty diagnostic 精确命名为 `TargetFeature::kAvx10_2_512`；删除 manifest 中已无消费者的 exact token | compile/old-symbol scan；不改 kernel body |

API freeze 为 `FROZEN`（2026-09-05）。`KernelTarget` 只表示 HYLE 已编译/实际选择的 kernel body，不表示
CPU 的最高能力，也不在不同架构间排序；Platform facts 只作为 SIMD-owned requirements 的输入。
`HYLE_SIMD_EXACT_TARGET`、`hyle_target_simd(... ISA ...)` 和 fat-kernel CMake 调用拼写保持不变，含义明确为
编译目标。无 operation/provider/loop/body 或 P1 winner 改动，故仅 P0；本轮没有未处置 `MUST`。

实现期 route generator 回放仍为 387 个 operation、422 个终态 scope、0 pending；仅把 route key 的 enum
owner 从 `provider::Target` 迁到 `KernelTarget`，没有重签任何 provider winner。集成 VQSort 的唯一 diff
是 target 字段注释随类型改名，依赖锁中的 integrated-source content hash 因而同步刷新，算法字节未改。
