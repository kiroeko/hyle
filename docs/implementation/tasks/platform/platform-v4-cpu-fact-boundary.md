# Platform v4：CPU 事实与执行 target 权威分离

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本轮由已关闭 Platform 的新产品事实触发。当前仅登记 `REOPENED` 证据；完成有限能力审计和
> closure card 以前，不授权修改产品源码。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform` / Platform /
  `platform-v4-cpu-fact-boundary` / `REOPEN`；
- baseline：`66dc97620d34dd05b566fdbc359f2a28ad981447`；前轮：
  `platform-v3-capability-factorization`；
- trigger：`PLATFORM-CPU-FACT-TARGET-AUTHORITY-2026-09-05`；
- `PLAT4-R1`：公开注释把 `platform::Isa/best_isa()` 定义为主机 CPU 能运行的最高 ISA，但 x86
  实现实际复用了 Highway target group：所谓 AVX2 还要求 FMA、F16C、LZCNT、BMI1/2 与 SSE4.2，
  AVX-512 又要求 Highway AVX3 的完整 group。它是 provider/codegen policy，不是单纯 CPU/OS 事实；
- `PLAT4-R2`：SIMD 通过 `using Isa = platform::Isa` 同时把该类型用于实际 compiled family target，
  而 `simd::provider::Target` 又描述 implementation TU target。一个公开类型因而同时承担 host fact、
  product selection 与 provider body 三种语义，已经击穿前轮的唯一 authority 和自然 API 结论；
- `PLAT4-R3`：当前 llama.cpp `427291b5b34cd914a31b3fd3b61a68f6184f4b9f` 的随仓 ggml tree
  `9bf0d932917ba3ccfe7e3d9fd1723390dc870ec0` 用 ARM DotProd/FP16/SVE/SVE2/I8MM/SME/SME2
  与 x86 AVX512-VBMI 等运行时事实选择真实推理 backend；Platform 现有公开事实缺 DotProd、ARM
  FP16、SSE3、AVX512-VBMI，且 macOS/Windows ARM 对已经公开的 SVE/SVE2/I8MM 等字段未探测；
- `PLAT4-R4`：当前锁定 Highway 参考 `63f4b16cda024e2adff8a18dee81e067f8a87d61` 也明确把
  raw CPUID feature bits 与 target groups 分开。该证据支持分离事实与策略，不授权把 Highway 的完整
  API 并集复制进 Platform；
- 初步受影响面：`include/hyle/platform/cpu.h`、隔离 probe TU、CPU tests、Platform 用户/架构文档，
  以及直接使用旧字段或 `Isa/best_isa/supports` 的 platform-thread、SIMD 与 Container；platform-fs
  产品源码不受影响，只回放 Platform façade 交付；
- 初步性能级别：`P0`。探测仍只允许一次有界初始化，随后 feature query/membership 为 O(1)；本轮
  不改变 Parker、filesystem、topology、memory、VM 或 NUMA 算法，也不重跑与 diff 无关的 P1 数字合同；
- 初步非目标：新增 kernel 或 SIMD provider、修改现有算法 winner、device inventory、scheduler
  policy、异步 I/O、Native FFI/C API、稳定 C++ ABI、公共性能仪器和 Formal 认证。

只读 replay verdict 为 `PRODUCT_TRIGGER`。下一步必须冻结 CPU feature-set 的自然公开形状、有限特征
清单、三平台探测合同、直接 consumer 迁移和验证预算，再进入 `SCOPED`；当前没有产品修改授权。

## Closure card（`SCOPED`）

- 真实用户任务：推理 backend、自定义 kernel、Container/SIMD dispatcher 在不知道最终客户 CPU 时，
  取得本进程可以安全执行的单项 CPU/OS facts，再由能力 owner 对照自己实际编译 TU 的 requirements；
- 冻结公开形状：`CpuFeature` 是无顺序的具名事实；`CpuFeatureSet` 是可值传递的固定小型 set，提供
  `has/has_all/has_any` 与显式构造，不含 provider、target、score 或“最好”概念；
  `cpu_features()` 返回一次探测后缓存的 immutable set，`cpu_feature_name()` 只供诊断；删除
  `platform::Isa/best_isa/supports(Isa)/isa_name`，不保留 alias 或 deprecated bridge；
- 冻结语义：普通向量事实表示 CPU 声明且 OS 已启用必要扩展状态后，本进程可安全执行该指令族；AMX
  使用带 `Hardware` 后缀的事实，仅表示 CPUID hardware capability，per-thread permission 仍唯一归
  `platform_thread`；未知/无稳定 OS ABI 的事实返回 absent，不猜测；
- 有限新增 facts：x86 的 SSE3、AES、PCLMULQDQ、AVX512-VBMI/VBMI2/BITALG/VPOPCNTDQ、GFNI、
  VAES、VPCLMULQDQ 与 AMX-FP16 hardware；旧 AVX10.2 字段收紧为 version ≥ 2、512-bit width 和
  OS ZMM 均成立的 `kX86Avx10_2_512`。ARM 新增 AES、PMULL、DotProd、FP16，以及 macOS/Windows
  对已经承诺的 SVE/SVE2/I8MM/SME/SME2 的稳定 OS 查询。现有 AVX/VNNI/BF16/FP16/AMX、SVE
  subfeatures 与 RVV facts 全部迁移，不能静默丢失；APX/全量 CPU catalog 没有当前准入，不进入本轮；
- 直接依赖不变：`platform -> PUBLIC core,error`；没有新第三方产品依赖。参考只绑定 clean llama.cpp
  `427291b5b34cd914a31b3fd3b61a68f6184f4b9f`/ggml tree
  `9bf0d932917ba3ccfe7e3d9fd1723390dc870ec0`、Highway
  `63f4b16cda024e2adff8a18dee81e067f8a87d61` 与 standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`；它们不是产品链接依赖；
- tier / normal cost：`P0 / closure-correctness-v1`。probe 一次 O(number of fixed CPUID/sysctl/
  auxv queries)，cached access O(1)，set membership/all/any O(1)、零 allocation/lock/syscall；类型不携带
  normal-path metadata 之外的动态存储；
- correctness：bit-set algebra、invalid enum、feature-name totality、synthetic x86 CPUID/XCR0 gating、
  synthetic ARM Linux bits、当前真机 invariants、concurrent cache initialization；三平台 build/test，
  WSL ASan/UBSan，AArch64 compile，public-header self-contained；
- delivery / direct consumers：精确 `platform` factor 与完整 Platform façade的 source、relocated
  static/shared；SIMD/Container 必须在自己的合法轮中迁移，随后回放 Mem/Sched 的 selected closure；
- 易用性与文档：Platform guide 以“查单项事实”和“让 owner 匹配完整 requirements”为主，不再教用户
  对不同架构 enum 数值排序；SIMD guide 单独解释 host fact 与 selected kernel target；
- 总预算：本三模块协同轮最多四个 engineering hours；只允许一次 API 实现、一次跨平台机制修正和一次
  direct-consumer 修正。任一适用平台/交付门在预算内不可裁决则进入 `BLOCKED`，不扩建公共仪器；
- 停止条件：有限 facts、三平台探测、无 target policy 的 Platform API 与全部直接迁移通过；Parker/fs/
  topology/memory/VM/NUMA 的旧 P1 证据因产品路径无 diff 只做 provenance replay，不重新计时。

进入 `SCOPED` 后只能完成以上边界的双向审计；完整 capability map 与有限 `FIX` 集冻结前仍不得改产品。

## Audit verdict（`AUDITED`）

正向能力源绑定本机 clean Highway、llama.cpp/随仓 ggml 与 Windows 26100 SDK 的实际 feature ABI；
反向扫描覆盖全部公开 CPU 入口、`cpu.cc/cpu_internal.h`、platform-thread、SIMD、Container、测试、CMake
注释和两层当前文档。结果不是“Platform 和 SIMD 都应提供 ISA”，而是两个不可互换的值域：

| Family | admitted journey | current | verdict / owner |
|---|---|---|---|
| immutable host facts | 自定义 kernel/backend 判断本进程能否执行单项扩展 | 字段式 bool 可用但不可组合、缺事实 | `PARTIAL/FIX`，Platform |
| OS-safe wide state | x86 YMM/ZMM 与 ARM OS capability | x86 主干已有，若干子事实未统一 gate；ARM 三 OS 不齐 | `PARTIAL/FIX`，Platform |
| execution target | HYLE family 报告/选择实际 compiled body | `platform::Isa` 混入 Highway group 和排序 | `ROUTED/FIX`，SIMD v5 |
| thread ISA state | SVE/RVV width、AMX permission | 机制完整但读取旧字段 | `COMPLETE/FIX`，platform-thread v2 |
| non-CPU Platform | topology/cache/NUMA/VM/fs/thread/wait | 本轮无产品 diff | `COMPLETE/KEEP`，旧 receipt |

公开 `CpuFeatures` 的 43 个旧字段逐项追踪后全部迁入新 set；新增 11 个 x86 与 4 个 ARM 推理相关
事实，不复制 APX 或完整 CPU catalog。x86 decoder 以 CPUID registers + XCR0 为纯输入，保证 FMA/F16C/
AVX 家族及 AVX-512 子扩展按相应 OS state gate；AMX 四项明确为 hardware fact。ARM Linux 用稳定
HWCAP/HWCAP2 位；macOS ARM 用 `hw.optional.arm.FEAT_*`，macOS x86 以 Apple 指定的
`hw.optional.avx512f` 修正 lazy XCR0/Rosetta 边界；Windows 用稳定 `PF_*` 数值，未知查询诚实 absent。

本轮冻结 FIX 集：

| FIX | frozen implementation | cost / verification |
|---|---|---|
| `PLAT4-F01` | `CpuFeature` + 两个 `uint64_t` 的 `CpuFeatureSet`；`has/has_all/has_any/set`；删除 `Isa/best_isa/supports/isa_name` | constexpr O(1)、16-byte、无动态状态；header/algebra/invalid tests |
| `PLAT4-F02` | x86 纯 decoder 与完整旧/新增 feature mapping、YMM/ZMM gate、AMX hardware 命名 | 一次固定 CPUID/XGETBV；synthetic register/masked-XCR0 tests |
| `PLAT4-F03` | ARM Linux/macOS/Windows 与现有 RISC-V facts 迁入 set；补 AES/PMULL/DotProd/FP16 和稳定 OS queries | 一次固定 auxv/sysctl/PF 查询；synthetic Linux + 三真机 invariants |
| `PLAT4-F04` | `cpu_feature_name()`、cache、公开指南/架构与旧符号扫描 | cached access/set query O(1)；三交付与 docs |

API freeze 为 `FROZEN`（2026-09-05）：Platform 公开事实没有跨架构排序、best/score/provider 或 kernel
概念；用户按名字查单项或把 requirements 构造成 set。没有未处置 `MUST`，无第三方产品依赖，无
Native FFI slice；本轮 P0 timing 不适用。实现只能执行 F01--F04，发现新的能力族或 OS 机制即退回审计。

## Candidate（`CANDIDATE`）

- product commit/tree：`d015b0de1cf6d5899da8336fef94104974bcd90e` /
  `1456eea7575fa72ce48626a9db13a80f0aa31c09`；公开 facts、纯 decoder、OS probes、测试与文档同树冻结；
- W1-WIN MSVC 19.50 shared 候选预验证：独立 SIMD 闭包的 Platform/SIMD 7/7、Container 闭包的
  Platform/Container 11/11、完整 Platform façade 6/6 全部通过，包含公开头自含；
- route verifier 为 387 operation / 422 scope / 0 pending；documentation 411 Markdown / 1155 links 与
  module closure 25 components 均通过；这只证明 P0 结构和候选完整性，不重签旧 Platform P1 数字；
- `AUDITED -> CANDIDATE` 前 registry snapshot 位于外部验证目录
  `platform-v4-target-boundary/module-status-before-candidate.json`，SHA-256
  `1A7715C50BC342F5DA90469E3C92A3ACA862AAD91EB7E3FF030259886386AF6F`；
- W1-WSL、M1-MAC、三种交付、exact/fat artifact 与最终红队尚未完成，因此不得写 `VERIFIED/CLOSED`。

## Candidate verification（`VERIFIED`）

验证仍绑定 product candidate `d015b0de1cf6d5899da8336fef94104974bcd90e` / tree
`1456eea7575fa72ce48626a9db13a80f0aa31c09`。候选后的
`a47f7c769a4ec553958e7ba58295c3191dd805fb` 与
`9070a3cdeefdff437451bf5af0003fcc3fd5318c` 只给两个 GoogleTest 条件分支补花括号，消除 GCC/
AppleClang `dangling-else`；candidate 到最终验证提交的 CMake、公开头、实现、sample 与 script 产品路径
无 diff。最终验证提交/tree 为 `9070a3cdeefdff437451bf5af0003fcc3fd5318c` /
`64039b8101c44271cb92f44c0373505bd371b6b5`，验证 source archive SHA-256 为
`42AEB10865510981AD5D040CD9C92F54C3A6783CC22358CFC3185FBB38DE9AD9`。

### Correctness、delivery 与 P0 evidence

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| selected Release/Werror | Platform façade `6/6`；相关 SIMD `7/7`、Container `11/11` | immutable archive，排除只依赖 `.git` 的 LOC tooling 后 `45/45` | immutable archive，同样排除 LOC tooling 后 `45/45` | `PASS` |
| sanitizer | Windows 产品回归通过 | affected Platform/Platform-thread/SIMD/provider/Container：ASan `5/5`、UBSan `5/5` | Release 真机覆盖 arm64/NEON | `PASS` |
| source embedding | Platform `1/1` | Platform `1/1` | Platform `1/1` | `PASS` |
| relocated shared SDK | Platform `1/1` | Platform `1/1` | Platform `1/1` | `PASS` |
| relocated static SDK | Platform `1/1` | Platform `1/1` | Platform `1/1` | `PASS` |

source archive 没有 `.git`，所以历史对象型 `hyle_loc_report_test` 明确为 `NOT_APPLICABLE`，不是产品
失败；candidate LOC 已从 Windows canonical Git tree 独立复算并 self-check。SIMD、Container 各自的
三平台三交付也全部通过，因而本轮不是只证明 Platform 单库能编译。

P0 structural verdict 为 `PASS`：探测只在函数局部 static 初始化中执行一次固定数量的 CPUID/XGETBV、
auxv、sysctl 或 processor-feature 查询；缓存读取、单项 membership、all/any 都是固定两个 64-bit word
的 O(1) 运算，无 allocation、normal-path syscall、虚调用或 target 排序。产品候选没有触碰 Platform
原有 wait/topology/cache/VM/NUMA/fs/Parker 算法，故不把旧 P1 点值冒充本轮新证书。

### Usability、依赖与红队

- 用户只有两个问题：查询具名 host/OS fact，或把自己真实编译 TU 的 requirements 组成 set；Platform
  不再回答“哪套 kernel 最好”；SIMD target、Container 私有 requirement 与 per-thread ISA state 均有
  唯一 owner；
- `platform -> PUBLIC core,error` 没有新增产品或第三方边。Platform-thread、SIMD、Container 均真实读取
  同一 Platform cache；Mem/Sched 的 selected closure 通过中央回归，没有错误引入 SIMD/Container 反向边；
- hostile scan 未发现旧 `platform::Isa`、`best_isa()`、`supports(Isa)`、旧 `simd/dispatch.h`、
  `provider::Target` 或跨架构 enum 排序；x86 parent/OS-state、CPUID 7.1/24、Apple lazy AVX-512/
  Rosetta、ARM stable OS queries 与未知 feature/enum 均有边界覆盖；
- 当前没有 SVE/RVV 真机，故本轮不签 scalable-vector timing 或新算法结论；既有编译/正确性资格不被
  夸大为本轮真机性能。macOS static SIMD consumer 的重复 `libhyle_platform.a` warning 来自静态 archive
  正确链接顺序，consumer 成功且无依赖环/运行成本，裁决为 `ACCEPTED/NON_BLOCKING`。

七项 aggregate gate（scope、architecture、correctness、delivery、usability/docs、red-team、engineering
performance）均为 `PASS`，无 medium-or-higher 未决 finding。`CANDIDATE -> VERIFIED` 前 registry
snapshot 位于
`C:\public\project-hyle\_validation\platform-v4-target-boundary\module-status-before-verified.json`，
SHA-256 `AD5A05627E04D1C734CA8F160CAD834FC29565639CFEF3A45A3ABFA7EFBA3A5E`。

## Closure decision 与 terminal receipt

本轮 terminal verdict 为 **`CLOSED`**。Platform 现在只拥有无顺序、可组合、OS-safe 的 host facts；
SIMD target policy、Container TU requirements 与 thread-local ISA state 全部有独立 owner。四项 FIX、七项
aggregate gate 和所有适用三平台/三交付验证均为 `PASS`，没有 `PARTIAL + KEEP`、`INCONCLUSIVE` 或
未执行的适用门。Formal 保持独立 `PENDING`。

已知限制：

- `CpuFeature` 是当前已准入的有限事实集，不是通用 CPU catalog；没有稳定 OS 查询的 feature 诚实返回
  absent，而不是猜测；
- AMX `Hardware` facts 不代表当前线程已经取得 tile permission；SVE/RVV 当前 vector length 仍由
  Platform-thread 所有；
- facts 没有强弱顺序，也不回答哪个 kernel/provider 最快；custom TU 必须声明与真实 flags 一致的完整
  requirements；
- 本轮只签一次有界 probe、O(1) cache/set 的 P0 结构成本；未重签 Platform 旧 P1、scalable-vector
  真机性能、端到端推理或普遍 SOTA。

只有以下新事实可以重开：

1. 具名真实 consumer 需要属于 Platform 且有稳定 OS ABI 的 host fact，但当前集合无法表达；
2. 支持 OS/CPU/虚拟化环境证明某 feature 的 parent、extended-state 或查询 gate 不安全/错误；
3. 公开 fact API/layout/语义、直接依赖、probe 正常成本或 source/static/shared delivery materially 改变；
4. Platform 再次开始选择/排序 SIMD、backend 或 provider，形成第二个 target authority；
5. profile 或 mature same-contract peer 证明 probe/query 存在更优复杂度档或 material normal-path 回归。

candidate LOC self-check（snapshot `d015b0de1cf6d5899da8336fef94104974bcd90e`）：全库代码
443,000；自实现包含测试 112,489；自实现不包含测试 71,135；Platform 实现 4,982 行。LOC 只记录规模，
不证明功能或性能。product candidate/tree 为 `d015b0de1cf6d5899da8336fef94104974bcd90e` /
`1456eea7575fa72ce48626a9db13a80f0aa31c09`；final verification evidence commit 为
`206864ad`。后续只按本节 triggers 重开。

end receipt commit 为 `c3a8b0beb24a9bfce95f9117fc7df6f194b0e966`；registry anchor commit 为
`195aa9a3d2ce90db1ddb249a5319edb351224988`，即首个保存本 `CLOSED` state/candidate 与全部 terminal
registry 字段的人读/机器快照。`VERIFIED -> CLOSED` 前态 SHA-256 为
`144E9E3536B48F05F556C529F037E5399F1A9782254CDA8FF7C60BE1988B1C86`；anchor 校准前态 SHA-256 为
`AEE21C072A7FBFC02FF3D13C865E5B0A4CFA4199CD90E1B5EB50CB4D45F08E7A`。本次校准只记录非自引用 Git
identity，不改变产品、状态或 verdict。
