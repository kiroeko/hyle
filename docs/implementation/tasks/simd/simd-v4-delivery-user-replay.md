# SIMD v4 交付形状用户回放 — 2026-08

<!-- CAPABILITY REPLAY RECORD: COMPLETE -->

> 本页以“需要 SIMD 加速能力的外部 C++ 用户”回放已经 `CLOSED` 的 `simd-v4-expert-sdk`。它是
> product-non-mutating 的存量审计，不替代
> [`module-status.json`](../../module-status.json)，也不把文档可发现性问题伪造成新的产品轮。

## 回放合同

- component / replay：`simd` / `simd-v4-delivery-user-replay`；
- reviewed state / round：`CLOSED` / `simd-v4-expert-sdk`；
- closure task：[`simd-v4-expert-sdk`](simd-v4-expert-sdk.md)；
- candidate / tree / end / registry anchor：
  `1e543dec85ad6cef411f4499f7449294a2507b1c` /
  `4e4e82e5e22e09f55c7af165b45de9d4dfbb4896` /
  `184610788421fe83c5c5e176a4875e530aebe2e3` /
  `184610788421fe83c5c5e176a4875e530aebe2e3`；
- replay base / evidence commit：
  `56bfbed98662b6602d61af91672a09121576da4d` /
  `61a72b9d18358e73b51426a6fedba2e85aa9eaf9`；reviewer：Codex；日期：2026-08-31；
- 用户画像：需要 whole-buffer 算法的普通应用用户，以及 whole-buffer 产品面没有所需 primitive、需要在
  自己 kernel 中直接使用 Highway/xsimd 的 C++ 性能用户；
- tier / instrument：`P0 / closure-correctness-v1`。本轮不改变算法、route 或 runtime dispatch，timing
  budget 为 0，只裁决交付层次、构建成本结构和真实 consumer 可用性；
- 调查预算：30 分钟；一个新的 direct-expert Windows consumer，复用 immutable candidate 的三平台
  source/static/shared package evidence，并逐项检查普通、direct、fixed、fat 四条用户路径；
- 范围：入口可发现性、最小 CMake 写法、provider 类型边界、target/ISA 粒度、源码/静态/动态同形、
  relocation/offline、普通用户隔离和已声明负空间；
- 非目标：重测 387-operation 性能、修改 route winner、包装完整 vendor register API、增加 compiled
  contrib、Native FFI、稳定跨工具链 ABI，或因示例偏好重开产品。

candidate 到 evidence commit 的 `include/hyle/simd/**`、`src/simd/**`、SIMD CMake/helper 与两个 SIMD
sample 产品路径为零 diff。中间变化只有 SIMD terminal 记录、独立 build/package Unix 入口以及本次用户
文档澄清，因此本轮审查的是同一个产品 identity。

## 用户从需求到入口

| 用户任务 | 最小入口 | 用户承担的部署事实 | 交付判断 |
|---|---|---|---|
| 未知 CPU 上调用常用连续数组算法 | 链接 `hyle::simd`，调用 pointer/span whole-buffer API | 批量调用；遵守 alias/精度/workspace 合同 | `NATURAL`；普通路径不暴露 vendor 头或 target 细节 |
| HYLE 没有该语义，但当前 target flags 已合适 | 单独 OBJECT target 链接 `hyle::simd_expert`，直接写 `hwy::` / `xsimd::` | provider 类型不跨 TU；该入口不替用户探测 CPU | `NATURAL`；三种交付同名、无 HYLE runtime wrapper |
| kernel 只运行在已知 ISA | 对专用 OBJECT target 调 `hyle_target_simd(... ISA ...)` | artifact 的最低 ISA 由用户明确选择 | `NATURAL`；宏/模板单态化，固定路径无 selector |
| 自定义 kernel 要覆盖未知 CPU | `hyle_add_fat_kernel` + provider-coherent impl header | 显式列 canonical variants，pointer/POD entry | `ADVANCED BUT PROPORTIONATE`；每个 whole-kernel 调用一次缓存入口 |
| 整个 HYLE artifact 只面向一种 ISA | 配置 `HYLE_SIMD_EXACT_TARGET` | 整个 HYLE binary 的部署最低 ISA 改变 | `NATURAL`；与单个 consumer kernel 的 fixed helper 明确分层 |

这里没有把 Highway/xsimd 数百个原语重新命名进 `hyle::simd`。普通语义由 HYLE 拥有，provider-native
能力由 expert target 交付；这既避免 HYLE 复制一套不完整 DSL，也让 vendor 升级继续由上游维护。用户
只有在进入自定义 kernel 时才需要理解 ISA、TU 和 provider 类型边界。

## 实际 consumer 回放

本轮新建了一个只有 `main.cc` 的 consumer：它只链接 `hyle::simd_expert`，同时包含
`<hwy/highway.h>` 与 `<xsimd/xsimd.hpp>`，用 Highway 完成 8-lane 可缩放求和并检查
`xsimd::batch<float>::size`。它没有链接 ordinary `hyle::simd`、没有调用 fixed/fat helper，也没有手工
增加 provider include path 或 definition。

| journey | 当前结果 | 证据边界 |
|---|---:|---|
| source `add_subdirectory()` direct expert | `PASS`，1/1 | MSVC 19.50 / Ninja；只构建 consumer 与 INTERFACE dependency |
| relocated shared SDK direct expert | `PASS`，1/1 | candidate shared prefix；configure/build/run 均无需下载或手工 include |
| relocated static SDK direct expert | `PASS`，1/1 | candidate static prefix；与 shared 使用完全相同 CMake/source |
| ordinary + fixed + fat package journey | inherited `PASS` | closure task 的 Windows/WSL/macOS 三平台 × source/shared/static，均 1/1 |
| ordinary target 的 expert 隔离 | inherited `PASS` | package sample 对 link/include property 作负向断言；普通 target 不传播 vendor surface |
| provider ABI / relocation / offline / license | inherited `PASS` | consumer 侧实例化，边界为 pointer/count/POD；SDK 自带锁定 headers 和许可证 |

当前 direct replay 位于
`C:\hyle-scratch\simd-v4-expert-sdk\user-delivery-replay\{source,shared,static}`；scratch 只保存本次机器
事实，不是 canonical 产品来源。candidate 的跨平台 fixed/fat 证据仍以 closure task 为权威，不把本次
单机 replay 冒充新的三平台证书。

## 用户红队

| finding | severity | disposition / evidence |
|---|---|---|
| 用户看不出“不固定 ISA、只取得锁定 provider 头”的最小路径 | documentation | `FIXED`：evidence commit 在指南加入 direct `hyle::simd_expert` OBJECT target；三种交付真实 consumer 1/1 |
| 指南把 target 级 helper 口语化为“给该 TU 设置 ISA”，用户可能对多源 app target 整体施加 AVX2 | documentation | `FIXED`：明确 helper 作用于整个 CMake target，并给出专用 OBJECT target 写法；CMake 行为和公开合同未变 |
| ordinary、direct、fixed、fat 的选择可能混成一套“自动最快”承诺 | usability | `DISPROVED AFTER CLARIFICATION`：决策表分别写清 runtime selection、consumer flags、fixed minimum ISA 与 fat boundary |
| static/shared 用户可能拿不到与源码用户相同的 native headers/helper | delivery | `DISPROVED`：direct 本轮 1/1；fixed/fat 候选三交付 1/1；target 名均为 `hyle::simd_expert` |
| expert target 可能给普通用户带来 parse/runtime 成本 | performance/build | `DISPROVED`：普通 imported target 负向断言；expert 为 opt-in INTERFACE target，无 runtime artifact |
| fat helper 的 variant 声明和 selector 比 ordinary API 复杂 | usability | `ACCEPTED`：它是未知 CPU 的自定义多版本 kernel，不隐藏 ABI/variant 列表；普通和 fixed 用户不承担这份复杂度 |
| header 可见可能被误认为完整 Highway compiled contrib 已交付 | contract | `DISPROVED`：user/architecture/closure 均明确只承诺 header surface；需要上游 `.cc` 的产品另行准入 |

没有 finding 改变公开合同、算法、layout 或正常 workload。两项改动只是把架构和 CMake 已经存在的事实
变成可复制的用户选择；因此不满足 `PRODUCT_TRIGGER`。

## 裁决

- 产品 verdict：`CONFIRMED`；`simd` 保持 `CLOSED`，产品实现改动为 0，没有 `PRODUCT_TRIGGER`；
- 交付 verdict：用户可以从简单到高级依次选择 ordinary、direct expert、fixed target、fat kernel；四层
  没有 static/shared/source 拼写分裂，也没有把 vendor ABI 塞进 HYLE binary；
- 易用性 verdict：ordinary 与 direct/fixed 路径可由短 CMake 片段完成；fat 路径是显式高级接口，其
  必要复杂度与未知 CPU 多版本部署相称；
- 性能 verdict：ordinary 保持一次 whole-buffer 间接调用，direct/fixed 没有 HYLE runtime selector，fat
  保持每次 whole-kernel 调用一次缓存函数入口；本轮没有新增 timing/SOTA 声明，Formal 仍为 `PENDING`；
- 文档 verdict：[`SIMD 使用指南`](../../../user/simd-usage.md)现在先给选择表，再分别给 direct/fixed/fat
  写法和陷阱；[`交付指南`](../../../user/distribution.md)明确 helper 的 target 粒度；
- 重开条件仍以 registry 和 closure task 为准：真实 consumer 复现交付/API/correctness 缺陷、锁定 vendor
  升级影响合同、支持 ISA/toolchain 改变，或 concrete compiled-contrib/foreign-runtime 需求通过准入。
