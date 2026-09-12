# Graph v4：真实 provider 成本事实合同

## Closure card

- component / owner：`graph` / Graph；round=`graph-v4-real-provider-cost-truth`；
- baseline commit：`a21379be51a5ae423ad331e87b73ee7ca9b9d8d7`；上一轮为
  [`graph-v3`](graph-v3-constants-regions.md)；
- trigger 与证据：首个真实、含 opaque JIT artifact 的 oneDNN region 审计证明 v3 把 proposal 的 compile/
  persistent/workspace estimate 与 artifact persistent bytes 的 `0` 同时定义为“none”。oneDNN 公开 ukernel
  API 不报告 JIT code/metadata bytes，因此 provider 只能把未知伪装成零；同时 `dispatch_count` 未定义 CPU
  executor fan-out、worker task 与 leaf call 的计数边界。这会让 CompileReport 和 region cost 失真；
- 正常 consumer 任务：编译 Graph 后，用户能区分“精确为零”与“provider 无法报告”，并能跨 CPU/GPU
  provider 一致理解 dispatch，而不会从未知值推导错误的容量或性能结论；
- 主纵切：oneDNN BRGEMM + HYLE SIMD gated MLP region；反偏置纵切为未来 GPU multi-kernel region，证明
  计数不是为一个 CPU benchmark 定制；
- scope：给无法精确获知的 cost 使用自然的 optional 事实，提升 region extension 版本；定义 dispatch 为
  provider submit 内部的 backend launch/同步 executor episode 数，worker 数和元素/chunk 不计数；更新报告、
  测试和文档；
- non-goals：改变 partition 算法、增加选择策略、动态 shape、跨设备 placement、算子目录、provider 实现；
- 直接依赖/消费者：依赖不变；直接消费者为所有 region provider、CompileReport 和诊断/容量规划代码；
- 性能 tier：`P0`；仅冷路径 POD/optional metadata 与 report copy，steady plan submit 不读取这些字段；
- 调查/纠正预算：一次 API/schema 修改、一次 tests/docs 修正；
- 预声明验证：exact zero/known/unknown round-trip、extension version mismatch、operation fallback unknown、
  report lifetime/copy、region tests、headers、Graph tests/三交付、P0 diff、docs/closure；
- stop：若 optional 进入 steady submit 或使 provider extension 无法版本隔离则阻塞；不得发明 sentinel、把
  unknown 写成零，或为获得数字去 patch/探测第三方 allocator。

## 能力与形状审计

| 能力族 | 层级 | 来源 | 语义与生命周期 | 当前路线 | 覆盖 | 动作 |
|---|---|---|---|---|---|---|
| 可缺省的冷成本事实 | 原子机制 | oneDNN/未来 CUDA opaque artifact | `nullopt=unknown`，`0=精确 none`，值只作报告/冷选择输入 | v3 混淆二者 | `PARTIAL` | `FIX` |
| backend dispatch 计数 | 原子机制 | CPU executor 与 GPU launch 正交纵切 | 一次同步 executor fan-out 或一次设备 command launch 计一；worker/task/chunk 不计 | v3 无定义 | `PARTIAL` | `FIX` |

全部既有公开 Graph journey 保持不变；这是首个真实 provider 触发的诊断合同修正，不是 benchmark-only API。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| G42 | estimate 的 zero/unknown 不可区分 | proposal 的 compile/persistent/workspace 与 artifact/report persistent 改为 `std::optional`；默认 unknown，显式 `0` 才是 none | tests 覆盖三态；用户文档不再把未知说成零 |
| G43 | extension ABI 已改变 | `kRegionProviderExtensionId` version 从 1 提升到 2；旧表不会被 reinterpret | query 精确版本，测试 provider 同步迁移 |
| G44 | `dispatch_count` 边界含糊 | 保留字段名并在 SPI/report/docs 定义 backend execution episode；provider 必须给 exact nonzero 值 | CPU/GPU 示例可无歧义填报，worker count 不污染 |
| G45 | 单-op fallback 没有 persistent-byte provider fact | report 为 unknown，不继续伪报 0；未来 Compute artifact 真正提供事实时再接入 | fallback step `persistent_bytes==nullopt` |

### API / usability freeze

- 普通用户读取 `CompileStepReport::persistent_bytes` 前检查 `has_value()`；值 `0` 明确表示没有常驻字节；
- `RegionCostFacts` 中只有 `estimated_net_savings_ns` 仍是保守选择所必需的已知正值；其余三个可能未知的
  proposal estimate 使用 optional，未知不参与当前 weighted interval score；
- `RegionArtifactProperties::workspace`、copy/sync/dispatch 仍须精确，只有第三方 opaque persistent bytes
  允许 unknown；
- 这些字段只在 compile/report 冷路径读写，不改变 Invocation 大小、workspace 或 submit call graph。

## 七问摘要

1. 不新增第三方；触发来自锁定 oneDNN 3.13.2 的公开能力边界。
2. 修复真实 provider 无法诚实接入的功能缺口；不扩大 Graph 产品范围。
3. 继续复用 Compute/Mem/Container；没有第二套 cost registry。
4. P0：只改变冷 metadata，steady path 零新增成本。
5. Native FFI 仍 N/A。
6. Graph 公开头与三交付必须一起回放；extension 版本阻止错表解释。
7. optional 比 sentinel 自然，guide 明确 unknown/zero 与 dispatch 语义。

Graph 保持 `AUDITED`，本卡只授权 G42--G45。

## Candidate

- commit：`2a2a133e3e4fe60561efe00ad2c225866ce86a5f`；
- public API/layout/算法/常态成本变化：`RegionCostFacts` 的 compile/persistent/workspace estimate、
  `RegionArtifactProperties::persistent_bytes` 与 `CompileStepReport::persistent_bytes` 已改为 optional；region
  extension 精确升至 v2。partition/selection/Invocation/submit 未改，optional 只存在于 compile/report 冷对象；
- 直接消费者迁移：测试 provider 已迁移并覆盖 unknown/exact-zero/known、fallback unknown 和 v1 mismatch；
  真实 oneDNN region 将直接实现 v2，不保留 v1 adapter；
- candidate 后仅证据/文档变化：Windows Graph 逐头、Graph/region tests、源码嵌入、installed shared/static
  独立 consumer 与 shared export hygiene 均已回放；后续 Graph 产品实现不得改写本候选合同。

## 红队与裁决

- 红队结果：`Invocation::submit`、内部 plan step 和 workspace 未取得 optional 字段；只有 compile 生成
  `CompileReport` 时复制 artifact fact。v1 ID 查询明确失败，fallback 保持 unknown，known zero 不被折叠；
- P0：产品提交没有修改 `src/graph`，steady call graph、allocation、workspace 与分支不变；新增状态只在
  proposal/artifact/report 冷 metadata；
- Windows 验证：Graph 3/3（含逐头）通过；source embedding、installed shared/static Graph consumer 均编译
  运行通过；shared hygiene 检查 10 个 HYLE libraries / 911 exports；closure/docs/diff 通过；
- 未决限制：本卡 G42--G45 已完成；真实 multi-op provider、跨平台/sanitizer 与 Graph 终局资格由当前
  `graph_provider_onednn` 及 Graph 总闭包继续承担，不能用本候选冒充 Graph CLOSED；
- reopen triggers：不适用，当前为 `AUDITED`；
- LOC：终局统计；
- 最终状态：`AUDITED`（G42--G45 候选已固定，Graph 总闭包继续施工）。
