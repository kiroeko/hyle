# Graph v5：可移植的 region admission 与确定性选择

## Closure card

- component / owner：`graph` / Graph；round=`graph-v5-region-admission`；
- baseline commit：`bcf25f2c7fa91ef57607d489ca9634fa18e68628`；上一轮候选事实保存在
  [`graph-v4`](graph-v4-real-provider-cost-truth.md)，不回写其 G42--G45 记录；
- trigger 与证据：首个真实 oneDNN gated-MLP region 的同叶短 preflight 在 W1-WIN 上随执行顺序于约
  `-4%..+4%` 反转，不能证明一个跨机器、跨 shape 的正纳秒下界；当前 prototype 却只能硬编码
  `estimated_net_savings_ns=1` 才能被默认 policy 接纳。ggml backend scheduler 以显式 backend 顺序、
  storage locality、capability/offload 规则划分，而不是要求 provider 从微基准制造可移植纳秒常数。
  这是 Graph 冷路径选择合同缺陷，不是继续增加样本可以修复的 provider 性能问题；
- 正常 consumer 任务：调用方显式把 region provider 排入 Context 后，Graph 可按确定、可解释的结构策略
  选择它；若部署方只允许有正量化 estimate 的 route，也可选择严格策略。provider 不需要为了进入产品
  伪造纳秒值，未知与精确零仍可区分；
- 主产品纵切：oneDNN BRGEMM + HYLE SIMD 的四节点 gated MLP；正交纵切为未来 CUDA/Metal 的多 kernel
  region，二者的 launch/dispatch 和 shape 成本都不能由一台 CPU 的常数代表；
- 跨模块 journey：Compute Context 保存显式 provider 顺序；provider 提交 capability 与可选 estimate；
  Graph 冷路径拥有 admission/interval selection；CompiledReport 只报告事实；steady plan 不读取 cost；
- scope：把净收益 estimate 改为 optional；用两个意义不同的 policy 取代含糊的 `kConservative`；默认
  policy 对调用方显式提供的 region 做“最大覆盖、较少 dispatch、provider 顺序”确定性选择，严格 policy
  只接受正 estimate 并按 estimate 选择；迁移 tests、真实 provider、guide 与 extension version；
- non-goals：扩大算子目录、修改 region pattern/kernel、运行时自适应/autotune、进程全局 profile、跨设备
  placement/copy、动态 shape、用 benchmark 数值写回 public API；
- 直接依赖/消费者：依赖不变；直接消费者是所有 region provider、Graph compile/report 和后续 GPU
  provider。`graph_provider_onednn` v2 prototype 是首个迁移者；
- 性能 tier：`P0`。只改变 compile-time POD 与 O(Q log Q) interval score；steady Invocation/submit 不变。
  oneDNN 短 preflight降为触发/诊断，不签胜负；
- 机器/验证：Windows Graph/region tests、public headers 与 source/static/shared；WSL sanitizer 与两台 Apple
  在 Graph 候选终局回放。固定 material margin 不适用于本轮 P0；
- 调查/纠正预算：ggml 和一个 graph-partition peer 各一次源码对账；一次 public schema/extension bump、
  一次 selection 实现、一次 tests/docs 修正；
- 预声明验证：unknown/zero/positive estimate、两种 policy、重叠/相邻 proposal、provider 顺序、预算、
  invalid enum、report round-trip、extension mismatch、P0 steady diff、三交付和红队；
- stop：若结构策略必须读取热路径 profile、引入全局 registry，或不能在 O(Q log Q) 内给出确定结果，则
  `BLOCKED`；不得把测不准的纳秒值改名后继续使用。

## AUDITED 结论（2026-09-13）

### 能力与形状

| 能力族 | 层级 | 当前问题 | 冻结动作 |
|---|---|---|---|
| provider 显式 opt-in region | 复合机制 | Context 已有显式顺序，但默认 admission 又要求不可移植的正 ns | `REPLACE` 为结构优先 policy |
| 有 estimate 的保守部署 | 原子策略 | `0` 同时承担 unknown/无收益 | estimate 使用 `optional`，严格 policy 只收正值 |
| 重叠区间选择 | 冷路径算法 | 只按纳秒和覆盖节点加权 | 两个 policy 分别使用独立、可解释的 lexicographic score |

公开入口只有 `CompileOptions::region_selection` 与 provider SPI 的冷成本字段；GraphBuilder、Graph、
Executable、Invocation 和 steady submit 不改变。调用方显式安装 region provider 本身就是 opt-in，默认
策略不再假装拥有跨机器 timing oracle。

### 有限改动集

| ID | 问题 | 允许改动 | 验收 |
|---|---|---|---|
| G50 | 正 net ns 是必填谎言 | `estimated_net_savings_ns` 改为 `optional<uint64_t>`，extension v2→v3 | unknown/0/positive round-trip，旧表不解释 |
| G51 | `kConservative` 混合 capability 与 cost | 替换为 `kPreferProviderRegions`（默认）和 `kRequireEstimatedBenefit` | 默认接纳 unknown；严格模式只接纳正 estimate |
| G52 | 一个 score 无法表达两种策略 | prefer 模式最大覆盖、再最少 dispatch、再 provider 顺序；strict 模式最大正 estimate、再覆盖 | 重叠、相邻、多 provider 决定稳定且 O(Q log Q) |
| G53 | oneDNN prototype 伪造 `1 ns` | 删除常数并报告 unknown；保留精确 dispatch/copy/sync/workspace facts | 默认仍选择；strict 下自然不提案，绝不补 benchmark-only op |
| G54 | 文档/错误仍承诺旧语义 | 更新公开注释、guide、active provider task 和 invalid enum | 三交付、逐头、docs/closure 通过 |

### API / usability freeze

```cpp
// 默认：把显式安装的 region provider 当作 opt-in，确定性优先覆盖。
auto compiled = hyle::graph::compile(graph, context, options);

// 部署方要求 provider 给出正的量化 estimate 时显式选择严格策略。
options.region_selection =
    hyle::graph::RegionSelectionPolicy::kRequireEstimatedBenefit;
```

`kPreferProviderRegions` 的完整顺序是：最大化被 region 覆盖的逻辑节点；覆盖相同时最小化 backend
execution episodes；仍相同时优先 Context 中更早的 provider，最后用 route/proposal 顺序稳定裁决。
这不是性能胜负声明。provider 仍必须只为自然、正确、由自己愿意承担的 route 提案，并在 report 中披露
copy/sync/dispatch/threading。严格策略才把正 estimate 当 admission 与第一 score。

本卡已完成 scope/audit，Graph 保持 `AUDITED`，只授权 G50--G54；候选前不得扩入 symbolic shape、
placement 或 operator catalog。

## Candidate

- 尚未实现。

## 红队与裁决

- 当前状态：`AUDITED`；等待 G50--G54 实现和预声明验证。
