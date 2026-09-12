# Compute Provider oneDNN v4：Graph region 内部 kernel 复用边界

## Closure card

- component / owner：`compute_provider_onednn` / Compute；round=`compute-provider-onednn-v4-region-kernel-seam`；
- baseline commit：`a21379be51a5ae423ad331e87b73ee7ca9b9d8d7`；上一轮 runtime 迁移记录为
  [`compute-provider-onednn-v3`](compute-provider-onednn-v3-runtime-replay.md)；
- trigger 与证据：已进入有限审计的 `graph_provider_onednn` 需要复用 v2 已验证的 x64 BRGEMM ukernel。
  复制 `onednn_provider.cc` 的 kernel/tiling/scratch 实现会产生两个可漂移 owner；让 Compute 反向依赖
  Graph 又违反固定依赖方向。这是一个具名直接消费者证明的私有实现 seam 缺口；
- 正常 consumer 任务：既有 direct Compute 用户仍通过 `hyle::compute::onednn::provider()` prepare/submit
  单个 MatMul；Graph-only provider 在同一 SDK 内私有复用完全相同的 prepared BRGEMM mechanism；
- 主产品纵切：f32 gated MLP 的 gate/up/down projection；反偏置纵切为既有 direct dense f32 MatMul，确保
  抽取不会把通用 kernel mechanism 特化成 Graph API；
- 跨模块 journey：本 factor 继续拥有 oneDNN lock、BRGEMM artifact、tile 与 scratch mechanism；Graph
  provider 只经未安装的内部 C++ seam 借用。公开 Compute/Graph API 均不出现 oneDNN handle；
- scope：抽取一个 opaque、allocator-owned BRGEMM plan 及 prepare/destroy/task execute/query seam；保留
  direct provider 的 route policy、mapping、Completion 和公开 API；为 shared factor 显式导出这条私有
  C++ seam，并验证 static/shared/source 链接；
- non-goals：新增公开 API/算子/dtype/layout、改变 M<16 fallback、weight packing、oneDNN Graph API、
  AArch64 oneDNN、修改 dependency lock、Native FFI；
- 直接依赖 / 直接消费者：公开边仍为 `compute_provider_onednn -> compute`；新直接私有消费者边为
  `graph_provider_onednn -> compute_provider_onednn`，绝不反向；oneDNN 依赖与 package registration 仍只有
  本 factor 一个 delivery owner；
- 性能 tier：`P0` refactor gate。若 single-op 生成代码、task 数、scratch、mapping 或 execute call graph
  发生语义变化，原 v2 P1 失效并在候选后定向重跑；否则只回放 allocation 与 codegen；
- 调查/纠正预算：一个内部 header + 一个 implementation split，direct provider 最多一轮迁移、一轮修正；
- 预声明验证：原 provider 全部 normal/holdout/fallback/workspace/concurrency/nested tests、六 shape warm
  allocation、内部 seam 的 pair execute、shared imports/exports、source/static/shared、P0 diff、docs/closure；
- stop：若必须暴露 dnnl 类型/头、复制 JIT/tiling、增加 hot indirect allocation/lock，或安装私有 header
  才能交付，则停止并重新裁决独立 build mechanism，不能用重复实现绕过。

## AUDITED 结论与有限改动集

源码审计确认 `OneDnnPlan`、`create_kernel()`、tile arithmetic 和每 task execute 是唯一需要复用的机制。
Graph region 的 gate/up 具有完全相同 shape，可共用一个 immutable plan，并在一次 hardware-context 区间内
连续执行两组权重；down 使用第二个 plan。公开 `ParallelExecutorRef` 已明确同步、不可失败、支持嵌套进展，
不需要新线程抽象。

| ID | 问题与证据 | 允许改动 | 验收 | 当前结果 |
|---|---|---|---|---|
| OD40 | BRGEMM mechanism 被封在匿名 namespace | 新建未安装 `brgemm_internal.h/.cc`，用 opaque plan 和 POD task range 隔离 dnnl | Graph consumer 不 include dnnl；一个 artifact owner | 已在 `6306ab2f` 抽取；真实 Graph consumer 待 GP20--GP26 |
| OD41 | direct plan 把 executor context 与 kernel artifact 混在一起 | kernel plan 只保存 immutable shape/tile/JIT；execute 显式接收已借用 executor | direct 生命周期与 task 数不变 | 已完成；原 route/task/workspace tests 通过 |
| OD42 | gate/up 若分别调用 single-op submit 会重复 mapping 与 executor fan-out | 私有 seam提供同 plan、共享 left 的 pair-task execute；不包含 activation/Graph 类型 | 一次 hardware context 内两次 BRGEMM，warm 零分配 | pair 差分、task range 与单 episode test 通过 |
| OD43 | shared Graph provider 必须调用 mechanism | 只导出未安装 header 声明的 C++ 符号；中央图登记单向 PRIVATE 边 | PE/ELF/Mach-O 无 C/FFI 入口，installed shared 可装载 | shared producer 已通过；installed Graph consumer 待 GP26 |
| OD44 | refactor 可能漂移既有路由 | direct provider 保留 M<16、N tile、workspace、map/flush 与 Completion 逻辑；增加差分/codegen 回放 | 原测试、allocation 和 P0 账单一致 | 11 个 provider tests、六 shape warm allocation 与新 P1 通过 |

公开 C++ API、默认值、对象布局和用户文档不变。本卡只准入 OD40--OD44；量化、post-op、packing 或新的
route policy 都不属于本轮。`compute_provider_onednn` 保持 `AUDITED`，允许下一提交修改其产品实现。

## 实现检查点（尚非 lifecycle Candidate）

- 产品提交：`6306ab2f058d9df1daadb06bb3d3a34603df6a5f`；P1 合同依赖选择修正在
  `f130c42ef09ae0bc4763eb6b4101623ac0cd8826`；
- public API/layout/算法/常态成本变化：公开面、路由、tile、scratch、mapping 与 Completion 不变；冷路径
  多一个小型 wrapper allocation，失败时先销毁已生成 plan；warm submit 每 episode 多一条同 factor 的
  非虚内部调用边，不增加 allocation、search、JIT、copy、lock 或 task；
- 直接消费者迁移：待 `graph_provider_onednn` 实现；
- lifecycle Candidate：尚未进入；OD43 的真实 shared consumer 与 Graph 直接边仍待完成。

Windows shared 定向 provider/header/composition 为 `3/3 PASS`；新增 fault injection 精确命中 plan 已创建、
prepared wrapper 分配失败的路径，终局 allocation/deallocation 相等。static allocation probe 在六个推理
shape 上仍为 warm `global new=0`。由于源码 call graph 确实改变，未沿用 v2 P1：新鲜 W1-WIN、MSVC
19.50、Release/static/LTO-off 的 n=15 回放中，八个交叉 strata 的 median ratio 为
`0.9904--1.0207`，全部 5% non-inferiority `PASS`；七项 `PARITY`，一项方向 `INCONCLUSIVE`，没有固定
loss。A/A、known-delta、timer-floor、placement、artifact/build identity 全部通过。证据位于
[`compute-onednn-provider-p1-v1-f130c42e-w1-win`](../../../../evidence/performance/compute-onednn-provider-p1-v1-f130c42e-w1-win/)。

首次短 probe 在计时前暴露合同只选择 provider factor、却漏选 benchmark 直接调用的 CPU backend；
`f130c42e` 只把 `compute_backend_cpu` 加入 `HYLE_DEV_FACTORS`，没有向 shipping provider 增加错误产品边。

## 红队与裁决

- 红队重点：私有 ABI 是否意外安装、Graph 反向依赖、两份 oneDNN registration、single-op codegen 漂移；
- 已排除：公开头/dnnl 泄漏、direct route/task/workspace 漂移、pair 每 task 两次 hardware-context、第二次
  metadata allocation OOM 泄漏、warm allocation 与 material provider 边界税；
- 未决限制：真实 Graph source/static/shared consumer、Linux/sanitizer 与 Graph 组合红队；
- reopen triggers：不适用，当前仍为 `AUDITED`；
- LOC：终局统计；
- 最终状态：`AUDITED`。
