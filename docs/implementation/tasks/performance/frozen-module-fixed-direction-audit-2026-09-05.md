# 新 FROZEN 模块固定方向性能审计 — 2026-09-05

<!-- PERFORMANCE EVIDENCE TASK: COMPLETE -->

## 范围与判别

- baseline：`ddf0cb89c2bb17028538a2e92dac428d77fd5b4b`；
- 状态输入：[`module-status.json`](../../module-status.json) 中带新 `freeze_confirmation.marker=FROZEN`
  的 8 个组件；
- “固定方向落后”只表示一个 workload 的全部必需 order/placement/allocation strata 中位
  `advantage_ratio < 1`。它不是 `LOSE` 的同义词，也不覆盖 CI、material margin、comparison class 或
  产品两把钥匙；
- 同时旁审 retained 报告中仍明确写着的 HYLE `LOSE` 或单平台下界落后，避免旧数字被藏在历史段落；
- 本 task 只复核既有合同、raw summary/control、当前/retained source 与相关 object code，不重跑采样，
  不修改产品、benchmark、旧 bundle、旧 contract 或 `module-status.json`。

## 全量筛查

| FROZEN 组件 | 性能级别 / 当前证据 | 固定方向 HYLE 点值落后 | 筛查结论 |
|---|---|---|---|
| Core | P0 structural | 无 peer timing | 不适用；没有伪造纳秒排名 |
| Container-intrusive | P1 | 无 | `forward-arena` 为 `0.9823–1.0089`，方向随 strata 翻转 |
| Util-core | P1 | short state transition，`0.9986145–0.9999718` | 进入机制复核 |
| Error | P1 | TaskGraph success，`0.9923150–0.9986028` | 进入机制复核；另旁审 v3 fixed-backing copy |
| Obs | P1 | model-load structured route，`0.9715451–0.9820841` | 进入合同/机制复核 |
| Time | P0 registry / retained P1 host evidence | 无 | host 为 `0.9869–1.0084`，方向翻转 |
| Timer-queue | P1 | 无 | 8/8 strata 均 `WIN` |
| Mem-reclaim | P1 | 无 | 两项分别为 `1.3649–1.3812` 与 `1.3911–1.4028`，均 `WIN` |

因此，当前证据没有被漏看的其他固定方向行。Container-intrusive 和 Time 虽各有低于 1 的 stratum，
但它们不是固定方向；继续挑一个顺序解释输赢反而违反仪器合同。

## 逐项复核

### Util-core：名义 `0.0028%–0.1385%`，没有产品机制

- comparison 为同合同：短、无争用的 `FiberStackPool` 状态转移，对手是 oneTBB `spin_mutex`；
- 四个 CI 全部跨 1；同一 bundle 的 A/A 中位 ratio 为 `0.9982764 / 0.9999371`，已经出现同方向且不小于
  产品行的偏移；
- retained object
  `C:\hyle-scratch\util-core-v2-perf-build\bench\util_core\hyle_util_core_bench.dir\Release\util_core_bench.obj`
  SHA-256 为 `F071929882607D8B60DAC125F6770F5970ED0634203B9FDC14F2A7112D042FB2`。反汇编确认本 workload
  经过的 relaxed test、`xchg` acquire、状态更新和 release store 指令骨架相同；差异只在未走的 contention
  backoff loop；
- 分类：**仪器/代码布局分辨率内无机制**。保持
  [`benchmarks-util-core-v2`](../../evidence/performance/benchmarks-util-core-v2.md) 的 `PARITY/PASS`，
  不满足产品修改或重开所需的任一 material 消费钥匙。

### Error current success：名义 `0.14%–0.77%`，没有产品机制

- 四个 current ratio 为 `0.9986028 / 0.9923150 / 0.9985339 / 0.9929537`，全部 CI 跨 1；A/A 为
  `0.9999113 / 1.0009608`；
- retained source-consumer object SHA-256 为
  `9D464BD5273FF9C36CA356F14BDBD19A2AEB87A46C03F2A3CE636EEB051CA0D3`。HYLE/Abseil 的 noinline
  success node 与 64-node evaluate 热路逐指令相同，OK carrier 都使用同一个 one-word 编码；成功路径
  不执行两侧不同的析构 slow target；
- v4 同一真实 source-consumer 的四 strata 为 `0.9984–1.0018`，方向没有跨版本固定；
- 分类：**代码布局/运行分辨率内无机制**。保持
  [`benchmarks-error`](../../evidence/performance/benchmarks-error.md) 的 `PARITY/PASS`；不能为这组点值
  改 Status/Result layout 或 TaskGraph 形状。

### Obs model-load：真实额外同步税，但不是同合同实现落后

- 四个 route ratio 为 `0.9740009 / 0.9803485 / 0.9715451 / 0.9820841`，完整 structured callback 点值
  慢 `1.8%–2.85%`；A/A 为 `0.9951576 / 1.0018098`，不能解释全部差距；
- 两侧 record、registry、adapter 与 195 条输出相同，但 HYLE `LogIntervalLimiter` 对 768 次 check 做
  relaxed atomic load，并在 192 次 interval admission 上用 CAS 保证 racing caller 精确至多一次；llama.cpp
  source-derived gate 因 stage callback 是 single-owner，只做 scalar load/store。MSVC binary 中也能看到
  HYLE loop 的 `lock cmpxchg`；
- 这是唯一明确找到的机制，但 comparison class 必须是 candidate-stronger `NATURAL_ROUTE`，不能把
  `1.8%–2.85%` 写成 limiter 实现对同合同 peer 的 `LOSE`；
- HYLE 是否应再提供一个按 single-owner/approximate 语义命名、无 CAS 的 interval state，是有真实 llama
  provenance 的**候选产品档位**。当前差距未超过预冻结 5% route budget，也没有 profile 证明 material，
  因而本审计不把它升级成 Obs reopen trigger；若后续实际推理 consumer/profile 证明该档位必要，应以
  capability task 退回 API freeze，而不是把 scalar control 偷换成现有并发精确类型的“优化”。

### Error v3 fixed-backing copy：测量稳定，归因不成立

- retained v3 四个 ratio 为 `0.9200565 / 0.9250407 / 0.9200341 / 0.9202135`，HYLE 约慢
  `7.5%–8.0%`；99.26% CI 均低于旧 5% 边界，timer floor 53.96x，A/A 约
  `0.9989345 / 0.9980798`。因此“仪器确实测到稳定差距”是历史事实；
- 该 fixture 永远复制各自同一个 cache-hot 29-byte backing，allocation 顺序固定且未记录或分层对象
  address/cache-line/page offset。它不是 copy 分布或正常 ownership handoff；
- retained benchmark object SHA-256 为
  `6E291F69F687F6823E8B936EC1E97B0539C33361C44758F16B308EB5FB57F2B2`。两侧 copy lambda 的 retain、
  sink 与 heap-tag branch 逐指令相同；只在析构 call target 不同。HYLE/Abseil release 都是 acquire load +
  非末次 acq_rel `lock xadd`，HYLE 非末次路径还少了 peer 的 stack prologue/epilogue，汇编不存在能支持
  8% HYLE 实现落后的机制；
- 后续三平台自然 owning-message copy 又得到 HYLE `1.038x / 1.045x / 1.314x`，方向反转；
- 分类：**保留的未归因 data/code-placement diagnostic**。原始数字不删除，但 comparison/product
  verdict 继续作废。只有地址分层诊断或真实 copy-heavy consumer 同时复现机制和 material 影响才恢复。

### Error 对较弱标准下界：不属于同合同固定落后

历史 W1-WIN 的 code-only construct 对 `std::error_code` 名义慢约 4%，四帧 Result OK 对
`std::expected<int,std::error_code>` 慢约 16%；Linux/M4 的后一项则由 HYLE 胜出约 7%/9%。标准下界只
携带 category/value、平凡析构，不能替代 HYLE 的 canonical message/native/context carrier，因此它是
平台特定的较弱能力成本提示，不是同合同 peer 失败，也没有跨平台固定方向。该事实继续留在 Error 用户
选型与历史报告中，不据此削弱默认错误合同。

## 终态裁决

1. 当前新 FROZEN 集合中，**同合同且找到 HYLE 产品落后机制的固定方向行：0**。
2. 两条同合同 current `PARITY`（Util-core、Error）确有固定点值方向，但一次有界机制检查均不支持产品
   缺陷；保留原始点值，不追加姿势、不改产品。
3. Obs 有真实额外同步机制，但两侧线程保证不同；它只产生 route-level 语义税和一个未来单 owner 档位
   候选，不产生 primitive `LOSE`。
4. Error v3 copy 是唯一 retained exact `LOSE`；测量事实保留，产品归因因固定 address 形状、反向后续
   结果和不支持差距的汇编而继续拒绝。
5. 本 task 不满足任何现有产品 reopen trigger，8 个 registry `FROZEN` marker 均保持不变。今后的 P1/P2
   收口必须按新增的固定方向复核规则逐项留下同样结论，不能只写“在 5% 内”。
