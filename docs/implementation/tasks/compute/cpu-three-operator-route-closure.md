# CPU 三项成熟接缝、规则绑定与有限清退

<!-- MODULE REVIEW: SCOPED -->

- owners：compute_provider_ggml、compute_provider_native；baseline `a98bdbf239d2854fa04acbbf2d43793b13cb311e`。
- trigger：用户批准有限接线/清退；[合格新版复比](../../../../evidence/performance/cpu-three-operator-current/README.md)
  已证明六个实际劣势，并找到 ggml F32 fused 未接的具体源码反例。
- consumer：Qwen3.5-2B hidden2048 的 prepared token/small-batch RMS；正交 plain tensor materialization；
  已准入 CPU attention/GQA 和实际 causal KV replay。不是按一个 benchmark 新造算子或公共 API。
- 只处理 CPU 三项，不扩自研优化，不恢复旧187域新版资格，不改 GPU core、Graph planner、SIMD/Mem 或公共仪器。
- 严格内存/动态整域按具体需求裁决；没有 consumer 的重复数值路径可明确撤回，不作为免退役理由。

## 同合同与有限 FIX（审计后冻结）

1. G1：仅 CPU-only `ffa4e8b8` 接缝使用原 F32 flash-attention、F32 accumulation，原布局/output/mask成本完整计入。
   不把 F32 KV 转 F16 偷换精度；保留当前 BF16/旧 mixed SDK 的已资格路径。mask 存储/构建溢出与空shape合同审计。
2. N1：仅当前已测 i9-13900K/MSVC195035728/Release/单线程/新版 CPU-only core 绑定17个 Copy/RMS WIN格。
   相同 op、dtype、gamma、epsilon、dense物理形状和rows条件匹配；其它域成熟优先，执行无搜索/计时。
   已测F32 RMS rows32不获优选；Graph叶规则仍不能拆成熟region。旧core只沿旧记录，不承接新证据。
3. N2：审计真实调用后清退 Native strided Copy 重复循环；保留 dense Mem接缝和有用途的 dense dynamic Copy/RMS编译复用。
   若无具名消费者要求 bounded-strided Native-only能力，则明确撤回该子能力；普通strided由已验成熟接缝接替。
4. G2/N3：成熟 F32 fused 正确性/资源/实际ordinary两入口通过后，固定候选只复比原4个SDPA格。
   无 Native 完整优势则整项退役；有优势只保留精确有证子域，不再优化或扩性能调查。
   mask/布局/严格内存差异单列，不以原库unknown私有内存保留普通重叠数值实现。

公开 API、ownership/完成/并发/错误、Invocation layout、路线和依赖图不扩。若源码反证要求改变这些合同，
退回AUDITED重新定界，不能用弱化数值检查来签资格。源码审计完成前不改产品。

## 验证与预算

- Windows canonical，Linux WSL只作前向验证副本；源码/static/shared低并行build2、test1，重型串行。
- owning ggml/Native、直接Compute/Graph CPU consumer；source普通入口与installed static/shared批量24域规则消费。
- F32 fused：B1/B2、BSHD/BHSD、causal/noncausal、GQA、Dk/Dv、rows1/32与长KV、空/边界、变输入replay、并发、close、OOM；
  复用未变底层资格，新增语义必须实跑，保持旧F16/BF16回归。三交付同时验证规则指向实际实现。
- 资源：dense bounded Copy/RMS复用；Native strided拒绝和普通成熟bitwise物化；有界预算/错误/生命周期实际消费。
- P1：W1-WIN，原4个SDPA形状，单P核CPU2/above-normal，实际完整submit+wait、1线程；两执行/两分配顺序，
  fresh workload A/A与公共controls，5% practical门，15独立进程/stratum，唯一一次，无n31升级。
  Copy/RMS仅复用tree/算法/合同/peer相同的刚完成P1，不能因文档/状态/规则冷绑定重跑。
- 最多一批定向纠正，性能失败先看源码与成本；未资格不猜PASS，不无限改姿势。
- 最终有限删除后数学路径不变则不重跑P1；selected correctness/交付、closure（previous-ref明确）、docs、
  独立终审/统一LOC、cleanup、[skip ci]提交推送完成后关闭。本卡收口不代表GPU或整模型SOTA。

当前SCOPED，G1–N3待源码审计定界；七门待验，不以旧CLOSED替代本轮。
