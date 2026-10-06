# CPU RN1/RN4：成熟低精度接线资格

固定产品 candidate `775b2bedace019de423edc5d8a6477772b8ef3ec`，Windows canonical，Linux 前向副本。
本批只验收[RN1/RN4](../../../docs/implementation/tasks/compute/cpu-residual-scatter-rope-numeric-retirement.md)，
不签剩余自研性能胜负或整个 CPU goal 完成。

## 实现与同合同边界

- Native F16/BF16 ScatterAdd 明确冷拒绝；删除两个模板中逐项 storage 舍入的 Add 路径，F32 Add、
  各 storage Assign 和 SIMD 保留。原 ATen/ggml 成熟累加实际接替，旧 ordered-rounding 测试迁移。
- 原 ATen RoPE 开放 F16/BF16 data、I32/I64 positions，frequency 仍 F32。原 Double phase 数学不改；
  RoPE 在公共 numeric 入口保留 typed Tensor，保证未旋转 tail 的原始位，版本为 programs-6。
- RN 初案对 RoPE 的不同 phase 比较已撤回。原模型 Float32 phase 程序、Native I32 与 ggml 不改；
  Double oracle 不是 Float32 模型的相同计算程序，不签数值 FAIL/性能 LOSS。解释见
  [同合同纠正](../cpu-residual-scatter-rope/README.md)与[精度审计](../../../docs/implementation/tasks/compute/cpu-rope-phase-contract-audit.md)。

## 实际门

| 验证 | Windows | Linux |
|---|---|---|
| Native owning factor，含独立 ATen | 144/144，无失败/skip | 144/144，无失败/skip |
| 普通 CPU／具名 RoPE 直接消费者 | 16 通过、2 旧 Qwen fixture skip | 同左 |
| 普通 Graph／具名 RoPE 消费者 | 22/22，无失败/skip | 22/22，无失败/skip |
| 公开头 | 1/1 | 1/1 |
| 普通低精度 Direct/Graph consumer | source shared/static、installed shared/static 全通过 | 同四种旅程全通过 |
| 低精度生命周期 consumer | F16/BF16 实际通过 | F16/BF16 实际通过 |

普通 consumer 验证两个 index widths 的重复 ScatterAdd，以及 I64 低精度 RoPE：实际成熟身份、
下游 Copy、变化输入重放、真正 wait/close、Native-only 拒绝。unit 覆盖原 ATen I32/I64、两 layout、
odd byte stride 和 sNaN／signed-zero 尾部位。补充 consumer 验证 axis0、empty、dense input0 alias、
同 Prepared 的两条独立 Queue/invocation 并发与三次变化输入重放、close拒绝及所供 allocator 的
逐点冷 prepare/invocation OOM 回滚与平衡。原库私有 Tensor heap 未注入故障，仍如实声明 unknown。

两个 skip 都要求外部 HYLE_D5_QWEN35_FIXTURE，不计 PASS；本批实际新 dtype/math/资源与普通
Direct/Graph 由独立带值消费者完成。未变模型原程序只按源码身份保留既有资格，不称本轮整模型
重跑或新模型验证完成。

## 原件、成本、清理

首次候选 e4f4bd44 的 Windows factor 143/144，失败为 sNaN tail 在转 F32 时变化；保存原 XML/log。
一次具名产品修正后通过。local driver 初次误选所有 CPU 模型，运行 272.89 秒有界中止，保存
非零 receipt，未签全目录 PASS；只运行实际受影响的普通/具名消费者，未修改公共治理工具。
生命周期 fixture 的一次 C++ CTAD compile failure及拆分声明修正也保留，不归因产品或 vendor。

P0 沿原 typed packing、原 Tensor 数学/转换/clone、完整 index 检查与真完成边界，managed、private
heap unknown、capture/thread 条件保持；没有新 payload 数学、热搜索、计时或线程政策。
1338 个产品/test/构建/consumer 输入 forward hash 匹配，109 个 SIMD/其它成熟程序/组装器输入不变。
复用未改 SDK-off、普通原模型和公共生命周期接缝，不冒充此次重跑；新 dtype 的直接旅程及资源
风险实际执行。

统一脚本计 Compute+Graph 实现 **61,386 行 / 249 文件**，不含 tests/vendor。六个自有临时
consumer 目录已清理 **917,628,251 logical bytes**，SDK/producer/cache 保留，不保证等量物理空间。
完整源码、manifest、drivers、XML、失败/中止原件、两平台生命周期、LOC和cleanup在
[qualification.zip](qualification.zip)（101 entries，230,035 bytes），SHA256/摘要见
[receipt.json](receipt.json)。本批七门与P0通过；剩余合格自研域仍按有限批次裁决，不能因低优先级
fallback永久保留，也不能删SIMD本体。

## 两个原 skip 的实际补跑

检查确认 `.scratch/d5-qwen35-six-profile-fixture` 已有真实权重/参考输出；原 skip 的直接原因是
测试进程未设置 `HYLE_D5_QWEN35_FIXTURE`，不是用户原模型缺失。设置仅子进程环境后，固定产品
候选的两个测试在 Windows/Linux **各2/2实际通过，无skip**，不重建：真实2048维BF16输出投影及
原Q/K norm权重参与的RMSNorm→RoPE→GQA Attention普通准备/重放。
5个输入文件hash在Linux前向副本匹配，8MiB级临时副本已清理；原Windows fixture保留复用。
初次16通过/2skip的XML与receipt不回写，两个补跑通过覆盖原未运行项，当前合并为18/18已执行。
完整原件/driver/hash见[fixture-followup.zip](fixture-followup.zip)和
[fixture-followup.json](fixture-followup.json)。这仍是片段验收，不是整模型推理。
