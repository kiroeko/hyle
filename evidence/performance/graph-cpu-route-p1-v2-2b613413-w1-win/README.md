# R3 Graph v2 工程 P1：资格通过，性能未决

候选 `2b613413cd30af845f0b66911e75a2fb455acd2a`；产品投影 `d9712e68`；合同
`graph-cpu-route-p1-v2`。W1-WIN / MSVC 19.50 / i9-13900K，caller CPU 2、workers 0/4，
三个独立物理 P 核，low-latency；两种代码布局 × 执行/分配正反，n=15→31。

实际 workload A/A 16 进程 / 192 ratios 的既定 5% factor-median 与 10% gross 门 PASS；
全部五个公共 controls、build/artifact validators 和 deep verify PASS，engineering QUALIFIED。
12 项中 11 项非劣门 PASS；双 worker 同实现 direct fused 比较两项 CI 下界越过
1/1.05=0.952381，聚合 performance_gate=INCONCLUSIVE。本轮停止，不追加样本。

## 全部方向

ratio>1 有利于 Graph；严格方向直接检查各分层 CI 与 1 的关系，不采用仪器的 5% equivalence
标签代替方向。范围只便于阅读，不能把不同分层或 workload 合并成一个统计估计。

| Workload | median ratio 范围 | 全部分层 CI 外界 | 严格方向 | 非劣门 |
|---|---|---|---|---|
| decode-fused-boundary-1x256x512 | 0.970136–0.974529 | 0.962630–0.981172 | LOSS | PASS |
| decode-natural-sequence-1x256x512 | 0.981288–0.989639 | 0.973949–0.999395 | LOSS | PASS |
| short-prefill-fused-boundary-16x512x1024 | 0.994235–1.000967 | 0.981550–1.009335 | INCONCLUSIVE | PASS |
| short-prefill-natural-sequence-16x512x1024 | 0.990111–1.005592 | 0.981816–1.017910 | INCONCLUSIVE | PASS |
| holdout-fused-boundary-17x129x257 | 0.984310–0.989053 | 0.979737–0.998289 | LOSS | PASS |
| holdout-natural-sequence-17x129x257 | 0.987358–0.993463 | 0.982557–0.997530 | LOSS | PASS |
| decode-native-sequence-1x256x512 | 0.999597–1.008232 | 0.992717–1.015219 | INCONCLUSIVE | PASS |
| short-prefill-native-sequence-16x512x1024 | 0.978004–0.987949 | 0.966585–1.001661 | INCONCLUSIVE | PASS |
| holdout-native-sequence-17x129x257 | 0.959773–0.964887 | 0.954126–0.970966 | LOSS | PASS |
| two-worker-fused-boundary-17x129x257 | 0.970361–0.982654 | 0.950646–1.002706 | INCONCLUSIVE | UNRESOLVED |
| two-worker-natural-sequence-17x129x257 | 1.157200–1.173222 | 1.137193–1.202477 | WIN | PASS |
| two-worker-native-sequence-17x129x257 | 1.268071–1.317464 | 1.242161–1.330614 | WIN | PASS |

阻塞项的八组 median ratio=0.970361–0.982654，Graph 中位耗时约 29.37–30.51us，
direct fused 约 28.91–30.18us。两个未决 CI 分别为 [0.951462,0.984325] 与
[0.950646,0.982562]；不是已经证实超过 5% 的 material regression。
固定小幅 LOSS 保留；是否存在可消除机制另按有界源码审计裁决，不因 PASS 自动签 parity。

## 保留与可复核性

`raw-bundle.zip` 完整保留原 output 的 469 个文件，包括两个采样阶段、A/A、controls、
observations、原 manifest，以及实际 driver、运行日志、非交互 Windows recipe 和 verify。
summary/manifest/controls/contract-lock/verify 的可读副本同时放在本目录，仅规范 LF 行尾；原包字节不变，路径与结果未修改。
SHA-256 在 sha256.json。工程包仍引用 Windows 外部冻结 checkout、build artifact 和工具链：
`C:/hyle-scratch/r3-cpu-20260916/p1-graph-v2` 及 `p1-graph-v2-build`，原样保留。
这是当前机器上可深度复核的工程证据，不声称脱离这些 artifact 的永久 Formal 资格。

命令：外部 qualify-graph-v2.ps1；重放使用
`python scripts/hyle-perf.py verify --contract bench/contracts/graph-cpu-route-p1-v2.json --manifest <原目录>/manifest.json`。
当前 owner 的停止/恢复合同见
[验收卡](../../../docs/implementation/tasks/compute/cpu-graph-external-caller-qualification.md)。

统一 loc-report.py --self-check PASS：Compute + Graph 全部实现 31,023 行，不含测试；
原始统计见 loc.json。该数包含旧 backend/provider，不是新基础架构完成百分比。
