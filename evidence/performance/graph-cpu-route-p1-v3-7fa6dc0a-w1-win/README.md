# R3 CPU Graph prepared-region 修正后的工程 P1

产品候选 `1a3a8d0e40a2a17250119369331c712718a4d880`；runner source
`7fa6dc0a8b3b857a1835525de0d0e7be5f8db5e9` 仅追加合同/状态，产品投影相同。
合同 graph-cpu-route-p1-v3；Native peer 绑定产品候选，oneDNN 绑定
`71094df451909e95ceeefc8f64c275cdcbbcdff5`。W1-WIN / MSVC 19.50 / engineering。

qualification_gate=PASS，performance_gate=INCONCLUSIVE；
12 个 workload 的非劣门 11/12 PASS。完整 raw 保存 n=15 与 n=31 阶段，已到冻结样本上限。
deep verify 的 paths/provenance 为 true/EXACT；这是身份复核，不替代实际 performance gate。
修正只把 shared-region 的重复资源校验移回冷阶段，并保留当前 Queue owner/admission/completion；
不改变数学、默认规则、Sched、核拓扑或统计门。旧 v2 未决包保留，没有原样重跑或追加第三档样本。

## 完整性能方向

ratio>1 有利于默认 Graph。每个 workload 的所有分层 CI 都在 1 以上才记 WIN，都在 1 以下才记
LOSS；其余记 INCONCLUSIVE。5% 工程非劣 PASS 不自动改名为 parity。以下方向只有在本包
qualification_gate=PASS 时才具有合格证据资格；不得跨候选直接据此计算修复加速比。

| Workload | median ratio 范围 | 全部分层 CI 外界 | 严格方向 | 非劣门 |
|---|---|---|---|---|
| decode-fused-boundary-1x256x512 | 0.966131–0.974263 | 0.951737–0.984423 | LOSS | UNRESOLVED |
| decode-natural-sequence-1x256x512 | 0.987477–0.995959 | 0.977373–1.005240 | INCONCLUSIVE | PASS |
| short-prefill-fused-boundary-16x512x1024 | 0.990410–1.002805 | 0.985584–1.014731 | INCONCLUSIVE | PASS |
| short-prefill-natural-sequence-16x512x1024 | 0.993661–1.003142 | 0.985273–1.015241 | INCONCLUSIVE | PASS |
| holdout-fused-boundary-17x129x257 | 0.987678–0.995114 | 0.981875–0.999792 | LOSS | PASS |
| holdout-natural-sequence-17x129x257 | 0.991623–0.994412 | 0.983759–1.001743 | INCONCLUSIVE | PASS |
| decode-native-sequence-1x256x512 | 1.005643–1.012878 | 0.999680–1.021756 | INCONCLUSIVE | PASS |
| short-prefill-native-sequence-16x512x1024 | 0.978415–0.993894 | 0.970855–1.000433 | INCONCLUSIVE | PASS |
| holdout-native-sequence-17x129x257 | 0.962867–0.967719 | 0.955836–0.971654 | LOSS | PASS |
| two-worker-fused-boundary-17x129x257 | 0.972652–0.977621 | 0.961814–0.993454 | LOSS | PASS |
| two-worker-natural-sequence-17x129x257 | 1.162959–1.175137 | 1.144884–1.187616 | WIN | PASS |
| two-worker-native-sequence-17x129x257 | 1.300473–1.320614 | 1.277643–1.333040 | WIN | PASS |

directions.json 是从已保留 summary 逐项读取的辅助表，另含各侧绝对吞吐范围、单片段微秒范围和
所有未通过分层的完整 CI；没有重算统计量或更改工具裁决。

## 保留与限制

raw-bundle.zip 保留完整输出、全部原始产品/控制日志、实际 driver、非交互 Windows recipe、运行库
相邻部署 hash 和 verify。四个 MSVC DLL 来自原候选会加载的同一 System32 文件，已在首次运行前
复制到 exe 目录；产品为 static，没有临时改 Provider 或 loader 搜索路径。
JSON 可读副本只规范排版和 LF，结果/路径不变；sha256.json 绑定目录文件。

工程 build/source 保留在 Windows 外部 scratch，本包不声称脱离这些 artifact 的永久 Formal
资格。该结果只覆盖冻结的完整 f32 FFN 形状和资源/完成合同，不外推算子大全、完整推理引擎、
其它硬件或 GPU 路线。[本次正确性与交付](../../correctness/graph-cpu-replay-cost/README.md)
和原 Direct P1 分开签证。
