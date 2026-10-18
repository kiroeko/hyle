# G01 CPU resident：实际W1同合同P1通过

CPU产品d91a不变，性能source `af9dc884448249ef0ad87d28525e5303ec087401`，独立clean checkout，contract
`d766ab8f24fbc7f0ea8ad0fffa66949e2f04dba675188a8f014a75b7e62606f2`，原ggml CPU peerFFa4。实际W1-WIN、原registered
CPU2/priority/affinity，单native线程；[receipt](receipt.json)、[summary](summary.json)、
[manifest](manifest.json)、[raw.zip](raw.zip)、[verify](verify-receipt.json)保存实际证据。

资格PASS、三个点5%非劣PASS、原verify退出0；16 A/A、八strata×n15和30 controls完成，
未进入n31。BF16 BHSD GQA8/2容量cache，Q1/128 KV1024 D256 causal与Q17/KV257 D64
noncausal，同公开PreparedOperation/CPUQueue/Completion/caller-output，原oracle不放宽。

| 点 | candidate/peer吞吐比 | 非劣 / band | 相对1方向 |
|---|---|---|---|
| cpu-resident-decode-d256 | 0.9977–1.0021 | PASS / PARITY | INCONCLUSIVE |
| cpu-resident-prefill-d256 | 0.9987–1.0040 | PASS / PARITY | INCONCLUSIVE |
| cpu-resident-holdout-d64 | 0.9977–1.0011 | PASS / PARITY | INCONCLUSIVE |

v1冷诊断显示104paired rounds过重，正式product samples前停止，原851.5s保留于
[停止记录](../../correctness/inference-g01-cpu-p1-cadence-stop/README.md)。v2在采样前固定
16paired rounds并校验遥测，数学/peer/n15/31/5%/公共controls不变，公共工具不改。
本轮实际2598.99s；stale ownmutex的0.297s原FAIL也保留。

两侧query/output/mask/cplan费用与完整完成边界计费，native private upper仍UNKNOWN，
不签总RSS预算、全引擎token/s或整个G01。多请求实现有独立新card，不能继承本片段性能。
