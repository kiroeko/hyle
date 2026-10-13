# CPU native ragged 同合同P1

产品candidate `973d3ffbf642aff06f6316fcec4b3cc99eba2e4d`，独立clean source `9cfa7b50db652010da5d4d544da9ee01c10644bc`，
contract `8751dfe583f516b1559256a23d14db91d832972884260d744a6da854adf91633`，peer固定原ggml CPU ffa4e8b8。实际registered
W1-WIN/CPU2/priority/affinity，单native线程。[receipt](receipt.json)、[summary](summary.json)、
[manifest](manifest.json)、[raw](raw.zip)、[verify](verify-receipt.json)保实际命令/原始样本/身份。

资格 `QUALIFIED`，性能非劣 `PASS`，verify退出
`0`；16 A/A、120产品process、30controls，实际sample
stages `[15]`，累计2188.17s。两个request BF16 packed NHD/GQA8/2，
Q1/17 KV1024/768 D256 causal、Q64/128同KV长prefill、Q1/17 KV31/47 D64 noncausal。
同原BatchProgram/两slots/完整topology update/metadata上传/F1/Queue/Completion/output
owner return，peer独立原flash图，原double bound8e-3和readonly/tail不放宽。

| 点 | candidate/peer吞吐比 | 5%非劣 / band | 相对1方向 |
|---|---|---|---|
| cpu-ragged-mixed-d256 | 0.9980–1.0030 | PASS / PARITY | INCONCLUSIVE |
| cpu-ragged-prefill-d256 | 0.9983–1.0017 | PASS / PARITY | INCONCLUSIVE |
| cpu-ragged-holdout-d64 | 0.9992–1.0046 | PASS / PARITY | INCONCLUSIVE |

16 paired rounds、4ms epoch floor，长prefill实测约169ms后在正式前固定min1，保原
n15/31/controls/双顺序/placement/5%阈值。所有资格/统计使用原工具，不由诊断点值
修改公共API或称PASS。min2诊断raw误覆盖仅15.641s工具耗时保留，min1完整9.422s
及compile FAIL另存[预检与交付](../../correctness/inference-g01-cpu-ragged-factory-delivery/README.md)，
它们未进入本轮正式samples。

已知cold query/mask/metadata/workspace计费，native private pool上界UNKNOWN。该有限
片段不签全模型token/s、paged/capture/private总预算或整个G01/十项goal完成。
