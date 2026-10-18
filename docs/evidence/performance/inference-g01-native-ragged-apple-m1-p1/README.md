# Apple native ragged 实际同合同P1

产品280、clean source `c6d10f66e37cebafd134eee88e82aea603319cbc`，contract `1f8e811f4765c4aa50a0a0194e601151d6d5ffdba04b3ecd0f40f93e1218a24a`。
[receipt](receipt.json)、[summary](summary.json)、[manifest](manifest.json)、[raw](raw.zip)、
[verify](verify-receipt.json)保实际命令/身份/原始样本/失败；actual Mac16,12/M1-MAC工程
资格，原interactive QoS，独立硬件Formal未签。

资格 `QUALIFIED`、性能非劣 `PASS`，verify退出
`0`，16 A/A、120产品process、30controls，实际stages
`[15]`，累计845.54s。F16 packed NHD/GQA8/2、Q1/17 KV1024/768
D256 causal、Q64/128同KV长prefill、Q1/17 KV31/47 D64 noncausal。原double bound2e-3/
readonly/tail不放宽，同原Program/Invocation/两slots/ready input owners/checkedcompletion/
单packed owning output，两边private native内存上界UNKNOWN。

| 点 | candidate/peer吞吐比 | 5%非劣 / band | 相对1方向 |
|---|---|---|---|
| ragged-mixed-d256 | 0.9997–1.0071 | PASS / PARITY | INCONCLUSIVE |
| ragged-prefill-d256 | 0.9827–0.9894 | PASS / PARITY | LOSS |
| ragged-holdout-d64 | 0.9961–1.0038 | PASS / PARITY | INCONCLUSIVE |

原16轮/min1/4ms、n15/31、双execution/allocation与placement、controls/阈值不改。
原MLX fast SDPA或独立原ggml shader按实际pin身份核验，不调用candidate attention
lower/math。只有本有限multi域，不签全模型token/s、paged/growth/capture或整个G01。
先前比较器compile/SIGSEGV诊断保[预检](../../correctness/inference-g01-native-ragged-p1-preflight/README.md)，
Apple首次身份脚本错误路径配置FAIL保jobs，不换树抹掉失败或累计成本。
