# Vulkan native ragged 实际同合同P1

产品bc0ba41f7、clean source `4f224878a3ca1c015c55ac77de3e711b11998034`，contract `d9a54a85a07e0f04093fb51cdc10c1fd09f2ec62569ac62efb78374e4fddb1ea`。
[receipt](receipt.json)、[summary](summary.json)、[manifest](manifest.json)、[raw](raw.zip)、
[verify](verify-receipt.json)保实际命令/身份/原始样本/失败；actual Mac16,12/M1-MAC工程
资格，原interactive QoS，独立硬件Formal未签。

资格 `QUALIFIED`、性能非劣 `PASS`，verify退出
`0`，16 A/A、120产品process、30controls，实际stages
`[15]`，累计931.80s。F16 packed NHD/GQA8/2、Q1/17 KV1024/768
D256 causal、Q64/128同KV长prefill、Q1/17 KV31/47 D64 noncausal。原double bound2e-3/
readonly/tail不放宽，同原Program/Invocation/两slots/ready input owners/checkedcompletion/
单packed owning output，两边private native内存上界UNKNOWN。

| 点 | candidate/peer吞吐比 | 5%非劣 / band | 相对1方向 |
|---|---|---|---|
| ragged-mixed-d256 | 1.0545–1.0642 | PASS / INCONCLUSIVE | WIN |
| ragged-prefill-d256 | 0.9993–1.0162 | PASS / PARITY | INCONCLUSIVE |
| ragged-holdout-d64 | 1.0014–1.0084 | PASS / PARITY | INCONCLUSIVE |

原16轮/min1/4ms、n15/31、双execution/allocation与placement、controls/阈值不改。
原MLX fast SDPA或独立原ggml shader按实际pin身份核验，不调用candidate attention
lower/math。只有本有限multi域，不签全模型token/s、paged/growth/capture或整个G01。
先前比较器compile/SIGSEGV诊断保[预检](../../correctness/inference-g01-native-ragged-p1-preflight/README.md)，
原280/0e9两次合格FAIL独立保留；全部jobs保累计，不换树抹掉失败或累计成本。
