# Vulkan native ragged 实际同合同P1

产品0e9e011b8、clean source `0006944b74ba7f93c68ee08b715f7c17a33ee854`，contract `d9a54a85a07e0f04093fb51cdc10c1fd09f2ec62569ac62efb78374e4fddb1ea`。
[receipt](receipt.json)、[summary](summary.json)、[manifest](manifest.json)、[raw](raw.zip)、
[verify](verify-receipt.json)保实际命令/身份/原始样本/失败；actual Mac16,12/M1-MAC工程
资格，原interactive QoS，独立硬件Formal未签。

资格 `QUALIFIED`、性能非劣 `FAIL`，verify退出
`0`，16 A/A、120产品process、30controls，实际stages
`[15]`，累计929.18s。F16 packed NHD/GQA8/2、Q1/17 KV1024/768
D256 causal、Q64/128同KV长prefill、Q1/17 KV31/47 D64 noncausal。原double bound2e-3/
readonly/tail不放宽，同原Program/Invocation/两slots/ready input owners/checkedcompletion/
单packed owning output，两边private native内存上界UNKNOWN。

| 点 | candidate/peer吞吐比 | 5%非劣 / band | 相对1方向 |
|---|---|---|---|
| ragged-mixed-d256 | 1.0258–1.0439 | PASS / INCONCLUSIVE | INCONCLUSIVE |
| ragged-prefill-d256 | 0.9884–0.9965 | PASS / PARITY | INCONCLUSIVE |
| ragged-holdout-d64 | 0.8018–0.8260 | FAIL / LOSE | LOSS |

原16轮/min1/4ms、n15/31、双execution/allocation与placement、controls/阈值不改。
原MLX fast SDPA或独立原ggml shader按实际pin身份核验，不调用candidate attention
lower/math。只有本有限multi域，不签全模型token/s、paged/growth/capture或整个G01。
先前比较器compile/SIGSEGV诊断保[预检](../../correctness/inference-g01-native-ragged-p1-preflight/README.md)，
原280短D64 qualified非劣FAIL保前一轮独立证据；全部jobs保累计，不换树抹掉失败或累计成本。
