# Vulkan native page16 实际同合同P1

产品520027a57、clean source `66af56852d0602ce14cdfde9aba3de373a7472c6`，contract `5bd7e044430c33d00eb4e85238df7357805751f2a36d9361fe547c5e7d485777`。
[receipt](receipt.json)、[summary](summary.json)、[manifest](manifest.json)、[raw](raw.zip)、
[verify](verify-receipt.json)保实际命令/身份/原始样本/失败；actual Mac16,12/M1-MAC工程
资格，原interactive QoS，独立硬件Formal未签。

资格 `QUALIFIED`、性能非劣 `PASS`，verify退出
`0`，16 A/A、120产品process、30controls，实际stages
`[15]`，累计1493.41s。F16 packed NHD/GQA8/2、Q1/17 KV1024/768
D256 causal、Q64/128同KV长prefill、Q1/17 KV31/47 D64 noncausal。原double bound2e-3/
readonly/tail不放宽，两边显式cold prepare/native replacement、owned I32 F2 table upload、exact page guards；同原Program/Invocation/两slots/ready input owners/checkedcompletion/
单packed owning output，两边private native内存上界UNKNOWN。

| 点 | candidate/peer吞吐比 | 5%非劣 / band | 相对1方向 |
|---|---|---|---|
| page-mixed-d256 | 6.4314–6.6996 | PASS / WIN | WIN |
| page-prefill-d256 | 5.8188–5.9560 | PASS / WIN | WIN |
| page-holdout-d64 | 2.0305–2.0691 | PASS / WIN | WIN |

原16轮/min1/4ms、n15/31、双execution/allocation与placement、controls/阈值不改。
原MLX fast SDPA或独立原ggml shader按实际pin身份核验，不调用candidate attention
lower/math。只有本有限multi域，不签全模型token/s、growth/capture/联合分页或整个G01。
本轮对照独立double/逐exact-page readonly预校验和Vulkan脚本配置FAIL保[预检](../../correctness/inference-g01-native-page-bench-preflight/README.md)，
当前原生页数学域和原始FAIL累计保留；不换树抹掉失败或修改原5%门。
