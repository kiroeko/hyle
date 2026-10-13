# G01 Vulkan 驻留直接端口：原合同合格P1通过

不可变candidate `3d92eee97a70f7788dfb9bbcda73f6e0f801959f`，同原canonical合同
`4ae724d684be30814e2f28d7d933f771b7e03dda56b86cb51464cef706031720`、原peer、
数学/输入/完整生命周期，实际Air Mac16,12工程资格，非Formal或全引擎token/s。
[receipt](receipt.json)、[summary](summary.json)、[manifest](manifest.json)、
[raw.zip](raw.zip)、[verify](verify-receipt.json)保留可复核命令/产物/所有样本。

16个A/A通过；按原固定[15,31]规则自动追加，未人工扩大样本或改变阈值。n15的
120产品+30 controls与n31的248产品+62 controls全部保留，实际3669.39s。
资格PASS、原verify退出0、三个点5%非劣全部PASS。最终八strata中位吞吐比：

| 点 | candidate/peer范围 | 5%门 | 相对1的方向 |
|---|---|---|---|
| Q1/KV1024/D256 causal decode | 0.9941–1.0017 | PARITY / PASS | INCONCLUSIVE |
| Q128/KV1024/D256 causal prefill | 1.0010–1.0032 | PARITY / PASS | WIN |
| Q17/KV257/D64 noncausal留出 | 1.0035–1.0092 | PARITY / PASS | WIN |

decode方向INCONCLUSIVE；prefill与留出各stratum CI下界>1，方向WIN，但仍在5% band内，不宣称全模型或模块全局胜出。原两轮decode
0.687–0.710、0.719–0.727的合格性能FAIL仍分别保留，不能覆盖或删除。此次原自然
路线的额外query/KV Copy3→0已由公开成本与原数学反证，原精度和生命周期不减弱。

此结果只签三点固定消费域和端口修正；CPU正式P1、多请求和其余G项仍未完成。
原三个P1 source/build/output都保持原字节，新fresh4独立现场继续保留。
