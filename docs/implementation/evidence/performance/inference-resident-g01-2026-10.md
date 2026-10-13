# G01 resident SDPA 性能增量账

2026-10-13；这是[历史全库账](full-library-scorecard.md)的当轮增量，不重写旧快照或
统计核心。仅实际Air Mac16,12工程资格、固定F32 BSHD GQA8/2、完整生命周期，非Formal
或全模型token/s。ratio大于1有利于HYLE；5%非劣、practical band和相对1的方向分列。

| source/candidate | decode D256 | prefill D256 | noncausal D64留出 | 资格/性能门 |
|---|---|---|---|---|
| 原d91a，source9087 | 0.687–0.710，LOSS | 0.962–0.969，LOSS | 0.921–0.977，LOSS | PASS / FAIL |
| RR a400 | 0.719–0.727，LOSS | 0.962–0.968，LOSS | 0.947–0.961，LOSS | PASS / FAIL |
| 直接port 3d92 | 0.9941–1.0017，INCONCLUSIVE | 1.0010–1.0032，WIN | 1.0035–1.0092，WIN | PASS / PASS |

原两次LOSS均保留：[d91a](../../../../evidence/performance/inference-g01-resident-vulkan-m1-p1/README.md)、
[a400](../../../../evidence/performance/inference-g01-resident-vulkan-replay-m1-p1/README.md)。
最终[3d92收据/summary/raw](../../../../evidence/performance/inference-g01-resident-vulkan-direct-ports-m1-p1/README.md)
经过原verify回放，三个点5%非劣均PASS、band均PARITY；prefill/留出每stratum CI下界>1
所以方向WIN，decode混合/跨1故INCONCLUSIVE，不把band当精确方向。

原合同SHA4ae、peer pin456172ec、precision/math/input/完成边界和阈值未改。原停止规则
[15,31]自动追加，16个A/A、368产品process、92 controls与3669.39s完整保留。真实费用
反证为外层3个warm Copy，原native记录修正不足；最终合格port Copy3→0，未放宽oracle。
CPU正式比较另有v1冷诊断STOP/v2固定排期，尚未获资格，不混入此账。

CPU v2现已[实跑资格/性能PASS](../../../../evidence/performance/inference-g01-resident-cpu-w1-p1/README.md)：
W1-WIN原registered CPU2、BF16 BHSD GQA8/2、完整PreparedOperation/Queue/Completion，
source af9、CPU产品d91a、contract d766；16 A/A、120产品process与30 controls，
2598.99s、原verify退出0，未进入n31。decode比0.9977–1.0021、prefill0.9987–1.0040、
D64留出0.9977–1.0011，三点5%非劣PASS/band PARITY，相对1方向均INCONCLUSIVE。
不继承到新ragged入口；v1 STOP与v2 stale-marker FAIL保留，原限额/peer/math没有改写。

CPU native ragged也已[独立资格/性能PASS](../../../../evidence/performance/inference-g01-cpu-ragged-w1-p1/README.md)：
产品973/source9cfa、contract8751dfe5，同原BatchProgram/F1/metadata update/两slots/Queue/
Completion/output owner return，对照独立原flash图；实际W1-WIN原CPU2/priority/affinity。
16 A/A、120产品process与30 controls，n15停止，2188.172s、verify0。两request mixed
Q1/17 KV1024/768 D256 causal比0.9980–1.0030，Q64/128同KV prefill0.9983–1.0017，
Q1/17 KV31/47 D64 noncausal0.9992–1.0046；全部5%非劣PASS/band PARITY，方向全部
INCONCLUSIVE。cold/native private费用仍分别已知/UNKNOWN，不签全模型或全部G01。
min1在正式前按169ms长prefill与4ms floor固定；原16轮/n15/31/controls/阈值未改。

Apple native ragged也已[独立资格/性能PASS](../../../../evidence/performance/inference-g01-native-ragged-apple-m1-p1/README.md)：
产品280/source c6d10f66、contract1f8e811f，实际Air Mac16,12/M4、M1-MAC engineering
interactive QoS；原MLX fast SDPA pin1f8e74e3与所链接SDPA/Metal源码核验。两边同原
Program/Invocation/两slots、F16 packed ready sources、checked completion与单packed
owning output，16 A/A、n15产品120/controls30，845.543s、原verify0，5%非劣全部PASS。

| 点 | candidate/peer吞吐比范围 | 5%非劣 / practical band | 相对1方向 |
|---|---|---|---|
| Q1/17 KV1024/768 D256 causal | 0.9997–1.0071 | PASS / PARITY | INCONCLUSIVE |
| Q64/128同KV长prefill | 0.9827–0.9894 | PASS / PARITY | LOSS |
| Q1/17 KV31/47 D64 noncausal | 0.9961–1.0038 | PASS / PARITY | INCONCLUSIVE |

长prefill稳定落后约1–1.7%，保留LOSS，不由5%门或band改写成方向parity，也不据此
扩大产品FIX。native/private内存两侧UNKNOWN；只签有限F16多请求工程域，不签独立
硬件Formal、BF16性能、paged/growth/capture或完整引擎token/s。首次配置脚本错误路径
FAIL4.868s保raw jobs，修私有身份脚本后的原同产品/合同新fresh2继续，未抹掉失败。

Vulkan native ragged的[合格失败](../../../../evidence/performance/inference-g01-native-ragged-vulkan-m1-p1/README.md)
也保留：产品280/source c6d10f66、contractd9a54a85，实际Air同M1-MAC工程身份，
16 A/A、n15产品120/controls30、928.101s，QUALIFIED、verify0、整体5%非劣FAIL。

| 点 | candidate/peer吞吐比范围 | 5%非劣 / practical band | 相对1方向 |
|---|---|---|---|
| Q1/17 KV1024/768 D256 causal | 1.0189–1.0408 | PASS / INCONCLUSIVE | WIN |
| Q64/128同KV长prefill | 0.9913–1.0042 | PASS / PARITY | INCONCLUSIVE |
| Q1/17 KV31/47 D64 noncausal | 0.8656–0.8885 | FAIL / LOSE | LOSS |

短D64固定loss约11–13%，不是仪器未资格或几次诊断点值，阻断本有限Vulkan NR轮关闭。
先核提交/输出结构再裁决真实owner，不能改容差/shape/阈值或无限重采样寻找PASS。

原region融合0e9的[第二次合格失败](../../../../evidence/performance/inference-g01-native-ragged-vulkan-rf-m1-p1/README.md)
也保留：同contractd9a54a85/source0006944b7，16 A/A、n15产品120/controls30、
929.176s，QUALIFIED、verify0、整体5%非劣FAIL。混合D256比1.0258–1.0439方向
INCONCLUSIVE/非劣PASS，长prefill0.9884–0.9965方向INCONCLUSIVE/非劣PASS，短D64
0.8018–0.8260方向LOSS/非劣FAIL。3managed段→1改善不了这条warm资格缺口：原
fixed_attention_graph拒第二个flash与Concat，导致整合graph每步走managed编码。
必须复用原record-safe/actual retirement机制并实证多flash记录，否则不结案。
