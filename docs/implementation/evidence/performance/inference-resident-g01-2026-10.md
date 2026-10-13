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
