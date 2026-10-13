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
