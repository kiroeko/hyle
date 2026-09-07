# 冻结性能审计逐格矩阵（既有证据摘录）

本表由 [audit-data.json](audit-data.json) 派生，只记录基线审计事实，不生成或续签性能证书。
任务与判定依据见 [project audit](../../docs/implementation/tasks/project/frozen-performance-opportunity-audit-2026-09.md)。

`r > 1` 有利于 HYLE；负向幅度使用吞吐下降 `100 × (1-r)`，不是延迟增长 `100 × (1/r-1)`。
CI 为原记录约 99.26% 的逐层中位数区间；全精度保留在 JSON。原 exact/NI verdict 不等于本次审计新签 PASS。
`RAW` 表示核对 CURRENT summary hash；`DOC` 表示当前文档舍入值；`MISSING` 表示没有逐格数值。
platform_fs 的 CI 只保留“包含 1”的文档断言；八个 ratio 没有逐层标签，保持未分配，不猜测它们与 order 的映射。
intrusive 只保留全层范围，不将端点分配给任何顺序。对 ORDER_OR_PLACEMENT 的命名不表示因果已证。

## container_intrusive / counted-membership-4096

原 combined exact：`INCONCLUSIVE`；审计：`NO_ACTION`；证据：`CURRENT_DOCUMENT_ONLY`。

CURRENT 文档只给范围 1.0118–1.0539；未知值不补成 1。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-candidate-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-peer-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-peer-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-candidate-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-candidate-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-peer-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-peer-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |

## container_intrusive / forward-arena-4096

原 combined exact：`PARITY`；审计：`ORDER_OR_PLACEMENT`；证据：`CURRENT_DOCUMENT_ONLY`。

CURRENT 文档只给范围 0.9823–1.0089；未知值不补成 1。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-candidate-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-peer-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-peer-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-candidate-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-candidate-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-peer-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-peer-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |

## container_intrusive / hlist-callbacks-4096

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`CURRENT_DOCUMENT_ONLY`。

CURRENT 文档只给范围 1.2206–1.277；未知值不补成 1。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-candidate-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-peer-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-peer-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-candidate-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-candidate-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-peer-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-peer-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |

## container_intrusive / sequence-sort-merge-4096

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`CURRENT_DOCUMENT_ONLY`。

CURRENT 文档只给范围 1.0881–1.1188；未知值不补成 1。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-candidate-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-peer-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| primary-peer-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-candidate-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-candidate-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-peer-first-candidate-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |
| placement-peer-first-peer-allocation-first | 未知 | 未知 | 未知 | 未知 | MISSING |

## util_core / const-borrowed-record-traversal-4096

原 combined exact：`PARITY`；审计：`NO_ACTION`；证据：`CURRENT_DOCUMENT_ONLY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.002090 | 0.000 | [0.977770, 1.012020] | PARITY | DOC |
| primary-peer-first | 1.003100 | 0.000 | [0.993860, 1.025030] | PARITY | DOC |
| placement-candidate-first | 1.005940 | 0.000 | [0.997960, 1.019660] | PARITY | DOC |
| placement-peer-first | 1.003150 | 0.000 | [0.989670, 1.008560] | PARITY | DOC |

## util_core / nontype-bound-member-record-traversal-4096

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`CURRENT_DOCUMENT_ONLY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.246360 | 0.000 | [1.233580, 1.256660] | WIN | DOC |
| primary-peer-first | 1.264370 | 0.000 | [1.228980, 1.289010] | WIN | DOC |
| placement-candidate-first | 1.245230 | 0.000 | [1.234000, 1.260880] | WIN | DOC |
| placement-peer-first | 1.257650 | 0.000 | [1.232200, 1.264940] | WIN | DOC |

## util_core / nontype-free-record-traversal-4096

原 combined exact：`PARITY`；审计：`ORDER_OR_PLACEMENT`；证据：`CURRENT_DOCUMENT_ONLY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 0.994930 | 0.507 | [0.989010, 1.013630] | PARITY | DOC |
| primary-peer-first | 1.002370 | 0.000 | [0.997670, 1.016800] | PARITY | DOC |
| placement-candidate-first | 0.991560 | 0.844 | [0.976220, 1.009830] | PARITY | DOC |
| placement-peer-first | 0.999240 | 0.076 | [0.977130, 1.005780] | PARITY | DOC |

## error / join-source-failed-owning-handoff

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.523805 | 0.000 | [1.490422, 1.587407] | WIN | RAW |
| primary-peer-first | 4.784033 | 0.000 | [4.749943, 4.816328] | WIN | RAW |
| placement-candidate-first | 1.518154 | 0.000 | [1.491662, 1.535313] | WIN | RAW |
| placement-peer-first | 4.657731 | 0.000 | [4.598242, 4.766159] | WIN | RAW |

## error / parallel-source-first-error-handoff

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.418017 | 0.000 | [1.378193, 1.455771] | WIN | RAW |
| primary-peer-first | 3.914315 | 0.000 | [3.818441, 3.922673] | WIN | RAW |
| placement-candidate-first | 1.410344 | 0.000 | [1.353389, 1.437973] | WIN | RAW |
| placement-peer-first | 3.862229 | 0.000 | [3.825519, 3.903959] | WIN | RAW |

## error / registry-fixed-table-borrow-64

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.729760 | 0.000 | [1.697103, 1.751486] | WIN | RAW |
| primary-peer-first | 1.750790 | 0.000 | [1.737629, 1.768272] | WIN | RAW |
| placement-candidate-first | 1.728223 | 0.000 | [1.684965, 1.753244] | WIN | RAW |
| placement-peer-first | 1.751532 | 0.000 | [1.721296, 1.772665] | WIN | RAW |

## error / taskgraph-source-owning-last-failure-64

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.216233 | 0.000 | [1.185491, 1.247762] | WIN | RAW |
| primary-peer-first | 2.509675 | 0.000 | [2.435063, 2.574534] | WIN | RAW |
| placement-candidate-first | 1.211841 | 0.000 | [1.181357, 1.230584] | WIN | RAW |
| placement-peer-first | 2.444188 | 0.000 | [2.385558, 2.530578] | WIN | RAW |

## error / taskgraph-source-success-64

原 combined exact：`PARITY`；审计：`ORDER_OR_PLACEMENT`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 0.998603 | 0.140 | [0.980225, 1.014927] | PARITY | RAW |
| primary-peer-first | 0.992315 | 0.769 | [0.967282, 1.002550] | PARITY | RAW |
| placement-candidate-first | 0.998534 | 0.147 | [0.980811, 1.015450] | PARITY | RAW |
| placement-peer-first | 0.992954 | 0.705 | [0.983568, 1.006976] | PARITY | RAW |

## obs / owner-scheduler-metrics-iteration

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.119264 | 0.000 | [1.099045, 1.151123] | WIN | RAW |
| primary-peer-first | 1.105286 | 0.000 | [1.093162, 1.118360] | WIN | RAW |
| placement-candidate-first | 1.144708 | 0.000 | [1.130008, 1.158760] | WIN | RAW |
| placement-peer-first | 1.145162 | 0.000 | [1.125385, 1.159568] | WIN | RAW |

## obs / shared-scheduler-metrics-iteration

原 combined exact：`PARITY`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.004207 | 0.000 | [0.993261, 1.012135] | PARITY | RAW |
| primary-peer-first | 1.005011 | 0.000 | [0.999131, 1.013020] | PARITY | RAW |
| placement-candidate-first | 1.007760 | 0.000 | [1.001407, 1.016349] | PARITY | RAW |
| placement-peer-first | 1.007027 | 0.000 | [1.000102, 1.017987] | PARITY | RAW |

## timer_queue / scheduler-timer-lifecycle-64

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 1.181484 | 0.000 | [1.130835, 1.232768] | WIN | RAW |
| primary-candidate-first-peer-allocation-first | 1.129346 | 0.000 | [1.091085, 1.166424] | WIN | RAW |
| primary-peer-first-candidate-allocation-first | 1.133844 | 0.000 | [1.104514, 1.159744] | WIN | RAW |
| primary-peer-first-peer-allocation-first | 1.167819 | 0.000 | [1.130541, 1.187781] | WIN | RAW |
| placement-candidate-first-candidate-allocation-first | 1.175409 | 0.000 | [1.130979, 1.219472] | WIN | RAW |
| placement-candidate-first-peer-allocation-first | 1.189774 | 0.000 | [1.153378, 1.216741] | WIN | RAW |
| placement-peer-first-candidate-allocation-first | 1.175765 | 0.000 | [1.147778, 1.216286] | WIN | RAW |
| placement-peer-first-peer-allocation-first | 1.173953 | 0.000 | [1.135283, 1.202623] | WIN | RAW |

## platform / inference-host-planning-snapshot

原 combined exact：`INCONCLUSIVE`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.015725 | 0.000 | [0.996623, 1.059769] | INCONCLUSIVE | RAW |
| primary-peer-first | 1.025350 | 0.000 | [1.003257, 1.072766] | INCONCLUSIVE | RAW |
| placement-candidate-first | 1.024989 | 0.000 | [1.004412, 1.096608] | INCONCLUSIVE | RAW |
| placement-peer-first | 1.037168 | 0.000 | [1.016681, 1.053644] | INCONCLUSIVE | RAW |

## platform_fs / model-file-open-read-close-4mib

原 combined exact：`PARITY`；审计：`ORDER_OR_PLACEMENT`；证据：`CURRENT_DOCUMENT_ONLY`。

CURRENT 文档只给范围 0.987503–1.012026；未知值不补成 1。

CURRENT 原序列（未分配到 stratum）：0.996402 / 0.999517 / 1.012026 / 0.998621 / 0.987503 / 1.005656 / 0.994822 / 1.00209。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 未知 | 未知 | 包含 1；界限缺失 | PARITY | MISSING |
| primary-candidate-first-peer-allocation-first | 未知 | 未知 | 包含 1；界限缺失 | PARITY | MISSING |
| primary-peer-first-candidate-allocation-first | 未知 | 未知 | 包含 1；界限缺失 | PARITY | MISSING |
| primary-peer-first-peer-allocation-first | 未知 | 未知 | 包含 1；界限缺失 | PARITY | MISSING |
| placement-candidate-first-candidate-allocation-first | 未知 | 未知 | 包含 1；界限缺失 | PARITY | MISSING |
| placement-candidate-first-peer-allocation-first | 未知 | 未知 | 包含 1；界限缺失 | PARITY | MISSING |
| placement-peer-first-candidate-allocation-first | 未知 | 未知 | 包含 1；界限缺失 | PARITY | MISSING |
| placement-peer-first-peer-allocation-first | 未知 | 未知 | 包含 1；界限缺失 | PARITY | MISSING |

## mem_reclaim / immutable-snapshot-read-4096

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.381227 | 0.000 | [1.344060, 1.496045] | WIN | RAW |
| primary-peer-first | 1.377683 | 0.000 | [1.353511, 1.489980] | WIN | RAW |
| placement-candidate-first | 1.379940 | 0.000 | [1.354210, 1.531933] | WIN | RAW |
| placement-peer-first | 1.364918 | 0.000 | [1.335764, 1.405280] | WIN | RAW |

## mem / isolated-heap-lifecycle-64

原 combined exact：`PARITY`；审计：`NO_ACTION`；证据：`CURRENT_DOCUMENT_ONLY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 1.024100 | 0.000 | [1.008700, 1.036500] | PARITY | DOC |
| primary-candidate-first-peer-allocation-first | 1.011000 | 0.000 | [0.989100, 1.037900] | PARITY | DOC |
| primary-peer-first-candidate-allocation-first | 1.011200 | 0.000 | [0.986700, 1.037300] | PARITY | DOC |
| primary-peer-first-peer-allocation-first | 1.013900 | 0.000 | [0.993000, 1.034600] | PARITY | DOC |
| placement-candidate-first-candidate-allocation-first | 1.020200 | 0.000 | [1.000200, 1.041100] | PARITY | DOC |
| placement-candidate-first-peer-allocation-first | 1.006800 | 0.000 | [0.976900, 1.039400] | PARITY | DOC |
| placement-peer-first-candidate-allocation-first | 1.017600 | 0.000 | [0.995500, 1.040600] | PARITY | DOC |
| placement-peer-first-peer-allocation-first | 1.020600 | 0.000 | [0.981100, 1.031200] | PARITY | DOC |

## mem_vm / model-loader-lazy-release-64

原 combined exact：`INCONCLUSIVE`；审计：`NO_ACTION`；证据：`CURRENT_DOCUMENT_ONLY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.033100 | 0.000 | [1.010800, 1.048100] | PARITY | DOC |
| primary-peer-first | 1.030700 | 0.000 | [1.012100, 1.040200] | PARITY | DOC |
| placement-candidate-first | 1.038500 | 0.000 | [1.008100, 1.057000] | INCONCLUSIVE | DOC |
| placement-peer-first | 1.046700 | 0.000 | [1.016500, 1.054600] | INCONCLUSIVE | DOC |

## simd / expert-top-k-f32-256-6

原 combined exact：`PARITY`；审计：`ORDER_OR_PLACEMENT`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 0.990501 | 0.950 | [0.966840, 1.022986] | PARITY | RAW |
| primary-peer-first | 0.989142 | 1.086 | [0.962797, 1.010234] | PARITY | RAW |
| placement-candidate-first | 0.984152 | 1.585 | [0.954637, 1.008147] | PARITY | RAW |
| placement-peer-first | 0.986975 | 1.302 | [0.981475, 1.000915] | PARITY | RAW |

## simd / split-swiglu-f32-11008

原 combined exact：`PARITY`；审计：`ORDER_OR_PLACEMENT`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.002530 | 0.000 | [0.981742, 1.028910] | PARITY | RAW |
| primary-peer-first | 0.998048 | 0.195 | [0.982911, 1.036344] | PARITY | RAW |
| placement-candidate-first | 0.997435 | 0.256 | [0.966932, 1.021713] | PARITY | RAW |
| placement-peer-first | 1.000243 | 0.000 | [0.975307, 1.022532] | PARITY | RAW |

## simd / weighted-rmsnorm-f32-4096

原 combined exact：`PARITY`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first | 1.009253 | 0.000 | [0.982091, 1.038734] | PARITY | RAW |
| primary-peer-first | 1.001350 | 0.000 | [0.977328, 1.013539] | PARITY | RAW |
| placement-candidate-first | 1.002011 | 0.000 | [0.994902, 1.033538] | PARITY | RAW |
| placement-peer-first | 1.014199 | 0.000 | [0.978542, 1.028392] | PARITY | RAW |

## container / model-cache-lru-mixed-access-4096

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 1.224251 | 0.000 | [1.173355, 1.274881] | WIN | RAW |
| primary-candidate-first-peer-allocation-first | 1.249704 | 0.000 | [1.219795, 1.284056] | WIN | RAW |
| primary-peer-first-candidate-allocation-first | 1.242946 | 0.000 | [1.224562, 1.273262] | WIN | RAW |
| primary-peer-first-peer-allocation-first | 1.245972 | 0.000 | [1.207314, 1.278704] | WIN | RAW |
| placement-candidate-first-candidate-allocation-first | 1.249797 | 0.000 | [1.213752, 1.265777] | WIN | RAW |
| placement-candidate-first-peer-allocation-first | 1.222328 | 0.000 | [1.193518, 1.290414] | WIN | RAW |
| placement-peer-first-candidate-allocation-first | 1.272756 | 0.000 | [1.218113, 1.303650] | WIN | RAW |
| placement-peer-first-peer-allocation-first | 1.238841 | 0.000 | [1.213136, 1.287827] | WIN | RAW |

## container_concurrent / read-mostly-membership-8x4194304

原 combined exact：`PARITY`；审计：`ORDER_OR_PLACEMENT`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| candidate-first | 0.988241 | 1.176 | [0.961304, 1.025337] | PARITY | RAW |
| peer-first | 0.977651 | 2.235 | [0.954083, 1.005561] | PARITY | RAW |

## container_queue / unbounded-mpmc-handoff-4p4c-262144

原 combined exact：`WIN`；审计：`NO_ACTION`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| candidate-first | 1.288767 | 0.000 | [1.102234, 1.496586] | WIN | RAW |
| peer-first | 1.261436 | 0.000 | [1.084888, 1.593879] | WIN | RAW |

## container_text / model-metadata-utf8-to-utf16-4096

原 combined exact：`PARITY`；审计：`ORDER_OR_PLACEMENT`；证据：`HASH_VERIFIED_CURRENT_SUMMARY`。

| stratum | r | 负向幅度 % | CI | 原 exact | 来源 |
|---|---:|---:|---|---|---|
| primary-candidate-first-candidate-allocation-first | 0.999618 | 0.038 | [0.973024, 1.004833] | PARITY | RAW |
| primary-candidate-first-peer-allocation-first | 0.996306 | 0.369 | [0.975916, 1.011956] | PARITY | RAW |
| primary-peer-first-candidate-allocation-first | 1.002737 | 0.000 | [0.986343, 1.020683] | PARITY | RAW |
| primary-peer-first-peer-allocation-first | 0.986317 | 1.368 | [0.977586, 1.012847] | PARITY | RAW |
| placement-candidate-first-candidate-allocation-first | 0.992644 | 0.736 | [0.966531, 1.008585] | PARITY | RAW |
| placement-candidate-first-peer-allocation-first | 0.995786 | 0.421 | [0.986332, 1.018994] | PARITY | RAW |
| placement-peer-first-candidate-allocation-first | 1.007151 | 0.000 | [0.977683, 1.029230] | PARITY | RAW |
| placement-peer-first-peer-allocation-first | 1.010171 | 0.000 | [1.002220, 1.024873] | PARITY | RAW |
