# G01 Apple native ragged 有限轮终审

状态：CLOSED，2026-10-13；owner `compute_route_apple`，不可变产品candidate
`280f6f92ea97bf3b424113346f0264fb7b20df16`。授权与NR-01/02/03见
[活动轮卡](inference-g01-native-ragged-layout.md)。只关闭Apple已验有限域，整个G01仍未完成；
Vulkan短D64性能FAIL另处理，原F5与其它Apple数学/状态证据保留。

## 七门与七问

| Gate | 实际证据与边界 | Verdict |
|---|---|---|
| scope | owned host-known ragged layout，F16/BF16、D64/128/256，两/八request、empty/reorder、packed capacity与active output；NR-01/02/03固定 | PASS |
| architecture | 原View/SDPA/Concat→原MLX retained callable/Program；无新math/Queue/controller，全部output只一次Concat | PASS |
| correctness | [source](../../../../evidence/correctness/inference-g01-native-ragged-stage/README.md)：两tests含三D/两dtype、独立double oracle、readonly/tail、old output/slots、empty/reorder/zeroquota；[宿主风险](../../../../evidence/correctness/inference-g01-native-ragged-candidate/README.md)ASan/UBSan/TSan两tests各PASS、真实TU插桩核对 | PASS |
| delivery | 同上实际relocated static/shared各两public tests、installed public batch头object编译；SDK只链hyle::compute，不借private product头 | PASS |
| usability_docs | [Apple指南](../../../user/apple-compute.md)公开冷prepare/原slots与ready sources、layout变化/旧结果/borrow者先释放规则，明确paged/capture/private限制 | PASS |
| red_team | 两新增入口及四产品文件独立只读候选复检；owned version非caller spans，capacity/indptr/GQA/finite scale/domain已校验；[直接消费者](../../../../evidence/correctness/inference-g01-native-ragged-direct-consumers/README.md)原native及CPU+Apple exact/control/state回归PASS | PASS |
| engineering_performance | P0没有按Q/head展开KV；[独立多请求P1](../../../../evidence/performance/inference-g01-native-ragged-apple-m1-p1/README.md)QUALIFIED、5%非劣PASS、原verify0，方向与阈值分别记录 | PASS |

借原MLX fast SDPA pin1f8e74e3及Metal源码身份认证，依赖锁/许可证不改；原math/runtime
及actual retirement复用，不复制成熟算法。Context/资源/allocator仍沿原borrow合同，
Program保owned布局，warm不遍历metadata或调用lower。query/output O(QHD)、原mask
以及vendor private费用仍收费；不是zero-total-allocation或capture保证。

无FFI/public vendor新类型/兼容wrapper/新factor；现有source/static/shared原公开factory
消费自然可发现。已发现MemoryResource inline default ctor的incomplete-type反例沿原
G10/交付接口有限修正待办保留，不由此重写本冻结候选或声明所有Apple public接口完美。
这里验收factory initializer与本NR入口，不代签该未修默认构造旅程。

## 性能、失败与停止

实际Air Mac16,12/M4、macOS27.0.1、AppleClang21，M1-MAC engineering interactive QoS；
clean source c6d10f66、contract1f8e811f。同原Program/Invocation/两slots、F16 packed
ready sources、checked completion和单packed owning output；16 A/A、n15产品120、
controls30，845.543s、原verify退出0。混合D256比0.9997–1.0071方向INCONCLUSIVE；
长prefill0.9827–0.9894稳定LOSS约1–1.7%；D64 noncausal0.9961–1.0038方向INCONCLUSIVE。
三个5%非劣门PASS；长prefill LOSS不改名为方向parity，见[增量总账](../../evidence/performance/inference-resident-g01-2026-10.md)。
不签独立硬件Formal、BF16性能或全模型token/s。

原compile/测试池/比较器borrow teardown失败、错误peer文件路径配置FAIL4.868s均保留。
直接回归首次helper猜bin路径失败，实际父工程产物在hyle子目录；同树识别唯一可执行后
增量build0.062s/run3.696s通过。未改产品/math/容差/阈值；不换树重置失败成本。
ASan detect_leaks=0，原平台限制不签LSan/shader/driver/privateGPU内存，两侧native
private上界UNKNOWN。有限FIX与适用门已裁决，停止重复本域性能/全库调查。

统一LOC绑定candidate280/self-check PASS：全库2,871,923，自实现不含测试175,049，
Apple模块实现4,262。分页/CPU联合准确页G02、producerG03、metadata/growthG06和private
预算G10继续属于原goal；不是全部模式对称证书。

reopen trigger：本已验域可复现数学、直接KV/结构成本、owner/实际退休或allocator失败
错误，或具名真实消费者需要原合同未有的机制。历史todo或再看一遍不重开。
