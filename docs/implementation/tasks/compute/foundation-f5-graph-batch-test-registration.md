# F5 Graph batch源码测试归属补齐

状态：CANDIDATE验证补充，product candidate25e4fb398不变；当前授权仍在
[候选卡](foundation-f5-candidate-validation.md)。只有tests/graph_batch测试与归属登记，
不改产品/API/算法/依赖图、公共checker或SDK合同，不重开A/F4/原Graph。

## 反证与有限FIX

公共component configure合同的CPU/tests=ON真实源码parent失败：graph_batch已是选中factor，
但没有tests/graph_batch/CMakeLists.txt。SDK tests=OFF与private实际消费者不代签这项。
[原失败](../../../../evidence/correctness/foundation-f5-component-test-owner-failure/README.md)
及原103.437秒完整保留，不能以关tests、改checker或空fragment绕过。

主consumer为公开MaterializedBatchProgram→materialized_graph可保活的固定Graph配方；
正交为不含模型/token的分段矩阵/资源batch。已有CPU/native实际Direct/Graph数学与故障
证据继续负责执行路径，这个无GPU依赖测试只验共同公开Graph配方/owner/失败合同。

| FIX | 固定范围 | 验收 |
|---|---|---|
| GRAPH-BATCH-TEST-OWNER | 新增tests/graph_batch/CMakeLists.txt，登记batch.h独立TU及真实共同Graph测试；现有公开配方的paged/ragged×F16/BF16端口/输出描述与artifact独立保活、invalid program拒绝、caller Graph allocator OOM不产生半成品 | 普通源码tests=ON可配置；所选Graph batch测试及头实际通过；原component合同同预算续跑 |

不人为要求新增GPU math、通用mock Queue、CPU route TEST_DEPS或生产路径改动。
测试直接调用graph_batch共同公开API，只依赖已有PUBLIC graph/compute_batch/error/mem闭包。
三交付源码/API不变，SDK8格与CPU/common sanitizer范围保持原证据资格。

预算：一次有限测试归属修复。component-contract总configure300已用103.437，同配置/机器/
工具链沿原失败预算续跑；新本factor所选测试按F5原configure300/build1800/test120/头120，
同身份已有tree/cache复用，不回放旧算子全集。若产品事实要求新API或依赖，先退AUDITED，
不能在测试补充里扩产品。每轮docs/closure/diff；最终F5七门仍全部按完整goal裁决。

三项正常登记测试与batch.h独立TU已在MSVC19.50/shipping/static/WERROR实际PASS；
[证据](../../../../evidence/correctness/foundation-f5-graph-batch-test-registration/README.md)保留
首个测试Result={}模糊赋值的编译FAIL、同root累计及逐cold allocation回滚记录。
只测试补充，product25e无变化。公共component配置合同正在同原总预算复验，不先写PASS。

同budget复验已通过原CPU/tests=ON位置，随后停于graph-only要求完整Compute façade的
旧工具断言；[独立工具跟进](../tooling/foundation-f5-component-facade-assertion.md)保存精确事实。
本模块不修改公共工具/产品依赖来掩盖问题，聚合component门未PASS；测试归属FIX本身已验。
