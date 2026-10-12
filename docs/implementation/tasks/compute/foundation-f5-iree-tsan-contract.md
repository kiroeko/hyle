# F5直接消费者的IREE TSan配置合同修复

## Closure card

- 状态：AUDITED；owner compute_provider_iree_vulkan，baseline176fabcc331f193d9c93d1f525f9798319277762。
- trigger：actual Air Vulkan F5 TSan消费者原数学通过后IREE wait count异常；
  [具体反证](../../../../evidence/correctness/foundation-f5-native-vulkan-tsan-layout-failure/README.md)
  证明同一公开operation在adapter64字节/runtime56字节，count偏移80/72不同。
- 原CLOSED范围：[Vulkan terminal](vulkan-parity-final-audit.md)，原candidate a27a9b17f043f557726dd73a023e38e70e3e270d；
  只由这个新可复现sanitizer构建缺陷reopen，不回放数学目录/原native性能。
- consumer：普通Compute/Graph Vulkan F5公开Direct/Graph与原native风险消费者。
  正交是相同原async semaphore operation的提交/完成ABI，不含模型语义。
- scope：owner内部CMake原IREE sanitizer开关保持adapter/runtime结构一致；无新API/算法/layout
  合同、Provider选择、依赖版本/许可/中央factor边、kernel/runtime或公共build工具改动。
- finite FIX：IRE-TSAN-01，当HYLE_SANITIZER=tsan时在原private block设置IREE_ENABLE_TSAN=ON；
  保原ASan路径，其余配置尤其san=off无新增选项/费用。仅明确运行必要配置，不扩san功能集。
- 性能P0：只san配置，普通warm对象/算法/代码没有变化，不跑P1；比较除新条件分支外文件投影
  完全相同，compile flags与原IREE官方选项必须实证，不能只改header宏伪造兼容。
- 预声明：同实际Air TSAN root/source cache/Clang/MoltenVK身份，configure300/build1800/
  两consumer180原累计继续；源码bundle单向同步。原失败和原探针封存，修复后actual runtime
  与adapter均instrument，原公开四域和native故障/append/F4/close/guard消费者，实际flags/ABI核对。
- 普通交付：固定条件投影证明san=off、原ASan/UBSan配置无diff，原18 SDK不凭新commit重跑；
  若发现普通输入/target/链接或合同变化则撤销相应资格并按实际影响补验，不拿“只两行”代证明。
- 调查：一次原因核对已以实际sizeof/offsetof与vendor头确认；一次固定修复与同预算验证。
  stop为原stage预算耗尽、出现超scope事实或新的运行反证，按协议裁决/退阶段，不无限扩仪器。
- 每轮selector影响计划、docs/closure/diff；registry变更用明确previous-ref。Windows产品真相，
  Air只验证副本；只原owner/真实直接消费者，跨机器重型串行、实时内存j4/j2/j1、consumer1。

## 审计结论与七问

这是BORROW原IREE TSan选项的配置缺陷，不自建同步或修改用户operation。官方头把tsan_bridge
放在条件字段中；adapter带TSan时不能链接未带TSan的runtime。原ASan已按同一原则设置原选项。
准入由真实consumer与原ABI定义共同证明，唯一FIX已定，无benchmark驱动入口或领域策略。
原锁/许可不变；原HYLE/F1/F2/native owner复用；成本P0仅instrumented配置；无新FFI；
source与san配置必须匹配、普通SDK投影复用；无新用户API，文档只记录san资格和失败预算。
实际SCOPED记录已先于产品独立提交；现按原sizeof/offsetof反证固定唯一IRE-TSAN-01，
完成SCOPED→AUDITED后只实施该开关同步。其余原F5完成定义、失败预算和未裁决门不变。
