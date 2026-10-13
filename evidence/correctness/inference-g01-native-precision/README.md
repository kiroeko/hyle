# G01 原生 Vulkan 精度候选证据

candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`，owner见
[原生精度卡](../../../docs/implementation/tasks/compute/inference-g01-vulkan-native-precision.md)。
[receipt](receipt.json)与[raw.zip](raw.zip)保命令、输入/原生/binary关联及全部失败，不签closure。

实际Windows GPU-only source public consumer通过：F32/F16、D64/128/256、B2/GQA8/2、
Q1/Q17、causal/noncausal、KV31/capacity37/offset1，两个seed与释放旧结果后的重放，共72
数学episodes；原oracle/容差不变，padding不参与，source不写，独立slot与旧输出保留。
原数值FAIL保在[前一阶段](../inference-g01-resident-stage/README.md)；本阶段首次还含
错误的bind拒绝断言，核原实现后改成submit拒绝，产品生命周期没有因此改变。

额外原生反例让N1/MHA在normal和strict中都走同scalar tiles，交错两模式且双顺序创建。
strict的131072大指数F32值保持有限，normal结果前后不受污染，证明precision key不混用
原half pipeline。该反例含内部类型，只签风险/承载，不冒充public SDK资格。

借的是已有full-F32 scalar SPIR-V；16个原host context edits与before/after hash一致，实际
build-local native source after hash核对通过，shader源码未改。私有slot tag、float shared
memory admission与cache precision位，不更改device features或另造queue/runtime。
source-contract六项通过；这不签device kernel/driver私有内存上界或全域资格。

首次native setter不可见改用原tensor私有参数数组，私有consumer采用仓库既有MSVC
intentional alignment flag；原build失败保留。本机4 jobs、tiny run约1.4秒，没有大模型run。
selector的global保守计划与仅一个dependency/两个content hash的delta证明保留；没有将
未运行global rows写PASS。其他peer/source数学不变，其版本/opaque/capture限制仍保留。

CPU resident子批已有[模块/默认入口证据](../inference-g01-resident-stage/README.md)。
本批三交付、适用san/OOM/最后review和定向P1尚未完成，三个owner仍CANDIDATE；
多请求直接驻留、Apple/Vulkan联合分页及G03–G10继续由goal实施，完成数仍0/10。
