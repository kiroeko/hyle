# F5 native host-control sanitizer验证范围

状态：CANDIDATE验证卡补充，产品/API固定25e4fb398；不新增产品能力或修改公共仪器。
承接[活动候选卡](foundation-f5-candidate-validation.md)预声明的native host-control适用风险。
common与CPU三san证据不代签本页。

## 精确facet与执行

| 平台/facet | 实际验证 | 不由此证明 |
|---|---|---|
| Windows actual RTX Vulkan host wrappers/Graph factory/callback/owner | MSVC ASan，原materialized_native与public_native两真实消费者；正常四域、append/F4、old results、caller OOM、native错误/unknown/close/drain | GPU shader/driver heap未被ASan instrument；不是GPU内存或性能证书 |
| actual Air Apple host wrappers/Graph factory/worker边界 | Clang ASan，原materialized_native与public_native两真实消费者，actual Mac16,12身份先核对 | MLX/Metal私有GPU storage/预编译库未完全instrument |
| Linux actual CUDA FI host adapters/Graph batch coordinator/native callbacks | GCC ASan，原public CUDA与native Direct/Graph风险消费者；原C++宿主 instrument，kernel/driver不冒充 | GPU kernel字节访问须靠已声明实际数学/故障/权限证据，不称ASan能看GPU |

控制API均thread-compatible且externally serialized，不制造同一controller的并发control测试。
immutable layout/页/版本分享并发已由common TSan覆盖；新宿主跨线程退休边界如仍有适用
TSan/UBSan风险，必须逐代码/平台定界并补实际证据，不能把整个类别NA。此卡不先签任何PASS。

每个平台只原F5 owner与真实直接消费者，不跑旧算子catalog/全库/P1。复用身份匹配的源码
cache；sanitizer配置必须独立binary tree，不能把ordinary SDK binary当san。普通Compute/Graph
测试parent仅添加选定消费者并调用原hyle_apply_sanitizer，实际compile flags必须核对HYLE
controller/coordinator以及消费者。原native runtime/library借用边界/退休语义保持。

本native平台配置的F5新gate各configure300/build1800/consumer180（普通gtest120），第一次
记录后同root累计，所有FAIL/TIMEOUT保留。不是重置已用common/CPU/native ordinary预算。
库按实时内存j4/j2/j1，consumer1，跨机器重型串行。因公共工具/平台限制无法给必需verdict
则精确记录独立条件，不能用未instrument运行或CPU替代证明。
