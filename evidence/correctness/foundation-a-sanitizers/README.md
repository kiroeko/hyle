# Foundation A 定向 ASan / UBSan

固定product candidate546b0d2468e5de29031c34668f4afa4bc31b9a17，Windows完整archive9d0aa797a
同步到可丢弃WSL副本。Ubuntu24.04/GCC14.2，原独立ASan/UBSan树、shipping HYLE_TESTING=OFF，
仅graph factor closure（core/container/util/error/math/platform/mem/compute_ops/compute/graph），
未构建/测试全库或GPU算子。按实际内存j4/test1，原HYLE_WERROR=OFF政策保持，不压warning。

各 sanitizer：40项ExactNativeVariants/NativeReachability/GraphBuilder/ViewSpec PASS，加1项
Graph invalid/out-of-range/foreign query PASS，再加1项全图effects footprints预检 PASS。
最后原子用例覆盖跨片段同range不同ID、duplicate/missing、disjoint范围、tensor alias和quota OOM；
故新shape/control/alias与提取后的原effect校验都实际执行，不能以GPU运行替代此风险。

追加 [effect_preflight_test.cc](../../../tests/graph/effect_preflight_test.cc) 是测试专用overlay，
产品src/include未变，private CMake DEFER仅向现hyle_graph_test追加fixture。候选后的canonical
测试注册也只改tests；不新开产品candidate、不扩公共验证工具。

ASan首次错误把既有internal HYLE_DEV_FACTORS树与product/route selector混合，两个原配置
FAIL保留；同树移除product/route选择，精确graph factor配置后成功。预算各阶段累计，未因
恢复日志重置。UBSan直接用已确认的维护配置通过。命令、原失败、输入hash、消耗与结果见各job。
GPU/native mechanism、SDK和TSan未由本证据代签；本轮无新增global/runtime并发机制，TSan
适用性沿活动卡裁决，actual多slot/cancel仍以Win/Air public旅程证明。
