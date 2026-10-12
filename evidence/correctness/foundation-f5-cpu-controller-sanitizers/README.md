# F5 CPU控制器三种sanitizer定向验收

固定数学/API输入77df，WSL GCC14.2/shipping/static，沿common已用的同source/tree与
configure300/build1800/consumer180剩余累计预算，未改root重置。private parent只加实际
公开CPU与materialized CPU消费者；普通Compute/Graph CPU组装，选择两个target，不跑旧
算子目录。调用原hyle_apply_sanitizer；ninja实际commands核对两个consumer、Compute/Graph
controller及原leased Graph宿主均有对应sanitizer编译flags，不以无instrument binary代签。

ASan/LSan、UBSan、TSan两consumer各PASS：public四域Direct/Graph数学；完整CPU八域
F16/BF16、paged/ragged、causal/非causal、NaN padding、same-generation/旧结果；
caller cold OOM、实际accepted-error停止后继、F2失败不发布/F4 head不变、精确页、append
重排/F4成功发布、取消/关闭与metadata quota0。覆盖以真实日志为准。

初次ASan private runner误选tests-on graph_batch，而旧disposable source没有该test ownership
fragment，配置FAIL保留；改用既有普通Compute/Graph tests-off组装和两个明确target，不删
产品测试/不修改公共CMake。三mode原common预算累计保留；TSan沿已证明的process-local
setarch -R，不加suppression、不改系统ASLR或算法。原common首次exit66亦保留。

这份证据只签本CPU宿主控制器与实际组合，第三方预编译CPU库未全instrument，
CUDA/Apple/Vulkan/private GPU/native host-control适用facet继续单独裁决。新的CUDA public
link-only修复不改变CPU源码/数学/依赖闭包，后续候选资格须按输入映射保留其精确范围。
所有命令/身份/hash/累计见[receipt](receipt.json)/raw.zip；最终F5 closure尚未签。
