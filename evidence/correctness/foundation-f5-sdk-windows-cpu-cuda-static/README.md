# F5 candidate Windows cpu,cuda static迁址SDK

固定candidate77df1c10c，MSVC19.50 Release/shipping/static，普通Compute/Graph SKU。
同身份原producer tree/cache，只构建compute_batch、graph_batch与cpu_cuda直接路线。
公开CPU与CUDA缺席消费者复制进隔离consumer-source；只从实际rename后的prefix发现头/库。
CPU四域paged/ragged×F16/BF16的Direct/Graph独立mean oracle PASS。actual RTX4070
CUDA Direct BatchPlanning给具名Unimplemented；typed ragged Graph执行前具名拒绝。
Windows未准入原生batch，不能用此诊断或CPU数学代签Linux FlashInfer/D256。

14个实际供应头独立TU PASS；Apple/Vulkan各三头缺席，缺route find_package具名拒绝。
staging实际rename、CMake metadata无源码/build/staging路径；dumpbin imports、干净runtime
PATH实际运行及DLL hashes见[receipt](receipt.json)与raw.zip。static HYLE不假称第三方
runtime全部静态。测试源码hash、原命令/身份/累计时间完整保留。

初次static helper在install后错误把已选CUDA头加入Vulkan缺头断言，纠正测试脚本；
产品代码未变，原成功stage及elapsed全部保留。本次没有产品FAIL。

本格只签CANDIDATE SDK资格，适用san/P0/LOC/guide/红队和最终closure继续；20 owners
保持CANDIDATE，goal active。没有全库回放或P1。
