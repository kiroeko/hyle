# Foundation A CPU+CUDA static SDK

immutable product candidate 546b0d2468e5de29031c34668f4afa4bc31b9a17；snapshot9d0aa797a
只追加状态/施工卡。复用 Windows CPU+CUDA static shipping source 树，configure51.140秒、
实际 graph_route_cpu_cuda closure build22.750秒/j4、install2.891秒 PASS；没有全库测试。

实际安装后将 staging rename 为 relocated，原目录不存在。Hyle CMake metadata 不含
canonical源码/producer/staging路径。只以搬迁后的 prefix find_package(Graph)，没有添加
source include：native_variants.h、cpu_cuda.h 各独立 TU，普通 public finite consumer 编译/运行
PASS。consumer configure50.828秒/build10.938秒/j4，RTX4070/SM89 运行1.016秒 PASS。
dumpbin/runtime hashes与机器身份在job/identity/import日志；运行 PATH 只含消费者目录与系统。

缺 Apple/Vulkan 请求各 configure 失败且包含预期“SDK does not contain requested Compute
route”诊断，记 PASS_EXPECTED_ABSENT；不把它们当运行通过。整个 static gate PASS。
shared、其他路线、WSL、全部适用风险和终局仍待证据；本目录不签 A CLOSED 或 F4/F5。
