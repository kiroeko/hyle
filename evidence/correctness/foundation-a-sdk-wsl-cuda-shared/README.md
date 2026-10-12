# Foundation A WSL CPU+CUDA shared SDK

candidate546b0d2468e5de29031c34668f4afa4bc31b9a17，产品等价snapshot9d0aa797a；
Ubuntu24.04/GCC14.2/CUDA13.3/真实RTX4070，原配置树按可用内存并行，test1。
有限两个exact shape、表外拒绝、CPU→CUDA→host数学、旧result与family quota明确拒绝 PASS。
实际install→rename迁址、两个公开头独立TU、ldd/runtime与缺Apple/Vulkan诊断 PASS。
原锁定LibTorch CUDA12与nativeCUDA13的第三方loader warnings保留，不升级依赖。
实际命令、耗时、身份及日志hash见job.json。
