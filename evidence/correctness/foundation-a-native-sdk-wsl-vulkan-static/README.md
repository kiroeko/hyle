# Foundation A wsl native-only vulkan static SDK

candidate546b0d2468e5de29031c34668f4afa4bc31b9a17；只选原生vulkan路线，无CPU。
同配置构建树增量target闭包，原锁与source cache，库按可用内存j4/j2/j1，test1。
真实install→rename迁址、公开头独立TU、runtime依赖、缺CPU及其他GPU路线明确诊断 PASS。
软件llvmpipe没有合格GPU：按活动卡仅签source/包构建、头、loader与迁址，GPU数学NOT_APPLICABLE。
实际硬件/工具链、命令、耗时与日志hash见job.json，非其他SKU代签。

原始日志逐byte保存在raw-logs.zip；job stage SHA指archive原条目。旁列.log仅移除行尾空白，
其mirror SHA另记readable_logs；不修改原失败、命令或结果。
