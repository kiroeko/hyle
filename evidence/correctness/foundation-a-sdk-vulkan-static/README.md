# Foundation A Windows CPU+Vulkan static SDK

candidate546b0d2468e5de29031c34668f4afa4bc31b9a17，snapshotd86ea3b98产品等价。
不存在可复用的同配置CPU+Vulkan static树，故以共享锁源码cache/独立binary树准备，实际库j4。
build target仅graph_route_cpu_vulkan的交付闭包；原ggml shaders、oneDNN/IREE等依赖库编译
不是全算子验证，没有共享其他route/linkage的binary树或新增数学kernel。

install后真正rename迁址，原目录不存在，Hyle metadata无canonical source/build/staging路径。
普通SDK finite native+hybrid、control-only失败、public effect正常/error/cancel/zero-output/
close-old-result、四公开头独立TU、运行库imports与缺CUDA/Apple诊断全部 PASS。
实际命令/elapsed/j4/机器、每阶段终局与日志hash/runtime DLL hashes在job.json。未代签Air/WSL。
