# Compute历史私有CPU MatMul文件清理

<!-- MODULE REVIEW: SCOPED -->

- owner `compute`；baseline `e8fec259ec6e687acce81affd09deec39c0e302b`；Windows canonical。
- trigger：用户批准[Native MatMul退役](cpu-native-matmul-retirement-decision.md)。两份私有kernel位于
  src/compute历史目录，引用扫描只有Native include/CMake；不是Compute runtime或SIMD公共机制。
- scope/FIX K1：删matmul_kernel_impl.h与matmul_kernels.h；配合operation_test两份既有矩阵fixture
  迁成熟ATen并保留矩阵schema/完整数值覆盖。其它runtime/公共头/路由/治理/四路线代码不改。
- 性能P0：删未被runtime使用的私有算法，不新增热metadata/dispatch；不重跑P1。
- 验证复用相邻MM批Win/Linux Native factor内编译的operation_test、普通Direct/Graph与三交付；
  Compute runtime和公共头源码identity全部未变，两私有文件不在runtime CMake编译单元/导出中。
  selector因目录归属选出的无实际include依赖的GPU消费者给出未变源码/编译闭包理由，不重审GPU。
- 生命周期/error/公共C++ ABI与原MR路由资格复用，仅原Native-only矩阵成功预期撤回。
- 每轮closure/docs/selector/统一LOC；一次批量验收，实际未运行不PASS，不扩大为Compute重审。
