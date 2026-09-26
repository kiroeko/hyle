# D5：Metal 原生 workspace 分配失败

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- owner：`compute_ggml_runtime`，从 M1-A CLOSED 按可复现资源缺陷重开。
- baseline：`cb16f5669`；Windows 为唯一产品真相，MacBook Air 是验证副本。
- trigger：MG4 完整 recurrent region 首次使用四个原生 cold workspace；静态资源测试
  `GgmlMetalFaults.CompleteGateColdWorkspaceAllocationsRollBack` 在第三个 C 分配故障点 SIGSEGV。
  macOS crash report 的首帧为 `ggml_metal_buffer_init`，源码 `calloc` 后直接写 `res->dev`，
  没有检查空指针。普通数值批次已通过，不用正常执行代签 OOM。
- consumer：`compute_provider_ggml_metal::NativeProgram` 准备完整 gate+GDN，分配失败应返回
  `ResourceExhausted`，释放已取得的前序 buffer/backend，不提交或发布部分结果。
- scope：固定 ggml Metal `ggml_metal_buffer_init` 的元数据失败与本函数部分初始化回滚；
  通过已有 exact patch manifest 接缝修复，不升级上游、不改数学、kernel 或路线接口。
- non-goals：ggml 全库审计、其它 backend 的分配器重写、新公共治理/性能仪器、优化。
- 性能：P0，正常路径仅增加空值判断；无 API/layout/正常算法改变，不启动 P1。
- 验证：exact patch 哈希/原版文件匹配，Air static Provider 整套（含四 buffer 冷分配逐点故障），
  Apple shared Provider/Direct/Graph 的新 recurrent 定向批次；未改变的其它路线按产品输入身份复用。
  资源修复随 D5-A5 三交付批次统一接收，不把 factor 静态测试冒充三交付。
- 预算：一次 constructor/cleanup 源码审计，一个有限 FIX 批次和一次针对同一分配路径的纠正。
  若需要改变原生资源所有权或继续扩大函数集合，停止并重新定界。
- stop：失败可报告、前序 owner 回收和普通 recurrent 输出通过后形成候选，交付门待 D5-A5；
  不把未运行的 native OOM 子路径写成已测 PASS。

## 待审定边界

补查本 constructor 已分配 host memory/MTLBuffer/residency set 后的错误返回是否完整清理；
仅把同一函数的具名漏清理纳入本次有限 FIX，不扩成其它 ggml 能力或全库资源审计。
