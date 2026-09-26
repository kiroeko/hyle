# D5：Metal 原生 workspace 分配失败

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- owner：`compute_ggml_runtime`，从 M1-A CLOSED 按可复现资源缺陷重开。
- baseline：`cb16f5669`；Windows 为唯一产品真相，MacBook Air 是验证副本。
- trigger：MG4 完整 recurrent region 首次使用四个原生 cold workspace；静态资源测试
  `GgmlMetalFaults.CompleteGateColdWorkspaceAllocationsRollBack` 在第三个 C 分配故障点 SIGSEGV。
  macOS crash report 的首帧为 `ggml_metal_buffer_init`，源码 `calloc` 后直接写 `res->dev`，
  没有检查空指针。普通数值批次已通过，不用正常执行代签 OOM。
- consumer：`compute_provider_ggml_metal::NativeProgram` 准备完整 gate+GDN，分配失败应返回
  `ResourceExhausted`，释放已取得的前序 buffer/backend，不提交或发布部分结果。
- scope：固定 ggml Metal `ggml_metal_buffer_init` 的元数据失败、部分初始化回滚与紧邻的
  `ggml_backend_metal_buffer_type_alloc_buffer` 所有权交接；
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

## AUDITED 有限 FIX

| ID | 源码证据与范围 | 固定修复与验证 |
|---|---|---|
| MB1 | `ggml_metal_buffer_init` 的 calloc 后未检查即写入；已实测崩溃 | 空值直接返回；现有逐点 C 分配故障验证 |
| MB2 | 同一 constructor 的 MTLBuffer/residency set 失败仅 free 元数据，泄漏已取得的 native owner | 调用已有 `ggml_metal_buffer_free`，保持释放顺序；不重写 allocator |
| MB3 | 紧邻 backend wrapper 调用 `ggml_backend_buffer_init`，其 throwing new 失败会丢失原生 buffer | 在本交接点异常清理后重新抛出，交由既有 HYLE adapter 翻译；逐点 C++ new 故障验证 |

清理审计：已有 free 会移除 residency set、release MTLBuffer/rset、释放 owned shared host memory，
最后释放元数据；rset remove 对 nil 直接返回，尚未加入集合的对象也可安全移除，ObjC nil release 安全。
MB2 的原生 MTL 分配/rset 系统失败分支以源码审计支持，不冒充已被 calloc/new 注入覆盖。
MB3 复用同文件已存在的 checked import 异常清理模式，不改变后端接口或上层 ownership。

正常路径只增加一次空指针分支和零成本异常清理区域；没有新 allocation、copy、锁或热执行成本。
manifest 与中央锁哈希随这三个限定修改更新；其它路线未采用 Metal manifest，实际源码输入不变。
