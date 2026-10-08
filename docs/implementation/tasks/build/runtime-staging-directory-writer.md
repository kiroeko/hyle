# Windows runtime 邻置：同目录 writer 协调

<!-- MODULE REVIEW: SCOPED -->

## 工具任务合同

- baseline：`cf41d82a29c91a4cf935b0a55a804b891bacbb55`；Windows canonical。
- trigger：CUDA-only shared的193步构建在第161步失败；两个runtime staging目标同时向
  `C:/hv/m2b1-cuda/bin/torch_global_deps.dll`写入，`COPY_FILE`报output Permission denied。
  原日志在`.scratch/validation/cuda-final-u3-delivery/windows-cuda-shared/`，原编译对象保留。
- owner：公共构建工具`cmake/HyleRuntimeStaging.cmake`；registry没有build模块身份，
  这是独立tool task，不在CUDA模块内重写公共治理，不制造空component或重开运行时模块。
- scope：Windows同输出目录的构建writer互斥；源码和installed helper同一实现。
- non-goals：不改数学、runtime loader、API、Provider范围、DLL选择顺序、package resolver、
  closure checker或性能统计；不在应用执行时加锁，不串行化整个编译。
- 原机制：`HyleAppleRuntime.cmake`已有同目录`file(LOCK ... GUARD PROCESS TIMEOUT 60)`。
  Windows邻置没有相应协调；两计划都合法，但`ONLY_IF_DIFFERENT`不提供互斥。
- 有限FIX候选：生成的Windowscopy脚本创建目录后、遍历DLL前，取得同目录process锁。
  同目录writer串行、不同目录和编译仍可并行；已有conditional／owner／idempotent合同保持。
- 预声明：既有RuntimeStagingContract补第二个同目录consumer、j2初次／增量／无变化／
  DLL-only更新与两实际exe加载；实际CUDA shared的失败目标增量续跑；docs／closure／selector。
- 预算：一次源码审计、一行工具修复、一批fixture与实际CUDA交付验证，不触发算子P1。
  若是应用持有目标DLL而非writer冲突，锁不授权杀用户进程；具名保留诊断。
- stop：审计后固定上述有限FIX进入AUDITED，工具与实际构建验收完成即CLOSED。
