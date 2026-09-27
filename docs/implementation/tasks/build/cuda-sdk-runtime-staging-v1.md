# CUDA SDK DLL 的公开 staging 接缝

<!-- MODULE REVIEW: SCOPED -->

- baseline：`9401a0c3`；Windows canonical。owner：`compute_backend_cuda`。
- reopen trigger：D5-A5外部SDK消费者编译/链接完成，去掉SDK PATH、隐藏producer后，
  两个程序启动均为`0xc0000135`。PE导入审计确认缺少`cublasLt64_13.dll`与
  `cublas64_13.dll`；HYLE自身DLL均已由公开staging入口布置。原记录在
  `C:/hv/d5cd/receipts/shared-corrected`，该拒绝不是模型数值失败。
- 正常旅程：用户通过源码或`find_package(Hyle COMPONENTS compute graph)`选择CUDA，
  调用既有`hyle_stage_runtime_dependencies(target)`，在exe旁取得所需系统SDK运行库。
- 现状：CMake CUDA imported library target缺少可被`TARGET_RUNTIME_DLLS`遍历的SHARED
  DLL节点；现有staging只看到HYLE DLL。系统SDK仍由用户安装，不能伪造源码锁或把
  驱动DLL打包；不改变公共staging函数或package registry的通用协议。

## 有限FIX与审计

| FIX | 允许改动 | 验收 |
|---|---|---|
| S1 | backend-owned私有CMake package复用`find_dependency(CUDAToolkit 13.3)`；Windows为cudart/cublas/cublasLt现有import library声明其真实SHARED runtime依赖。保留原CUDA目标与include/link语义，缺DLL在configure明确失败 | source与installed使用同一文件；关闭CUDA不探测；当前Toolkit的三个DLL均来自其实际目录 |
| S2 | 仅backend的CMake接入并通过既有dependency registry安装该私有config、声明find_dependency；不改数学、C++、通用staging/checker、其它Provider选择 | shared SDK重新生成后，普通staging将CUDA DLL邻置；producer隐藏、SDK PATH移除仍运行 |
| S3 | D5-A5原8个真实片段测试完成source/static/shared；保留失败与最终PE/文件来源收据 | 改动前后相同模型/形状/误差门、两入口、实际状态反馈；选择器与docs/closure |

P0：纯配置/交付metadata，无热路径成本、对象/API/算法改变；D5-A6不由本卡代签。
依赖版本、许可来源与CUDA SDK安装前提不变。Imported runtime节点不构建替代数学库。
非Windows继续原SDK目标；本次不宣称重签Linux专有FlashInfer。

预算：一次上述接缝修复与一次三交付批次，最多一轮明确构建接线纠正。
沿D5-A5复用现有shared构建，source/static共用producer；无新增全库build或P1。
若需要改全局package/staging协议，停止并另行定界，不在本卡扩张。
满足S1-S3及适用门后才CANDIDATE/CLOSED；未运行项保持未验。
