# Apple Direct 权重迁移

使用`hyle/compute/apple_weights.h`，只链接`hyle::compute`。本例在普通CPU主存准备权重，
通过两槽原生Metal迁移池上传选中ID，然后由原MLX程序执行实际MatMul。

源码：`cmake -S . -B build -DHYLE_SAMPLE_SOURCE=<hyle-source>`；安装包：
`cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk>`。两者选择`HYLE_COMPUTE_ROUTES=apple`，
并由`hyle_stage_runtime`部署真实metallib和动态库。构建使用`--parallel 1`。

共享物理内存仍有原生提交与完成顺序。`binding`拒绝pending/failed数据，返回的array及所有alias
保留槽；`close`只排空迁移，不覆盖仍被计算程序/结果引用的权重。资源和allocator须晚于最后alias。
本例已在MacBook Air完成source/static/shared原生验收。
