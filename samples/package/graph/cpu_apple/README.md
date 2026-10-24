# CPU + MLX/Metal 片段组合

普通入口为`hyle/graph/cpu.h`、`hyle/graph/apple.h`与共同Graph API，
选择`HYLE_COMPUTE_ROUTES="cpu;apple"`和`graph;graph_progress`产品，
链接`hyle::graph`与`hyle::graph_progress`；进度服务公开提供所需Sched依赖。
原生资产与实现依赖自动加入；单Apple路线的普通提交无需CPU或进度服务。

示例包含CPU RMSNorm/普通主存投影→完整原MLX affine-Q4 FFN→CPU消费，以及CPU分块变换→
Apple权重迁移/MatMul/RMSNorm→CPU结果。前者CPU大权重16KiB，native边界容量仅512字节，
不要求全部CPU权重变成GPU暂存。两独立Invocation验证重复执行和旧result lease禁止覆写。

源码：`cmake -S . -B build -DHYLE_SAMPLE_SOURCE=<hyle-source>`；安装包：
`cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk>`；构建使用`--parallel 1`。

调用顺序为`compile → create_invocation → bind → prepare_replay → submit → wait/result`。
编译环境持实际CPU/Apple Source和存储域，普通Invocation用QueueBinding绑定CPU Queue；
`prepare_progress`复用原槽推进，GPU保留原Context/stream。WeightBatch通过`graph::apple::bind`保留原上传owner，
无需调用方先wait。更换绑定需要重新prepare，
不会重新选择Provider。原生边界为显式host completion，`HostWaitPolicy::kReject`会拒绝这项组合。
资源预算只管请求字节；MLX私有分配未知，需要显式`allow_unattributed=true`，报告不承诺总物理内存。
保留result lease到CPU读取/GPU使用结束；外部payload、资源、Queue、Scheduler与allocator仍由应用管理。

迁移后的本例已在MacBook Air用当前host闭包与已有SDK实跑，含原Q4数学、资源和旧结果检查；
新的source/static/shared交付尚未重新运行，不沿用旧专用执行器的交付验收结论。
