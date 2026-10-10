# CPU + MLX/Metal 片段组合

普通入口为`hyle/graph/cpu_apple.h`，链接`hyle::graph`并选择`HYLE_COMPUTE_ROUTES="cpu;apple"`。
组合factor、CPU/Sched闭包和原生资产自动加入；单Apple路线没有这项CPU/Sched依赖。

示例包含CPU RMSNorm/普通主存投影→完整原MLX affine-Q4 FFN→CPU消费，以及CPU分块变换→
Apple权重迁移/MatMul/RMSNorm→CPU结果。前者CPU大权重16KiB，native边界容量仅512字节，
不要求全部CPU权重变成GPU暂存。两独立Invocation验证重复执行和旧result lease禁止覆写。

源码：`cmake -S . -B build -DHYLE_SAMPLE_SOURCE=<hyle-source>`；安装包：
`cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk>`；构建使用`--parallel 1`。

调用顺序为`compile → create_invocation → bind → prepare_replay → submit → wait/result`。
CPU片段使用应用借给HYLE的CPU Queue与Scheduler，GPU保留原Context/stream。更换绑定需要重新prepare，
不会重新选择Provider。原生边界为显式host completion，`HostWaitPolicy::kReject`会拒绝这项组合。
资源预算只管请求字节；MLX私有分配未知，需要显式`allow_unattributed=true`，报告不承诺总物理内存。
保留result lease到CPU读取/GPU使用结束；外部payload、资源、Queue、Scheduler与allocator仍由应用管理。

H2仍在实施，当前示例待原生source/static/shared验收，不据源码或配置测试宣称完成。
