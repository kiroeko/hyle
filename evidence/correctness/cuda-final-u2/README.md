# CUDA U2：普通 preset 与架构传播子批

普通 `compute-cuda` preset 已实现：Compute+Graph、CUDA ON，其它三路线／tests／bench OFF，
Release、build j2。用户无需传内部 Provider flags；mixed 可显式 CPU=ON。
来源与有限 owner 范围见 [U2 卡](../../../docs/implementation/tasks/compute/cuda-final-u2-build-contract.md)。

## 实际配置与反例

初版 `f1937b27` 的普通 preset 在 Windows Toolkit13.3.73／MSVC19.50 实际 configure 通过，
同时发现 ggml 没收到 backend 的局部架构默认，使用原版更宽的
`75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real` 集合。
最初观察保留在该轮工具输出与 U2 SCOPED 卡；scratch 的初版 configure.log 被后续同目录配置替换，
不把归档里的修复后日志冒充初版原始日志。

修复在 CUDA SDK 成功发现后向父作用域传播现有 `80;89-real` 默认，以及用户 CUDAARCHS；
显式 CMake 变量优先，CUDA OFF 早返不探测、不设置。
原因是 ggml 在 enable_language(CUDA) 前定义自己的默认值，单纯“有环境变量就不设默认”仍不够。
没有改变数学、CPU 参数／resource overlay、Provider 源码／资格或公共对象布局。

| 配置检查 | 实际候选 | 生成 CUDA command 的 arch | 结果边界 |
|---|---|---|---|
| 默认／恢复默认 | 76929103 | 80＋89 | configure 与 154 条 CUDA FLAGS 行检查通过 |
| 显式 CMake arch89-real | 76929103 | 89 | configure 与生成 flags 通过 |
| CUDA OFF，独立目录 | 76929103 | 无 | 无 Toolkit／CUDA compiler probe，生成 CUDA FLAGS 为0 |
| CUDAARCHS=89-real | 7217b109 | 89 | configure 与实际 ggml／其它 native flags 通过 |
| CMake arch80 覆盖 CUDAARCHS89 | 7217b109 | 80 | 显式变量优先配置通过 |

这些是有精确 commit 的配置证据，**不是 compilation／operator／runtime／三交付 PASS**。
当前默认 producer 保留在 `C:/hv/cuda-final-u2`；最终 7217 的普通默认／全部交付随稳定接线候选集中验收。
一次具名环境优先级纠正后停止本子批追加配置姿势，不为文档重建／重跑 P1。
完整日志／收据／selector 计划见 [配置归档](configure-cases.zip)；产品／头／交付 selected gates 保持待运行。

U1 审计已结束；U2 B1 和 B4 架构接缝有实现及配置验证，B2 软件事实与 B3/B4 完整交付／能力仍未完成。
source/static/shared、DLL 邻置／搬移／offline、实际 CUDA 算子与状态，以及后续比较／规则／清退尚不签完成。

统一 LOC `--self-check` 通过：Compute+Graph 全部实现 **60,161 行／249 文件**，不含测试/vendor/bench。
`C:/hv/cuda-final-u2-off` 与 `C:/hv/cuda-final-u2-env` 只含本批临时配置，原进程已终态、路径及归档 SHA 已核对；
复杂与显式 literal-path 清理均被执行策略返回 `blocked by policy`，未实际删除，暂保留。
默认 producer、依赖 cache、本地模型及既有 sealed 证据未删除。
