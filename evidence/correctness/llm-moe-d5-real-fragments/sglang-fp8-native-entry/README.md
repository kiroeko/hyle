# SGLang FP8：CUDA 普通 Direct / Graph 接线检查点

候选 `3e098f5d3b20eb944681a6859f21e3a7553a66c0` 通过 Windows 原生普通入口与资源检查。
[收据](receipt.json)保存测试过滤条件、真实命令、结果和耗时；不是整格画像或P1收据。

- 公共低精度描述符6项通过，包括N1/127/128/129/576的尾块scale大小与输出形状。
- CUDA Provider4项通过：新FP8分配失败回滚、并发两invocation、Completion保活、无热分配、
  不支持格式/错误payload大小/I32越界冷拒绝，以及原Provider的两项生命周期回归。
- Direct和Graph各运行9签名×M1/M3×两次输入，共72次。共享同一官方投影夹具，包含
  DeepSeek UE8M0权重scale精确扩宽和GLM F32scale；保留原始FP8 codes，不常驻展开权重。
  夹具使用独立oracle和精确输出断言。普通入口选择成熟Provider，未调用HYLE自写数学。
- 共享输入事实脚本13项通过，文档与模块状态检查另行执行。

原版SGLang `06992c92` block128程序的cubin以固定字节嵌入模块，零runtime Python/JIT。
准备检查SM89/Driver13040、加载并完成原生函数初始化；submit只绑定地址并发射一次。
无caller workspace、复制或同步。Driver内部持久内存无法准确计数，报告unknown。
原版程序的生成来源和数值探针见[前置证据](../sglang-fp8-native/README.md)。

首次configure因生成vendor manifest时Windows Path排序忽略大小写而未匹配CMake的排序，
锁文件更正后构建与运行通过。原configure失败和对应收据保留，不改写为PASS。
构建37.66秒，所选测试1.23秒，无失败或跳过。

**仍未签署：** 动态激活量化、官方完整MoE/状态链、其它路线的该FP8接线、独立source/static/shared
consumer以及A6性能。本次是局部接线通过，完整画像ledger仍为140/176。
selector因共享依赖锁变更列出的扩展交付/跨组件门尚未整体运行，其计划随收据保存；没有用本批
4个CTest套件代签这些门，也没有重跑无关平台或公共性能仪器。

[统一实现统计](loc.json)：59,173行/235文件，排除测试、第三方源及嵌入的第三方生成程序。
