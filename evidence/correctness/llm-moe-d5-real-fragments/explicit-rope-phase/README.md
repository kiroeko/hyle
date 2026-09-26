# D5 F35：显式 RoPE 相位修正的 CPU 定向证据

任务见 [F35](../../../../docs/implementation/tasks/compute/llm-moe-d5-explicit-rope-phase.md)。

`4f18811a` 的 CPU Direct/Graph 完整 DeepSeek attention 和长位置压缩状态均失败。
三个 ggml adapter 原本构造 F32 reciprocal，再由 ggml 除回；Apple 已直接乘频率。
`receipt.json` 保存不依赖产品实现的 IEEE F32 两步标量复现、候选身份、原始日志 SHA256。

`b111e3b7` 使用锁定 ggml 原语直接计算 phase，未修改测试 oracle、误差门或测试位置。
Release/WERROR 编译通过；实际执行并通过7个 GTest：Provider的3个 RoPE/类型/布局回归，
CPU Direct的完整 DeepSeek attention及stateful compression两项，Graph对应两项。
覆盖两份GGUF完整attention和四份DeepSeek画像适用的F32压缩语义，含65536附近位置、
实际状态反馈和重置。新增Provider回归另覆盖batch/head广播、零和负频率、partial tail、
interleaved/half-split。官方FP8投影与MTP没有由这些F32片段代签。

定向命令同时选中的CPU executor binary匹配零个测试，明确不计定向覆盖。
同一候选随后完成完整CPU owner回归，9个CTest目标全部通过：ggml Provider43、Compute Ops57、
Direct141、Graph156、runtime7个GTest（合计404）；另有CPU executor、runtime源码合同及两组
原生分配故障检查。完整日志没有SKIP。CPU Direct/Graph分别261.87/260.34秒；这是测试耗时，
不是P1性能结论。本收据暂不增加完整画像格、不签CUDA/Apple/Vulkan、不签三交付或P1。
原失败与修正后的实际命令、完整日志、JUnit、工具链缓存身份均以gzip保存。

P0：helper最多创建约28个tensor/view节点（包含类型转换与partial尾部），复用既有冷metadata
arena；phase/cos/sin按sequence×rotary-pairs广播到head/batch。CPU和CUDA每recipe节点的32/64
预算以及Vulkan RoPE的29个节点加输入/固定余量覆盖该上界。新增节点和workspace有成本，
不能继承旧fused RoPE性能结论；D5-A6仍待最终稳定路线的同语义验收。

统一脚本当前Compute+Graph实现58,965行、233个文件，排除测试及第三方源码，见`loc.json`。
