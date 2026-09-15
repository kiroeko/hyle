# Compute/Graph 基础架构接入与调度边界调查

## 本轮范围

- baseline：`0b5da87c`。用户要求把基础架构的大改风险查清，同时保留性能优先和渐进式 best-of。
- 保留现有 Device/Context/Queue/Completion、Graph/Executable/Invocation；本轮不写产品代码，
  不更改 registry、依赖锁或治理系统，不填算子目录，不创建新的大 goal。
- 固定问题：CUDA 压缩 kernel 的数值/冷配置边界；MLX 的公开资源、线程、异步完成/错误和重放；
  HYLE 能控制的全局调度与不能控制的第三方内部行为；局部替换的格式、融合和完整成本条件。
- 证据：前轮 CUDA 真机结果复用；核对当前 vLLM/SGLang/ggml/ds4 和 MLX 的具名入口，Vulkan
  只核对公开同步/队列合同。本轮只做一个 MLX C++ 接入实验，不签新 backend 或 Provider。
- MLX 调查预算至多两小时，一个实验、一轮 fixture 纠正。机器仅缺构建工具时可使用官方同版预编译
  runtime 作 C++ API 研究，不安装/修改全局开发环境；它不代替未来源码交付资格。
- 预声明实验：MLX v0.32.2 官方同版 headers/dylib；F32、M=4/K=1024/H=256 的
  gate/up MatMul → SwiGLU → down MatMul，两个提交线程各三次变化输入；host double oracle，
  逐元素 `abs(actual-expected) <= 1e-4 * max(1, abs(expected))`，不测完整模型精度；
  无复制输入的实际可用条件、完成前保活和完成后释放；同一编译函数跨提交线程是否再次 trace。
  不做 P1、不更换数值阈值跑到绿；不能主动制造 GPU hang/device loss。异步故障与未测场景按源码
  证据和未验证限制分别报告，不能冒称 fault-injection PASS。
- 该实验只裁决 MLX 自身的 import/lifetime/线程缓存，不冒充 HYLE Metal backend、跨 Provider
  FFN、量化、全局调度或性能纵切。正式纵切须从 HYLE Graph 进入并验证真实边界与失败回收。
- 输出：已有合同是否足够、真实矛盾需要的最小修正、后续实现门；文档/closure/diff 检查。
  调查完成不等于基础实现或性能已经通过，更不承诺永远不改任何内部实现。

## 结果

进行中。
