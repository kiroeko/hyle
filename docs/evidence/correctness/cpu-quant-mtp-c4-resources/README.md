# C4 资源、失败与独立运行状态

固定消费者/产品候选 `216242d2fabdfae8e69d00872786aac0148442c1`。
Windows/Linux 各 10 项针对当前实际 Provider 与普通 CPU façade 的资源边界检查通过：
[Windows](windows/summary.json)、[Linux](linux/summary.json) 保留准确测试名、命令、JUnit。
没有重建产品或修改资源实现。

- ggml persistent 独立 slots、executor/close 与 invocation 分配失败回滚/恢复。
- 原 ATen 的真实分配失败、accepted failure 退役 slot、新 slot 恢复、重复/越界 expert；
  四个 artifact decoder 与三个 MLX linear 的独立 slot 并发、metadata OOM 与额度回收。
- 普通 Direct factory 冷组装 OOM 回滚，已有成熟 program replay、close 拒绝与 metadata 回收。
- 普通 Graph 独立 invocation/replay 状态、多个发布输出、close 拒绝与 owner 回收。

本批组合没有新的私有资源机制：GGUF region 委托同一 ggml invocation，原量化程序
委托同一 ATen managed boundary；常量通过只读 binding 保活，mutable cache 由各 caller
独立持有。实际 C4 source/installed 消费已执行 changed replay、真实 cache 反馈/wrap/reset、
共享 embedding/head，以及完成后读取输出；资源边界按相同未变实现复用上述针对性资格。
组合路径见 [三阶段](../cpu-quant-mtp-c4-three-stage/README.md)、
[Qwen](../cpu-quant-mtp-c4-qwen-sidecar/README.md)、
[Flash](../cpu-quant-mtp-c4-flash-sidecars/README.md)。

这些资格不宣称全链事务回滚：后续 operation 失败不会撤销此前已完成输出。
失败 invocation 退役后须重新创建；消费者负责自己的 cache 提交/回滚策略。
metadata allocator 的回收不是 ATen 全进程 RSS/私有分配硬上界。
没有全实权重模型、采样/接受策略或推理引擎验收。
