# 旧 C1：实施事实索引与授权终止

2026-09-19 用户要求重写计划并重启 goal。原 C1 没有完成，原 A–AB 自写/逐算子队列终止；
本页不再是活动施工卡。当前唯一队列为[总计划 M0–M6](llm-moe-coverage-completion-plan.md)，
当前 scope 见 [M0](llm-moe-m0-provider-scope.md)，所有已有自实现受
[冻结与退场合同](llm-moe-self-operator-freeze.md)约束。

## 保留的事实，不继承的授权

- CPU Native 已有半精度 tensor/norm/reduce/gate/index/scan 等；oneDNN 已接 BRGEMM、
  RMSNorm/LayerNorm 与 Add/Mul post-ops。Native 不是成熟库接入，现有算法冻结。
- CUDA 既有 cuBLASLt/Marlin/SGLang 和 HYLE kernel/CUB 路径；SGLang AB 已实施，
  `ebe5af51` W1 static provider/default Direct/Graph 与头/源码合同通过，其他适用门仍未完成。
- Apple 已接较多 MLX typed/region；HYLE checked/ordered Scatter Metal 仍属冻结自实现。
- Vulkan ggml managed region 已接 norm/gate、TopK/Gather/F32 Scan、Sum/Mean/Softmax 及
  Y35 稳定变换。Y34/Y35 的适用 correctness、W1/Air 三交付与所选工程资格保留；无完整性能优选结论。
- CPU Q2 测量资格与 ggml CPU runtime 阻塞保留原恢复条件。局部 PASS 不签整包 C1 或全矩阵。
- 原完整 A–AB 事实可用 `git show ebe5af51:docs/implementation/tasks/compute/llm-moe-c1-tensor-state.md`
  读取，不删除 Git 历史，不恢复其命令式施工授权。

## 历史证据入口

| 记录 | 用途 |
|---|---|
| [Provider 目录](llm-moe-c1-provider-reuse-audit.md) | 已重写的模型→Provider→接入方式与当前事实 |
| [C0](llm-moe-c0-local-contracts.md) | 本地输入、参考、量化及来源资格 |
| [oneDNN AA](llm-moe-c1-onednn-normalization.md) / [Q2](llm-moe-c1-normalization-aa-qualification.md) | 数值/交付事实与独立测量 blocker |
| [CPU Z](llm-moe-c1-cpu-matmul-precision.md) | 必要精度准入修复的完成事实 |
| [CUDA L](llm-moe-c1-cuda-device-status.md) / [SGLang AB](llm-moe-c1-sglang-primitives.md) | 错误/寿命及成熟片段接入的局部事实 |
| [Vulkan Y](llm-moe-c1-ggml-vulkan-region.md) | managed region、TopK/Gather/Scan 的接入与交付 |
| [Y33](llm-moe-c1-vulkan-layernorm.md) / [Y34](llm-moe-c1-vulkan-reduction.md) / [Y35](llm-moe-c1-vulkan-elementwise.md) | 对应批次已完成的资格与未决性能 |

原C1通知约定曾由M3-01承接；2026-09-20用户已取消该专门通知/暂停，改为**M5开始前暂停并汇报，
等待明确继续**。当前约定以总计划为准。不把移交、文档提交或旧测试数算完成。
矩阵和 registry 仍为当前能力/模块状态的对应真相。
