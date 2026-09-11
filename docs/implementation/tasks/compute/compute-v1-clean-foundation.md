# Compute v1：清洁异构执行基础与 CPU one-op 纵切

## Closure card

- component / owner：`compute` / Compute；round=`compute-v1-clean-foundation`；
- baseline commit：`601b25ad4fe54baa1f8dd19153bb1230085b44d9`；
- trigger 与证据：Compute/Graph 长期需求已由
  [`compute-design`](../../../architecture/compute-design.md)与
  [`graph-design`](../../../architecture/graph-design.md)接受；用户明确要求旧实现不兼容、不得继续污染
  新设计。registry 把 `compute` 记为 `QUEUED`，中央组件图却仍构造旧 target/product，且旧公开头、源码、
  tests、bench、sample 与可选 gate 合计进入约 12.5k 行待审资产；
- 正常 consumer 任务：C++ 推理引擎在 CPU 上创建显式 Context/Queue，把 caller-owned Mem storage 绑定为
  TensorView，准备一次 RMSNorm 并重复提交，以 Completion 管理输出与 workspace 生命周期；
- 主产品纵切 / 具名来源：ggml/llama.cpp、ds4、vLLM、SGLang 的 normalization/低层 operator 调用；
- 正交反偏置纵切：工程计算消费者使用 typed host custom kernel 完成一个 SAXPY 类直接任务，不构造
  Graph，不取得模型语义；
- 跨模块 public journey：Platform 提供 host/ISA 事实，Mem 拥有 storage，SIMD 提供叶子向量核，Error
  表达失败，Compute 拥有 Device/Context/Queue/Completion、tensor binding、prepare 与 submit；
- scope：逐入口裁决并删除旧 Compute 产品面；建立最小 CPU Device/Context/Queue/Completion、
  TensorDesc/TensorView、Operation/PreparedOperation、CPU reference 与 HYLE-native RMSNorm、typed host
  custom-kernel direct path；为 one-op Graph 暴露稳定的窄编译/提交合同；
- non-goals：兼容旧 Device/Stream/Event/DevicePool/GgmlBackend/Kernel API；完整算子目录；CUDA、Metal、
  MLX、MPSGraph、Vulkan、HIP/ROCm；Graph 多节点编译；Native FFI；重开已关闭依赖；
- 直接依赖：候选为 `core`、`error`、`mem`，`platform`/`simd` 只在实现确实需要时登记窄 PRIVATE 边；
  不因 coroutine convenience 公开依赖 Sched；
- 直接消费者：本轮 `graph` one-op 纵切及独立 source/static/shared C++ consumer；
- 性能 tier 与原因：`P1`；RMSNorm 叶子与 prepared-submit dispatch 是真实重复热路，先做 P0 静态成本，
  再与相同 storage/layout/precision 的直接 SIMD/native route 对拍；
- 适用机器 / 仪器 / material margin：W1-WIN CPU 为工程性能机，统一 instrument；抽象税同时报告绝对
  ns 和百分比，5% 只作 material action 门，不把稳定方向改写为 PARITY；GPU 不适用本轮；
- 调查与纠正预算：资产/能力各一遍，API 纠正最多两轮，性能根因纠正最多两轮，每个合格 shape 最多
  15 个 paired samples；达到上限按协议裁决，不扩 benchmark；
- 预声明验证：旧入口/target/gate 负扫描；新 header isolation；正常、OOM/unsupported、move/destroy、
  submit/completion、并发 Context 与关闭测试；Windows/Linux/macOS CPU；ASan/UBSan，适用时 TSan；
  exact factor 与 Compute product 的 source/installed static/installed shared consumer；P0/P1、red-team、
  closure/docs/diff；
- stop / blocker 条件：Tensor/storage owner 无法与 Mem 合法组合、Completion 需要隐藏 process-global
  runtime、one-op 税有无法解释的 material loss、或中央 factor 无法表达独立交付时停止在明确状态。

## 资产隔离原则

旧源码不会复制到新的 `legacy/`、`reference/` 或兼容目录。审计只把仍有价值的事实浓缩到本卡：

| 旧资产族 | 审计问题 | 默认处置 |
|---|---|---|
| Device/Stream/Event 与 global GPU poller | 是否符合显式 Context/Queue/Completion 和有界 shutdown | `DELETE`，除非某个无依赖算法可原样证明 |
| DevicePool 与 compute memory factories | 是否错误复制 Mem storage owner | `DELETE`，需求回到新的 storage bridge |
| generic Kernel/host ops | 是否能成为 typed custom-kernel 路线，而非裸参数袋 | 只保留语义教训，重新实现 |
| ggml bridge/FFN façade | 是否泄漏 provider/model/runtime owner | `DELETE`；未来按 region provider 独立审计 |
| CUDA/Vulkan backend | 是否满足新 backend 合同 | 源码删除；能力分别在后续 factor round 重建 |
| tests/bench/sample/CMake gates | 是否只验证旧形状 | 与旧入口一起删除，不迁移成新验收证据 |

Git 历史负责恢复旧字节；历史 ggml storage 缺陷只通过架构页和既有调查记录保留。

## 当前状态

当前为 `SCOPED`。本卡尚未授权产品修改；下一步只做能力/入口/依赖审计并冻结有限
`KEEP/FIX/BORROW/DELETE` 集，随后才能进入 `AUDITED`。
