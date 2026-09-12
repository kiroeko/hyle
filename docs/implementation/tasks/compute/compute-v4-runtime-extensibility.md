# Compute v4：异步 runtime、storage bridge 与 provider 扩展合同

## Closure card

- component / owner：`compute` / Compute；round=`compute-v4-runtime-extensibility`；
- baseline commit：`98295cc24d28f35437b51b0ef1fdf1ee76872d5a`；上一轮已关闭的 storage
  合同与证据保存在 [`compute-v3-block-storage-format`](compute-v3-block-storage-format.md)，本轮不回写
  该历史记录；
- trigger 与证据：当前公开 `Completion` 对任意有效对象都立即 ready，`TensorView` 只能绑定 host
  `mem::BufferView`，`ProviderKind` 是封闭枚举且 `ProviderList` 固定最多四项。与此同时，已接受的
  Compute 架构和 ggml、llama.cpp、vLLM、SGLang、ds4 的真实 GPU/异步路线都要求 device-owned
  storage、异步终态、显式 provider/backend identity 与有界关闭。这是现有公开合同缺失，而不是
  benchmark 输赢；
- 正常 consumer 任务：C++ 调用方创建一个 backend context 和 ordered queue，显式绑定 host、pinned、
  device 或 unified storage，冷路径准备 artifact，重复异步提交并 poll/wait，最后 close/drain；运行中
  不重新搜索 provider、不隐藏 staging/copy/JIT/heap allocation，所有借用活到 completion 终态；
- 主产品纵切 / 具名来源：量化推理的一次 GPU/CPU prepared-region replay，参考 ggml backend
  buffer/event/graph plan、llama.cpp backend scheduler、vLLM/SGLang CUDA stream 与 ds4 CUDA/Metal/ROCm
  tensor/queue 路线；
- 正交反偏置纵切：renderer/工程程序把已有 Vulkan/Metal/CUDA resource 和 queue 导入 Compute，提交
  一个 typed kernel 后把 completion 交回原资源管线；不引入模型、KV 或 serving 语义；
- 跨模块 public journey 与每段 owner：Mem 拥有 host storage/resource 语义；Platform 拥有 OS/device
  事实；Compute 拥有 backend context、queue、device storage/interop、provider artifact 与 completion；
  `compute_ops` 将拥有 Tensor/Operation schema；Graph 只消费本轮合同并拥有跨节点计划；Sched 只通过
  后续窄 bridge 组合等待/取消；
- scope：审计并冻结 async-capable Completion、Context/Queue close/drain、开放 provider identity 与冷路径
  provider sequence、backend-neutral storage owner/view/resource bridge、typed native module/kernel/command
  seam，以及现有入口的 `KEEP/MOVE/REPLACE/DELETE`；只实现审计后有限 `FIX` 集；
- non-goals：扩大公共算子目录、Graph IR/partition、任何具体 CUDA/Metal/Vulkan/HIP/MLX 实现、模型或
  KV/batching/sampling/serving、训练/autograd、distributed collectives、Native FFI、字符串属性袋、
  process-global provider registry；
- 直接依赖 / 直接消费者：当前直接依赖为 Core/Error/Mem，私有使用 Math/Platform/SIMD/Util-core；
  本轮审计可把 Tensor/Operation schema 移到 `compute_ops` 后登记真实边。直接消费者为 direct Compute、
  Graph plan、oneDNN/ggml 与后续 device backend/provider factors；
- 性能 tier 与原因：先做 `P0`，登记 submit/wait/poll、storage view 与 provider dispatch 的 allocation、
  atomic、lock、syscall、copy、metadata 和最坏复杂度；若最终 runtime seam 比成熟同合同 event/queue
  多出未由语义解释的热成本，再对 immutable candidate 做一次 `P1`；
- 适用机器 / 仪器 / material margin：Windows、WSL Linux、两台 Apple 做 CPU fallback、lifetime、headers
  与三交付；真 GPU 只由各 backend 后续卡资格化。本卡如启动 P1，使用统一仪器、W1-WIN 主机和固定
  5% non-inferiority margin；
- 调查与纠正预算：四类推理来源各一次定向扫描；ggml backend、CUDA/HIP event、Metal command buffer、
  Vulkan timeline 与一个独立成熟执行 runtime 各一次合同扫描；API/owner 最多两轮；每个冻结 FIX 最多
  两轮根因纠正；candidate 后适用验证各一次；
- 预声明验证：invalid/move/destroy、同步与模拟异步 terminal、poll/wait/cancel、late completion、
  close/drain、queue/context mismatch、storage domain/range/alignment/ownership、unsupported/device loss、
  OOM/error injection、并发 invocation、逐头、sanitizers、四机 CPU fallback、source/static/shared、
  factor selection、export/interop hygiene、P0 账单及必要 P1；
- stop / blocker 条件：若 device storage 或 completion 的最小自然合同必须先由一个尚未登记的独立
  backend factor 拥有，先登记该 factor 而不把 vendor handle 塞入通用对象；若无法在不引入全局 runtime、
  隐藏等待或常态 heap allocation的条件下表达安全 late completion，则精确记录 blocker；不得缩回同步
  CPU-only API 冒充异构 runtime。

## SCOPED 待审计项

1. `Completion` 的 state owner、poll/wait/terminal/cancel、callback/coroutine bridge 与 late completion
   销毁顺序；哪些属于 core，哪些应留给可选 `compute_sched`；
2. `Context::close()`、`Queue::close()` 与显式 `drain()` 的 admission、in-flight work 和析构边界；
3. host、pinned、device-local、managed/unified storage 的 owner/view、mapping、copy 与 native interop；
4. `ProviderKind`、`ProviderList<4>` 与 provider/version/backend identity 如何替换成开放、冷路径且不让
   steady submit 支付搜索或 allocation；
5. CPU default/native fallback 如何在拆出 `compute_ops` 后保持自然调用，同时不让 schema factor依赖
   runtime/SIMD；
6. `Device`/`QueueOptions` 当前的封闭或空形状是否只是最小测试纵切遗留；
7. 公开入口逐项 `KEEP/MOVE/REPLACE/DELETE`，以及 source/static/shared 和后续 backend factor 的唯一
   构建投影。

AUDITED 前不改 Compute 产品代码、公开头、依赖或 backend/provider 实现。
