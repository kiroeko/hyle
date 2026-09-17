# C1-L：CUDA checked operation 的异步算法错误

## Closure card（SCOPED）

- owner / round：`/root`，`compute_backend_cuda`，LLM/MoE C1-L。
- baseline：`2aff9e352e65fc9f03e0a8497bf3bb513221248e`。
- 上位合同：[C1](llm-moe-c1-tensor-state.md) 的 F02/F19 checked Gather/Scatter、I32 Scan；
  [唯一计划](llm-moe-coverage-completion-plan.md)。本卡只补直接依赖，不重审 CUDA backend 全部能力。
- 合法 reopen trigger：原 [terminal](../build/vulkan-r6-product-delivery.md#终局2026-09-17) 允许
  “真实消费者必需能力/公开合同变化”。GPU resident token IDs、状态 scatter 索引和 scan counts
  必须在 GPU 检测越界/溢出，普通 Completion 返回实际错误；当前 CUDA EventSlot/poll/wait 只有
  CUDA runtime 错误，没有异步算法状态来源。prepare 阶段看不到这些运行数据。
- 当前具体源码：`cuda_backend.cc` 的 `poll_slot` / `wait_slot` / `finish_sequence_impl`、
  `wait_completion_impl`、capture begin/end/launch；`cuda_internal.h` 的 EventSlot/QueueState。
  现有 `EnqueueResult::status` 只能报告 enqueue 时已经知道的错误，不能代签 GPU 执行后判定。
- 主纵切：普通 GPU embedding Gather → norm/residual，输入索引每次变化，两独立捕获槽；坏索引
  不非法访存、不污染 CUDA context，不把结果误报成功，保留已接纳 prefix 与 owner 至终态。
- 正交纵切：I32 offsets Scan 的溢出；带错误的已接纳 CUDA prefix 被另一同 context Queue 导入
  依赖、原 Completion 被观察/丢弃后，错误与状态存储不能丢失或被下一轮重用覆盖。
- scope：CUDA 原生程序与完成边界的私有设备状态、原生 capture、等待依赖、failure/close/reaper。
  普通 C++ Compute/Graph API、通用 Completion/vtable、其余路线和公共构建/治理不改。
- 直接消费者：后续 CUDA checked Gather/Scatter/Scan；现有 CUDA provider/default Direct/Graph
  回归不能因该机制产生额外常态 GPU dispatch/copy。算子实现仍在 C1 的独立有限审计中固定。
- tier：P0；正确性先用真实设备状态写入及原生 capture/Completion，实际算子接线后随 C1 bundle
  完成 W1/WSL、source/static/shared、同 checked 语义的等责 P1。未执行的整包门不写 PASS。
- 预算：本机制定向调查≤60分钟、两项具名平台来源；产品纠正最多两轮。若需修改共同 API/
  Completion 所有权或其它路线，先重新定界；没有具体新反例时不扩大。
- 停止条件：先固定有限 FIX 和成本/生命周期，owner 独立提交进入 AUDITED 后才改产品；缺少
  必需平台机制则记录具体 blocker，不用 unchecked、host payload copy 或 device assertion 代替。

## 初步核对（还不授权产品改动）

1. CUDA 数据/索引仍留设备；只允许常数大小的状态回报，不新增每算子 CPU 等待。准备和 capture
   录制不能执行输入计算；成功的热重放不得申请状态存储或重新搜索程序。
2. Graph 的 capture 以 segment 录制，可能有多个 captured artifact；状态必须属于各自运行槽，
   并进入实际提交 prefix 的 Completion。不能把录制时 EventSlot 地址直接烘焙为以后任意 prefix。
3. 必须保留 `cudaStreamWaitEvent` 的非阻塞依赖能力；GPU 软件错误也要进入依赖方的最终状态。
   原 CUDA event 等待不会替软件错误判定。源槽存储须保留至依赖方不再读取，不能仅保留事件。
4. 没有 checked operation 的已验收路径不得新增 reset/readback/merge GPU 命令。新增的冷期
   metadata、设备/固定页 host 字节、状态 reset/readback 与必要合并命令均计入 P0/实际片段。
5. 平台依据：[CUDA 异步调用合同](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
   与 [CUDA Graph 的捕获和重复执行](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。
   实际编译/资格仍固定本机 Toolkit13.3，细节以安装 SDK 头与真实设备测试验证，不能从网站的较新版本
   推导本机通过。成熟实现的裸 device assertion 不满足本轮可恢复错误合同。

目前仍是 SCOPED；没有设备错误机制或 checked CUDA 索引的 PASS。
