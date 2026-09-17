# C1-L：CUDA checked operation 的异步算法错误

## Closure card（AUDITED）

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

## 审计结论

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

安装 SDK 的 `cuda_runtime_api.h` 同时确认：event wait 绑定调用时的 event record；GraphExec 的
多次 launch 由原生 runtime 排序。两者都不替 HYLE 保留独立的软件错误字或传播其含义。现有 Graph
按 segment 捕获和提交，最后已有聚合 Completion；本轮不改 Graph/通用完成 vtable。

### 有限 FIX（实施前固定）

`KEEP` 原生 stream/event/Completion/reaper、shared provider、capture segment 和输入输出 owner
合同；`BORROW` CUDA 原生 event 顺序、device/pinned allocation、async memset/copy；`OWN` 私有
软件错误字与 first-error 合并。设备错误是本轮新 consumer 的原子基础机制，不是新的通用执行器。

| FIX | 固定范围 | 语义 / 成本 / 必需验证 |
|---|---|---|
| L01 | CUDA prefix 的私有设备错误状态和 provider bridge | 非空 checked 工作取得当前已接纳 sequence 的状态字；首次使用在原 stream reset，后续 checked 操作复用且不清掉先前错误。状态字/固定页 host 镜像随 native completion pool 冷分配，受 allocator/OOM/close 合同约束；无 tensor payload 回传或新的普通 C++ API。原子错误码限 checked index 越界和 I32 prefix overflow，最终为 OutOfRange；未知内部码明确 Internal |
| L02 | 真实完成、跨 Queue 错误传播和保活 | 仅使用状态的 prefix 在 event 前排一次常数大小 D2H，event 完成后 poll/wait/abandon 才解释；CUDA runtime failure 与已知 terminal hint 不被清除。device wait 后将源软件错误合入目的 prefix；源状态必须持有至目的 GPU 读取结束，观察/丢弃源 Completion、并发新提交及池重用都不能覆盖它。已接纳后任一步失败仍返真实 prefix Completion；释放先于资源销毁，不在 CUDA host callback 中销毁 CUDA 对象 |
| L03 | 原生 capture 的独立状态与本地构建 | 每个使用 checked 工作的 captured artifact 有冷期固定的状态地址与 reset 命令；录制不能借未来任意 EventSlot 的地址。launch 后把其状态合入实际 prefix，允许多个 artifact、普通工作和跨 Queue 依赖组合；artifact/state 保活到其 GPU 读取结束，独立运行槽互不污染。用 CUDA owner 内一个微小原生 merge kernel 保留 first error，冷绑定、不逐次查询/编译。backend 自己启用 CUDA TU，沿现有 arch/CRT/sanitizer/可选路线规则，不改公共构建系统 |
| L04 | 有界真实设备验证、已有消费者回归与文档 | 真实 GPU 判定的 valid/invalid index 与 overflow 状态、pending→terminal、fresh invocation 恢复、两槽变更数据、capture abort/replay、多个 captured artifact 与 checked/pure 混合 prefix；跨 Queue 延迟源/目标、提前观察/丢弃源、close/abandon/reaper、OOM 回滚、稳定重复执行不分配。旧纯 CUDA 工作及默认 Direct/Graph 回归；完整 C1 实际算子/三交付/P1 未运行前保持未证明 |

### 固定生命周期与 P0

- 软件状态属于一次实际提交 prefix，不能属于可覆盖的全局 last-error。完成池的源记录在被依赖
  时不能提前回收；捕获状态由 artifact 拥有，运行记录保留其引用。依赖/捕获的引用元数据按实际
  在途记录有界池化，预留/回滚在冷期或接纳前完成；不可用时明确拒绝，不在成功热重放中扩池。
- 同一含软件状态的 captured artifact 若尚有本轮 GPU 读取未结束，禁止不受控地并发重用其状态；
  普通 Graph 已由独立 Invocation 保证这一点。两个 Invocation 各自捕获，不增加全局排他。
- 未使用该机制的 prefix 不发 reset/readback/merge 命令。使用时，每 prefix 一次 reset 和一次
  4字节 D2H；每个含状态的 captured artifact 另有录入图中的一次 reset，每个含状态的 imported
  dependency/captured artifact 至多一次原生 merge。记录
  实际 native cold metadata、device/pinned bytes 和 driver 存储，不把物理分配粒度写成4字节。
- first-error 合并不承诺整个 Graph 的事务回滚。后续算子卡必须先验证自身 checked 数据再写
  自身结果；最终错误的图结果不得当作有效结果消费，先前已接纳工作仍必须完成/保活。
- 不将新状态字塞入所有 Provider 的 Prepared 布局，不改 CPU/Apple/Vulkan、通用 Completion
  回调或 Sched/Mem。kernel 与状态/依赖的实现细节可在此有限集合内定向验证；如需跨越这些 owner
  或改变上述成本/完成合同，退回此卡重新定界。

本次只固定 L01–L04；没有设备错误机制或 checked CUDA 索引的 PASS。原生状态的测试 kernel
只验证此基础机制，不能代替后续实际 Gather/Scatter/Scan 或整条路线验收。
