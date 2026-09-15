# Compute CUDA backend v3：同步存储操作的实际完成边界

## Closure card / SCOPED

- owner：`compute_backend_cuda`；baseline `85f8bbb2`；前轮证书为
  [CUDA v2](compute-backend-cuda-v2-endpoint-identity.md)，产品候选 `db164bb7`。
- trigger：公开 Q4/cuBLASLt consumer 在首次运行得到未更新输出，见
  [原始失败日志](../../../../evidence/correctness/compute-cuda-resource-sync-v3/original-q4-preflight.log)。
  更重要的独立源码证据是 `CudaResourceBase::write/fill/copy` 直接返回 CUDA native 调用，与 Mem 明确的
  synchronous range-operation 合同不等价。原样例复跑与显式 upload-sync control 均通过，故不把偶发
  数值失败单独当作已经证明全部因果关系；补受控 native-stream 顺序反例后实施。
- 具名来源：[CUDA Driver 同步行为](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html)：
  pageable H2D 可在 DMA 完成前返回，D2D 不保证 host 完成，device memset 也可能异步。
- 主纵切：加载/更新设备权重后立即从 non-blocking Queue 执行 native/vendor kernel；正交纵切为不涉及
  MatMul 的 Buffer fill/copy 与两个独立 allocation。同步方法没有额外 Completion 可供用户等待。
- scope：只修正 CUDA MemoryResource 的同步 range 操作，不重开 Mem、Compute/Graph 对象模型、
  Queue、事件池、算子或公共构建。异步 copy/fill/submit 保持原实现与 overlap 能力。
- non-goals：全设备/全 Context 等待、额外全局缓存/线程/stream owner、供应商算法调参、P1。
- 预算：一个实现方案、一轮产品纠正、一次候选后红队。默认流配对若无法满足同步合同或引入跨 Context
  风险，先重新审计，不把等待搬进 Graph 或每次 warm kernel。

## 能力与有限 FIX

| 能力族 | 层级 / 当前状态 | 动作与边界 |
|---|---|---|
| CUDA 同步 write/fill/copy | 原子存储机制，PARTIAL | FIX：提交后等待实际使用的默认流完成；显式保留 native error |
| CUDA read | 原子存储机制，COMPLETE | KEEP：DtoH 本身保证 host 返回前复制完成 |
| pinned/managed host read/write/fill | 原子存储机制，COMPLETE | KEEP：现有 memmove/memset，不新增 CUDA 等待 |
| Queue async copy/fill/native submit | 异步执行机制，COMPLETE | KEEP：Completion/跨 stream 依赖不改，不插 host/global barrier |
| 其余 v2 公开入口、C++ 交付与依赖 | 既有合同 | KEEP，不以本缺陷扩张能力或兼容旧错误行为 |

| ID | 允许改动 | 验收 |
|---|---|---|
| CU30 | 私有同步 helper：同一 ScopedPrimaryContext/TU 下 native operation 后 `cuStreamSynchronize(nullptr)`，匹配 driver 的默认流模式；提交失败也保留原错误并观察流终态 | H2D/fill/D2D/host-resource copy 返回后可被独立 Queue 使用；不创建常态 metadata，不使用 device/context-wide synchronize |
| CU31 | CUDA factor 受控顺序反例、同步操作组合与 Q4 原始样例回放；指南区分同步初始化和异步热传输 | 错误候选反例失败；新候选通过，不保留 sample workaround 冒充产品修复 |

## API、七问与成本

公开接口、ABI/layout、组件依赖、Toolkit 13.3 与许可证不变，FFI 仍 N/A。`BufferView::write/fill/copy_from`
无需用户再查询 CUDA 默认流；调用方仍负责先完成与这段数据冲突的异步工作。默认流方法可能等待该流此前
工作，但不能因此推导任意 non-blocking Queue 都已完成。需要 overlap 的正常热路径继续用 `copy_async` /
`fill_async` 与显式 Completion，而不是同步 convenience 方法。

复用 Mem 的同步接口、Error Status/native-error 与 CUDA 已有 ScopedPrimaryContext；不私造第二套错误、
所有权或调度机制。source/static/shared 均消费同一 CUDA factor 实现，用户安装 Toolkit，构建不代为下载。
形状来自存储生命周期与 CUDA 标准，不来自 Q4 benchmark。所有旧范围保持，只有上述原子机制修复。

P0：复制/填充仍 O(bytes)，新增的是同步合同要求的 O(1) driver wait 调用，无新增 allocation、lock、
字段或数据搬运。DtoH 与直接 host 方法无新增调用；CUDA native `copy` 的同步分支明确等实际默认流。
不得把未等待完成的旧实现当作等价性能 peer，也不得把本轮判为纳秒级 WIN；异步 Queue 路线不变。

## 预声明验证

- 原始失败样例只构建本身，不重新填算子；控制变量保持 Q4 字节、cuBLASLt algorithm、数值容差不变。
- CUDA factor：默认流 gate 下 fill/copy 不能提前返回；同步 write/fill/device-copy/host-copy 后通过独立
  non-blocking Queue 读数据；现有 runtime/late-failure/逐头与并发生命周期回放。
- Windows 与 Linux WSL 真 CUDA；ASan/UBSan 沿用 v2 已单独对照的 WSL driver 限制并逐项重跑相关测试。
  不引入新共享可变状态，TSan 不因增加 driver wait 扩张；这不是 GPU race sanitizer 资格。
- source、installed static、installed shared 的 Q4 carrier；原有跨 lane/capture 示例及 Graph CUDA 相关
  测试验证 async 路线未被串行化。公共 CMake/API 未变，不跑全库 package matrix 或 CPU P1。
- Apple 没有 CUDA 且本轮不改探测/组件图，为 N/A；docs/closure/previous-ref/diff 均需通过。
- 当前所有新增验证均为 UNPROVEN；scope/architecture/correctness/delivery/usability_docs/red_team/
  engineering_performance 七个最终 gate 待候选证据裁决，不提前写 PASS。

候选与终局结果在此续写；Q4 Provider/residence、Apple 与整体 foundation 尚未完成，不计入本证书。
