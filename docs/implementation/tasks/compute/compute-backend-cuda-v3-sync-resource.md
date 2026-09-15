# Compute CUDA backend v3：同步存储操作的实际完成边界

## Closure card / CLOSED

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

P0：复制/填充仍 O(bytes)，新增的是同步合同要求的常数次 driver wait 调用，无新增 HYLE allocation、lock、
字段或数据搬运；等待耗时还取决于流的在途前缀，并不承诺 O(1) 时间或 driver 内部无锁/分配。
DtoH 与直接 host 方法无新增调用；CUDA native `copy` 的同步分支明确等实际默认流。
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

## 实现前反证

`cd00f76d` 的 CUDA runtime 在未改产品的 Windows RTX 4070 上实际 13/14 通过。新增 default-stream
host gate 保持未放行时，device fill 与 device copy 都提前返回；两个断言均失败。见
[受控原始日志](../../../../evidence/correctness/compute-cuda-resource-sync-v3/original-stream-gate.log)。
这个反例不依赖 cuBLASLt、Q4 数值或亚纳秒计时：未满足执行前置条件，同步操作就已经返回成功。
100ms 是给 host worker 执行的观察窗口，不是产品性能门槛；放行并 join 后仍验证完整数据与清理。

因此 CU30/CU31 已固定，进入 AUDITED。选用与当前 native 操作同 TU 的默认流完成等待，避免引入新
stream owner/资源锁/全设备 barrier；原 Context/Buffer 生命周期与异步 Queue 合同保持。完成 helper
后先重放这一旧候选失败用例，再运行本卡列出的精确验证，不靠多跑随机 Q4 样例掩盖缺陷。

## 候选与 Windows 定向验证

`be2d8b059ca858d2e9b3393b38d2530d4eb3dbe3` 完成 CU30/CU31 产品与测试：一个私有 helper、四个 native
range-operation 调用点；read、host 直接方法、Queue/Completion 与对象布局未变。Windows Release CUDA
runtime 15/15、late failure 1/1、逐头、Graph executable 284/284 通过。
[原始日志](../../../../evidence/correctness/compute-cuda-resource-sync-v3/windows-runtime-graph.log)同时记录
旧失败 gate 修复，以及 write/fill/D2D/普通 host resource copy 后立即从独立 Queue 运行 kernel 的四种组合。
编译最多两 job、CTest 串行/120s；无 P1 或无关模块测试。

进入 CANDIDATE。Linux、sanitizer、三交付/Q4 与候选后复核尚未完成，不能将 Windows 结果扩张为最终
CLOSED 或整体 foundation 资格。

## 终局复核与裁决（2026-09-15）

同一产品候选 `be2d8b05` 完成所有本轮适用门。除上述 Windows 结果外，Linux GCC 14/CUDA 13.3 的
runtime 15/15、late failure、逐头均通过；ASan 与 UBSan 各实际回放 15/15。ASan 使用 v2 已由无 HYLE
control 证明的 `protect_shadow_gap=0` 和唯一 `/usr/lib/wsl/lib/libcuda.so.1` LSan suppression；仍恰好
两笔/280 字节 WSL shim 退出泄漏，没有关闭 leak detection 或忽略 HYLE 内存。

Q4 原始默认路线（不加 upload-sync control）在 source、installed static、installed shared 均通过：
M=1/7/33、N=65、K=96、各三次变化输入，native 与真实 cuBLASLt 输出正确。前两形状 vendor workspace
为 0，M=33 为 98,736 bytes；Q4 权重 3,510 bytes，显式冷 F32 权重 24,960 bytes，额外占用没有藏掉。
原有双 lane commands/capture consumer 在三种交付也全部通过。原始日志、缓存和复现命令集中在
[证据索引](../../../../evidence/correctness/compute-cuda-resource-sync-v3/README.md)。

候选后红队由主 agent 在实现后的独立一遍源码复核完成，不冒称另一 agent：四个调用点均在原
ScopedPrimaryContext 内等待；helper 保留首个 native error 并观察完成状态；generic host source 在完成
等待后才 unmap；空范围、DtoH 和直接 host 操作没有新增等待。CUDA 13.3 header 的 PTDS/PTSZ 映射使
native 操作和 `cuStreamSynchronize(nullptr)` 匹配同一默认流模式。无新可变状态、字段或 owner；
Queue、事件池、异步 copy/fill/capture 未改，无 Graph workaround。未发现剩余本轮产品反例。

| Gate | 证据与边界 | Verdict |
|---|---|---|
| scope | 仅 CU30/CU31，原始 counterexample 与标准一致 | PASS |
| architecture | Mem 同步合同由 CUDA adapter 实现；不转嫁到 Graph/用户/异步 Queue | PASS |
| correctness | Windows/Linux 真 CUDA、受控 gate、四种存储组合、既有错误/生命周期，ASan/UBSan 如上 | PASS |
| delivery | 三种交付的 Q4 与双 lane/capture 实际运行；SDK/consumer DLL 身份核对 | PASS |
| usability_docs | 公开形状未改；Compute guide 明确同步初始化与 async overlap 的使用边界 | PASS |
| red_team | 候选后的错误、默认流、unmap、并发和生命周期复核 | PASS |
| engineering_performance | P0：常数个 native wait，无新增 HYLE metadata；不把错误提前返回当同合同速度 | PASS |

Apple/CUDA 为 N/A；没有新共享可变状态，不新增 TSan。未运行 P1、GPU race sanitizer 或签发厂商
私有 allocation 上界，不把它们写成通过。本轮仅关闭同步 adapter 修复，不关闭 Q4 Provider/Apple 或
整个 Compute/Graph foundation。新的有效同步操作提前返回、默认流/context 配对错误或真实 async
consumer 回归才触发本 owner 重开。

同候选统一 LOC/self-check：全库 **687,918**，自实现不含测试 **126,320**；CUDA factor 实现 **2,034**
（含 split public header）。Compute **17,000**，Graph **14,135**，合计 **31,135**；样例不计产品实现。
