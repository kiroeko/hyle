# Compute v4：异步 runtime、storage bridge 与 provider 扩展合同

## Closure card

- component / owner：`compute` / Compute；round=`compute-v4-runtime-extensibility`；
- baseline commit：`98295cc24d28f35437b51b0ef1fdf1ee76872d5a`；上一轮已关闭的 storage
  合同与证据保存在 [`compute-v3-block-storage-format`](compute-v3-block-storage-format.md)，本轮不回写
  该历史记录；
- trigger 与证据：当前公开 `Completion` 对任意有效对象都立即 ready；`TensorView` 已能绑定 provider-neutral
  `mem::BufferView`，但 Compute 没有 backend resource factory、queue copy/sync 或 resource/device compatibility
  合同；`ProviderKind` 是封闭枚举且 `ProviderList` 固定最多四项。与此同时，已接受的
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

## AUDITED 结论（2026-09-12）

### 对账结论

- ggml `e91ded11bdcd` 把 backend/device、buffer type/buffer、stream/event、async tensor copy 与 graph plan
  分开；CUDA 13.3 的 stream/event 和 Vulkan timeline 都证明“有效 completion 恒 ready”不能承载真实提交；
- oneDNN `6b7f1fbbb18e`、vLLM `51da0ca66c80`、SGLang `dcebe8c4733a` 与 ds4
  `f62ca29a3087` 均把可共享编译 artifact 与 per-stream/per-request mutable state 分开；
- Mem 已提供开放 `MemoryResourceRef`、opaque `Buffer/View`、host mapping、coherence 与 domain identity。
  因而本轮**不再发明 Compute Storage 类**；缺口是 backend 暴露资源并让 Queue/Provider 正确解释它。

### 有限入口与 owner ledger

| ID | 当前形状 | 裁决 | 冻结合同 |
|---|---|---|---|
| R40 | vendor 名称组成的封闭 `BackendKind` | `REPLACE` | 开放 `BackendId` + 通用 `DeviceClass`；CUDA/Metal 等 identity 由 typed backend factory 提供 |
| R41 | `Device::cpu()` 与只返回一项的全局枚举 | `MOVE` | CPU factory 进入 `compute_backend_cpu`；每个显式 `Backend` 枚举自己的 Device，不使用全局静态注册 |
| R42 | Device 只有 kind/ordinal | `REPLACE` | 小型 copyable type-erased Device，保存静态 backend vtable + inline identity payload；复制、查询不分配 |
| R43 | TensorDesc/StorageFormat 混在 runtime target | `MOVE` | schema 进入 `compute_ops`；头路径仍保持 `hyle/compute/*`，用户不接触内部 factor 名 |
| R44 | TensorView 绑定 `mem::BufferView` | `KEEP/FIX` | 保留四字 borrowed resource view；prepare/submit 校验 resource domain、device、range、alignment；不复制 payload |
| R45 | `ProviderList<4>` | `REPLACE` | `ContextOptions` 借用任意长度 `span<const Provider>`；create 冷路径一次性复制，steady path 无 allocation/search |
| R46 | `ProviderKind`/`kMixed` | `REPLACE` | 开放 `ProviderId` 与 name/version；单 artifact 返回精确 identity，Graph report 自己枚举多个 provider |
| R47 | Context 内固定四槽 | `REPLACE` | Context state 与精确 provider slots 一次分配；创建失败逆序回滚，不依赖 Container |
| R48 | inline terminal-only Completion | `REPLACE` | move-only immediate/async type erasure；`poll()` 返回 pending/success/error，`wait()` 返回终态，`request_cancel()` best effort |
| R49 | Context/Queue 共用一个 bool state | `FIX` | 独立 ref-counted backend context/queue state；atomic admission；`close()` 不再接收，`drain()` 显式等待 |
| R50 | provider `execute()` 同步返回 Status | `REPLACE` | prepared submit 接收具体 Queue/bindings并返回 Completion；同一 ordered queue 不逐 node host-wait |
| R51 | `submit_host()` 位于通用 runtime | `MOVE` | caller-thread typed escape hatch 进入 CPU backend；仍不保留 callable、不分配、不抛异常 |
| R52 | 公共通用 `Module/Kernel/CommandList` 长期占位 | `DEFER/REFINE` | 共用 lifecycle 只到 Queue/Completion；CUDA/Metal/Vulkan/HIP 分别提供 typed module/kernel/interop extension，首个 backend 证明共同形状后才上提 |
| R53 | callback/coroutine bridge | `DEFER` | 核心只 poll/wait/cancel；需要时登记 `compute_sched`，避免 Compute 强制依赖 Sched |
| R54 | 一次性 prepare+submit convenience | `DEFER` | prepared 是正常热路；真实一次性 consumer 出现后以显式 cold-work 名称补充，不污染本轮 |

`Provider` 本身继续是显式、copyable、inline 配置值；其 vtable 只在 context/prepare 冷路径调用。可选 Graph
region 能力通过对象局部的具名 extension query 暴露，不使用 process-global registry，也不让 Compute
依赖 Graph。普通 built-in `Operation` 不因 provider 扩展支付字符串查找。

### 生命周期与错误矩阵

- Context owner 析构等价于 close + release，**不 drain**；Queue 析构 close + release，**不隐藏等待**。
  Queue/Prepared artifact/Completion 各自保活所需 backend state，迟到完成后才允许最终释放；
- `Queue::drain()` 只等待该 queue 先前接受的工作；`Context::drain()` 等待其所有 backend work。close 与
  drain 可分开调用；设备不支持抢占时 cancel 返回 `Unimplemented`，但 borrow 仍活到真实终态；
- submit 前错误（invalid binding、wrong device/domain、closed queue、OOM）直接返回 unexpected Status；
  已接受工作的终态错误只从 Completion 观察。device loss 必须成为 terminal error，不能永久 pending；
- CPU immediate completion 完整 inline，不分配、不加锁、不做 atomic；异步 completion 允许使用 backend
  event/pool，但常态 HYLE wrapper 不额外分配，具体成本由 backend 卡签发。

### 冻结实现边界

`compute` component 作为历史稳定 identity 承担概念上的 `compute_core`，不再另造同义 component。
CPU 执行轴与 HYLE native kernels 分别由已登记的 `compute_backend_cpu`、`compute_provider_native` 承担。
本轮 Runtime FIX 为 R40--R51；R52--R54 明确不进入本候选。P0 账单目标是：provider/device 枚举 O(N)
冷路径；Context create O(P) 且一次 metadata allocation；prepare O(P)；prepared submit O(bindings) + 一次
预绑定 dispatch；poll/wait O(1)，不含设备实际等待时间。

## Implementation checkpoint（2026-09-12）

- R40--R51 的第一份产品实现已落地：开放 `BackendId` / `ProviderId`、type-erased Backend/Device、任意长度且
  在 Context create 时复制的显式 provider sequence、独立 Context/Queue state、move-only immediate/async
  Completion，以及返回 Completion 的 prepared submit 均已替换旧封闭 runtime；
- Context 的 provider metadata 保持一次 allocation；provider context 创建失败按已成功项逆序回滚。测试用
  六个 provider 明确越过旧四槽上限，并销毁原始配置后验证 Context 仍只引用自己的副本；
- Completion 的 immediate success/error 与模拟 async pending/poll/wait/cancel/move/destroy 已覆盖。共享因子
  测试发现其内部 factory 原先没有跨 DLL export，现由 `CompletionAccess::make` 的 out-of-line Compute symbol
  修复；provider create hook 也改为接收 Context 拥有的 Provider 副本，不能保留调用方临时配置；`prepare()`
  取得 Context admission token 后才进入 provider 冷路径，不再以非原子的 `is_open()` 检查和 close 竞争；
- Windows 上 15 项 Compute/Graph 定向测试，以及独立 source、relocated static、relocated shared Compute
  consumer 均通过；旧 `ProviderKind`、`ProviderList<4>`、`Device::cpu()`、通用 `submit_host()` 和
  `host_kernel.h` 产品入口已无代码消费者；
- 已交付的单输出 transform 与 elementwise family 提供只做 cold output inference 的 typed `prepare()`；Fill
  另提供自然的零输入 submit，Select 另提供自然的三输入 submit。这不是 R54 所排除的 prepare+submit 一次性隐藏路径：artifact、
  invocation、workspace 与 completion owner 仍全部显式；
- Graph G35 证明仅靠 operation schema 的 alias 上界不足以规划具体 artifact：例如 Binary 语义允许 alias，
  Native 广播路线却不能安全原地执行。R50 的 prepared provider 合同因此补充冷路径
  `alias_support(output,input)` refinement，返回当前 shape/layout 的支持与 view byte offset。Compute submit
  也在接收工作前复核该结果；普通不重叠绑定不新增 vtable 调用，Graph compile 后的 steady route 不搜索。

这不是 terminal record，状态保持 `AUDITED`。真正异步 backend 的 late-completion 保活、close/drain 与
device-loss/error injection，sanitizers、Linux/两台 Apple、shared export hygiene、P0 codegen 账单，以及依赖
本 runtime 的全部 v1 provider/Graph 仍待完成；不能用 CPU immediate 模拟测试替代异构生命周期资格。

## Object-local provider extension checkpoint（2026-09-13）

- Compute runtime 增加具名、精确版本的 provider extension identity，以及只供 factor 间使用的
  `compute/detail/provider_extension.h` bridge。查询发生在 Context/Graph compile 冷路径；普通
  `prepare()`/`submit()` 不查 extension，也没有进程全局 registry；
- extension handle 保活创建它的精确 Context/provider state，但所有 call/submit 仍经过 Context/Queue
  admission。submit 同时验证 Queue 属于同一 Context、全部 Tensor storage 属于同一 backend/device；关闭
  Context 后 retention 不会错误授予新工作；
- 普通用户只看到 `Context::provider_count()` 与 `provider_identity()` 诊断。原始 extension query/call 没有
  暴露成半截公共 API；Graph region、后续 CUDA/Metal/Vulkan typed factor 在内部 bridge 上建立自己的类型
  安全 façade；
- Windows shared 构建中的 provider extension owner-retention、错误版本、跨 Context Queue、close admission
  与 Graph region 调用测试通过。它解除 Graph G37 的 runtime blocker，但真实异步 backend 的 late
  completion 资格仍属于 R48--R50 终局验证，Compute 继续保持 `AUDITED`。

## Ordered sequence 与 late-lifetime checkpoint（2026-09-13）

- `Queue` 现有 admission 上增加仅覆盖 host enqueue 的单-producer sequence guard。direct operation 仍每次
  自动打开一个 sequence；Graph 可在同一 guard 中连续调用已 admission 的 operation/region bridge，从而
  不插入逐 step host wait，也不允许另一个 producer 把命令插到 plan 中间；
- provider submit 的冻结语义已写入单-op 与 region SPI：成功 Completion 代表截至本命令的完整 ordered
  Queue prefix，同步 `Result` 错误表示本命令没有接纳工作。Graph 因此只保留最终 prefix Completion；若
  后段同步拒绝，则在已接纳前缀的 Completion 上追加错误而不丢弃 work；
- Completion 可附着一个位于被保活 owner 内的内部 lifetime record。正常 poll/wait 观察终态时先销毁
  provider completion state，再调用 terminal/release；用户直接丢弃 pending handle 时则把 record 交给
  backend，由 backend 在真实终态后释放。public Completion 只新增一个 attachment pointer，不为 direct
  CPU 路线增加 heap allocation、lock 或三字 callback 常态字段；
- direct `OperationInvocation` 也以同一附件保活 provider invocation，并以 `in_flight()`/重叠 submit 拒绝
  防止一个 mutable lane 被并行复用；Graph 的内部 admitted submit 则由最外层 Graph Invocation 一次性保活，
  不为每个 step 重复加引用。inline-terminal CPU completion 在内部可直接识别，因此不做 execution-owner
  引用交接，不为同步热路径增加一对原子 refcount；
- Windows 模拟异步 region 已验证无逐 step wait、pending handle 遗弃、Invocation handle 先销毁、terminal
  error/cancel、同步 tail rejection 和 exactly-once release；模拟 direct async operation 也验证外层 handle
  先销毁和重叠 submit。Compute 选择产品的 Windows source、installed/relocated shared 与 static consumer
  均通过。真实 CUDA/Metal/Vulkan backend 仍须在各自卡证明 event 销毁与 late callback 实现符合该合同；
  这份模拟证据不替代真设备资格。

本 checkpoint 完成 R48--R50 的通用组合机制，但 Compute 仍为 `AUDITED`：真实异步 backend、Linux/两台
Apple、sanitizer、P0 codegen 与异构 provider 交付回放尚未完成。
