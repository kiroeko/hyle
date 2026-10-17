# 原生后端与 CPU/GPU 组合

共同语义和冷路由见[Compute](compute-design.md)，Graph的计划职责见[Graph设计](graph-design.md)。
本页解释为什么保留原生差异。实际接口与设备限制见[用户Compute指南](../user/compute.md)。

## 后端、模式和 Provider

CPU、CUDA、Apple MLX/Metal、Vulkan是构建能力；CPU、GPU、CPU＋一个GPU是应用执行模式。
SDK含多种后端，不表示单次运行自动跨设备。Provider是后端内的实现来源，不是普通用户选择项。
同一家库的CPU和GPU实现需要不同资源接入，不能因名称相同就视为一个互换executor。

共同层复用错误、端口、资源与选择语义，不强制统一native Queue、array、buffer、Program或完成对象。
配置与交付由[构建架构](build-system.md)说明；初始化入口见[环境准备](../developer/setup/README.md)。

## CPU

Platform给出OS/ISA/NUMA事实，SIMD拥有自己的kernel分派；Compute不重复探测。
CPU执行需要持久executor、packing和每槽scratch；小任务或嵌套调用可使用合法的inline路径，
不能每个op建线程或用逻辑核心数推导所有shape的最优线程数。

oneDNN、Native、ggml/ATen等实现按实际端口和资源绑定。第三方自有线程团队必须如实表达，
不能将进程全局开关伪装成对象局部配置。复用现有Sched executor组合，避免第二个调度runtime。
部分任务接纳后排空其前缀，嵌套等待不能因占满同一线程池而自锁。

## CUDA

cuBLAS/cuBLASLt、原生kernel和ggml等完整region互补。准备时完成适用的handle、stream、tactic、
packing、代码和scratch绑定；执行时用固定调用或合法capture replay，不逐token重新搜索算法。

stream/handle/workspace的绑定顺序和并发使用是实际合同。capture需要稳定地址、合法操作与
launch topology；“尺寸不变”不是充分条件。ggml自有stream、pool和graph状态必须通过真实桥接组合，
不能把私有stream强转成Hyle Queue。

记录accepted prefix及event/stream终态；enqueue成功不代表计算成功。事件记录失败、晚到错误和
取消都不能提前释放仍被设备使用的allocation或代码。CUDA Toolkit是构建SDK，driver/device是运行环境。

## Apple MLX/Metal

MLX GPU后端本来使用Metal，优先在同一runtime内复用成熟compiled region、Primitive或合法kernel扩展，
避免每个op跨runtime。必要时可使用独立Metal或ggml-Metal region，但须承担真实的buffer、格式、
时间线、完成及边界成本。统一物理内存不代表任意array/buffer可零复制互换。

MLX生成array由实际owner保活，不为统一out-buffer形状强制影子复制。cache hit仍可能构造执行
arrays/metadata，应按实际managed成本描述；不能从compile名称推断无分配重放。
提交线程、stream、metallib与错误观测具有原生约束，host可见也不等于GPU工作成功完成。

Metal pipeline的static threadgroup memory上限与全局GPU buffer容量是不同限制。
编译或验证选项可能改变某个pipeline的实际用量，应查询创建后的pipeline再决定合法tile/路径，
不能把一次超限推导成统一buffer上限，也不能假定所有kernel的用量按相同比例变化。
合法的小tile备用路径仍需再次核对；仍超限则正常返回错误并退休工作，不改变dtype、维度或转到CPU。

Metal feature、SDK、部署版本、dtype和设备限制共同决定可用路径。GPU矩阵加速与独立Neural Engine
是不同硬件能力；某代芯片的结果不代表另一代。公开Python互操作也不自动成为独立C++ interop承诺。

## Vulkan

Vulkan提供跨厂商计算路径；macOS的portability部署与Apple原生路径仍有不同能力和成本。
准备时固定device/queue family、memory type、pipeline/SPIR-V、descriptor与每槽command资源，
按实际feature/limits选择specialized程序，而非强制所有设备使用最弱通用shader。

execution顺序与memory availability/visibility分别成立。transfer/compute、跨queue/family及
non-coherent host memory需要真实的stage/access、ownership和flush/invalidate处理。
recorded push constants不会随host变量自动变化，pending command/pool/buffer不得reset或释放。

ggml-Vulkan可能每次编码并分批提交，IREE与原生程序也有各自runtime状态；它们不因都返回Program
就具有相同replay成本。同一VkDevice或同一GPU仍不自动保证跨Provider对象可以共享。
取消、OOM、提交失败和device lost分别处理；缺feature明确拒绝，不虚构同成本fallback。

## CPU与GPU组合

先区分存放与计算：host保存权重、分批送GPU，不等于CPU承担模型数学；只有CPU与GPU各执行片段，
才需要两端计算实现。引擎提供允许的放置、expert IDs/范围、驻留与预算，Hyle提供合法片段、搬运、依赖和完成。

组合编译复用同一typed DAG，将粗粒度片段交给原生compiler。它拥有node coverage、canonical边界、
依赖、liveness和whole-plan预算，不建立万能Queue、第二套调度器或逐kernel抽象。
CPU＋CUDA、CPU＋Apple、CPU＋Vulkan保留各自native endpoint和绑定类型。

每条跨片段边保留完整descriptor、物理编码/layout、owner、ready/完成及转换方式。
同格式才可byte-copy；不同时使用真实转换或拒绝。host staging、repack、上传下载、等待与融合损失
进入整体成本。pinned/pageable、MLX owner、Vulkan queue family与semaphore条件不能被统一接口抹掉。

固定容量staging/device槽在上次真实终态后复用；流式权重作为受绑定输入，不在warmup偷偷完整驻留。
独立搬运可与计算重叠，但返回pending或换线程阻塞不等于设备copy/compute overlap。
硬件或Provider需要host等待时保留并报告它，不为“全异步”标签牺牲正确性。

动态expert IDs改变的是数据选择，不重新搜索Provider。引擎拥有预取/命中/淘汰政策，
Hyle拥有有界迁移、在途owner、绑定更新和native依赖；不复制模型cache。

编译失败释放partial artifact；运行失败停止依赖后继并保留原错误，不到CPU重算。
取消停止未接纳后继，已接纳native工作仍需退休。组合owner结束使用时只退休自身工作，
stop/close/drain按各原生API合同区分，不关闭应用借用的全部Context；旧结果与外部backing保活到其最后使用。
