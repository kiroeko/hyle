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

已准备的ggml-Metal region可在原命令缓冲尾部发布SharedEvent，MLX消费者显式编码GPU等待。
MLX event的stream标记不是ggml的原生Queue身份，不能据此省略等待。原输入array、输出和Source
保活到原生终态核对；事件就绪不代替错误检查或公共Completion的权限退休。关闭公共接纳后，
原owner仍排空已接纳工作的完成通知。整张Graph的Source交接与内部region事件桥接分别资格检查。
MLX→ggml也可在原命令缓冲等待输入事件；冷期根据实际region能力准备这条路径。
算子的设备检查结果由原Invocation保活，退休时按region顺序核对，保留首个错误并阻止失败输出发布。
后续原生算子需在CPU读取索引时，仍先完成前缀检查与退休；所选MLX region通过原生事件交接GPU依赖。
冷期为有界原生region准备单命令缓冲，实际具备完整事件能力的计划复用最后一个原生region的完成通知。
通知仅将原Invocation的退休Job交回原owner；完整MLX/原生前缀、输入Source、错误与权限仍按原合同核对。
整段计划可以交接待完成array，不增加终态GPU命令缓冲。
同一Metal端点可以包含ggml与MLX两条原生队列；Graph内部region的纯控制依赖也须显式导入，
不能凭端点相同省略等待。依赖容量在冷期按实际Plan预留，执行时保留原生产者的提交代际，
等待其原生/MLX事件，并在原owner退休时核对完整失败结果。数学与复制Job复用这套原生依赖交接；
CPU必须读取的索引或复制仍先核对生产者，空复制也不能丢掉尚未完成的控制依赖。

GPU复制在原MLX stream提交blit及本次代际的事件，完成通知只把原复制记录的退休Job交回原owner。
MLX的原buffer使用untracked资源；同一stream的blit pass通过冷期创建的原生fence交接访存，
不能只依靠主机FIFO提交顺序。原Runtime保活一个fence，由原owner编码等待与更新，见
[Metal blit访存合同](https://developer.apple.com/documentation/metal/mtlblitcommandencoder)。
显式复制依赖向ggml和MLX计算导入实际事件；原生blit不在MLX的资源fence表中，同stream也不省略这条依赖。
需要CPU读取的结果及尚未稳定的原生输出地址仍须核对实际前缀。前缀观察不结算公共Completion，不发布lease，
也不允许提前复用；公开复制结果还核对原依赖闭包及生产者的完整结果。多阶段复制的依赖容量
按声明的phase/resource上限冷期预留，复用原权限与退休。退休核对也等待原命令缓冲的错误处理
回调完成，遵循[Metal完成合同](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted%28%29)。

设备长度控制的分页追加复用原CopyRecording、fence和完成通知；原PagedAppendPages保留COW写权限、
继承前缀和完整Graph的成功发布合同。追加与注意力可有不同的物理query容量，但须共享原KV长度值及
同一schema的分页容量；值相同的另一个Graph输出不能代替原控制身份。读取程序在GPU检查原追加的
状态字后才访问页面，失败仍由原依赖和完整Graph退休结算。已编码的MLX输出可在原提交代际被固定后
交接稳定地址，同时携带其依赖闭包中的原GPU错误字。错误array保活到消费者实际退休；GPU追加先检查
全部错误字再访问输入和页面，不以生产者事件就绪代替成功。错误字目录在冷期有界预留，Tier2设备按
[Metal间接资源合同](https://developer.apple.com/documentation/metal/tracking-the-resource-residency-of-argument-buffers)
声明原buffer驻留，原生与MLX访存事件分别导入。普通ready-array导出仍要求成功完成。
该原生交接不等于完整MTP设备闭环；引擎的后端设备长度能力须由完整消费者单独验收。

整张原生Graph的Source可由具备实际终态端点的原提交导出；冷期逐一核对消费者是否能导入生产者的
所有终态端点，并为原执行槽预留依赖容量。接纳期间借用原RetainedCompletion，防止并发drain消费
仍在交给后端的句柄；后端在回调返回前保留原Job和提交代际。每段消费者按自己的原生队列导入，
不能凭公共端点相同跳过后段。消费者真正退休后仍核对原生产者的完整结果，失败不得发布输出。
这条路径保留原输出writer与权限；显式create_handoff仍是先等待完整生产者的HostWait合同。
等待按真实队列及编码器来源区分：MLX导入ggml或原生复制事件，ggml导入MLX或原生复制事件；同一原生队列的后续
region复用首次成功提交的等待。原生产者的完整错误与代际仍独立核对，不用MLX的stream标记推断队列身份。

原生程序实际退休后可以归还输入绑定；后继仍在读取时，保留原输出array和提交代际，继续禁止重新绑定、
提交和复用生产者。外部结果观察者仍阻止输入归还，输入归还也不代替后继的真实退休。Source依赖引用的
原状态提示在该代际内保持有效，完成通知不移空它；下一次合法提交才重置本代际的提示。

BF16 的无批次薄矩阵（2–8 行）复用 MLX 的逐行 GEMV，保留原权重视图，避免物理填充行改变
单行归约次序。设备长度分页注意力按设备上的实际历史选择算法：少于 1024 行时保留 MLX vector
的 Q 预缩放、连续分量归约及原生线程组 softmax；较长历史继续使用 F32 部分和。两者可以共存于
同一个容量内，不把容量当成实际长度，也不承诺两种算法逐位相同。所有原写入判定先于控制及载荷读取。
原 Context 上的冷资格查询检查真实配置、索引范围及预算；具体 Graph、Source、页面权限和引擎路线仍须单独准备。

固定原生 MLX 配方可以导入尚未完成的原输出事件，并借用原命令缓冲区的完成回调；不追加完成专用缓冲区。
公开结果和复用仍等待全部原输出事件、错误及原 owner 的退休。冷期显式准备的对齐同设备共享存储复制可在 GPU
核对原生产者的错误字后执行，错误时不读取载荷或写目标；普通 Graph／主机复制不隐式改用该机制，
普通主机地址、未对齐及 memmove 情形保持原路径。
收集错误字使用冷期有界的遍历空间，每次仅访问一个原对象及其 epoch 一次；它不持有执行权限或替代 Source 保活。
这些机制不改变模型数学、状态发布或引擎的设备闭环准入。

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

原生packed GDN目前接纳F32 Q/K/V、深度不超过256的状态行；准备时检查实际descriptor、buffer与workgroup限制。
每次执行先通过原checked-dispatch机制核对整组序列偏移和快照长度，再写两个输出；空序列逐位保留状态，
不读取padding。错误结果仍等待原生完成与退休，不由设备错误字授权提前复用资源。

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
