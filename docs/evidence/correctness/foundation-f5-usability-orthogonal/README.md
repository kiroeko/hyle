# F5公开用法与正交资源旅程核对

固定product25e，独立按user guide→公开调用→实际证据回映；不把文档存在写为未运行API PASS。

| 普通任务 | 公开consumer与完成条件 | 实际证明范围 |
|---|---|---|
| CPU direct/Graph | public_batch_cpu_consumer，只hyle::compute/hyle::graph；显式recipe/prepare/bind/submit/read | 四storage/dtype域、有效区/GQA/乱页/tail/old output，四路线CPU SKU已运行SDK |
| Linux native FI | public_batch_cuda_consumer，公开provider/同schema plan/coordinator、外部输出writer | actual D256四域Direct/Graph、metadata generation/old output，两installed SDK已运行 |
| Apple/Vulkan typed direct/Graph | public_batch_native_consumer，公开device/staging、原F2 ready lease、typed controller/factory | 各actual GPU static/shared SKU、同generation/old result/readonly output；native-only不需要CPU/SIMD |
| 未准入/软件设备 | Windows CUDA absence consumer、WSL Vulkan软件probe | 具名unsupported/执行前拒绝与软件loader；不借CPU/软件数学宣称actual native资格 |

上表证据是原18格矩阵，public cpp实际复制到隔离目录并从rename prefix编译；没有trusted
src/private桥。guide简短CPU片段明确value只为展示顺序，完整例子检查Status/Result。
raw库文件、内存owner、Context/Queue/allocator晚于work/result/use退出；完整消费者的drain
guard在相关resources销毁前，结果保活期间prepare拒绝，后续episode显式prepare/update。
公开caller不需要重写最后使用/镜像版本/失败排空或内部target传播；原shared FI反证已修。

正交是设计中的分段矩阵与资源版本旅程，不把attention schema假装通用Mem。数据流具体
是两个数值段的Q/K/V、physical页2/0/3或packed offsets，旧readonly资源版本→原F2 dirty
clone/append和tail初始化→新布局/计算→F4 expected-base成功发布；失败复制不发布，旧
资源继续可读。ragged第二段重排与paged精确页权限使用原fixture/实际四路线路径，不含
模型层、token、RNG、admission/eviction/材质策略；opaque entry/segment IDs由consumer解释。
这可映射资产候选更新和分段矩阵批次，共享/权限/退役机制不会读模型语义。此核对不宣称
另有完整渲染器或模型引擎，也不新增第四种通用executor/API。

实际两段非uniform/append/read/旧新旧/F4证据在
[ragged旅程](../foundation-f5-ragged-append-first/README.md)与原paged controller/risk证据；
算子/状态数学及素材策略仍各归原owner。guide明确reference额外gather/GQA/F32/mask/
copy/waits、UNKNOWN、capacity padding、schema identity、borrow/old-result/close边界。
single/multi请求的高性能是否足够不能由这里的可用性决定，仍须最终基础库缺口/engine评估。

这份核对只usability/正交接口映射；分离红队、剩余native san与全goal completion audit
还要独立执行，不用本文代签未验gate或整个CLOSED。
