# Graph：逻辑计算与原生执行计划

Graph将模型无关的typed DAG编译为可复用程序，负责边界、依赖、放置和资源计划。
Compute拥有算子与原生执行，Provider保留片段内部程序；模型构图、生成和请求政策归引擎。
用法查[Graph指南](../user/graph.md)，实现选择查[Compute冷路由](compute-design.md#冷路由)。

## 图、制品与运行状态

逻辑图表达多输入／输出、常量、typed operation／custom execution、View／Copy、alias、资源访问及控制边。
数值与物理编码属于合同，不能把opaque或量化bytes解释成普通标量。

不可变制品共享计划、代码、常量和合法packing；Invocation独占可变bindings、scratch、原生状态和结果。
执行槽按并发准备，Session状态按各自寿命拥有，不能为每个驻留会话复制执行器。
若多个Invocation共享caller IO，须对整套载体排他；不同Invocation本身不授权并发覆盖。

输出可由caller、Graph或Provider拥有，alias仍保留真实owner，不为统一ABI制造复制。
旧结果读完且原工作真实退休后才能复用槽。共同Invocation接纳后失败须退休并重建，接纳前拒绝不等于槽已失效。
编译不执行用户输入或改变副作用，跨制品cache由消费者显式拥有，不建立裸地址或进程全局隐式cache。

## 形状与副作用

图可表达有限exact形状与有界symbols／guards。派生维度由只读shape program维护；checked加乘、整除和广播足以表达需求，
不引入模型脚本解释器或通用符号求解器。接纳前核overflow、元素数、stride、alias span、workspace及整个形状域。
不能证明全域时缩窄域或用exact形式，不以端点抽样代替证明。

View保留实际stride／alias、资源与控制依赖，不因没有kernel就自动删除，也不默认强制dense复制。
schema明确read／write／read-write，调用者以data／control边排序同一canonical范围上的冲突；
node编号不是执行顺序，Graph不靠全局effect锁串行无关工作，Provider不能谎称未知副作用为pure。
跨Invocation的共享可变状态仍由调用者约束。

## 冷编译与有界规划

编译先核语义和描述，再发现叶与region候选；region可以不连续编号，但收缩不能形成离开后重入的新环。
公开中间值、额外读者、数值及effects边界必须保留，不枚举全部节点子集。

| 先处理 | 再处理 |
|---|---|
| 数学、端口、资源域、设备能力和硬限制 | 合法候选的融合、copy／repack、同步、dispatch与workspace成本 |
| 实际已发现且可接纳的候选 | 有界全图覆盖和依赖计划 |
| 可行性及明确拒绝 | 优化预算用尽时的受限结果 |

覆盖更多节点不证明成本更低，region、物理融合与capture也不能互相推证。
仅接纳前、无副作用的能力／预算拒绝可尝试合法后备；原始metadata分配失败或已接纳错误保留原错，不能换实现重算。

对齐是实际端口资格。资源落位后按真实存储根、View偏移与reader要求核定，必要时通过原copy／预算机制物化，
不把分配器偏好的256B对齐当成全部算子的数学限制。Copy和materialize产生独立物理根，不能沿旧逻辑别名误判。
动态地址及原生生成输出在其真实边界继续验证。

## 并发、内存与时间线

Graph先建立必须alias的存储组，再汇总全部读者寿命。可选原地写须证明最后读者happens-before该写，
不能只看节点序号；已发布结果按仍有读者处理。workspace依偏序复用，不以每层临时上界相加或偷偷串行化替代计划。

冷常量复制／packing同样遵守接纳点和原Completion。失败时partial payload、真实资源／allocator、lease、workspace及预算
保活到实际退休，成功后才发布或缓存；借用TensorView本身不构成保活。resident与transient费用按各自寿命归还。

每条跨端点边保留实际描述、编码、owner、依赖和转换方式。相同格式才能byte-copy；统一内存或同GPU不证明interop合法。
Graph表达聚合完成，具体runtime保留真实Queue／stream／memory dependency，必要host wait显式存在。
取消只阻止后续接纳，已接纳prefix仍保活至终态；结束使用不销毁借用Context、executor或allocator。

## 原生组合与重复执行

每个region保留原生程序，原Invocation持绑定、workspace、结果和在途owner。
单路线与CPU＋GPU使用同一DAG和完成合同，不新增模型、算子或Provider名单；具体interop见[原生路线](compute-routes.md)。

### 准备、capture与权限

冷计划保存实际端口／Copy／seal／segment等固定索引，热准备仍验证当前描述、资源范围、lease、Queue和原槽状态。
索引及相同物理span只提供查找事实，不缓存ready或授予读写权限。

capture需要原生计划、workspace、metadata载体和实际参数保持相容；资源改变前失效对应录制，准备失败保留未就绪状态。
完整View沿真实物理根解析，Copy结果独立；partial／未知别名、私有结果及其它Queue继续走完整资格检查。
支持prepared重放不代表所有路径都无metadata／array分配，managed成本按实际声明，不隐藏JIT／搜索／重规划。

### 分页布局与写入

program容量、实际query端口行数和layout活跃行数分别表达；padding不授予额外KV或输出写权限。
query拓扑不变可按Provider能力更新KV布局，拓扑或workspace改变须重新准备。
owned layout的metadata版本、真实页guard与Source保留同一owner，不在后端复制公共权限表。

PagedAppend是外部写effects节点，不伪造reader或tensor输出。writer／reader能力与Graph依赖共同授权精确页写，
不能从整bank地址取得写权。COW有效前缀、尾写和后继沿原Invocation／Queue／Completion执行；
真实退休后取回事务，只有whole Graph成功才可发布新页。失败或取消后的writer须实际退休并显式discard／rollback，
析构事务不证明页已恢复可用。具体接口与编码限制查[Graph指南](../user/graph.md)及[分页准备尺度](compute-design.md#分页计算为何需要独立准备尺度)。

### 主机进展装配

可选Graph＋Sched的`prepare_progress`推进原SubmissionCursor的ready frontier，按真实`blocks_host`与transfer mode
把阻塞段交给BlockingPool；不接管Provider内部线程团队或另建执行器。
原Completion导入依赖并判断真实终态，通知／HostWait只负责进度；Scheduler任务、栈及有界定时观察的成本如实保留。
不用该装配的Invocation继续直接提交。

模型／资产版本的通用增量失效仍是独立未来需求，不给每次submit预埋dirty系统。
逐批实现与实验可从Git历史查证：`git show 23b204d92:docs/architecture/graph-design.md`。
