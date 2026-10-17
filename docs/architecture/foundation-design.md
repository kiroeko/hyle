# 基础机制：事实、所有权与成本

Core、Error、Platform、Mem、Container、Time、Math、Obs和Util各有公开产品，但不需要各复制一套
生命周期、错误、交付与验证规则。本页保留共同原则及无法由它们直接推出的实现取舍。
具体接口见[用户文档](../user/README.md)与[公开头](../../include/hyle/)。

## 五项共同原则

1. **事实、机制与政策分开。** 编译目标、运行硬件、可用实现和应用选择是不同事实；底层不猜测业务政策。
2. **owner决定寿命，view不制造所有权。** 异步、并发和外部借用必须能说明最后使用；地址相同不代表身份相同。
3. **发布、执行与退休分开。** 接纳前拒绝与部分接纳后失败不同；取消/观测错误不能冒充已完成，
   新状态只在所属接口的发布条件满足后可见。
4. **按实际使用付费。** 可选追踪、扩展、缓存和诊断不向未使用者引入第二runtime或常态元数据。
5. **完整成本可解释。** 分配、复制、原子争用、同步、系统调用、峰值及未知开销分别说明，不能仅比较内部快路径。

这些原则不意味着所有类型都线程安全，也不保证所有操作失败后回滚。每个owner仍须说明自己的
借用、并发、失效与完成边界；无法提供相同语义的路线明确保留差异。

## 编译事实与运行事实

Core只处理当前TU的编译/布局事实、必要的可移植注解和不变量诊断。标准C++已有自然形状时直接使用，
不为nodiscard、delete或普通模板再制造宏语言。assume/restrict/unreachable是优化语义承诺，不能当输入检查。

公开布局中的cache spacing由编译目标决定，不能随运行机器改变sizeof/alignof。
Platform报告运行时CPU/OS、cache/NUMA和资源事实，SIMD/Compute据此选择实现；
设备名、逻辑CPU编号和“最高ISA”不能替代实际能力及OS许可。prefetch只是hint，不建立同步或保活。

Platform提供线程、等待、VM、文件系统和动态库机制，不拥有线程池、内存淘汰或设备计算政策。
支持程度保留原生差异，缺少OS事实不编造默认拓扑；文件原子替换、可见性和持久化也不能混为一件事。

## 失败值为何不承担诊断系统

可恢复的OS、文件、资源、设备和协议失败使用Status/Result；程序不变量损坏使用fatal assertion；
运行日志归Obs。Status本身不做I/O，不保存全局handler或TLS last-error，也不成为任意payload registry。
正常poll无结果、队列满/空等属于操作协议时可用值/枚举，不能机械转成运营错误。

常见OK/code-only保持小值和零分配；可选message/native detail由不可变owner保活，OOM至少保留恢复码。
跨平台恢复看canonical code，需要原生恢复才看固定domain/code，不能解析message取得机器值。
跨DSO不依赖std::error_category singleton地址或回调模块的卸载寿命。

Result的错误槽不能装OK；引用结果明确借用而不假装拥有。错误传播只在增加有用语义时加context，
不逐层复制完整函数名/调用栈。保留共享历史与连续移动传播具有不同复制成本，不能一并声称零开销。

## 内存为何保留多种owner

通用堆借用mimalloc；固定块池、owner-local cache、arena、VM/mapping和quota分别解决不同寿命与资源问题。
它们不是可任意互换的“更快allocator”：硬容量池限制可用块数，cache只是归还/保留策略，
arena按region寿命批量回收，quota按声明资源域计费。

allocator借用背后的resource须活得更久，释放带回原owner及实际size/alignment。
有跨线程free不能假设线程退出就可销毁backing；共享cache也不能依赖池活到进程结束。
object pool的重构与保留对象状态是不同合同，不能为了复用而跳过析构或隐藏活对象。

Buffer/View、逻辑范围、mapping与typed对象分开。offset相同或两个wrapper不同都不能证明alias安全；
物理backing共享、读写权限、generation与最后使用共同决定复用。CPU可访问的mapping也不自动成为设备buffer。

NUMA、huge page、提交/驻留和pressure是实际机制，应用决定是否值得使用。
受预算的已知payload与driver/外部库未知开销分别报告；局部quota不等于总进程RSS硬上限。

## 容器为何不复刻标准库

只有真实的布局、复杂度、固定容量、稳定identity、allocator或并发进展需求才需要新容器。
普通vector/map/heap等已能自然满足时继续使用标准库，模型cache/eviction和tokenizer不因内部有索引就下沉。

flat存储、稳定node、侵入式membership与并发reader reclamation分别付出不同成本。
侵入式容器不拥有节点：hook membership和对象寿命分开，移动已链接对象会破坏关系；
member recovery须满足实际C++对象模型，不能以container_of技巧掩盖未定义行为。

并发/锁自由结构借成熟算法，但仍需Hyle的allocator、失效、关闭与内存模型边界。
低元数据并不意味着可以删除保护ABA、reader或独占修改的协议；generation安全也不等于地址稳定。
text interner、registry与一般容器各有owner，不为多个专用实现提前建立万能SlotMap/trie。

文本/UTF/Base64借成熟simdutf核心，保留caller buffer、容量、编码与partial failure语义；
不为便利返回string而隐藏分配，也不将第三方option bitmask变成公共接口。

## 时间与标量数学

duration、monotonic deadline、Unix epoch及thread/process CPU time不是可直接相加的裸整数。
单位、clock domain、overflow和舍入显式表达；系统时间跳变不能改变monotonic等待，
CPU accounting不能用wall time替代。timer取消复杂度和可能的保守early wake属于数据结构取舍。

Math只补标准库未自然提供的标量checked arithmetic与明确数值合同；SIMD承担bulk，
Compute承担tensor/设备语义。不能让helper因常用而携带模型、线性代数或第二套向量类型。

## 工具与观测

同步borrowed FunctionRef与owning callable不同；registry的mutable注册、freeze与查询阶段不同。
稳定entry拥有callable/capture，冻结可移除热路径锁，但freeze不是卸载仍被调用的代码。
plugin reload还需要跨对象与callback排空，不能从DynLib加Registry推导完整管理器。

Obs同步sink不拥有无界后台队列。异步日志/export由真正拥有队列、背压和flush/drain的宿主承担。
metrics优先caller-owned值，trace关闭时不链接profiler或增加热路径代码；开启后共享一个真实backend。
同步日志成功不等于持久化，禁用日志参数也不应产生隐藏计算。崩溃observer只观察，不能恢复执行。

授权与第三方来源见[许可](licensing.md)；交付、宏传播与依赖隔离统一见[构建架构](build-system.md)。
