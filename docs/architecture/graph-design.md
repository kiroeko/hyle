# Graph：逻辑计算与路线专属执行计划

本文定义目标设计，不保留旧实现兼容要求，不签产品完成或 SOTA。当前接口见[用户指南](../user/graph.md)，
状态见 [registry](../implementation/module-status.json)，证据与未完成项见[六项复核](../implementation/tasks/compute/six-point-architecture-review.md)。

## 1. Graph 解决什么问题

Graph 把模型无关的逻辑 DAG 编译成可重复执行的程序：验证语义、应用已验证的实现/融合规则、决定
边界资源/转换/依赖、规划自己拥有的 workspace，并导出执行和诊断事实。它不是模型 runtime 或通用任务池。

用户先选择 CPU、CUDA、Apple MLX/Metal 或 Vulkan 路线。CPU+GPU 分担模型计算不在范围内；
host 控制/上传下载和同 GPU 多 Provider 组合仍在范围内，见
[Compute 边界](compute-design.md#13-四条独立路线与明确非目标)。

## 2. 三层职责，两个公开产品

Compute 拥有直接计算与原生执行；Graph 拥有语义图和冷编译；compiled plan 是内部执行制品，不另造
第三个 Plan 产品。一套计算语义可以被四条路线各自 lowering，不要求一个万能 compiler/执行器。

普通用户选择路线/设备、提供图和数据、显式准备并运行；不手工构造三轴 ID 数组、Provider 优先级、
性能 profile 或逐节点 native binding。专家再控制资源、并发/lanes、预算或 Provider。具体类名/函数
不继承旧测试，使用指南只描述真正已交付 API，不把本页概念示意写成用户可调用代码。

## 3. 生命周期职责

builder 构造和验证，完成后的逻辑图供冷编译读取；不可变可共享的执行制品保留代码、权重和计划；
每个并发运行槽独占 bindings、scratch、可变 native state 与结果。这些是职责，不强制四个旧类布局。

- 编译不偷执行用户输入或改变副作用。常量只按显式 borrow/materialize/packing 政策读取与准备；
  借用和复制分别说明，制品销毁前不能释放仍依赖的 owner。
- 相同数值/格式/device/packing 合同的 immutable constants 可按真实 owner 共享；不同或未知
  兼容性不能因为裸地址/字节数相同而合并。跨制品缓存由消费者显式拥有，不建进程全局 cache。
- 每次并发有独立运行状态；同一槽未完成不能重绑/重放/重置。排他制品排他到真实终态，不只锁 enqueue。
- 输出可以来自 caller、来源 alias、Graph backing 或 Provider 生成结果。后者由真实 owner 保活，
  不为了旧 out-buffer ABI 强制复制；导出、借用与复用见
  [Compute 结果合同](compute-design.md#36-输出-binding-与生成结果的-owner)。

## 4. 最小完整计算语义

DAG 表达多输入/输出、显式常量、typed operation/custom execution、数值/物理格式、View/Copy、
alias 候选、资源访问与控制顺序。公共语义不使用 vendor op 枚举或字符串属性袋。未知实现可以拒绝，
不能把 opaque/quantized bytes 当普通标量计算。

有限 exact shape 与 bounded symbols/guards 均可表达；无界、数据依赖输出 shape、任意 if/while、
递归和模型脚本解释器不属于基础阶段。recurrent/speculative 等由引擎重复调用适当程序；将来有真正
通用需求再扩控制流，不由模型新闻自动增加基础 API。

### 4.1 执行、存储与时间线

target 决定计算资格，storage 决定实际数据与计费域，lane 决定执行顺序。多 stream 不制造多个假
storage/target；unified memory 不取消依赖。第三方 owner 可以提供自己的 timeline，不要求伪造 HYLE Queue。

编译先限制计算路线，再验证资源可达性/import、数值/格式与同步，只有真实不兼容边界才 Copy。
host buffer 不是 CPU 模型计算资格，同 GPU 的两个 runtime 也不自动共享 stream/allocator。

### 4.2 并发、workspace 与预算

先根据 data/control、lane order、跨 lane 依赖形成 happens-before，再决定内存复用。只有全部旧
使用先于新写入时才能 alias 同一 backing；不能按一个拓扑编号的区间忽略真实并发。

默认保留并发，预算不足就拒绝；只有显式允许才增加可报告的串行依赖换内存。必要的资源计划求解
有确定的时间/状态/内存预算和保守选择，不进行实测计划竞赛，也不追求一般 DAG 的无限全局最优解。

Graph 只给自身 backing 和必要边界转换调用 Mem lifetime/quota primitive；Provider 私有 heap、
runtime 生成输出由该 owner 规划。完整内存报告分别计编译峰值、常驻唯一权重/制品、全部有限 variants、
并发运行 scratch/capture、外部借用与 driver/private 开销，不能把局部 metadata 计数当 RSS。

相同物理 allocation 不重复计费，不同并发槽不能按相同 shape 复用可变状态。仍有 capture/异步
引用就不能回收。unknown 不是零；有限完整预算必须有上界，否则拒绝或明确只限制已知部分。
成功/失败都按实际 owner 归还 reservation；无配额时不创建无用 tracker，不在 submit 逐项计费。

### 4.3 有限派生 shape

flatten 的 `B*T`、concat 的和、head split 的整除和广播相容由只读 shape program 表达。独立维度
由用户绑定，中间维度由 schema 派生，不能要求引擎在图外再维护一套相等关系。

限定为 checked add/multiply/exact-divide/broadcast 与 typed guards；接纳前检查 overflow、非整除、
元素数、layout、alias span 和 workspace。保守区间/关系无法证明全域时缩窄域或 exact-only，不用
端点抽样冒充证明，也不引入通用符号求解器。计算为有界 metadata 遍历，不扫描 tensor payload。

View 默认保持真实 stride/alias，不自动 dense/materialize，不因无 kernel 就丢失资源/控制依赖。
长 alias 链复用已推导结果，lifetime 延长到 backing 根。bounded 制品必须涵盖实际布局和内部节点，
边界 shape 固定不能掩盖内部动态；一次 exact 成功不签整个域。

### 4.4 按资源描述副作用

typed schema 声明 tensor 与其它资源的 read/write/read-write。同资源冲突必须有顺序；可证明独立
的资源可并行；未知副作用保守排序。不用一个全局 effect 锁串行化所有独立工作，也不允许谎称 pure。

编译/绑定核对真实 canonical owner/subrange，两个 ID 不会把重叠 backing 变为独立；跨 Invocation
共享可变状态由调用方显式约束。Provider 保留片段完整副作用和外部顺序，不能以 opaque 为由省略。
引擎拥有 KV eviction、请求优先级、communicator 创建等策略，Graph 只记录实际访问/依赖。

## 5. 冷编译与 Provider 组合

冷编译按所选路线验证图/推导描述，匹配研发时已验证的实现与融合规则，确定合法片段与边界，建立
native 顺序和 liveness/alias 计划，创建制品，生成路线专属程序和诊断。实际资源约束可反馈调整；
这仍是针对真实 DAG 的规划，不是预存所有整图，也不是在模型准备时试跑多个 Provider 争夺赢家。

region 是可收缩 node set 与明确 ports，而非必然连续编号。收缩不能产生离开后重入的路径/新环，
入选集合不重叠并完整覆盖需要执行的节点；只审计受限提案，不枚举所有子集。

三种接入机制见 [Compute](compute-design.md#2-两条用户路径与渐进替换)：原生调用、完整 Provider
region、在 Provider 支持的扩展点内提前插入 HYLE lowering。MLX 内的 Metal kernel 不需要先切开
runtime；ggml-Metal 与 MLX 不因都用 Metal 就能任意交叉串联。

资源/数值/失败硬约束先于选择；候选的局部宣称、覆盖节点数和 dispatch 数不是 SOTA 证据。
显式 caller output 的必要 Copy、Provider 结果导出、layout/repack、融合与同步代价都进入完整路线。
准备失败必须事务化：仅明确无副作用的能力/资源拒绝允许试下一个，真实错误不吞掉；接纳后不换实现重跑。

## 6. 热路径与选择

### 6.1 工作负载约束与既定优选规则

默认面向长寿命、单次运行的低延迟；吞吐/并发、确定性、prepare 预算及内存约束可显式指定。
无 replay 次数就不任意摊销 compile 时间；不把 bytes、ns 和线程数混成魔法分数。

默认选优遵守 [Compute 的研发/交付/绑定分工](compute-design.md#41-开发时验证-best-of准备时应用规则)。
Graph 消费随实现交付的融合与片段规则，不要求消费者提供完整图的 timing profile，也不按实测
profile 重建多个计划再择优。已有 profile 接收/搜索代码不因存在就成为目标架构必须保留的能力。

规则只适用于已验证的数值/格式/shape、device、实现版本、资源/完成边界和 workload；不同 Provider
的局部 timing 不能相加冒充完整图收益。匹配不到胜出规则时采用预定的合格成熟基线；没有合法实现
则失败。没有性能证据的兼容范围不签最优；固定小输仍如实保留，不改名为 parity。

融合候选重叠、依赖、alias 和预算需要有界的结构判断，不能因此取消必要的 Graph 规划；按既定规则
与硬约束得不到可行计划时明确拒绝。规划预算耗尽但已有合法计划时可报告优化受限，严格政策可拒绝。
这些判断不执行 tensor payload、不计时、不搜索新的性能赢家；真正的错误不被规则匹配吞掉。已选
Provider 的必要编译/packing/warmup 仍遵守 Compute 的准备与隔离合同。

### 6.2 Shape、capture 与动态参数

部署固定不代表 position/KV 长度/输入内容恒定。只有影响程序合法性或真实策略的事实进入有限 key；
其它变化通过数据/参数 buffer 表达，不能每 token 新建 graph 或 capture。

声明域须被已准备的 finite variants/general 完整覆盖。准备/绑定期验证并选择已有程序；域外走显式
冷准备，不在热 submit 搜索 Provider。消费者拥有模型阶段/bucket 策略，HYLE 不从数据猜 batch 模式。
capture 不与所有模式强制绑定：CUDA 是原生 graph，Vulkan 是原生录制程序，MLX 是自己的执行方式。

可变输入、shape、地址和参数更新遵守各自 API。Vulkan 已录制 push constants 不随 host 变量更新；
CUDA capture 有真实更新限制；MLX Python/C++ captured scalar 与输入 array 不等价。需要重编译/
重录或改变地址时必须显式准备并保留真实成本。

### 6.3 路线内编译与特化执行

一套必要语义，四套可独立 compiler/Program/run-state；不以旧搜索器、segment loop 或公共对象布局
作前提。CPU 用 prepared block + 持久 worker；CUDA 用 vendor/native 调用与合格 capture；MLX 保留
GPU region/内扩展；Vulkan 使用自己的 commands、descriptors、barrier 与池。

低延迟入口只运行已准备、已绑定的程序，不隐式 prepare。热路径不遍历原始 DAG、推导 shape、查询
设备/Provider、JIT 或常量 packing。整段 native 程序不逐 kernel 经过多层 HYLE wrapper；确有多个
片段/lanes 时才付必要边界协调，事件按真实提交前缀而不是每个节点分配。

不变验证移到 bind/prepare，必要 in-flight/alias/接纳检查仍保留。managed Provider 的实际运行
成本不能算到“HYLE 零成本”之外而从端到端计时中消失。固定调用不等于零指令；资格见下一节。

## 7. 依赖、构建与诊断

Graph 依赖 Compute 的计算/执行合同，不反向依赖；Mem 提供通用 backing/quota/lifetime 算法；
Sched 提供 CPU executor/任务组合；Platform 提供硬件/OS 事实；SIMD 用于真正的 CPU 计算；Obs
提供可关闭的事件。没有具体缺口不重开底层，也不为文档画图造空 bridge 因子。

公共语义核不取得 CUDA/Metal/Vulkan/oneDNN/ggml 全家桶；选定路线的 factor 才引入对应依赖，
source/static/shared 同样隔离，见[构建规则](build-system.md)与[路线规则](compute-routes.md#1-隔离边界)。
Compile report 独立保存必要 metadata，不意外保活 payload；说明 route、边界 copy/sync、数值政策、
内存/并发与拒绝原因。关闭观测不执行热态诊断工作。

## 8. 正确性与性能资格

正确性覆盖真实单/多 op、View/alias、shape/overflow/非法布局、量化数值、独立/冲突资源、并发槽、
部分准备/接纳失败、晚到错误、取消/关闭、结果跨次保留和资源回收。合成 Provider 单测只能证明
协议分支，不能代替真实 runtime；第三方可运行不证明全部隐含分配和失败路径可恢复。

性能分开做同实现 direct 包装税和同层完整片段竞争。锁定精度、decode/prefill、shape/布局、资源、
线程/Queue、cold/warm 和成功完成边界；记录 allocation/copy/lock/wakeup/dispatch。不能拿裸 kernel
对比含 prepare、copy 或同步的对手，也不能用宏或单次点值证明“几乎无成本”。

## 9. 基础验收与停止点

四路线分别完成实际直接调用、图内成熟实现与 HYLE 扩展的组合、资源/同步/错误寿命、构建隔离与
三交付，再取得相应完整性能证据，才叫基础实现/资格完成。Apple 同-runtime 组合不签任意外部
MTLBuffer/Queue 互操作；Vulkan 单机探针不签其它厂商/MoltenVK；M4 不签 M5 NAX。

架构研究只需要有限证据裁决系统约束，不为研究一次扩全算子、跑全故障×设备矩阵或宣称四路 SOTA。
具体任务和缺口只由[六项复核](../implementation/tasks/compute/six-point-architecture-review.md)及后续活动卡
记录，不复制第二张进度表。只有真实反例推翻共同语义/所有权/成本结构才重开架构；新 op、shape
或 Provider 能在既定边界表达时局部演进。
