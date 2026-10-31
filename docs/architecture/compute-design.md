# Compute：计算语义、实现选择与资源边界

Compute为Direct调用与Graph编译提供模型无关的计算、资源和完成合同。它不依赖Graph，
也不解释模型文件、生成或服务政策。用法见[Compute指南](../user/compute.md)，后端差异见[原生路线](compute-routes.md)。

## 语义与执行分开

operation明确全部端口的dtype／encoding、shape／layout、alias、状态、副作用和资源要求。
Provider将相同合同绑定为原生primitive、完整region或合法扩展；能分配内存不等于能计算任意格式。
Graph决定跨片段覆盖、依赖和资源计划，Provider保留片段内部融合、packing及原生执行。

Direct不需要图编译器；Graph和引擎也不为调用Compute复制线程池、Queue、allocator或错误模型。
有价值的抽象应减少真实搬运、同步、分配或维护成本，不能只将成熟region拆成逐kernel虚调用。

## Provider 与资源所有权

Provider有三种合法接入：直接primitive、自有编译程序、明确扩展点内的kernel，按实际能力与完整成本选择。
自有tensor／allocator／stream／workspace形成原生执行岛，边界通过真实import或显式copy／repack连接。
输出由真实owner保活，不为统一out-buffer而复制到影子存储，也不伪造第三方buffer或调用私有字段绕过权限。

caller-owned interop须核device、范围、编码、coherence、borrow／ownership和完成边界。
CPU外部指针能力不能推广到CUDA／MLX／Vulkan；同GPU或同runtime名称不自动保证内存与时间线互通。

## 冷路由

普通调用者选择设备与实际请求，不排列Provider或填写算子timing表。Context捕获环境，prepare／compile绑定实现，
submit消费已准备制品，不重新搜索Provider或算法。

| 冷事实 | 作用 |
|---|---|
| RouteEnvironment | 实际OS／ABI、设备／ISA、features／limits、driver／runtime及Provider身份 |
| workload／ModelArtifactProfile | 可选的制品revision、配置身份和phase／region意图；不持权重或parser |
| RouteSubject | 全部真实端口、Graph region、canonical边界、state／effects和资源约束 |

未知与零值不同，规则所需事实缺失就不命中。模型名、文件名及region意图都不能替代真实端口和DAG检查。
先排除语义／资源／设备不合法实现，再比较完整计划的融合、搬运、等待和workspace。
注册顺序、覆盖节点更多或规则更具体均不证明更快；合法通用基线也不能冒称已测赢家。

只有接纳前、无副作用的能力或预算拒绝可尝试声明过的合法后备。原始OOM、制品创建错误及接纳后的执行失败
保留原错，不吞成能力拒绝，也不暗中换设备或实现重算。

## 制品与执行状态

不可变程序、代码、常量和合法packing可共享；bindings、scratch、mutable原生状态和结果按并发合同独占或池化。
同一槽未到真实终态且旧结果仍有读者时，不能重绑、覆写或回放。共同Invocation接纳后失败会退休该槽，复用须重建；
接纳前拒绝可以按合同修正后重试，重建不修复device loss。

形状域、数值／物理格式、环境／policy版本或相关常量身份变化需要重新准备；普通payload与域内尺寸仍是输入。
跨制品cache由消费者明确拥有，不用裸地址、模型名或隐式进程全局cache代替完整身份。

prepared重放复用已绑定代码与资源，不在submit做JIT、搜索或packing。managed路径可承担已声明的metadata／array
及cache／shape工作，不能借此隐藏重新规划、copy或同步。prepared／managed、capture和是否阻塞分别描述。

## 存储、同步与生命周期

Mem拥有allocation、view、quota与寿命原语，Compute解释native资源和异步使用，Graph提供liveness／alias依赖。
host metadata、payload、第三方heap及driver私有内存分别计量；未知内存不写成零，固定池归还也不等于物理释放。

只读租约可共享，独占写要求对应物理范围没有读者或在途使用。alias按实际存储范围判断，
成功写入后才能发布ready读内容。device仍引用的payload、索引、参数、代码和真实owner必须活到最后使用。

CPU happens-before、CUDA event、Metal hazard／event及Vulkan执行／内存依赖保留各自语义。
统一内存不消除同步。取消只停止未接纳后继，事件记录失败或Status非OK不证明此前工作已经退休；
未知完成状态不能变成可回收状态。停止接纳、观察错误、close与显式drain分别按owner合同处理，不靠析构隐藏无界等待。

原Completion的保活附件承接实际已提交工作的owner；失败不能消费原引用，inline完成可同步释放。
观察报错、放弃公开句柄和取消均不授权提前回收，也不能让结果反向拥有自己的槽形成引用环。
具体调用边界见[使用合同](../user/usage-contract.md)。

## 边界值、结果与状态

边界值保留逻辑TensorDesc、真实owner及实际Mem或原生表示。caller输出、source alias、Graph backing、常量和
provider-generated输出分别声明；生成输出不额外申请一份影子workspace。

句柄可用、数据ready与最终成功是三件事。没有真实Mem投影时明确拒绝该投影接口，不能将非空payload当作已完成结果。
只读导出必须来自真实producer权限；共享coherence domain不授予资源族或范围权限。

片段内部保持原生表示。跨片段只在真实interop允许时零拷贝，否则冷计划明确插入copy、Cast、contiguous或编码转换，
计入工作、预算与寿命。原Source／Completion接续依赖；不得为统一接口增加第二套执行器或提前复制仍须升级为writer的输入。

## 资源权限与有界流送

冷索引可缓存端口坐标、资源分组与候选范围，不缓存本次ready、权限或准备成功。
每次绑定仍核完整resource／handle／offset／length、generation、alias、Queue及原producer；失败保留旧工作与owner。

copy冷准备固定范围、groups、slots与字节上限。独立copy不隐含目标重叠或批内依赖，有依赖时显式分phase；
全批预检后仍可能部分接纳，须保留实际accepted prefix和所有可能被设备引用的控制数据。
copy只搬物理bytes，不替代编码转换。

文件流送由显式产品`compute_stream`提供，普通Compute／Graph闭包不带IO。它组合原资源和Queue中的copy：

```text
File／Mapping → 有界staging → native copy → group ready → 计算使用 → 实际退休后回收
producer完成 → read lease → D2H／共享可见 → positional write → 完成
```

容量满产生背压，旧ready／use lease继续占槽和预算。成功前缀可消费，后面失败不把整批写成成功。
取消不保证中断pread／pwrite；格式、文件发布及模型cache政策归引擎／IO，不由流送服务接管。

## 集合发布不是多设备事务

状态schema声明有限opaque成员与资源约束。只读版本共享owner；draft共享未改成员，显式clone或用新输出替换修改成员，
不能修改base或把外部仍可写alias当immutable。

```text
base → draft共享／clone → 新输出全部ready → expected-base核对 → 发布新只读版本
 └──────────── 旧读者与native uses继续保活 ────────────┘
```

发布只改变集合可见性，不回滚外部device写、不代表文件持久化。冲突保持current；token绑定store／schema／generation，
不按分支完成顺序自动选赢家。旧版本、目录和copy槽有明确上限，clone按实际dirty bytes计费。
模型checkpoint、接受前缀、pending与RNG仍归引擎，算子的initial／final state不证明任意中间前缀都可恢复。

## 分页计算为何需要独立准备尺度

Paged／ragged保持typed数学schema，普通Tensor不变成含混分页对象。arena、page IDs、indptr、tail、active counts、
metadata／device mirror、plan与payload须有相符generation和owner；接纳前核单调、边界、dtype／layout、overflow、GQA及causal位置。

| 阶段 | 合同 |
|---|---|
| program准备 | 固定算法域、dtype／head／layout、容量、workspace与slots，绑定实现 |
| layout准备 | 冻结并验证host快照、metadata上传和必要native planning／等待，产生原slot／generation的事实 |
| native提交 | 执行已准备批次，不读取可变caller span、不搜Provider、不隐含新planning |

容量相同不证明split／tile或workspace相同；Provider对真实拓扑的更新能力负责，Graph在绑定变化前失效不兼容录制。

选页与读取权限按选中制品的真实bank端口、受检访问足迹及准确页guard表达。
逻辑bank描述不授予整bank权限，padding不是有效KV，非法ID不得回退读取row0。
ready读取能力不自动接纳pending来源；pending必须保原Source、全部accepted phases成功事实及对应Queue的实际依赖，
不能只看最后一个phase或用FIFO顺序证明成功。布局镜像成功退休后才发布；同版本复用不在submit重新上传。

### 多会话缓存与共享前缀

Compute提供有限存储、版本和copy机制；token前缀目录、会话接纳／淘汰和完整检查点政策归引擎。
具体对象分工见[推理架构](inference-design.md#多会话多请求与缓存)，不在本页复制缓存流程或支持矩阵。

### 生成记录与缓存重算

生成账簿由引擎记录真实已提交边界，按原输入分段、pending与随机地址恢复完整状态。
Compute不持另一份会话历史，也不从token相同推定状态相同；见[完整提交](inference-design.md#从输入到一次完整提交)。

## 推理执行体系的重构形状

请求、模型数学、冷程序与执行的分工统一见[推理整体架构](inference-design.md#职责与依赖)。
Compute只承担真实计算／资源／完成合同，不在此建立第二份推理迁移清单或进度。

## 推理负载与数值边界

storage、activation、乘法来源、accumulation、output和cache／反馈状态精度独立表达。
FP8、GGUF、AWQ／GPTQ、NF4、MLX affine、NVFP4／MXFP4等编码各有codes、scales、group和舍入规则，名称相似不表示数学可互换。
attention的mask／GQA／位置与变长输入、递归／conv反馈、MoE路由／重复IDs／combine顺序均属于真实语义。
具体typed算子及限制查[Compute接口](../user/compute.md)与[模型覆盖](../implementation/model-operators.md)。

### 精度配方与冷编译

模型配方明确输入与输出类型，转换通过真实Cast／量化运算表达，冷编译选择满足要求的实现。
F32输出或F32累加不授权TF32／BF16乘法；phase也不授予降精度许可。若需更窄数学，使用显式配方或算子已有许可。

例如`F32 X × Cast(BF16 W→F32)`可融合为读取原BF16权重并精确扩宽，不需要整份F32权重，
但不能顺带窄化X。`DenseLinearF32`表达相应显式投影时仍按实际端口类型接纳。
中间转换存在公开读者或后备需要时，须保留相应表示并计费；只跳过热Cast不保证冷期未物化整库。

### 中间值与反馈状态的反例

- 融合保留原算子可见的舍入、求值与非有限数行为；多个BF16算子不能只在整条链最后舍入一次。
- GDN／conv的BF16输入、F32 gate／累加／反馈状态不是同一精度；只支持primitive不能代替外围归一化和动态gate。
- 连续状态检查让每个实现消费自己的上轮状态；每轮注入oracle只证明单步，不能证明长期反馈误差。
- checked索引须在寻址前验证全部所需IDs，不能用改变ID、读row0或放宽权限掩盖不支持路径。

这些合同由实际typed实现和选中region兑现，公共Graph不识别模型名或checkpoint版本来改变数学。

## 依赖与性能取舍

公共语义核不拉入GPU SDK，各runtime依赖按[构建图](build-system.md)装配。
性能比较同时看数值、实际dispatch、packing／变换、提交、最终完成和资源峰值；同Provider包装成本与不同Provider整体竞争力分别说明。
局部融合、workspace减少或kernel数减少不等于整模型提速，最终数据只在[性能资料](../developer/performance/decode.md)维护。

2026-10-31整理前的逐批实现、反例和实验说明保留在Git历史：
`git show 23b204d92:docs/architecture/compute-design.md`。
