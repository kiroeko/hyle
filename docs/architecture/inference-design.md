# 推理系统：职责与执行合同

本页说明推理底座如何分工、何时可以发布状态和回收资源。长期取舍见[两项核心期许](inference-principles.md)；
当前任务只在[进度看板](../developer/inference-progress.md)，实现范围在[工程计划](../implementation/inference-roadmap.md)。
运行命令见[引擎指南](../user/inference.md)。逐批实验不在本页追加。

## 简明执行合同

| 边界 | 必须保持的事实 |
|---|---|
| 产品 | `hyle`是当前引擎CLI；Compute／Graph等基础库可独立使用。`engine/inference`是源码内共用机制，尚非稳定的引擎C++ API。 |
| 控制 | 一个owner串行控制Model、Session、请求和目录。设备异步执行不授权其它线程直接改Session；在途守卫不是线程同步。 |
| 提交 | prepare形成未发布候选；submit只表示工作已接纳。完整状态、pending、RNG和已消费输入在原发布边界一起推进。 |
| 观察 | 同一提交结果只发布／交付一次；pending查询不推进状态。报告、交付或后续维护失败不回滚已提交结果，也不能重新采样或执行。 |
| 权限 | 物理地址、已准备事实、数据就绪和读写权限分别判断。固定地址或capture命中不能替代当前Source／Queue／owner检查。 |
| 取消 | cancel记录意图并阻止后续发布；已接纳工作仍须真实退休。单成员取消不能改变同批幸存者的数学结果。 |
| 失败 | 原Status、已知参与者及真实退休事实分别保留。未知成员或native故障不能伪装成单请求失败；接纳后不暗中换设备重算。 |

## 职责与依赖

```text
CLI／应用
  → 请求、会话、兼容分组与生成政策
    → 模型数学、完整状态及静态步骤适配
      → Graph：逻辑图、放置、资源计划和依赖
        → Compute／原生后端：计算、权限、完成与退休
```

| 所属 | 唯一职责 | 主要入口 |
|---|---|---|
| 请求与缓存 | 请求身份、FIFO接纳、turn输入、credit／ACK、停止、淘汰与恢复政策 | [RequestLoop](../../engine/inference/request_loop.h)、[RequestProgress](../../engine/inference/request_progress.h)、[OutputDelivery](../../engine/inference/output_delivery.h)、[SessionCache](../../engine/inference/session_cache.h) |
| 共同执行编排 | 同步与连续步骤的准备／提交／观察、后继去留和失败收尾 | [drive_steps](../../engine/inference/step_execution.h)、[request wave](../../engine/inference/request_wave_execution.h)、[ExecutionEpisode](../../engine/inference/execution_episode.h) |
| 冷程序与IO | 有限程序目录、具体程序的原IO／物理槽、事务及资源交接 | [ColdShapePlan](../../engine/inference/cold_shape_plan.h)、[PreparedProgramFamily](../../engine/inference/prepared_program_family.h)、[ProgramIo](../../engine/inference/program_io.h)、[ProgramSlots](../../engine/inference/program_slots.h) |
| 模型私有实现 | 权重角色、几何、数学／精度、状态schema、真实阶段和合法修复／发布顺序 | [Qwen模型目录](../../engine/models/qwen35/README.md)；其它家族保留自己的状态与数学适配 |
| Graph／Compute | 算子、融合、资源域、copy、原生程序与真实完成 | [Graph设计](graph-design.md)、[Compute设计](compute-design.md)、[原生路线](compute-routes.md) |

职责分开不要求每行变成新库、基类或registry。共同层复用原Graph／Compute，不建第二执行器、资源owner、
账簿或权限表；底层不依赖模型名单。模型适配采用静态直接调用，保留合法融合、布局和专用快路。

## 模型、会话、请求、批次是不同对象

| 对象 | 持有什么，何时结束 |
|---|---|
| 已加载模型 | 共享权重、冷图、有限程序与存储作用域；不是每请求复制一份 |
| Session | 完整已提交状态头、位置、pending和随机游标；可跨turn保留、重建或换出 |
| 请求／turn | 本次输入、输出额度、停止／取消和主机观察进度；不独立推进另一份RNG或模型状态 |
| batch／episode | 本次参与者、候选状态、原Source／Completion及提交事实；实际退休并结束结果读取后归还槽 |
| 物理执行槽 | 可变IO、workspace、原Invocation及原生载体；按执行并发准备，不按驻留会话数复制 |

驻留版本数`N`、批内序列数`R`、在途深度和物理载体数是不同容量。目录别名共享同一版本时不重复扣费；
字节预算按真实backing／metadata计量，不能由N／R推定全进程或物理显存上限。
Qwen的有限流水使用三个物理载体不构成共同层的固定要求，R2也不表示两个多成员Invocation可以同时执行。

## 冷准备与热路径

冷期确定制品身份、模型几何与精度、数学图、放置、真实资源域、有限形状、程序端口、输出来源、槽和预算。
Qwen的构图、target数学、Family及Phase准备已分别位于[模型私有冷单元](engine-overview.md#模型与冷准备)。
有限具体图可直接组合ProgramIo／ProgramCarrier，不必伪造一套Qwen形状族。

| 冷期准备 | 每次执行仍必须处理 |
|---|---|
| 编译、实现选择、packing和固定端口／索引 | 本次token、位置、有效长度、动态logits输入和候选状态 |
| 有限容量、可复用metadata及原Invocation | 当前RequestKey、epoch、Source／Store／Append租约及真实读写范围 |
| 已证明的capture候选、固定拓扑和资源事实 | 当前Queue、完整tuple、输入来源成功、槽状态及最后读者 |
| 有界exact key目录及淘汰规则 | 新key的明确冷接纳、拒绝及实际退休后的销毁／重建 |

新形状只在有界冷接纳点准备，不预编全部上下文。精度、布局或相关资源身份变化须重新准备；
KV增长也可能改变原生tile／split和workspace，地址相同不足以复用录制。

Capture查询只回答当前绑定是否具有已准备候选，不能授权执行。完整View须沿真实物理根解析，
Copy结果保留独立根；partial／未知别名、输出变化、私有状态和Queue仍按原门检查。
一次未提交后继的typed准备拒绝可保留原Invocation和固定IO，在前驱真实退休后取回原事务；
真实执行错误不能被包装成可重试的资格拒绝。

架构费用以实际冷入口、分配调用、metadata和完整消费者判断。warm不重复冷构图／物理准备是目标；
逐episode owner、动态权限检查和实际读回仍按其寿命处理，不能从live字节不变宣称零分配。

## 多容量设备批次设计提案

本节是依据[成熟引擎源码](references.md#多容量设备元数据与投机解码)形成的实施提案，尚未交付新API或K3提速。
目标是让容量和有效值各自承担正确职责；候选数不再由K1的两种结果决定，底层也不认识K3或接受政策。

### 容量、有效值与计划

| 概念 | 准备时确定 | 执行时的数据 |
|---|---|---|
| 请求容量R | 批次能容纳的请求数、各请求的原始身份与资源边界 | 活动请求及其位置映射；不能指向未接纳的请求 |
| query／token容量Q | 物理输入／输出跨度、每请求上限、总token上限 | 每请求有效行数、前缀偏移；填充行不推进缓存或recurrent状态 |
| KV容量C | Session允许的上下文上限、已预留页面及可访问的最大范围 | 当前有效KV长度、选中前缀、追加位置；设备值不能授予新页面 |
| 原生计划 | kernel／CTA／split、工作区布局、固定地址与兼容域 | 可在该计划兼容域内更新的metadata内容 |

容量档位由消费者以有界集合配置，不能只把2换成4再形成新的全局上限。
首次验证覆盖多个独立R／Q／C配置、非档位实际长度和档位边界；例如R=1/2/4、Q=1/2/4/8、C=1K/4K/16K/32K。
这是验证集合，不是所有后端的默认值或已支持范围。配置缺少合适档位时拒绝或走原有显式回落，不隐式扩大预算。

Q表示物理总行数，不与R混用；每请求最大行数另行检查。C的Session上限、arena物理页数和本次已接纳页面也不同。
同一大schema下可有多个query／原生计划档位；跨schema或cache解释的布局仍不能互换。
context容量配置决定资源上界，不能为每个实际KV长度录一张新图。

### 分工

1. 引擎先按已配置上界选择能覆盖本次请求的容量档位，预留原页面／状态／输出额度。
   即使本轮接受长度尚未回主机，也能依据上界准备下一轮；不为R个请求枚举所有接受组合。
2. 模型图产生token、接受前缀和下一步位置等设备值。GDN、conv、hidden、draft lag、EOS、额度和RNG仍在模型／引擎层。
   底层只接收有界控制值，不接收模型名、K或“接受”政策。
3. Graph表达控制值的生产／消费以及对已接纳资源的读写依赖。候选页、Store和Source沿原Invocation／ExecutionEpisode持有，
   不新建执行器、状态账簿或另一套权限证明。
4. Compute／provider预先证明原生计划兼容域并持有稳定metadata与工作区。GPU只更新已获准的有效值／选择，
   包括必要的页索引、有效长度和原生调度表；不能借此提交任意调用者可写页表。
5. 原Completion成功且实际使用退休后，模型的原事务只发布选中的完整前缀。未选中写入保持私有并撤回；
   CPU输出观察可以稍后发生，下一轮GPU提交不以观察动作作为必要前置。

公共能力应描述“已准备容量＋允许的设备控制域＋后端资格”，而不是“恰好四个完整布局”。
有限、预检查的布局集合可以作为首个后端实现方式；每请求的长度／映射表示必须保留，避免批量增长时组合爆炸。
更一般的设备元数据更新必须是封闭、可校验的原生操作，不能把任意Tensor解释成可信页表。

### 原生计划和兼容

容量足够、地址相同，都不等于可以重放同一计划。CTA、KV分块数、merge索引、有效block mask、工作区偏移和输出载体
必须按实际provider资格验证。计划不兼容时，在接纳前明确拒绝／选择已准备的其它档位；提交后的真实故障走原失败退休路径。
同一槽的设备metadata、页权和工作区不可在原最后读者退休前重写；元数据上传Queue不能越过这个寿命边界。

已有FlashInfer调度器探针使用真实2B几何QH8／KVH2／HD256、page16和4070：
44组固定Q4、不同KV长度的计划字段／工作区大小一致；Q1～Q4混用时block mask偏移从624变为640字节，整数工作区从670变为686字节。
该探针只调用原生host规划，未运行attention、未证明调度表内容／页面权限或CUDA replay兼容。
因此首个实现需要固定容量下的原生metadata规范化与真实GPU检查，不能只放宽Q2检查或改数组长度。
日志在`.scratch/inference-mtp-k3-review-20261105/native-plan1/`。

旧二路API及其零／非零语义保持；新增的有限索引／有效长度使用独立opt-in和版本化provider能力。
新控制值必须校验范围、来源和完整资源域，非法控制记录原Queue错误并使用安全地址，不能先越界再靠主机拒绝发布。
新API的正式类型名、默认档位和ABI布局在原生验证后确定。本提案不承诺所有后端具有相同动态计划能力。

### 验证与采用

先验证多容量原生组件，再接模型；组件通过不能代签K3吞吐。检查包括非档位长度、跨页／计划边界、填充行、
非法索引和长度、外来schema、捕获重绑、生产者失败、取消／重试、旧快照、并发槽隔离及真实退休。
设备控制不能把未初始化或未授权页面变成可读范围。每个分支与独立固定布局执行比较，GDN等模型状态再逐前缀核对。

K1保持原路径作为回归对照。K3接入后仍须满足现有卡片：八轮同窗每臂n3、512项输出一致、图调用比≤2、后继提交>0、
合并decode≥K1的105%、逐轮≥97%、ready增量≤100ms；独立采样资源，冷开销保留在完整进程。
阶段范围和通过情况按实测记录，不把设备草稿单图、可录图或组件规划成功写成轮间无同步与端到端提速。

## 当前调整的边界

共同`drive_steps`控制准备、提交、取消检查、完成、后继保留／撤回和异常恢复。同步适配无需空的successor方法，
连续适配借原Step、Source与credit；函数退出不保留回调或新增执行状态owner。
可选主机游标只记录当前Graph待完成及已决定的后继分支；跨pump恢复不重准备、重提交或提前发布，原A／B仍持有全部native工作。
封闭CLI仅在当前批次已完成后继提交后等待原Completion，不提前等待已提升的后继；对外pump可返回Graph pending，完成后的结果读回和回调仍同步。
GLM实际普通续算与K1候选消费同步分支，Qwen普通／MTP目标步骤消费同一驱动；模型仍决定自己的数学和Head发布。

普通输入／小结果方式在每个物理槽冷期选定，容量拒绝可保留原独立IO；运行中的写入、复制或观察错误直接进入原失败路径，不重新选择方式重算。

`ExecutionEpisode`组合原Invocation、资源交接和退休事实。准备事实可以复用，动态权限不能缓存成另一份“已就绪”。
后继失败时保留前驱已提交事实；前驱失败、EOS、取消或credit不足使后继失效时，已接纳后继仍须真实退休。
跨pump／turn保留候选须核对完整cohort、RequestKey、输入边界和额度，不能把R2的Source直接当成R1准备结果。

整体架构是否完成由具名任务和相应消费者判断，不能以共同driver接通或一个模型通过代替跨模型、寿命及费用结论。

## 从输入到一次完整提交

```text
接纳／恢复Session → 预检完整turn与输出额度 → 形成兼容work
  → 选择已准备程序并准备原IO／状态事务 → submit到原Queue
  → 原Completion完成，取选中前缀的完整结果 → 模型必要修复／追赶
  → expected-base核对 → 发布状态＋pending＋RNG＋已消费输入
  → 请求按提交序号观察一次 → 交付、续算或归还执行槽
```

普通推理有自己的无draft目标图。MTP家族可以融合提议／验证，也可以递归产生多草稿，不能强制成固定三次提交。
共同`AcceptedPrefix`表达合法消费／输出前缀；模型保留真实tap、训练块宽、frontier／lag、修复顺序和RNG地址规则。

pending是已返回但未消费的输出，下一轮恰好消费一次。拒绝候选不发布，bonus或最终输出仍可留pending。
state-only中间prefill块只推进真实输入和状态，不投影词表、发token或消费RNG；输入偏移只观察真实提交后推进。
完整turn的上下文资格在可能产生部分提交之前检查，不能在中间块之后再次扣除完整输入。

发布前完成可失败检查，Head成功后立即记提交见证。Qwen、GLM等可有不同的状态／journal维护顺序；
晚维护或报告失败只补观察，不能重跑已提交数学。旧Snapshot、输出读者及候选独立持租约，不能因Head换新而提前回池。

## 共享请求能力与边界

- `RequestWindow`以固定槽和世代管理身份，可在合法边界接纳新请求；旧key不会因槽复用而复活。请求资源真实退休后才释放槽。
- `RequestProgress`借用稳定turn输入并观察真实提交，不复制模型pending／RNG权威；整轮资格、模型容量与数学阶段仍由适配层提供。
- `OutputDelivery`管理credit和已交付前缀的ACK。背压暂停该请求，不能误记为turn完成；未排空输出保留key，即使Session已释放。
- 多成员普通／MTP批次可只撤回停止成员的发布，幸存者使用原批次结果。单成员、全部停止或明确整批abort沿原中止与退休路径。
- 错误携带原Status、能证实的参与者和独立退休事实；未知native失败不根据异常字符串自动重试。普通C++参数异常仍有各自API边界。

在线排队、公平性及输入／未交付输出字节上限由调用owner选择；网络协议不是这些底层能力的前置。
当前有限CLI预留输入和输出账簿，不承诺在线服务的排队时延或全进程内存上限。

## 多会话、多请求与缓存

PrefixDirectory／SessionCache维护token键、LRU与pin；模型提供可恢复的完整检查点和真实驻留版本查询。
腾位逐项移除可淘汰记录并重新查询真实版本数，不由目录条目数推断已经释放足够字节，也不把仍有读者的版本强行回收。

检查点覆盖schema／边界、KV、递归与conv状态、hidden/frontier、位置、pending和RNG等实际续算内容。
生成记录足以重算时按原分段与随机地址恢复；它不保活全部历史GPU版本。只缓存KV或token前缀不能证明完整状态相同。
外部动态logits modifier及应用对象不放入检查点，新请求使用自己的合法政策。

完整Session换出换入复用StateTransfer和HostStateStorage。数据搬完、发布可见、写权限取回和物理释放分别判断；
固定池归还槽不等于物理显存下降。失败保留原状态与实际在途owner，恢复不重置已观察输出、turn游标或未ACK数据。

## 尊重平台、精度与状态差异

存储、激活、乘法／累加、输出和反馈状态精度分别表达，遵守[Compute数值边界](compute-design.md#推理负载与数值边界)。
模型提供实际状态schema；KV／MLA、GDN／KDA、causal conv及不同MTP前缀不能统一成一块含混“cache”。
shape相似或编码名称相同不授权替换数学、舍入、状态布局或原生能力。

Graph可表达多个资源域、端点与聚合完成；当前引擎Backend装配仍以一个主端点及可选CPU辅助端点为主，
不代表多GPU分片、跨Context等待或多机已实现。CPU／CUDA／Apple／Vulkan保留实际interop与同步条件，
不因统一内存、同GPU或同Provider名称假定零拷贝或相同执行成本。

## 新模型接入与实施边界

新家族实现自己的制品角色、数学／精度、完整状态和静态适配，组合已有请求、程序IO、状态与退休机制。
先用结构不同的真实组件核对共同合同，再补确有重复的机制；不以Qwen层数、K1、两步流程或单KV布局定义公共接口。
新数学与模型私有修改应限制在其编译边界，不要求修改其它家族；具体融合和专用快路由冷期选择保留。

完整引擎当前仍以Qwen为主；GLM／Kimi／DeepSeek的组件证据不等于第二完整模型或任意拓扑验证。
通用loader、C++ tokenizer、server／应用API、图片产品链路及多卡／多机未交付。

## 实施包与收敛条件

任务顺序、百分比和正在做的事项只维护在[看板](../developer/inference-progress.md)，本页不复制进度或逐批检查清单。
完整目标见[工程计划](../implementation/inference-roadmap.md)，有限结果按实际模型／平台／配置读取原证据。
2026-10-31整理前的逐批接线与实验说明保留在Git历史：
`git show 23b204d92:docs/architecture/inference-design.md`。
