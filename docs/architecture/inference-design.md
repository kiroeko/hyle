# 推理系统整体架构与模块边界

本轮统一整个单机推理底座：普通推理、MTP、多轮、多会话、多请求，以及它们的缓存、
批处理、执行准备和资源寿命。覆盖 CPU、单 GPU、CPU＋GPU 和 Windows／Linux／macOS；
不同硬件的真实能力分别表达，不以接口统一冒充全部组合已经可用。
当前仍以底层支持为主，CLI 是真实模型驱动；server／应用 API 后续使用同一底座。

本文描述本轮有限收敛的职责和检查依据。当前 Qwen 已共用普通／MTP
自然请求循环；后端装配、输出事实、读回和退休保活已共享。
`RequestProgress` 已共用主机进度、turn 输入、额度／停止和提交观察；版本接纳查询和执行观察事实已共用，
CLI formatter 已独立编译。Qwen 与真实 GLM StateHead 组件已消费共同的请求窗口、目录和进度规则。
`RequestLoop`现承载自然请求的FIFO、驻留腾位、turn切换和待退休身份；Qwen的静态适配器提供实际状态与执行。
兼容分组与请求循环已共用，Qwen适配器提供真实工作形状、兼容条件、完整恢复算法和数学阶段；GLM／DeepSeek／Kimi仍为组件消费者，
尚无第二个完整引擎模型验证整套编排复用。
实施状态只在 [进度看板](../developer/inference-progress.md)更新，功能和硬件边界见[实现范围](../implementation/inference-roadmap.md)。

## 职责与依赖

| 部分 | 只拥有的内容 | 不需要了解的内容 |
|---|---|---|
| 薄入口 | 输入／输出协议、参数、CLI、报告和诊断 | 算子路由、状态布局、原生提交 |
| 请求与会话控制 | turn、输出额度、初始采样身份、接纳、空闲淘汰、恢复和一次观察 | target／draft 方程、codec、Provider |
| 批次与生成编排 | 兼容 work 分桶、普通／MTP 阶段组合、接受规则及提交顺序 | 平台宏、物理权重格式、某模型的状态内存布局 |
| 模型数学与状态适配 | 几何、target／draft、tap、含 pending／RNG 的完整提交边界、合法前缀、修复／追赶 | 用户身份、排队政策、HTTP、其它家族的状态容器 |
| 制品绑定 | 来源身份、角色、codec／scale／布局、精度和参数变换 | 请求调度、缓存淘汰、热 Provider 选择 |
| 已准备模型 | 原 Graph、有限 child、实际端口／资源域、执行槽及固定绑定 | 聊天协议、用户队列、源格式名路由 |
| Graph／Compute／后端 | 算子、融合、冷放置与搬运、权限、执行、Completion 和真实退休 | 模型白名单、会话政策、生成文本的含义 |

这些是职责边界，不要求每行变成新库、抽象基类或目录。共同机制逐步归入
`engine/inference`，模型数学及实际状态布局留在模型目录。复用现有 Graph／Compute，
不另造 IR、执行器、通用 registry 或第二份端口表。

## 当前调整的边界

共同调度按真实调用分工：请求窗口、兼容分组、接纳／淘汰、输出额度和停止属于
`engine/inference`；模型适配负责容量事实、实际工作形状、准备／提交／完成事实、完整检查点与恢复。
优先静态分派和已有类型，不为搬出Qwen循环先造大`ModelOps`接口。组件复用与第二个完整模型接入
分别记录，不能用前者代签后者。CLI的`run(argc, argv)`也不冒充可安装的引擎服务API。
Qwen特殊token集中在模型自己的`generation_tokens.h`，请求额外停止策略仍由`RequestStopPolicy`管理。

首轮提取的`dispatch_batches`已由自然生成与固定输入两个循环调用。它只遍历原先冷预留的
`BatchIndices`，通过静态调用取得资格、数量上限、兼容性及实际完成前缀；缩批后的尾部仍按原
候选顺序等待，不插队。载荷、接纳、模型执行、提交观察及退休仍归调用者。执行或晚观察抛错时
直接退出，不自动重试或回滚已提交事实；该提取不是完整模型无关调度器。

版本腾位共用`make_version_room`：模型提供真实版本容量、剩余目录项和完整接纳查询，目录只负责
移除可淘汰项。每次移除后重查；别名被删而版本数不变是合法结果，容量增长或目录不减则拒绝。
尝试次数受初始目录项数限制。它不推定页、原生工作或字节已经退休，亦不取代`SessionCache`的原账本。

`RequestLoop::pump_once`只推进一波，可在owner事件之间暂停。它复用原`RequestWindow`，
自己保存有限的空闲顺序与待退休key，不拥有Session、RNG、pending或设备工作。静态适配边界为
进度／真实驻留查询、接纳预检、完整恢复与开始turn、执行一波、发布完整turn、原生退休观察。
预检先于淘汰；未创建Session的活动请求也预留名额。执行适配器先解析整波前缀查找，再提交并发布。
完整turn按“发布→离开active→原生退休→归还身份”推进，未知退休保留原key，下次只观察原owner。
普通稳态不遍历历史slot寻找待退休项。`RequestLoopFailure`保留逻辑请求key、原Status及真实退休事实，
不把请求slot解释为某个模型batch的成员索引。

开放输入没有后续turn时保留请求并pause；新输入到达后用同一个key唤醒，关闭空闲输入不重新执行旧turn。
中途pause／wake不重置已消费输入、已观察输出或额度。适配器必须在返回前观察真实提交；晚观察抛错
不能撤回已提交结果。固定输入teacher的cohort与phase规则不同，继续只复用共同批次遍历，不冒充自然FIFO消费者。
[有限主机、真实CPU续接及CUDA普通／MTP对照](../evidence/correctness/inference-request-loop-2026-10-26.json)
支持这次迁移，不以有限CLI日程代签完整请求控制。

请求可在enqueue时借用自己唯一的`OutputDelivery`；省略表示同步消费。`credit`只查询原游标，
`acknowledge`先核对完整key再确认实际已交付前缀，不修改模型提交或RNG。信用不足时暂停调度，
有效ACK仅自动唤醒因信用暂停的请求。最终输出尚未ACK时，原Session可以在真实退休后释放，
请求key与已提交输出仍保留在单独等待表；它不会成为虚假的设备等待或阻止其它请求执行。
最终ACK在下一次pump归还身份，旧generation的控制不能影响复用后的slot。

`cancel`沿原模型`stop`返回真实Status：排队／暂停请求撤回，活动请求先记意图，原波结束后才移除，
不破坏借用的active span。普通R2以上已提交batch沿原成员撤回合同完成幸存者；其它角色仍遵守原
整体取消范围，不把MTP假装成普通单成员取消。可选的静态提交后观察点让同一owner在原submit之后处理控制，
同步调用省略时不引入回调表。高层生成读取真实`EvaluationDisposition`，取消成员不发布token、ledger或prompt history。
原生退休与输出排空分别判断；只有无native错误的预期取消可作为正常取消结果，真实设备错误继续保留。

当前控制由单owner串行驱动；这与设备执行并发是两个维度。未来应用可向owner投递请求、取消和输出额度，
观察有序事件，不能直接并发改Session。已有错误事实包括底层`Status`和带成员／退休信息的`EvaluationFailure`，
尚未统一全部引擎异常。后续API需同时保留错误类别、影响范围（成员／批次／模型）和真实原生退休事实；
不靠异常字符串区分容量可重试、输入无效、取消与设备故障。

自然消费者的执行错误在最终缩批后的bucket仍有效时桥接到`RequestExecutionFailure`。
仅原`EvaluationFailure`已确认的成员索引映射到完整`RequestKey`，不使用active或slot索引替代；
未知成员、越界成员及计算／退休中的native故障保持批次级。桥接保留原计算Status、成员索引及独立退休观察，
请求身份本身不授予回收、重试或取消权限，正常路径不创建额外成员表。
[有限检查](../evidence/correctness/inference-request-error-identity-2026-10-27.json)覆盖四组主机身份／错误事实及正常B2多轮CUDA报告一致；
未注入真实模型或设备故障，不把正常生成通过当作失败隔离证明。

Provider身份可表达第三方实现来源；例如[DeepSeek来源接口](../../include/hyle/compute/providers/cuda.h)按数学、类型、形状及设备接纳，
不读取模型身份。CPU的`kWins`只在锁定的已测构建／硬件条件下调整合法候选的[冷偏好](../../src/compute_route_cpu/default_policy.cc)，
未命中仍保留正常候选。完整CUDA组合仍自动携带ATen模块，覆盖稳定Softplus、BF16融合SDPA及部分制品解码；
同一模块也承载原生packed路径，不能把移除模块等同于只去掉ATen调用。
显式Provider组合控制数学候选，不表示完整CUDA安装闭包已可裁去ATen；当前保留该能力覆盖与依赖取舍。

当前Model仍每Phase单Invocation、同时一个episode。普通B2参考也逐episode同步，单槽不是已证实的差距原因。
CPU提前准备下一步仍值得独立设计：元数据准备与设备提交分别推进；设备token接力需要真实生产者、
依赖fence和未发布候选的权限；后继以候选为base时须按序发布，前驱失败撤销所有后继。
停止／取消／背压还要约束提前预留的输出和无效后继。单纯加第二份buffer不能满足这些合同。

提前准备不得改已提交head、ledger、RNG或交付游标；它失败时只撤回自己的预留。
后继设备提交另须取得前驱候选的真实读取权限、原生顺序及完整终结事实。
模型层给出实际依赖、工作形状与资源需求，共同层管理有界推进，不把Qwen阶段表搬入Graph。

已有机制和剩余缺口如下：

- Graph的`InputSource`可保留尚未完成的不可变生产者，按原Queue选择device／host wait，
  在消费者真实退休后补核生产者完整Status。`OrderedGraphSubmission`及typed pending frame
  已用于原批次搬运组合，Compute已有有序读取writer的内部权限原语。
- `Invocation::result/output`继续只导出成功终结的输出。同build的`detail::SubmissionSource`
  接管一次原提交proxy、Queue和完整终结事实，只有一个producer保留；`OutputSource`是固定caller-owned I32视图，
  各自核原writer／pool、精确范围及Plan强持的真实`StorageOwner`，不公开pending字节或制造ready租约。
  一组视图用原子`OutputBinding` bundle绑定，依赖按共同Source身份去重，实际资源权限仍逐值接纳。
  bundle保全部视图寿命，Source不反持bundle；原生退休另保真实owner，不增加观察者或执行器。
- 一次A→B依赖链与持续有限槽轮换分别实现。完整成功且实际退休后，view的`detach()`取得原只读权限；
  所有借用解除后，调用方显式`SubmissionSource::detach()`结束该代导出并归还producer保留。
  B仍读旧输出时，A才可绑定另一块真实writer继续执行，不能覆盖B尚在读取的块。
  `poll/wait`不隐式关闭导出。数学失败不发布ready；失败已实际退休且没有借用时仍可归还载体，原错误继续传播。
  detach不等待pending工作，最后owner释放仍有正常析构／分配器清理成本。
KV其余入口、索引融合读取及持续轮换仍须另接，不能由token链代签。
- CPU可以先准备不依赖候选数值的形状、位置、容量和独立scratch，不能并发改当前`Phase`输入、
  layout、共享`input_scratch`、tail或工作态。两个capture槽也各自保持物理参数与权限寿命。

后继以原前驱候选及其代际为base，按原依赖执行。只有前驱成功、expected-base仍一致且本成员
仍有额度时才顺序发布；前驱失败／取消使其未发布后继失效，已接受原生工作继续保活至真实退休。
EOS、停止或背压使无效后继只退休、不发布token或推进RNG，提前预留的输出信用计入上限。
后继事件成功不能替代前驱的采样有效性、原生错误或完整状态事实。

容量按实际未发布候选计量。R1／K1若在前驱发布前提交后继，必须同时保留旧head、前驱目的及后继目的；
仅两块状态会使后继覆盖失败回退需要的旧状态。容量计划现显式接收在途批深D：
状态槽为`N+K×D×R`、版本槽为`N+D×R`，KV额外COW尾页为`D×R`；
每批的请求、query及page-ref数学上限不随D扩大。默认D=1保留原占用，实际调度启用更深在途前必须传入真实D。KV保留原追加／COW权限与各候选
真实长度，不提前扩大已发布帧的可读范围；MTP接受未知时等待真实接受事实，或由实际实现修正
候选长度，不能按全接受页界执行。普通`SlotBindings`及`PagedAppendPages`仍保ready／真实写权限合同；
在途只读视图独立表达，不把未发布候选包装成ready状态。当前首批Graph KV接线见下，持续追加仍需另接。

一次设备token依赖链及前驱checked错误传播已取得[有限CUDA检查](../evidence/correctness/inference-graph-output-source-2026-10-27.json)：
host gate实际阻止前驱完成时，后继仍可用原Queue提交并取得正确结果；前驱完整错误阻断后继发布，真实权限随后退休。
消费者只走Commands与已有metadata-only准备合同，普通submit不能绕过原有序读权限，capture准备也拒绝；
成功返回Completion后才消费一次提交资格，接纳前拒绝仍可修正后重试。原producer若已将writer成功发布为read，
提交使用该真实只读权限，不制造第二份状态。ready分支未在本批单独运行。
后续[有限CUDA检查](../evidence/correctness/inference-graph-output-detach-2026-10-27.json)另验证：
A成功退休、B仍在途时detach，A改绑第二块真实writer提交下一代，B仍读取旧值，旧source的终结事实不受新提交影响。
实际使用两块输出缓冲，未减少其容量；没有复制载荷或增加执行器。该适配尚未接引擎，
同一次原提交的[共同owner检查](../evidence/correctness/inference-submission-source-2026-10-27.json)另已完成：
两个不同I32输出送入同B两个输入，source／device-wait均为1；后项无效时整批绑定保持，原错误及真实退休分离。
旧单输出入口已删除，不保兼容重载；Source本身不限制consumer，每个实际bundle保自己的绑定／一次提交身份。
接下来仍需KV其余入口、索引融合后继读取及有界轮换；下方Store及首批KV精确视图已另有有限检查。
Store／append取回与publication不能继续依赖已被take移动的内部writer指针；KV已有append_reserved及native-use保护，
不能为接力直接撤掉。普通请求的有序发布、额度与停止及MTP接受／修复仍待接通，没有完整重叠或提速结论。

引擎的Phase现把输入、读回和attention布局按执行槽持有，普通Text外部token采用`[B*T,1]`，
再以原View还原embedding所需`[B*T]`，T1可与采样的`[B,1]`原输出直接对接。
非空caller读回把完整WriteLease交给原Graph，重复提交沿原发布／取写权门复用，
不能以裸TensorView代替Source权限。跨Invocation重绑需先撤回原绑定并满足实际结果、Source与读者寿命；
reset释放原绑定后的同池重新allocate也是原有合法路径。
[三项合同及四条CPU／CUDA双会话检查](../evidence/correctness/inference-relay-carriers-2026-10-27.json)
验证本批普通／MTP单槽路径；独立诊断入口只适配源码，未单独编译。实际引擎仍单槽，
未声称完整设备token接力、重叠执行、capture或新tokens/s通过。

剩余接线按以下共同职责收敛，尚未实现的部分不能由现有token检查代签：

- 一次原Graph提交只持一份proxy、观察锁、实际依赖及完整终结事实。token、Store和KV视图共用此提交身份，
  沿原`InputSource`按owner及实际Queue去重等待／终结核对；每个值的实际读权限仍单独接纳。
  去重要求共同terminal adapter与提交identity同时相同，不能为Store／KV各造新vtable后仅复用identity。
  不用Plan代际表示提交代，不为每层状态搬动Completion或增设观察者／执行器。
  `poll/wait`只观察，不隐式关闭导出；各typed view先独立，调用方再显式关闭该代导出并归还producer保留。
  此控制须核实际退休，与数学成功分开：失败不产生ready权限，但不应因后继仍保留终结记录而永久挡住原事务回收。
- typed pending读凭据须独立保活原block、generation及精确范围；append还核原epoch与已写前缀。
  它不是ready租约，不公开pending字节；仅原typed owner可导出，不能借任意TensorView取得权限。
  前驱成功实际退休、所有借用指针已分离后，才解除原结果保留并沿原take／publication发布。
  初次发行应在原producer成功submit之后：提前增加block引用会触发现有writer唯一性门，不能全局放宽该门来迁就凭据。
  导出封住本写代的新写接纳和可写地址出口；只在原publication后、所有读者归还、真实独占的try_write中开启下一写代。
- Store发布须证明原writer已退休，允许已接纳的只读者存续；KV另须支持受检的pending前缀读取及原epoch发布。
  新append仍须等旧读者，或使用真实独立／COW页，不能把原保护门整体移除。
  `WorkingFrame`继续只表示已提交ready范围，不承载未发布候选权限。
- 深度二按实际候选增加状态／页／载体预算，先支持已证明的单shared Queue。
  真实多segment／managed路径还须由原提交游标给出可等待frontier并保留完整聚合Status，
  不把最后一个event当成整图成功，也不以单Queue组件通过代签完整模型。

普通完整writer的独立`PendingReadTicket`已实现，不含writer指针、可读视图或第二Completion。
它保原block／domain／generation／精确范围，并允许同一原有序读接纳跨越writer移动与真实publication；
新的写接纳和可写地址出口由本写代的封写标记机械拒绝。
[有限检查](../evidence/correctness/inference-captured-submission-2026-10-27.json)覆盖受控Completion下的寿命、
失败分离、关闭与唯一ready重开写；该批次未接Store目的槽、KV append或真实GPU状态ticket。

已有append页的在途读另用独立`PendingAppendReadTicket`保原block、generation、epoch和
`committed+reserved`精确前缀，不持`AppendLease`指针，不提供ready视图。发行前由F2 owner证明
原追加实际提交，发行后封住本epoch的新写接纳和可写地址；实际追加登记为write，读者只登记read，
其取消或失败不污染前驱append epoch。原追加成功且所有真实write退休后，受检发布桥可在只读后继
仍在途时发布该精确前缀。旧ReadLease不扩界，公开publish及reserve／rollback仍保原全部native-use限制；
新reserve才开启下一epoch并使旧票据失效。已有页权限原语不单独证明Graph KV或完整流水；后续接线另列。
[有限检查](../evidence/correctness/inference-hybrid-append-2026-10-27.json)中的三项新增append合同
使用受控Completion，不是GPU KV接力证明；四项原append合同及原ticket相邻检查同镜像通过。

共同`ResourceSource`现沿原资源表核selected／nominated writer、原代际及ready／full／append权限。
固定票据元数据在前驱提交前准备，提交后的短metadata visit只借原slot，capture后不留writer、appender或交易指针。
ready只覆盖后继仍用的旧页，不把COW复制源或controls混进读取足迹；同根旧prefix由append票据覆盖，避免重复接纳。
全部地址及owner预检在首seal前完成，包括无副作用的原append精确prefix检查。
token、Store和资源视图沿原唯一bundle／terminal table／proxy／Queue／whole verdict接纳；
同Invocation拒绝另拆第二个pending bundle，防止依赖去重后由哪个代表决定prepare门。

`graph::batch::detail::PagedAppendSource`负责原arena／layout／物理page前缀的数学解释，
Graph只管理具体读凭据和原selected-reader资格，不反向依赖GraphBatch。其installed detail头归GraphBatch产品。
后继query_indptr可不同，但每个KV页的最大所需前缀必须由前驱实际来源覆盖。
显式metadata仍归原configuration；owned-layout则由原selected region的冷query导出实际mirror的tensor与ready guard，
不增加虚构Graph输入。bundle在bind／prepare／submit逐项复核同一实际region、原block、generation及完整范围；
私有controls按真实root冷去重并计入原权限容量，在原LeaseUseBatch接纳read。保留guard会阻止mirror.try_write，
实际carrier改变仍须先走原before-command-change及capture参数校验；Prepared路径同样核guard。
fresh转换和已有尾页推进都先核整图成功、原交易及真实写退休，再沿Compute/Paged原池发布；
前驱源关闭后仍保原whole verdict。fresh转换后旧full ticket不再接纳新读，已有读者可正常退休；
后来的读者须用真正发布的bounded ReadLease。原public publish、reserve及rollback门保持。

[首批真实CUDA检查](../evidence/correctness/inference-paged-submission-source-2026-10-27.json)覆盖两份KV资源共用一次来源、
ready／fresh／已有尾页／真实COW、不同query布局、0新增录制的后继读取，以及后继仍读时take／delta提交／ready重试。
三项Paged资源检查另覆盖非法ticket／行数／旧页、失败frontier和真实读退休。
[owned-layout补充检查](../evidence/correctness/inference-paged-owned-source-2026-10-27.json)中两种实际shared Queue reader同镜像通过，
非法末控制项整束拒绝后原束仍可绑定；owned节点仅Q／K／V三端口。相同layout仍可准备，改page IDs被真实mirror guard拒绝，
原metadata全部字节不变；未推证失败后原capture仍可复用。delegated路径仍待补，不能造ready frame绕过。
首批B只读attention；现`make_append_from_source`可在A提交前按其真实布局和地址创建B的原append交易，
B持共享typed事实，不留A writer指针。COW沿原Ranges及原生程序复制到同arena的不同fresh页，不夺A尾权。
实际COW prefix作为原effect读登记，按该真实step加入原InputSources，和token／state／KV共用Source及原子bundle；
普通submit缺精确ordered map时拒绝。新full来源另记真实可读前缀，不能把整页padding算作已初始化。
提交只接纳实际copy覆盖的票据；满页边界没有copy时仍登记完整前驱依赖，不发明range或tensor输入。
A沿原发布桥取得真ReadLease后填固定共享cache，B成功退休后才从该cache继承未改页；票据不能冒充ready。
[两条真实CUDA追加检查](../evidence/correctness/inference-pending-append-source-2026-10-27.json)覆盖COW及满页新页、
一次Source／device-wait、原take／发布／快照、真实数据与128／129字节边界。
连续来源现将未改页扁平继承到原权限块及具体slot；capture先核B原交易的真实前驱effect附件，再封本代writer。
权限块不含Queue、Completion或前代Source，实际终结依赖仍归原提交；不递归保留终结观察链。
原typed发布取得全部真实ReadLease后，先核原block／generation／完整有序范围，才更新本来源的ready权限；
继承项只能核对、不能扩大原权限。接纳先用真ready，否则保原ticket及原错误；full转append尚未发布时不伪造可重试状态。
原Source销毁及最后继承pin释放按slot清真实资源，保留一页不能拖住同元数据内已被COW替换的其它页。
[连续CUDA检查](../evidence/correctness/inference-inherited-page-source-2026-10-27.json)覆盖满页及COW两条A→B→C旅程、
A发布后C读原页与B在途新页、旧快照及逐页回收；未新增故障注入。普通ready贡献仍保原双层核对，
新在途继承按物理范围排序及查找，不据此宣称所有主机准备已消除。索引状态承载见下；同尾有序写仍需接线。
完整模型、连续轮换及吞吐尚未验证，不能把本片当作KV流水完成。

Store精确视图现由`StoreSource::prepare`在A提交前预留固定票据／坐标元数据，保留原预算与实际allocator owner。
成功原提交后`capture`重核原交易、代际、范围与资源，再封写导出票据；之后不留writer指针。
原destination IDs以精确bounded ready guard保活，不另上传、不取得整池ready权限；同代第二视图的几何检查
只比较原writer范围，不因封写而请求新的可写地址。此guard也会阻挡旧控制载体过早重写，不能当作无背压的连续轮换。

混合bundle先用原逻辑Graph核实际checked Gather的pool／IDs边和当前Invocation具体形状，再核原selected reader的投影资格。
token、capacity及真实ready IDs沿原`InputAccess::bind`整束提交，失败不部分改绑；成功后只无失败地恢复精确pending投影及只读权限。
共同terminal adapter保一份原Status／Queue依赖，实际每块读权限仍单独接纳。
原A实际退休后可显式detach共同Source并take交易；`publish`须同时证明整图成功、真实退休及返回writer与ticket精确相同，
再沿原`publish_after_writes`发布，不能以detach成功代替数学成功或放宽public publish。
[有限CUDA检查](../evidence/correctness/inference-pending-store-source-2026-10-27.json)覆盖非连续槽0／2、真实后继在途时取回／发布／reset、
同形状诱饵IDs拒绝与旧绑定、第二视图、失败传播及未选槽／padding；不是KV append、索引GDN／conv后继读取或完整模型流水证明。
Store视图自身元数据已预冷；共同Source、混合bundle及原prepare仍可能分配冷元数据，完整模型接入前须在N之前按真实预算有界预留。

索引状态现在用`slots::SourceMode::kImported`在冷准备时区分源投影和目的槽。后继只持自己的目的writer／IDs，
前驱原pool、ready IDs及读权限由同一Source束拥有；冷桥从实际Store／Gather及已选region生成投影，不让引擎识别Provider。
`store_ready`只接受该真实StoreSource全部目的的原root／generation／坐标及成功退休证明，用于首次真实捕获；
不带ready guards的在途来源仍只能复用已有捕获。查询重核实际束附件、完整投影及IDs guard，裸TensorView不能冒充来源。
`take`只回收后继目的权，保留无权限物理配置；`release_outputs`须等本消费者真实退休、先take受影响writer，
再沿原native全部预检撤本束来源及guard，失败也可清理，不能提前撤在途读权限或牵连其它输入。
私有ID值不变时复用原成功发布的真实carrier，免去重复上传及同Queue等待；改值仍须原try_write、传输及真实退休。
[有限CUDA旅程](../evidence/correctness/inference-pending-indexed-source-2026-10-27.json)覆盖F32 GDN／conv、原捕获复用、
前驱退休而后继仍读、真实数学错误及退休清理；当前是完整目的集／K1承载，MTP非连续接受子集、有界持续轮换、
完整模型和吞吐仍未完成，不能把组件检查当作完整模型重叠执行已交付。

当前Qwen入口仍用一个串行载体；下一步接入有以下具体约束：

- 普通token输入`[B*T]`与采样输出`[B,1]`不一致，可将外部输入改为`[B*T,1]`后立即用原Graph View展平，T1直接沿同一资源接力。`PreparedReadback`当前仍绑定裸TensorView，设备接力需按实际输出事实移交full writer，并从Graph发布的原租约读回。
- I/O须按执行槽独立持有，不能只改槽数。普通K1保留old head与两代候选时，[状态预算](../../engine/models/qwen35/session_state.h)下界为N＋2R，额外KV COW尾页至少2R；首次真实边界／新shape可串行，稳态两个载体须均已冷备imported来源及真实tuple捕获。
- [batch准备](../../src/compute_batch/batch_execution.cc)和[append布局上传](../../src/compute_batch/paged_append_program.cc)仍同步等待原Queue。完整流水须安排真实提前准备或有所有权及退休证明的异步承载，不能删wait后把未初始化metadata当ready。
- finish须沿原Store／Paged Source受检发布及整束退休清理，不能再对在途读页用普通publish／unbind。设备token读须checked；采样valid、EOS、取消、expected-base、ledger及RNG仍在原host提交门判断，后继只写独立候选，不能提前发布head。

Region的prepare／seal已统一为metadata、资源、owner及未提交recording准备，不读取载荷或接纳设备工作；
[真实Store检查](../evidence/correctness/inference-graph-output-region-2026-10-27.json)已解除按hook存在与否拒绝的过度限制。
ggml CUDA普通Commands仍可能维护内部graph cache、惰性分配或走同步路径，保留managed／host-blocking事实；
`record_compatible`只表示原encoder可录制，不是普通执行无阻塞的证明。
Graph已增加窄`detail::ReplayAccess`沿原准备实现复用已完成的捕获：核同一实际Queue、物理key、
稳定地址／私有参数及实际range／gap覆盖，pending输入只允许复用，所有range预检命中后才选择原记录。
miss在begin capture及variant eviction之前拒绝，不新录制、不降回managed Commands，普通`ReplayOptions`不扩维。
`prepare_outputs`固定该bundle选择的Commands、Captured或Hybrid模式，submit仍核实际完整路由与原有序读权限。
[有限CUDA检查](../evidence/correctness/inference-graph-output-capture-2026-10-27.json)证明原ggml Gather→Add区域
在生产者pending时取得0次新录制／1次复用，物理tuple不匹配时拒绝，原块读写权限交接及结果正确。
仍执行原metadata hook／预算及权限准备，不表示零分配。真实双Concat I32生产者的pending接力另已通过有限CUDA检查。
当前共同冷分类也允许真实Hybrid：range必须有序、合法且持有效原捕获，未覆盖的command仅允许
SharedQueue／Prepared／非阻塞，所有managed／blocking command仍须实际覆盖。生产者核本次真正launch的span，
Commands不拿休眠的记录作覆盖证明。消费者仍核每段既有物理tuple与私有参数，缺任一段就拒绝，
公开RequireCapture仍要求全部command可捕获。单个[真实Hybrid检查](../evidence/correctness/inference-hybrid-append-2026-10-27.json)
已覆盖两Concat＋checked Gather捕获段与Store间隙、换址miss、同carrier ready→pending、0新增录制和B原proxy接管。
撤回ready投影时仅实际Operation的稳定后端绑定、无native输入与原私有参数保持证明允许保记录；Region仍保原显式合同。
未覆盖managed间隙、多segment、完整模型或持续轮换，不能改写Provider静态能力或推证吞吐。

真实Qwen接力另有已核实的接线缺口：`PreparedReadback`目前以裸TensorView绑定token输出，
不是Graph持有的完整writer；普通最终ID为dense I32 `[B,1]`，下一Text T1输入为`[B]`。
优先用既有typed reshape统一普通最终token端口与原writer所有权，不为该消费者另造Source reshape机制。
已有正式CUDA窗口的ordinary与MTP decode均为Hybrid；记录未保留逐gap覆盖，不能声称完整capture。
当前已按原`captured_ranges`／`capture_spans`与提交游标区分已捕获command和Prepared／nonblocking gap，
保留同一Queue、原完整Status与reuse-only物理tuple／私有参数检查；实际模型仍需逐gap核查其资格。
token接力不能替代validity、KV／状态、RNG及停止的联合发布。

普通多成员请求已先接[CPU整数准备](../evidence/correctness/inference-cpu-ahead-2026-10-27.json)：
冷阶段按执行元数据预算预留scratch，GPU执行期间仅填下一步位置与旧页拓扑，不提前取得状态／tail或布局版本。
下一次真实prepare完整匹配Session控制块、epoch、长度、成员顺序及采样形状后才使用；
停止、换序、取消、失败或不匹配均丢弃。R1及MTP跳过，布局创建和所有原权限检查仍在正常prepare。
这只是CPU工作提前，不是后继GPU提交、有序候选发布或完整重叠的完成证明。

索引GDN／conv状态池保持稳定物理池与版本化旧／新槽，不把旧Snapshot变成可变视图。
它与已撤回的固定IO复制方案不同：引擎已接原池及统一Store事务，CUDA tiny普通／MTP通过有限检查；
CPU／CUDA的B2独立接受前缀也已检查，Apple／Vulkan普通与MTP K3完整tiny入口另已通过有限检查；
两个tiny制品接受草稿数为零，不代表正向多token接受。完整CUDA普通／MTP窗口另有实测，跨平台性能仍需核查。
先复用现有独立状态端口、`LeasePool`及`ReadLease::try_write`表达槽权限：普通会话优先轮换两槽，
旧Snapshot／prefix／fork仍占用时从原有界容量取得备用槽，不能保证所有场景只用两槽。
真实退休并撤回绑定后才回收；MTP每个可接受前缀保留独立候选槽，接受时选槽，不按奇偶覆盖旧版本。
固定槽不自动消除provider复制：CUDA ggml的K1闭合片段已有GDN直接状态输出，不能再把旧的
packed状态拆分复制当作当前K1热点；Apple仍分配packed输出。K>1的前缀槽及非连续多请求索引
计算需独立承载：私有CUDA GDN／conv及Graph融合已通过下述有限检查，真实Qwen消费者已观察到原生融合，
不能把K1承载或地址复用记成整段能力已完成。

具体实施沿原`LeasePool`的单份backing增加`LeaseSlotLayout`：固定item、stride与容量，
只从同一domain的完整原租约推导槽号和generation；地址载体不变成整池读／写许可。
源可共享，候选仍取原独占writer；history与GDN各用自身池的槽号，不假设两池分配次序一致。
槽资源接口的25项Windows组件检查已通过；Graph写事务与CPU／CUDA／Apple／Vulkan Store已接通，
CUDA窄链索引GDN／conv另有下述有限集成结果；共同批量控制及引擎接线已实现，
Vulkan原码16位Gather／Store及精确短租约同步I/O已接通，仍不放宽权限或读写调用者padding。
接口检查及测试预算修正见[槽资源记录](../evidence/correctness/inference-lease-slots-2026-10-26.json)。

逻辑图保留选槽、原GDN／conv数学和写回效果，冷规划再选择原生融合。
GDN融合从源槽读取，按时间升序直接写各候选槽。conv的普通图由原精度history Gather、
与新chunk的Concat、Cast F32及CausalConv组成；每个前缀的history直接Slice原精度samples，
沿候选轴Concat后reshape为batch-major `[B*K,L*C]`，交原Store，不从F32结果反向降精度。
原生conv融合只替换寻址：从history槽或新chunk取样并直接写候选history，保留原F32计算顺序，
避免整段拼接和窗口物化。普通K1与MTP的差别是候选槽数量；CUDA两者共用此融合。
原精度F32／F16／BF16载荷与history同类型，系数和数学结果保持F32；history从原码取样，
不经F32结果回转。B>1且K1的普通图窗口可有batch间隙，须沿原Copy等合法物化再交稠密Store。
有普通稠密观察者时仍须物化该结果，不能静默删掉公开值；无索引能力的实现保留合法原路径。
外部写资格仍对应精确原节点和所选artifact，先支持纯节点与一个外部写节点，
GDN和history各保独立写区域。所有目标与源及保留快照不重叠，仍走原接纳、Completion、
真实退休及联合发布；失败和取消不覆盖旧head，不增加原`N+K×R`载荷容量。
Graph对一个指定外部写加纯节点的资格与单次／批量撤回已通过[有限组件检查](../evidence/correctness/inference-fused-writer-2026-10-26.json)，
CUDA普通浮点选行也已接通；这两项是前提，尚不等于整个索引状态region已经实现。
`graph::slots::Schema`只保几何，`Store`描述原码写效果；`SlotBindings`按实际源／目标数量
预分配元数据、整批检查后才移交原writer，并保存generation。它不执行或发布状态，
已通过[6组资源事务检查](../evidence/correctness/inference-slot-bindings-2026-10-27.json)。
实际Store region持有该事务，避免另占KV已使用的Invocation资源payload。
普通CPU／CUDA路线冷装配fallback，按原Queue逐真实writer复制原码；小I32索引由实际租约地址生成，
借用输入载体与其真实只读控制权限分开登记。准备和seal核对原父Invocation的写权限，
reset后保留事务不代表可继续执行；真实退休后可以take回收owner，不将回收当作计算成功。
[13项有限检查](../evidence/correctness/inference-slot-store-2026-10-27.json)包括5个实际Store检查与8个相邻合同。
这条fallback不支持capture，索引上传可能等待；引擎按同一普通图与事务接线，是否融合由冷规划决定。
原生融合用于消除状态物化和逐层metadata等待，
不能拿这个正确性入口代替性能实现。共同绑定与退休应继续复用，不能在每个Provider再复制一套事务。
融合沿用`slots::prepare/bind/take`入口和唯一的内部槽绑定owner，
fallback与融合region只各持一个实例；selected-artifact冷扩展交回该实例，不新增热分派表。
当前CUDA冷匹配从原Store回溯同item的GDN和Gather，只消解无逃逸读者的稠密状态，Y可有外部读者。
原生Recipe以显式effect roots保留副作用：即使Y不逃出，也在原ggml graph中执行状态写入，
不能用伪公开输出维持活性。源ID与目的ID共用一次小metadata上传，保持各自原真实权限；
绑定失败回滚本次权限后保留调用方事务，换载体前销毁原录制。
[有限集成检查](../evidence/correctness/inference-indexed-gdn-graph-2026-10-27.json)已覆盖精确五节点链、
Y发布与仅写状态、reset撤权回收及原13项Store／投影合同；实际选择一个融合region，
候选状态与原稠密GDN逐位一致，源槽、未选槽及padding不变。普通与MTP共享同一K参数，
但这不代表MTP接受／回滚或完整引擎已经接通。带逃逸状态／别名、其它图形及跨后端读绑定另计。
conv复用相同Artifact、Invocation、StoreBinding与退休回调，不另建执行器或公开模型专用算子。
[7项有限GPU检查](../evidence/correctness/inference-indexed-conv-graph-2026-10-27.json)覆盖conv与原GDN回归，
包括更换目标ID后的真实录制复用；未代签引擎接受／回滚或完整吞吐。
共同层按真实ExecutionSource、Queue与resource分组固定小索引载体，各Store保留独立切片和原事务。
整组绑定先验全部实际租约，再一次上传该组控制；配置及权限全部成功后才无失败地移交所有owner。
拒绝时撤回本次已安装的写效果和源投影，不消费调用方事务。原生路线仍用真实ExecutionSource，
不伪造Compute Queue。组内尚有未取回事务时不得重写共享载体，各组不隐式互相重绑。
取回的SlotBindings可clear旧source／候选后refill，保留原layout、计数、预算与元数据容量，
仍核完整真实租约及generation，不缓存裸槽号代替权限。引擎现按layer／状态角色保留原事务和元数据容量，
每步先准备完整新租约，再共同绑定；已有完整CUDA普通／MTP四臂窗口，其它工况仍待检查。
共同机制已有[有限CPU／CUDA检查](../evidence/correctness/inference-batched-slot-controls-2026-10-27.json)，
覆盖同组全体绑定、固定切片、未全部取回时拒绝重写及晚期权限失败回滚。
原生copy仍可有每组临时元数据分配，不把事务refill零分配推广为所有后端bind零分配。

[引擎接线检查](../evidence/correctness/inference-engine-slot-state-2026-10-27.json)覆盖CPU／CUDA旧快照、状态复用、镜像及真实Qwen前缀；
后续[MTP异成员检查](../evidence/correctness/inference-mtp-batch-slots-2026-10-27.json)覆盖B2/K3的不同接受长度与独立目的槽，
[完整CUDA性能窗口](../evidence/performance/inference-indexed-current-window-2026-10-27.json)另报普通／MTP四项对照。
各结果只支持其实际制品、工况和镜像，不代签其它平台或自然生成任务质量。
引擎的history与递归状态分别绑定原池布局，不能复用对方的slot ID。普通K1选最终边界；
MTP按batch-major `[B,K]`选接受前缀的原writer，未接受候选不公开。成功路径先保留所有普通输出，
再由原whole-Graph成功见证取回KV append，最后取回状态Store并发布候选；
`slots::take`只证明真实退休，不替代计算成功。清空取回事务中的旧source后才进入原联合head发布，
失败仍沿原退休、回收和旧head保留合同。控制载体归执行预算，状态载荷仍归原缓存预算。

拒绝融合时仍须由实际reader证明合法投影，不能让模型按Provider分叉构图或伪造整池ReadLease。
可移植接线由冷准备关联原Graph的Store与源Gather，不包含Provider身份；
共同层核checked axis0读取、池几何和私有ID，再由实际选中实现证明只读选中行。
跨region时还须在准备／seal核原父Invocation实际绑定的池与源ID载体，
不能只保留合法guards、却允许调用方替换索引读取未授权槽。融合与fallback复用同一事务和退休回收。
Apple／Vulkan现已沿原ExecutionSource装配Store：共同事务持有实际source，只有共享Compute路线才需要Queue，
私有索引由该source原冷传输能力准备，Store沿原FIFO／Completion和真实退休回收，不新增执行器。
Apple原类型选行与Vulkan合格原码选行保留实际stride，不复制整池；资格仍以实际lowering及全部读者核验为准。
[有限检查](../evidence/correctness/inference-native-slot-store-2026-10-27.json)覆盖两原生路线各3项及共同层11项回归，
不代表完整引擎提速。Apple现复用原CopyRecording按全部实际范围冷准备、FIFO解析成功producer后先验：
对齐范围仍走原blit，合法的非对齐范围精确host copy；不重算、不扩写padding，原Shared门及退休机制保持。
[5项有限检查](../evidence/correctness/inference-apple-slot-byte-ranges-2026-10-27.json)覆盖2／6字节生成状态及原回归。
Vulkan的整行与pitch均4字节对齐时，原lowering已可按I32字别名选行并保持F16／BF16原码，
不会经浮点数值转换。Store的精确偶数字节复制现沿原Run使用原码16位shader，4字节对齐范围仍走transfer；
先验全部真实范围、feature与dispatch，Completion通过原Run保活pipeline，不能依赖外壳仍在。
[五项有限检查](../evidence/correctness/inference-vulkan-halfword-store-2026-10-27.json)分属两实际镜像，包含2／6字节及外壳先释放。
非4字节行或pitch现有[原码16位Gather](../evidence/correctness/inference-vulkan-raw16-gather-2026-10-27.json)：
原checked-index先生成安全ID，实际uint16 shader只读选中行，精确描述符资格来自所选pipeline，不按数学dtype猜。
BF16／F16部分源租约到Store、越界拒绝及原word路径已有有限检查；SPIR-V不含浮点转换。
同步I/O现对合法偶数origin／长度剥离至多两个2字节边缘，中段保留原1MiB staging快路。
边缘复用原16位CopyRecording，独立8B逻辑device scratch先全初始化，不读写调用者的相邻字节。
两边缘在中段读写前一起预检；所有失败出口撤回两份准备，未知退休继续由原Runtime标错并保活实际Run。
独立scratch资源不反向retain父MemoryState，避免控制器所有权环；原aligned与whole-owned私有padding路径保留。
[三组有限检查](../evidence/correctness/inference-vulkan-halfword-io-2026-10-27.json)覆盖2／6字节、跨staging上限和释放wrapper后的短租约，
没有设备故障注入结论。odd partial仍拒绝，通用ready租约导入门也未放宽，不能推广为任意字节I/O或导入。
[完整tiny消费者](../evidence/correctness/inference-vulkan-engine-slots-2026-10-27.json)另通过普通与MTP K3入口，
实际验证候选／拒绝修复与资源收尾；该制品接受草稿数为零，未覆盖正向多token接受。
CPU默认路线对plain非稠密及F16／BF16 Gather复用原Native选行，按真实stride直接搬原码，
不依赖模型或硬件白名单；稠密F32／I32及encoded路线保留原选择。
[11项有限检查](../evidence/correctness/inference-cpu-storage-gather-2026-10-27.json)覆盖普通Compute／Graph、默认槽投影及相邻回归，
不是单独强选Native；显式ggml的带padding读取未因此取得投影资格。
history接线保留CastF32之前原精度samples的逐前缀窗口；它与F32 GDN各用自己的池和槽号。
私有CUDA索引GDN沿原内核的F32运算体寻址：pool保留真实`[S,S,H,P]`及槽stride，
源为I32`[B]`，目的为batch-major I32`[B,K]`，K>1写最后K个边界的时间升序状态。
它保留原op／tensor／backend及旧fused-cache布局，冷接线须核真实native契约符号；
元数据检查不把任意device IDs变成权限。[三组原生逐位检查](../evidence/correctness/inference-indexed-gdn-native-2026-10-27.json)
覆盖K1、T>K和BF16载荷，包含旧融合预检遗漏的失败及修正；不代签Graph接线或吞吐。
原生工作真实退休后，集成层可用`release_read_projection`只撤回指定ready容量及其guard和透明alias，
使旧源槽有机会重新取得写权；其它绑定、payload与pending来源不被重置。
原生导入沿原反向释放链清理，只有精确reader和实际backend同时证明时才保留录制。
[11项有限检查](../evidence/correctness/inference-projected-read-release-2026-10-27.json)覆盖两类投影、失败／关闭、alias、录制资格及原unbind合同。

Qwen活动会话现持有不可拷贝的`WorkingState`。`head`只借读当前完整边界；保留旧边界必须在空闲时
显式调用`Session::checkpoint()`，把原独占`State`转为`SnapshotRef`，不复制载荷或`VersionRef`。
冻结本身不改地址和epoch；分配失败保留原边界。已冻结的对象永不变更或回收为候选。
未冻结的旧元数据在原生读者撤回、完整成功后清空租约／页权／版本再复用，通常轮换两份主机元数据；
这两份不是GPU状态双缓冲。候选仍取得原有池权限，失败和取消保留旧提交状态。

expected-base及可追加尾权同时核对epoch，避免元数据地址轮换造成身份复用。采样恢复仍从冻结的
canonical状态派生pending／RNG wrapper，不覆盖其原预测。前缀、fork、旧读者和未知退休继续持有真实
权限与预算。KV帧reframe现由下文工作帧及变化页事务替代；GDN／conv现通过原池Store选槽发布，
不再逐层导出普通Graph状态输出。实际融合与捕获复用仍须检查所选artifact，不能从主机元数据复用推导吞吐收益。[有限主机、CPU 与 CUDA 普通／MTP 对照](../evidence/correctness/inference-working-state-2026-10-26.json)
记录了这次迁移的实际检查范围和保留边界。

GDN／conv原池状态用共享R容量的FIFO保留原`WriteLease`：只在完整成功、原生
读者撤回后，从独占的旧工作态调用`try_write`回收，保留原generation和地址。未冻结的普通K1
可轮换两块原状态；共享旧Snapshot、失败候选及未知退休均不回收。decode、冷创建和镜像恢复
消费同一缓存，不增加原N+K×R载荷容量；缓存不是额外不可用的占位。此项不改变KV帧发布、
原生索引计算或MTP接受前缀合同；最初的caller-owned输出检查见[回收与定位](../evidence/correctness/inference-reusable-state-2026-10-26.json)。

同一Graph的多个append可用`batch::take_appends`一次取回：先检查全部原节点、目的及真实退休，
再按原writer／reader资格撤回权限、压缩一次匿名绑定表，最后移出原事务。完整成功事实仍只
标记一次；失败和取消不授予ready页。部分取回后仍可用单项接口取其它原事务。普通、MTP target
和独立NextN共用此接口；它不改变`ready()`或快照布局。有限检查及直接A/B见
[批量取回](../evidence/performance/inference-bulk-append-2026-10-26.json)，未证明吞吐收益。

KV持续工作帧由Compute在原arena／页权限上维护工作布局、guard表及epoch；
`PagedArenaRead`保持永久不可变，`PagedArenaFrame`显式区分可推进工作态与封存态。
Graph直接接收各会话的footprint，保留本次执行需要的真实页租约；原append成功退休后只导出变化页。
引擎先为各接受布局准备ticket，并通过ticket的proposed view检查完整MTP续算状态；
K/V、GDN、采样、ledger和所有expected-base全部预检后，才在无失败发布区移交原工作帧、
提交ticket并交换会话head。可变基态不提前搬空；冻结基态在准备时显式fork实际guard元数据。

checkpoint先检查整个集合可封存，再分配共享控制块；分配失败保持原State、VersionRef及工作帧，
成功后无分配地永久seal各帧，再公开只读Snapshot。前缀共享、分叉和换出使用封存帧，
不按引用计数重新解冻。布局的唯一页与行界限由同一版本共享，稳态只替换变化guard，
页集合改变才准备完整替代表。失败不推进旧边界，未知退休继续保活；
MTP接受较短前缀时不得倒退已发布tail，仍丢弃后缀写权并COW。
这些机制复用原页池、tail、Completion及Graph权限，每步真实页权限接纳仍不可省略。
接口与控制边界见[工作帧指南](../user/batch-attention.md#连续追加的工作帧)，实际检查范围见进度看板。

自然生成入口现将 `--prefill-chunk-tokens` 接到已有真实分块原语：冷形状、ledger及dispatch元数据、
最大query行数和实际 `prepare_generation_call` 使用同一量子。冷键区分中间StateOnly与末块EmitNext；
每块返回原RequestLoop，不在模型内部循环独占完整prompt。无界或未计划prefix尾长仍走原有界exact接纳，
后续turn的pending只进入首块一次，取消／重建保留真实分块和采样边界。
[有限CLI检查](../evidence/correctness/inference-prefill-chunk-cli-2026-10-27.json)覆盖三条CUDA旅程及参数拒绝；
当前仅普通自然生成，不是MTP分块、全模型varlen或设备执行预热。

分片事实归共同层`PrefillPolicy`：`next(total, consumed)`返回原输入起点、实际行数及末块标志，
`parts`计算剩余片数；零量子／越界游标明确拒绝，整前缀命中返回零行且完成，计数避免SIZE_MAX加法溢出。
冷形状、ledger容量、query上界、modifier输出预检、prefix后缀及实际切片共用它。
模型适配仍负责pending入turn、prefix查找、阶段／completion映射和成功后推进原游标；共同策略只保存量子，
不拥有会话或第二份进度。普通和MTP保留各自实际可支持的分块边界。
有限量子的尾块按不超过真实剩余行数的二次幂分解；默认无界行为不变，MTP继续拒绝有限量子。
`parts`是精确片数但不单调（量子8的8行一片、7行三片），未知prefix的容量使用`parts_bound`。
前缀启用时，`for_each_shape`将已知输入范围内可能的真实尾块行数／completion与成员支持并入原Family计划，
仍按批宽与采样配置生成exact child，不扩展递归算子的padding语义，也不另建形状或预热注册表。
额外小块可能增加episode和提交次数，合法数学形状的舍入仍可不同；这一政策本身不构成吞吐或质量结论。
[有限尾块检查](../evidence/correctness/inference-bounded-prefill-2026-10-27.json)覆盖六组主机合同和一次真实生成后前缀命中，
剩余T2以B2执行、ready后新增exact family为零；首次N4旅程未命中目标覆盖，保留记录后仅将驻留数改为N2。
[有限检查](../evidence/correctness/inference-prefill-facts-2026-10-27.json)覆盖四组纯主机边界与原Q2／B2多轮CUDA非时间报告一致，
不把这一算法抽离记作吞吐提升或其它平台运行结果。

`Model::prepare_execution`可在空闲且缓存已配置时，通过原有有界接纳准备新增普通／MTP-prefill后缀；
不会预先接纳Session或另造编译缓存。额外exact容量在初始冷计划结束时按完整键数冻结（至少1），
与会话数量N解耦；仍用原FIFO、真实drain和资源预算，动态未见形状可能在请求边界准备并淘汰旧额外family。
`--warmup-shapes planned`只遍历原`planned_slot_uses`中有实际用途的完整形状键，在ready前调用同一准备入口；
默认lazy保留按需载体准备。它不提交设备工作、不推进状态或RNG；有限量子与前缀启用时覆盖已知范围内的可能尾块，
但不保证所有未来输入形状已准备，也不代表capture已录制。
所有调用计划先冻结再创建载体，不复制一份预热注册表。额外载体和输入暂存的真实分配可能增加启动内存，
执行预算与会话cache预算仍分别生效，不能把`--cache-bytes`宣称为整个Model显存上限。
[有限CLI检查](../evidence/correctness/inference-warmup-shapes-2026-10-27.json)覆盖普通B2多轮lazy／planned非时间报告一致、MTP K2、fixed及非法policy拒绝；
不把这些短进程的墙时当作吞吐证据。
新`shape_admission_seconds`记录该次同步接纳并计入`step_seconds`；后者从成功输入预检后计到主机发布完成，
不含输入预检、初始加载和失败接纳。原`seconds`仍是提交到完成的区间，不把CUDA Graph重录耗时当新形状编译延迟。
长度分桶、设备执行预热和异长合批仍需分别接入；已有变长页表能力不能自动消除Qwen当前等T密集计算的约束。

会话淘汰后的完整恢复当前仍可重算已消费输入；已有页复制／传输不等于完整会话换出。
主机状态镜像涵盖KV、递归／卷积状态、pending、RNG和来源配置，分别计量主机／设备预算，
保留共享页、旧版本及真实退休，恢复时不复用失效设备指针。此项属于底层状态能力，HTTP不是前置条件。
镜像还须分别保存canonical已计算预测与请求自己的历史pending／RNG，保留采样wrapper的语义。
换出使用原CopyBatch、租约及Completion；完整复制成功并真实退休后才发布镜像、退役原Session。
换入在原池中重建真实PageRef及完整State，全部成功后安装新head，不重放输入或重新采样。
当前状态bank是固定backing；释放一个会话只归还槽，不保证显存占用下降。仍被旧快照保留的页和版本
继续计费；真正缩减设备backing需要单独的池容量机制，不能由逻辑换出推导。

`PagedArenaRead`与`PagedRowCacheRead`现提供`page_prefix(physical_page)`，供原CopyBatch读取不可变frame
实际覆盖的整行前缀。返回原ReadLease的裁剪子权限，保活原backing，不暴露容量尾部，也不会随后续append扩大。
它是搬运源，不自动成为可重新交给arena.read的整页guard；换入仍经原writer与publish路径取得新权限。
单页查询是冷路径线性扫描，不在每token路径增加页表。完整镜像用新增`page_prefixes(output)`，
按不可变layout的逻辑引用顺序导出每个前缀，重复物理页分别受各请求自己的末页长度限制。
它先检查全部输出再替换句柄，无分配，按原已排序guard二分查找，复杂度O(P log U)，避免逐页扫描的平方成本。

`HostStateStorage`只保原Backend身份下的独立主机资源；CPU／CUDA沿用原`StorageOwner`，
CUDA使用pinned host，Apple使用独立shared Metal资源，Vulkan使用原Runtime的staging资源。
原生身份token只防跨Backend及地址复用，不保活Backend或Model；Vulkan资源保留其设备所需Runtime。
复用`ResourceBudget`限制
实际托管backing及原池／搬运已纳入预算的元数据字节，并独立限制镜像数量。逻辑描述等标准容器元数据
仍由模型上下文和镜像数约束，不冒充完整进程内存上限。镜像只保存逻辑层／页、完整有效字节和弱模型身份，
不保权重、物理页权限、原VersionRef或执行器。借用payload的搬运必须同时保活完整镜像。
CUDA pinned host不承诺CPU／GPU并发coherence；原复制成功并退休之后才发布ready，通过资源自己的
read／map接口读字节，不能把额外的`HostDeviceCoherent`属性误当成完成后可读的必要条件。
`StateTransfer`使用原CopyBatch及Completion接管临时源／目的资源，成功wait或实际退休的poll后才能取回完整原writer；
未知退休时不归还容量，不把错误当作退休。Apple／Vulkan复用原typed Program／Invocation／Update，
不伪造common Queue；Apple的退休查询与附件共享原观察锁。临时scope也将实际分配器owner保留到
共享控制块释放结束。模型adapter负责完整清单、新页布局和wrapper恢复。
搬运与hidden-output副本通过公开`CopyBatchUpdate`附加保活，不再借出原Completion或由引擎操作Apple观察锁。
common搬运直接用原Context准备Program／Queue，原对象已保留Context状态；Queue wrapper仍固定在scope中直到真实退休。
`poll`分别返回状态和真实退休事实，暂时pending不缓存成永久失败；确认退休后重新取得原Update的稳定终态。
`request_cancel`只转发原端点能力，失败或不支持不替换计算结果，也不授予提前回收权限。
关闭CopyBatch禁止新接纳，但仍可取回原已成功退休的writer。模型当前的换出／换入调用仍同步，
这些底层能力不冒充已接通异步会话腾位。
共同请求adapter可显式选择主机镜像恢复；未选时仍按原已消费token重建。CPU／CUDA普通与MTP已作有限消费者检查，
CUDA另核lag1／prior完整往返与原catchup续算；[检查范围](../evidence/correctness/inference-host-image-lag1-2026-10-26.json)
不等于完整任务质量或性能结论。Apple／Vulkan的原生字节合同与原335角色完整普通／MTP／lag1消费者
均已作[有限检查](../evidence/correctness/inference-native-host-image-2026-10-26.json)。原tiny Vulkan流对齐问题
已由[显式初始化载体](../evidence/correctness/inference-stream-padding-2026-10-26.json)修复；受控退休与真实设备故障按各自记录区分。

`SessionCache` 的逻辑驻留目录已归入 `engine/inference`，不再依赖 Qwen 或 benchmark 命名空间。
它只管理独占 pin、空闲 LRU、撤回和原 owner 的退休回调；`PrefixDirectory` 管前缀索引，
模型适配器保证完整状态，原状态池管理物理版本。三个容量与寿命不同，不合成第二套缓存所有者。

模型给出数学，Graph 根据真实能力冷选 Provider 和等价融合。模型驱动无需逐算子指定
实现；新增后端融合不应要求修改请求循环。共享机制也不能反向依赖 Qwen 的字段或平台宏。

CPU线程和executor配置在冷建Backend时传入原`compute::cpu::DefaultContextOptions`，CPU主端点和显式CPU辅助端点共用原工厂。未传配置仍用原默认值；不隐式创建CPU端点，不修改全局线程设置。Model只接收已准备的Backend，不参与线程池配置。

一个CPU Provider Context拥有一个延迟创建的原生工作团队，派生Program和执行槽共同保活；每槽仍独立拥有graph、plan、workspace及取消状态。同一团队的完整原生compute串行执行，不随Invocation数复制线程池；不同Context各自独立。团队沿用原生默认的有限轮询等待，空闲超过轮询范围后休眠。执行器分发时不持团队锁，同线程重入在触及原生池前拒绝。Context关闭仍禁止新提交，已派生内部资源按原退休规则释放。外部allocator／executor须活到全部派生制品和原生工作退休，executor并发度与`native_threads`是不同预算。[有限并发、取消、分配失败及真实模型对照](../evidence/performance/inference-cpu-context-team-2026-10-26.json)支持资源调整；[等待策略对照](../evidence/performance/inference-cpu-polling-screen-2026-10-26.json)分别报告吞吐与CPU时间代价，不代表跨平台吞吐或所有并行负载均已测定。

默认executor为空，回调在调用线程执行，仍可显式配置多线程原生团队。借用executor用于表达调用者需要的执行位置和分工；库必须遵守该回调，不能擅自绕过。应用自己的Scheduler并不意味着每个完整原生程序都应多一次工作线程交接；[调用线程配置对照](../evidence/performance/inference-cpu-caller-executor-screen-2026-10-26.json)说明这项选择有实际成本，不能从executor并发度直接推导性能。

CPU原生region冷发现允许无副作用、无控制依赖且未发布的常量View链连接数学节点，避免独立的权重视图切断已有聚合。View不进入原生数学成员，仍沿原Graph的alias、offset、租约和拓扑排序执行；动态、发布及受控View保留原边界。聚合保留每节点声明的存储精度，不删除BF16舍入点。[有限数值与模型对照](../evidence/performance/inference-cpu-constant-view-regions-2026-10-26.json)支持该规则的资源收益，不承诺聚合节点越多就越快。

Qwen实现内部的`create_execution(..., cpu_options=nullptr)`继续使用原有限workspace预算公式，
再把可选配置转发给CPU主端点或显式CPU辅助端点；省略参数保持原构造调用。
例如实现内可用`create_execution(geometry, plan, cpu_layers, selection, &cpu_options)`，
随后将原Backend移入Model；这不是已安装SDK导出的公共引擎API。只有GPU的选择会沿原Backend入口拒绝CPU配置，
GPU-only构建只需配置类型的前置声明。`native_threads`不配置ATen的共享线程，
后者仍须使用原显式启动接口；没有新增全局设置或自动核心数策略。

Qwen的冷制品准备已独立为`weights.h`／`weights.cc`：原Tensor存储、LoadedWeight和冷加载／投影绑定接口留在模型内，Recipe只消费绑定结果，加载器不依赖Model、Session或请求循环。绑定仍借稳定map节点，存储仍归原ColdResources，不复制权重银行或新增owner。Tensor的热读写／view／move保原inline；移出的freeze仅在冷导入和常量构图调用。同一weights源由每个实际消费者按自己的后端能力宏编译，避免把完整产品的Backend布局强加给Vulkan等factor消费者；没有新增用户构建选项。拆分后的原CPU请求流程及[工程构建检查](../evidence/correctness/inference-cold-weights-split-2026-10-25.json)已通过，不据此声称吞吐提升。

流式权重的文件范围与TensorDesc保留逻辑长度L；载体按四字节补齐为P，实际分配和背书预算计P。
F3只读L并初始化尾部；`output()`返回L，`initialized_storage()`返回明确完成的P，二者共享原租约。
`ConstTensorView::from_initialized_storage`显式保留这段只读范围，旧`create`仍裁L；Graph常量和不可变别名
只传递原授权范围，普通数学复制仍取L。更大的根分配本身不构成读取尾部的权限。

KDA／Kimi NoPE MLA 组件的参数准备与数学追加已分开：调用者提供真实参数 Value，
同一个 append 返回原端口；测试权重工厂也调用这一个数学体。来源变换、只读租约及完整状态发布
仍归调用者，append 不创建另一份 owner 或状态表。普通 BF16 投影与 F32／量化 MatMul 的
逻辑布局分别检查，不在模型数学内按仓库名选择。
[host 接线与原 CUDA 状态旅程](../evidence/correctness/kimi-supplied-parameters-2026-10-25.json)
使用原几何的合成权重；随后[原23角色 CUDA 组件](../evidence/correctness/kimi-original-attention-state-2026-10-25.json)
也已完成两步数值及完整5资源＋Epoch提交、旧读者、分叉、过期拒绝与未发布候选取消。
它使用显式 llama/config QK epsilon `1e-5` 合同；已查 vLLM reader 为 `1e-6`，不能宣称两者等价。
输入为标明的上游 fixture，KDA0 到 MLA3 跳过中间层；不代表完整模型、质量或吞吐通过。

## 模型、会话、请求、批次是不同对象

| 对象 | 内容及寿命 |
|---|---|
| 已加载模型 | 固定来源、数学／精度、放置与容量合同；共享权重、冷图、程序和存储作用域 |
| 会话 | 可续算的完整状态头、位置、pending 与采样游标；请求结束后仍可保留或重建 |
| 请求／turn | 本次输入、输出额度、停止与取消政策、主机观察进度；不各自复制模型 |
| 执行 episode | 本次实际参与者、输入／输出权限、候选状态、原 Completion 和提交见证；退休后归还槽 |

`N` 限制驻留状态版本，`R` 限制一次真实 batch 的活动序列数；请求总数可以大于 N，
通过等待、空闲淘汰和恢复推进。目录与会话别名共享一个实际版本时不能重复扣费。
当前 Qwen 驱动也把驻留 Session 数限制在 N 内；用户数量不是底层硬件容量的同义词。
字节预算另按真实 backing／metadata 计费，不能由 N／R 推定全进程或物理显存上限。

当前一个 Model 只有一个在途原子 batch 和串行 carrier；R2 是同图真实批处理，
不表示两个独立 Invocation 同时执行。增加在途数需要真实独立槽和权限支持，不能改报告便算支持。
同一 Model、Session 及缓存目录的控制操作由一个 owner 串行调用；现有在途守卫不是线程同步。
多用户不意味着用户线程可以并发 prepare／stop／finish。跨线程取消应交由该 owner 执行，
不能直接修改会话字段，也不为这一合同在每 token 路径增加锁。

请求控制读取已提交边界，不再保存一份可独立推进的 pending／RNG。`RequestProgress`
只持有主机进度与新请求的初始采样身份；实际游标、已消费输入和 pending 必须由模型的
同一发布事务推进。模型程序和权重跨会话共享，状态写权限、取消及输出观察按实际参与者区分。

请求的 turn 合同借自冷准备的不可变输入，最大输出量独立于 vector capacity。共享进度只借
原输入缓冲，将真实已提交 pending 放入下一 turn 的唯一首槽；剩余用户 suffix 不另复制。
已提交输出按连续序号观察一次，额度／停止和 turn 推进共用，缺号及容量错误在主机变更前拒绝。
完整缓存命中的输出不伪造 Graph 提交；导入历史只接纳真实已提交记录，不重新发射旧输出。
模型仍负责把自身 ledger／StateHead 投影为这些事实，不把模型修复或随机游标交给请求进度。
Qwen 的自然 `GenerationRequest` 统一要求冷准备原提交账簿，已删除另行拼结果的无账簿分支；
直接使用 raw Model／Session 不承担请求进度的接口仍按原合同工作。批成员先全部预检，
提交后日志失败也从原账簿观察两请求的输出、游标及报告事实，不能重新消费输入。
[有限 CPU 检查](../evidence/correctness/inference-request-ledger-2026-10-25.json)已覆盖真实 R2 晚日志失败和原 pending 续算。
批成员在容量接纳和前缀淘汰前按实际Session去重，不按共享快照去重。
[重复Session反例](../evidence/correctness/inference-duplicate-session-2026-10-25.json)证明旧路径会先淘汰无关前缀；
前置拒绝后缓存、账簿、head及RNG保持原样，同请求可直接重试。

自然请求在首次接纳一个 turn 时，由模型根据实际 head 长度检查完整输入及该 turn 剩余额度减一能否放入上下文。
输入已包含应消费的历史 pending；末枚输出仍是 pending，不额外占一行。此检查先于 Session／前缀导入，
分块输入只检查整轮一次，不能在中间块提交后再次扣除原完整输入。共同 `RequestProgress` 仍只管理主机额度与观察。
[有限 CPU 对照](../evidence/correctness/inference-turn-context-admission-2026-10-25.json)复现旧路径的部分提交后晚拒绝，
覆盖新提前拒绝、恰好满上下文及仅 pending 的末行续轮；满前缀、分块及极大整数边界本轮仅源码复核。

前缀命中的历史、已消费偏移和 Session 只在实际资源接纳及 frontier 成功后一起发布；
先预检原账簿与进度，再局部准备，最后写入原预留记录。提交前资源拒绝后可重试同一请求，
不重建账簿或重置随机流；turn 启动、计时和目录 LRU 仍可保留幂等准备的变化。
[有限 CPU 检查](../evidence/correctness/inference-prefix-admission-retry-2026-10-25.json)覆盖真实字节门拒绝、
同请求重试与无拒绝对照，以及原快照／随机游标保留；不代签其它 native 路线。
共享 frontier 的可写输入也须先检查原退休保留节点；到 submit 才拒绝，不能保护前面的 identity 写入。
[提前门禁检查](../evidence/correctness/inference-frontier-retirement-gate-2026-10-25.json)保留旧 owner；
host 用例覆盖 pending 时不写入，未注入原生设备故障。

当前有限 CLI 接收 schedule，按 FIFO 接纳，在完整 turn 结束后让出；已接纳的 turn 逐波推进。
它预留全部输入和输出账簿，N／R／缓存字节预算不限制等待输入或主机输出积压。
未来在线服务须在请求层明确排队请求数、输入字节和未交付输出上限及背压／公平策略；
底层按真实状态容量、执行槽和退休事实接纳，不暗中扩容，也不承诺当前没有的等待时延保证。

## 共享请求能力与边界

普通推理、多会话和多任务使用以下已有共同机制，模型消费者的实际依据和有限边界见下文。
引擎选择排队、公平性和交付政策；网络协议、聊天应用和通用产品管理不在本轮。
接口存在不代表所有模型、数值或后端组合已运行；有限CLI未暴露某种政策也不等于缺少底层原语。

| 共同机制 | 现有实现与所有权 |
|---|---|
| 有界运行中接纳 | `RequestWindow`固定槽和FIFO，世代拒绝旧身份；owner先准备payload，再在合法波次边界接纳。`RequestProgress`借用稳定输入，open turns按turn数、累计prompt token及输出额度接纳；传输字节和输入所有权仍归调用者 |
| 请求级取消与失败 | 等待撤回不触及设备；已提交普通多成员批次可只丢弃停止成员，幸存者提交原候选。其它模式或明确放弃整批沿原中止／退休；已定位参与者失败与设备失败分别处理，交付失败保留已提交事实 |
| 输出消费与背压 | `OutputDelivery`的credit／ACK与提交观察分开。提交前检查episode最大输出；满时由原窗口暂停该请求，让其它就绪任务推进。owner保留未交付输出，异步交接不让`CompletedEvaluationView`／原生租约逃逸 |
| 有界prefill工作量 | 原`TextCompletion`表达state-only中间块与最终输出；ledger保真正已消费偏移，中间块不投影词表、发token或消费RNG。原窗口的队尾重排允许其它ready decode推进，quantum由owner选择 |
| 多Model资源与调度 | 每个Model保原端点、队列和单在途合同；共用`ResourceBudget`按指定真实资源行接纳已知权重／缓存／workspace。未知私有分配仍标未知；跨Model优先级归owner，不把共享计数称为物理显存上限 |

这些机制保持请求政策、模型数学和后端执行的边界；后续新增需求仍先接真实消费者，再提取确有重复的内容。
不增加第二套状态所有者、缓存索引、执行器或大回调接口。chunked prefill 改变工作分段与浮点路径，
需要独立的数值／状态检查；吞吐、首 token 和并发等待时间分别比较，不把公平性变化冒充同工作量加速。

同一个 `RequestWindow` 使用固定容量FIFO和请求槽身份：空间随同时占用的槽数变化，
不再随历史 turn 数增长。`enqueue` 可在运行中使用空槽；每次重新占用取得新世代，冷重置也不会复活旧身份。
排队撤回只将该请求标为 Retiring；终态请求经原资源实际退休、owner 显式 `release` 后，槽才可复用。
Qwen 有限消费者和 GLM 原组件消费者已迁移到该合同；真实生命周期检查及其范围见下文。

普通分块 prefill 沿原 `Text` 图族增加冷输出模式 `TextCompletion`，不增加推理阶段或执行器。
中间块提交完整状态，pending 为 -1，不构造词表投影、采样或主机 token 读回；最后一块才生成下一 token。
输入偏移只在真实 ledger 提交被观察后推进。同一请求通过原窗口重新排到队尾，让就绪 decode 有机会执行。
历史记录保留每块的输出模式，恢复时按原分段重放；只有具有有效 pending 的完整前缀可以发布到缓存目录。
中途因调度量或输出背压暂停的请求可以淘汰 Session，再用原 key 唤醒；恢复只重建模型状态，
不重新开始 turn、消费输入、观察输出或重置交付游标。完整 turn 的旧检查点必须覆盖当前已提交 ledger，
否则沿原重建路径恢复；同 key 的目录命中也须恢复该请求自己的 pending／RNG，并核对完整物理边界。
[有限 CPU 旅程](../evidence/correctness/inference-sampling-and-midturn-resume-2026-10-25.json)
覆盖真实淘汰、其它请求推进、状态块及未 ACK 输出后的恢复；其它状态家族与 CUDA 调用未在本批重跑。
[原 BF16 P154／Q64 检查](../evidence/correctness/qwen-ordinary-chunked-prefill-2026-10-25.json)
覆盖状态恢复、取消后重试和 pending-only 续轮；目前是有限底层消费者，不宣称 CLI 已提供调度量选项或性能收益。

[两个 Model 的实际寿命检查](../evidence/correctness/qwen-dual-model-lifetime-2026-10-25.json)
确认同一设备上的独立 Hyle Backend／队列／资源可分别退休：A 销毁后 B 的原已提交工作和新续算继续。
CUDA 实际共享 primary context；这个合同依赖各自原资源与 Completion，不能把销毁一个 Model 实现为设备 reset。
这不提供跨 Model 的物理显存总额上限或公平调度；现有 `ResourceBudget` 仍按真实 resource 计量。

已知字节的共用接纳沿一个 `ResourceBudget` scope 扩展：指定多个真实资源行共享总额，
其副本沿原分配、Graph 与租约传播，保留同 scope 的 backing 去重。不会自动合并独立预算，
也不把无资源归属的私有存储归到某个设备。验证消费者已可显式移入原 Backend 和共用 scope：
权重池先接纳再分配，原文件 pipeline、已知 Graph workspace、caller Tensor 和 readback
沿同一 scope 计量；常量沿原 ready lease 去重。默认不传 scope 时沿原分配路径。
[两个 Model 的有限检查](../evidence/correctness/inference-shared-known-byte-sources-batch1-2026-10-25.json)
已覆盖加载拒绝、A 实际退休与 B 继续；FP8／NF4 的新预算路径仅源码接线，不以此代签量化实跑。
原 `--cache-bytes` 与 cache-only 计量保留为局部约束。原 `LeasePool`／`PagedArena` 的冷创建可另传共用 scope，
在实际 backing 分配前同时通过两种限额，并沿原分配身份去重、保留到实际释放；局部 metadata 分配器向共用 scope 计费。
cache 的 schema、inventory 和完成标记在全部准备成功后一次发布，失败释放额度后可在同一个 Model 重试。
[有限检查](../evidence/correctness/compute-cache-local-and-shared-admission-2026-10-25.json)覆盖八项 host、
原 C16 双 Model CUDA 拒绝／重试及逃逸快照寿命；正式四个原层回归 TU 仅严格编译。
native 完成结果通过原 `GraphOutput::storage_lease(local, additional)` 留存，
一次取得原只读 lease，再按同一授权资源施加两种约束。成功读者及子视图保有同一导出 owner 和两份费用；
第二门拒绝时原结果及权限仍有效，第一份费用按实际导出 owner 的寿命归还：持久 Domain 可以继续保费，临时导出销毁则可退费。
先前独立的未计费导出和 Provider 私有槽不会被追溯计费，也不承诺两份计数的原子回滚。
[有限 host 检查](../evidence/correctness/inference-generated-result-dual-scope-2026-10-25.json)覆盖生成结果、拒绝后重试和别名子视图；
Model 状态／logits／hidden 的新接线仅严格编译，没有新增 Apple／Vulkan native 实跑。
这属于完成结果的留存接纳，不能预留 Provider 未来的私有分配。
原文件映射、普通 STL 及 opaque 私有存储也仍在此范围外，不能仅设置总额参数便宣称全部受控。
已接纳租约的费用须随其导出 owner 和实际退休延续；raw Buffer 转租约时
应转移同一实际费用，不能重复计账或先退费再接纳，移动赋值也须先释放旧 backing。

源精度在准备状态／图之前确定：从同一份已校验角色得到不可变 `PrecisionProfile`，
由 Recipe、缓存 schema、hidden 与读回共同使用。存储为 F16 本身不能推定计算精度；原 GPTQ 的 BF16 计算保持原合同。
NF4 的原 F16 配置沿此路径执行，GDN 递归状态及规定的累加仍为 F32。
NF4 加载器保留原六个数字操作数，JSON 只作冷期认证，offset 保原 F32 位值；不预先展开整库。
Recipe 表达原 Decode→View→MatMul，由同一冷规划器选择 CPU／CUDA packed 融合或原分解，
报告实际融合数、分解数及全图 workspace，不能把所有后端／行数称为无展开。
显式 F16 冷导入工具与驻留加载器共用原状态校验；需要展开的调用者仍承担实际空间及退休合同。
MLX affine4／8 group64 也借入原 U32 权重及 BF16 scale／bias，由既有 `MlxQuantized` 执行。
准备器保留原物理身份，将已转换的 norm gamma 和 conv 描述映射为显式参数语义；不再套 HF 的加一。
embedding 保留 BF16 乘法／加法舍入，绑定同一权重的 head 使用 F32 affine 线性计算，不能共用预展开的 embedding 表冒充两者。

## 从输入到一次完整提交

```text
接纳／恢复会话
  → 按阶段、真实行数和资源容量准备兼容 work
  → 普通 target，或 MTP 的草拟／验证
  → 等待原 Completion，捕获选中前缀的完整结果
  → 必要的模型状态修复／追赶
  → expected-base 核对，原子发布状态＋pending＋RNG＋已消费输入
  → 按提交序号观察一次，再决定完成、下一轮或归还执行槽
```

普通推理使用真实 target 图、文本权重和无 draft 的计划；它不是禁用草稿的 MTP。
MTP 的草拟数、训练块宽、Markov／confidence、target tap 和状态修复由真实模型合同承载，
不能把所有家族硬塞进 Qwen 的 NextN 循环。

共同的是输入限制、接受前缀和完整发布合同，计算阶段并非相同顺序：Qwen 递归草拟后可做
一次多行 target 验证；GLM 要围绕真实 target tap 续算 NextN／indexer；DeepSeek 有固定训练块、
Markov 和相应历史追赶。它们直接组合已有 `SamplingLimits`、`AcceptedPrefix` 和原状态事务，
保留各自 RNG 地址规则。若一层所谓“阶段协议”只转发回调或再存一份参数，它不能减少职责耦合，
不应新增；真正重复的接受／尾部算法才归入共同机制。

pending 已返回但尚未消费。只有真正消费的输入进入 KV、位置和检查点键；拒绝的候选不发布，
最后一枚输出和 bonus 仍留 pending。下一轮将上轮 pending 与新增用户输入拼接，恰好消费一次。
EOS、额度和上下文边界共同限制实际接受前缀，不先消费末尾输出再回退。

合法接受结果共用 `AcceptedPrefix`：`consumed` 包含必须消费的当前输入，
`consumed_drafts` 只计真正进入状态的草稿。零草稿接受仍选择 after-current，
不能退回消费当前输入之前的 base。greedy 和随机路径都由同一前缀规则处理 EOS／额度；
模型据此捕获完整状态及原 pending 租约，采样游标仍按实际使用的 RNG 地址推进。

所有可失败的选择、状态捕获、修复及合法性检查先完成，再以不抛异常的 head 交换发布整个 batch。
提交前失败保留旧状态和游标；提交后 trace／报告失败不能重抽样或重复返回。
提交记录是权威事实，观察序号保证幂等；异常本身不能代表“模型没有前进”。
MTP、提交前及最后一个成员停止仍走整批中止。已提交且无draft状态的普通多成员批次，
停止一个成员只撤回其发布，设备计算正常完成，幸存者不重算；原生／导出失败仍不能部分发布。
停止成员不改head／RNG，不提交其账簿、不调用成功观察函数；`EvaluationDisposition`区分未发布、已发布和已取消。
`stop()`确认请求被接纳，不能把这次成功撤回伪装成原生退休错误。等待幸存者完成期间，任一成员的退休／drain
返回尚未退休，不隐式撤销整批；原owner仍可明确中止整个episode。输出观察的局部掩码仅在撤回时准备，
普通R>2不增加每步缓冲分配。当前[有限CPU检查](../evidence/correctness/inference-review-fixes-2026-10-26.json)
覆盖greedy R2、空账簿、原候选发布／继续和全部取消；不代签GPU物理在途、随机取消或设备故障。

普通／MTP 和固定回放共用模型无关的执行观察记录。模型完成接口只接收原提交目标、事实
目的地和明确的捕获需求；名字、路径、默认打印及 trace 写入由入口处理。CLI formatter 只读取
事实并写入调用方的流，不推进 ledger、pending 或 RNG。

完成后通过静态类型的同步观察函数交接现有记录、已提交状态及必要的只读输出；原 native
lease 和 carrier guard 覆盖整个观察调用。view 不能逃逸或重入准备，也不增加一份状态所有者。
关闭诊断不捕获／读回大张量。报告查询仍在提交之后；晚 IO 失败仍按原 ledger 幂等观察，
不承诺任意文件失败后的字节级 exactly-once。冷准备的图／route 诊断仍是独立的准备期职责。

## 多会话、多请求与缓存

三层各自承担有限职责：

- **缓存目录**索引真实保存的完整状态边界，负责查找和 LRU；radix 分叉不能凭空制造中间状态。
- **模型状态存储**保留 KV 页面、conv／GDN、MLA／indexer 等实际状态及原租约；目录不复制载荷或另记一份物理容量。
- **请求接纳**根据驻留版本、活动 work、预算和真实退休决定等待、淘汰或恢复；Provider 不做用户调度。

新请求复用 evaluated state 时使用自己的 pending 选择和 RNG；恢复同一请求则保留它的历史
pending／游标。缓存键只含已消费输入及稳定的来源／数学／布局身份；采样 wrapper 共享状态，
不能继承另一请求的独占写尾页权限。

同一 Model 可在冷准备时冻结有限的 greedy／温度／top-k／top-p 配置；种子、stream 和 draw 仍属请求。
Session 借用一个稳定配置，Family／Phase 的身份包含该选择，兼容策略的请求继续真实合批。
它们共享原权重和 Backend；不动态修改全局 sampler，也不为不同策略复制 Model。
新请求完整命中另一策略产生的状态时，从实际 carry hidden 按自己的策略选择 pending；
同请求恢复按 ledger 的历史 pending／RNG 续算。Ledger 用原资源 owner 的弱身份拒绝跨 Model、
过期或地址复用，不为一份历史记录保活整份权重。
纯 greedy 单配置保留原来不保存 hidden、不另执行前沿 head 的快路；本轮多策略仅适用于普通 Text，MTP 沿原默认策略。

采样政策在冷期决定数值路径。T>1 时，合法 F32 两端先按二进制缩放再居中，避免差值溢出；
TopK 仍按原 logits 排序，避免缩放使极小不同值变成并列。T≤1 保留原图。
[六项 CPU 检查](../evidence/correctness/inference-sampling-and-midturn-resume-2026-10-25.json)
覆盖宽有限范围、巨大共同偏移、极小值排序及原无效输入政策，不代签 GPU 或完整生成质量。

完整命中可跳过模型前缀计算；部分命中只执行真实后缀。未命中恢复使用已消费输入重建，
不重新向用户返回历史输出，也不冒称跨批宽逐位重放。
显式固定 token tape 有已知未来输入，仍是独立重放载荷；它与自然请求共享状态／预算协议，
不能靠预先补自然 token 来复用静态调度。

一波先完成 lookup／接纳，再执行兼容 batch，最后发布新检查点，避免同波次请求观察顺序依赖。
按同一已准备模型、阶段、真实行数和状态合同分桶；MTP EOS 后按实际验证宽度拆组，
不 padding 到最大 R／K，也不把 q／hidden 下载到 CPU 以统一格式。

共享请求控制管理 turn 输入／额度／停止、一次提交观察、有限的 waiting／active 下标和空闲 LRU；模型适配提供
真实驻留情况、完整检查点、恢复／重建和可提交 work。缓存目录的 radix／LRU 可以复用，
其载荷仍是模型的真实检查点，不增加第二份状态所有权或通用 KV 容器。
Qwen `State::state_identity()`的canonical指针和持有`StateVersion`的`StateIdentity`不是同一种身份；
`StateStore` 的版本统计还可能包含候选与 metadata fork，不能直接当作 Qwen 的 N。
接纳按原存储的版本、字节、槽位与退休事实证明；无法证明独占替换时，淘汰后重读实际容量，
不预支退休空间。目录命中只授予保留读者，写权限必须由原状态机制取得。

公共请求控制由三项小机制组成：进度负责 turn 输入、输出额度与提交观察；索引窗口负责
FIFO 接纳、N 个活动名额和整 turn 归队；typed 目录负责 token radix、LRU 和 pin，直接保存
一个模型自有的检查点载荷。三者已接入 Qwen
普通／MTP，以及 GLM Pro 既有原角色 StateHead 消费者。GLM 检查绑定明确初态之后的有限
suffix；两请求两轮、采样 fork／历史恢复、真实续算和取消／stale 已运行，不代表任意 GLM
prompt prefill 或完整模型服务。原 StateStore 版本和读者仍按实际预算计费，窗口 N1 不是独占证明。
R 继续交现有分桶及真实执行容量处理。模型提供容量查询、命中后的 fork／恢复、实际执行和
检查点构造，不增加大 `ModelOps` 接口、热回调表或第二套调度器。

有界续轮仍由同一 `RequestProgress` 承载：输入 owner 在冷期预留稳定的 turn 存储，
`bind_open_turns` 固定 turn、用户输入 token 和总输出边界，`extend_turns` 只检查新增后缀，
不复制或每轮重扫旧输入。已暴露的 prompt／额度不可改写；关闭输入与结束一轮是不同事实。
最后已知 turn 完成但输入未关闭时保持原 request 身份；后续 turn 仍只消费一次真实 pending。
实际提前停止只返还未使用的输出预留，重复提交观察不再次返还。模型的上下文、KV 和实际
输入接纳仍归模型 owner；主机 token 边界不代替设备容量证明。
上述[开放续轮检查](../evidence/correctness/request-incremental-turns-2026-10-25.json)覆盖 host 原语，
原 Qwen 普通 CUDA 的运行中续轮和背压另由真实生命周期消费者核验；CLI 仍预读有限日程。

窗口的 `pause` 只释放活动名额，原 request key 和 payload 槽仍占用；可用输入或输出 credit
到达后，`wake` 将同一 key 放回原 FIFO 尾。未完成 turn 的 survivor 用 `requeue`，不能冒充
已完成 turn；`retire_active` 只表达 owner 已证明整批退出后的终态，不执行设备取消。
`cancel_waiting` 覆盖排队与暂停状态；取消请求、真实退休和槽位复用仍分开。
暂停且确实驻留的 head 可参与原空闲淘汰政策，但 owner 必须重读实际预算，不能假设释放
活动名额就回收设备字节。上述[状态检查](../evidence/correctness/request-window-suspension-2026-10-25.json)
已通过主机用例。

整批中止的正常终态只属于被中止的 episode。确认所有原生工作实际退休后，存活成员保留
未发布的 head、ledger、pending 和 RNG，再用原 key 归队；不能把 peer 的正常取消永久写入
它的 Session 退休错误。被停止成员仍保留取消事实；其它错误、原生错误及退休不明不能
按这一例外丢弃。[真实 Qwen 普通 CUDA 用例](../evidence/correctness/qwen-live-request-lifecycle-2026-10-25.json)
已复现并修复存活 B 最终停止时残留取消的问题，随后完成续跑、新请求接纳和完整退休。
该用例观察到提交在途，不证明取消时 GPU 物理工作尚未完成，也未注入设备错误。
随后退休返回的真实错误须能替换此前的纯取消状态；第一条真实错误仍保留。最终合并结果在
销毁 episode 前返回给调用 owner，即使当前尚未退休也不能只返回较早的取消。正式消费者的
七项主机错误合并检查已通过；未用主机合成状态冒充真实设备故障注入。

Session、Model、冷资源与逻辑缓存统一返回 `RetirementObservation { retired, status }`，
没有 bool 转换或旧兼容出口。owner 遇错仍沿原收集流程；确认退休后归还旧会话、窗口槽或
目录费用，再报告原错，失败的接纳不继续装入新任务。单纯尚未退休不写成持久故障。
等待普通部分撤回的幸存者发布时，成员`drain`不代替原owner完成或中止，返回尚未退休。
其它路径中未停止 Session 的 `drain`只驱动原执行并观察错误，不退休其head，也不固化peer的纯取消；
停止后的观察持续保留第一条真实错误。实际释放权限仍来自原 Graph／页／原生保留门。
[有限检查](../evidence/correctness/inference-retirement-observation-2026-10-25.json)已通过三组缓存
host 合同及 Windows CPU 的真实 Session／R2 准备与续算；合成状态不代签设备故障。
其余 GPU／平台消费者已按接口迁移，本轮未编跑，不据此宣布全局错误隔离完成。

引擎的 `check`／`take` 用 `EvaluationFailure` 保留原 Status 及可选的原批成员序号，
不再只剩错误字符串。仅普通 Text 在发布前成功读出采样有效性、恰一个值为0且其余为1时，
才定位该成员；设备／读回错误没有这个定位。原 abort 提供真实退休与错误观察，逻辑采样错误
不写进整批持久错误。owner 只有在干净退休后才终止坏请求、按原 key 重排其它成员；
其它情形继续原错误／退休处理，不能推定仅一个请求失败。序号属于原调用的成员顺序，不是窗口 slot。
[有限 CPU R2→R1 检查](../evidence/correctness/inference-request-sampling-failure-2026-10-25.json)
覆盖实际坏 A／好 B、B 的状态／ledger／RNG 不变和同 key 续算。默认 CLI 仍遇错退出；
未注入 native 读失败、退休失败或多个坏成员，不以此宣称所有入口的错误隔离已完成。

普通请求的停止 token 政策是冷不可变输出配置，独立于 KV、采样图和 FamilyKey。
ledger 冷期值拥有同一政策，完成阶段记录 stopped，观察阶段消费该事实；前缀命中使用
新请求政策，原请求暂停／重建保持自己的政策。非默认政策进入未绑定 ledger 时必须在
发布前拒绝，不能在 finish 首次复制配置。[host 与 CPU 旅程](../evidence/correctness/inference-request-stop-policy-2026-10-25.json)
已检查这些边界；MTP 目前只接受默认停止合同。

普通随机请求在开始 turn、前缀查找、恢复及等待窗口接纳前，用实际剩余输出证明整轮 RNG 地址可用；
只读原 live head、原 ledger 末提交或新请求 initial cursor，不预占地址，也不另存可推进的游标。
新请求不能借用缓存发布者的低游标；state-only 不消费，最后 pending 不另消费。raw Text 没有整轮
额度，仍在实际 episode／输入变更前证明本次一个 draw。[有限 CPU 旅程](../evidence/correctness/inference-turn-sampling-budget-2026-10-25.json)
已覆盖早拒、同请求重试、完整命中、后续轮、淘汰恢复及 MAX 边界；不代签 GPU 或 MTP。

`OutputDelivery` 只借用原有界输出记录，保存成功交付的绝对前缀游标；重复或较旧的确认
不重复推进。它既不观察模型 commit，也不拥有第二队列；模型 turn 完成后仍可保留最终
未交付输出。owner 先核 request key，再用当前记录的实际长度计算 credit，不能采信晚事件
携带的旧长度。提交前按该 episode 最大真实输出前缀检查 credit；这是单 owner、每请求至多
一个在途 episode 下的预检，不是另一套并发预留器。credit 不足时等待或选择原来支持的
较窄 episode，不能把不可写当作 turn_done。输入关闭、模型结束、最终交付和资源退休各有
自己的完成事实；释放 payload 槽前须满足实际寿命和交付／明确丢弃合同。
每次准备接收本 episode 的 output credit，而 turn 的原输出上限保持不变；零 credit 在接纳和
状态访问前拒绝。原 Qwen 生命周期用例已检查 credit=1 暂停、其它请求继续、确认后唤醒及
晚确认的世代隔离；[交付游标检查](../evidence/correctness/output-delivery-credit-2026-10-25.json)
另覆盖主机合同。这些结果不代表 MTP 窄化或所有后端均已运行。
GLM 同一原角色案例还复用真实 turn 进度：已有单输出 K1、新请求采样 fork 以及两次 ordinary
续算均交共享算法观察，第二 turn 只消费历史 pending。它不计算任意用户 suffix，也不替代完整模型服务。

## 冷准备与热路径

固定来源、采样政策、精度、layout、资源绑定、容量与放置共同构成准备合同。
当前构造配置固定，草稿政策仅首次准备前可设；改变冷合同就重建／重新准备，
不在每 token 重算配置 hash 或重新查 Provider。

运行期 metadata 与原生计划拓扑分别判断。KV 增长即使仍属同一 Schema，也可能改变
split／tile 数和 workspace，不能仅凭容量或地址相同复用捕获。Provider 证明原计划可更新，
Graph 在绑定或原生资源变化前失效旧录制；模型和请求层不解释后端的 chunk 规则。
Provider 的调优策略受完整冷容量和原字节预算约束，查询与实际规划使用同一策略。
短上下文的收益不自动推广到长上下文，减少 kernel 或重规划也不等于完整生成提速。

已准备对象组合原图、同一 exact child 的输出事实、真实资源域、执行端点和有限槽位。
只为 caller 输出准备目标；generated 输出从原 Invocation 取得。读回按实际 dtype／storage，
不能按 CUDA／Apple 名字猜测。形状相同也不代表捕获的资源／权限身份仍有效。
未知 suffix 只在有界的冷接纳处准备实际 child，不预编全部 context，不引入热动态图解释器。

计算工作区与已发布状态可以有不同寿命。当前 K1 MTP 对实际 caller-owned hidden 输出
保留地址稳定的 Phase 工作块，在原 Graph 成功及接受前缀确定后，经原 CopyBatch 只复制
选定 carry／prior 到原 canonical 池；旧 Snapshot 继续持有独立租约。复制实际退休、原
Graph 解绑且全部别名退出后，才能恢复工作块 writer。工作块和复制控制走原执行预算，
不占用缓存状态额度，不增加状态所有者或执行器。其它输出保持原路；是否采用此类暂存
取决于完整消费者收益，不把分页 KV 推广成所有循环状态和输出都已地址稳定。

模型构图使用冷装配提供的常量、字面量和位置，不直接挑 Backend 资源或 Queue。
当前 Qwen 已将这些职责从 Recipe 移入 GraphAssembly；CPU FFN 放置在数学调用之外，
普通主干、draft、接受采样和前沿采样沿用各自原图。后续共享化以真实跨模型共同点为依据。

原参数接入可用现有 GraphConstantBinding 借入同一 ReadLease；冷期重排只保必要的派生布局，
不为构图复制大矩阵或添加第二权重 owner。Kimi [有限绑定检查](../evidence/correctness/kimi-original-attention-binding-2026-10-25.json)
已核真实租约、零主机常量载荷及旧 reader 寿命；原23角色数值随后按上述有限组件合同通过。

N1／R1 是共同控制链中的直接路径，继续使用 R1 原图和同一 prepare／submit／finish／commit。
prefix 关闭且能证明 head 独占时，避免目录和无关请求扫描；N1 本身不能证明没有逃逸读者。
自然循环只对多成员桶执行外层缩批查询；最终单成员桶仍由执行入口检查真实版本接纳，
不重复查询，也不把 N1 当作可省略权限或预算证明的理由。
整数上传 scratch 和分桶索引已冷预留复用。索引暂存只持有完成标记和下标，
兼容规则与接纳政策仍由调用方表达；缩批只标记真正完成的前缀，固定回放仍先检查整波输入。
q、状态、输出租约及历史报告按各自寿命保留。
不能为文件拆分增加热分配、复制、转换、kernel 或 CPU 往返；最终仍以实测性能判断。

## 尊重平台、精度与状态差异

Backend 在冷期选择实际编入的端点，编入能力不代表设备可用；失败不暗中回退。
`has_cpu_assist()` 只表示 GPU 主端点附带 CPU 辅助；纯 CPU 主端点由 `host_only()` 表达。
CPU＋GPU 通过冷放置和 Graph 的跨域边表达。模型方程不应决定某平台的搬运、Queue 或执行包装。
Qwen 的 CPU FFN 放置由参数清单的归属派生一次，加载器、量化辅助角色与目标图共用
同一不可变事实，已删 loader 的名称解析和第二份层数；默认纯主端点不建立角色表。
Backend 仍负责实际端点与资源，NextN 保原主端点。[元数据及合并 CPU 消费者](../evidence/correctness/inference-ffn-cold-placement-2026-10-25.json)
已通过；本批未运行新的混合 GPU 或 MTP native 检查。
原生 FIFO／ExecutionSource 和显式 Queue 可以不同，不为统一接口制造多余 Queue。

物理编码和计算精度分别绑定。保留 GPTQ F16 舍入、CT／NVFP4 的原 scale 约定、FP8 block128
与 channel／token 差异，以及 dense／GGUF 直接 F32 head 和部分量化 BF16→F32 head 的原数学。
共享采样规则不表示可改写各家族的 RNG 地址排布。

采样的分布数学已集中在 `sampling::distribution`。家族批图保留行提取、分布、
身份提取和抽样的原构图顺序；完整前缀则直接提供身份。端口形状、批量 greedy
TopK、读回和发布属于各自调用者，不为合并几行装配增加地址描述或延迟回调。

受控logits已沿`sampling::ordinary_selection`接入普通批图和完整前缀重采样。冷配置只决定bias／allowed端口是否存在，每个episode借入typed ready租约，owner可以在下一步换值；载荷不进入KV、快照或历史账簿。默认图保原批量greedy和读回，不增加modifier端口。原logits的NaN／正无穷、非有限bias及全mask在head／ledger／RNG发布前拒绝；可定位的坏行沿原退休和幸存请求重排机制处理。StateOnly不需要这些输入；历史恢复只重建已消费状态并恢复原pending／游标，不能用当前mask重采历史。原CPU小Model的[有限请求流程](../evidence/correctness/inference-logits-modifiers-2026-10-25.json)已通过，不代表其它后端、质量或吞吐；MTP没有新增此政策。

CT BF16 block128的[共同冷图片段](../evidence/correctness/ct-block128-supplied-graph-2026-10-25.json)接收实际Value／TensorSpec及原租约绑定，固定形状和symbolic token行共用同体，按真实几何追加既有Graph数学；原fixture复用同体。它不拥有存储、调度或源格式注册表；BF16 block QDQ、channel／token与F32-scale code-dot仍由制品策略明确区分，不能仅按SF形状猜数学。[普通引擎冷入口](../evidence/correctness/ct-block128-model-binding-2026-10-25.json)由源配置选择reader、角色保留原W／SF及local FP4舍入，Recipe将片段生成的真实NodeId交回原GraphAssembly记录placement；后端继续消费同一Graph，不增加运行时路由。

投影的逻辑 `[N,K]` 由已验证的 `ProjectionBinding` 解释一次：冷发图复用
`projection_detail::logical_shape` 的输出宽度，各格式分支只表达各自数学和原操作链。
dense／GGUF 权重银行仍为 `[N,K]`，通用 MatMul 的 `[K,N]` 来自原元数据 View；
block128 编码 carrier 的 `[K,N]` 保留独立解释。这不改变编码、舍入、累加或融合资格。
[扩展边界记录](../evidence/research/inference-extension-boundaries-2026-10-25.json)区分
十种格式的源码等价复核与本次实际运行的既有 BF16 CPU Model 检查，不代签全格式执行或测速。
逻辑形状和分数政策的有限类型分派均有编译期穷尽检查；新格式须显式补充解释，
不能静默套用已有 FP8 格式的形状或报告。当前类型与精度规则保持不变。

[状态与请求所有权后续复审](../evidence/research/inference-contract-followup-2026-10-25.json)
按实际 Kimi cursor／完整 StateVersion 扩展及 Qwen 背压恢复／旧 Model 退休逐链检查。
必要的数学、状态声明和后继绑定仍归模型；这些有限源码结论不表示 Kimi 已接完整 token 请求，
也不把 GLM 已知前沿后的 suffix 消费者当作任意 prompt 的完整引擎。

缓存的拓扑、存储解释与页权限分别表达：

- 每个 bank 的 `CacheStorage`／schema 决定真实 dtype、K/V scale、行大小和 native recipe。
- `BatchSchema.with_cache_storage` 显式派生不可变存储变体，共享原 metadata 域、generation 与分配预算；独立创建的 schema 仍属于不同域。
- Arena、原页 owner、ready 读者和写入／退休权限仍由原资源机制证明，布局共享不授予页权限。

布局以 `origin_schema()` 查询拓扑出处；构图和 native program 从冷 spec 或实际 Arena 取得存储解释，
复用同时核对原域与完整存储解释。一个 Graph metadata 输入仍须绑定同一个实际 layout，逐层 scale
共用这份原元数据；后端计划私有的 compact indices／scratch 另计，不能把 CPU follower 的零上传
报告推广为所有 native 程序没有暂存或搬运。布局单参数 Graph 形式使用其来源的存储解释；存储变体使用既有 typed capacity
加 owned metadata 接口，不能从 layout 反推实际缓存 scale。旧 `.schema()` 查询已删除，无兼容别名。
[有限检查](../evidence/correctness/inference-cache-topology-storage-2026-10-25.json)覆盖 Windows CPU
39项及普通消费者、Apple 原 E4 Q1 续算，以及相关原生 host 严格编译；未运行所有后端的双 scale
共享域案例。原 E4 编码／读取设施继续复用。

普通 CT 制品的缓存政策与权重／activation 精度分开：准备器显式选择 BF16 校准省略或
E4 原校准，保留按实际 attention 层排序的独立 K/V BF16 标量。冷期一次读取精确两字节并扩宽，
同一份已准备来源给出实际 carrier 行字节，随后规划容量并移入 Model；容量搜索不重复源读取。
`Plan::row_bytes` 显式承载物理事实，不再由模型几何硬编码两字节。GDN／conv／hidden 不随之改变。
每层原 Arena、append 和 typed attention 使用同一存储变体；K 在原 norm／RoPE 之后、V 在原投影之后
各编码一次，仅处理新行，旧编码页直接共享或 COW。可追加页按实际行字节保留旧前缀与后缀权限，
不放宽 fresh generation、成功发布、真实退休及后端复制对齐条件。E4＋MTP 仍冷拒绝。
[有限检查](../evidence/correctness/ordinary-e4-kv-2026-10-25.json)覆盖准备器／角色、物理字节规划、
CPU分页六项、CPU默认BF16及CPU／CUDA双层实际Model续算、分叉和旧读者；生成夹具借入四个原标量，
不代表原27B数学／质量或吞吐。Apple另用原生编码→同页追加→reader核独立字节、输出与预算归还；
Vulkan本批仅严格host编译。CLI的初始化诊断另加回前移的metadata准备耗时，未把参数规划时间算成模型加载；
该报表修正只做源码检查，实际运行的Model定义与最终源码相同。

追加的逻辑token行数与物理carrier第一维分别表达：由原`cache_carrier_description`的单token描述取得首轴倍率，
恢复query_rows后再核完整描述和strides，不另列dtype／codec白名单；例如Q8_0将头数合入U8第一维。
普通prepare与Apple artifact缓存query_rows，原生绑定用同一规则检查。Arena复用创建时已验证的实际整行字节；
各block只在行内，COW和独占尾权保留原block、世代及旧读者权限。RowCache显式排除。
F32直接复制原producer行，不为统一接口新增identity编码或转换。后端reader的资格与原生复制对齐分别检查。
[Q8接线及共享路径回归](../evidence/correctness/ordinary-q8-append-2026-10-25.json)与
[F32 CPU数值／字节检查](../evidence/correctness/ordinary-f32-append-2026-10-25.json)分别记录实际范围，不推导新GPU reader支持。

冷图的有效历史长度不能由固定物理容量推定。GLM Flash 的原2051项选择状态保持不变，
实际闭合池与尾窗决定参与 attention 的前缀，零池省去评分；参数／codec／排序差异仍由
原数学路径表达。[原 CPU 启动组件](../evidence/correctness/glm-flash-ordinary-startup-2026-10-25.json)
已从空历史续算过首次池化；MLX／GGUF 构图检查不代签其原参数 native 数值。

ModelOpt 静态 FP8 的准备器、角色选择和投影绑定保留原 E4 权重及独立 F32 SA／SW；
Recipe 仅发出现有十二节点公开链，融合或分解仍由冷图选择。所选软件 reader 的
倒数／量化／两次 scale 乘法顺序写入源身份，producer 版本只作出处。
[有限冷链检查](../evidence/correctness/qwen-static-fp8-cold-chain-2026-10-25.json)已通过元数据、
准备器及合成 CPU 分解数值／租约。随后一个[原 k_proj 组件](../evidence/correctness/qwen-static-fp8-original-component-2026-10-25.json)
以原 E4 权重及独立 SA／SW 通过 CPU／CUDA 各两次 Invocation、共8192个输出的解析误差界检查；
默认 CPU 走六步分解，CUDA 走一步 packed，保旧输出及原权重租约。该组件不代表整制品或吞吐。

Qwen 分页 COW Snapshot 与 GLM／DeepSeek StateHead 不强制换成同一容器；共用的是
完整选择、提交和退休协议。原 state entries／端口仍是唯一声明，不加同义映射表。
Qwen 的整批预检／无抛发布与 StateHead 的单头原子发布保留各自原子范围。
公共接受结果不拥有状态，模型仍负责自己的完整发布事务。
输出重绑也区分只读常量、输入别名和自有写入块；保护留在真正拥有其权限的后端，
不能让模型循环特判某个原生 buffer。

准备期资源作用域保留权重、池、原生分配控制器和执行端点。未知 Completion 退休时，
应用退出或观察失败都不能释放在途资源；停止接纳、撤销可见性和实际释放是不同阶段。
调度器不放入可能最后由退休回调释放的资源作用域，避免自等待。
状态池按实际对齐计入相邻 live 行间隙；有效 payload 不能直接当作多槽 backing 容量。
已知缓存、workspace 和未知原生私有开销分别报告。

## 实施包与收敛条件

后端装配、真实资源域、输出读回、退休保活、版本接纳查询和有限分桶暂存已有共同机制。
不因模块不同就再次包装这些原语；制品注解只属于已加载权重，通用 Tensor、状态和激活
不承担源 codec 的接线职责。上一轮 MTP 基础架构按以下三个交付包收敛，并以有限扩展检查核对实际修改范围。

| 顺序 | 交付包 | 有限完成条件 |
|---|---|---|
| 1 | 生成合同与完整前缀适配 | Qwen／GLM／DeepSeek 直接使用已有 limits、接受结果和原状态事务，清除仍重复的通用接受／尾部算法；逐一核对模型修复、pending／RNG 和失败发布。保留真实阶段顺序，不另造统一控制器或包装已有 `AcceptedPrefix`，不只迁移测试判定器 |
| 2 | 多会话／多请求与缓存政策 | waiting／active、空闲淘汰和目录索引不依赖 Qwen `Session`／`SnapshotRef`；turn 输入／额度／停止和提交观察规则由共享请求进度承载。Qwen 普通／MTP N2／R2 多轮淘汰恢复保持原结果，一个真实 StateHead 适配复用这些主机算法。模型负责容量证明、完整检查点和恢复；显式固定 tape 保留独立载荷 |
| 3 | 完成事实与入口 IO | 请求 JSON、默认打印及 trace 写入移出模型完成路径；保留提交前预留见证和必要的条件捕获，晚 IO 失败仍可幂等观察。关闭诊断时不增加捕获、读回或分配 |

扩展检查覆盖新家族适配、同族两种真实量化配置，以及平台／后端融合接入的修改范围。
它检查依赖与实际调用，不要求再跑完整大模型或全后端矩阵，也不把未运行写成通过。
架构完成须同时满足：共同流程确实有不同模型消费者、原资源与失败边界保持、无新增热路由，
以及新增模型不必复制请求／缓存算法。之后进入普通／MTP 分账的性能定位与正式对照。

扩展后的合理改动范围：同家族新 codec 主要改制品绑定；新模型主要补数学、状态和角色；
后端融合主要改 Provider 及冷资格；请求／会话政策主要改控制层。
若仍需每次改动多个无关层，就继续纠正边界，不以增加接口数量作为完成标准。

有限迁移依据分别见[自然请求与晚 IO](../evidence/correctness/inference-natural-requests-2026-10-24.json)、
[进度与版本预算](../evidence/correctness/inference-request-boundaries-2026-10-24.json)、
[冷图装配](../evidence/correctness/inference-graph-assembly-2026-10-24.json)、
[执行观察](../evidence/correctness/inference-execution-observation-2026-10-24.json)与
[跨模型 K1 接受边界](../evidence/correctness/inference-k1-acceptance-2026-10-24.json)、
[有界分桶索引](../evidence/correctness/inference-batch-indices-2026-10-24.json)及
[权重源注解边界](../evidence/correctness/inference-loaded-weights-2026-10-24.json)与
[共享请求窗口](../evidence/correctness/inference-request-window-2026-10-24.json)，以及
[生成合同与 DeepSeek 接受去重复](../evidence/correctness/inference-ds-greedy-2026-10-24.json)和
[typed 目录与真实 StateHead 接线](../evidence/correctness/inference-typed-prefix-directory-2026-10-24.json)及
[完成事实与同步入口 IO](../evidence/correctness/inference-completion-io-2026-10-25.json)，以及
[请求进度与有限扩展检查](../evidence/correctness/inference-request-progress-2026-10-25.json)。
各记录区分实际运行、源码等价和未触发分支；它们不替代整模型质量、全平台或正式性能测量。
