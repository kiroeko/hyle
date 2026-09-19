# M1-R：模型条件与原生路线的冷准备合同

## Closure card

- owner：`/root`；baseline：`6933817f`，Windows main，工作树无既有产品修改。
- component：`compute`、`graph`、`compute_backend_cuda`、`compute_route_apple`、
  `graph_route_apple`、`compute_route_vulkan`、`graph_route_vulkan`。
- trigger：用户要求模型家族/revision/配置、阶段/片段以及设备/驱动/Toolkit/Provider版本可影响
  固定规则；当前普通prepare/compile不能向Provider传递模型条件。CUDA backend也缺少与Context
  寿命一致的设备/软件事实快照。此新需求只重开具名backend边界，不重审CUDA执行/内存内核。
- 先决：M1-G已按[资源卡](llm-moe-m1-ggml-runtime.md)CLOSED；不恢复旧C1或R1–R7队列。
- 主纵切：本地Qwen3.5-2B的模型配置身份、prefill/decode与region语义随普通Direct/Graph冷准备
  到达成熟Provider；相同类型/shape的另一个revision或配置不能误命中模型专用资格。
- 正交纵切：无模型注解的工程MatMul/归一化片段继续自然执行；两个不同模型/阶段的准备结果
  并存，共享常量只读，调用方源字符串销毁、owner move、并发使用和OOM不留下悬空元数据。
- scope：可选、不可变模型身份；冷请求传递与已绑定事实；各路线原生设备/软件事实；固定资格
  匹配的真实入口验证。复用现有数值/schema/多输出/effect/lease检查，不重造共同执行器。
- non-goals：自研算子、shader或性能比较；模型loader；通用规则DSL/全局注册表/跨进程plan缓存；
  改造oneDNN阻塞、Sched/Mem、性能仪器或selector；本卡不签64域覆盖与整模型执行。
- tier：P0。只新增冷元数据/资格接缝；成熟路径完整消费性能仍在M2，不在这里启动自研竞速。
- 预算：本seam源码/设计总2小时；每个实现根因最多两次纠正，出现owner/寿命/常态成本变更重新审计。
- 机器/交付：Windows/WSL CPU+CUDA，Air/Mini CPU+Apple，Windows/Air Vulkan；适用source/static/shared
  与普通Direct/Graph消费者。重型串行parallel2；Windows DLL邻置；无UAC/电源策略修改。
- stop：下述有限接缝及适用门完成后关闭；模型专用计算方法仍须具名Provider/M2规则证据，
  不把加字段或合成Provider证明当作模型优化/覆盖完成。

## 能力与形状审计

| 能力 | 层级 / 承载 | 已核对事实 | 有限处理 |
|---|---|---|---|
| 模型/配置身份 | 消费方事实 + Compute机制 | 当前ProviderRouteKey只标实现；ProviderInstanceId只在当前进程内标slot | 新可选不可变身份；精确内容比较，不用地址/Provider key冒充模型身份 |
| phase/region | 消费方计算声明 | prefill/decode影响同一模型的调用形状；HYLE不猜payload或读取模型文件 | 准备请求显式携带；实际Operation/DAG/数值与effects仍是合法性真相 |
| Direct冷请求 | Compute / BORROW现有prepare | ProviderOperationRequest、ProviderPrepareRequest和CustomKernel请求没有模型事实 | 一路传递到propose/prepare；执行只用已绑定artifact，不新增submit参数或查找 |
| Graph region冷请求 | Graph / BORROW现有候选规划 | propose只看DAG；propose_resources与compile共享RegionCompileRequest | 在资源/资格阶段传入条件即可拒绝未匹配region；不为此重写发现接口或oneDNN源码 |
| 资源/常量/状态 | 各路线既有owner / BORROW | common Graph有多输出、effects、constant artifact作用域；Apple/Vulkan保留各自Program与结果lease | 保留现有拒绝/保活；状态片段以明确输入/输出或已支持effects表达，不能用模型名绕过检查 |
| 硬件/软件条件 | backend/原生route / BORROW | Apple/Vulkan已有Environment；CUDA已有设备/context owner但缺同寿命版本快照 | 复用原生事实与锁定版本；CUDA定向冷采集，失败保留native码；不做万能Environment |
| 计划/缓存身份 | artifact owner / BORROW | 现有常量cache局限一次不可变Graph family，没有跨模型全局plan cache | 保留绑定workload并做完整相等比较；配置/规则/环境变更须重新准备，不新增全局cache |

CPU/CUDA便利factory只建立Context；模型阶段随每次prepare/compile而变，故不把它绑死在
DefaultContextOptions，也不为传参无故重开CPU/CUDA route owner。`compute_provider_ggml`及两项
oneDNN仍保持原BLOCKED；如M2要恢复CPU Provider，另建SCOPED卡并引用M1-G满足的恢复条件。

### 有限 FIX（下述AUDITED决定固定R1–R5）

| ID | 允许改动 | 必需验收 |
|---|---|---|
| R1 | Compute可选ModelIdentity owner与WorkloadContext：family/revision/configuration独立精确内容；phase与具名region identity | 一个冷分配、源字符串可立即销毁；空注解零分配；copy/move/并发读/OOM/size溢出；不同revision/config/phase/region不相等 |
| R2 | Direct请求、Graph的compile/resource/region请求与原生Apple/Vulkan冷入口传递；artifact保留绑定事实 | 自然入口至真实成熟路径；无注解原路径；准备后元数据owner保活；不增加执行期模型读取、规则搜索或计时 |
| R3 | CUDA Context冷采集设备/SM及driver/runtime/build toolkit事实；复用Apple/Vulkan Environment与Provider lock/version | 同设备原生API对照；错误翻译/关闭寿命；关闭CUDA零SDK探测；更换环境事实不能复用其专用资格 |
| R4 | 用固定资格条件验证命中/缺失/未知/配置不符；普通Direct与Graph的合法成熟计算验证相同冷请求 | 条件匹配不能绕过schema/资源/状态；unknown返回合法baseline或明确拒绝；缓存/已绑定身份区分；不宣称未验证的专用算法收益 |
| R5 | 文档、头自足与适用三交付，有限selector门 | 消费者不排序Provider/提供timing profile；源码/静态/动态同合同；适用门未运行不得PASS |

### API / ownership / 成本方向

- `ModelDescription`只描述调用方已知的family、revision和configuration identity，不含权重或模型
  JSON。身份字段按字节精确比较，不对未知模型猜测等价；缺失必要条件不启用专用资格。
- `ModelIdentity::create(description, allocator)`一次冷拷贝拥有这些字符串。使用现有Mem与Util
  引用计数，默认空对象不分配；创建失败返回Status。allocator backing活到最后一个保留者释放。
- `WorkloadContext`带可选ModelIdentity、phase和region语义标识；不带Provider排序/计时/用户规则。
  其副本可跨线程读。注解与stage只在prepare/compile消费；Program/Executable保留绑定事实供冷诊断。
- 普通入口增加可选准备参数；无注解调用继续使用已具备资格的默认路径。原生Apple/Vulkan
  仍返回各自Program；CPU/CUDA仍使用现有Context/PreparedOperation/Graph，不统一其执行器。
- Provider的固定条件可读模型内容、实际描述与本路线原生Environment；真正的数值、资源、
  alias、effect和失败检查仍先于接受。模型名不是支持能力或放宽精度的凭证。
- 单模型只付一次identity字符串分配；多个准备artifact共享不可变owner。新增工作只在冷准备和
  最后销毁，submit不复制/读取这些字符串、不增分配/系统调用。绑定identity不冒充序列化cache key。
- 当前没有跨模型plan cache；现有family-local常量cache不得跨workload。应用持有旧Program时
  仍代表原准备合同；要改变模型/阶段/规则或相关软件环境必须重新prepare，不能原地换标签。

## 预声明验证

- R1先完成owner/lifetime/overflow单测；R2/R3/R4稳定后才冻结一个完整candidate做selector适用门。
- 原生CPU/CUDA/Apple/Vulkan真实成熟片段，至少一个多输出/常量保活案例；model与generic分别覆盖。
  单元测试可反证条件匹配/冷调用次数，但不作为Provider覆盖或完整路线验收的替代。
- Windows/WSL的ASan/UBSan适用部分；共享身份并发只读与最后释放的TSan；GPU资源沿既有原生
  correctness，native driver不以CPU sanitizer代签。对未改资源路径复用准确身份的证据。
- 公共头/跨factor请求变化按selector扩展真实直接消费者与package门；oneDNN仅作直接消费者
  correctness/build，原性能BLOCKED和预算保持，不重跑其Q2。
- 文档/状态提交只跑docs/closure与diff。P1在本冷接缝卡不适用；模型/片段默认性能由M2逐批签署。

## AUDITED决定与七问（2026-09-20）

1. **来源/升级**：模型需求和成熟Provider来源继续使用M0的已锁映射；本卡不更新模型或库pin。
   ModelDescription是consumer声明，HYLE不读取HF/JSON/权重。Provider自己的版本/原生能力由对应
   实现读取；未知/缺失事实不能伪造为已资格化。无通用规则解释器，无数学实现变更。
2. **消费/语义**：普通调用形状固定为可选WorkloadContext传给Direct prepare与Graph CompileOptions。
   同模型可建立不同phase/region的artifact；ModelIdentity拥有字符串，copy保留同一个不可变owner，
   相等用完整内容而非指针。known规则必须完整匹配family/revision/configuration，且仍检查真实IR。
3. **下层复用**：Mem allocator与Util已有RefCounted负责所有权，不另造原子计数或分配器。
   CPU硬件机制复用platform；CUDA只在已有Context创建边界采集事实；Apple/Vulkan沿现有Environment。
   不重开这些下层模块，不新增全局CPU/device探测器或模型registry。
4. **成本**：有注解时ModelIdentity一次连续冷分配保存header+三个字符串，空对象零分配；
   handle复制/最后释放用已有引用计数。WorkloadContext只进入冷请求和artifact元数据，大小/保留位置
   随候选记录；Invocation/submit不复制它，不执行字符串比较、native query或新锁/系统调用。
   本卡没有新的算术/吞吐结论，P0只证明上述边界；后续模型规则的完整成本属于M2。
5. **接口**：ModelDescription三个独立string_view只借用至create返回；ModelIdentity是一个私有state
   指针的普通C++ owner。空family不得带孤立revision/configuration；缺失revision/configuration可记录，
   但不能匹配完整专用资格。WorkloadPhase限unspecified/prefill/decode，region语义用带version的
   namespace/id强类型；非法enum或半初始化region在准备前拒绝。没有Native FFI/稳定跨版本ABI承诺。
6. **交付/依赖**：现有compute component承载新小头与compiled实现，已有util/mem私有依赖足够。
   不改变中央组件图，不引入新第三方target。Provider私有request追加冷字段并提升合同版本；
   RegionCompileRequest能在propose_resources阶段匹配模型，因此无需修改既有oneDNN发现回调或
   重开其BLOCKED实现。跨factor真实消费者与三交付由selector在完整候选上验证。
7. **易用/停止**：调用方只提供自己已知的模型/阶段声明和路线，不配置Provider优先级、设备版本
   猜值或timing profile。无注解仍是普通入口；已有artifact不支持原地改模型标签，相关事实变化
   必须重新准备。R4分别证明资格条件/请求传递和真实成熟计算，不能据此宣传尚未接入的模型优化。

先实施R1不可变身份与失败/寿命验证，再接R2/R3/R4，最后固定候选执行R5。schema/多输出/effect/
lease已有路径逐项复用；若模型片段暴露具体能力缺口，登记在对应M2批，不以本卡扩张自研。
本次owner进入AUDITED；R1–R5之外无产品改动授权。M1-G仍CLOSED，旧ggml CPU与oneDNN仍BLOCKED。
