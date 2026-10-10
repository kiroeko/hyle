# H1：CPU+CUDA 粗片段、异步传输与流式权重

<!-- MODULE REVIEW: AUDITED -->

- baseline：`061b46700`（C3按用户接受边界CLOSED）；Windows canonical。
- 当前owner：graph、compute、graph_route_cpu、graph_route_cuda、compute_route_cpu、
  compute_route_cuda、compute_backend_cpu、compute_backend_cuda、compute_ggml_runtime、
  compute_provider_ggml_cuda、graph_provider_ggml_cuda。
- trigger：用户新准入的真正CPU/GPU片段执行、有界异步传输/依赖、真实重叠和引擎驱动expert片段迁移。
- 新增独立可选owner `graph_route_cpu_cuda`：身份6f74cb2d先QUEUED，本轮恢复SCOPED。
  它拥有组合/进展，不成为新用户产品；普通cpu;cuda自动取得，单路线不取得CPU/Sched依赖。
- 不重开Apple/Vulkan、Mem/Sched或无具体缺陷的算子owner；是否需要定向底层修复以有限反例为准。
- C3十二项WSL Vulkan实GPU运行保留UNQUALIFIED，用户明确不block；不代签GPU数学，不阻止本H1。

## 正常任务与两条纵切

主纵切：静态typed推理子图含明确CPU-required片段与CUDA数学片段；引擎声明权重存放/允许执行
位置和本次expert IDs/范围。编译绑定真实dtype/physical encoding/layout、model/artifact、hardware/
driver/toolkit/recipe；重复运行只更新实际payload/绑定，程序固定。另验纯host权重/GPU算术的
固定容量分批路径，不能把它当CPU数学参与。

正交纵切：CPU分块数值变换→CUDA矩阵/归一化完整片段→CPU消费结果的管线；两槽在途，
保留旧结果、取消未接纳后继、独立plan关闭。无需模型loader/token/KV/专家缓存策略。

复用Graph既有IR/placement/region/resource/replay、CPU成熟Provider与CUDA原生程序、Mem backing、
CUDA streams/events/Completion、现有结果lease与Sched。Provider成熟完整region优先，不拆成逐kernel
调度；不建万能Queue、另一套后台runtime或第二套路由IR，不写数学算子或扩大模型目录。

## 需在AUDITED前固定的合同

1. Graph拥有跨片段逻辑边界、真实转换/资源/依赖、compile/runtime分离和组合终态；原普通同Device
   compile保护保持，新组合入口显式CPU+一个CUDA端点，不隐式GPU失败回CPU。
2. CPU/设备/staging/Provider-private成本分别预算；未知不当零。pageable host先显式有界staging，
   同CUDA context的pinned backing才可作为实际async源；physical encoding不同必须合格转换或拒绝。
3. 提供阻塞完成的自然基线和有界异步进展；真实独立copy/compute可重叠，而非后台线程阻塞整图。
   CPU不能消费CUDA pending事件时显式进展/等待并报告，借既有Sched而不建新线程池。
4. 至少两个独立slot；旧lease/在途DMA/native region结束前不复用、改绑或释放。输出只成功ready后发布。
   borrow资源/allocator/executor寿命可发现；取消只阻止未接纳后继，close只排空本plan工作。
5. expert迁移只提供IDs/范围验证、实际选择/搬运、固定容量暂存、绑定更新和在途保活；引擎管理
   cache/predict/prefetch/evict/模型放置/循环。变payload或expert不重选Provider；变格式/shape域则重新准备。

## 已核实的有限输入

[原生接缝预审](../../../../evidence/architecture/compute-entry-design/h1-native-preflight.md)记录具体源码：
同Device guard；CPU Queue同步/不接受async completion；CUDA transfer仅接同context原生storage；
原CPU程序ABI没有ggml backend factory，通用mixed ggml CPU backend15实际commands未启AVX2；
普通Graph输出是borrowed而非可覆盖旧lease。它们是有限集成约束，不是H1已完成或性能LOSS。

本轮先审计成熟ggml scheduler的完整region承载与现CPU优选/程序资格是否可保留；不盲目初始化
通用CPU backend。若不合格，复用现合格CPU/CUDA完整片段和原生边界，不为形式复用牺牲语义/优化。
调查上限：一遍已具名源码接缝、一次自然调用草图反证与最多一个必要的有界native实验；
固定有限FIX后实施。不无限重选架构，不以benchmark行设计API。

## 预声明验收与停止

- 一批接线后冻结不可变候选；Windows/WSL实际CPU+CUDA主/正交完整数学、三交付，build2/test1。
- 正常/空/错shape/错encoding/坏ID/范围/溢出、资源/OOM、重复payload/绑定、两槽busy、旧lease、
  accepted错误终态、request_cancel、plan close、丢弃completion后owner保活与销毁。
- 原生stream/event依赖与copy/compute进展证据，host wait/转换/融合边界完整成本；不凭async命名签重叠。
- 公开头/target/运行资产与普通C++guide；CPU-only/CUDA-only原入口与明确缺组合能力诊断定向回归。
- P0为常态账单。只在API/算法/完整费用固定且源码有未决消费者预算机制时做一次有预算P1；
  一家成熟承载不做无限Provider赛跑，复用既有单路线证据，不改公共仪器。
- 每轮docs/closure/diff；按selector真实owner/消费者选择。文档/状态不重建/P1。
- 具体阻塞说清，不能用两个Context、mixed SDK、copy/mock或空target冒充本包。

当前有限审计已完成并进入AUDITED，仅允许下列H1-01至06产品FIX；实现尚未验收。
H2/H3按其native域后续独立施工，增量图继续DEFER。

## 有限审计裁决

| 能力族 | 当前证据/缺口 | 承载与动作 |
|---|---|---|
| 双端点静态typed DAG与粗片段 | 普通guard拒绝跨Device；现NodePlacement/region/liveness/copy规划可复用 | FIX显式cpu_cuda入口；普通guard保持；CPU/CUDA仍用原合格Context/Provider |
| 成熟混合完整region | 原scheduler存在，但当前CPU ABI不导出backend且通用CPU backend不具备当前AVX2资格 | BORROW既有CPU/CUDA完整region，不新增低资格CPU backend；whole scheduler作为本轮未选承载，不声称其性能失败 |
| 跨native资源与依赖 | CUDA只接受同context native storage，CPU仅同步Queue；现Graph cold routes、native events完整 | FIX显式有界pinned/device桥与分段进展；转换只走合格程序；不造新Queue |
| 异步运行/终态/槽/lease | 原submit遇host wait会阻塞；普通output borrowed，ResultSlot已有旧lease防覆盖 | FIX独立组合Invocation与既有结果槽；按segment驱动，不把整图扔到后台线程 |
| host权重/expert片段迁移 | 原native transfer可用，缺固定容量选择/槽/更新的自然组合 | FIX CUDA route的有界byte-range/ID选择与owned在途状态；数学和模型缓存不下沉 |
| 交付与构建隔离 | generic Graph不依赖Sched，CUDA-only不可拉CPU math；组合有独立依赖/测试消费者 | FIX新可选组合factor；中央图自动选择，三交付/原路线隔离定向验收 |

### 固定有限FIX

H1-01：新 `graph_route_cpu_cuda` 声明/公开 `hyle/graph/cpu_cuda.h`/实际源码与普通路线集合闭包；
private依赖既有Sched/Util，public依赖Graph/CPU/CUDA真实类型；不为CUDA-only添加组合依赖。

H1-02：Graph内部显式跨Device编译接缝，仅由新typed CPU+CUDA adapter使用；
它先验证exact CPU/CUDA Context及同CUDA context的host/device资源；沿现IR、NodePlacement、
all-port workload、Provider region和冷规划，不删普通sameDevice guard、不改CPU/CUDA优选。

H1-03：可暂停的segment进展复用原计划、prepared span/依赖/terminal retention，
组合owner借现Sched处理CPU片段，CUDA原生事件驱动后继；保留原阻塞run基线和实际host wait报告。
只处理当前静态typed DAG；无法保持effects/同步合同的native片段明确拒绝，不静默忽略。

H1-04：组合Invocation固定槽/owned结果、payload绑定更新、cancel/close/晚到完成；
旧lease持有槽owner，复用前检查全部在途完成/结果引用。借用Context/资源/Sched寿命显式声明；
close只本组合停止admission并排空，不能关闭应用其它Context/plan。

H1-05：CUDA有界权重/专家片段迁移机制，预先分配至少两组pinned/device容量；
校验IDs/范围/字节长度/格式边界后选择复制，发布真实DMA completion并在数学队列记录依赖；
支持每次改变选择/地址但不重选Provider，CPU参与数学与仅host权重上传分别验收。

H1-06：普通双端点C++guide、推理与分块数值真实consumer、目标测试、三交付/安装头/asset与
无组合能力诊断；接线批量结束后才冻candidate验证，不逐算子/每小改动重建全集。

### Public shape与生命周期冻结

- 组合名空间 `hyle::graph::cpu_cuda`，由显式CPU/CUDA Context和host/device native资源准备。
  `compile`返回现CompiledGraph；target身份固定CUDA0/CPU1，placement仍用现NodePlacement。
- `Invocation`作为组合独占可变槽owner，封装原Graph Invocation及CPU/transfer/compute native边界；
  创建分配/预备workspace，bind/prepare更新固定容量，submit返回现Completion/OwnedResult形式，
  阻塞run只等待相同终态。至少两slot可供独立上传与计算真实重叠，slot不供重入/并发变绑。
- 固定结果payload只提供成功ready的只读tensor视图及其保活owner；普通external输入仍借用至终态，
  外部mutable输出不声称immutable，owned输出旧lease持有时不覆盖。
- 迁移payload为实际byte ranges及stable expert ID/容量信息，格式/shape边界由真实TensorDesc/编译规则
  验证；不生成模型角色、cache策略或新的量化格式。错误发生在接纳前可拒绝；已接纳前缀以失败
  Completion保活并排空，后继不提交，不在运行时换CPU重算。
- 成本：cold图/region/slot分配；steady仅固定segments与选中fragments线性工作，必要pinned拷贝/
  DMA/CPU等待计费；metadata opt-in组合factor，普通route对象不加组合常驻字段。

这固定系统性合同与有限改动边界；内部state/容器/native callback细节在FIX中按实源码完成。
未实现能力仍MISSING/PARTIAL，不把本审计或身份登记算成完成。

### 首批产品接线（尚未native验收）

H1-01/02已接入实际组合factor与cpu_cuda.h/compile.cc：普通cpu;cuda Graph自动取得组合，
CPU/CUDA/Vulkan单路线不取得该factor或Sched；typed入口固定CUDA0/CPU1，要求exact-context
pinned/device资源。Graph内部cold admission传入两backend身份并沿原variant/region/预算路径，
普通compile同Device保护仍保留；无普通运行对象layout或hot字段变化。

`compute_routes_test.py`原11 tests实跑通过，实际manifest增加混合/单路线隔离反证。
首次新增case发现不必要PUBLIC route边及夹具目录分号，改为真实PRIVATE边/规范路径后通过；
未放宽checker或改产品路线解析。selector针对首批产品授权READY，无unknown路径；其共享
manifest superset不表示执行全部库数学，后续按本卡真实半径批验。CPP/数学/native库尚未编译或
运行，不写PASS；H1-03～06继续施工，整包保持AUDITED，不将该入口当联合/异步完成。

首批产品提交6f7684f07；统一脚本对该不可变commit、全部compute/graph根及新组合源码根统计，
[LOC结果](../../../../evidence/architecture/compute-entry-design/loc-h1-first-wiring.json)self-check PASS：
Compute+Graph62,657 code lines/272 files，全库自实现不含测试157,884。不是自研数学行数，
本片未新增数学算子；源码接线增加149行实现。下一步只继续H1-03～06，不启动已完成C3矩阵。

H1-03分段core接线：将原提交算法拆为private SubmissionCursor begin/advance/finish，
复用原admission、prepared bindings、native segment调用和最终多lane accepted-prefix assembly。
组合advance遇host依赖pending可返回，不阻塞该提交调用；CPU/managed片段仍遵守其真实同步成本。
普通submit使用borrowed stack cursor、原阻塞等待，不增加普通Invocation字段或额外retain原子。
空前缀拒绝恢复Idle；已接纳错误仍以真实Completion保活并退役。private cursor意外丢弃会排空，
组合owner须保活至finish，不把该内部cleanup当公开Completion析构行为。
当前仅源码接线/反查，不签CPP/native正确性或async/overlap；下一步仍需Sched/native唤醒、
owned槽、权重迁移与集中验收。
