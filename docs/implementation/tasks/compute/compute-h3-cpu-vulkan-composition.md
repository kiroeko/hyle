# H3：CPU+Vulkan粗片段与offload基础机制

<!-- MODULE REVIEW: AUDITED -->

- baseline：84e8bc28724df5759cd7d28202dbe78a06e714fe（H2 terminal、Windows唯一产品真相，工作树原干净）。
- owner：compute_route_vulkan、graph_route_vulkan、新graph_route_cpu_vulkan；其他已关闭owner无具名缺陷不重开。
- trigger：用户准入CPU+Vulkan联合数学、有界异步传输/专家片段以及普通CPU存储、共享预算/准备诊断。
  原native route只有单GPU fixed Program/Tensor与同步upload/download，不能自然承担新组合。
- 不撤销C3 accepted boundary；旧WSL Vulkan十二项仍UNQUALIFIED、不代签PASS。H3是新能力新资格。

## 已核实的接缝与当前边界

1. Vulkan Context的upload/download在vulkan_transfer.cc临时allocate staging/create_run并同步wait；
   原Buffer、native容量/队列族、fence、pending owner与异常观测保活可借用，固定两槽与重用缺失。
2. 原Vulkan Program/Invocation拥有命令、managed ggml/IREE region、native输出/结果slot；
   用原native程序承载GPU片段，不复制CUDA Queue或用逐kernel万能vtable重写成熟资源管理。
3. 原Tensor retains Buffer，native_owner与borrow_ready要求same VkDevice/queue-family且已完成交接；
   外部pending semaphore不能混成ready。普通CPU权重大于staging时仍必须直接CPU计算。
4. 原Context memory_quota计actual VkDeviceMemory；不能冒充覆盖CPU、metadata、所有槽/旧lease、
   private packing的共享预算。复用现ResourceBudget并单列native容量/实际分配观察与私有unknown。
5. 原native Runtime已在fence wait时释放host mutex，独立producer可提交；单VkQueue命令没有隐含
   全部串行保证，但fence代表有序prefix，实际overlap只由原生时间线签，不从pending数量推断。

## 固定任务/候选FIX（本节不是产品实现授权）

| ID | 本包机制 | 验收 |
|---|---|---|
| H3-01 | 自然graph::cpu_vulkan入口、typed DAG冷partition/value-ID目录、required CPU/Vulkan位置；CPU/原Vulkan编译器保留完整regions/workload/全部端口合同 | CPU→完整GPU FFN→CPU与正交数据管线，原单路线guard保持 |
| H3-02 | 原native device/staging资源适配与普通CPU主存边界；精确编码/布局、同Device/queue-family、flush/invalidate/visibility | 大CPU权重不全量staging；dtype/alias/范围/empty/overflow/OOM |
| H3-03 | Vulkan Compute固定两槽WeightSlots/WeightBatch，ID/描述/容量有界选择、原始编码、原VkBuffer copy/barrier/fence与Completion保活 | 两槽/更新/旧引用不能覆盖、坏ID/format/容量、close/abandon与独立transfer/compute原生取证 |
| H3-04 | 组合每Invocation独立buffer/程序/结果，借现Sched CPU进展与原Vulkan Completion/retirement，失败排空不CPU重算 | 真数学、错误/取消/close/晚到、旧lease、未接纳和接纳后的区分 |
| H3-05 | ResourceBudget共享CPU/设备/staging/metadata/packing/所有槽及lease；现原生未知private明确strict reject或partial，冷transfer/replay/resource报告 | 去重/已知与unknown不相加、配额admission/rollback、host-wait拒绝及bind/reset失效 |
| H3-06 | 唯一cpu;vulkan集合自动选组合factor，Direct compute/Graph入口、三系统三交付、头/资产/搬迁/缺路线/指南与终审 | Windows、WSL和Air独立source/static/shared；Windows DLL同exe目录，旧入口定向回归 |

主纵切：CPU norm/projection与CPU-resident普通大权重→现成熟Vulkan完整FFN/合格量化片段→CPU消费。
正交：CPU分块变换→Vulkan矩阵/归一化→CPU结果；动态expert ID只决定有限payload，不重挑Provider。
引擎管理模型loader、KV/offload比例/层/专家cache/predict/evict；多GPU、推理引擎、算子扩张和增量图不做。

## 有界审计、复用与验收预算

- 先完成上述具名seam的一遍源码审计，以及锁定ggml Vulkan原队列/设备/资源/退休入口一遍；
  至多一项必要原生copy/fence/owner小probe。不重新翻模型或历史算子任务，不改变总体架构无具体反例。
- 公共shape/错误/ResourceBudget/Mem/Results/Sched和既有CPU Graph全部BORROW；独立counterexample
  才增补owner。技术细节可有界查原标准/锁定源码；不在本卡重写公共治理/性能仪器。
- 审计必须固定native承载、API、冷/热边界、ready/borrow、错误/弃置和资源账单后AUDITED再写代码。
- 批量接线/consumer迁移后API固定一次candidate，三系统三交付集中验收，复用相同身份不变域；
  private/source小修只补其因果半径，不重跑旧全集。build1/test1，重型run串行，Windows余量/commit
  及owned RSS guard，WSL限制12GiB unit，Air实际身份/余量，缓存/对象复用且不反向覆盖Windows。
- 正常/empty/坏ID/shape/encoding/range/overflow、payload/metadata OOM、same device/family、flush与
  visibility、两槽更新/绑定、scope total/unknown/rollback、cancel/close/abandon/accepted error、旧lease。
- 真实typed数学与原规则、源/安装头、runtime/native shader资产、真正SDK rename与missing route。
- 本包P0：cold固定程序/目录/容量，hot沿粗片段及必要copy/barrier/等待，managed private成本如实列出。
  native copy/compute actual GPU timeline证明适用机制；不因新math竞争为空制造P1或速度WIN。
  硬件/Provider无合法重叠域时列真实限制，不用mock/copy/枚举签混合能力。未运行一律不PASS。
- docs/state-only只跑closure/docs/diff；到有限预算/真实阻塞按协议裁决，不无限研究或假关闭。

## 当前状态

新身份QUEUED登记后，本卡将三个owner绑定SCOPED。尚未AUDITED、尚无产品实现或H3资格。
下一动作为固定有限native承载与FIX，然后批量实施。H2 CLOSED只作可复用机制证据，不授权改其产品。

## AUDITED：原生承载、API职责与有限FIX冻结

具名seam与原native ownership源码已完成一遍审计：allocate_buffer保留原ggml Device，actual
memory quota与Buffer/视图owner已有；create_run/submit_locked/observe的原池/command/fence和
pending_owner可复用，异常观察不释放未退休资源。现transfer确实每次创建staging并同步等待。
Vulkan标准的submission order本身不引入execution/memory依赖；保留同队列合法独立submission，
不为证明async创建第二线程/runtime或不同VkDevice。必要barrier限定到正确buffer/阶段；fence是
ordered prefix，冷报告不能假称独立copy已完成或跨family/deviceready。原生timeline单独验证。

固定公开文件/职责：compute/vulkan_memory.h承载原VkBuffer device/staging MemoryResource与
native Tensor/native owner映射、原编码和flush/invalidate；compute/vulkan_weights.h承载固定
catalog/两槽的异步迁移、借原Completion/fence与native ready binding；graph/cpu_vulkan.h承载
原typed DAG的cold partition/value映射与输入目录、独立Invocation/Results及必要Sched粗进展。
不修改CPU Graph/Provider规则，不新增数学；各片段继续原CPU/原Vulkan compiler/native程序。

设备资源保持同VkDevice与queue family，普通CPU内部存储走system/host resource，跨界才
固定native边界/可映射暂存。两槽cold创建命令池/fence/backing，hot只做有界ID选择、必要host
准备及原VkBuffer copy/barrier；相同shape/encoding的payload更新不重新选实现。native Tensor
与last owner/alias护住槽，ready只在真实终态后发布；cancel停止后继，失败排空已接纳prefix。
共享预算借ResourceBudget：payload接纳在分配前，controlled metadata借Mem StdAllocator，
managed region/private runtime保留unknown并strict拒绝或显式partial。配额一直保留到原Buffer
最后owner销毁；native capacity/actual VkDeviceMemory观察分别列示，不能合并当requested usage。
现memory_quota独立控制actual native分配，不替代共享CPU/staging/packing/metadata账单。

H3-01至06就是有限产品FIX集，三个owner进入AUDITED后实施；借现Mem/Results/Sched/common
语义，无新基础缺陷不扩owner。原生copy/数学单元后API固定、集中三交付及因果半径验证。

WSL继承C3用户已接受的真实GPU缺件边界：Windows/Air必须真实数学/寿命/重叠；WSL编译/安装
三交付和可适用头/config仍验，未具备GPU的数学运行继续UNQUALIFIED，不写PASS或N/A，也不
为同一硬件缺件自动再阻塞goal。这与C3 record已明确的后续H1-H3授权一致，不重新请求同一决定。

### 首批产品接线

vulkan_memory.h/vulkan_memory.cc接真实route Buffer/Device为独立device/staging资源；staging借
原host-visible VkDeviceMemory mapping并保留coherency/whole映射flush与invalidate。Mem opaque
handle是本路线allocation token，不把VkBuffer当CPU指针。native Tensor allocation保留原Buffer，
新shared ResourceBudget按原四字节native capacity在分配前接纳，原mem::QuotaAllocator仍独立
按actual VkDeviceMemory计账；buffer最后owner销毁、先释放native内存再退budget，views保留root。
控制块复用Mem StdAllocator，普通无预算Buffer只多一个cold accounting指针，不在hot查scope。

这是有限FIX H3-02/05第一部分，尚未原生编译/数学/三交付；不据源码或资源枚举计本包完成。
迁移固定池、native Program资源接线与组合继续批量实现，API稳定后按固定半径集中验收。

### 组合与固定迁移池源码接线

固定 TransferSlot 复用原 command/fence/retirement，WeightSlots 两槽按 catalog ID 与完整物理描述
选择 payload，原生 Tensor/native owner 保留会拒绝覆盖。普通 CPU 存储留在主存；仅跨端值使用
原 VkBuffer device/staging。CPU+Vulkan 粗片段编译保留原 CPU Graph 与完整 Vulkan 程序，准备时
固定输入目录、原生输出和内部引用基线；执行不重新选 Provider 或重新规划。原 Vulkan Completion
已经在原生工作和 Provider 尾项实际退休前维持 pending，组合进展沿用该保证与原 Sched。

共享预算接 payload、controlled metadata 和所有槽，私有 managed/runtime 继续明确 unknown；
准备报告区分主存、device、staging、native completion 等待和权重准备等待。新增普通消费者为
Direct migrated-weight MatMul、CPU norm/projection→完整 Q8_0 FFN→CPU 与正交 abs→GPU矩阵/归一化
→CPU；单独 contracts 覆盖坏 ID、容量、编码字节、OOM 回滚、旧 owner/结果、更新、取消和晚到失败。

Windows MSVC 已完成新增 TU/公开头语法检查，修复一个环境字段名并补正确 private include 后，
受影响 TU 与新增两消费者语法检查通过。这仅为接线诊断，尚未链接、运行 GPU 数学或三交付，
不签本包完整 PASS。下一步集中 source 数学/合同，然后固定候选的 installed static/shared 与原生
重叠取证。该接线快照统计 Compute+Graph 68,078 行/301 个实现文件，测试与样例不计入。

### source 数学/寿命验收与 H3-07 定向接缝

产品候选 ae0283e3b890d96945ac5116fd5afd3294a8f1cc：Windows 原两入口、Direct 权重 MatMul、组合
Q8 FFN/正交管线和 contracts 已通过；Air 同批五项通过。首批组合重复执行暴露了下载槽内部
output owner 未计入冷基线的问题；候选把该固定引用在 cold sealing 前纳入，保留普通外部 Tensor
保护。仅重编受影响两个 target、复跑两个组合测试，并定向复核受改 close 路径的 Direct 权重入口，
未重跑原两个不受影响入口。Windows 整批 owned peak 0.560 GiB，修复批 0.274 GiB；Air 646 MiB。
统一不可变快照统计：Compute+Graph 68,095 行/301 文件，全库非测试实现 163,322 行/652 文件。
这些是 source 资格，installed static/shared、WSL build-only 与重叠尚未闭合。

一次原生 P0 探针用原 RecordedProgram/sample affine dispatch 和真实 WeightSlots VkBuffer copy，
GPU query 装饰原调用，独立核实 512 个 dispatch 的 -1 不动点及 16,776,960 字节的 0.5 payload。
RTX 4070/617.14 原 queue family 2 的时间线无交集；把 query reset 移至测量前仍无交集，不能签
GPU transfer/compute overlap。不得把该结果解释为算子性能 FAIL 或硬件不存在独立队列。
具名源码反例：native_runtime_impl.h 的 Native/with_queue 只暴露原 compute queue；锁定 ggml
已创建 transfer_queue，但 H3 upload 仍经 compute queue。原生传输队列能力未接出，属于已准入
迁移/重叠机制的实际缺口，不是 benchmark 新造数学/API。

H3-07 新定向 owner compute_provider_ggml_vulkan 从 CLOSED→SCOPED，本轮 baseline 同为
ae0283e3b890d96945ac5116fd5afd3294a8f1cc，旧 [macOS loader CLOSED](../build/vulkan-macos-loader-discovery.md)
资格不撤销。现有 route owner
仍只按已审计 H3-01至06实施；新增接缝在补充审计为 AUDITED 前不得写产品。

有限 FIX：借原已创建 transfer queue/family 与原提交锁，不创建 VkDevice/线程/万能 Queue；必要
payload backing 明确合法跨队列共享，transfer→compute 原生 semaphore/visibility 交接，公开 ready
仍代表实际可用、可继续是 compute ordered prefix。原单队列设备保留有界正确路径并如实报告限制。
跨队列接纳后失败必须排空实际已接纳队列、保留原诊断和旧 owner，不能丢 pending 或 CPU 重算。
scope 只含 native bridge、route buffer/固定 transfer/原退休及相应 P0、寿命反例；不扩算子目录或
改变数学/default Provider。先固定追加 seam 一遍源码审计/有限实现，再一批定向原生验证；保留上述
数学事实，按实际改变的调用链补受影响验收，不重跑历史全集。停止预算：原生正交时间线两端各
一个固定有界候选；失败先查实际队列/同步成本，有具体缺陷才修，不换 workload 无限试。
