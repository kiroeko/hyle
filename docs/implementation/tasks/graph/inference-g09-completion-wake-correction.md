# G09 完成驱动修正

状态：**AUDITED，2026-10-17**。SCOPED checkpoint `13cbba94b`；用户明确要求“我们先修复g09”。
baseline `8119dc6c2c02f5b3e588cc6e3b64b36b41222c59`；原 G09 产品候选
`3524413272cb52d06af5e64d634edbfef09b9864`、原功能/交付/CUDA P1 与三组未资格结果保留。
本卡替代本轮三个 owner 的施工入口，不重新授权旧 profiler/重试脚本。

## 需求与范围

- owner：`graph_route_cpu_vulkan`、`graph_route_cpu_apple`、`compute_route_apple`。
  前两者由 BLOCKED 回 SCOPED；Apple native owner 因真实组合消费者缺少主动完成通知而重开。
- trigger：[R29 因果诊断](../tooling/inference-performance-recovery-2026-10-17.md)在完整数学/
  结果/退休不变的 Windows Vulkan 消费旅程中确认周期观察策略有毫秒级成本。
  诊断占用 Scheduler worker，不可直接采用。Apple 原完成者虽主动执行，hybrid 仍按 timer 观察。
- 主纵切：同一 DAG 的独立 CPU/native 数学支路及 join，两个插入顺序；两个独立 invocation
  slots 的 CPU→GPU→CPU microbatch。反偏置纵切：批量数值分支与必要串行链，无 token 身份。
- 普通调用：Graph/placement→compile→create_invocation→bind→prepare→submit→checked owning
  result→release→close。native 数学、Queue、transfer/page、LeaseSubmission 和结果 owner 复用。
- 有限范围：让原 fiber 由真实完成/有界等待执行者唤醒，保 single worker、多在途、新 ready
  native、错误/取消和 accepted prefix 实际退休；处理通知/等待任务尾部后才允许复用。
- 非目标：G04/G05、CUDA 数学/Graph、公共 Scheduler/BlockingPool/Completion 改造、新线程池、
  新 Provider、全库重新资格化、采样/估计量/阈值调整。
- 直接依赖：原 Graph frontier/leases、Sched CV/Event/BlockingPool、原 Apple/Vulkan 完成者。
  直接消费者为现有 hybrid contracts/pipeline、native page/state joint、公开 SDK sample。

## 准入与暂定形状

能力是原 G09 联合进展的复合机制，SELF 只拥有协调；等待、数学、资源、取消和退休 BORROW
原 owner，模型分组/请求策略仍为 APPLICATION。所有既有公开入口继续归原 J1–J4 能力族。

| 项 | 已证不足 | 本轮有限设计 |
|---|---|---|
| W1 Vulkan | Runtime 无后台 observer；1ms timer 在无 ready native job 时观察 pending | 借 caller-owned BlockingPool；每 RunState 冷内嵌一个 waiter；不在 Sched worker 上阻塞原 Completion |
| W2 Apple | 原 FIFO worker 完成后，hybrid 未收到主动通知 | 原 invocation owner 提供内部 opaque 通知，Graph 保有 node/epoch/CV；Compute 叶不依赖 Sched |
| W3 退休/错误 | queued job 取消不等于 GPU 完成；common Completion 的 const 观察会修改 handle | 冻结独占观察、destroy/release 尾部、拒绝/取消/close/异常进度与引用归还协议 |
| W4 使用/交付 | Vulkan 当前创建入口没有借用池 | 冷创建改为 create_invocation(Scheduler&, CPU Queue&, BlockingPool&)；无兼容双入口，迁移真实调用方 |

已有[最小设计与两条独立源码审查](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g09-production-fix-design.md)
作为本轮输入；`subscribe_retirement`/attach_lifetime 不会主动观察 GPU，不作为主动通知源。
Apple 单订阅/已完成竞态与 Vulkan pool 拒绝/queued shutdown 的唯一具体协议尚须审完，
在本卡与 registry 进入 AUDITED 前不改产品。

公共调用需要的 pool 必须活过 invocation/所有 waiter destroy/accepted native work；thread/stack
由原 pool 上限约束并明确计费。正常热路径不创建 JoinState/job、不新增 executor 或每 op fiber。
API/owner、失败/并发/常态成本在 AUDITED 冻结后批量实现，不边实现边改变合同。

## 预算与停止

- 设计审计 90 分钟检查点；实现与普通验证 4 小时检查点。慢 native 构建按原同配置保留累计
  时间，不用另目录或新卡重置原性能失败。每条路线只采用本轮冻结的一种机制。
- 普通失败最多三轮定向修复；架构/API/正常成本变化回 AUDITED。发现必须改公共等待/统计核心
  则留独立具名 blocker，不在本卡扩建。不能用更短 timer、忙轮询、更多 workers 掩盖进展问题。
- 重型任务全局串行；Windows/WSL/Air 最多 4 核/jobs、BelowNormal/nice10，native 编译最多2
  top-level jobs。Apple 仅 MacBook Air，先读取实际硬件/系统/工具链。
- API/算法和 correctness 固定成不可变 candidate 后，独立审查新有限测量入口。
  复用原 normal/holdout、四顺序、5%/10% workload A/A 与 n15→31 合同；不追加 epochs/n、
  不改尺寸/Provider/精度/估计量。新候选每未资格 profile 最多一次新入口资格尝试，任一
  preflight/A/A 失败即停止该 profile 并保原阻塞；不在模块内继续开发仪器。
- 本轮结构/功能修复和原 G09 聚合 P1 分开裁决。适用门未过写 BLOCKED/INCONCLUSIVE，
  不由单机有限提速或旧未变 CUDA PASS 代签新路线。

## 预声明验证

普通验证先 selector，仅改动 owner/真实直接消费者与公开闭包；旧未变叶证据按身份复用。

| 风险/交付 | 有限门 |
|---|---|
| 真实数学/进展 | 现有 hyle_cpu_vulkan_contracts / hyle_cpu_apple_contracts 和 pipeline；1/2 workers、两个 slots×65 步、两插入顺序、必要串行/纯 CPU 留出 |
| 新等待/通知 | 已完成/登记竞态、active waiter 时出现新 native ready、多 pending、同步/异步通知、generation/slot reuse，所有新路径用定向测试而非 sleep 证明 |
| 生命周期/运营失败 | cancel/abandon/close、旧 output、先错后仍在途；pool submit 拒绝/queued shutdown/running job、cold OOM/首 worker 拒绝；无早发布/无假退休 |
| 真硬件 | Windows Vulkan validation、Air Apple Metal validation、Air Vulkan；原数值/只读/页与状态 joint 直接消费者 |
| 宿主风险 | 改动 owner 的 ASan/UBSan/TSan 适用配置，runtime 不支持项按真实原因/替代门记录，不冒充未运行 PASS |
| C++ 交付 | 两个 hybrid 与 Apple native 受影响公开头、source/static/shared 迁址 consumer；未变 vendor/SDK 复用，不能拿旧映像签新 API |
| 性能/P0 | 冷 O(V+E)/有限 waiter/node 元数据、原 pool/thread 预算；warm allocation、观察独占、通知/锁/wake/copy；固定 candidate 后原完整 consumer 合同与资格 |
| 维护 | docs、closure --previous-ref、write-status、diff；最终统一 LOC/独立红队 |

SCOPED 以上门均待本轮运行；审计后补精确 filters/构建入口与适用风险，不默认通过。

## 七问与阶段

1. vendor/version/license 均不变，native 数学沿原成熟 owner；本轮不升级依赖或引入 fallback。
2. 只补原 J1–J4 的通知/等待/退休旅程；两种分支、两 slots 和串行留出共同约束形状。
3. 复用已有 file-stream 的 embedded BlockingJob destroy 边界、Sched CV/Event 与 Apple FIFO
   完成者；不复制执行器/定时器/资源权限。
4. 新线程/栈与冷节点费用公开，原所有性能结果保留；新收益待正式资格。
5. 普通公共面仍为 C++；内部 opaque callback 不构成 Native FFI 产品。
6. 按实际 headers/export/直接依赖检查 source/static/shared，未变依赖不全量重建。
7. Vulkan pool 借用及关闭顺序写在类型附近、guide 与公开 sample，Apple 保原普通调用。

## AUDITED：有限 FIX 与合同冻结

独立 Apple owner 审查与 CPU/Sched 审查已分别完成并与原设计对照；原 SCOPED 的候选方案
据此收敛如下。旧卡 J-F5 的“无新 public API/OS thread”仅描述旧候选，不能约束新需求掩盖
实际等待资源。本轮 Vulkan 明确增加 caller-owned pool 借用，费用与能力取舍一并公开。

| FIX | 唯一采用的机制 | 有限产品文件/直接验收 |
|---|---|---|
| W1 | Vulkan 每 episode 一个冷内嵌常驻 BlockingJob。Sched root 启动后先等 job started，才接受该 episode 任何 CPU/native 运行前缀。job 在私有 monitor 下接收原 Completion wait 请求，root 用原 fiber-aware CV 等响应，活动句柄由 job 独占；另有 ready native/CPU 可继续推进 | cpu_vulkan.h、cpu_vulkan/invocation.cc；真实两顺序/两 slots、1/2 workers，active wait 时新 ready、多 pending |
| W2 | Apple native invocation owner 内部按值三指针 notification packet；subscribe_completion(CompletionDependency, CompletionNotification) 为窄 private C++ export，普通 public API/Completion 不变 | 新 apple_completion_internal.h，apple_context.cc、cpu_apple/invocation.cc；已完成/inline/登记竞态、普通及 paged |
| W3 | Vulkan pool 拒绝/queued shutdown 发生在 started 前，零本 episode 运行前缀，Status 失败；started 后 job 已 RUNNING，原 pool shutdown 不取消它，它继续服务所有错误/取消的 native wait/needs_drain；最后 stop→destroy_done 后才发布终态/复用 | 无 poll/collect fallback、无 Scheduler worker 上的 native wait；cold/worker OOM、queued/running shutdown、cancel/abandon/close、旧结果 |
| W4 | Vulkan 唯一冷创建形状 create_invocation(Scheduler&, compute::Queue&, sched::BlockingPool&)，迁移所有实际 in-tree 调用；Apple 保原入口 | 公开头、contracts/pipeline/page/state joint、SDK sample/user guide；source/static/shared 与逐头 |

### Vulkan 独占、进展与费用

每 RunState 一个 embedded job/cell，warm 不创建 JoinState/job heap；原 pool 可懒创建线程，
cold worker/stack/锁与每 episode 一次队列入场计费。job started 前拒绝或 queued cancel 不会
留下本 episode 的 native 前缀；Scheduler root 未 admission 时也不投递 job。纯 CPU episode
不启动 native waiter。冷 prepare 的既有显式 native barrier 保持原独立合同。

running job 在 std CV 上等请求/stop，Sched root 在原 Sched CV 上挂起。请求、完成与回交都在
同一私有 monitor 下发布；job 处理期间 root 不 poll/wait/move/reset 该 Completion。
root 仍推进其它 ready/pending 项；有其它未桥接 pending 时可保原有界观察，不能把单 waiter
改写成通用 wait-any。native 完成通知的谓词/epoch 必须在一轮观察前 capture，避免 poll→park
间丢唤醒。所有 accepted 前缀都由原 checked wait/结果/退休处理，绝不靠 job cancel 签 GPU 完成。
stop/destroy_done 与 admitted root/ref handoff 阻止旧 episode 尾部写入新 episode。

pool 容量是调用方的有界进展预算：max1 可以排队，**不承诺任意容量下跨 slot 乱序完成**。
本轮主消费者两个活跃 slots 配 max2，另测 max1 的有限无环 liveness；单 Scheduler worker
不因此要求额外数学 worker。pool worker 不得同步等待一个依赖同池 waiter 的 Graph 结果。
pool/allocator/native contexts 必须存活到全部 Graph/job/native 尾部返回；通常先 close Graph，
再 drain Scheduler，最后 shutdown pool。并发 shutdown 的 queued/running 分支分别验证，
不将关闭中的 pool 悄悄换成另一执行器。native 永久无法证明退休时保留原等待与所有权，
Scheduler worker 仍可运行，不伪造成功。

### Apple hint 与寿命

成功订阅消费调用方一份引用，ready→release 恰一次，可 inline；失败不消费。只接受原 native
vtable 和现成 terminal hint，未知/重复 pending subscriber 返回错误；不为未知 Completion 加 timer fallback。
native finish 第一段锁只存 terminal/移出 native_uses，解锁完成 retire_native 后，第二段锁才
exchange abandoned/hint/notification 并发布 done；这同时覆盖退休窗口内晚到的 Completion
destroy。done 后仅原 wake、局部 callback 和 worker ref 尾部，不再写当前代状态，避免 late-inline
通知后新一代 bind 仍撞旧 native_uses。随后沿原
wake/deliver 尾部后 ready/release，不把 hint 当 actual retirement。Graph 在锁下先计 outstanding/retain，
解锁再登记；ready 只更新 epoch/通知，release 归还 outstanding/ref，尾部归零后才复用/完成。
epoch 在本轮 poll/advance 前 capture，避免通知落在最后 poll 与 park 之间。原 LeaseSubmission
与 native owner 仍负责实际资源退休；普通/paged fragment 返回原 native Completion 已只读确认。

### 测试与阶段约束

沿原 hyle_cpu_{vulkan,apple}_{contracts,pipeline} 及 native page/state joint；新增定向 tests
只覆盖本表新增的竞争、拒绝和资源尾部风险，不复制一套 fake 数学去代签真实设备进展。
Apple 内部 seam 可用 owner 级定向 fixture 验恰一次/inline/拒绝；真实 Graph 流水仍必跑。
Windows Vulkan validation、Air Metal validation 与三个 host sanitizer profile 适用风险保留。
每个 selected source/static/shared 必须绑定新源码及重新链接后的新 owner，未变叶与 shader DATA
按精确 hash 复用。公共 API 仅 W4，组件图不改，不重开公共等待/构建/统计核心。

当前 AUDITED，允许上述有限实现；候选/实际验证/性能入口/终审和 LOC 尚未签发。

实际调用映射后 W2 缩域：`cpu_apple::drive_stage` 的 upload/download 是原同步复制；唯一
通知 caller 收到的普通/paged fragment 都是原 native Invocation Completion。CopyBatch 的
FIFO/source lifetime 不需要本轮通知 seam，故去掉无实际 caller 的 copy packet/订阅/三项新增
copy 测试；原 copy 产品/测试保持 KEEP。本轮 native 四项定向测试另覆盖合法 inline terminal，
不以测试行数推动能力。此收窄不改变已冻结公共形状或扩大 owner。

### 首个普通 checkpoint 与有限纠正，2026-10-17

实现 checkpoint `6c6222e51` 已固定 W1–W4，仍为 AUDITED，并非性能 CANDIDATE。
两个独立源码审查无 must；Windows source 的 Vulkan `compile.cc`、`invocation.cc` 及新 shared
owner 已真实编译链接。首个完整 fixture build 随后因新私有 CMake wrapper 遗漏 encoded
消费者原本需要的 `GTest::gtest_main` 而停止；尚未运行任何 fixture/device/A/A/P1，不能签功能 PASS。
同时 compiler 指出原 contracts 日志的 `%zu` 与 `uint32_t` 不匹配，本次只校正该格式及显式转换。

原准备的 export-subst 身份拒绝、plan 的错误 DATA 路径拒绝及首次 build 失败全部保留。
只读补充已证明实际 vendor 目录 152 份 shader DATA 与原同配置资格逐项相同、内层 generator
无工作；56 个 C++ 单元均有实际 compdb/dirty 理由，未重编 shader/CUDA 数学。configure/manifest/
两项 dry-run 的已测命令 265.985 秒，首次 build 95.141 秒，累计 361.126 秒；不含审查/编排。

下一有限普通入口只修正私有 fixture 对原 GTest 的借用、实际 DATA 路径、设备 unit 的环境与
分阶段结果文件；继续原八个消费者，encoded 必须实际执行 Vulkan 三项/Apple 四项 tests，
共享/静态 SDK 仅复用同机真实 GTest target export，HYLE 头和库仍只来自新迁址 SDK。
产品算法/API/正常成本与 W1–W4 不变，既有失败、三轮定向修复及四小时检查点预算均继承。

本轮 selector 静态计划另揭示 Windows 错选 Apple hybrid correctness 的既有 policy 漏项。
由[独立维护工具任务](../tooling/validation-selector-apple-hybrid-platform.md)单列一行 Darwin
适用性修正与既有平台表测试，旧 policy 的 Windows/Linux 两项红证据及完整 60 tests 绿证据已存。
不以空 target 签 correctness，也不将工具单测代签 G09；后续用真实 VS 环境重新 plan/run。

### 第二普通 checkpoint 与下层缺陷，2026-10-17

固定 checkpoint `e2a55aaab` 的 Windows source build 与八个真实消费者全部通过，包括两个
slots×65 步、第二 native 支路和 max1 pool 有限进展。实际设备层运行在第四个 encoded fixture
停止：三项 GTest 中两项通过，故意 NaN 的 failed-encoder 项在原测试强制要求成功的同步
阶段收到原数值域错误。原 API 允许 bind/submit 较早拒绝已知失败，不能只接受最终 wait 报错。
有限 consumer 纠正须接受这些合法原边界，并严格保留原错误、尚在 caller 手中的 owning result
观察、accepted prefix 实际 drain、未提交 writer 归还、不发布坏 page 和旧 reader 字节不变。

设备前三个消费者通过、第四个失败、后四个未运行；四份日志无 Khronos VUID/SYNC-HAZARD，
但保留 loader 的 OBS manifest 错误和弃用警告，不能写原始日志零错误或推断 OBS 为因。
[普通 checkpoint 2](../../../../evidence/correctness/inference-g09-completion-wake/ordinary-checkpoint-2.json)
保留完整冻结入口、实际计划、原日志/JSON、artifact/DATA/源码身份与独立复核；累计命令时间
718.405 秒，Air/SDK/sanitizer/A/A/P1 仍未开始，原失败和四小时/三轮预算均不重置。

这次源码调查另发现原 Vulkan 完成者在权限退休结束前发布完成的独立竞态。按原 G08 的寿命
reopen trigger，以[有限 native 子卡](../compute/inference-g09-vulkan-retirement-barrier.md)
将 `compute_route_vulkan` 从 CLOSED 重新 SCOPED；它是 W3 的原 owner 修正，不能把新问题
留在 Graph 外层掩盖。该竞态不是本次 NaN fixture 失败的已证原因。子卡完成 AUDITED 之前
不得修改该 native owner；主卡三个 owner 仍 AUDITED，不恢复旧 profiler/性能重试入口。

### 第二次有限纠正的实现边界

native 子卡 AUDITED checkpoint `602614c8b` 后，V1–V3 只改原 private Runtime/Run 和既有
lifecycle 测试。新增每 Runtime 一个 CV、终态每次一次额外 Runtime 锁往返；正常 lone observer
不等 CV，非终态 poll 不通知，无新增 heap/容器/线程/native fence/checked readback。
实际 sizeof 未取数；不把源码固定空间账包装为实际字节测量。原 W1–W4 API/算法不变。
encoded consumer 只修合法错误边界及原 owner/permissions 的实际收尾断言，不放宽数值错误种类。

普通 native 验证的轮次明确保留：第一轮在 fixture build 遗漏原 GTest 依赖处停止；第一次
定向纠正后的第二轮八项 source 通过、设备 encoded 失败；本次为第二次定向纠正，最多再允许
一次有审计依据的普通纠正。此前 export-subst 准备与 DATA plan guard 两次失败均发生在对应
native build/设备执行前，原 receipt/时间完整保留，不作为通过，也不因上述轮次口径被删除。
全部工作仍受原 14:08 +0800 起四小时检查点约束，新的 native 子卡与 v3 不重置任何成本。

新普通 v3 仅增加原 Vulkan lifecycle TU 的三项有限 filter（原 managed 独立进展一项、新
retirement 两项），并按不可变源码逐名/计数冻结；八项公开 consumer、Apple 原四通知项、
SDK 八项加公开头与所有身份/152 DATA guard 不变。两次入口独立源码审查通过后才允许运行；
实际 configure/plan/build/device 仍分阶段裁决，不将入口静态审查当 correctness PASS。

有限性能入口的原公共 runner 在预验证非 PASS 后仍尝试首个 formal child，历史依赖私有
ready guard 拦截。以[独立工具任务](../tooling/performance-qualification-failfast.md)补唯一
qualify fail-fast 边界，保留 probe/PASS、统计和 manifest；相关65项 Python 测试通过，不由此
启动 P1 或签发 G09 性能资格。

### 第三普通 checkpoint：实际 device FAIL 与最后纠正

`adbe3a248` 的八个公开 consumer、三项 native GTest 在普通模式全部通过；设备层九进程
和三项 encoded/三项 native GTest 的数学断言也通过，但 encoded 正常单请求纯 native
分支出现一条真实 resetFenceInUse VUID，device 聚合必须 FAIL。
[checkpoint 3](../../../../evidence/correctness/inference-g09-completion-wake/ordinary-checkpoint-3.json)
保存九份实际层激活、唯一真实 VUID、176 行 raw loader 错误、原 receipt/身份及全部费用。
累计命令 958.640 秒，所有 native 停止；Air 仅完成同源 sync，入口 PATH 缺目录的拒绝亦保留。

原 native 子卡以 `3ed603a48` 预签最后 V4：跨 queue 的 compute handoff 不证明原 transfer
fence signal 已结束；必须观察实际接纳的两枚 fence，拒绝 handoff 则去重为仅接纳的一枚。
原 raw 不能唯一定位句柄，V4 由规范/可达源码的独立缺陷驱动，不能预签此次 VUID 已消失。
实现限原两文件，增加确定主 ready/transfer proof withheld、真实双 fence wait、65 次变输入
上传与 partial handoff 单 fence 的一项 native 回归；其余 V1–V3/W1–W4 不变。
Windows/Air native 默认四项须全部 RUN，SKIP 不能通过；只有预先证实不适用才另冻结具名子集。
下一 v4 继续原所有门与预算，再次普通失败即停止，不追加 profiler/环境/入口变体。
