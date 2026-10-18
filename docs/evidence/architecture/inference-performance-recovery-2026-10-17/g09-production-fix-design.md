# G09 完成通知与阻塞等待桥：最小产品修复评估

2026-10-17，输入 HEAD `8e04eb0c0`。本页汇合独立 Graph/完成所有权审查与 CPU/Sched
审查，只读评估候选设计；**没有修改产品、公开 API、registry、SDK 或公共性能仪器，也没有
运行新的 GPU 工作**。它不是产品 closure card，不把诊断代码直接提升为产品。

## 诊断支持什么

R29 的四个 execution/allocation 组合均完成原 oracle、退休、记录和身份检查。独立重算的
shape0 原观察/直接 native wait 的中位耗时比为 1.467/1.508/1.536/1.413，shape1 为
1.380/1.359/1.468/1.387；shape2/3 无干预记录，也没有同方向的大幅变化。原件见
`g09-native-wait-29/run/receipt.json` 与四个日志，独立汇总见本目录
[native-wait-analysis.json](native-wait-analysis.json)。

这支持固定 Windows Vulkan 独立支路的**观察/唤醒策略有实质成本**。原等待是 fiber-aware
`sched::ConditionVariable`，诊断则阻塞一个 OS worker，差值不能全部解释成纯 1ms 延迟或
GPU 精确完成时间。它没有解决原 A/A 资格，也不能外推 Apple；直接阻塞 Scheduler worker
不能作为单 worker、多 invocation、一般 DAG 的进展保证。

## 已有机制与不能混淆的职责

| 事实 | 当前源码 | 对修复的含义 |
|---|---|---|
| Vulkan frontier 已无 ready native job、仍有 pending 时每 1ms 重新观察 | [cpu_vulkan invocation.cc:543](../../../../src/graph_route_cpu_vulkan/invocation.cc#L543) | 应补真实完成驱动或等待桥，不靠更短 timer、更多 epochs 或忙轮询 |
| Apple hybrid 的普通 pending 路径同样定时观察 | [cpu_apple invocation.cc:201](../../../../src/graph_route_cpu_apple/invocation.cc#L201) | 有相似结构，但尚未取得本轮 Apple 因果结果 |
| `OutcomeSources::subscribe_retirement` 要求预分配、不能等待或分配 | [dependent_completion.h:9](../../../../include/hyle/compute/detail/dependent_completion.h#L9) | 是寿命协议，不是自动 GPU observer |
| 实际订阅仅逐个给 source Completion 加 retention | [retained_completion.cc:461](../../../../src/compute/retained_completion.cc#L461) | 保留的 common Completion 仍需 poll/wait/析构才能 settle；单靠订阅不能让 frontier 自行醒来 |
| `attach_retention` 把 attachment 存在 common handle；`settle/reset` 才交给 backend destroy | [completion.cc:51](../../../../src/compute/completion.cc#L51)、[completion.cc:174](../../../../src/compute/completion.cc#L174) | 不应把 active notification 与既有 actual-retirement attachment 混为一项能力 |
| DependentCompletion 的 poll/wait 显式推进 sources；proxy 销毁且 suffix 真退休后才 abandon sources | [retained_completion.cc:213](../../../../src/compute/retained_completion.cc#L213)、[第291行](../../../../src/compute/retained_completion.cc#L291) | 为唤醒而提前 abandon 会改变 suffix/source 借用顺序，不能作为通知捷径 |
| Apple 原 Runtime 已有 FIFO worker；原 Job 做 native evaluate/synchronize 后 finish | [apple_runtime_internal.h:19](../../../../src/compute_route_apple/apple_runtime_internal.h#L19)、[apple_context.cc:186](../../../../src/compute_route_apple/apple_context.cc#L186)、[第457行](../../../../src/compute_route_apple/apple_context.cc#L457) | 可以在已有主动完成者上增加窄内部回调，不需要新线程 |
| Vulkan Runtime 只有 pending 链；collect/drain 的调用方实际执行 fence/region 观察 | [vulkan_internal.h:191](../../../../src/compute_route_vulkan/vulkan_internal.h#L191)、[vulkan_run.cc:599](../../../../src/compute_route_vulkan/vulkan_run.cc#L599) | ggml/native Vulkan 没有一个可直接挂接的后台 observer；只加 callback 字段不能取消轮询 |
| Vulkan 完整退休包括 final fence、managed region/tails、状态检查、native uses 和回调 | [vulkan_run.cc:200](../../../../src/compute_route_vulkan/vulkan_run.cc#L200)、[第288行](../../../../src/compute_route_vulkan/vulkan_run.cc#L288) | 不能拿 kernel end 或单 fence signal 冒充 public Completion/资源退休 |
| IREE NativeProgram 已有原 proactor notification，wait/poll 包含 callback 返回 | [native_program.h:119](../../../../src/compute_provider_iree_vulkan/native_program.h#L119)、[native_program.cc:173](../../../../src/compute_provider_iree_vulkan/native_program.cc#L173) | 只适用于已经走该 owner 的实际路径，不为普通 ggml Vulkan 强行引入 IREE 或切换数学 Provider |

结论是：**Apple 可用窄内部通知做到不新增线程、不增加公开 Completion 字段；当前普通
Vulkan 不能同时满足“没有新完成驱动、没有外部等待执行器、没有周期观察”这三个条件。**
共享 Compute 的 retirement 协议不能凭改名填补这个主动进展来源。

## 路线一：Apple 原 owner 的内部完成通知

最小 owner 为 `compute_route_apple` 与 `graph_route_cpu_apple`；若涉及 F2 则仍由原 copy
owner 完成。共同 Compute 只复用已有 opaque fn-pointer/lifetime 形状，不反向依赖 Sched，
不修改公共 Completion layout，不新增 worker、Queue 或通用 callback runtime。

1. Graph 在冷创建时准备有限 subscription node，含精确 episode/generation、目标 RunState
   引用及 opaque wake callback。callback 的函数体由 Graph 拥有，Compute 叶只接收
   `void*` 与不抛函数指针，不包含 Event/Waker/Scheduler 类型。
2. 在原 invocation/copy 的完成 owner 内登记；登记和“已经完成”的判定使用同一原 monitor。
   要先准备完整 node 和所持引用，再调用登记函数，因为已经完成可能同步回调。登记不能
   等待、热分配或偷偷执行数学，拒绝须保留原 native prefix 的所有权。
3. 原完成者在正确语义边界摘下 node，并在释放 native/queue monitor 后回调；callback
   只记录 epoch/状态并通知原 Graph CV/Event，不调用 Graph 数学、prepare、wait 或重新提交。
   根 fiber 再沿原 poll/finish/read 裁决结果。只有 notify、没有 epoch/ready 谓词，会被当前
   frontier 的 predicate 忽略，仍不能去掉 timer。
4. 不能直接把 Apple 的 `done==1` 当作“retirement callback 已返回”。当前
   [finish:469](../../../../src/compute_route_apple/apple_context.cc#L469) 先发布 done，随后才 retire
   native uses、wake、deliver；新内部协议应明确它是 ready hint 还是 actual retirement。
   若承诺后者，须把订阅的 ready/retired phase 放在完整原退休边界，或用私有阶段记录覆盖
   这个窗口，不能仅检查现有 done 就提前释放订阅引用。
5. cancel/abandon 不等于取消回调寿命：旧 node 必须存活到原完成者不再访问，native owner
   和 Graph RunState 各自保有需要的引用。回调 generation 不能唤醒下一次 episode；node
   在 release 尾部完成前不得重绑/复用。关闭顺序必须涵盖 callback tail，不能在 callback
   内 destroy/re-invoke 原 native slot，亦不能持反向 Graph/native 锁等待彼此。

这是一条可施工的窄方案，但不是当前已有 API，也未签实现/资格。是否需要单订阅或有界多
订阅，应由真实 composition owner 和既有 observers 冻结，不能无需求制造通用事件总线。

## 路线二：Vulkan 复用既有 BlockingPool 的有界等待桥

本节汇合 CPU/Sched 审查。最小 normal FIX 可只在 `graph_route_cpu_vulkan` 放一个冷准备
waiter cell/RunState；两个 invocation slots 各有自己的 cell，不做全局单 cell 争用。
原数学、native queues/fences、frontier、LeaseSubmission、输出 owner 均复用。

已有模板就是 [file_stream.cc:355](../../../../src/compute_stream/file_stream.cc#L355)：固定 slot
内嵌 `sched::detail::BlockingJob`，run 执行阻塞工作，**destroy callback** 才发布 job_done，
根 fiber 用原 CV 挂起。[BlockingPool worker:104](../../../../src/sched/blocking.cc#L104) 明确先
run 后 destroy；普通 `spawn_blocking` 每次创建 JoinState/job，不能拿它宣称零热分配。

当前 [create_invocation](../../../../include/hyle/graph/cpu_vulkan.h#L110) 只接 Scheduler/CPU
Queue，Scheduler 不拥有 BlockingPool。因此借 caller pool 需要一个明确的冷注入边界，
例如增加已有 `BlockingPool&` 参数或窄 invocation options；这不要求修改 BlockingPool
或公共 Scheduler 本身，但也不能谎称完全不改变 Graph 调用形状。不要自建隐藏池、偷 IO
私有池或用进程全局 runtime 代替所有权合同。若本轮拒绝任何注入边界，就应保留这个路线
未决，而不是把不存在的主动 Vulkan 通知写为已完成。

建议冻结的最小 normal 域：

- 仅在原 R29 前提下 arm waiter：已接纳一个 native prefix、暂无 ready native job。cell
  保存原 job ID、原 Completion 指针、状态/done、执行与否以及 RunState 引用。
- pool 接管期间，root 不得 poll/wait/move/reset 同一 Completion。common `wait() const`
  实际会 settle 可变 handle 字段，不能因 const 就并发观察。job 的 destroy_done 经原
  monitor 发布后，root 才接回原 poll/finish/read/清理。
- root 不能 arm 后只 join(done)：CPU lane 可能生成新的 ready native job。等待谓词应
  同时包含 waiter_done、readyNative、cancel/error，保持原 ready admission；active cell
  的那个 handle 仍跳过观察。后来出现其它 pending 时，最小方案可对其它项保留原观察
  路线，不把“一次单 prefix 诊断”扩成通用 wait-any runtime。
- pool.submit(false) 或 shutdown 取消 queued job 只表示等待任务未执行，不表示 GPU
  prefix 退休。cancel_fn 不在 shutdown 线程阻塞 wait；destroy_done 通知 root 恢复原
  poll/collect，原 accepted owner、native uses 和错误必须保留。
- outer close/cancel 先停接纳，join active waiter 的销毁边界，再处理其它 accepted prefixes。
  现 [drive_graph 清理](../../../../src/graph_route_cpu_vulkan/invocation.cc#L575) 仍有直接
  Completion.wait，必须一并裁决单 worker 故障路径：纳入原 offload 或保留能进展的
  fiber-aware poll/collect fallback，不能仅优化成功主路。
- pool 必须活过 Graph、wait job destroy 和全部 accepted work；先排空 Graph 后再销毁池。
  [submit:161](../../../../src/sched/blocking.cc#L161) 可能因 shutdown/首 worker 分配失败拒绝；
  [shutdown:242](../../../../src/sched/blocking.cc#L242) 会取消 queued jobs、等待 running jobs。
  有限共享池也须排除“所有 pool workers 同步等待本 Graph 结果，而 Graph waiter 排在其后”
  的依赖环，不能靠默认 512 线程或无限扩线程掩盖。

这样不增加一种执行器，不改变 Compute 叶依赖方向，并把新增线程/堆栈预算交给已有 caller
pool。它仍有线程启动、队列、锁和唤醒成本，应明确 cold/warm 与 pool 容量，不能预签低成本。

## 最小验证与产品裁决边界

开始写产品前，由真实 owner 合法登记/回退并冻结一条路线、有限 API/FIX 与风险门；本页不
产生施工授权。不能一次为了“统一”扩共享 Completion、所有后端、公共 Scheduler 和新观测
系统。以下是所选路线必须覆盖的直接风险，不是新的全库审计：

| 风险 | 最小验收 |
|---|---|
| 单 worker、两个 slots/多个 invocation | CPU 独立支路、native pending、另一 invocation 与新 ready native job 都能推进；不用多加 Scheduler workers 隐藏问题 |
| 已完成/同步完成与注册竞态 | 完成早于登记、登记与完成交错、callback inline、取消同时到达；恰一次通知与引用归还，无丢唤醒 |
| 多 pending、乱序与 bounded capacity | active waiter 之外仍可接纳/观察；较晚接纳但先完成的独立支路不被错误串行；容量/submit 拒绝可解释 |
| cancel、abandon、close、pool shutdown | queued 未执行与 running 分开；接受后错误/unknown 不早释放；旧 output/lease 保活；关闭不销毁仍在 native/callback/job 尾部使用的 owner |
| 生命周期与 generation | 每个 node/cell 不在 release/destroy 尾部前复用；旧 episode 通知不能污染新 episode；native/Graph/allocator/pool 借用合同可发现 |
| 失败与资源 | 冷逐分配 OOM、首阻塞 worker 创建失败、已接纳后的 native/观察错误；保原 first error/实际退休，不伪造 ready |
| 成本与交付 | warm 无重复 job/JoinState heap；记录真实 pool/thread/wake 成本；选定 owner 的 source/static/shared、逐头及真实 consumer，不重建全库 |

只在固定 API/算法和 correctness/生命周期通过后，才按原工具卡取得各平台自己的 A/A/P1
资格。R29 的 Windows Vulkan 因果证据不改变 CUDA 已有 PASS，也不替 Apple/Vulkan 未跑门
签字。任何产品修复完成与正式性能结论都仍未在本页签发。
