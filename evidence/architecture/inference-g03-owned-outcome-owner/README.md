# G03 来源到原结果槽的单一观察 owner 反证

2026-10-14，产品固定 9404af694/terminal16d677，compute_batch SCOPED。本证据不
改产品，不签完整 G03。[限制](limitations.json)、[累计记录](result.json)、
[原始编排/flags/日志](raw.zip)，原型[观察 owner](pending_outcome_owner_probe.h)、
[Apple](pending_outcome_owner_apple_probe.cc)、[Vulkan](pending_outcome_owner_vulkan_probe.cc)，
[Apple 结果](apple-admission-2.log)/[编译](apple-compiled-inputs.json)、
[Vulkan 结果](vulkan-admission-2.log)/[编译](vulkan-compiled-inputs.json)。

MacBook Air Mac16,12 / arm64 / macOS27.0.1 / Apple Clang21，最多四 jobs/串行。
沿原 Runtime/FIFO或queue/fence、原 F1 exact guards；同一 CopyBatchUpdate 两控制
移到唯一观察者，保原 OwnedResult/ResultSlot。原 native结果和 sources 的 lifetime
由既有 CompletionRetention 关联，没有第二 executor/queue/completion/result slot，
也没有新增普通 native 对象字段。prototype 使用现有 static artifact，不重编产品。

Apple 实际三场景、Vulkan 实际五场景：成功保同一 payload，旧 result lease 仍阻止
原 native slot reuse；source 在 suffix 接纳之后请求取消，聚合返回 CANCELLED，
重复观察保持该错误、不发布结果；abandon 后 source仍活到实际 suffix 退休，再释放。
Vulkan 额外用实际 GetFenceStatus OOM 验证 unknown不释放 source/guards、最终实际
fence wait 保第一 OOM/no publication，以及该 unknown 情况下 abandon 的正确保活。
actual submit/fence/wait 未替换；其 pending 留存仍仅遮住成功退休观察，不声称阻塞
真实 GPU shader。Apple 用原 GPU checkpoint 留住真实 queued prefix。

原型将 source borrow 的观察推迟到 suffix 实际退休之后，保原 prefix/guards。它只
是 result()/wait() 的 shape/lifetime 反证。真实 Graph native fragment 还先消费
completion()（Apple/Vulkan原 invocation.cc），产品必须在同一 completion语义上
合并来源；不能直接把此 façade 当完整接口。原型 const_cast 只访问实际非 const
对象，产品必须使用经审计的 trusted Result access 接缝。

原型每 episode 一次 make_shared 冷元数据；没有 warm 无分配/OOM预算证据。输出
只核原 native arity与identity，没有独立数值 oracle/64steps。未证明 CPU/CUDA
owner、未知 CUDA cached-error 的 blocking drain、所有 producer/accepted phase 的
actual retirement、pending page初始化/late native output/全七路线。原 common/
Vulkan CopyUpdate.wait 的错误不总能单独证明 phase.actual retired；后续以原
retired/pending及原 native callbacks裁决，不能按一个 Status假定退休。

两轮累计 11.316 秒，全部最终 exit0。八场景输入/source-target编译 SHA核对，原
数学/P1/全模型/全局构建未运行。[活动观察接缝 scope](../../../docs/implementation/tasks/compute/inference-g03-owned-outcome-observation.md)
继续固定有限 API/FIX；完整[主卡](../../../docs/implementation/tasks/compute/inference-g03-pending-page-consumption.md)
仍SCOPED，源码/SDK普通consumer尚没有这些新公共能力。
