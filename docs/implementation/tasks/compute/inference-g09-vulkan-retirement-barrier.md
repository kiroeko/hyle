# G09 Vulkan 原完成者退休发布修正

状态：**AUDITED，2026-10-17**。SCOPED checkpoint `508551f0c`；owner `compute_route_vulkan`；baseline
`e2a55aaabdd06349f2e2d5efb82423d34e30305c`。本卡是
[G09 完成驱动修正](../graph/inference-g09-completion-wake-correction.md)的有限下层纠正，
不另开性能诊断或重置原预算。原 [G08 terminal](inference-g08-final-closure.md)保留。

## 缺陷、消费者与范围

合法 reopen trigger 是源码可确定重现的原完成/权限寿命缺陷，而不是性能点值：
`vulkan_run.cc::observe` 在 Runtime 锁下清 `pending`、摘链并移走 `native_uses`，随后
才在锁外 `LeaseAccess::retire_native`。另一观察者可在权限退休结束前返回 Complete；
已在锁外 fence/IREE wait 的第二 observer 同样可回锁走提前完成路径。
这破坏 checked completion 后归还/再次借用同一 `max_batches=1` pool 的正常旅程。
shared_ptr 只保对象寿命，不能证明权限已归还。尚无证据表明此次设备 NaN 用例失败由此引起。

主消费者是 G09 两个 slots：pool waiter 等原 Completion，另一个 slot 的原 submit/collect
也可观察同一 Runtime。正交消费者是原 native Invocation 的单 slot、真实 GPU 数学、原
LeaseUseBatch 和 checked wait→release→65 次复用，不依赖 Graph/模型/新 waiter API。
普通 public Context/Program/Invocation/OwnedResult/Completion/LeaseUsePool 形状不变。

有限范围仅 `vulkan_internal.h`、`vulkan_run.cc` 和既有 `lifecycle_test.cc`。
不改数学、copy/page owner、公共 Completion/Sched/BlockingPool、构建/统计核心、vendor 或线程数。
原 native fence wait 释放 Runtime mutex 的合同保留。

| 能力族 | 准入与层级 | 当前不足 | 承载 |
|---|---|---|---|
| 原生完成与权限退休 | 原子机制；原 checked Completion 与 NativeLeaseInputs | PARTIAL：正常原生完成先于权限 retirement 发布 | SELF FIX，原 Vulkan Run 是唯一完成 owner |
| 资源/错误/lifetime | 原 LeaseUseBatch、CompletionLifetime、Runtime close | 已有协议复用，须覆盖并发 observer 与 late abandon | BORROW，不能复制权限或另建 executor |
| Graph/请求编排 | 原 hybrid consumer | 等待原拥有完成即可 | KEEP；请求/模型策略仍 APPLICATION |

## 预算与有限验证

继承主卡 14:08 +0800 开始的四小时实现/普通验证检查点、最多三轮定向纠正与全部既有失败；
本卡设计审计最多 20 分钟，不以子卡、目录或新提交清零。只采用一种退休发布机制。
需要新公共 API、容器/线程或无法固定单一 terminal 顺序时停止并裁决，不能用 timer 规避。
重型 native 全局串行，四逻辑核以内、BelowNormal/nice10、compile -j2；Apple 仍只用实际 Air。

预声明：既有 private PFN/LeaseAccess seam 配真实 GPU completion，resource-owner release gate
确定卡在 active_batches-- 之前；该窗口 poll 必须 Pending，wait/collect(drain)/close 与 abandoned
lifetime 不得提前完成。覆盖已登记第二外锁 observer、数值/native 错误、晚到 destroy、旧结果与
65 次 slot reuse；使用 guard 释放测试 gate 并 join，不靠 sleep 或 fake 数学签 GPU。
运行原 Vulkan lifecycle 的适用 filters、八个 G09 真实消费者、Windows Khronos validation，
Air Vulkan 普通消费与适用 host sanitizer；与主卡共享 source/static/shared、selector、docs/closure
和最终独立审查门。未运行项保持 UNPROVEN，不把纯源码证明当 runtime PASS。

性能 tier 为 P1，合并原 G09 完整 consumer 的固定资格合同；P0 先审每 Runtime 固定 CV、
锁/通知/metadata 与正常 fence 外锁等待。不另造微基准或新性能资格尝试，不改变原阈值与失败预算。

## 七问与当前裁决

1. vendor/版本/许可无变化；原 Vulkan/IREE/GGML 与原错误 owner 继续承担数学和设备故障。
2. 两个真实消费者约束 checked completion 后可复用；不扩全库能力或模型域。
3. 复用原 mutex、Run、observer、retiring、LeaseUseBatch、CompletionLifetime；不复制 scheduler。
4. 冷固定空间、正常非阻塞 poll、无新增 native wait；成本与完整 consumer 资格仍待审计/验证。
5. 无新 C/FFI/export/API 或跨语言需求。
6. 原 native 叶 source/static/shared 重新构建链接，直接 hybrid/page/state 消费验证；未变依赖按 hash 复用。
7. 原 public 使用形状不变；新实现必须使原 checked wait/close 的已有寿命承诺成立。

## AUDITED：唯一有限 FIX

实现者与独立 Graph 寿命审查均确认以下最小形状，未发现必须增加 public API/generation 的事实。

| FIX | 冻结机制 | 验收 |
|---|---|---|
| V1 | 每 Runtime 一个私有 condition_variable，复用原 Run.retiring；pending/list/pending_owner 保留至权限及 abandoned 交付尾部结束 | gate 窗口 poll Pending，wait/drain/close 无早完成 |
| V2 | 唯一退休者先 drop 自己的 observer，再等其余已登记外锁观察归零；各 loser 回锁先合并本次错误，再 drop/notify，不能二次 checked/retire | 已登记第二 observer、first-error 不被 OK 权限发布掩盖 |
| V3 | 原 retire_regions/checked/cancel 与 terminal 优先级在 observers=0 后确定，外锁 retire_native；第二锁收 late abandoned，无 packet 则同锁直接 publish，有 packet 则外锁 deliver 后同锁 publish | abandon/lifetime 恰一次，原 native 错误和数值拒绝，65 步复用 |

每个 IREE wait/poll、最终 callback tail、native fence 外锁回点均重查 retiring/pending。
poll 有其它 observer 时只记录本次结果并返回 Pending，不能 claim 后 CV 等待或留下无人推进的
retiring；wait/drain 才能 claim 后等零。遇已有 retiring 的 poll 立即 Pending，wait/drain 直接
CV 等 publication，不落入原 1ms 的异常未证 quiescence backoff。notify 仅用于 observer 退出/
retirement 发布，不新增每 kernel observer、容器、heap、线程或普通 timer。

pending destroy 只 handoff abandoned/completion_owner，不清 retiring；已 publish 的正常 destroy
独占自己的原 deliver 尾部并清理/通知。publish 后旧退休者不再写 Run，防止覆盖新代/正常
destroy。原 completion owner 与 closed admission 继续限制复用，不新增公共状态机。

定向测试在真实 GPU fence 完成后，以原 private resource-owner 回收 gate 卡住 permissions
retirement；受控 episode 必须释放其它 LeaseDomain refs，先确认 gate 命中和 active_batches=1。
gate 只等待外部测试信号，不从回调同步 wait/close 自身；失败 guard 先释放 gate 再 join。
原内建 owner 回收路径未查到同步 close 本 Runtime 的实例；同一退休 callback 同步等待自己的
retirement 不在原可组合合同内，本轮不为假设自环增加 thread identity/public 重入 API。
非阻塞 collect 重入必须安全返回 Pending。

允许上述三文件有限实现与原消费者测试纠正；实现/source/运行审查、candidate 和 P1 仍未签发。
