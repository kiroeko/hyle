# DEFER_TASKRUN 接回 debug（EEXIST 线程绑定 + kCancelToken livelock —— 已定位并修复）

> 调查日志（已结案）。关联:[`../performance/benchmarks-io.md`](../../implementation/evidence/performance/benchmarks-io.md)("DEFER_TASKRUN 接回"节)、[`../architecture/io-design.md`](../../architecture/io-design.md)、commit `285c68a`。
> 结论先行:DEFER_TASKRUN 需 SINGLE_ISSUER,而 SINGLE_ISSUER 把 ring 的 issuer 绑到**创建线程**;HYLE 主线程建 ring、worker 线程 reap → worker 首 enter **EEXIST**。修:`R_DISABLED` 建禁用 ring,worker 首次 reap 时 `ENABLE_RINGS`(per man,enable 调用者成 issuer)。修完又撞第二个根因:net_cancel 的 cancel-op CQE 用 `kWakeToken` → reap 跳过 inflight 递减 + 重武装 wake-read → DEFER 下 **livelock**;改用专用 `kCancelToken`。

## 0. 背景
DEFER_TASKRUN(+SINGLE_ISSUER,≥6.1)把完成 task_work 全延到 GETEVENTS enter 才跑,理论最契合"每核一 ring、单消费者"。曾因"挂死基本 send/recv"撤下(见 io-pipelined-hang-hunt 的 reap-path 次要因),本轮认真接回。

## 1. 根因一:SINGLE_ISSUER 线程绑定(探针 errno=17 铁证)
加 DEFER 到 setup flag 后,`hyle_io_core_net_test` 的第一个用例 `CoAwaitSendRecvRoundTrip` 就挂。**先疑自己**:先前撤下时我归因"reap 路径 submit-only enter 不带 GETEVENTS",以为修好 reap 就行——但修了 reap-path GETEVENTS-peek 后仍挂。

**探针**:在 reap 的每次 `io_uring_enter` 后打 `errno`(env-gated)。铁证:
```
[defer] reap0-peek submit=1 r=-1 errno=17   ← 第一个 enter 就失败
```
errno **17 = EEXIST**。查 liburing `io_uring_setup.2`:
> IORING_SETUP_SINGLE_ISSUER ... The submission task is either the task that created the ring, or if IORING_SETUP_R_DISABLED is specified then it is the task that enables the ring. The kernel enforces this rule, failing requests with **-EEXIST** if the restriction is violated.

**真根因**:HYLE 在 `IoContext` 构造函数(主线程)建所有 ring,却在各 owner worker 线程 submit+reap。SINGLE_ISSUER 把 issuer 绑到创建线程(主线程)→ worker 首次 enter 违反,EEXIST。**reap-path 只是次要因,线程绑定才是主因**——我第一次撤下时没挖到底。

## 2. 决定性实验:独立探针程序验证 R_DISABLED 生命周期
不在 HYLE 里瞎改,先写 `defer_probe.cc`(标准 io_uring syscall,无 HYLE)验证方案:
| 步骤 | 结果 | 判读 |
|---|---|---|
| setup(R_DISABLED\|SINGLE_ISSUER\|DEFER) | fd>=0 | 可建 |
| worker 线程 ENABLE_RINGS | rc=0 | worker 成 issuer |
| worker 线程 enter(GETEVENTS) | rc=0 | **issuer 可 enter** |
| **主线程 enter** | **EEXIST(17)** | **非-issuer 被禁(正是 HYLE 的 bug)** |

方案确认:主线程建 `R_DISABLED` ring,**worker 首次 reap 时 ENABLE_RINGS** → worker 成 issuer。

## 3. 修法一(ring 生命周期)
- `init(entries, defer_worker_thread)`:opt-in。`true`(IoContext 跨线程 reap)→ setup 加 `R_DISABLED|SINGLE_ISSUER|DEFER_TASKRUN|SUBMIT_ALL`,置 `needs_enable`。`false`(bare Proactor 同线程)→ 现状不变。
- **wake-read 武装延到 enable 后**:原在 `try_init_uring`(主线程)武装 + enter → R_DISABLED 下失败、且绑错 issuer。挪到 worker 首次 reap:`ENABLE_RINGS` 成功后才填 wake-read SQE(此时 worker=issuer)。
- reap 顶部 `needs_enable` 块(一次性):`ENABLE_RINGS` → 武装 wake-read → 清 flag。
- `register_buf_ring` 在 `needs_enable` 时 return false(主线程 register 与 DEFER 互斥,仅 test/bench 命中、已自 skip)。
- 内核版本回退阶梯:defer→coop→submit_all→无。

## 4. 根因二:kCancelToken livelock(修完根因一后 net_cancel 挂)
根因一修好,net_test 全过,但 `net_cancel_test.TokenCancelUnblocksRecv` **挂死**。

**先疑自己 + 读代码**:net_cancel 的 cancel-op 自身 CQE 用 `Proactor::kWakeToken` 当 drop-token。而 reap 见 kWakeToken 的分支:
```cpp
if (c.user_data == Proactor::kWakeToken) {
  wake_fired = true;   // → 循环后重武装 wake-read
  ++head; continue;    // 跳过:不投递、不递减 inflight("永久 infra,从没计数")
}
```
但 cancel-op 是 `submit_one` 提交的(**`++inflight`**),它的完成却走 kWakeToken 分支 → **inflight 不递减(泄漏+1)+ spurious 重武装 wake-read**。

**为什么 COOP 下良性、DEFER 下 livelock**:COOP 下 spurious 重武装的 wake-read 完成不紧密循环;DEFER 下完成只在 GETEVENTS enter surface,重武装的 wake-read + 泄漏的 inflight 让 worker 在"有在途 op → 阻塞 GETEVENTS reap → 收到 kWakeToken → 重武装 → 又有在途"里打转。

## 5. 修法二
net_cancel 的 cancel-op CQE 改用专用 `kCancelToken`(net_timeout.h 定义,`~0 & ~(1<<1)`):
- 它经 `complete_timed` 的 `if(user_data==kCancelToken) return` 丢弃(不 route 成 slot)。
- 关键:它**不是** kWakeToken,故 reap 作**普通完成**处理 → **正常递减 inflight**、不重武装 wake-read。
- 这修好了一个 COOP 下就存在、但被掩盖的 **latent bug**(inflight 每 cancel 泄漏)——DEFER 把它逼成了显式 livelock。

## 6. 验证
| 面 | 结果 |
|---|---|
| 探针 chosen_flags | `0x30c0`(DEFER\|SINGLE_ISSUER\|R_DISABLED\|SUBMIT_ALL)真选中,非静默回退 COOP |
| 曾 EEXIST 挂死的 send/recv | 过 |
| net_cancel(曾 livelock)| 过 |
| pipelined 512 流高扇出 | 过(DEFER task_work 交付路径的真考验) |
| 9 套件 + WSL TSan | net/cancel/timeout/http/sched 净 |
| 双平台 | Windows IOCP net 3/3 + cancel 2/2(DEFER io_uring-only,IOCP 无影响) |
| 性能 | DEFER echo-1w ~1.35-1.47µs、4w ~0.51-0.71µs,与 COOP(1255/581ns)**同量级、无显著额外提升** |

## 7. 教训
1. **先疑自己要挖到底**。第一次撤下 DEFER 我归因 reap-path(次要因)就收手,漏了真主因(线程绑定)。探针 errno=17 + 读 man 才见 SINGLE_ISSUER 的 issuer-task 规则。**"修了一个原因还挂"= 原因没找全,不是修错了。**
2. **独立探针程序验证方案再动主代码**。`defer_probe.cc`(纯 syscall、无 HYLE)先证"R_DISABLED + worker-ENABLE_RINGS 可行、主线程 enter 必 EEXIST",才敢改 HYLE 的 ring 生命周期。省了在大代码里试错。
3. **一个新机制会照出旧代码的 latent bug**。kCancelToken 的 inflight 泄漏在 COOP 下潜伏已久,是 DEFER 的严格 task_work 语义把它逼成显式 livelock。**上更严格的机制 = 免费的旧 bug 探照灯。**
4. **正确接入 ≠ 有性能收益,必须实测**。DEFER 能跑对了,但单 worker loopback 本就每 drain 都 enter ring、几乎无 IPI 可省 → 与 COOP 同量级。诚实登记"机制就位待更重负载显收益",不吹成又一个 -33%。

## 8. 相关 commit
- (先前)`93b1e23` — COOP_TASKRUN+SUBMIT_ALL 落地(DEFER 首次撤下)。
- `eaa9254` — 定位根因一为 SINGLE_ISSUER(非 reap-path)+ reap-path GETEVENTS-peek dormant 就位。
- `285c68a` — **真修**:R_DISABLED + worker ENABLE_RINGS + wake-read 延后武装 + kCancelToken 修 livelock + 回填文档。
