# sched-06：协程胶水（co_await → fiber 让出）

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | sched-04（Waker 脊 / fiber yield / mem）|
| **状态** | ✅ 完成（双平台 3 Coro + 全 sched 41/41 + WSL TSan×3 净，协程帧跨 worker 迁移零 race）|

> **范围**：sched-design §4 的胶水——`co_await` 让出的是**fiber**（fcontext 切栈），不只是协程，故深调用链可 await 而无需每帧都是协程。`coro.h` header-only，建在 Waker 脊 / fiber yield / mem 上，**无外部前置**。
> **cloc**：生产 175（`coro.h`）+ 测试 130。

## 为什么是它（第一节定的分工要落地）

sched-design 第一节定了 fiber（有栈）作调度单元、C++20 协程（无栈）作异步外壳；第四节讲二者不是二选一而是分工（§4.2）。本步是二者的胶水。核心洞察：协程帧**活在 fiber 栈上**，所以一次 fiber 挂起/恢复自然冻结/解冻它承载的整条协程 await 链——不需要 continuation 链，深层库函数想让出就靠 fiber，写法顺滑靠协程。

## 核心决策：三个原语，co_await 落 fiber 让出

- **`Task<T>`**：frame 走 `mem::Allocator`——promise 的 `operator new` 用**分配器作协程第一参**的标准惯用法（`PromiseAllocBase`），**不提供默认 `operator new`**，故忘了分配器参数的 Task 协程**编译不过**（charter：处处显式分配，无隐藏堆）。lazy start（`initial_suspend` = suspend_always，driver 调 resume 才跑）；`final_suspend` = suspend_always 保帧活着让 driver 读 result，Task 析构销毁帧。
- **`SuspendFiber`**：`co_await` 让出 fiber——`await_suspend` 调 sched-04a `suspend()` 做 fiber-yield，**返 false** 使 wake 后协程续跑而非二次挂起。这是 §4.1 "co_await 落 fiber 让出" 的正确实现关键：返 void 会让协程**第二次**挂起（把控制交回 resume() 的调用方），而等待其实已经在 fiber 层发生过了——那是错的。
- **`CaptureWaker`**：两步 register-then-suspend——先把 fiber 的 Waker 交给异步源、再 SuspendFiber，消除"挂起早于注册"的窗口（`Waker w = co_await CaptureWaker{}; source.arm(op, w); co_await SuspendFiber{};`）。

## 关键：为何 await_suspend 返 false（fiber↔协程桥）

`SuspendFiber::await_suspend` 在 fiber 栈上跑。`suspend()` 设等待标志并 fiber-yield（fcontext 切回 worker），**直到** `Waker::wake()` 重新注入本 fiber 才返回。一旦返回，fiber（及其上冻结的整条协程栈帧）已复活——所以返 **false** = "别让协程继续挂起，立即恢复它" → 控制流进 `await_resume` 与 co_await 之后的代码。这样**一次 fiber 挂起覆盖整条 await 链**，协程从不独立于其 fiber 挂起。

## co_await 子 Task（有栈的极简 awaiter）

`Task<T>` 自带 awaiter：`await_ready` = done()，`await_suspend` 在**本 fiber** 上把子 Task `resume()` 到 done（其内部的每个 co_await 让整个 fiber yield、冻结父帧），返 false 不 park 父帧。因为 HYLE 是有栈的，fiber suspend 覆盖了子 Task 的内部 await，**无需 continuation 链**——这是 io-08 TLS 泵能用嵌套 Task 的原因（TLS 是首个嵌套 Task 消费者）。

## 对抗式自审（挑适用类）

- **① 正确性**：双平台 3 Coro + 全 sched 41/41——Task 分配器背书跑通、SuspendFiber wake 后续跑、CaptureWaker 两步无窗口。
- **③ 规范**：`coro.h` header-only；frame 走注入 allocator（无默认 operator new，忘参数编译失败=编译期强制）。
- **④ TSan**：WSL TSan×3 净——**协程帧跨 worker 迁移零 race**（帧活在 fiber 栈，随 fiber 迁移）。
- **⑥ 过度复杂**：只做胶水（§4.3），最小化——start/done/result + 三 awaiter，组合子（when_all/select）留 sched-10；热路径仍可用裸 fiber+Waker（§4.3）。

## 交付物

- `include/hyle/sched/coro.h`（Task<T> + Task<void> + PromiseAllocBase + SuspendFiber + CaptureWaker + 子 Task awaiter）
- `tests/sched/`（3 Coro，全 sched 41/41）

## 参考

- sched-design §4（协程与 fiber 统一：§4.1 await_suspend 落 fiber 让出、§4.2 两层正交、§4.3 边界与热路径裸接口）。sched-04a（Waker 脊 / suspend / current_waker）。mem-01（Allocator，frame 背书）。
