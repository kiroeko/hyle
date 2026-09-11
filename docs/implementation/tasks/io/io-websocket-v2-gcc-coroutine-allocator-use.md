# WebSocket v2：GCC coroutine allocator 源码交付修复

## Closure card

- component / owner / round：`io_websocket` / IO maintainers / `io-websocket-v2-gcc-coroutine-allocator-use`；
  kind=`REOPEN`；canonical checkout=`C:\public\project-hyle\hyle`；baseline=
  `db277a579609a429c408706842e4845df28413e9`。
- trigger 与证据：全库正确性资格 Wave E 在受支持的 GCC 14.2、shared、WERROR 源码构建中稳定失败；
  `src/io_websocket/websocket.cc:914,943` 的两个 coroutine allocator 参数只被编译器用于 frame
  allocation，GCC 仍把函数体中的参数判为 unused，并以 `-Werror=unused-parameter` 阻断交付。
- 正常 consumer 任务：从源码或 shared 组件构建并使用 WebSocket，不要求消费者关闭 HYLE 的 warning gate。
- 主产品纵切：Wave E 的完整 IO/Process factor 构建；正交纵切：Windows MSVC Release shared/WERROR。
- 跨模块 public journey：HTTP Upgrade → WebSocket connection；本修复不改变该旅程的运行语义。
- scope：仅让两个私有 coroutine 明确消费 frame allocator 参数，并回放 WebSocket 公开测试。
- non-goals：公开 API/layout、协议状态机、allocator 路由、算法、性能、全局 warning policy、Compute/Graph。
- 直接依赖 / 消费者：依赖不变；`io_http`/`io_tls` 与 WebSocket 用户不迁移。
- 性能 tier：`P0`；预期语句在所有优化构建中零代码，不重跑旧 P1。
- 调查与纠正预算：一次源码/编译器诊断、一个产品修复、至多一次纠正。
- 预声明验证：GCC 14.2 shared/WERROR 重新构建 `hyle_io_websocket` 与测试；WebSocket/HTTP Upgrade
  CTest 硬 timeout；Windows MSVC Release shared/WERROR；closure/docs/diff gate。
- stop / blocker：若消除 warning 必须改变 coroutine allocator 签名或运行语义，退回 `AUDITED` 重新裁决；
  不以放宽 `unused-parameter` 结束本轮。

## 审计与有限改动集

`sched::Task` 的 promise 通过 `PromiseAllocBase::operator new(size_t, AllocatorRef, ...)` 从 coroutine
函数的第一个参数取得 frame allocator；因此不能删除、改型或绕过该参数。GCC 14 已在生成的 frame
allocation 中使用它，却不把这种隐式使用计入 `-Wunused-parameter`。两个失败函数的 body 都会把后续
异步子任务的 allocation failure 转为 typed `Status`，问题仅是源码交付 warning。

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| `W02-01` | `read_exact` / `write_frame` 的 frame allocator 被 GCC 判为 body-unused | 在两个私有 body 中加入显式、可优化掉的 `(void)allocator`；不改签名或调用 | GCC WERROR 编译通过；优化代码、API/layout/状态机不变 |

明确拒绝全局关闭 `unused-parameter`、修改 coverage 工具规避本组件、删除 allocator 参数或改用默认 heap。
公开 C++ API、ownership、failure、并发、teardown、默认值和 source/static/shared component 名称全部沿用 v1。
P0 账单没有新增 allocation、copy、atomic、lock、syscall 或常驻 metadata；不适用新的 P1。

七问差量：第三方路径、功能范围、下层复用、Native FFI、交付拓扑和 guide 均不变；唯一失败的是 GCC
源码/WERROR 交付。修复只表达现有 coroutine allocator 合同，不产生 benchmark 入口或消费者迁移。

当前状态为 `AUDITED`。[旧 v1 terminal record](io-websocket-v1-rfc6455.md) 保持历史事实，本卡只允许
`W02-01`。

## Immutable candidate

产品 candidate 为 `35af7e364a5b0ff5bcada42e975375f4ce141b99`（tree
`e1f68b2194cdaf09ec0da73b1a3b6d9c9db268e7`）。`W02-01` 已按审计实现；同一提交还修正 HTTP Upgrade
测试夹具里的同类显式参数使用，不改变 `io_http` 产品字节。

- GCC 14.2 coverage/shared/WERROR：`hyle_io_websocket` 与测试编译通过；WebSocket runtime/逐头 2/2，
  HTTP Upgrade 1/1；
- Windows MSVC 19.50 Release/shared/WERROR：上述三组 3/3；
- 公开 API、layout、协议算法、ownership 和常态成本均未变化；旧 P1 仍适用且不重跑；
- candidate LOC：全库 556,948，自实现不含测试 100,236，`io_websocket` 实现 2,079 行。

当前进入 `CANDIDATE`；后续只允许本卡的终局文档、治理与证据，不再修改产品字节。
