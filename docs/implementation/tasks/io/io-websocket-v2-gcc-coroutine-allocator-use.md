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

当前状态为 `SCOPED`。[旧 v1 terminal record](io-websocket-v1-rfc6455.md) 保持历史事实，本卡只处理
上述可复现源码交付缺陷。
