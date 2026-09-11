# IO Core v5：匿名本地字节流接管

<!-- MODULE CLOSURE RECORD: SCOPED -->

## Closure card

- component / owner / round：`io_core` / IO core maintainers / `io-core-v5-anonymous-stream-adopt`；
  kind=`REOPEN`；canonical checkout=`C:\public\project-hyle\hyle`；baseline=
  `f0e6dc130d5bad292789b466cc9a5c10646eed44`；前轮 terminal record 是
  [`io-core-v4-file-region-streaming`](io-core-v4-file-region-streaming.md)。
- trigger：G03 [`process-v1-managed-child`](../process/process-v1-managed-child.md) 的 AUDITED 源码/owner 审计
  证明 parent stdio pipe 需要进入既有 `IoContext`/proactor。当前 `LocalStream::adopt` 强制要求可 connect 的
  pathname/abstract/named-pipe endpoint，调用方只能伪造地址或复制一套异步 pipe。
- 正常任务：把调用方已经唯一拥有、已按当前 OS 的异步要求创建的 anonymous local byte-stream handle 接入
  一个 IoContext worker，随后沿现有 LocalStream read/write、deadline/cancel、close 合同使用。
- scope：只增加无名字、仅 adopt 的 endpoint/factory；既有 named endpoint/connect/listen 行为与对象大小保持
  不变；无额外 allocation、copy、runtime、queue 或 public native operation。
- non-goals：创建 OS pipe、stdio 方向类型、process 生命周期、handle duplication/inheritance、跨进程协议、
  公开 Proactor、改变 LocalStream 并发合同、Native FFI、G04 或 Compute/Graph。
- 依赖/消费者：依赖边不变；唯一新直接消费者是 AUDITED `process`。io_core 不能反向依赖 process。
- 性能：`P0`。adopt 与既有入口相同，一次 validation + bind；普通 connect/listen/read/write 没有新增常态
  字段或分支。没有同合同 P1 问题。
- 预声明验证：invalid handle/context/worker、anonymous adopt/read/write/EOF/close、anonymous connect/listen
  拒绝、既有 endpoint 回归；Windows/Linux/macOS、逐头、process integration、source/static/shared、sanitizer
  由 G03 候选矩阵共同承担。
- 调查/纠正预算：一次 LocalStream/backend 源码审计、一个产品批次；不运行全 IO P1，不借此扩张 pipe API。
- stop：若现有 Proactor 不能正确 bind POSIX socketpair 或 Windows overlapped named-pipe parent handle，则
  `BLOCKED` 并回到 G03 backend；不得降为 blocking file wrapper。

本卡只登记窄重开。registry 进入 `SCOPED` 后进行一次 API/backend 审计，在 `AUDITED` 前不得改产品代码。
