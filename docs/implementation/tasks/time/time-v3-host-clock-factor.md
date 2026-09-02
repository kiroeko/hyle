# Time v3：host-clock 质因数与可注入时间

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`time` / Time / `time-v3-host-clock-factor` / `REOPEN`；
- baseline：`25b13f5a18a0164a5fa81d01c4ef5d1a32f85cf4`；前轮：`time-v2`；
- trigger：`TIME-FACTORIZATION-2026-09-02`；
- `TIME3-R1`：已接受的 Stage C 证明 host clock/value 与 caller-owned timer node 是两个独立消费者、依赖
  和成本闭环。现有 `time -> container_intrusive` 迫使只需要 `UnixTimestamp` 的 Platform 取得 timer
  storage 依赖，满足“component dependency/delivery shape changed”的重开条件；
- `TIME3-R2`：能力账本 `TIME-A-17` 已由 Seastar manual clock、Abseil simulated clock 与确定性
  scheduler/timeout 测试共同准入，但当前 HYLE deadline helpers 固定读取全局 clock，只能给 queue 显式
  传 `now`，属于已确认 `MISSING`；
- `TIME3-R3`：time-v2 candidate 后，Time 实现已迁移到新的 Core environment contract，当前源码 identity
  不再等于旧 candidate；本轮以当前 baseline 重新绑定受影响的依赖、交付和 codegen，不伪装旧证据覆盖；
- 最新参考身份（2026-09-02 clean fast-forward）：llama.cpp
  `b96806d96061049a5b574269b049bf6241d63d46`（随仓 ggml tree
  `bc886da1a6c9d69660b0043d407200015667d609`）、standalone ggml
  `d4716378882593333721eb33f153144b6885caf2`、Seastar
  `991cf9a908482a3f11ca2fc34a1b3970a06d8622`、Abseil
  `cc05267c15127e41d497ddfc985328e59992ea53`；
- 初步受影响面：host clock/value/conversion/sleep 与 injectable/manual clock 的公开自然形状；从 Time
  迁出 timer headers/source ownership；Platform/Sched/IO 的直接边；`hyle::time` façade 的双 factor
  聚合；correctness、P1、三平台三交付、architecture/user 文档；
- 初步 tier：`P1`。monotonic/profiling/coarse reads 与 timer lifecycle 都可能处于 runtime 热路；本轮
  只在 API freeze 后分别冻结 host-clock 与 timer-queue 正常 workload，不用 clock-call 亚纳秒点值制造
  接口；
- 暂定非目标：device clock、timezone/tzdb/locale、协议文本、callback/executor、周期/跨线程 timer、
  cancellation propagation、Sched/IO 重写、Native FFI/C API、稳定 C++ ABI、公共仪器改造与 Formal。

当前只登记足以击穿旧结论的新事实；closure card、完整 public trace、有限 `FIX` 集和 API freeze 尚未
完成，因此不授权产品代码修改。
