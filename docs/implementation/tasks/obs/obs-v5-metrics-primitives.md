# Obs v5：推理运行时指标聚合原语

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由已冻结模块的真实推理消费者回放触发。当前仅为 `REOPENED`：本页保存可复现的新产品事实与
> 最小影响面；在后续 `SCOPED/AUDITED` 分别冻结 closure card、能力账本和有限 `FIX` 之前，不修改
> Obs 产品源码。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`obs` / model-independent metrics aggregation /
  `obs-v5-metrics-primitives` / `REOPEN`；baseline
  `c51599be2fdcd4caeacc5b05aa0f61f725c62491`，tree
  `40e9c1d341548357bfdccc667d0c3af7d6ae0549`；
- trigger ID：`OBS-METRICS-PRIMITIVES-2026-09-06`。锁定 llama.cpp
  `73a43d1f69345aee8bb186ef4b3172cef892f2e5` 的
  `tools/server/server-common.h:server_metrics` 与
  `tools/server/server-task.cpp:server_task_result_metrics::to_metrics` 分别维护累计 counter、即时 gauge
  和窗口 bucket，并在 `/metrics` 才转换成 Prometheus 文本；
- 锁定 vLLM `808f8cd3ac6751278d8b7be97f6fde5295ca4653` 的
  `vllm/v1/metrics/loggers.py:PrometheusStatLogger.record` 按 scheduler iteration、token、request 和 KV block
  eviction 更新 counter/gauge/histogram；锁定 SGLang
  `5bebe7a033ab57751fac3c9d8db219da62ca8c72` 的
  `python/sglang/srt/observability/metrics_collector.py:SchedulerMetricsCollector` 以同三类原语记录 queue、
  token、cache、latency、transfer 和 storage 事实。三条独立生产旅程证明基础 producer/aggregation
  vocabulary 是通用 Obs 能力，不是模型、backend、HTTP 或 benchmark 形状；
- HYLE Obs v4 只有同步 structured log 与编译期 trace。把每个数值编码成 `LogRecord` 会支付 fan-out、
  format/sink 生命周期，并丢失 counter 单调累积、gauge last-value 和 histogram bucket 语义；Tracy plot
  又只服务 profiling build，不能组成 production scrape snapshot。因此现有路线不能自然承载上述任务；
- 成熟实现反证边界：Folly
  `folly/ThreadCachedInt.h` 与 `folly/stats/Histogram.h` 分离 update/snapshot、线程语义和固定 bucket；
  推理项目也都把 metric producer 与 Prometheus/exporter、label/cardinality policy 分开。HYLE 只考虑
  无名称注册表、无 exporter、无后台线程、无隐式 clock 的数值原语；
- 受影响范围只含 `include/hyle/obs` 下新的 metrics public surface、Obs tests/docs/sample、selected
  component 的 source/static/shared header delivery，以及新增路径的 P0/P1 成本裁决。现有 log registry、
  record layout、Tracy route、`core/error/time` 公共依赖和 private `mem_reclaim` 路线默认不改；
- 明确不做 Prometheus/OpenTelemetry wire format、metric name/description/labels、动态 registry、时间窗口、
  exporter、网络/持久化、后台 flush、自动 task context、GPU event timing、模型/KV/scheduler policy、
  Native FFI/C API、稳定 C++ ABI、公共性能仪器或相邻模块重构；
- `CLOSED -> REOPENED` 前 registry snapshot：
  `C:\\public\\project-hyle\\_validation\\obs-v5-metrics-primitives\\transitions\\before-reopened.json`，
  SHA-256 `8473C0DC32E5459DBE421603D04DB45A34EEB34E7F66BEBD866EC356056CE167`。

## 初步停止边界

- API 必须按 metric 语义命名并让 ownership、并发和 snapshot 一致性可见；不能用 `fast/light` 或
  benchmark 名称制造两套含糊入口；
- 普通更新不得分配、加锁、读取时钟、查名字/标签或调用 log sink；bucket 查找的复杂度必须显式；
- exporter 与 label/cardinality 生命周期继续由宿主拥有。若审计证明基本数值原语也必须依赖动态
  registry、Container、Mem 或异步 runtime，本轮停止并重新裁决 owner，不顺手扩大依赖图；
- 原 Obs v4 日志 P1 合同只做未受影响回放；新指标性能只比较同语义、同线程安全、同 snapshot 保证的
  路线，不能把 thread-confined plain integer 与 concurrent exact atomic 当成同合同 peer。
