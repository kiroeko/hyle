# Compute 待设计边界

> `compute` 当前是 `QUEUED` 身份预留，唯一状态见
> [`module-status.json`](../implementation/module-status.json)。本页只冻结问题边界，不把现有源码、
> 旧 task、旧 benchmark 或某个第三方 adapter 当成已经接受的产品设计。

## 为什么需要 Compute

HYLE 的推理引擎北极星需要把 host 数据、设备资源、异步提交与完成通知组合起来，同时允许工程计算
消费者直接运行 kernel。这个问题域通常包括：

- device/queue/stream/event 的发现、拥有和生命周期；
- host↔device 与 device-local buffer、映射、同步和显式复制；
- kernel/module 编译或加载、参数绑定、launch 与错误；
- CPU、GPU 和其他 accelerator 的 provider 选择；
- completion 与 HYLE scheduler/cancellation 的组合；
- profiling timestamp、能力查询和资源预算。

这些是未来 capability audit 的输入，不是当前功能承诺。只有真实推理 workload、标准和成熟 peer
共同证明的语义才进入 `MUST`。

## 已确定的跨模块边界

- host storage、alignment、pin/NUMA 与 allocator 归 Mem；Compute 只组合设备可见性或传输。
- OS、CPU、driver 与 loader 事实归 Platform；Compute 不复制探测器。
- CPU 向量原语归 SIMD；高层 tensor/kernel 不塞进 SIMD。
- fiber/task、取消和唤醒归 Sched；provider 不引入第二套公共 scheduler。
- 失败使用 Error 的值语义；第三方异常、错误码和 callback 在 adapter 边界转换。
- 模型格式、tokenizer、KV policy、batching、sampling 和 HTTP serving 属于推理引擎。

第三方 provider 的公开类型不得泄漏到 HYLE 普通 API。若某 provider 必须自行拥有 device allocation
或 graph allocator，HYLE 应诚实表达这个 owner，而不是用伪零拷贝或 borrowed handle 掩盖生命周期。
ggml adapter 曾暴露过这一类问题；其历史调查见
[显存归属边界](borrowing-ggml-memory-model.md)和
[调试记录](../developer/debugging/ggml-zerocopy-null-buffer-hunt.md)，但该 adapter 不自动成为新设计。

## 开工前必须重新裁决

1. 从版本化推理 consumer slice 建立最小端到端任务，而不是从现有类名恢复旧设计。
2. 对 CUDA、Metal、Vulkan、CPU 与成熟 runtime 做能力族和 ownership 对比；选择借用层级。
3. 决定 portable core 与 backend-specific extension 的界线，恒定失败入口不得冒充 portable 能力。
4. 冻结 buffer/event/stream/kernel 的 lifetime、thread/fiber affinity、failure、cancel 和成本合同。
5. 重新审计当前 `include/hyle/compute`、`src/compute` 和 CMake 边；保留、替换或删除都不需要兼容旧面。
6. 建立真实 kernel/transfer/completion workload 后才设计性能证据；旧 microbench 只作诊断输入。

在这些条件完成前，源码可用于实验，但不得从本页推断其 API、依赖、交付或性能已经验收。
