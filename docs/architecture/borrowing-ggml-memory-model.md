# ggml provider 的 storage 与执行边界

> 本页记录一条长期 adapter 规则，不选择 Compute 的唯一实现。`compute` 当前是否可用只查
> [`module-status.json`](../implementation/module-status.json)。旧 compute-05 调试证明“伪造 ggml device
> buffer metadata”不可接受，但当时的 ggml/CUDA 行为不是所有版本、backend 和 provider 的永久事实。

## 稳定结论

借第三方算子时，HYLE 不能假定自己拥有对方内部的 device storage、allocator、stream、graph capture
或线程池。对每个锁定 provider/backend 组合必须逐项确认：

1. 能否从 caller-owned memory 创建合法 tensor/buffer；
2. import 是借用、接管还是复制，owner 必须活多久；
3. host-visible、pinned、device-local 与 unified storage 分别支持什么；
4. queue/stream 和 completion 能否导入、导出或等待；
5. layout/repack、upload/download 和同步发生在哪里；
6. graph allocator、packed constants、workspace 与 provider context 由谁销毁；
7. CPU、CUDA、Metal、Vulkan、HIP 等 backend 是否有不同限制。

只有 provider 的公开 API 明确支持且有正常/失败/销毁测试时，才能把 external memory 写成零拷贝。
设置内部字段、伪造 dummy buffer、依赖未公开 layout 或把 native address 塞进不透明 handle 都属于拒绝路线。

## 两种合法形状

### Provider-owned island

若 provider 必须拥有 allocation 或 graph allocator，Graph compiler 把相应 region 编译成一个 provider
island：

```text
boundary input
  -> validated import, or explicit copy/repack when necessary
  -> provider-owned tensors + region + workspace
  -> owning result, or validated export / explicit copy to caller destination
```

不可变制品与每次可变执行状态分开；Compute 的真实完成边界决定 ingress/egress、workspace 和外部
Buffer 何时可复用。结果可继续由 Provider 拥有，不强制下载或复制到 HYLE 影子 buffer。报告列出实际
边界 copy、同步和 ownership，完整成本计入这些代价，不能只比较 island 内的 kernel。

### Supported external-memory interop

若 provider 明确接受 caller storage，Compute 用 backend-specific typed adapter 把 Mem Buffer 与 provider
对象绑定，并保存 resource/device/offset/size/coherence identity。owner 和 view 活到所有 completion 结束。
CPU `buffer_from_ptr`、Vulkan external memory/semaphore、CUDA IPC/external memory、Metal no-copy/shared
storage 或 MLX interop 都要按各自公开合同审计，不能从其中一条路线类推另一条。

## 历史证据怎样使用

旧 ggml CUDA bridge 曾尝试让 HYLE allocation 冒充 ggml tensor buffer。单节点或一致 shape 可以偶然
通过，多节点 varied-shape 路径在 gallocr/metadata 使用中崩溃；关掉 CUDA graph 也不能消除问题。这个
调查的永久价值是：

- synthetic metadata 通过 smoke test 不证明 ownership 合法；
- 单 op 成功不证明 multi-op graph、varied shape 和销毁正确；
- CPU external pointer 能工作不证明 CUDA/HIP/Metal/Vulkan 能工作；
- adapter 必须以 provider 公布的 import/create/allocate API 为边界。

完整复现和当时版本事实保留在
[ggml 零拷贝调查日志](../developer/debugging/ggml-zerocopy-null-buffer-hunt.md)。新 Compute 任务只把它当
回归用例来源，仍须绑定新的 ggml commit、backend、driver 和公开 API 重新验证。
被删除的旧 adapter、测试与施工计划只从 Git 历史取得；本页与上面的调试日志已经保留新 provider
仍需继承的全部长期约束，不授权恢复旧实现或旧 API。

## 对新架构的约束

- 当前 `compute_provider_ggml` 是 CPU-only 的阻塞审计身份，不是 Compute 本身；CUDA、Metal 与 Vulkan
  carrier 必须各有独立身份和验证，用户普通 API 不出现 ggml 类型。
- 同一 Graph 可以把部分 region 给 ggml、部分给 HYLE native 或其他 provider；选择发生在编译阶段。
- provider-owned storage 不削弱 Mem 的 owner/view 价值：Graph boundary、host mapping、external buffer、
  composite shard 与 HYLE native region 仍使用 Mem；provider island 如实保留自己的 owner。
- 若跨 island copy 使 ggml 路线整体更慢，应选择另一 provider 或扩大合法 region，而不是伪造零拷贝。
- provider 的自有线程/stream/graph capture 必须在 capability 与 cost 中声明，并由 Sched/Graph 避免
  nested oversubscription；HYLE 不假装能注入一个 provider 并未接受的线程池。
- ggml 升级若改变 buffer import、allocator、threading、graph capture 或 completion 行为，会触发该
  provider factor 的定向重审，不自动重开无关 Compute/Graph 因子。

## ggml graph 怎样成为 HYLE 可调用制品

HYLE 不要求 ggml 把私有 CUDA/CPU kernel 改成公共叶函数。合法 adapter 在冷路径把 HYLE 的一个 op 或
可收缩 region 翻译为一个可复用 `ggml_cgraph`：建立并保有 tensor metadata、operation nodes、backend
对象、常量/packing 和需要的 invocation state；steady submit 只绑定本次 boundary storage，调用公开的
backend graph compute，并把真实异步终态桥接为 HYLE Completion。一个单节点 `ggml_cgraph` 可以充当
function-like artifact，多节点 region 则保留 ggml fusion 与内部 layout。HYLE 不逐次重建逻辑图；
backend 内部是否仍 encoding/分配按 prepared/managed 单独判断，不能从 `graph_compute` 名称推导
零成本 replay。例如 ggml-Vulkan 的实际提交仍可能逐次编码和分批提交，见[路线设计](compute-routes.md)。

对当前锁定参考版本，公开 `ggml_backend_graph_compute_async()`、backend event 和 provider-owned buffer
足以证明 delegated CUDA island 值得独立审计，但还没有证明 shared-runtime 零成本接入：公开 API 没有
签发把任意 HYLE CUDA allocation 包装为 ggml CUDA buffer、也没有把任意 HYLE stream 设置为 ggml 执行
stream 的通用合同。`ggml_cuda_compute_forward`、`ggml_cuda_mul_mat` 等实现细节不是稳定公开 seam。
因此 CUDA carrier 开工时必须在以下路线中明确裁决，而不能先写 adapter 再靠 smoke test 猜：

1. ggml-owned allocation/stream/event 的 delegated region，并把 ingress/egress、同步和内部常驻显存计入
   完整成本；
2. 锁定版本后来提供并经验证的 external-memory/stream interop，升级为 shared-runtime 或零拷贝 island；
3. 对确实值得逐 leaf 替换且有 material 证据的 kernel，建立独立 source-patch factor；否则继续使用
   ggml region 或 HYLE/vendor provider。

渐进式 best-of 发生在可实现的完整候选间：原版 region、独立 HYLE leaf/融合，或 Provider 公开扩展允许的
内嵌 HYLE kernel。ggml 不因此自动具备 MLX 的 custom Primitive 机制；切分产生的 copy、repack、event
或 fusion 损失超过收益时保留更大 region。不能只比较私有 kernel 时间后忽略边界成本。

总的所有权规则见 [Compute 设计](compute-design.md#5-storage同步与生命周期)；region、Executable 与
运行状态的关系见 [Graph 设计](graph-design.md#3-生命周期职责)。
