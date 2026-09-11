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
HYLE/Mem input
  -> explicit upload/repack
  -> provider-owned tensors + compiled region + workspace
  -> explicit download/repack
  -> HYLE/Mem output
```

`graph::Executable` 持有该 island 的不可变 artifact；`Invocation` 持有每次执行状态；Compute Completion
决定 ingress/egress、workspace 和外部 Buffer 何时可复用。CompileReport 必须列出边界 copy、同步和
provider ownership，cost model 也必须把它们计入，不能只比较 island 内的 kernel。

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
当轮范围、版本和结果见历史事实记录 [compute-05 ggml bridge](../implementation/tasks/compute/compute-05-ggml-bridge.md)；
该记录不授权恢复旧实现或旧 API。

## 对新架构的约束

- ggml 是候选 `compute_provider_ggml`，不是 Compute 本身；用户普通 API 不出现 ggml 类型。
- 同一 Graph 可以把部分 region 给 ggml、部分给 HYLE native 或其他 provider；选择发生在编译阶段。
- provider-owned storage 不削弱 Mem 的 owner/view 价值：Graph boundary、host mapping、external buffer、
  composite shard 与 HYLE native region 仍使用 Mem；provider island 如实保留自己的 owner。
- 若跨 island copy 使 ggml 路线整体更慢，应选择另一 provider 或扩大合法 region，而不是伪造零拷贝。
- provider 的自有线程/stream/graph capture 必须在 capability 与 cost 中声明，并由 Sched/Graph 避免
  nested oversubscription；HYLE 不假装能注入一个 provider 并未接受的线程池。
- ggml 升级若改变 buffer import、allocator、threading、graph capture 或 completion 行为，会触发该
  provider factor 的定向重审，不自动重开无关 Compute/Graph 因子。

总的所有权规则见 [Compute 设计](compute-design.md#5-storage同步与生命周期)；region、Executable 与
Invocation 的关系见 [Graph 设计](graph-design.md#3-四个生命周期对象)。
