# F2 实施前源码审计

2026-10-11，SOURCE_ONLY。此页是[有限 F2 卡](foundation-f2-batched-transfers.md)的定界输入，
不改变 registry、不授权产品修改、不签运行资格。F1 当前仍为 CANDIDATE；F2 需在其 terminal 后
绑定实际 HEAD、依法完成 SCOPED/AUDITED，再实施原七项 FIX。
当前读取的 HYLE 产品与 `3034fc595abf5c2570e06dedbf195b34daf3fe72` 一致。

## 已有原子与复合缺口

| 原 owner / 实际源码 | 已有可复用合同 | F2 仍需固定的有限接缝 |
|---|---|---|
| compute：[transfer](../../../../src/compute/transfer.cc)、[QueueSequence](../../../../include/hyle/compute/detail/queue_sequence.h) | checked 单 byte range；序列排除提交交错，accepted prefix 携原错，最终一次 Completion；device-side wait 可用性由实际 endpoint 查询 | 冷固定 endpoint/capacity、整批 physical alias/范围校验、有限 metadata 与 bind/replay；一次 phase 不逐 range 创建 Completion |
| compute_backend_cpu：[CPU backend](../../../../src/compute_backend_cpu/cpu_backend.cc) | caller-thread 同步 `destination.copy_from(source)`，原 resource/Queue/Status；完成是 inline | 有限多范围沿原 byte copy；明确空批/self-copy、独立区间与有依赖 phase；不假装 CPU 异步 |
| compute_backend_cuda：[CUDA backend](../../../../src/compute_backend_cuda/cuda_backend.cc) | 原 same-context CUDA device/pinned/managed storage，非 legacy stream、原 `cudaMemcpyAsync` 和 EventSlot/QueueSequence retirement | batch 能力/ABI/stream 冷资格与同路线原命令 fallback；producer ordering 与 control arrays 保活；fallback 的 O(R) native calls 明示 |
| compute_route_apple：[Apple memory](../../../../src/compute_route_apple/apple_memory.mm)、[原 Runtime](../../../../src/compute_route_apple/apple_context.cc) | 原 shared Metal carrier 与 ready array owner；现有 host copy 是同步边界；原 Context 已有自己的 native owner/Job admission | 原生 blit/command 的批量与 CPU/GPU 可见性、完成/取消/部分失败必须接原 owner；现有 host memcpy 不能代签 native batch |
| compute_route_vulkan：[TransferSlot](../../../../src/compute_route_vulkan/vulkan_transfer_slot.cc)、[Run](../../../../src/compute_route_vulkan/vulkan_run.cc) | 冷 command/fence 与 staging；实际 eligible upload 的 transfer queue/semaphore→compute prefix；legacy/imported/download 原 compute queue；原 flush/invalidate/unknown drain | 多 backing/pair 分组 VkBufferCopy；全范围四字节/tail/usage/physical alias；每阶段实际前缀与 group ready；不复制 Run/Device/native retirement |
| F1：[lease 核](../../../../include/hyle/compute/resource_leases.h)、[完成代理](../../../../src/compute/retained_completion.cc) | bounded uses/read/write、未接受 rollback、实际退休、首错与 dropped Completion 仍保活 | 搬运 source/destination 与 descriptor/index/producer owner 一起保留；失败不能发布 ready 或复用未知 prefix |

四路线原资源和执行 owner 已明确；共同层可以承载 byte ranges/checked arithmetic/有限 metadata，
不能因此推导出共同万能 Queue。新的 prepared copy/phase 是尚缺的复合功能，不能把原单 copy API
直接记作完整 F2。普通用户构建仍只选 Compute/Graph 与 routes，不新增内部 factor/provider 开关。

## 固定成熟来源与本机 SDK 核对

本次未更新研究 checkout，实际 HEAD 与[原研究](../../../architecture/compute-resource-streaming-design.md)相同：
vLLM `276fbcff2717bd934cfa37c8a2e4c391f3e7237b`；
SGLang `234ccba7f625e12639e621044496bcc6197c3758`。

- vLLM `vllm/v1/kv_offload/cpu/gpu_worker.py:143` 的 CopyPlan 按实际 fragment offset/stride/size
  建立 host-known ranges，CPU/GPU 粒度与方向独立；不能只支持整 tensor 等尺寸。
- 同版本 `csrc/libtorch_stable/cache_kernels.cu:110` 的 batch 具有 host pointer/size arrays、冷缓存
  proc-address 和 legacy stream fallback。ANY 只在明确没有并发 producer 的 source 使用；普通默认 STREAM。
- SGLang `python/sglang/srt/mem_cache/l2_transfer.py:103` 在 producer event 之后准备索引，
  record_stream 保留实际 control/source，失败排空；layout gate 与逐 layer ready 是独立合同，
  F2/F3 只承载通用 group/tag，不带 layer/model policy。
- 本机 Windows CUDA13.3.73 的 `cuda_runtime_api.h:6464` / `driver_types.h:2429` 明确：
  batch 整体有 stream order，批内无顺序；dependent copies 同批不合法；pointer arrays 同长度、
  attribute index 分段有界且首项0；source STREAM/DURING_CALL/ANY 的寿命与排序各不相同。
  该 SDK 的 runtime signature 与上述 peer 的12.8 driver typedef 不能直接互换；实施 audit 需选定实际 ABI，
  不从头文件存在推断 driver 能用，也不热解析能力。这里没有运行 native batch 或签性能。

## 实施前必须冻结的决定

1. 现有 component 中实际需要改动的 owner/直接消费者，普通产品闭包与 public/native 类型依赖方向。
2. endpoint 的 physical backing identity/offset/capacity、ranges/catalogue IDs 与 mutable destination 权限；
   别名校验不能只比较包装对象地址，原数据编码与 native padding 的拥有范围分开。
3. cold prepare、bounded bind/replay、independent batch/explicit phase、单组/分组 ready 的自然 C++ 形状；
   空批/exact self-copy、满容量、旧结果与 pending/failed slot 的错误形状。
4. native capabilities/fallback、producer dependency、one-phase completion，以及 descriptor/index/source owner
   的实际保活；拒绝、partial acceptance、unknown observation、取消/close/drain 逐项裁决。
5. P0 的 bind 校验 O(R log R) 或更好与 submit O(R+G)；冷 arrays/groups 有界，vendor/native calls、
   host waits/staging/bytes 明示。没有性能反例不进入 P1，缓存控制不改变旧普通数学 Provider。
6. 原 F2-01至07、至少二组二槽、真实量化/BF16搬运后 Direct/Graph 数学、三平台及三交付的有限验收。

本页不先发明未审计的公共 API。只读核对完成后，按原卡合法定界；F3、GPU-produced dynamic IDs、
复制/数学 kernel、跨 GPU 与应用缓存策略不混入本批。
