# Compute v13：运行实例的额外存储事实

## Closure card / AUDITED

- component / owner：`compute`；baseline `9d4108e9`；承接 [v12](compute-v12-metadata-ownership.md) 和其
  未完成 C93/foundation。已有 metadata owner 不改变。
- trigger：Graph G106 将 runtime allocator 和 workspace 放入同一共享配额；Provider 的 persistent
  字节不能代表 invocation 私有 heap，且不能重复计算已走 supplied allocator 的对象。
- 主旅程：Native exact/general 的独立 Invocation 共享 Graph 预算；正交旅程：typed custom kernel
  可以明确声明额外 private storage，也可以保留 unknown。默认无预算执行与 direct API 保持。
- C130 FIX：PreparedOperation 和 KernelPreparedProperties/私有 ProviderPrepared 披露
  `invocation_private_bytes`：不经过所供 allocator、不在 caller workspace 中的每实例额外存储上界。
  覆盖 domain instantiate/create/replay/teardown，prepared 域对全部合法点负责；不是 persistent 字节。
- C131 KEEP：未声明为 unknown，不猜零、不强迫普通 direct caller 配置；Native N21 的已知全部分配
  走 allocator，因此声明 0。不改 kernel、submit、Context/Queue 或 CLOSED backend。
- 版本：私有同 build Provider contract 随布局变化递增；没有 C/FFI 或旧版本兼容层。
- 验证：facts 的 exact/domain/typed 传播、未知与零区分、原 Compute/Native/CPU/oneDNN/Graph、逐头、
  Windows source/static/shared 和 CUDA；共同候选再做 Linux/两台 Apple、sanitizer、独立终局红队/P1。
- 性能：P0；一个冷 prepared 字段/查询，不进入 submit；总预算、Graph capture 和实际计费归 G106。
- 预算/stop：一轮表示/实现、一轮纠正；若需要 vendor 私有 API 或热态追踪所有 malloc，停止该路线，
  明确 unknown，不借本卡扩 Provider 目录。

本卡在同一已 AUDITED owner 下固定有限后继，不签候选或 terminal；完整基础 gate 的剩余范围不缩减。
