# Compute v14：编译私有存储上界

## Closure card / AUDITED

- owner：`compute`；baseline `1108d591`；承接 [v13](compute-v13-invocation-storage-facts.md)、未完成 C93
  与共同 foundation，不改变 Queue/Completion 或 prepared replay。
- trigger：Graph G106 的 compiler quota 已能取得 allocator 请求，但现有 persistent_bytes 混合了已追踪
  metadata 与不经过该 allocator 的 Provider 存储，而且 prepare 后的数字不能约束 prepare 峰值。
- C140 FIX：新增 `ProviderCompilationStorage`，以 optional resident/peak 字节上界表达 supplied allocator
  之外的编译存储；peak 包含 resident，覆盖 prepare、失败回滚与 destroy。unknown 不猜零；两个已知值
  必须 peak >= resident。放在已有 proposal cost 中，编译前取得，不增加热路径或公共管理器。
- C141 KEEP：typed KernelProposal 原样传播该 cost；Native 编译只使用 supplied allocator，因此为 0/0。
  其它 Provider 默认 unknown，不从旧 persistent_bytes 推断；同 build 私有版本递增。
- 验证：valid/unknown/zero/逆序上界、typed propagation、原 Compute/CPU/Native/oneDNN/Graph、逐头、
  Windows 三交付与 CUDA；完整候选再做跨平台/sanitizer/终局红队/P1。
- 性能 P0：仅冷 proposal 字段/检查；没有新 kernel、submit 分支、线程或分配层。
- 预算：一轮表示/实现与一次纠正；若要猜 vendor malloc 或追踪进程 RSS，保留 unknown，不扩本卡。

Graph 管总量和真实生命周期；本卡只提供非重复、编译前可用的私有上界，不独立声称全内存验收。
