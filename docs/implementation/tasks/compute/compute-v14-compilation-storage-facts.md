# Compute v14：编译私有存储上界

## Closure card / AUDITED

- owner：`compute`；baseline `1108d591`；承接 [v13](compute-v13-invocation-storage-facts.md)、未完成 C93
  与共同 foundation，不改变 Queue/Completion 或 prepared replay。
- trigger：Graph G106 的 compiler quota 已能取得 allocator 请求，但现有 persistent_bytes 混合了已追踪
  metadata 与不经过该 allocator 的 Provider 存储，而且 prepare 后的数字不能约束 prepare 峰值。
- C140 FIX：新增 `ProviderCompilationStorage`，以 optional resident/peak 字节上界表达 supplied allocator
  之外的编译存储；resident 覆盖保留至最终 destroy（含 teardown scratch），peak 另含 prepare/失败回滚。
  peak 包含 resident，unknown 不猜零；两个已知值
  必须 peak >= resident。放在已有 proposal cost 中，编译前取得，不增加热路径或公共管理器。
- C141 KEEP：typed KernelProposal 原样传播该 cost；Native 编译只使用 supplied allocator，因此为 0/0。
  其它 Provider 默认 unknown，不从旧 persistent_bytes 推断；同 build 私有版本递增。
- 验证：valid/unknown/zero/逆序上界、typed propagation、原 Compute/CPU/Native/oneDNN/Graph、逐头、
  Windows 三交付与 CUDA；完整候选再做跨平台/sanitizer/终局红队/P1。
- 性能 P0：仅冷 proposal 字段/检查；没有新 kernel、submit 分支、线程或分配层。
- 预算：一轮表示/实现与一次纠正；若要猜 vendor malloc 或追踪进程 RSS，保留 unknown，不扩本卡。

Graph 管总量和真实生命周期；本卡只提供非重复、编译前可用的私有上界，不独立声称全内存验收。

## 2026-09-15 实现检查点

`bcd7ecbe` 已实现 C140/C141，私有 Provider table v12；Native 声明额外 0/0，typed factory 原样传递，
非法上界冷拒绝。公共 `KernelRequest` 补明既有成功 proposal 零分配合同，不改变 callback 形状。
Compute 值测试、Graph private-bound/失败/重选反例与 source/static/shared 的 CPU/CUDA consumer 已通过；
命令、环境、LOC 和未验证项统一见 [Graph 编译存储检查点](graph-v10-reviewed-foundation-contracts.md#2026-09-15-g106-编译-requested-storage-检查点)。
本卡仍 AUDITED，继承 C93/foundation 的未完成事项；未运行的跨平台/终局性能不签 PASS。
