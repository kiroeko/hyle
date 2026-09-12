# Compute Provider oneDNN v3：开放 runtime 合同回放

## Closure card

- component / owner：`compute_provider_onednn` / Compute；round=`compute-provider-onednn-v3-runtime-replay`；
- baseline commit：`71f9de0cce8e6dd93a1bf3b2c6eb8f1e1f1b2cdb`；上一轮能力、性能与交付证据保存在
  [`compute-provider-onednn-v2-ukernel-resume`](compute-provider-onednn-v2-ukernel-resume.md)，不回写历史记录；
- trigger 与证据：已 `AUDITED` 的 Compute v4 正在以开放 `BackendId`/`ProviderId`、显式 provider sequence、
  独立 Queue state 和返回 `Completion` 的 provider submit 合同替换旧封闭枚举与同步裸指针 vtable。
  oneDNN 是直接消费者，必须迁移并回放交付；这是公开依赖合同变化触发的合法重开；
- 正常 consumer 任务：CPU 用户显式组合 oneDNN 与 HYLE native provider，冷路径准备 dense-f32 MatMul，
  重复在 CPU Queue 上提交；oneDNN 不成为隐藏 fallback，也不泄漏其类型到通用 runtime；
- 主产品纵切 / 具名来源：短 prefill dense-f32 MatMul；正交形状为小 decode 自动落到随后显式的 native
  provider；
- scope：迁移 provider identity/context/submit vtable、CPU storage mapping、显式 fallback sequence、测试、
  文档与 source/static/shared 投影；确认 v2 BRGEMM 算法和选择阈值没有无关变化；
- non-goals：增加新 oneDNN primitive、改变 BRGEMM tiling、重新跑无产品漂移的完整 P1、CUDA/Metal/Vulkan、
  Native FFI；HIP/ROCm 无限期不在当前产品线；
- 直接依赖 / 直接消费者：公开依赖 Compute，CPU backend 是私有运行时事实；直接消费者为 Compute/Graph
  CPU 产品与 package sample；
- 性能 tier 与原因：`P0` 合同回放；若迁移改变已计时 execute 路径或常态 allocation/copy/dispatch，才对
  不可变候选重跑原 v2 P1；
- 调查与纠正预算：一次 vtable/API 迁移、每个 correctness defect 最多两轮；不得借机扩展算子目录；
- 预声明验证：显式 provider 顺序、unsupported fallback、identity/version、workspace、mapping/coherence、
  queue/context mismatch、close/drain、逐头、Windows/WSL/两台 Apple 负能力、sanitizer、source/static/shared、
  export/依赖、P0 diff 与红队；
- stop / blocker 条件：若 Compute v4 候选仍变化，保持本轮 `SCOPED/AUDITED` 而不伪造 terminal；若 locked
  oneDNN 无法表达新 Completion/Queue 合同，记录具体 blocker，不保留旧同步 ABI 作为永久兼容层。

## 待审计项

1. oneDNN provider 的 open identity 与版本应如何进入 artifact/report；
2. CPU backend compatibility 和 host mapping 由谁验证，怎样避免通用 runtime 私藏 CPU 路线；
3. immediate CPU Completion 是否保持 submit 零新增 heap/lock/syscall；
4. 旧 `ProviderKind`、隐藏 native fallback 与固定 `ProviderList<4>` 的全部测试/文档消费者如何删除；
5. Compute v4 改动对 v2 候选性能证据的实际失效范围。

`AUDITED` 前不修改 `compute_provider_onednn` 产品代码。
