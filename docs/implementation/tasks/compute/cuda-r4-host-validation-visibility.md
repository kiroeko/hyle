# R4 CUDA：host 验证的同步可见性

## Closure card / AUDITED：有限重新定界

- owner：`compute_backend_cuda`，延续 R4 C6 的 AUDITED，仅重新冻结验证可见性的有限 FIX；本卡从属
  [R4 主卡](cuda-v1-four-route-foundation.md)，不能独立代替完整 CUDA 路线结案。
- baseline：`6541e801`；保留当前尚未完成的原生成本 benchmark 修改。
- trigger：WSL GCC14 TSan 的实际六因子检查 11/13 PASS，backend 与 Graph 测试产生十条告警。
  其中 Queue teardown/slot recycle 遵循 `util::RefCounted` 的 release RMW + final acquire fence；
  GCC TSan 不认识 standalone fence。独立 32 次双线程析构程序复现误报；同程序的 final acquire
  load 模型及 acq_rel 对照均无告警，析构读取的 payload 均正确。不能据此重开 Util 算法。
- 另一个机制：测试的 CUDA host callback 用 native event 保证真实完成，TSan 看不到驱动内的
  happens-before；复用 callback 所在栈区时报告后续 Graph/测试代码竞争。只允许在测试门上显式
  发布/观察 callback 的最终退出，不抑制 Graph/CUDA 的全部 race。
- 范围：CUDA backend 的 sanitizer-only 最终销毁同步投影、现有 callback 测试夹具，以及本卡
  原始日志/复现实验。普通 shipping 行为、对象大小、公开 API、引用计数策略和热路径均不改。
- 依赖：既有 `RefCounted::is_unique()` 已含 acquire load，可用于最终释放后的 TSan 模型观察；
  不添加 Utility API，不修改公共 sanitizer/selector/统计工具。
- 有限候选 FIX：V1，TSan-only QueueState 最终销毁前观察现有 refcount acquire load；V2，测试
  callback 在最后一次访问后 release 发布 finished，owner 在真实 native 完成后 acquire 观察。
- 审计预算：15 分钟，仅上述十条报告与两个源码机制。若发现独立实际数据竞争另行定界，不用抑制
  掩盖；若 V1/V2 不能消除已归因报告，保存具体残余并停止本机制纠正，不无限换 runtime 参数。
- 验证：复用本候选 ordinary/ASan/UBSan 的已完成项，产品修改后定向 backend 与真实 Direct/Graph
  consumer；WSL TSan 重跑有具体变化的两项，未变化的十一项不重跑。Windows 定向测试；P0 证明
  sanitizer=off 的预处理路径/ABI/热代码未改。本卡不启动 P1；完整 R4 在稳定后统一测量。

## 审计冻结

V1/V2 均为 FIX。release-sequence 和终态资源寿命的产品语义保持；`is_unique()` 的返回值在已归零的
最终销毁处不用作产品判断，仅令 TSan 观察同一计数器上的 acquire load。代码只在已存在的
`HYLE_DETAIL_SANITIZER_THREAD` 条件下生成，shipping 无额外 load 或布局变化。测试 final-store
之后不再访问 callback 对象，owner 在已完成原 native 等待后 acquire 观察，不能用测试同步代替
产品必须提供的 GPU 完成边界。记录十条初始告警，不使用全库/整个 CUDA 的 race suppression。

本项没有新增能力、API 或常态成本等级，属于原 AUDITED 的 host 验证范围内具名实现细节。
registry 不制造不合法的 AUDITED→SCOPED 转移，延续 AUDITED 并绑定本次有限重审结果。
原 TSan 报告不得写 PASS，原始数据保存在本轮 qualification 工作区，终审时归档。完成 V1/V2 后
owner 回到 R4 主卡继续全部验收，不能把验证投影算作新路线能力或独立 CUDA CLOSED。
