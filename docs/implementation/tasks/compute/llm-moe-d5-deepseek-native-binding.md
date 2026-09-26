# D5-F32：成熟 DeepSeek hash 路由接缝

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- baseline：`cf0e985f666d89539d75edb308d1ab3f38d15470`，Windows canonical，工作树干净。
- owner：当前已 AUDITED 的 `compute_ops`、`compute_provider_cuda`；保留原 D5/F25 未完成门。
  不重开 CLOSED route、backend、SIMD，也不改公共治理或性能仪器。
- trigger：[真实角色审计](llm-moe-d5-deepseek-remaining-audit.md)证明固定 DeepSeek 前三层
  从 I32 token→expert 表选专家，已有 score TopK 不能表示它；固定 SGLang 已有完整原版程序。
- scope：借用同一 `06992c92` 的 `hash_topk.cuh::moe_hash_topk_fused`，接通正常 CUDA
  Direct/Graph；其它路线用已有成熟 Gather/Softplus/Sqrt/Reduce/Divide 组合。此 carrier
  不被误称为四路线通用原子。四画像的完整 MoE 消费者纳入同一 DeepSeek 批次。
- non-goals：新数学/kernel、修改通用 Softplus、模型 loader/engine、其它模型/专家数、
  新 Provider/依赖版本、无边界检查的快速入口、为本小项单独重建八入口或启动 P1。
- dependencies：原 SGLang in-tree 依赖与 Apache-2.0；仅追加原文件与来源/内容 hash。
  复用 CUDA primary context、QueueSequence、device status 和 Completion，不另建错误通道。
- budget：一次共享接线批次；具体失败最多一轮定向纠正。与剩余 DeepSeek 片段接齐后冻结候选，
  再集中做 selected schema/provider/Direct/Graph 和原 score-router 回归。三交付/P1归 D5-A5/A6。
- stop：真实 hash 层完整链通过或记录精确成熟域拒绝；单独 kernel/片段通过不增加完整画像格。

## 已固定语义与有限 FIX

`model_ops::SqrtSoftplusHashRouting` 输入 F32 logits `[T,E]`、I64 token IDs `[T]`、
I32 table `[V,6]`，输出 F32 weights `[T,6]` 与 I32 IDs `[T,6]`。plain dense、T/V非空，
E256/384。输出保持表中顺序；权重为选中 logit 的稳定 sqrt-softplus 除以 `sum+1e-20`。
原 kernel 的 shared-expert 分支关闭。输出不含模型的1.5/2.5缩放，消费者用已有成熟 Multiply
应用一次缩放后连接原 grouped expert/shared MoE。重复 ID 按表保留，不私加去重或排序策略。

原版 kernel 在检查前索引 `table[token_id, lane]` 和 `logits[token, expert_id]`。
HYLE 接入不能继承不受检查的外部索引：仅增加词表上界和既有 device status 端口，先检查
token，再以 warp 一致的方式检查六个被选专家，任何非法索引均不得读对应 payload。
有效输入的 score、SUM、epsilon、输出和128线程/4 warp launch policy 保持原版；PDL关闭。
源码适配必须注明不是逐字节原文件，不增加另一个数学 kernel。非有限 logits 不承诺有意义权重，
但仍不能导致非法访问。失败通过真实 Completion 报告；失败 invocation 的输出不可消费，
不承诺不同 token 行之间的事务回滚。
失败行先写安全的零 ID/权重，避免已排队的下游拿未初始化 ID 寻址；这不是失败输出可消费的承诺。

| FIX | owner | 有限动作与验收 |
|---|---|---|
| F32.1 | compute_ops | 完整 region descriptor、schema/shape/type/overflow拒绝；真实V129280、E256/384、K6 |
| F32.2 | compute_provider_cuda | BORROW 原 hash 程序；namespace/host binding及上述边界防护；保留原数学和warp SUM |
| F32.3 | 同一 owner | 原 SGLang provider 冷绑定、普通入口可达、资源/alias/完成与依赖内容锁；不搜索/计时 |
| F32.4 | D5 shared consumers | 表选择不同于TopK、变化token/table/logits、tiny/large logits、all-underflow、非法token/专家、多个invocation和OOM；接实际MoE下游 |

P0：prepare固定kernel/launch，submit仅原生索引检查和kernel；无host读回/排序/热分配/新线程。
复用既有异步device-status初始化与completion读回；错误检查成本明确保留，不能漏报为无检查。
Provider没有独立可变invocation数据，普通输入/输出由原Completion保活，禁止alias。
空shape与unsupported域在冷路径拒绝；运行数据的坏索引在GPU拒绝；共享三交付不新增TVM/Python运行时。

## 同一模型批次的剩余接缝

带sink的attention、CSA/HCA压缩状态和官方MTP仍由真实审计列出；它们不由本卡提前签署。
需要产品端口的sink接缝在写产品代码前另补有限FIX；其它已能用成熟计算表达的模型程序只写consumer。
本卡不授权用GGUF/BF16代签官方FP8动态激活量化。当前完整画像数仍为128/176。

## 2026-09-27 批内接线检查点（未运行）

F32.1–F32.3 已写入候选：公共描述、固定 SGLang hash kernel 的来源/内容锁、
prepare 冷绑定及原异步索引失败通道。数学保持成熟源码；没有新增 HYLE 数学实现。
F32.4 共享消费者已接入四路线 Direct/Graph：四画像的真实词表/专家域查表，
Flash 两个 GGUF 的完整 routed/shared MoE，下游保留 MXFP4 与 Q8_0/BF16 的逐角色格式。
官方完整 MoE 仍待 FP8 shared 接缝，不能由 router 局部测试代签。

新增测试代码覆盖变化 token/table/logits、重复专家、极端分数、1/4/5/513 token 的 warp 尾部；
CUDA 另覆盖两个独立 captured invocation、坏索引后恢复、普通 Direct、分配失败回滚和
未完成 Completion 保活。这些 C++ 测试**尚未编译或运行**，此检查点不是 PASS。
与 sink/压缩状态等同一 DeepSeek 批次接齐后统一冻结并验证，不为本小项新建构建树或启动 P1。
批内轻检查：closure 57组件/schema4、文档合同及模型输入测试13/13通过，diff检查无错误；
selector WORKTREE plan为READY，影响面与skipped理由保存在本地
`.scratch/validation/d5-deepseek-f32-plan.json`。这些检查不代签尚未运行的native门。

## 首批 CUDA 的恢复流程纠正

候选 `87f2cb03` 完整 CUDA 批次中，原生 Provider 通过；Graph hash 测试在首个坏索引完成后，
错误地复用已退役的 Invocation，第二轮 submit 因而被拒绝。公开 `graph/executable.h` 明确要求
终态失败后从同一 Executable 新建 Invocation；源码 `Invocation::submit` 也拒绝 failed phase。
这不是 hash 数学或执行器缺陷，不重开 CLOSED route。沿 F32.4 的一轮具体纠正，仅修消费者：
断言失败槽不可再提交，然后重新创建、绑定和 capture 该槽；另一个槽及 Queue/Executable 保持，
直接 Compute 仍检验其本来的可恢复合同。保留失败证据，与 F35 定向 CUDA 批次合验。
