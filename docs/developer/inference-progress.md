# 推理看板

更新：2026-11-01。看板由用户维护；agent 按顺序做任务卡，完成后只把该卡标为完成并附提交号和关键数字。规则见 [AGENTS.md](../../AGENTS.md)。

**目标：** 在一两个有人在乎的场景做出可复现、明显领先 llama.cpp 的推理结果，并公开讲清原因。长期方向见[两项核心期许](../architecture/inference-principles.md)。

## 里程碑

| 里程碑 | 完成标准 | 状态 |
|---|---|---|
| M1 首轮冷开销 | 完成 M1-5 收尾，整理一篇讲清根因的技术文章。性能以同窗实测和 M1-5 验收标准为准。 | 进行中 |
| M2 主场景 | 按 M2-1 的实测选定一个场景，在该场景领先 llama.cpp 至少 1.3 倍，或做到它做不到的事 | 未开始 |
| M3 发布 | README 第一屏是结果表和一条复现命令；文章公开 | 未开始 |
| 上游 PR | 1–3 个性能 PR 合入 vLLM、SGLang、llama.cpp 或 FlashInfer，方向取自 M1、M2 查清的问题 | 未开始 |

**当前执行：** [2.D输入与块间连续执行](base-ip.md)已完成，提交`1715d7c9c`、`5f5aef3db`、`319114961`。16K／32K普通prefill为参考101.9%／101.7%，普通／MTP的32K decode准备为1K的1.09／1.01倍；最终decode不低于原始基线98%，显存最大采样峰增量4MiB。[性能表](performance/decode.md)保留初验三格失败、五进程确认和有限检查边界。2.B、2.C原完成状态保留；本goal结束，2.E／2.F待启动。下方M1条目保留历史验收边界，不重做已提交成果，2.D结项不代签M1整体完成。

**暂停：** GLM、Kimi、DeepSeek 组件，主场景用不到的后端，基础库新功能，多卡，通用 loader、tokenizer、server。[base-ip总计划](base-ip.md)继续维护0～3阶段；2.B～2.F及阶段3仍保留。

## 任务卡（按顺序做）

### C0 收尾工作树里的 1.C 改动｜已完成
- **完成：** `975f03ffd`；驱动8组、CPU／CUDA普通接线检查及两项CTest通过；普通与MTP greedy两轮各128个输出ID均与10-31窗口一致。
- **目标：** 处理工作树里尚未提交的普通流水接线改动（目前约 15 个文件，在 `engine/models/qwen35/`、`engine/inference/`、`tests/` 等处）。
- **验收：** 相关目标编译通过，改动涉及的测试通过，Qwen3.5-2B 普通和 MTP greedy 两轮输出 ID 与 10-31 窗口记录一致，然后提交。做不到就把改动存成 `.scratch/c0-wip/wip.patch`、恢复这些文件，再汇报。
- **不做：** 不扩大 1.C 的范围，不做新的性能测量。

### M1-1 一条命令的按轮对照｜待做
- **目标：** 把 `.scratch/inference-default-product-window-20261031/window.py` 的做法整理成提交到仓库的 `scripts/inference-compare.py`，模型、参考程序和输出目录都由参数给出。
- **验收：** 一条命令在同一时段交错运行 Hyle 与参考的普通、MTP × 单轮、两轮，每臂 3 个计时进程和 1 个独立内存进程，输出一张表：按轮 decode 的中位数和范围、prefill、启动到 ready、显存采样峰、Hyle 每轮录图次数、双方版本。本机按M1-4的新协议跑一遍；旧预知schedule／缺少参考warmup的数字不能作为验收锚点。
- **不做：** 不改引擎代码，不生成证据 JSON 或 ZIP。

### M1-2 启动时完成录图和预热｜待做
- **目标：** 首轮 decode 不再付录图和流水预热的成本。10-31 日志里，普通和 MTP 都在首轮前 5 步录图（visible 155–164）；普通每次录图那步多约 4–8ms，流水接力到第 9 步才开始。
- **先做：** 确认 CUDA Graph 的复用条件不绑定具体会话的地址；如果绑定，停下汇报。
- **验收（用 M1-1 的命令）：** ready 之后 decode 的录图次数为 0；普通和 MTP 的单轮、两轮 decode 中位数都不低于同时段参考；启动到 ready 不慢于参考；显存采样峰增加不超过 64MiB；greedy 输出 ID 与改动前一致。
- **不做：** 不改 kernel 和稳态路径，不处理多请求合批。

### M1-3 MTP 后继门控｜验收未通过（候选保留）
- **目标：** 按接受历史／可用置信度决定是否提前提交后继 B，减少猜错后的无效执行；提交时保留原设备接力。
- **负载：** 固定8个不同的聊天、代码、中英文、摘要等 prompt；用模型本地 tokenizer 生成 ID，命令写入提交信息。原重复 prompt schedule 仅作专项回归，不能证明真实多轮收益。
- **策略：** 先记录信号与接受结果，从连续接受规则开始；不够准再用草稿 top1 概率或 top1／top2 差值。缺失信号如实记为 unavailable，不把历史次数当概率。
- **显存：** 启动仅预录 T2 Round／CatchupRound；T1 首次使用时录图，其冷成本计入实际 decode。停止追查分配内部字段。
- **验收命令：** `.scratch/inference-mtp-gating-20261101/run.py timing` 与独立的 `memory` 阶段，完整参数见同目录 `commands.txt`；正式计时每臂至少3个独立进程，同窗交错。
- **通过标准：** 不同 prompt 每轮 decode ≥ max(关后继, llama.cpp)；重复 prompt 第3–8轮 ≥ 原开后继的97%；greedy ID 不变；显存采样峰相对 `88e0f96b6` 增加 ≤64MiB。逐轮报告接受率、猜错次数及门控信号的真实含义。
- **不做：** 设备端接受、kernel、ready 时间。约94%的盈亏接受率仍是粗估，不能当已测门槛。
- **当前结果：** 固定token负载可用于功能回归；旧预知schedule测试的性能／显存结论作废，仅留排查日志。等待M1-4有效协议重测，不据旧结果结项。引擎候选未提交。

### M1-4 Prefill不依赖预知长度｜M1-4c按用户调整范围收尾
- **收尾：** 设备连续MTP `67c6a9450`、有界有效行prefill `774abccaa`、独立冷准备 `eb767e022`。短普通／MTP合并decode领先3.37%／8.2%，同配置显存增量0；原短普通prefill为参考93.9%、普通ready慢80ms，95%／+50ms门槛未通过。短prompt约2ms与ready差距保留，不纳入M1-5。长16K decode退化已由 `cf56700c2` 修复。新增MTP teacher唯一第13项argmax不同、oracle分差0.085544586<0.1，teacher反馈14—16项一致；状态字节差异未自行设容差。原目标和验收保留作边界参照，详见[性能表](performance/decode.md)。
- **目标：** 冷期只依据已到达首轮与显式容量配置准备；后续请求长度未知。旧整schedule预知测试作废，不能另立为有效工况。
- **先测：** Hyle只暴露首轮给冷规划，未来prompt／配额不影响family、载体、workspace、replay策略或主机缓冲预备；参考使用固定batch容量和默认启动warmup。每轮报告prefill／TTFT、shape admission和真实载体创建时间；启动加载／编译／准备／warmup单列并计入完整进程。
- **再改：** 首次启动预热归引擎初始化，清理临时KV／GDN／RNG后才ready，测试只观察费用。优先复用现有有效长度prefill路径，采用与未来输入无关的少量容量档位／分块；工作区按配置上界复用。填充行不写KV、不推进GDN／conv；复用原权限与真实退休机制。
- **验收：** 普通和MTP，同窗交错，每臂至少3个独立进程；冷期不知道后续长度时每轮prefill不低于做过启动warmup的参考，ready不慢于参考，显存增量≤64MiB，greedy ID不变；完整进程耗时及逐轮decode同时复核。新协议下未测项不得用旧数据补齐。
- **不做：** Prefill数学kernel或decode优化，不靠挪ready标记、扣局部耗时或预读未来形状达标。若共享workspace需新生命周期接口，先报告范围与真实限制。

### M1-5 长上下文准备收尾｜进行中
- **已提交基线：** `cf56700c2`。16K 普通／MTP decode 已恢复至与参考接近；C2048 短负载不低于旧版98%，同配置显存采样峰增量0 MiB。详见[性能表](performance/decode.md)。不再重做已完成的容量诊断。
- **当前：** 32K 普通 prefill 为参考的92.7%；主机 decode 准备从1K到32K分别增长约3.0倍／3.7倍。布局关系缓存和MTP候选页引用合并均未测出收益，已撤回；待处理旧页权限按bank常驻与增量验证。
- **目标：** 一天内让 bank 页表／布局常驻，权限增量处理；decode 每步、prefill 每块只处理新增 token／页。保留原有权限、版本、有序发布和真实退休。
- **固定负载：** 普通／MTP 四档 1024／4096／16384／32736 token，全部 C32768，每进程生成16 token；真实文本经本地原生 tokenizer 生成，参考核对实际 target／draft 容量并使用默认 warmup。正式计时同窗交错，每臂3个独立进程；不使用 profiler。
- **验收命令：** `python .scratch/inference-long-context-20261101/acceptance.py timing --output <新目录> --hyle-version <实际版本> --baseline <cf56700c2构建> --baseline-version cf56700c2 --capacity 32768 --lengths 1024 4096 16384 32736 --modes both`；`python .scratch/inference-long-context-20261101/short_regression.py timing --output <新目录> --hyle-version <实际版本> --baseline <cf56700c2构建> --baseline-version cf56700c2 --modes both`；独立运行 `acceptance.py memory-run` 采样显存，并分别记录 1K／32K 的主机准备。
- **通过标准：** 两模式下 32K 的 decode 每步、prefill 每块主机准备均不超过 1K 的1.3倍；32K 普通 prefill 不低于参考97%；四档 decode 和 C2048 短负载不低于已提交基线98%；显存采样峰增量≤64MiB。
- **正确性：** 对同一已接受前缀比较 Round 与 Single 的 teacher-forcing；argmax 不同时 oracle top1／top2 分差须小于0.1。从两种状态各继续生成16 token，完全一致或只在近平局处分叉；核对版本、RNG、有序发布和真实退休。
- **不做：** attention kernel、短 prompt 的约2ms差距、ready 和 M2。到时限仍未达标就报告数据。

### M2-1 主场景实测｜待做
- **目标：** 用实测选出 M2 的场景。候选：(1) 混合架构加 MTP 的更大模型；(2) 12GB 显存加 64GB 内存跑 30B 级 MoE，专家放 CPU；(3) 混合模型的 agent 式多轮和分叉会话（重新生成、改写上一条、共享前缀）。
- **先做：** 为每个候选列出模型、量化和下载大小，交用户批准后再下载，遵守 AGENTS.md 的下载规则。参考尽量用上游最新发布的 llama.cpp。
- **验收：** 一张表，列出每个候选下 llama.cpp 的 decode、首 token 延迟、能否运行、显存和内存，以及版本；HYLE 现在能否跑、还缺什么的粗估；推荐一个场景并说明理由。
- **不做：** 不改 HYLE 代码，不做优化。
