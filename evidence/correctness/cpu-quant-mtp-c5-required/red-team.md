# C5 独立收口反证

候选 `c740abc74a5a62ffacf60b26798228ca484317df`。本遍在产品/测试冻结后重新从要求和负例看证据，
没有以实现过程中的测试数直接签全范围PASS。[聚合审计](aggregate-audit.json)保存精确XML/SHA和产品差异。

| 反证 | 实际核对 | 结论 |
|---|---|---|
| 对账头等于数学通过 | 104完整header身份与559 typed numeric XML逐ID相等；34 packed及45 plain亦逐ID相等 | 这些端口合同已证明；metadata不代签运行 |
| 一个最小projection等于完整模型 | All104Identity测试只执行各身份的真实projection；目录和summary均明确不称104整模型run | 保留区别；全profile聚合仍须联合semantic/state证据 |
| GGUF F32参数代签BF16/F16 | 80 GGUF参数与45非GGUF plain分开；Qwen zero-centered norm分开；低dtype convID9实际运行及更新SDK | 对应实际dtype/几何资格成立 |
| 交错/full256 RoPE代签目标Qwen | CPU-only actual helper三真实Q/KV几何、HalfSplit64、theta1e7、epsilon1e-6、实际KV反馈 | 原旧fixture不代签；新准确域有两系统收据 |
| MTP输入代签完整NextN | 新完整2.4T消费者额外执行GQA/MoE/shared gate/residual/head norm/full V；原Qwen sidecar和GLM NextN有完整独立程序 | 不能复用输入边界替代完整程序 |
| 量化对照用未量化点积 | F1/newNextN及Qwen sidecar含原activation codec/vec-dot；packed decoder保持原packing/scale/group/dtype | 非苹果对橘子；没有新性能赢家 |
| 降驻留capacity等于偷减内宽 | 检查所有F1/F2/45 plain实际K/N、完整V；expert8是计划允许合成驻留，router/top-k及native indexed数学真实执行 | 明确验证容量限制，不签offload引擎 |
| 库资源只通过metadata mock | 原allocator失败、failed slot退役/恢复、并发readonly/独立outputs、close和metadata回收实际运行；raw_deleter代理恢复原接口 | 资源复用只在同runtime/未变管理机制；unknown私有上界不写0 |
| 旧binary代签新接缝 | 大Q8 table有自己的64 updatedSDK；ID9 lowdtype conv有另360 updatedSDK，普通find_package两入口 | 新接缝安装资格成立；旧记录scope不扩大 |
| 两版ggml对象混入SDK | CPU-only新core，含GPU旧envelope；同源baseline/AVX2及private program名带身份；C2/C3配置/安装路径证据 | 一SDK一core；没有新GPU资格 |
| 旧Native WIN胜过新peer | C2编译期peer456172条件；新版CPU-only关闭旧187优选，实际consumer要求成熟实现/原程序ID | 不重开自研竞赛，不删除SIMD |
| 软件标记相同等于实现相同 | Git产品diff精确只两CPU adapter；旧Torch ID1–8及原math未变，新增ID9；encoded Gather只扩特定准入 | 逐差异/合同复用；不是笼统同tag复用 |
| 文档修复推动假百分比 | 未写104 full-model RUN_PASS；本轮Operator/parameter/complete-region收据分开，所有未聚合项仍INCONCLUSIVE | 不因文档或旧CLOSED宣布goal完成 |

## 尚未作出的聚合结论

技术接缝反证没有发现新产品缺陷。最终仍须对每个准入制品把typed端口、必要转换、
共同semantic/state程序、identity、生命周期与三交付联合证明，并给出每类所继承的准确合同。
这些单独证据不能通过改名为整模型run来消除作用域差异。

本遍只签上述有限技术反证，**整个scope/red-team聚合门目前仍INCONCLUSIVE**；
完成资格账本与公开使用文档后再裁决CLOSED。未提出新kernel/依赖版本/P1或跨路线恢复。
