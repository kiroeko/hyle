# G09 纠正预算接续与 Metal 失败归属审计

状态：**AUDITED（仅 M1Q1 资源诊断），2026-10-17**。用户明确要求“改一下机制，要不改一下上限”，并追问 Metal
断言与 32,768 字节的含义。本卡按[按根因记账规则](../../../developer/review/module-closure-methodology.md#31-纠正预算按根因记账)
接续审计。原 SCOPED checkpoint 为 `9b09b57af`；此前只读阶段事实保留。
当前仅解锁末节冻结的 M1Q1 查询入口，不授权产品实现、原 fixture 重放或性能运行。

## 身份、范围与现状

- owner：`graph_route_cpu_vulkan`、`graph_route_cpu_apple`、`compute_route_apple`、
  `compute_route_vulkan`，先从 BLOCKED 恢复 SCOPED，再按末节进入有限诊断 AUDITED。
  API/算法不因改预算而变成已验收。
- baseline：`401195a12232d7315f382f9c8658c4cad40d8cc1`；产品 checkpoint 保留
  `a0cc517332129c403b4e464a2e865712a2842290`。
- 原记录：[完成唤醒主卡](inference-g09-completion-wake-correction.md)、
  [Vulkan 退休子卡](../compute/inference-g09-vulkan-retirement-barrier.md)、
  [最终失败与费用](../../../../evidence/correctness/inference-g09-completion-wake/ordinary-checkpoint-5.json)。
  原三轮耗尽、Metal device FAIL、未运行项与所有 raw/SHA 保留，旧卡不回写成未曾阻塞。
- 真实旅程仍是原 DAG CPU/native 支路及两个 slots、D256 pending paged consumer、checked owning
  result/权限归还/close；数学、尺寸、精度、Provider 和生命周期不缩减。
- 有限范围为 M1/H1/H2；G09 已实现的 W1–W4/V1–V4 暂为 KEEP。G04/G05、CUDA、公共统计/
  构建核心、vendor 升级、新 API 不在此 SCOPED 授权内。

## 调整后的预算

旧口径把入口遗漏、fixture 断言和不同 native 缺陷共用三次，不能反映同一机制是否反复失败。
用户本次调整后采用：

| 项 | 当前规则与继承 |
|---|---|
| 产品纠正总额 | 累计最多 **6 个有限纠正批次**；按下表已有 2 次，剩余最多 4 次；新根因不刷新总额 |
| 同一根因 | 产品与验证纠正合计最多 2 次，分账不重复授额；下一次运行必须有新事实、具体改变和独立复核 |
| 入口/fixture/cache | 单列账本，不占产品次数；仍占同一个总时间，不能隐去失败或反复换目录 |
| 总时间 | 原 2026-10-17 **14:08 +0800** 起点保留；依据用户允许调整上限，本卡具体增加 2 小时，将原 4 小时调整为累计 **6 小时**，硬截止 **20:08 +0800** |
| 原已记费用 | **1228.157 秒**为已记录命令累计，另有原审计/编排墙钟；不得用命令时间代替总墙钟或从零计时 |
| 性能资格 | 原每缺失 profile 一次新入口、四顺序、5%/10% A/A、n15→31 合同保持；本次不增加额度 |

本卡登记前的停止仍是旧规则下真实裁决；新预算仅自用户调整后生效，不补写旧运行授权。
source/artifact/机器资源和语义隔离证明成立时，独立验证门可以在总预算内继续；未知寿命/UB、
共享身份漂移或影响未隔离时，停止全部相关执行。重型任务仍全局串行、最多四核、编译 -j2。

## 历史分类与当前有限审计

| ID | 已有事实 | 类别 / 纠正次数 | 当前动作 |
|---|---|---|---|
| E1 | 首个 wrapper 遗漏原 GTest main 依赖 | 验证入口，已纠正 1 次 | 保留原失败和 v2；不占产品次数 |
| T1 | NaN consumer 只接受 wait 报错，遗漏合法同步边界 | fixture，已纠正 1 次 | 原错误/accepted prefix/坏 page 禁止发布保持；不占产品次数 |
| P1 | Native 完成早于权限及 late-abandon 尾部发布 | 产品，V1–V3 为 1 次 | 原 source/Windows device PASS 保留；不补写其为 NaN 失败原因 |
| P2 | split transfer 的主 fence 不证明另一 queue 的 transfer fence 已 signal | 产品，V4 为 1 次 | 原 source/Windows device PASS 保留，历史 VUID 的唯一句柄归因仍未证明 |
| M1 | Air Metal 验证态报告 D256 threadgroup 58368 >32768，assert/exit -6 | 原因未判定，尚无纠正批次 | 审计原 shader 声明、编译后资源和验证插桩；不能先判产品缺陷或工具误报 |
| H1 | Apple TSan cache 留有 cpu route 和 G10 project hook | 验证 context，尚无本次纠正 | 设计真实 Apple route/model/cache 的新有限入口，不手改旧 receipt |
| H2 | Vulkan ASan 实际 `/usr/bin/c++` 与 context `/usr/bin/clang++` resolve 不等 | 验证 context，尚无本次纠正 | 审定实际 compiler identity 和明确 context，不绕原 guard |

export-subst/DATA/PATH 等准备拒绝，以及 LOC/归档命令错误仍保留在旧 evidence，不能因本次分类
消失或变为 PASS；它们未产生新的产品纠正批次，全部费用继续受总时间约束。

### M1 的问题边界

32,768 是日志使用的**每个 GPU threadgroup 的共享暂存内存限值**，不是整台 Mac、模型、
KV cache 或所有 GPU buffer 的总内存上限。Apple 的
[设备属性](https://developer.apple.com/documentation/metal/mtldevice/maxthreadgroupmemorylength)
定义该类内存限值，[feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
列出相关 Apple GPU family 的 32KB 上限。

原日志只证明 Metal validation 的明确资源断言使进程中止，没有证明无诊断的 GPU hang，
也没有证明 hyle 正常返回可处理的 Status。
[Apple Shader Validation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage)
会插入代码并改变资源开销。新只读源码线索是显式 threadgroup 声明合计 29,184 字节，而
原报告 58,368 恰为两倍；这个线索需要绑定已锁 vendor/patch 身份，且不能代替实际 pipeline
资源或插桩归因。M1 必须先审定这项差异，不能通过缩尺寸、关层或改产品来迁就未解释的观测。
本次[只读源码审计](../../../../evidence/correctness/inference-g09-completion-wake/metal-validation-static-memory-audit.json)
已绑定原 tar、顺序 patch、最终 header/JIT/host SHA 和数组算术：Q 为 16,896 字节、复用的 KV
区为 12,288 字节。仅此源码账不能证明编译后 pipeline 的实际总量或倍增原因，原设备 FAIL 保留。

### 普通推理与成熟实现的源码对照

[Qwen3.5-2B 官方配置](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json)明确
head_dim=256、Q heads=8、KV heads=2，与原 fixture 的几何一致；每四层包含一层 full attention。
Q1/Q17 混合 decode/prefill 和 KV 跨页增长属于普通推理旅程，不能通过降低 D256 或删除 prefill
来签支持。模型官方类型为 BF16，本次最先失败的是 F16 回归，不能把 fixture 说成整模型实测。

| 来源 | 已核事实与对本轮的约束 |
|---|---|
| [llama.cpp Metal 固定 peer](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/ggml/src/ggml-metal/ggml-metal-ops.cpp#L3610-L3650) | D256 属于支持形状；小 query batch 有 vec 路径，调优分块超 device 上限时退回 baseline Q=1；不能假定任意 runtime validation fatal 都会被捕获回退 |
| [vLLM Metal tile](https://github.com/vllm-project/vllm-metal/blob/9e0e1ea38fea3f92e1f16dae215613bfb1f728e8/vllm_metal/metal/paged_attention_kernels.h#L22-L76) | D256 使用 BQ16/KV16/64 threads，源码名义 shared 为25,344字节；修改内部 tile，模型维度不变 |
| [vLLM Metal 准入](https://github.com/vllm-project/vllm-metal/blob/9e0e1ea38fea3f92e1f16dae215613bfb1f728e8/vllm_metal/metal/paged_ops.cpp#L621-L640) | reducer 在编码前核实际 staticThreadgroupMemoryLength + dynamic bytes 与 device maxThreadgroupMemoryLength；该 guard 的具体作用域不能泛化成其所有 kernel 都有同一保护 |
| [SGLang Metal](https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/hardware_backend/mlx/kv_cache/attention_wrapper.py#L309-L340) | 实际调用 mx.fast.scaled_dot_product_attention，不能把其 CUDA FlashInfer 路径当成 Metal 路径 |
| [锁定 MLX 标准 SDPA](https://raw.githubusercontent.com/ml-explore/mlx/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/scaled_dot_product_attention.cpp) | 按 query/head/device 选择 vector、two-pass、full、head-split 或 unfused；小规模 D256 prefill 的标准 fallback 不等于 Hyle paged primitive 可直接套用的实现 |

Hyle patched HylePagedAttention 直接进入 native dispatch，没有经过标准 SDPA 的 use_fallback。
现有 MLX threadgroup 检查只查线程数，不能代替编译后 shared-memory 字节准入。下一有限设计
应先查实际 pipeline 静态内存/动态绑定/设备限值，再审合法内部 tile 或成熟备用路线；保持外部
D256、paged 表、active length、causal 与 lease 语义。若备用路线需要 gather K/V，额外 GPU
复制、暂存和成本必须明确计入，不能伪装为原零复制路径。

这些是源码机制与边界对照，不是三家在本 Air、同输入和同 Metal 验证层下的运行 PASS。
29,184→58,368 仍须有限诊断证据解释，不得直接按名义字节除二或把验证层限制调大。

## SCOPED 阶段的准入（当前以末节为准）

SCOPED 本轮只运行文档/closure selector 与维护检查，读取原源码、缓存元数据、官方文档和
已封存日志。不得启动旧 verify-v4、fixture、SDK、sanitizer、层变体或新 A/A/P1。

进入 AUDITED 前必须独立审查并冻结：M1 的已知/未知资源账、归属和有限 FIX 或有限诊断入口；
各 host 入口的真实身份与适用范围；产品/验证类别、次数、剩余时间；精确验证 stages 和失败
分支。只对已审完的入口解锁执行，H1/H2 未完成不阻挡 M1 读源码，但继续阻止各自宿主门。

全部真实失败与未运行风险继续阻止聚合 CLOSED。原 SDK、旧 CUDA PASS 和窄 Windows device
PASS 都不能代签新未运行门。当前仅解除机械次数上限导致的审计停止，尚未解除技术验证缺口。

## AUDITED：M1Q1 的唯一有限诊断入口

实现者与独立审查均已核定
[冻结合同](../../../../evidence/correctness/inference-g09-completion-wake/m1-pipeline-query-contract.json)
`48303c2b...afd9b` 及
[源码独审](../../../../evidence/correctness/inference-g09-completion-wake/m1-pipeline-query-source-review.json)
`62ffa4fc...c6df`，只签 SOURCE。入口源码与只读身份在
[冻结包](../../../../evidence/correctness/inference-g09-completion-wake/m1-pipeline-query-entry.zip)。
四 owner 进入 AUDITED 仅承载本节诊断；W1–W4/V1–V4 保持 KEEP，**没有新产品 FIX 授权**。
若查询形成具体纠正设计，须先更新有限 FIX、独审并提交，再改产品。

- 精确原 kernel：half、BQ32/BK16/BD256/BV256/WM4/WN1，原完整函数名/八项 Bool 常量、
  123,307 字节 MSL 与 SHA `5749a119...22e77f`；不以普通 SDPA 或 NAX 代替 paged 变体。
- 单私有 query TU，复用原两枚字符串对象、原 source facet `libmlx.dylib`、原共享 DSO 邻置
  metallib 与原 Safe/MSL 默认值。直接工厂因未导出不可用，复用原导出的 Device 工厂与完全
  相同 MSL；不修改 exports、不重编产品/vendor，不更新原 cache。
- 只创建/查询 pipeline：staticThreadgroupMemoryLength、device 上限、maxthreads、
  execution width 与 shaderValidation。helper 不创建 encoder、不调 eval/dispatch/queue submit；
  原 Device 冷初始化及 Apple compiler/validation 内部资源不声明为零。原 encoder 动态内存
  **NOT_OBSERVED**，不能填零。
- 三个 fresh process，顺序为无验证、API-only、API+GPU，各一次；实际层激活与 requested env
  分开记录。它们是资源归因诊断，不是减少原验收验证层或新增性能资格尝试。
- 固定阶段：configure（45秒）→plan（15秒）停审，实际必须只有 1 TU+1 link；再逐次 GO
  build（60秒）/otool（10秒）停审，再三次 query（各90秒）。独立入口总限600秒，与原20:08
  截止共同约束；nice10、串行，compile -j1。任一失败/超时保留结果，不自动重跑或追加dispatch。
- source/objects/library/metallib/CompileOptions 头/controls 每阶段前后核 SHA；结果只允许
  解释本 pipeline 的资源差异或保持 UNKNOWN，不能签原 correctness/device/SDK/P1 PASS。

H1/H2 只读方案已查明：Apple TSan 要显式 cpu;apple 并清旧 G10 hook，且全阶段恢复原模型；
Vulkan ASan 要以真实 `/usr/bin/c++` 独立 context 保留路径 guard，并核真实 Vulkan owner/
fixture 的 sanitizer 编译项。新 host 控制入口仍未冻结、未授权运行，不因 M1Q1 解锁而解锁。

### M1Q1 实际结果与归属

冻结入口按阶段停审后已各执行一次；[结果与身份](../../../../evidence/correctness/inference-g09-completion-wake/m1-pipeline-query-result.json)
及[完整 raw 包](../../../../evidence/correctness/inference-g09-completion-wake/m1-pipeline-query-result.zip)
保留原 controls、实际 plan/link、三个 fresh process 和旧失败。没有重编产品、重放原 fixture 或提交 GPU 数学。
[结果独审](../../../../evidence/correctness/inference-g09-completion-wake/m1-pipeline-query-result-review.json)
与[Apple 定义/枚举原件](../../../../evidence/correctness/inference-g09-completion-wake/m1-pipeline-query-primary-docs.zip)
单独封存；旧入口的源码独审不冒充结果独审。

| 实际状态 | pipeline 静态 threadgroup 字节 | device 上限 | pipeline shaderValidation |
|---|---:|---:|---|
| 无验证 | 29,184 | 32,768 | Disabled（2） |
| 仅 API 验证，激活日志已见 | 29,184 | 32,768 | Disabled（2） |
| API＋GPU 验证，两项激活日志均已见 | 58,368 | 32,768 | Enabled（1） |

三项完整 MSL、函数常量、原 MLX DSO、资源、工具链均绑定同一冻结身份，查询均 exit 0。
因此已确认**此 pipeline 的编译后静态内存随 Shader Validation 启用从29,184增加到58,368字节**；
静态量本身已经超限，足以解释旧断言的字节值。不能再写成原 kernel 在普通编译状态本来就需58,368字节。
这不证明内部 shadow 实现，不是所有 kernel 的固定倍数；原 encoder 动态历史仍为 NOT_OBSERVED，
原完整设备验证仍 FAIL，不能由查询替签 correctness/SDK/P1。

新命令累计1.990139168秒，继承原1228.157秒后为1230.147139168秒；产品仍已用2/6，P1未增加。
私有入口原 UTC 起点09:51:52.355187，最终 receipt mtime 给出完成上界09:55:36.163807，
root另在09:55:52 UTC确认已结束，均在600秒内。原 child 精确退出UTC没有记录。
原跨进程 monotonic guard 的可靠性未证明，旧macOS Python存在进程间基准风险；此前认为同一启动
周期即可比较的判断在此更正。本次以外部UTC证明时限，原入口/raw不修改、不重跑，后续入口使用
明确的跨进程时间合同。总截止20:08 +0800保持。

资源准入与合法备用分块进入下一有限设计审计；在该 FIX 独审并冻结前，仍不授权产品修改。
