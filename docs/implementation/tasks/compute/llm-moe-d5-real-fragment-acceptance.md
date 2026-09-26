# M3-D5：真实模型片段批量验收

<!-- MODULE REVIEW: AUDITED -->

**2026-09-26 当前进展：** 用户选择的完整 recurrent 数值边界已完成 MG1–MG5；
九个 Qwen 与三个 Kimi 画像已通过四路线 Direct/Graph 普通入口、变化输入和实际状态反馈。
当前 ledger 为 **96/176 RUN_PASS、80/176 VALIDATION_NOT_RUN**。Qwen 见
[九画像收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen-nine-profile-complete-gate/README.md)，
Kimi 见[三画像收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/kimi-remaining-batch/README.md)。
Metal workspace 分配失败已按[有限修复卡](llm-moe-d5-metal-buffer-failure.md)修复。
旧失败及撤回资格保留；下文0/48/72等数字是历史快照，不能覆盖当前 ledger。
其余10画像、D5 三交付/P1及 D6 尚未完成，不把完整片段正确性当作整个 goal 完成率。

**GLM 新增批次已验：** 六组共享 consumer 在四路线 Direct/Graph 各6/6通过，共48组，
包括真实宽度 FFN、压缩 MLA、DSA/Flash pool-tail、加权 mHC 和 NextN 输入边界。
F30 的64-head准入和F31的 CUDA 原生 I32 Concat 已验证；Vulkan布局与原生精度失败已归因。
见[批量检查点](../../../../evidence/correctness/llm-moe-d5-real-fragments/glm-remaining-checkpoint/README.md)。
完整 MTP、官方 FP8剩余域和完整画像门仍未签，ledger维持96/176；实现58,617行/232文件。

**GLM decoder/稀疏cache接合已验：** `3fb5e818` 的四路线 Direct/Graph 各10/10通过，
合计80组、零跳过，其中每入口新增NextN decoder实际hidden/cache回送和DSA索引直接驱动
compressed-cache Gather/Attention两组。完整GLM片段组由6增至8（八入口64/64），另有16组
MoE回归；原CPU失败和有限consumer纠正保留于
[新检查点](../../../../evidence/correctness/llm-moe-d5-real-fragments/glm-nextn-decoder-checkpoint/README.md)。
MTP索引跨轮carry/reset、全词表head组合及官方FP8剩余域仍未签，不增加完整画像格。

**下一共用批次已开始：** 其余13画像的 R01 复用已有八入口词表消费者，保留 Kimi 163840、
DeepSeek 129280、GLM 154880 的完整词表和真实 hidden。逐角色 BF16/Q8_0 由已保存的官方/
GGUF header 取证，写入 `tests/compute_models/d5_moe_profile_facts.json`，不重新下载模型。
测试 helper 参数化词表容量并处理整除分片的 Kimi；不增加产品算子或 Provider。
该批接线/编译检查不改变完整画像 ledger，其余 region 组合后才统一签整格、交付和性能。
候选 `9f88d8b0` 的 CPU Direct/Graph 定向 R01 各26项通过，见
[检查点](../../../../evidence/correctness/llm-moe-d5-real-fragments/moe-vocabulary-cpu-checkpoint/README.md)；
其它路线未运行，不据此增加完整验收格。

**共用语义核心批次（2026-09-26）：** `d5_moe_semantic_fragments.h` 复用现有八个
Direct/Graph consumer，接入六画像 KDA、三画像 Kimi SiTU、七画像 mHC。KDA 保留
96/64 heads、D128、K4、逐 key 通道的 `-5*sigmoid(exp(A_log)*(g+dt))`，八次变化输入
与路线自身输出状态回送；输出门是 sigmoid，不套用 Qwen 的 Softplus/SiLU 门。
mHC 保留 hc=4、真实 H=7168/4096、非均匀 Sinkhorn(20)、独立分支输出与原 residual；
旧 SiTU 测试采用 width=3584、beta=4、linear_beta=25；该宽度资格已在下文更正，不能代签
Kimi 的实际 expert/shared/dense 分支。这是 F32 运行核心和投影后边界的
验收，不代签 R03 投影、官方 BF16 参数转换或完整模型画像。权重内容为合成非零值。
候选 `f0bec3afe` 的四路线八入口各29项通过（含十三画像 R01），共232项、零失败/零 SKIP，见
[批量收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/moe-shared-core-batch/README.md)。
整格 ledger 仍为72/176；三交付与 P1 未运行。统一实现统计仍为58,255行／228文件。
公式来源为已登记 llama.cpp `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a`
的 `src/models/kimi-k3.cpp` / `deepseek4.cpp` 与 Transformers
`c1b80cf3ee183f94d8b21dfa7e312d1eea4942ad` 的 `models/glm5_next/modeling_glm5_next.py`。
本批仅测试接线，不新增产品数学，也不触发各 profile 的完整 selector、三交付或 P1。

**KDA 投影角色批次：** 从已保存 tensor header 提取六画像的51个 q/k/v/o、beta、f、g 投影角色，
按同一 profile、输入/输出形状和原生 dtype 归并为30个 MatMul 签名；每个原权重名均保留于
`d5_kda_projection_facts.json`，不按 artifact 的 Q 标签推断全部 dtype。普通八入口统一遍历同一张表；
BF16 保留 BF16 输入/权重/输出舍入，Q8_0 保留 native block，F32 保留物理转置布局。
候选 `40699b3d7` 的四路线八入口全部30签名均通过两次变化输入，见
[投影批量收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/kda-projection-role-batch/README.md)；
本批仍不是完整画像 verdict。
参数事实另保留 Kimi 官方 A_log[128] 与成熟加载器取前96项、GGUF folded ssm_a[96] 的关系，以及
GLM Flash 官方 BF16 conv/norm 与 GGUF F32 的区别；这些参数转换不由投影结果代签。

**十三画像 MoE 批次与形状更正：** tensor header 与固定 llama.cpp 的 latent MoE 公式确认，
Kimi 的3584是 latent expert 输入/输出维度；expert SiTU 中间宽为3072，shared 为6144，
dense 为33792。上一批3584宽测试只保留通用公式证据，撤回它对真实 SiTU 宽度的资格；
本批共享测试显式遍历三个实际宽度，不变更旧日志或将旧结果改写成新形状的 PASS。
`d5_moe_execution_facts.json` 固定13个真实 router（896/384/256/288专家及16/6/8路选择），
逐角色 BF16/F32 router、sigmoid/sqrtsoftplus、仅影响成员选择的 correction bias 和1/1.5/2.5权重缩放。
八个 GGUF MoE 使用实际 MXFP4、Q4_K/Q5_K、Q8_0、BF16 组合，包含 routed/shared 全链与 Kimi
latent down → experts → norm → up。只将 resident expert capacity 缩为 top-k+1，完整专家打分宽不缩小；
三个变化输入改变成员和次序，输出 expert ID 单独精确比较，不使用量化数值容差蒙混成员错误。
本批尚未完成四路线验收，官方 expert 打包、DeepSeek hash 层、其它 attention/MTP 仍待完成。

候选 `a11aac6a` 的 CPU/Apple/Vulkan Direct/Graph 本批各5/5通过；CUDA各3/5，数值均通过，
失败仅为 DeepSeek 普通 Softplus 仍落到冻结 HYLE 实现的成熟 Provider 断言。
原生 Clamp 的四路线边界已通过；初始 Q8 反例已由独立激活量化计算解释，修正合成 router
权重/输入比例并保持相同分数和原误差门，旧失败保留。详见
[MoE 部分收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/moe-execution-batch/README.md)。
下一有限接缝为固定 SGLang 原版 `moe_fused_gate` 的完整 DeepSeek routing 接入审计；
先确认 bias dtype/转换、tie/归一化合同、原 kernel 依赖与现有 owner，再固定有限 FIX，
不抽 helper 新写数学，不为该缺口重做全平台历史审计。官方 expert 等其余项目继续按上段保留。
整格 ledger 不变；统一 Compute+Graph 实现统计为58,309行／228文件。

**F25 融合路由批次已验证（`e25c5188`）：** CUDA 的 DeepSeek 得分路由已改由原版 SGLang
完整程序承载，小/大 token 分支、并列/负尾部、资源拒绝与 OOM 通过。十三画像得分路由与八个
GGUF 完整 MoE 链在四路线八入口全部通过；本批20个测试组、公共 schema55项、SGLang回归6项
及独立头2项均通过。见[收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/sglang-router-batch/README.md)。
这消除了上一批 CUDA 的成熟 Provider 断言失败，但不代签官方专家格式、DeepSeek hash 层或其它
attention/MTP。整格仍为72/176；最新实现为58,561行／232文件。进度报告区分这些已验片段和
完整画像，不把文档、构建步骤或固定百分比当作新增验收。

**D5-F23 明确重新定界：** 用户选择后，四个 ggml Provider、Apple Direct/Graph 与 Vulkan Direct/Graph
当前 AUDITED 范围仅增加[MG1–MG5](llm-moe-d5-model-gate-qualification.md)的完整 recurrent
region 接线。拒绝独立发布精度未合格的 gate；不改通用 Softplus、不新增数学。原 F1–F22
验证事实和 D5 三交付/P1 待办保留。

接线源码审计补充：`compute_route_vulkan` 的 `region_operation()` 使用六输入栈数组，无法将
合法八输入完整 recurrent schema 送达 Provider。该 owner 已为 F22 处于 AUDITED；本次仅将容量
扩为八，并仅对具名 `RecurrentGatedDeltaNet` 放行超过六输入的请求，其他准入域不变。

**D5-F24 有限重新定界：** 十三画像 MoE 普通入口确认 Binary 限幅落到冻结 HYLE 实现；
固定为[成熟 Clamp 接线](llm-moe-d5-mature-clamp-binding.md)的 F24.1–F24.4，
追加四个 ggml Provider 与 Apple/Vulkan Direct/Graph 当前 AUDITED 范围；不新增数学，
不放宽 MoE 数值门，完整片段数值失败另按中间输出定位。

### 官方专家格式共用批次（A2/A3 的有限测试接线）

baseline `2715701f`；owner 仍为 D5 `compute_ops` 的共享消费者，不修改产品或重开已关闭路线。
已保存官方 header 与固定 llama.cpp `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a`
的 `conversion/base.py::repack_mxfp4_blocks`、Kimi/DeepSeek 调用点确认：三者 FP4 codes 相同，
源布局为相邻元素高低半字节加独立 UE8M0 scale，ggml 布局为前后16元素配对加同一 scale。
该转换只移动位，既不反量化也不重新量化；属于模型消费层的准备动作，不增建产品 loader。

本批固定：①共享源 payload/scale 夹具和独立源格式数值 oracle，使用原版 Python 转换生成
逐字节 golden；②Kimi、DeepSeek Pro/Flash 的 gate/up 与 down 六个真实内宽签名，resident
expert 数仅缩为 top-k+1，普通八入口各两次变化输入/成员；③Kimi 官方完整 MoE 复用现有
router/latent/shared/SiTU 链，routed 权重改为上述真实 FP4，shared/latent 保持官方 BF16。
源 payload 与目标 storage 同时仅存在于准备夹具，执行不重排、不保留全量 F32 权重副本。
测试权重内容为合成值，不宣称巨型 checkpoint 内容正确性或原模型部署的激活量化策略。
一个共享接线批次后串行运行四路线两入口；原 GGUF MoE 同批回归。既有资源与公共 schema
未改，证据复用；不启动本批独立三交付/P1。具体失败只定向修正一次，否则保留精确缺口。
DeepSeek shared FP8 的 UE8M0 128×128 scale、GLM FP8 的 F32 128×128 scale 另行审计，
不能以 BF16、UE8M0 group32 或本批 FP4 代签。完整画像 ledger 不因单独专家格式通过而增加。

**本批结果：** 候选 `4c862164` 的六个官方 FP4 投影签名、Kimi 混合格式完整 MoE 计算链与
八个 GGUF MoE 回归，在四路线 Direct/Graph 共24/24测试组通过，零跳过；见
[批量收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/official-fp4-expert-batch/README.md)。
源格式与固定原版 converter 的逐字节 golden 相符，不新增产品数学；实现仍为58,561行/232文件。
本批没有暴露产品缺陷，无需重开已关闭路线。完整画像仍72/176；官方参数读取/转换、剩余片段、
三交付/P1 的未验边界详见收据，不以片段通过代签。

### 官方 FP8 dense 边界批次（A2/A3，AUDITED）

baseline `1c4e8444`，owner `compute_ops` 共享消费者；产品 FIX 为零。固定 GLM-5.3/Flash
及 DeepSeek Pro/Flash 的八个 gate/up、down 真实内宽。payload 为 E4M3FN；GLM scale 是
128×128 F32，DeepSeek 是128×128 UE8M0。后者可在消费层仅把 scale 无损扩为 F32 接入现有
block128 描述，不使用 group32 代签，也不展开权重。激活输入是已准备的 E4M3FN/group128；
此批明确不代签尚未接入的动态激活量化或 grouped expert 选择。

有限动作：①共享八签名/源格式 oracle 和两次变化输入；②Vulkan 复用原版 IREE dense 程序运行
普通 Direct/Graph，保留其 F32 临时 workspace 成本；③CPU/CUDA/Apple 普通准备/编译记录该
具名编码的实际拒绝码，关联既有原版库证据，不能把入口拒绝升级成所有上游方案皆不可能。
固定 oneDNN 3.13.2 在本机 CPU 的 FP8 域、CUDA 13.3 在 SM89 的 block128 scale 域已有
[具名拒绝](../../../../evidence/correctness/compute-llm-moe-m2-b6/qualification.json)，不重测相同
底层探针；Marlin 的 W8A16、SGLang 的 rowwise scale、MLX 的 group32 不能冒充完整动态
block128 合同。只做一次共享批次及一次具体失败纠正；未运行保留未验，不触发独立三交付/P1。
本批只签 dense 片段及拒绝边界，完整画像仍由全部 region 或精确终局缺口统一裁决。

### D5-F28：IREE FP8 原生 dispatch 准入（AUDITED）

baseline `3f8caf58`。F27 的真实八签名均在 Vulkan Graph 被 `native grid or workspace limits`
拒绝；Direct 夹具的 `.value()` 隐去了同一错误。当前 `compute_route_vulkan` owner 的有限范围
追加本项，Graph 复用该 route，不重开其它产品 owner。
对仓库锁定 `fp8.vmfb`（SHA256 `5ee3469dcb7db4e7f560fd19e6592803c1fa6a355d27c429adcfa8a85b4a059b`）
运行同 pin `iree-dump-module`，确认两个 FP8 export 的三次 dispatch 都是 X 轴
`min(rows*ceil(columns/64),65535)`、Y/Z=1；三域依次为 M×K、N×K、M×N。
旧检查却要求所有矩阵的元素数不超过65535，把元素数错当实际 workgroup 数。
F32 临时 allocation 为 `align64(4*M*K)+align64(4*N*K)`，输出为4*M*N，不能只扩大元素门。

有限 FIX：F28.1 仅替换 FP8 准入为上述锁定实际 dispatch、合并 workspace 的 descriptor range
及原有32位索引边界检查；不改 bytecode、shader、数学、Scan 或 BF16 conversion 域。
F28.2 八个真实 FP8 签名在普通 Vulkan Direct/Graph 各 M1/M3、两次变化输入通过；低网格上限、
合并 workspace 超限、巨大形状必须在 native allocation 前拒绝。原 IREE FP8/资源/OOM/Scan
消费者同批回归。复用其它路线已取得的冷拒绝，不重建它们。
预算仍限一次具体原因修正；稳定后的 D5 三交付/P1 待办保留。若锁定程序执行仍存在能力缺口，
保留实际错误，不修改原数学或继续扩大上限。

**F27/F28 批次结果：** 初始 Vulkan 拒绝确为上述 HYLE 冷准入问题，有限修复 `652d8083` 后，
八个 FP8 真实签名×M1/M3 在 Direct/Graph 均运行通过；最终 `4f60924b` 的逐值相等比较也通过。
grid/workspace/索引、混合 Graph、IREE Provider 与 quota 失败等15个回归项通过。
CPU/CUDA/Apple 的各16个普通入口冷拒绝均有实际日志；不将入口拒绝外推为所有上游方案缺失。
见[批量收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/official-fp8-dense-batch/README.md)。
动态 activation quantization、grouped FP8 及完整画像仍未代签；ledger 72/176不变。
实现为58,588行/232文件，新增仅冷准入检查，没有修改原版数学或 bytecode。

### KDA 完整链共用批次（A2/A3，AUDITED）

baseline `ede9d55e`；owner 为本卡 `compute_ops` 的共享测试消费者，产品 FIX 为零。
固定六个 Kimi/GLM Flash 画像，将已经验证的51个投影角色和 KDA 核心组合为普通完整 region：
q/k/v 投影、K4 causal conv、L2 norm、低秩 decay gate、beta、GDN、输出 norm/gate 与 o 投影。
Kimi 为全秩输出门，GLM Flash 为 g_a/g_b；真实 hidden/head/depth 与逐角色 BF16/Q8_0/F32
均按 `d5_kda_projection_facts.json`，不按模型全局量化标签推断。BF16 投影使用现有成熟
`BFloatLinearF32` 的 BF16 输入/权重、F32 累加输出合同；Q8 保留原生 storage 与上游点积。
这验证原始权重存储在 HYLE 普通入口的完整消费，不宣称与某一 PyTorch 部署逐位一致。

有限动作：①一个共享完整链 recipe 与独立数值 oracle；②保留 Kimi 官方 A_log[128] 的前96项、
GGUF folded negative-A，GLM 官方 BF16 conv/norm 的实际 Cast 与 F32 state；③八次变化输入，
卷积历史和 recurrent state 均从实际路线输出回送，oracle 独立推进；④共享多状态夹具迁移，
已有六画像 KDA 核心与九个 Qwen 单状态消费者同批回归。沿用既有 F32 与量化链数值门，不能
遇到失败后扩大容差。参数和权重为格式合法的合成非零值，不下载巨型 checkpoint。
接齐四路线八入口后固定一个候选并串行验证；只对具体失败定向纠正，不增加算子、Provider、
新构建树或独立 P1/三交付。通过后也只签完整 KDA region，其它 attention/MTP 等仍须完成。

参数复核更正：保存的官方 config 明确两模型 `rms_norm_eps=1e-5`；固定 Kimi mature source
同时把它用于 Q/K L2 与输出 norm，GLM KDA source 的 L2 独立固定1e-6。初始 `9bb6be90`
八入口通过但沿用旧夹具1e-6，不能签上述真实配置。有限测试纠正同时覆盖新完整链、旧 KDA
核心与 Kimi latent MoE 的 norm/oracle，追加原官方 Kimi 和 GGUF MoE 批次；容差不变。
旧核心与 MoE 收据保留为原参数的事实，撤回其对本项真实 norm 参数的资格；无需改产品实现。

**本批结果：** `c800b76e` 的完整 KDA、修正参数的核心和官方 Kimi/GGUF MoE 在四路线八入口
共32/32组通过，零跳过。共享多状态 helper 的九画像 Qwen 回归已有八入口合格收据并按未改范围复用。
见[完整链收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/kda-complete-chain-batch/README.md)。
ledger 新增48个 R06 `partial_receipts`，完整画像仍72/176；不把 region 通过改写为整画像通过。
最新实现仍为58,588行/232文件，没有新增产品数学，剩余片段、三交付和 P1 继续按 D5 主卡执行。

### Kimi 剩余三画像与 D5-F29（AUDITED）

baseline `21d81e66`；`compute_ops` A2/A3 的共享消费者批次补齐三画像的 dense FFN、MLA 和
Attention Residual。固定原始 header/config、llama.cpp `4c9233c` 的 `kimi-k3.cpp` 与
`conversion/kimi_k3.py`，角色事实保存在 `d5_kimi_remaining_facts.json`。
H=7168、dense FFN=33792、heads=96、Q rank1536、KV rank512、QK=128+64、V=128；
MLA 不执行 RoPE，但保留其64维附加分量与原192维缩放。GGUF 使用实际拆分后的 K/V 权重；
官方 KV-B 仅作原 converter 的 BF16 位转置/拆分。三画像各自存储格式不相互代签。
Attention Residual 对规范化数据打分、对原始 residual 加权，覆盖首层无 bank、12层边界
保存原输入并重启 prefix、普通层累加及末层混合。参数 norm epsilon 使用原配置1e-5。

F29 的具体反例：公共 SDPA schema 支持不同 QK/V depth；Kimi absorbed MLA 的 Q/K=576、
V/output=512 合法，但 ggml CPU/CUDA/Vulkan `supports` 仍要求 K description==V、output==Q。
固定原版 MatMul/Softmax attention lowering 已按实际 V 维度构图，未写死相同 depth；问题在 HYLE
冷准入。当前三个 Provider owner 的 AUDITED 范围追加 F29.1：仅放行本批 F32 576/512、96/1 heads
且 batch/sequence/输出维度一致的 native 域，保留其它 dtype 和已有等深域。不改 runtime、原数学、
公共 API、默认用户配置或 route owner。F29.2：完整压缩 MLA 链，真实 cache 输出回送与变化输入，
错配 K/V sequence/head/output 在准备前拒绝；已准入 attention、资源失败/OOM 同批回归。

共享 consumer 的 per-head 权重以现有 GroupedMatMul 绑定固定 head ID，保留一条成熟调用，不循环
逐 head 建图。BF16/Q8 storage、F32 norm/cache、吸收前后布局与 Attention Residual 融合参数分别
验证。全部四路线接线后冻结候选，一次串行批验；具体失败限一次定向纠正，不能通过改容差或新增
HYLE 数学绕过失败。R01/KDA/MoE 已有精确收据按身份复用；整画像必须等剩余 region 和对应完整
owner 验收齐备才更新，三交付/P1仍按 A5/A6。无上述实测证据不宣称画像完成。

### Kimi BF16 验收分类纠正（A2/A3，AUDITED）

`b0cc483d` 的 CPU Direct/Graph 暴露夹具合同问题：相同 F32 norm 中间量仅差
5.960464477539063e-8，却位于 BF16 的 -0.748046875 中点两侧，分别舍入到0xbf40/0xbf3f；
SiTU 的0.210449203849/0.210449248552 同样跨0x3e57/0x3e58。原独立 oracle 在每个投影前
舍入 BF16，但错误地要求整个 BF16 链仍满足 F32 逐点门；数学连续误差不保证离散舍入分支相同。
ggml 原版 `ggml-cpu/ops.cpp::ggml_compute_forward_rms_norm_f32` 使用 F32 乘法/均值/scale，
独立 oracle 使用 F64 中间计算；二者的这一差异本身在原 F32 门以内，不能据此修改原数学。

有限纠正撤回上一段对这两个新 BF16 完整链的 F32 数值分类，复用已验 Qwen BF16 三投影链
`kExpertRelativeTolerance=0.025`、`kExpertAbsoluteTolerance=2e-5`，不根据本次差值调参。
仅显式标记经过 BF16 投影的输出；独立 F32 输入 norm 仍用8e-5+4e-5*abs(ref)，residual bank
仍精确比较，Q8 原门不变。独立 oracle 与状态轨迹不读取实际结果；新增正负 BF16 中点及相邻
F32 的 Cast→Cast 逐值相等控制，防止数值分类掩盖转换错误。四路线共用一份规则，无平台特判。
新增输出保留 input norm 与 gate projection，诊断日志和旧失败均保留；不将新规则倒签旧候选。

残差失败的原因则是夹具把原模型 append 写成 assignment Scatter；原版 `res_push` 只做 Concat。
有限容量测试改为 Concat 后 checked Gather 保留8行，输入 indices 只表达有效槽位，数值与原追加
语义相同。每轮多搬运一行候选及一次8行选择，成本在本批收据记录；不接回冻结自研 Scatter。
本节仅调整共享测试消费者与验收分类，无公共数学、路由策略或产品 API 变化；新稳定候选必须
实际完成四路线八入口检查，剩余整画像/交付/P1 不由此代签。

**Apple 原生精度边界纠正：** 首次 Apple Direct 的 dense/残差/精确 Cast 全部通过，MLA 未通过。
运行日志明确选中 `kernel_mul_mv_id_bf16_f32_4`；锁定原版 Metal `kernels/mul_mv.metal:3367`
以 bfloat 权重和 float4 激活相乘，不像 CPU BF16 vec-dot 隐式舍入激活。共享 oracle 错将后者
套到 Apple，尤其在 V-B 输入近零相消处引入假差。仅把 Apple 的独立 per-head K-B/V-B 参考改为
保留 F32 激活；产品 recipe、输入、Provider、误差门均不变。其它路线保持已验参考；原 Apple
失败保留，修正后的 Direct/Graph 必须实际运行。官方完整形状 KV-B 拆分/转置及三种残差参数融合
已执行锁定原 converter 的五项逐值相等检查；全层 header 复核确认本批20/18/18个签名无遗漏异型。

**Kimi 三画像收口结果：** `9d8d90e4` 的 CPU/CUDA/Vulkan/Apple 完整受影响 owner suite
全部通过，零失败/零跳过；此前本批八入口48/48组通过。F29 只增加原版 MLA 的冷准入，
BF16 分类及 Metal 原生激活精度的参考纠正、完整原 converter 检查、全层权重签名、R01/KDA/MoE
回归和资源失败证据均保留于上述三画像收据。ledger 新增24格到96/176；Kimi 的旧 `expert_ffn=3584`
标签改为 latent3584/expert3072/shared6144/dense33792，不修改已正确接线的产品数学。
统一实现为58,606行/232文件。剩余10画像、三交付和 P1 继续执行，本批不代签。

### GLM 六画像剩余片段与 D5-F30（AUDITED）

baseline `56506f28`；共享 consumer 归 `compute_ops` A2/A3，原始 header/config 保存在
`d5_glm_remaining_facts.json`。固定 Transformers `c1b80cf3` 的 GLM DSA/Flash 原源码
与已登记 llama.cpp `4c9233c` 的 GLM DSA、`de253435` 的 GLM Flash；重新取得的两个
Transformers 文件与原登记 SHA256 相同，不升级依赖。

本批按真实剩余角色一次接线：①四个 GGUF 的 Q8_0 dense FFN（H6144/4096、N12288）、
压缩 MLA（heads64、KV512、Q-rank2048/1536、V256）；②六画像 NextN 的独立双 norm、
Concat、EH projection、carry/输出 norm，官方 EH 为 BF16，GGUF 为 Q8_0；③DSA 的
32×128 indexer、top2048、GLM interleaved RoPE 与 Flash K4 pool/tail、实际 cache 反馈；
④Flash mHC 的真实 fn/scale/base 与最终 mean，复用已验 Sinkhorn 程序。不能用后投影
F32 core、普通 logits 或 KDA 收据代签上述完整角色。只有全部必要 region 通过才增加整格。

GLM 普通版 QK 为192+64=256、V256；旧 ledger 的单一 head_dim192不是完整 QK。
Flash 为256+0，无 RoPE，KV512。两者 dense FFN 都是12288，不能沿用 expert2048。
RMS epsilon按各原生模型路径明确记录，indexer LayerNorm 独立1e-6；GGUF 的 q/k/v/o
采用原版 absorbed MLA，不能为绕过准入改为展开全部 KV 的另一程序。

F30 有限产品反例：已有三个 ggml Provider 的 F29 准入写死96 query heads，拒绝 GLM
合法的 F32 QK576/V512、64/1 heads；原 native MatMul/Softmax/MatMul 没有该限制。
在这三个已 AUDITED owner 仅追加64 heads，保留原 shape、dtype、batch、sequence、
output 和资源检查；Apple 与其它路线产品不改。共享正/负准入和 Kimi 回归同批验证。

官方 FP8 的 partial block（GLM KV-A N576/scale rows5）、动态激活量化、grouped expert
仍为明确未验边界；保留 F27/F28 实际拒绝，不能以 GGUF 或 BF16 全链代签。原 ggml
converter 展开 FP8 的事实也不等于 HYLE native FP8 已支持。本批不新增数学、Provider、
量化器或 loader；若上述原生接缝需要超出有限准入修复，记录精确后续范围。

全部八入口批量接线后固定候选，重型验证串行、已有构建树复用；数值错误先查源公式、
物理布局和原生精度，不扩大容差。三交付/P1按 A5/A6 在对应完整路线稳定后统一执行。
96/176已验格直接保留；DeepSeek和本批未完成 region 不因本次审计标为 PASS。

**GLM 共享接线检查点：** 六组 consumer 已同时接入四路线 Direct/Graph：GGUF dense FFN、
NextN 输入边界、GGUF absorbed MLA、普通 DSA indexer、Flash K4 pool/tail、Flash 加权 mHC。
DSA 保留32×128、top2048，使用2052个以上有效 cache token；Flash 从513个完整 pool 中选512个，
另检查左 padding 和0/1/2/3个尾 token。key/gate cache 和 mHC stream 均反馈实际上轮输出；
索引检查比较完整成员集合、唯一性和-1 padding，不只检查索引落在范围内。

CPU Direct/Graph 的本批六组和 Kimi 六组回归各12/12通过（工作树检查点，尚非整画像裁决）。
首轮失败是 consumer 接缝：RoPE改为现有原生 rank4/axis1，K4 Softmax 显式转成原生连续行，
加权 head 求和使用现有 Multiply/Reduce，ReLU 使用原版 Clamp，整数索引直接发布并在宿主检查。
诊断输出中的 view 显式 Copy 后再独立绑定；这些诊断搬运不能冒充最终无额外成本执行。
未增加产品数学或放宽误差；F30仍只有64-head准入。其余三路线待同一候选串行验证。

NextN当前只验证双 norm/Concat/EH/输入 norm，不能代签完整R13。原 `glm-dsa.cpp` 明确按
embedding、hidden 顺序拼接，转换器不交换 EH 的列，已纠正最初测试的相反顺序。
固定 GLM Flash 上游还显式拒绝 NextN graph；需继续核对其它已登记模型程序及状态边界，
不能仅凭该拒绝判定所有成熟路径不可用。官方 FP8 与完整 MTP、A5/A6仍未验收，ledger保持96/176。

**D5-F31（AUDITED，baseline `5c738eaa`）：** CUDA 的六组 GLM 中仅 Flash pool/tail
因整数索引拼接落到已冻结 `hyle-cuda` 而失败，Direct/Graph其余11/12组通过。已审计原版
`ggml-cuda.cu` 的 CONCAT 支持同类型1/2/4/8字节元素，`concat.cu`按原始元素复制，现有 lowering
也不转浮点；HYLE `tensor_program.cc` 的 representable 提前只接纳浮点而挡住 I32。
当前 `compute_provider_ggml_cuda` owner 已 AUDITED，本 FIX 仅让原 Concat 接纳非空、连续、
plain I32输入/输出，保留原尺寸、字节、rank、axis和推导结果检查；不扩其它整数算术或 strided
I32，也不增加 kernel。验收为 provider 原生大整数/-1逐位重放、无效域拒绝、原 Concat回归及本批
CUDA Direct/Graph 的全部 pool/tail。Vulkan 同批失败先定位到测试读回把 I32当F32，只修 consumer。

F31 冷初始化进一步确认：实际构建的 `source-977e0a6dff4b1d66` 接纳原版 I32 CONCAT/CPY，
但明确拒绝 I32 DUP。原 lowering 在首段额外建立 DUP，使扩准入后 invocation 仍被拒绝；这不是
Concat数学缺失。有限 FIX 同时将 I32 首段复制改接原版 `ggml_cpy`，保持整数类型与逐位复制，
包括单输入 Concat；浮点原路径不变。不改 runtime pin、上游 kernel 或一般 Copy准入。

**GLM Vulkan 失败定向归因（baseline `f373efce`，仅 A2/A3 consumer）：** I32 读回修正后，
普通 DSA 已通过，剩 Flash pool/tail 两类失败。两次 permute→Copy 属于 D4 明确保留的
字节布局 adapter，`vulkan_tensor.cc` 为 Copy 固定 operation=0，不执行数学；本批不为它重开
产品 owner。Direct 检查只对同形状/同类型 F32、非 canonical 输入→canonical 输出的单个
layout dispatch 承认此边界，所有数学仍要求 ggml/IREE。搬运成本继续计入完整片段。

实际 runtime `source-8dc7950171d4c70b` 的 `ggml-vulkan.cpp:9577` 起明确在 coopmat2 下将
F32 MatMul 两个乘数转为 F16，F32 accumulation 不撤销该转换。官方 Flash 仅 score 输出失败，
query/key/pool/cache 及精确索引成员未失败。有限核验为该 score 增加独立 F16 乘数 oracle，
沿用既有 Vulkan 原生矩阵乘法双门：对完整 F32 oracle 为 abs2e-4/rel2e-3，对原生舍入 oracle
为 abs8e-5/rel4e-5；不修改归一化、pool/cache、参数或索引验收，不按观测误差拟合阈值。
若该源码机制不能解释实际结果，保留失败继续定位，不用更宽门代签。其它路线不改验收边界。

**本项结果：** `264e8633` 的 Vulkan Direct/Graph各6/6通过，双数值门和精确成员检查通过；
CPU同候选各6/6通过。CUDA `7b0fb4f1` 和Apple `5c738eaa` 的各6组按不变程序/输入/原参考公式
复用；CUDA另有 `f373efce` 单输入 I32 Concat检查。全部失败日志及每个候选身份保存于上述检查点。

### GLM 完整 NextN 共用批次（A2/A3，AUDITED）

baseline `264e8633`；owner `compute_ops` 的共享 consumer，产品 FIX 为零。逐 tensor 审计
保存为 `tests/compute_models/d5_glm_mtp_facts.json`：普通版 layer78、Flash layer45 均没有
mHC tensor；Flash官方该层实存17个BF16、871个FP8、872个F32 tensor，不能把全局 FP8标签
套到所有角色，也不能把其它版本的“BF16 NextN”说明套到本画像。

有界源码依据新增本机已登记 SGLang `06992c92ad4494fe9f9636e75be3e35040512447`：
`glm5_next_nextn.py` 复用 `deepseek_nextn.py` 的独立双norm→EH→Deepseek decoder→shared head norm；
Flash无RoPE，decoder使用普通 residual，不复用主干mHC。`index_topk_share.py`明确实现迭代间
索引carry，其对象与文件hash已记录。固定 llama.cpp 普通GLM的NextN程序也确认双residual和
最终norm；Flash那个入口尚未实现仅是单一入口限制，不裁为全部成熟路径缺失。

有限动作：①四个GGUF画像共用现有EH、MLA、router/routed/shared MoE与norm recipe组合成
完整NextN decoder，使用真实H/Q-rank/KV-rank/专家宽度和逐角色格式；②变化embedding与实际上轮
hidden/cache回送，独立oracle推进，检查首轮索引生成与后续carry/失效边界；③最终norm输出连接
已有R01全词表/head合同，不用一个截短词表代签；④官方两个画像保留相同消费者图和精确FP8缺口，
不得把GGUF/BF16结果代签为官方已验。主干DSA选出的索引与attention cache消费也在同一共享
组合批次接合，不能只靠两个互不相接的小测试签完整稀疏attention。

先完成所有八入口共享接线再固定候选并串行批验；复用已合格片段，不单独重跑P1/三交付或扩建
仪器。一轮具体失败纠正后按实际结果裁决；若缺原版能力，记录具名格式/设备/程序拒绝及替代路径
调查边界，不新增HYLE数学。完整画像只在其全部region和完整owner gate完成后增加。

**接线检查点（下文已验候选保留首次接线范围）：** 四个GGUF共用NextN decoder已连接EH→MLA→residual→norm→
routed/shared MoE→residual→head norm，四次变化embedding、实际上轮hidden/cache回送；独立
参考函数复用原有投影/MLA/MoE公式，不以实际输出为oracle。另一个共享consumer把真实DSA
top2048（Flash为512个pool展开并附0..3尾token）直接送入compressed cache的checked Gather和
64-head SDPA，索引选择不经宿主读回；压缩cache与indexer cache按相同策略实际反馈。
两个组已同时接入八个普通入口。MTP跨迭代索引carry和R01输出头组合仍未由这些组代签。
GGUF NextN逐tensor格式已追加到同一facts文件；主干的mHC不能误接进NextN。

首次CPU批验暴露两处consumer错误，原日志保留：①只含过去/当前token的单query SDPA错误声明
noncausal，超出既有成熟causal入口；改为右对齐causal后所有已选token仍可见，不扩产品准入。
②Flash Q8完整链中专家2/4的次序随上游量化误差交换，成员与最终输出均未失败；原测试却按位置
逐项比较ID及权重。只对诊断端口增加原生TopK/Gather按expert ID排列，同时独立参考以相同ID
对应权重；成员仍精确相等、原误差门不变，实际MoE保留原生次序。额外诊断排序/搬运计入测试
成本，不宣称它是最优执行。Graph的初始search budget失败先按同一不可用SDPA归因，不改搜索器。

**有限纠正结果：** `3fb5e818` 的CPU/CUDA/Apple/Vulkan Direct/Graph本批均10/10通过，
没有修改产品准入、搜索预算、数学或误差门；首次漏选的八画像GGUF MoE回归也纳入纠正批次。
原始失败与最终通过日志见上方新检查点。两组新consumer共16组通过仍不代签完整R13、画像、
三交付或P1；后续严格限定到本节既定carry/reset、完整head组合与具名官方FP8缺口。

## Closure card

- owner：`compute_ops` 持有验收矩阵和公共测试 consumer。D5 默认不重开已关闭路线；只有实际片段暴露
  产品缺陷时，才以失败收据把对应 Provider/route/Graph owner 重新定界到 `SCOPED -> AUDITED`。
- baseline：`deae2ed23fd3bf9ebc30364874d94881fc8f16c3`；D4 已关闭，Windows canonical 工作树干净且与
  `origin/main` 一致。
- trigger：[M3 主卡](llm-moe-m3-provider-demand-reconciliation.md)的 D1-D4 已完成需求正规化、成熟
  Provider 接线和错误自实现退场，但 D3 candidate 只证明代表格式/程序，不等于 22 个真实 profile 的
  模型片段已运行。D5 必须消除这种证据缺口。
- 正常 consumer：普通 C++ 用户提供 exact model/artifact profile，选择 CPU、CUDA、Apple 或 Vulkan
  路线，以 Direct `prepare` 或 Graph `compile` 准备真实 region；同一 artifact 重复 submit 时不重新
  选 Provider、解码权重或建立 workspace。
- 主纵切：本地 `Qwen/Qwen3.5-2B@15852e8` 官方 BF16，直接读取用户已有 safetensors 中的真实 tensor
  slice，覆盖 dense projection、full attention、Gated DeltaNet、MTP 和状态。
- 正交纵切：Qwen3.8-2.4T IQ4 MoE、Kimi K3 MXFP4/KDA、DeepSeek V4 FP4/FP8+DSA/mHC、
  GLM-5.3 FP8+DSA 及 GLM-5.3-Flash KDA；使用固定 header/config 的真实内宽并缩小 token、expert
  capacity，不宣称整模型推理。
- scope：八个模型、22 个固定 profile、15 个语义 region；四路线各自的 Direct/Graph、两次 submit、
  profile/region 冷绑定、数值/状态、资源/完成/失败、源码/静态/动态交付，以及适用 P1。
- non-goals：模型 loader 产品化、tokenizer/采样/推理 engine、整模型常驻、下载其它巨型 checkpoint、
  新增或优化 HYLE 数学/kernel/shader、逐算子重开旧实现对比、视觉 encoder。
- 依赖与 consumer：D3 的 ggml CPU/CUDA/Metal/Vulkan native storage candidate、oneDNN/cuBLASLt/
  FlashInfer/MLX/IREE 限定域、公共 `WorkloadContext`/region signature 与 Graph compile；D6 消费本卡收据。
- performance：正确性稳定后的不可变 candidate 执行一次适用 `P1`。计入准备、必要转换、上传、同步、
  workspace 和完成；只验证自然成熟路线没有 material 工程退化，不与已删除自研算子竞速。
- machines：Windows RTX 4070 是 CPU/CUDA/Vulkan 产品真相；MacBook Air `Mac16,12` 是 Apple formal
  机，Mac mini `Mac16,10` 只签 correctness/portability。重型路线串行、构建低并行。
- budget：一个共享 profile/shape manifest、一个本地 safetensors 有界提取器、每路线一组参数化 Direct
  和 Graph consumer、每路线一次稳定后三交付与一次适用 P1。不得按 15 region × 22 profile 手写测试。
- stop：[`llm-moe-d5-acceptance-ledger.json`](llm-moe-d5-acceptance-ledger.json)的 176 个
  profile/route/entry cell 全部获得 `RUN_PASS`、`BACKEND_REJECTED` 或 `UPSTREAM_MISSING`；未运行数为
  0。若精确缺口需要新数学或 Provider，记录最小后续卡并停止，不在 D5 扩张。

## 接线优先的批次执行规则

依照[Compute 算子批次合同](../../../architecture/compute-design.md#61-算子接入按有限批次接线和验证)，
D5 不再以单个 op/片段作为构建和完整验证的单位。先按下表固定模型/权重角色及已接入的成熟
Provider，批量写完该组四路线 Direct/Graph 的必要接线、夹具和 oracle；批末冻结不可变候选，
一次选择受影响门并复用同一构建树。前一批完整证据可按精确身份复用。API/算法稳定后才做路线
三交付和适用 P1；未运行格及门始终保留 `VALIDATION_NOT_RUN`，不能用编译目标或单片段代签。
后续接线优先扩充共享 profile/权重角色参数表与四路线已有入口，跨模型复用相同 Provider carrier；
不得为了每个新算子或画像复制一套 adapter、测试目标、完整 selector 和收据。批内只在共享接缝
改变或出现具体失败时做静态/最小定向检查，完成本组所有已审定接线或精确拒绝后才冻结候选，
串行运行适用 owner/consumer、三交付和性能门。同一候选上的合格收据按身份复用；部分片段
即使通过，整格仍维持 `VALIDATION_NOT_RUN`，直到其全部 region 与该格正确性门实际完成；
三交付和 P1 另按 D5-A5/A6 裁决。

当前 profile 批次是 Qwen3.8-2.4T 官方 BF16。其 R10/R11 接线子组覆盖 512-way/top-10
路由、动态 grouped expert、独立 sigmoid shared expert 与最终相加，在 CPU、CUDA、Apple、
Vulkan 的普通 Direct/Graph 入口接通；F19 的四路线 F32 score 定向结果不能替代整条链。
该子组的定向测试与完整 owner suite 是 partial receipt，不能触发 profile 级全量 selector、
三交付或 P1。继续批量接通同一 profile 的 R01-R05/R13-R15 后冻结候选，集中验证四路线的
变化输入重放、近并列成员、独立数值 oracle、成熟 Provider report、资源/完成/失败和交付。
若某路线真实缺少固定上游能力，记录精确拒绝并停止该路线，不用 HYLE 自研数学补洞。之后的
Kimi、DeepSeek、GLM 和量化格式按同样有限批次推进，不逐算子重建。

**下一接线批次：共用 GGUF 词表 carrier。** 将 Qwen3.5-2B、Qwen3.8-27B、
Qwen3.8-2.4T 的六个精确 GGUF profile 合为一个 R01 批次。共享参数化测试表记录 embedding 与
output head 各自的真实存储格式、tied/untied 角色、H=2048/5120/8192 和 V=248,320；
四路线 Direct/Graph 共用分片遍历、独立解码 oracle、全局边界检查，再由路线适配器提供原生
准备、上传和提交。先写完六个 profile 的接线，不按 profile 或单个 Gather/MatMul 反复运行完整
owner suite、selector、三交付或 P1。批内只做编译/极小定向烟测以发现错误接缝；候选稳定后
一次选择并运行适用验证。单个通过的检查点不改变 176 格 ledger 状态。若真实上游拒绝某格式，
记录 profile、权重角色、路线、原生错误及资源成本，再定界该格，不以另一格式代签。

**共用 GGUF 词表 carrier 批次结果（`76ebc05d5`）**：六个精确画像的 R01
embedding/head 已在 CPU、CUDA、Apple、Vulkan 各自 Direct/Graph 普通入口通过
96/96 个定向用例。真实逻辑宽度、逐角色格式、全部 logits、跨片索引和越界失败
已检查；夹具权重内容为按真实 GGUF 格式生成的合成分片，不声称下载了巨型权重。
这是[一份批次收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen-gguf-vocabulary-carrier/README.md)，
不触发每个算子/profile 的全量 selector、三交付或 P1；其它 region 未组合验收，
ledger 维持 0/176 `RUN_PASS`。

**六个 Qwen 画像整格批量复核（`3b326cd1` → `b2ef3eb0` → `57cd5a67`）**：
Qwen3.5-2B 和 Qwen3.8-27B 的官方及两种 GGUF 画像，在四路线 Direct/Graph
共 48 格重新通过全部所需 region；R01 保留 V=248,320、真实逐角色格式、全部
logits、跨片/越界失败。Windows 不可变候选的六个路线 factor selector 全部 PASS；
后续两项纯测试修正分别在原隔离构建树增量编译并完整复跑 CPU Direct/Graph owner。
MacBook Air 的 Apple Direct/Graph 各自取得 selector `PASS`，内部 104/104 与
77/77，无 SKIP。批量收据见
[六画像完整词表复核](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen-six-profile-full-vocabulary/README.md)。
当前 ledger 为 48/176 `RUN_PASS`、128/176 `VALIDATION_NOT_RUN`；D5-A5 三交付和
D5-A6 P1 尚未执行。Mac 额外 CPU Graph 检查的 ARM FP16 oracle 与把 BF16
画像转为 F32 后要求 oneDNN 融合的测试形状仍失败，不能写作 Mac 全套 selector PASS；
它不代替本卡的 Windows CPU 产品机 verdict，后续须单独修正与复核。

**当前混合权重 MoE 批次（`ea5cb859` → `ceef59d8` → `2e3d4c8d`）**：Qwen3.8-2.4T
官方及两个 GGUF 画像共用完整 R10/R11 recipe；四路线八入口的六个真实投影角色、原生量化
grouped、官方整链和量化整链共 32/32 项通过。CPU 唯一产品修改是既有 ggml F32 native-layout
MatMul 的准入条件；公共 schema 的误判已纠正并撤销新增实现，GPU 无产品修改。完整 CPU
owner 的唯一旧路由断言已迁移并定向复测，其余用例通过；不改写初跑失败记录。见
[合并收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-mixed-moe-batch/README.md)。
当前实现为 58,047 行 / 228 文件。接下来继续整批接通两个 GGUF 画像其它角色/region 的真实
dtype、量化 grouped 错误 ID 和完整资源负例门，批末再做完整 selector、三交付和 P1；本部分
收据不改变 48/176 ledger，不重复构建已通过的单个算子。

**当前 BF16 full MoE 批次结果（部分收据）**：四路线各自的完整 Graph owner 与 Direct owner
suite 均已通过 512-way/top-10、动态 grouped expert、shared expert 和 final add 的三次变化输入
重放及独立 oracle。Graph 的 CUDA 256-byte workspace 对齐缺陷已按 D5-F20 修复；Apple Direct
消费者的惰性 MLX reshape 必须在下一次 bind 前求值。见
[Graph 收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-full-router-graph/README.md)
与 [Direct 收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-full-router-direct/README.md)。
这只证明 R10/R11 的完整链，不能代签官方 BF16 整个 profile 的其它 region、三交付或 P1；
R01 资格纠正后 ledger 为 0/176 `RUN_PASS`。其余模型继续按有限批次接线，到批末再选受影响门。
不可变 `c2c1f7e8d` 的提前 selector 尝试选中 13 门；docs、closure、共享模型输入工具与
W1-WIN/MSVC 配置已运行通过。它在 CPU factor 首次编译共用 ggml CUDA 对象时人为停止，
余下 factor 未获 selector verdict；这次试运行不改变上述 partial receipt 或 ledger，
完整 selector 留到官方 BF16 profile 接线稳定后的批末执行。

**R05 causal conv 批内检查点（`55ff8c517`）**：官方 2.4T 的
`(2×16+128)×128=20,480` 通道、K4 causal Conv1d 已通过四路线 Direct/Graph
八条普通入口的两次变化输入和独立 oracle；见
[定向收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-causal-conv/README.md)。
这不触发逐算子的完整 selector/三交付/P1。此前 R01 的七行词表测试不能签官方
248,320 行词表；后续共用 BF16 词表批内检查点已接通真实宽度的 embedding/head
边界，见下文。2.4T 官方八格继续保持 `VALIDATION_NOT_RUN`，不把定向测试记作全格通过。

**R01 资格纠正（`ae15979a8` 后审计）**：此前 Qwen3.5-2B 与 Qwen3.8-27B
六个 profile 的 48 个已签格，均将真实 248,320 行词表缩成七行完成 Gather/输出头；
这不符合本卡仅缩小 batch、token、活跃 expert 和上下文容量的规则。
旧收据保留为 R01 的有限接线证据，其它已运行 region 以 partial receipt 保留；
[`ledger`](llm-moe-d5-acceptance-ledger.json) 将 48 格退回 `VALIDATION_NOT_RUN`，
当前完整格为 0/176。先按共用真实词表宽度批量接通四路线 Direct/Graph，批末再
统一运行完整 profile gate；历史收据中的 48/176 是纠正前的快照，不再代表当前验收。

R01 的 248,320 行允许按硬件资源无损分片执行：每片保留该 profile 的真实 hidden、dtype、
权重格式与原始行号，全部分片的 logits 必须按全局行序拼成完整词表并检查；Gather 必须验证
跨片全局索引和越界失败。分片程序可复用，但不得只抽样几行或把分片宽度当作模型词表宽度。
这属于图/消费者的资源规划，不新增 HYLE 数学或让 Provider 热路径重新选择。

**R01 共用 BF16 词表批内检查点（`c4e2b0d0`）**：Qwen3.8-27B 与 2.4T
官方 profile 的 248,320 行输出头在四路线 Direct/Graph 上无损分片运行，保留各自
5120/8192 hidden 与 BF16 类型；两次变化输入的全部 logits 按全局行序核对独立 oracle。
Gather 已验证跨片和尾片的具名全局索引。32 个定向用例全部通过，八个 owner suite
均无失败；六个依赖本地 Qwen3.5 夹具的用例在每个 suite 中 SKIP。详见
[共用词表部分收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-full-vocabulary-batch/README.md)。
这仍不是 R01 或完整 profile 的验收：本批尚未新增 248,320 全局越界拒绝用例，
量化画像的精确权重格式未在这份 BF16 收据内运行，完整 selector、三交付与适用 P1
也未签署。完整 profile 接线稳定后再集中验证，ledger 保持 0/176 `RUN_PASS`。

**R01 Qwen3.5-2B 共用词表批内扩展（`ae54c66c`）**：同一分片与 oracle
消费者已复用于本地官方 BF16 profile 的 H=2048、V=248,320；四路线 Direct/Graph
各自的输出头与全局 Gather 共 16 个定向用例通过，使用既有隔离构建树增量编译。
见[2B 词表部分收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-full-vocabulary-batch/README.md)。
本次没有再启动完整 owner suite、全路线 selector、三交付或 P1；这些门仍在
profile 接线稳定后的统一批末执行。全局越界和真实量化格式仍需同批补足，
ledger 保持 0/176 `RUN_PASS`。

**R01 BF16 全局越界批内检查点（`d2250624`）**：上述 2B、27B、2.4T
官方 profile 的词表消费者在四路线 Direct/Graph 共 24 个定向用例上拒绝
`-1` 和 `248,320` 全局索引；绕过消费者检查提交尾片本地索引 `2560`
时，成熟 Provider 的 checked Gather 返回越界，失败不取代已发布结果。
见[共用越界部分收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen-full-vocabulary-bounds/README.md)。
BF16 词表边界的该项缺口已有运行证据；其它量化权重格式、完整 profile
组合、全量 selector、三交付与适用 P1 仍待稳定批末验收，ledger 保持
0/176 `RUN_PASS`。

**语义角色与 grouped ID 批次（`9e7c57928`，测试断言修正 `32772681b`）**：
两个 2.4T GGUF 画像的九类真实投影、F32 attention、MTP、GDN primitive/conv，
以及官方和量化完整 MoE 已通过四路线八入口。完整 CPU/CUDA/Apple owner/consumer
均通过；Vulkan Graph 104/104，Direct 初跑 191/193，两条旧末尾 region 断言改为
检查受保护发布后定向 2/2 通过，复用其余通过项，不改写初跑 FAIL。Apple 最终明确
开启 WERROR 后重新构建并通过 109/109、82/82 和 Provider 7/7；先前 targeted
缓存 WERROR=OFF 的结果不代签此门。见[合并收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-semantic-and-checked-id-batch/README.md)。
本批实现 58,117 行 / 228 文件；ledger 仍为 48/176。继续共用夹具补官方 BF16
投影角色和连通的 conv、gate/decay、norm、GDN state 边界；官方与 GGUF 已转换的
头排列必须分别表达。三交付和 P1 仍未签。本次没有增加自研数学或重新选路。

## 批量验收形状

| 模型批次 | profile 数 | 真实宽度锚点 | 必验 region |
|---|---:|---|---|
| Qwen3.5-2B | 3 | H=2048、FFN=6144、GQA 8/2×256、GDN 16×128、conv K4 | official：R01-R05、R13-R15；两份 GGUF：R01-R05、R14-R15 |
| Qwen3.8-27B | 3 | H=5120、FFN=17408、GQA 24/4×256、GDN K16/V48×128 | R01-R05、R13-R15 |
| Qwen3.8-2.4T | 3 | H=8192、E=512/top10、expert N=2048、GQA 64/4×256 | R01-R05、R10-R11、R13-R15 |
| Kimi K3 | 3 | H=7168、E=896/top16、latent=3584、expert N=3072、shared N=6144、attention 96×128 | R01-R03、R06-R07、R10-R12、R14-R15 |
| DeepSeek V4 Pro | 1 | H=7168、E=384/top6、expert N=3072、index H64×D128/top1024、hc4 | R01-R03、R07-R15 |
| DeepSeek V4 Flash | 3 | H=4096、E=256/top6、expert N=2048、index H64×D128/top512、hc4 | R01-R03、R07-R15 |
| GLM-5.3 | 3 | H=6144、E=256/top8、expert N=2048、index H32×D128/top2048 | R01-R03、R07-R11、R13-R15 |
| GLM-5.3-Flash | 3 | H=4096、E=288/top8、expert N=2048、KDA H64×D128、hc4 | R01-R03、R06-R15 |

真实宽度不缩小；只缩小 batch、token、活跃 expert 和上下文容量。量化 profile 必须保留实际 codes/scales、
block/group 与逐 tensor role 的混合格式。官方与 GGUF 是不同 profile；某条路线拒绝官方 FP4/FP8 时，
不能用同模型 GGUF 的通过结果代签。

## 能力审计与有限动作

| ID | 当前事实 | 有限动作 | 完成证据 |
|---|---|---|---|
| D5-A1 | 22 个 profile 身份和真实 shape 已固定，但尚无统一运行 ledger | 生成共享 manifest 与 176-cell ledger，测试和 evidence 只引用稳定 ID | schema/self-check 通过；每格回指模型来源和 region |
| D5-A2 | 本地 Qwen 权重已校验，现有 provider test 仍多为合成小 shape | 有界读取选定 BF16/F32 tensor slice，生成外部 scratch fixture 与独立 oracle | 文件 digest、tensor 名/offset/dtype/shape 和结果收据 |
| D5-A3 | D3 已运行真实量化 block，但没有和 exact model profile/region 绑定 | 参数化现有成熟 Direct/Graph 程序，传入 exact `WorkloadContext` 并使用真实内宽 | Provider/report、两次 submit、结果/状态、准备期拒绝 |
| D5-A4 | 生命周期测试分散在路线测试中 | 每路线批量复用现有 resource/completion/failure fixture；只补 profile 纵向组合缺口 | 无热选择/重排、失败不发布、teardown 平衡 |
| D5-A5 | D3 三交付只签代表格式，未签 profile 片段 | 候选稳定后每路线一次 source/static/shared 外部 consumer | 搬迁后运行普通入口；DLL 与 EXE 同目录 |
| D5-A6 | 完整 profile 成本尚未验收 | 每路线选一个主 profile 和一个量化/MoE 反偏置做一次适用 P1 | 完整成本、A/A 与双顺序合格；精确限制入账 |

以上六项是 D5 全部动作。A2-A4 属于测试消费者和证据；若它们揭示产品缺陷，先停止并重开精确 owner，
不得在测试内实现 fallback 或新数学。

### D5-F1：CUDA BF16 dense projection 有界修复

本地 Qwen3.5-2B 的首个真实 R03 片段暴露了一个此前合成测试没有覆盖的产品缺陷：默认 CUDA 路线的
`cuBLASLt` Provider 只接纳 FP16/FP32，`ggml CUDA` 又按既定分工把普通 rank-2 dense MatMul 留给
`cuBLASLt`，所以合法的 BF16 `[1,2048] × [2048,2048]` 在 prepare 阶段被全部成熟 Provider 拒绝，
最终只显示末位 tensor Provider 的 unsupported 原因。

本缺陷把 `compute_provider_cublas` 从既有 CLOSED 经 SCOPED 精确重开为 AUDITED，固定的 FIX 只有：

1. 让现有 cuBLASLt adapter 接纳 plain rank-2 BF16 row/column-major 输入和 dense 输出，以
   `CUDA_R_16BF` 存储、FP32 accumulation 运行；不新增 HYLE kernel、fallback 或热路径搜索；
2. 在 Provider 测试中覆盖 BF16 行/列布局、两条独立 invocation 和变化输入；
3. 让真实 Qwen R03 的普通 CUDA Direct/Graph 入口绑定 `cublasLt`，完成两次 submit 和独立 oracle；
4. 按 selector 运行 `compute_provider_cublas`、`compute_route_cuda`、`graph_route_cuda` 及公共检查。

除 BF16 数据类型映射、相应合同文字和上述测试外，不改变 cuBLASLt 算法选择、workspace、路由优先级、
其它 Provider 或任何算子域。若 CUDA 13.3 heuristic 对该真实形状仍拒绝，则记录精确 backend rejection，
不在 D5 内扩大实现。

**D5-F1 CLOSED（`67948310b`）**：cuBLASLt 13.6.0 在 RTX 4070 上接受 BF16 row/column-major
布局并以 FP32 累加运行。Provider 6/6、CUDA Direct 10/10、CUDA Graph 29/29 以及 Compute/Graph
源码、共享、静态和搬迁消费者均通过；不可变候选上的真实 R03 Direct/Graph 两次 submit 收据见
[CUDA Qwen3.5 R03](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-r03/README.md)。
两格仍是只覆盖 R03 的 partial receipt，不提前宣称完整 profile 或 P1。

### D5-F2：Vulkan BF16 decode MatMul 边界修复

同一真实 R03 片段在默认 Vulkan 路线暴露了不同的成熟 Provider 边界缺陷。固定 ggml
`456172ec733a135778adcd32d00e576a58232e45-hyle` 的 Vulkan backend 会把 rank-2、1 行 BF16
MatMul 送入 `mul_mat_vec`；该分支的公开选择条件允许 BF16 activation，但实际 pipeline 入口只接受
F32、F16 或 Q8_1 activation，因而执行时触发 `b_type` 断言。现有 rank-3 小形状测试没有覆盖这个
decode 分支。这是 HYLE 适配器对固定上游能力域限定不足，不是缺少新的 MatMul 实现。

`compute_provider_ggml_vulkan` 从 D3 的 CANDIDATE 退回 AUDITED；固定 FIX 只有：

1. 在现有 ggml Vulkan region 构图时，rank-2 BF16 MatMul 的 1–4 行 activation 先用上游 `ggml_cast`
   扩为 F32，再交给同一个上游 BF16-weight/FP32-accumulation MatMul；权重、输出语义和 Provider 不变；
2. Direct 与 Graph 用完整 Qwen3.5 `[2048,2048]` checkpoint 权重、精确 R03 workload、两次变化输入
   验证普通入口；补 1–4 行资格回归以覆盖固定上游的整个 small-matvec 分支；
3. 只运行 `compute_provider_ggml_vulkan`、`compute_route_vulkan`、`graph_route_vulkan` 及 selector
   选中的公共检查；候选稳定后再签 Vulkan 三交付。

不修改 ggml 源码，不新增 HYLE shader、MatMul、热选择或 fallback，也不改变 F16/F32、rank-3/batched、
大于 4 行或量化 MatMul。若 F32 activation 边界仍被固定上游拒绝，则记录精确
`BACKEND_REJECTED`，不在 D5 内扩张数学实现。

**D5-F2 CANDIDATE（`958b9d69f`）**：固定 ggml Vulkan 0.24.0 在 RTX 4070 上通过 activation-only
上游 cast 承载 BF16 rank-2 decode MatMul。1–4 行边界回归、完整 `[2048,2048]` Qwen R03 的
Direct/Graph 两次 submit、ggml/IREE Provider 检查、Compute/Graph owner suite，以及源码、共享、静态和
搬迁外部 consumer 均通过；未增加 HYLE 数学、shader、fallback 或热路径选择。收据见
[Vulkan Qwen3.5 R03](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-r03/README.md)。
两格仍只覆盖 R03，不提前宣称完整 profile 或 P1；Provider 回到 D5 总验收所需的 CANDIDATE 状态。

### Apple Qwen3.5 R03 检查点

不可变 tests-only candidate `78561cec7` 在 formal Air `Mac16,12` 上直接由 MLX 0.32.2 Metal 路径
承载完整 `[2048,2048]` BF16 checkpoint projection；Direct/Graph 两次 submit、独立 oracle、两个
完整 owner suite，以及源码、共享、静态和搬迁外部 consumer 均通过。未暴露产品缺陷，也未增加 HYLE
数学、Metal kernel 或 fallback。收据见
[Apple Qwen3.5 R03](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-r03/README.md)。
两格仍只覆盖 R03，不提前宣称完整 profile 或 P1；`compute_ops` 继续持有 D5 AUDITED 总验收。

### D5-F3：Qwen3.8 Gated DeltaNet grouped-head Provider 接线

Qwen3.8-27B 官方 profile 的 R05 真实形状是 K16/V48×128。公共 `GatedDeltaNet` schema 已要求 V heads
为 K heads 的整数倍，但四个 ggml adapter 都额外要求 K/V heads 相等。固定 ggml
`456172ec733a135778adcd32d00e576a58232e45-hyle` 的 op 与 backend 将该比例作为正式能力：上游
`test_gated_delta_net` 有 `v_repeat` 维度，CUDA kernel 用独立 Q stride 和 V head dispatch，Metal/Vulkan
supports-op 也没有等头限制。Windows baseline 的精确 K16/V48×128、T=1 consumer 在 CPU prepare
阶段稳定返回 `ggml CPU operation domain is not admitted`；失败收据见
[Qwen3.8 R05 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-r05-admission-audit/README.md)。

因此 `compute_provider_ggml`、`compute_provider_ggml_cuda`、`compute_provider_ggml_metal` 与
`compute_provider_ggml_vulkan` 从各自 D3/D5 candidate 退回本卡 AUDITED。有限 FIX 只有：

1. 四个 adapter 在既有 F32 dense/depth/token/size 边界内接纳 `V_heads % K_heads == 0` 且 K/V depth
   相同的 Gated DeltaNet；继续调用同一个上游 `ggml_gated_delta_net`，不新增 HYLE math、kernel、shader、
   fallback 或热路径搜索；
2. 共享测试 fixture 固定 `Qwen/Qwen3.8-27B@1d4bf0f`、官方 safetensors index digest、R05、
   K16/V48×128 与 decode T=1，以非零数据和独立 double oracle 覆盖头复用、state output 和两次 submit；
3. 每条路线在自己的 Direct/Graph 普通入口验证 Provider identity、workload identity、数值、状态、失败不发布
   与资源回收；Apple 在 formal Air 上运行，Windows 运行 CPU/CUDA/Vulkan；
4. 复用 D5 的每路线 source/shared/static 外部 consumer 和最终 P1 批次，不为本修复另造交付或性能仪器。

不改变公共 schema、route 顺序、GDN 状态布局或 KDA gate 语义。若某个固定上游 backend 实际拒绝该精确
比例，则该路线记录 `BACKEND_REJECTED` 和原生拒绝，不在 D5 内实现替代算子。

**D5-F3 CANDIDATE（`d7cb7e01`）**：四个 adapter 只放宽到公共 schema 已允许且固定 ggml 原生实现
支持的 `V_heads % K_heads == 0`。同一份官方 Qwen3.8-27B R05 K16/V48×128、decode T=1 fixture 在
Windows CPU/CUDA/Vulkan 和 formal Air Apple M4 的普通 Direct/Graph 入口全部通过，两次 submit 的
输出与新 state 均匹配独立 double oracle；完整受影响 owner suite 也通过。未增加 HYLE 数学、kernel、
shader、fallback 或热选择。收据见 [Qwen3.8 R05 grouped-head](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-r05-grouped-head/README.md)。
该收据仍只覆盖 R05；其余 region、三交付与 P1 未完成，所以 176-cell ledger 不提前改变任何 cell。

### D5-F4：Qwen3.5 CPU BF16 RoPE 类型边界

本地 Qwen3.5-2B 官方 profile 的 R04 使用 BF16 activation、GQA 8/2×256 和 F32 accumulation。批量
Graph consumer 在普通 CPU 路线准备成功后，于固定 ggml CPU 的 `ggml_compute_forward_rope` 中触发
`fatal error`：HYLE adapter 的资格函数接受 BF16 RoPE，但该上游 CPU 入口只执行 F16/F32；同一资格
函数反而拒绝上游原生支持的 F32。源代码条件与实际失败一致，收据见
[Qwen3.5 CPU R04 RoPE admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r04-cpu-rope-admission-audit/README.md)。

因此 `compute_provider_ggml` 从 D5-F3 candidate 退回本卡 AUDITED。有限 FIX 只有：

1. CPU adapter 的 RoPE 资格域与固定上游对齐为 F16/F32，并继续接纳 Qwen 官方 BF16；BF16 在冷准备的
   ggml 图内使用上游 `ggml_cast` 转成 F32，执行同一上游 RoPE，再由既有输出边界转回 BF16；
2. Provider 回归覆盖 BF16/F32、两种 RoPE layout 和变化输入；真实 Qwen3.5 R04 的普通 Graph 入口
   覆盖精确 8/2×256、两次 submit、Provider/workload identity 和独立 decode oracle；
3. 只运行 `compute_provider_ggml`、`graph_route_cpu` 及 selector 选择的公共检查；三交付和 P1 继续并入
   D5 路线批次，不为本缺陷新增仪器。

不修改公共 RoPE schema、ggml 源码、路由优先级或其它路线，也不新增 HYLE 数学、fallback 或热选择。

**D5-F4 CANDIDATE（`f340b0d4`）**：CPU adapter 以固定 ggml 的既有 Cast/RoPE 图执行 BF16，F16/F32
保持原生类型域；BF16/F32 Provider 回归、Qwen3.5 精确 GQA 8/2×256 Graph、ggml CPU 34/34、CPU
Direct 12/12、CPU Graph 34/34 与普通 cpu-executor 全部通过。未修改公共 schema、上游源码或路由顺序。
失败与修复边界收据见 [R04 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r04-cpu-rope-admission-audit/README.md)。

### D5-F5：Qwen3.5 CPU Direct BF16 SwiGLU 成熟接线

官方 Qwen3.5-2B R02 的普通 CPU Direct 路径按 BF16 存储边界逐算子准备：RmsNorm 与三次 MatMul
均绑定固定 ggml CPU，独立 `GatedActivation(SiLU, separate)` 却落到已冻结的 `hyle-native`。固定 ggml
0.24.0 已提供 `ggml_swiglu` 与 `ggml_swiglu_split`；其 CPU 实现原生执行 F16/F32，BF16 可在同一上游
图内 Cast 到 F32 并由既有输出边界 Cast 回 BF16。源代码与普通入口复现见
[Qwen3.5 CPU Direct R02 gate admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r02-cpu-direct-gate-admission-audit/README.md)。

因此 `compute_provider_ggml` 从 D5-F4 candidate 退回本卡 AUDITED。有限 FIX 只有：

1. CPU adapter 接纳公共 `GatedActivation` 的 SiLU packed/separate 两种形状，lower 到固定上游
   `ggml_swiglu`/`ggml_swiglu_split`；F16/F32 保持原生，BF16 通过上游 Cast 执行；
2. Provider 回归覆盖 F16/BF16/F32、packed/separate、变化输入与非 SiLU 拒绝；真实 Qwen3.5 R02
   使用真实 2048/6144 权重、两次 Direct replay、独立 BF16 边界 oracle 和成熟 Provider identity；
3. 只运行 `compute_provider_ggml`、`compute_route_cpu` 及 selector 选择的公共检查；三交付和 P1 继续并入
   D5 路线批次。

不修改公共 schema、ggml 源码、路由优先级、其它 activation 或其它路线；不新增 HYLE 数学、fallback、
热选择或性能竞速。若固定上游不能正确执行该精确域，则记录 `BACKEND_REJECTED`，不恢复 native 作为
D5 成熟覆盖证据。

**D5-F5 CANDIDATE（`3798fd11`）**：CPU adapter 已用固定 ggml 的 `ggml_swiglu`/
`ggml_swiglu_split` 接管 F16/BF16/F32 的 packed/separate SiLU；BF16 只在同一上游图内 Cast 到 F32，
再由既有输出边界转回。Provider 35/35、CPU Direct 19/19、CPU Graph 两个测试目标和 fixture 输入工具
13/13 均通过；未修改公共 schema、上游源码、路由顺序或其它 activation。

### CPU Qwen3.5 official Graph 完整格

不可变 candidate `f340b0d4` 已用本地 checkpoint 的 embedding slice、layer-0 norm/FFN/conv/GDN/out-proj、
layer-3 Q/K norm 与 MTP 权重，完成 R01-R05、R13-R15 的普通 CPU Graph 两次 replay、独立 oracle、
成熟 Provider identity、资源与失败检查；既有 R03 收据并入同一格。完整收据见
[CPU Qwen3.5 official Graph](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-official-graph/README.md)。
ledger 首次得到 1 个 `RUN_PASS`；CPU Direct、其它路线、三交付和 P1 仍未由该格代签。

### CPU Qwen3.5 official Direct 完整格

不可变 candidate `3798fd1129323c53c373dbdf7bd628b1a1f6365d` 复用同一 checkpoint fixture，
通过普通 `compute::cpu` 上下文逐算子冷准备并执行 R01-R05、R13-R15。R02 保留真实 BF16 边界并由
ggml SwiGLU 与 MatMul 承载；每个 prepared call 两次变化输入 replay，R01 另验证越界失败不发布，
既有 R03 收据并入完整格。完整收据见
[CPU Qwen3.5 official Direct](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-official-direct/README.md)。
ledger 现有 2 个 `RUN_PASS`；其它路线、三交付和 P1 未由这两格代签。

### D5-F6：Qwen3.5 CUDA BF16 SwiGLU 成熟接线

官方 Qwen3.5-2B R02 的普通 CUDA Direct 路径对精确 BF16 `[1,6144]` separate
`GatedActivation(SiLU)` 冷准备成功，但绑定的是已经冻结的 `hyle-cuda` fallback。固定 ggml
0.24.0 CUDA backend 已提供 `ggml_swiglu`/`ggml_swiglu_split`，原生执行 F16/F32；其 adapter 当前既不
资格化也不 lower 公共 `GatedActivation`。普通入口复现和固定上游源码边界见
[Qwen3.5 CUDA R02 gate admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r02-cuda-gate-admission-audit/README.md)。

因此 `compute_provider_ggml_cuda` 从 D5-F3 candidate 退回本卡 AUDITED。有限 FIX 只有：

1. CUDA adapter 接纳公共 `GatedActivation` 的 SiLU packed/separate 两种形状，lower 到固定上游
   `ggml_swiglu`/`ggml_swiglu_split`；F16/F32 保持原生，BF16 在同一 ggml CUDA 图内 Cast 到 F32，
   再由现有输出边界 Cast 回 BF16；
2. Provider 回归覆盖 F16/BF16/F32、packed/separate、变化输入与非 SiLU 拒绝；真实 Qwen3.5 R02
   使用精确 2048/6144 权重、两次 replay、独立 BF16 边界 oracle 和成熟 Provider identity；
3. 同一批完成 Qwen3.5 official CUDA Direct/Graph 的 R01-R05、R13-R15 普通入口验收；只运行
   `compute_provider_ggml_cuda`、`compute_route_cuda`、`graph_route_cuda` 及 selector 公共检查，三交付和
   P1 继续并入 D5 路线批次。

不修改公共 schema、ggml 源码、路由优先级、其它 activation 或其它路线；不新增 HYLE 数学、kernel、
fallback、热选择或性能竞速。若固定上游拒绝精确类型/形状，则记录 `BACKEND_REJECTED`，不恢复 native
作为 D5 成熟覆盖证据。

### D5-F7：Qwen3.5 CUDA BF16 RoPE 类型边界

F6 接线后的完整 CUDA Direct 执行批次中，R02 FFN 已通过真实权重与独立 oracle；R04 的精确 BF16
GQA 8/2×256 RoPE 则在普通入口 prepare 成功后，于 invocation 初始化/执行固定 ggml CUDA 的
`rope.cu` 触发 `src0->type == F32 || F16` 断言。adapter 的资格函数接受 BF16，但 lowering 将 BF16
原样交给只执行 F16/F32 的上游 CUDA RoPE；源码条件与真机失败一致。收据见
[Qwen3.5 CUDA R04 RoPE admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-r04-cuda-rope-admission-audit/README.md)。

`compute_provider_ggml_cuda` 已因 F6 处于 AUDITED；本次在任何 RoPE 产品修改前追加的有限 FIX 只有：

1. adapter 的 RoPE 资格域与固定上游对齐为 F16/F32，并继续接纳官方 BF16；BF16 在同一 ggml CUDA
   图内使用上游 `ggml_cast` 转成 F32，执行同一上游 RoPE，再由现有输出边界转回 BF16；
2. Provider 回归覆盖 F16/BF16/F32、两种 RoPE layout、变化输入与不支持类型拒绝；完整 Qwen3.5
   CUDA Direct/Graph 批次继续验证精确 8/2×256、两次 replay、成熟 Provider/workload identity 和独立
   decode oracle；
3. 验证集合保持 F6 的 `compute_provider_ggml_cuda`、`compute_route_cuda`、`graph_route_cuda`、selector
   公共检查与 D5 路线三交付/P1，不另造仪器或测试矩阵。

不修改公共 RoPE schema、ggml 源码、路由优先级或其它路线；不新增 HYLE 数学、kernel、fallback 或热
选择。固定上游若拒绝 Cast 后的精确图，则记录 `BACKEND_REJECTED`，不绕回 native。

**D5-F6/F7 CANDIDATE（`9917438b`）**：固定 ggml CUDA adapter 已用上游
`ggml_swiglu`/`ggml_swiglu_split` 接管 F16/BF16/F32 packed/separate SiLU，并以同一 ggml CUDA 图内
Cast 衔接 BF16 RoPE。Provider 24/24、CUDA Direct 18/18、CUDA Graph 35/35 在静态与共享树通过；
共享树启用 WERROR，DLL 与 EXE 同目录。不可变 SM89 candidate 的 selector 又完成 configure、四个选中
factor 并得到 PASS；没有新增 HYLE 数学、kernel、fallback 或执行期选择。

### D5-F8：GGUF 量化 embedding 原生行读取接线

Qwen3.5 `UD-Q4_K_XL` 的精确 embedding 权重为 Q6_K；目标 GGUF 画像还要求 Q4_K 与 Q8_0。
Windows baseline `0c7293ff38ac078c095e623062398087a5a89ba1` 上，普通 CPU Direct 以真实隐藏宽度
`[7,2048]` 准备 Q6_K table 时被 adapter 的通用 GGUF 端口门拒绝，最终落到 native provider 的
`plain scalar tensor storage` 拒绝。固定 ggml `b9d43bc809551dde` 的 CPU、CUDA、Metal 和 Vulkan
源码均已有 Q4_K/Q6_K/Q8_0 GET_ROWS；这是 adapter 接线缺陷，不是上游算子或 HYLE 数学缺口。收据见
[GGUF embedding admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-gguf-embedding-admission-audit/README.md)。

因此 `compute_provider_ggml`、`compute_provider_ggml_cuda`、`compute_provider_ggml_metal` 和
`compute_provider_ggml_vulkan` 从各自 candidate 精确退回本卡 AUDITED。有限 FIX 只有：

Apple 的普通 Direct 与 Graph 还各有一层 GGUF carrier/region 准入，因此本路线实施前另将
`compute_route_apple` 与 `graph_route_apple` 从旧 CLOSED 经 SCOPED 精确重开为 AUDITED；它们只负责
把同一 checked embedding 域交给 `compute_provider_ggml_metal`，不改变其它 Apple 运算或 MLX region。

1. 四个 adapter 只接纳 rank-2 Q4_K/Q6_K/Q8_0 embedding table、axis `0/-2`、I32 indices、dense
   F32 output，调用固定上游 `ggml_get_rows`；不写 HYLE 解量化器、kernel 或 shader；
2. 保留公共 checked Gather 语义：越界在结果发布前失败且输出不变；prepared program 以变化 indices
   重放两次，不重新选路、解码权重或分配持久 workspace；
3. 以 Qwen3.5 `UD-Q4_K_XL` 的 Q6_K `[7,2048]` 纵切在四路线普通 Direct/Graph 实际运行；后续
   Q4_K/Q8_0 profile 复用同一参数化 fixture，不手写路线乘画像测试；
4. 只运行四个 provider owner suite、对应 Direct/Graph factor 与 D5 最终三交付/P1。其它格式、轴、
   路由顺序和自研数学继续冻结；固定上游若在具体设备拒绝精确域，按合同记录
   `BACKEND_REJECTED`。

**D5-F8 CPU CANDIDATE（`42f0320f`）**：CPU adapter 已在固定 ggml 的
`ggml_get_rows` 上接纳 Q4_K/Q6_K/Q8_0 精确 embedding 域。Qwen3.5 primary GGUF 的 Q6_K
`[7,2048]` R01 在普通 Direct/Graph 中完成 profile 冷绑定、两次变化 indices replay、独立 packed-byte
oracle、checked 失败不发布和 Graph 零 replay metadata 分配。i9-13900K/RTX 4070、MSVC 19.50、
WERROR 的 selector 对 docs、closure、configure、Provider、CPU Direct 与 CPU Graph 全部 PASS；收据见
[CPU Qwen3.5 primary GGUF R01](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-primary-gguf-r01/README.md)。

**D5-F8 CUDA CANDIDATE（`889bb81a`）**：CUDA adapter 复用同一固定上游
`ggml_get_rows`，接纳完全相同的 Q4_K/Q6_K/Q8_0 embedding 域；checked indices 在任何 native graph
写输出前验证，失败不发布，invocation 的验证存储在准备时一次分配。Qwen3.5 primary GGUF 的 Q6_K
`[7,2048]` R01 已在 RTX 4070 上通过普通 Direct/Graph 的两次变化 indices replay、独立 packed-byte
oracle、checked 失败输出保持和 Graph 零 replay metadata 分配。CUDA 13.3.73、driver 610.62、SM89、
MSVC 19.50、WERROR 的不可变 selector 对 docs、closure、configure、Provider、CUDA Direct、
ggml CUDA Graph provider 与 CUDA Graph 全部 PASS；收据见
[CUDA Qwen3.5 primary GGUF R01](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-primary-gguf-r01/README.md)。

**D5-F8 Apple CANDIDATE（`94264ba2`）**：Apple Direct 的 GGUF carrier 与 Graph region 已把同一
checked embedding 域交给固定 ggml Metal `ggml_get_rows`。Qwen3.5 primary GGUF 的 Q6_K
`[7,2048]` R01 在 formal Air M4 上完成普通 Direct/Graph 冷绑定、两次变化 indices replay、独立
packed-byte oracle、越界提交前失败、输出保持及后续恢复；Provider owner suite 同时运行 Q4_K/Q6_K/
Q8_0。固定上游 Metal Q4_K 的独立解码资格界为实测 `0.4% + 2e-4`，Q6_K/Q8_0 保持 `1e-6`。
AppleClang 21、Release/WERROR selector 的 docs、closure、configure、Provider、Direct 与 Graph 全部
PASS；收据见 [Apple Qwen3.5 primary GGUF R01](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-primary-gguf-r01/README.md)。

#### D5-F8 Vulkan route reopen audit

Windows baseline `65138ac77` 的固定路径仍不能把量化 embedding 交给成熟实现：
`compute_provider_ggml_vulkan::region_supports` 只接纳 plain I32 unchecked Gather，导致 GGUF table 不能
成为 managed region；随后 `prepare_plan` 会在 native `lower_gather` 前以“compressed storage requires a
matching mature region”拒绝。既有 `lower_gather` 的 GPU bounds check、safe indices 和 checked publication
适用于公共失败语义，但它后续按逻辑 `data_type_size` 把 plain tensor 重排成 raw I32 rows，不能解释 GGUF
block storage。固定 ggml Vulkan 已有目标 Q4_K/Q6_K/Q8_0 GET_ROWS，因此这是精确的 route/provider
adapter seam，不是算子或 shader 缺口。

本事实满足 `compute_route_vulkan` 的 reopen trigger；该 owner 从旧 D4 CLOSED 只为 D5-F8 量化
embedding 边界进入 SCOPED。`graph_route_vulkan` 不需要新的产品 lowering，它继续复用同一 Compute plan，
只在候选上完成 Graph 普通入口验收。

有限 FIX 固定为：

1. `compute_provider_ggml_vulkan` 只新增 exact rank-2 Q4_K/Q6_K/Q8_0 F32 embedding table、axis
   `0/-2`、dense rank-1 I32 indices 和 dense F32 output 的 unchecked native region 资格，仍只 lower 到
   固定上游 `ggml_get_rows`；unchecked 是 provider 内部边界，不成为公共 API；
2. `compute_route_vulkan::lower_gather` 只对该 exact GGUF 域复用既有 GPU `lower_index_check`，把 safe
   indices 和原量化 table 交给一个 provider region，再用既有 checked publication 决定是否发布 F32
   output；不做 raw-word 重排、host readback、新 shader、新数学或 fallback；
3. Provider owner suite 运行 Q4_K/Q6_K/Q8_0、两次 replay 和错误域拒绝；Qwen3.5 primary GGUF Q6_K
   `[7,2048]` 在普通 Vulkan Direct/Graph 运行两次变化 indices、独立 packed-byte oracle、越界输出保持
   与恢复，并确认一个 managed region；
4. 只运行 `compute_provider_ggml_vulkan`、`compute_route_vulkan`、`graph_route_vulkan` 与 selector 选中的
   公共检查。本 partial R01 不启动 D5-A5 三交付或 D5-A6 P1，也不改变其它 Gather、格式、轴或路线。

若固定上游在 RTX 4070 上拒绝任一精确格式，记录具名 `BACKEND_REJECTED`；不得回接冻结的自研
Vulkan shader 或把量化 bytes 当普通标量 storage。

实施前 ledger 仍为 8 个 `RUN_PASS`、168 个 `VALIDATION_NOT_RUN`；CPU、CUDA 与 Apple 共六个
primary GGUF cell 只新增 R01 partial receipt，其余 region 未运行，因此不提前把完整 cell 写成 PASS。

**D5-F8 Vulkan CANDIDATE（`8af3a29d`）**：Provider 只为 exact Q4_K/Q6_K/Q8_0 embedding
开放内部 unchecked `ggml_get_rows` region；普通路线先运行既有 GPU bounds check，把 safe indices 与原始
量化 table 交给固定上游，再由既有 checked publication 决定是否发布结果。Qwen3.5 primary GGUF 的
Q6_K `[7,2048]` R01 已在 RTX 4070 上通过普通 Direct/Graph 的两次变化 indices replay、独立
packed-byte oracle、越界不发布、保留既有结果及 fresh invocation 恢复；owner suite 同时运行三个目标
格式。Vulkan SDK 1.4.357.0、driver 610.62、MSVC 19.50、Release/WERROR selector 的 docs、closure、
configure、ggml/IREE Vulkan Provider、Direct 与 Graph 全部 PASS；收据见
[Vulkan Qwen3.5 primary GGUF R01](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-primary-gguf-r01/README.md)。
ledger 计数仍为 8 个 `RUN_PASS`、168 个 `VALIDATION_NOT_RUN`，但四路线共八个 primary GGUF cell
现均有 R01 partial receipt；完整 cell、三交付与 P1 仍未代签。

### D5-F9：Qwen3.5 primary GGUF 混合投影批量验收

权重审计固定的 `UD-Q4_K_XL` 不是单一 Q4 文件：FFN、full attention 与 Gated DeltaNet projection
实际混合 Q4_K、Q5_K、Q6_K、Q8_0 和 IQ4_XS。D5-F9 不新增产品实现，而是用同一参数化 consumer
批量验证每条路线既有 ggml adapter 是否真正承载这些精确 tensor role、内宽和物理格式；只有实际失败
才按本卡规则重新打开对应 owner。

**D5-F9 CPU partial（`6834a6c9`）**：普通 CPU Direct/Graph 已运行五个精确代表 tensor：
`blk.0.ffn_gate.weight` Q4_K `[2048,6144]`、`blk.0.ffn_down.weight` Q6_K
`[6144,2048]`、`blk.3.attn_k.weight` Q5_K `[2048,512]`、`blk.8.ffn_gate.weight`
IQ4_XS `[2048,6144]` 和 `blk.0.ssm_out.weight` Q8_0 `[2048,2048]`。两条入口均冷绑定
固定 ggml CPU，以两个变化 activation 重放并匹配独立 packed-weight decode oracle；Graph 每项只有一个
成熟 region。不可变 MSVC 19.50/WERROR selector 的 docs、closure、fixture 工具、Direct 与 Graph
全部 PASS，未暴露产品缺陷，也没有新增 HYLE 数学。收据见
[CPU primary GGUF projections](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-primary-gguf-projections/README.md)。
本 partial 只签 R03/R14；R02、R04、R05、R15 和完整 cell 仍未代签，ledger 总计数不变。

**D5-F9 CUDA partial（`9dc37c98`）**：同一参数化 consumer 已在 RTX 4070/SM89 的普通 CUDA
Direct/Graph 上运行全部五个精确 tensor，均冷绑定固定 `ggml-cuda`；设备 activation 变化两次，结果匹配
独立 packed-weight decode oracle，Graph 每项只有一个成熟 region。CUDA 13.3.73、MSVC 19.50、
Release/WERROR 的不可变 selector 通过单架构 configure、Direct 与 Graph，docs/closure 从相同合同复用；
没有新增 CUDA kernel 或 fallback。收据见
[CUDA primary GGUF projections](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-primary-gguf-projections/README.md)。
本 partial 同样只签 R03/R14，完整 cell 和 ledger 总计数不变。

#### D5-F9 artifact region applicability correction

固定 revision `f6d5376` 的 Qwen3.5 `UD-Q4_K_XL` 与 `UD-Q8_K_XL` 均各有 320 个 tensor；
完整 header 名单的 MTP/NextN 匹配数都是 0，两个名单完全相同，且仓库树中没有 MTP sidecar。
因此模型家族的官方 BF16 checkpoint 继续要求 R13，但这两个 GGUF artifact 不要求 R13。当前 ledger
已按具体 artifact identity 删除这两份 profile 的 R13；不能凭官方 checkpoint 的能力为 GGUF 虚造权重，
也不能用官方 R13 收据代签 GGUF。审计收据见
[Qwen3.5 GGUF region applicability](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen35-gguf-region-applicability-audit/README.md)。

**D5-F9 Vulkan partial（`e31a6e7d`）**：同一组五个精确 tensor 已在 RTX 4070 的普通 Vulkan
Direct/Graph 上运行，均冷绑定固定 `ggml-vulkan`；两次变化的 device activation 匹配独立
packed-weight decode oracle，Direct 每项观察到一个成熟 region，Graph 的单 MatMul 通过成熟 managed
region 编译。Vulkan SDK 1.4.357.0、driver 610.62、MSVC 19.50、Release/WERROR 的不可变 selector
通过 docs、closure、configure、Direct 与 Graph；没有新增 HYLE Vulkan shader 或 fallback。收据见
[Vulkan primary GGUF projections](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-primary-gguf-projections/README.md)。
本 partial 同样只签 R03/R14，完整 cell 和 ledger 总计数不变。

**D5-F9 Apple partial（`ef9c0562`）**：同一组五个精确 tensor 已在正式 MacBook Air M4 的普通
Apple Direct/Graph 上运行，固定 `ggml Metal` 并实际加载 Q4_K、Q5_K、Q6_K、IQ4_XS、Q8_0
`mul_mv` pipeline；两次变化的 shared-buffer activation 匹配独立 packed-weight decode oracle。
AppleClang 21、macOS SDK 27.0、Release/WERROR 的不可变 selector 通过 docs、closure、configure、
Direct 与 Graph。首次完整 owner suite 使用了旧 R03 partial 夹具，4 个既有测试因缺文件拒绝；从 Windows
canonical checkpoint 重建并同步当前 12 张量夹具后，Direct/Graph 均 2/2 PASS。收据见
[Apple primary GGUF projections](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-primary-gguf-projections/README.md)。
本 partial 同样只签 R03/R14，完整 cell 和 ledger 总计数不变。

### D5-F10：Qwen3.5 primary GGUF 完整路线格

**D5-F10 CPU complete（`b2e6a27f`）**：在既有 R01、R03/R14 精确收据上，普通 CPU
Direct/Graph 已补齐该 artifact 的全部剩余适用 region。R02 运行完整
`2048 -> 6144 -> 2048` RMSNorm/SwiGLU FFN，gate/up 保留原生 Q4_K、down 保留原生 Q6_K；
R04 运行 F32 norm、half-split RoPE 和精确 `8 query heads / 2 KV heads x 256` GQA 语义；R05
运行 K4 causal conv 与 `16 x 128` Gated DeltaNet 状态更新。每条链以两个变化 activation/state
重放，逐中间结果匹配独立 oracle；R02 Direct/Graph 在准备完成后的重放区间均为零 metadata
allocation/reallocation。所有调用和 Graph compile report 只固定到 ggml CPU 或 oneDNN，没有新增
HYLE 数学或 fallback。

固定 GGUF artifact 不含 R13；这不是跳过适用能力，具体 header/repository 证据仍由 D5-F9 的 artifact
applicability correction 约束。不可变 MSVC 19.50 Release/WERROR selector 通过 docs、closure、fixture
工具、configure、CPU Direct 与 CPU Graph，127 个无关 gate 跳过或复用；P1 未启动。完整收据见
[CPU primary GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-primary-gguf-complete/README.md)。
ledger 现有 10 个 `RUN_PASS`、166 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍随稳定 CPU
路线批次执行，不由本 tests-only candidate 代签。

**D5-F10 CUDA complete（`ae2b789b`）**：普通 CUDA Direct/Graph 在既有 R01、R03/R14 收据上
补齐相同 artifact 的 R02、R04、R05 与 R15。R02 保留 Q4_K gate/up 和 Q6_K down；R04 按真实
模型合同使用 BF16 activation/storage、F32 accumulation，而不是无适用成熟 attention Provider 的
F32-only 测试形状；R05 覆盖 K4 causal conv 和 `16 x 128` Gated DeltaNet 状态更新。所有准备调用与
Graph step 均固定到 ggml CUDA、cuBLAS、SGLang/CCCL native 或 FlashInfer 成熟路径，没有新增 HYLE
数学或 fallback。固定 ggml CUDA 允许在首次 submit 建立 CUDA Graph metadata，第二次变化输入重放不再
发生 metadata allocation/reallocation。

不可变 Windows RTX 4070/SM89 candidate 以 `CUDAARCHS=89-real` 通过 MSVC 19.50 Release/WERROR
selector 的 docs、closure、configure、CUDA Direct 与 CUDA Graph；127 个无关 gate 跳过或复用。
完整收据见 [CUDA primary GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-primary-gguf-complete/README.md)。
ledger 现有 12 个 `RUN_PASS`、164 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍随稳定
CUDA 路线批次执行，不由本 tests-only candidate 代签。

**D5-F10 Apple complete（`690d5f2c`）**：正式 MacBook Air M4 上的普通 Apple Direct/Graph
在既有 R01、R03/R14 收据上补齐相同 artifact 的 R02、R04、R05 与 R15。R02 运行完整
`2048 -> 6144 -> 2048` F32 RMSNorm/SwiGLU 链，并保留 Q4_K gate/up 与 Q6_K down；R04 以
primary GGUF 模型身份运行 BF16 norm、half-split RoPE 和精确 `8/2 x 256` GQA；R05 以同一
模型身份运行 K4 causal conv 与 `16 x 128` Gated DeltaNet 状态更新。量化路径实际加载 ggml
Metal Q/K/IQ pipeline，其余计算固定到 MLX 0.32.2 或完整 ggml Metal region；没有新增 HYLE
数学、fallback 或热路径选路。

不可变 AppleClang 21、macOS SDK 27.0、Metal 17F109、Release/WERROR selector 通过 docs、
closure、configure、Apple Direct 与 Apple Graph；127 个无关 gate 跳过或复用。完整收据见
[Apple primary GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-primary-gguf-complete/README.md)。
ledger 现有 14 个 `RUN_PASS`、162 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍随稳定
Apple 路线批次执行，不由本 tests-only candidate 代签。

**D5-F10 Vulkan complete（`e3b2a5fa`）**：Windows RTX 4070 上的普通 Vulkan Direct/Graph
在既有 R01、R03/R14 收据上补齐相同 artifact 的 R02、R04、R05 与 R15。R02 运行完整
`2048 -> 6144 -> 2048` F32 RMSNorm/SwiGLU 链，保留 Q4_K gate/up 与 Q6_K down；Graph 将完整
逻辑链一次编译为一个成熟 managed region。R04/R05 以 exact primary-GGUF 身份复用已验的 BF16
`8/2 x 256` GQA、K4 causal conv 与 `16 x 128` Gated DeltaNet 语义。Direct 的每项均固定到
ggml Vulkan managed region 或限定 IREE dispatch；没有新增 HYLE shader、数学或 fallback。

不可变 MSVC 19.50、Vulkan SDK 1.4.357.0、driver 610.62、Release/WERROR selector 通过 docs、
closure、configure、Vulkan Direct 与 Vulkan Graph；127 个无关 gate 跳过或复用。完整收据见
[Vulkan primary GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-primary-gguf-complete/README.md)。
ledger 现有 16 个 `RUN_PASS`、160 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍随稳定
Vulkan 路线批次执行，不由本 tests-only candidate 代签。

### D5-F11：Qwen3.5 fidelity GGUF 完整路线格

**D5-F11 CPU complete（`74101b6e`）**：普通 CPU Direct/Graph 已完整运行固定
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` profile 的全部适用 region。测试没有从 artifact 名称推断全局
Q8：R01 使用真实 F16 embedding；R02 使用 Q8_0 gate/up/down；R03/R14 分别运行 Q8_0 FFN/V/O
投影和 F16 GDN-output/K/Q 投影；R04/R05 保留真实 `8/2 x 256` GQA、K4 causal conv 与
`16 x 128` Gated DeltaNet 语义，state/conv/norm 为 F32。每个 prepared chain 或 Graph program
以变化输入重放，匹配独立 Q8_0 packed-weight、F16 dense 或 F32 fragment oracle；无效 embedding
索引失败不覆盖已发布结果。所有调用与 Graph compile report 只固定到 ggml CPU 或 oneDNN，没有新增
HYLE 数学、fallback 或热路径选路。固定 320-tensor artifact 不含 MTP/NextN，因此 R13 仍按已审计
applicability 证据排除。

不可变 MSVC 19.50 Release/WERROR selector 通过 docs、closure、模型输入工具、configure、CPU Direct
与 CPU Graph；Direct 可执行文件 30/30、Graph route 可执行文件 45/45。首次 selector 进程缺少
Visual Studio `rc`/`mt` 环境而在 compiler configure 停止；同一 candidate 加载 `VsDevCmd` 后通过，
期间没有源码变化。完整收据见
[CPU fidelity GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen35-fidelity-gguf-complete/README.md)。
ledger 现有 18 个 `RUN_PASS`、158 个 `VALIDATION_NOT_RUN`。CUDA、Apple、Vulkan 六格以及 D5-A5
三交付、D5-A6 P1 仍未由本 tests-only candidate 代签。

**D5-F11 CUDA complete（`d407d9cc`）**：普通 CUDA Direct/Graph 已完整运行同一固定
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` profile。R01 的 F16 embedding 冷绑定原始 CCCL checked
Gather，并按其合同持有 workspace；R02 保留 Q8_0 gate/up/down；R03/R14 分别运行 Q8_0 与 F16
投影；R04/R05 保留精确 `8/2 x 256` GQA、K4 causal conv 和 `16 x 128` Gated DeltaNet 语义。
每条 Direct/Graph 路径准备一次并以变化输入重放，匹配独立 Q8_0、F16 或 fragment oracle；无效
embedding 索引失败不覆盖已发布结果。所有调用和 Graph report 只固定到 ggml CUDA、cuBLAS、
SGLang/CCCL native 或 FlashInfer，没有新增 HYLE 数学、fallback 或热路径选路。R13 继续按固定
artifact 的已审计缺失排除。

不可变 RTX 4070/SM89、MSVC 19.50、CUDA 13.3、Release/WERROR selector 通过 docs、closure、
configure、CUDA Direct 与 CUDA Graph；完整 Direct 29/29、Graph 46/46，127 个无关 gate 跳过或
复用。完整收据见
[CUDA fidelity GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-fidelity-gguf-complete/README.md)。
ledger 现有 20 个 `RUN_PASS`、156 个 `VALIDATION_NOT_RUN`。Apple、Vulkan 四格以及 D5-A5 三交付、
D5-A6 P1 仍未由本 tests-only candidate 代签。

**D5-F11 Apple complete（`65bdb5f5`）**：普通 Apple Direct/Graph 已完整运行同一固定
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` profile。R01 的 F16 embedding 由 MLX Metal 执行；
R02 的 Q8_0 gate/up/down 以及 R03/R14 的 Q8_0 投影由原始 ggml Metal 执行，F16 投影由
MLX Metal 执行；R04/R05 保留精确 `8/2 x 256` GQA、K4 causal conv 和 `16 x 128`
Gated DeltaNet 语义。每条 Direct/Graph 路径准备一次并以变化输入重放，匹配独立 Q8_0、F16 或
fragment oracle；无效 embedding 索引失败不覆盖已发布结果。所有调用和 Graph report 只固定到
MLX 或 ggml Metal，没有新增 HYLE 数学、fallback 或热路径选路。R13 继续按固定 artifact 的已审计
缺失排除。

不可变 MacBook Air M4、AppleClang 21、macOS SDK 27.0、Release/WERROR selector 通过 docs、
closure、configure、Apple Direct 与 Apple Graph；完整 Direct 59/59、Graph 32/32，127 个无关
gate 跳过或复用。完整收据见
[Apple fidelity GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-fidelity-gguf-complete/README.md)。
ledger 现有 22 个 `RUN_PASS`、154 个 `VALIDATION_NOT_RUN`。Vulkan 两格以及 D5-A5 三交付、
D5-A6 P1 仍未由本 tests-only candidate 代签。

**D5-F11 Vulkan complete（`123c87c1`）**：普通 Vulkan Direct/Graph 已完整运行同一固定
`qwen35_2b:fidelity_gguf:UD-Q8_K_XL` profile。R01 的 F16 embedding、R02 的 Q8_0
gate/up/down，以及 R03/R14 的 Q8_0 与 F16 投影均固定到成熟 ggml Vulkan region；R04/R05
保留精确 `8/2 x 256` GQA、K4 causal conv 和 `16 x 128` Gated DeltaNet 语义，并只使用已准入的
ggml 或有限 IREE Vulkan dispatch。每条 Direct/Graph 路径准备一次并以变化输入重放，匹配独立
Q8_0、F16 或 fragment oracle；无效 embedding 索引失败不覆盖已发布结果。候选没有新增 HYLE
数学、fallback、Provider 或热路径选路，R13 继续按固定 artifact 的已审计缺失排除。

不可变 Windows RTX 4070、driver 610.62、MSVC 19.50、Vulkan SDK 1.4.357.0、Release/WERROR
selector 通过 docs、closure、configure、Vulkan Direct 与 Vulkan Graph；完整 Direct 142/142、
Graph 53/53，127 个无关 gate 跳过或复用。完整收据见
[Vulkan fidelity GGUF complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-fidelity-gguf-complete/README.md)。
ledger 现有 24 个 `RUN_PASS`、152 个 `VALIDATION_NOT_RUN`。D5-A5 三交付和 D5-A6 P1 仍未由本
tests-only candidate 代签。

### D5-F12：Qwen3.8 CPU GQA 资格域修复

Qwen3.8-27B 官方 profile 的 R04 使用 BF16、24 query heads / 4 KV heads × 256 和 F32 累加。
普通 CPU Direct 在 RMSNorm/RoPE 准备成功后拒绝 SDPA；同一 executable 的 Qwen3.5 8/2×256 GQA
通过。源码定位到 `compute_provider_ggml` 的 SDPA 资格函数把成熟路径写死为 `Q=8、KV=2、D=256`，
而紧邻的固定 ggml lowering 已使用 `ggml_mul_mat` 的 head-batch 广播，没有 8/2 专用算法。失败和源码
边界见 [Qwen3.8 R04 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-r04-cpu-gqa-admission-audit/README.md)。

因此 `compute_provider_ggml` 从 D5 candidate 精确退回本卡 AUDITED。有限 FIX 只有：

1. 在既有 dense BF16/F16/F32、causal、无显式 mask、F32 accumulation 域内，将 SDPA 资格从固定
   8/2 改为正 head 数、`Q_heads % KV_heads == 0`、Q/K/V depth 相同、K/V 形状相同和既有输出形状；
2. 把 Qwen3.8 `attn_output_gate=true` 所需的公共 Sigmoid 只映射到固定 ggml 已有的
   `ggml_sigmoid`；保留 F32 内部执行和既有输出转换，验证 BF16、有限尾部、±∞ 与 NaN，不开放其它 unary；
3. Provider 回归同时保留 8/2×256 并新增 24/4×256，两者继续 lower 到同一固定 ggml program；
4. Qwen3.8 官方 CPU Direct/Graph 使用 exact profile、真实内宽、两次变化输入和独立 oracle 完成
   R01-R05、R13-R15；既有 R05 K16/V48×128 收据并入完整格；
5. 只运行 `compute_provider_ggml`、`compute_route_cpu`、`graph_route_cpu` 与 selector 选择的公共检查。

不修改公共 SDPA/Unary schema、ggml 源码、算法、路由顺序、native Provider 或任何 HYLE 数学/kernel；若固定
ggml 实际执行 24/4 仍失败，则记录精确 `BACKEND_REJECTED`，不在 D5 实现替代 attention。

不可变 candidate `667dda0d6a696eab0d3f243386a73f2f54cefe55` 已完成该有限 FIX。CPU Direct 与
Graph 均在官方 H=5120、FFN=17408、GQA 24/4x256、partial interleaved RoPE 64、MTP 10240
形状上通过 R01-R04、R13-R15，并合并既有 R05 K16/V48x128 收据；每个 artifact 使用变化输入
replay，失败路径不发布结果，编译报告只含 ggml CPU/oneDNN 成熟 Provider。Release/WERROR selector
的 docs、closure、模型输入工具、configure、ggml Provider、CPU Direct 与 CPU Graph 七个 gate
全部 PASS，126 个无关 gate 跳过或复用；未启动 P1。完整收据见
[Qwen3.8 official CPU complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen38-official-complete/README.md)。
ledger 现有 26 个 `RUN_PASS`、150 个 `VALIDATION_NOT_RUN`；CUDA、Apple、Vulkan 六格以及 D5-A5
三交付和 D5-A6 P1 尚未由本批代签。

### D5-F13：Qwen3.8 CUDA GQA 资格域修复

Qwen3.8-27B 官方 profile 的 R04 使用 BF16、24 query heads / 4 KV heads × 256 和 F32 累加。
普通 CUDA Direct 的 RMSNorm、partial interleaved RoPE 和输出 Sigmoid 均可准备，但 SDPA 在准备期被
默认路线拒绝；同一 executable 的 Qwen3.5 8/2×256 GQA 已通过。源码定位到 `compute_provider_ggml_cuda`
的 SDPA 资格函数把 query/KV head 数写死为 8/2，紧邻的成熟 lowering 已使用固定 ggml CUDA
`ggml_mul_mat` head-batch 广播，没有 8/2 专用算法。失败和源码边界见
[Qwen3.8 CUDA R04 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-r04-cuda-gqa-admission-audit/README.md)。
放宽 GQA 后，同一普通入口又显示输出门公共 Sigmoid 落入冻结的 `hyle-cuda`；固定 ggml CUDA
tensor program 已承载同组 unary，且上游已有 `ggml_sigmoid`，只是 adapter 的资格表和 lowering 漏接。

因此 `compute_provider_ggml_cuda` 从 D5 candidate 精确退回本卡 AUDITED。有限 FIX 只有：

1. 在既有 dense BF16/F16、causal、无显式 mask、F32 accumulation、SM89 资格域内，将 ggml CUDA
   SDPA admission 从固定 8/2 改为正 head 数、`Q_heads % KV_heads == 0`、Q/K/V depth 相同、K/V
   形状相同和既有输出形状；
2. 把公共 Sigmoid 只映射到固定 ggml 的 `ggml_sigmoid`，复用 tensor program 既有 F32 内部执行和
   输出转换；验证 BF16 有限输出，不开放其它 unary；
3. 保持 FlashInfer 的 8/2×256 已资格化原生 kernel 域不变；24/4 由固定 ggml CUDA 成熟图接管，
   不新增 HYLE kernel、fallback、热路径搜索或路由顺序；
4. Provider 回归保留 8/2×256 并新增 24/4×256，两者必须准备到成熟 Provider；Qwen3.8 CUDA
   Direct/Graph 使用 exact profile、两次变化输入、独立 oracle 和稳定 replay 完成 R01-R04、R13-R15；
5. CUDA 首次 submit 允许固定上游建立一次 graph metadata，资源稳定性从第一轮 warm replay 到第二轮
   replay 比较；只运行 `compute_provider_ggml_cuda`、`compute_route_cuda`、`graph_provider_ggml_cuda`、
   `graph_route_cuda` 与 selector 选择的公共检查。

不修改公共 SDPA/Unary schema、ggml 源码、算法、FlashInfer kernel 或任何 HYLE 数学；若固定 ggml CUDA
实际执行 24/4 仍失败，则记录精确 `BACKEND_REJECTED`，不在 D5 实现替代 attention。

**D5-F13 CUDA CANDIDATE（`80bbbba06`）**：固定 ggml CUDA adapter 已把既有 attention program
的资格域从固定 8/2 放宽到正整数倍 GQA，并把公共 Sigmoid 映射到上游 `ggml_sigmoid`；FlashInfer
仍只接管其已资格化的 8/2×256 域。Qwen3.8-27B 官方 H=5120、FFN=17408、GQA 24/4×256、partial
interleaved RoPE 64、MTP 10240 在普通 CUDA Direct/Graph 均以变化输入通过，首次 submit 后 replay
资源稳定，invalid Gather 不发布结果；既有 R05 K16/V48×128 收据并入完整格。Release/WERROR
selector 的 docs、closure、configure、ggml CUDA Provider、CUDA Direct、ggml CUDA Graph Provider 与
CUDA Graph Route 七个 gate 全部 PASS，125 个无关 gate 跳过或复用；未启动 P1。完整收据见
[Qwen3.8 official CUDA complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen38-official-complete/README.md)。
ledger 现有 28 个 `RUN_PASS`、148 个 `VALIDATION_NOT_RUN`；Apple、Vulkan 四格以及 D5-A5 三交付和
D5-A6 P1 尚未由本批代签。

### D5-F14：Qwen3.8-2.4T 官方 BF16 grouped expert 准入

固定 `Qwen/Qwen3.8-2.4T-A95B@207bd685` 官方 profile 的 1,609 个 tensor 全部为 BF16；
92 层主干中的 routed expert 按层堆叠，配置为 H=8192、512 experts、top-10、expert/shared
intermediate=2048。现有公共 `GroupedMatMul` 却把 dense expert weight 写死为 F32，四个固定 ggml
adapter 也重复这一限制，因此 R11 在进入成熟 Provider 前即被拒绝。固定 ggml `ggml_mul_mat_id`
本身创建 F32 输出，upstream backend test 明确登记 BF16 weight + F32 activation，CPU 使用 BF16
type trait 的 F32 dot 域，CUDA/Metal/Vulkan 的现有 backend admission/dispatch 均包含 BF16 weight。
完整源码边界见 [D5-F14 admission audit](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-grouped-admission-audit/README.md)。

因此 `compute_provider_ggml`、`compute_provider_ggml_cuda`、`compute_provider_ggml_metal` 与
`compute_provider_ggml_vulkan` 从各自 D5 candidate 精确退回本卡 AUDITED；`compute_ops` 已由 D5
持有 AUDITED。有限 FIX 只有：

1. 公共 `GroupedMatMul` dense schema 接纳 F32 或 plain BF16 weight；activation/output 保持 F32、ids
   保持 I32、accumulation 保持显式 F32，输出形状和现有 GGUF packed weight 的 F32 logical type 合同
   不变；
2. 四个固定 ggml adapter 只放宽 plain dense BF16 weight 的相同域，继续直接 lower 到
   `ggml_mul_mat_id`；CPU、CUDA、Vulkan 解除重复的 F32 admission，Metal 补齐其 native dtype 到
   `GGML_TYPE_BF16` 的映射；不接纳 BF16 activation/output，不增加转换、fallback、热选择或 HYLE 数学；
3. 四个 Provider 回归用 BF16 expert weight、F32 activation/output、I32 ids、变化输入 replay 和独立
   double oracle，保留既有 F32/量化 grouped expert 回归；
4. Qwen3.8-2.4T 官方 Direct/Graph batch 保留 H=8192、expert intermediate=2048、512-way router 与
   top-10 语义；允许按 D5 容量合同只实例化被选 active expert 子集，但不得缩小内宽、dtype、state
   或语义端口；
5. 只运行 `compute_ops`、四个 ggml Provider owner suite、对应 Direct/Graph route factor 与 selector
   选择的公共检查；API/算法和路线批次稳定后才进入 D5-A5/A6，不在本修复中提前启动 P1。

不修改 ggml 源码、Provider 顺序、路由键、量化格式、资源模型或任何 HYLE kernel/shader。若某后端
实际拒绝 BF16 `MUL_MAT_ID`，该路线记录精确 `BACKEND_REJECTED` 并停止，不以 F32 常驻复制伪造覆盖。

**D5-F14 CANDIDATE（`d707af1e562e5fd4c183d187238a58bb8eab6641`）**：公共 schema 和四个
固定 ggml adapter 已接纳 plain BF16 expert weight，并继续以 F32 activation/output/accumulation、
I32 ids 直接执行上游 `ggml_mul_mat_id`。Windows CPU/CUDA/Vulkan 与正式 MacBook Air M4 的
Provider、普通 Direct 和 Graph 定向用例均以变化输入通过；完整受影响套件分别通过 CPU 49/49、
64/64，CUDA 48/48、65/65，Vulkan 162/162、73/73，Apple 78/78、51/51。Metal 验收首次发现其
dtype 映射遗漏，最小补齐后实际编译运行 `kernel_mul_mv_id_bf16_f32_4`；没有转换、fallback、热选择
或 HYLE 数学。完整 partial 收据见
[Qwen3.8-2.4T BF16 grouped expert](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-grouped/README.md)。
本收据只签 R11；公共 API selector 选择的 Windows Compute/Graph 源码、共享、静态及搬迁 CPU consumer
均已通过。R01-R05、R10、R13-R15、D5-A5 各路线 profile 三交付与 P1 尚未完成，因此 ledger 仍为
48 个 `RUN_PASS`、128 个 `VALIDATION_NOT_RUN`。

### D5-F15：Qwen3.8-2.4T 官方 BF16 CPU 片段批量验收

tests-only candidate `041d9e63a` 把已验 Qwen3.8 CPU Direct/Graph 参数化 consumer 复用到官方
2.4T 的 H=8192、shared expert N=2048、GQA 64/4×256、GDN K16/V128×128 和 MTP 16384 输入。
普通入口分别运行 embedding/投影/head、shared FFN、RoPE/GQA/output gate、MTP 和 GDN 状态更新，
两次变化输入复用准备/编译结果并对照独立 oracle。现有 R11 BF16 grouped expert 收据继续独立有效。
Windows CPU Direct 全套 54/54、Graph 全套 69/69，无跳过；证据见
[Qwen3.8-2.4T CPU batch](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-cpu-batch/README.md)。
R10 路由/派发/合并和 R05 的完整模型投影/存储边界仍待验；本 partial 不改变 ledger 48/176，
不代签各路线三交付或 P1。

### D5-F16：Windows CUDA 运行库版本使 CCCL Provider 不可达

**SCOPED（baseline `0d25f9938`）**：在官方 Qwen3.8-2.4T CUDA 批次中，R01 的 BF16 Gather
准备失败；相同的 Qwen3.8-27B 既有 R01 用例在当前机器也重现。源码定位到
`compute_provider_cuda/collectives_provider.cc` 的 CCCL 资格规则：SM 8.9、编译 Toolkit 13030、
`CCCL-300304` 都匹配，但运行库要求精确 13030。当前 `v13.3` SDK 的 `CUDART_VERSION=13030`、
`CUB_VERSION=300304`，其 Windows 运行库和驱动经 API 均报告 13040；因此 CCCL 的 Gather、TopK、
Scan 在匹配阶段整体不可达，最后的 CUDA tensor Provider 只返回“未实现该算子族”。
[NVIDIA CUDA Runtime 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html)
明确说明 CUDA 13 起 Windows 使用随显示驱动提供的运行库，其版本可能与编译 Toolkit 不同；
[CUDA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
允许 CUDA 13.x 内的兼容组合，但具体 Provider 能力仍须本机验证。

本次只重开 `compute_provider_cuda` 的 CCCL 环境资格；不修改 TopK/Gather/Scan 数学、CCCL 调用、
其它 CUDA Provider 或优先级。D4 CLOSED 的算子清退结论仍有效。先完成 AUDITED 有限 FIX 定界，
再改产品代码；现有 CUDA 测试候选的 R01 失败保持原样作为触发证据。

**AUDITED 有限 FIX**：

1. 仅将 CCCL 的 `kRuntimeVersion` 从精确 13030 改为闭区间 13030..13040；SM 8.9、编译 Toolkit
   13030、`CCCL-300304` 仍精确匹配。该区间只覆盖已验旧机运行库和本次实际 13.4 运行库；不宽放
   CUDA 14、其它架构/Toolkits 或其它 Provider。
2. 在 RTX 4070 当前 13040 环境重新运行 CCCL Gather/TopK/Scan owner suite 与 Qwen3.8 27B/2.4T
   普通 CUDA Direct/Graph；对 2.4T 保留 H=8192、K16/V128、top-10、两次变化输入和独立 oracle。
3. 只运行 selector 指向的 Provider/Direct/Graph factor、输入工具与公共检查；本规则没有更改数学或
   公开 API，本 partial 不启动 D5-A5/A6。若 13.4 库实际执行不合格，则撤销扩域并记录精确拒绝。

**CANDIDATE（`05b374e61`）**：有限 FIX 已实现，RTX 4070 / SM 8.9 上 Toolkit 13.3 编译产物
与驱动提供的 13.4 runtime 组合实际运行通过。CCCL owner 定向 10/10、全套 42/42；CUDA Direct
全套 53/53、Graph 全套 70/70，均无跳过。Qwen3.8-2.4T 官方 BF16 的 Direct/Graph 各 5/5
片段通过；既有 27B Gather 也恢复。证据见
[Qwen3.8-2.4T CUDA batch](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-cuda-batch/README.md)。
这仍是局部片段证据：R10 与 R05 完整链、各路线 profile 交付和 P1 未签，ledger 维持 48/176。

### D5-F17：Qwen3.8-2.4T 官方 BF16 Apple/Vulkan 形状复用

tests-only candidate `10c878305` 将既有 Qwen3.8 Apple/Vulkan Direct/Graph profile consumer
参数化，加入官方 2.4T 的 H=8192、shared expert N=2048、GQA 64/4×256、MTP 16384 和
GDN K16/V128×128；仍复用同一普通入口、成熟 Provider 与独立 oracle。各路线此前的 R11 grouped
expert 收据独立有效。Vulkan RTX 4070 的新形状 Direct/Graph 各 5/5，完整 factor 167/167、
78/78；MacBook Air M4 的新形状 Direct/Graph 各 5/5，完整 factor 83/83、56/56，均无跳过。
Windows selector 选择 Vulkan 两个 factor；Apple 验证副本的 selector 选择 Apple 和 Vulkan factor，
由两台适用平台分别执行。证据见
[Qwen3.8-2.4T Apple/Vulkan batch](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-bf16-apple-vulkan-batch/README.md)。
R10 router/dispatch/combine 与 R05 完整模型投影/存储链未由这批片段覆盖，ledger 仍为 48/176；
不代签 D5-A5 三交付或 D5-A6 P1。

### D5-F18：Qwen3.8-2.4T R10 router 的成熟 Softmax 接入

**AUDITED（baseline `dbf492934`，owner `compute_provider_ggml`）**：固定 llama.cpp
`4c9233c` 的 `qwen35moe.cpp` 调用 `build_moe_ffn`，其实现对 512 个 router logits 先做
Softmax，再稳定取 top-10、抽取选中权重并重新归一化；之后按选中专家执行乘法、加权合并，shared
expert 另有 sigmoid gate。当前 HYLE ggml CPU Provider 已有这些基础图节点中的 MatMul、Cast、
TopK、Reduce 与 Binary，但 F32 router Softmax 没有 admission/translation，普通 CPU 入口选到冻结的
`hyle-native`。新增的真实 H=8192、E=512/top-10 Direct consumer 在成熟 Provider 断言处定向失败，
其余数值断言通过；这不能作为 R10 PASS。

有限 FIX 只在 `compute_provider_ggml` 接入固定 ggml 原版 `ggml_soft_max`：F32 plain/dense rank-2、
最后轴、无 mask、scale=1、标准 Softmax、F32 accumulation；不扩宽其它 dtype/轴/mask，不改路由
优先级、不加 HYLE 算法。用两次变化输入和稳定 tie 检查 512→10 的概率、ID 与归一化；随后接到
已有 BF16 grouped expert 的真实 selected IDs 和权重，验证 Direct/Graph 与四路线。适用 selector
factor、公共检查和 P0 成本结构先验收；本阶段不代签 D5-A5/P1，也不强迫插入上游 ggml MoE 路径没有的
Scan/Scatter 节点。

**CANDIDATE（`66f4d80fb`，R10 router partial）**：固定 ggml CPU `ggml_soft_max` 已按有限 FIX
接入；H=8192 BF16 router projection 产生 512 个 score，经显式 F32 Cast、Softmax、稳定 TopK、
Reduce/Divide 形成 10 个 ID 和重新归一化的权重。普通 CPU Direct 与 Graph 用两次变化输入和一次
并列 tie 各通过 1/1。`oneDNN=ON` 的正式受影响完整套件为 Provider 40/40、Direct 55/55、
Graph 70/70。`oneDNN=OFF` 的 Graph 全套 64/65，失败的旧 Qwen3.5 用例明确断言必须出现 oneDNN
region，故不能把该配置误记 PASS；它不是本次 Softmax 回归。完整局部证据见
[R10 CPU router](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-r10-router-cpu/README.md)。
该 partial 没有把 top-10 ID/权重接到 grouped expert、shared gate 和最终 combine，也未裁决 BF16
router MatMul 存储边界与固定 ggml 图的 F32 score 边界是否等价；不签 R10 或完整 profile cell。

**D5-F18 后续局部收据（`34a13aab`、`0c97b284`）**：十个 resident expert 的测试权重已
区分；CPU Graph 用同一 BF16 token 完成 512-way router、稳定 top-10、成熟 Gather 复制激活、动态
ID 驱动 BF16 grouped expert、归一化权重乘法和十路合并。两次改变顺序的 replay 对照独立 oracle，
Graph 定向 1/1、完整 factor 70/70。夹具变更在 CPU/Apple 完整 Direct/Graph 和 CUDA/Vulkan 定向
Direct/Graph 均通过。详见 [R10 CPU Graph dispatch partial](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-r10-graph-dispatch-cpu/README.md)。
这仍未接共享专家分支及最终相加，Direct 与其它三路线尚未接动态路由。固定 llama.cpp
`build_lora_mm` 的 ggml MatMul 产出 F32 logits，而现有公共 BF16 MatMul 合同先存 BF16 再 Cast F32；
需先定界这个精度差异，不能以当前 synthetic logits 正确就称模型等价。ledger 仍为 48/176。

### D5-F19：官方 BF16 router 的 F32 score 存储边界

**AUDITED（baseline `f32c85ef3`，owners `compute_ops` 与四个固定 ggml Provider）**：
固定 llama.cpp `4c9233c` 的 `build_moe_ffn` 通过 `build_lora_mm(gate_inp, cur)` 产生 router
logits，所调用的 `ggml_mul_mat` 在固定 ggml 源码中明确创建 `GGML_TYPE_F32` 输出。现有 HYLE
公共 `MatMul` 要求两输入同 dtype 且输出也同 dtype；四路线官方 BF16 consumer 因此先把 score 写入
BF16，再 Cast F32。512 路 top-10 的第 10/11 名可因这一额外 BF16 舍入变成 tie，改变专家成员；
目前的单对角 synthetic router 没有暴露这个风险。此为模型语义缺口，不以 P1 或性能猜测驱动 API。

有限 FIX 固定如下：

1. 在公共 model operation 中增加明确的 rank-2 `BF16 Linear -> F32` 语义，输入分别为
   activation `[T,K]` 和模型原始输出优先存储的 weight `[E,K]`，F32 输出 `[T,E]`；不改通用
   `MatMul` 的同 dtype 合同，不隐式转换已有图；
2. CPU/CUDA/Metal/Vulkan 四个固定 ggml Provider 只在该精确域接纳并直译原版
   `ggml_mul_mat` 的 F32 输出。Vulkan 的一行 BF16 小 matvec 复用已审定的上游管线限制处理：
   只将该激活在设备图内提升 F32，权重仍 BF16；验收含此必要成本。现有 GGUF、oneDNN、MLX、
   cuBLAS 等 Provider 顺序及准入不变；不增加 HYLE 数学、F32 常驻权重副本或设备往返；
3. 先以第 10/11 名接近的 BF16 输入制造 BF16 中间存储会改变 top-10 的红灯测试，再对四路线
   Direct/Graph 验证 F32 score、ID、归一化权重与动态 expert replay；失败时只记录精确 Provider
   边界，不能用 host 排序或修改测试 oracle 代替；
4. 运行 selector 选择的公共语义、Provider、Direct/Graph factor、三交付风险和公共检查；P0 审查
   转换/同步/常驻资源，稳定不可变候选的适用 P1 留在对应路线 profile 验收，不在本精度 FIX 中
   提前多次计时。

本 FIX 不签共享 expert、完整 R05、其它模型/量化画像或完整 profile cell；ledger 保持 48/176。

**路线接缝重定界（baseline `f595c7f51`）**：`BFloatLinearF32` 的固定 ggml Vulkan
Direct 定向实际执行已通过，但 Graph 编译返回 `Vulkan custom schema needs a cold native
lowering`，在到达同一 Provider 前被成熟 custom 白名单拒绝。Apple 的 Direct 与 Graph 路线同样
以固定 `fixed_ggml_model_operation` 列表划分 ggml region。新模型 consumer 是 F19 原生 F32
router；这构成 `graph_route_vulkan` 的新事实 reopen trigger（当前 SCOPED），并要求
`compute_route_apple`、`graph_route_apple` 从既有 D5 候选重定界为 AUDITED。既有 D4 删除裁决
和 F8 候选保留为历史证据。Apple 两 owner 的有限 FIX 仅是各自在冷期的
`fixed_ggml_model_operation` 分类加入此 custom schema，交由已审定 ggml Metal Provider 执行；
Direct/Graph 各以近并列 F32 score 和变化输入验证。
本轮只接通该 operation 至已 AUDITED 的固定 ggml Provider，不改变 Graph 调度、MLX region、
CUDA 路由、资源/完成语义、旧 GGUF 路由或自研数学。

**Vulkan Graph AUDITED（baseline `a6c79badb`）**：失败点精确在
`graph_route_vulkan/vulkan_lowering.cc` 的 `mature_custom` 分类，不在固定 ggml Vulkan
Provider、Vulkan shader 或用户自定义 lowering。有限 FIX 只加入 `BFloatLinearF32` 这一已由
Direct 实际执行的 schema；以普通 Graph 准备、两次近并列变化输入、F32 score、Provider region
report 与完整受影响 factor 验证。不修改其他 custom 准入或新增后备执行器。

**F19 部分候选（`774c3a32d`）**：公共 BF16 `[T,K]` × 原生输出优先 BF16 `[E,K]`
已接到四路线固定 ggml Provider 的 F32 score 边界。第 10/11 名近并列红灯先证明旧 BF16
中间存储会改变入选专家；新路径的四路线 Direct/Graph 单算子定向结果通过，MacBook Air 四个
相关完整 suite 也通过。Apple/Vulkan 的 Graph factor 辅助函数实际以 GraphBuilder 建单节点图并
调用路线 Graph compile；尚未签多节点 R10 整链。CPU Graph 的动态专家与加权合并仍为 partial，
其他路线
尚未连接该完整链。Windows selector 的 27 个受影响 gate 尚未全部完成：修正 Vulkan SDK 后的
全架构 ggml CUDA 构建已停止，具名中断 lease 阻止旧候选继续取得 selector verdict。
[F32 score partial receipt](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-router-f32-score/README.md)
记录实际测试和未签边界；三交付与 P1 不写 PASS，ledger 维持 48/176。

### D5-F20：CUDA Graph workspace 别名根对齐

**SCOPED（baseline `6f954ef5`，owner `graph`）**：本轮完整 Qwen3.8-2.4T
BF16 MoE CUDA Graph 首次将 `Cast` 的内部 F32 输出经零偏移 `View` 交给固定 ggml CUDA region。
Region 的冷期 `ResourceRequirement.minimum_alignment` 已声明 256 字节；Graph 的内部生命周期
规划却只按 tensor 基本对齐和 64 字节默认值分配根，未把下游 port 要求传到别名根。
MSVC Release/WERROR 编译通过，普通 Graph submit 的完成状态实际报
`CUDA buffer offset does not satisfy the requested type alignment`，属于新的可复现重开事实。
同批 CPU、Vulkan Graph 整链定向已通过；CUDA 失败不能记作模型 cell PASS。

本缺陷只涉及 Graph 冷期的 workspace/alignment 规划，不修改 ggml CUDA 的既有
256 字节合同、CUDA 算子、Provider 次序或公共路由。AUDITED 前核对具体失败值、
端口 proposal 与别名偏移；有限 FIX 应使选中 Provider 的每个输入/输出最小对齐传到物理
实例及零偏移别名根，对非零偏移不能满足的组合在准备期拒绝。验收只选受影响 Graph
编译/执行、别名和四路线原有 owner 回归，候选稳定后按 selector 签署；不重跑无关 P1。

**AUDITED（baseline `49c6c63b`）**：固定 ggml CUDA region 的
`propose_resources` 对需要向量化的边界端口已提交 256 字节要求，Graph 的
`discovery.requirements` 在冷期持有该值；`prepare_placed_steps_attempt` 建物理实例时没有保存
这个要求，`plan_lifetimes` 建根请求时仅取 `max(64, TensorDesc.required_alignment())`。
因此缺陷在 Graph 消费已声明的端口约束，不在 Provider 提案或设备枚举。有限 FIX 固定为：
保存被选中 operation/region 的输入输出端口最小对齐；把零偏移 View/alias 的要求向根传播；
工作区请求使用传播值，非零偏移冲突在准备期返回错误。只改 Graph 冷期规划，不增加
新 Provider、数学、运行时搜索或公共治理逻辑。定向重放 CUDA 三组变化输入、近并列成员及
shared/final oracle，随后运行 Graph CPU/CUDA/Vulkan 受影响 owner 和 Mac Apple 回归；
完整 selector、三交付及适用 P1 仍在稳定批次签署。

**Graph 批次部分收据（candidate `825c41f8d`）**：CPU、CUDA、Vulkan 与
MacBook Air Apple 的完整多节点 R10/R11 Graph 均以三组变化输入通过独立 ID、routed、
shared、combine 和 final oracle；四路线完整 Graph owner suite 及适用 Graph 核心回归通过。
Graph 对齐 FIX 已在 CUDA 真实执行中验证。详见
[完整 MoE Graph 批次](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen38-24t-full-router-graph/README.md)。
这是 Graph 子批结果，Direct 整链、完整 selector、三交付和 P1 尚未签署；ledger 保持 48/176。

### D5-F21：Qwen3.8-2.4T GGUF grouped expert 原生权重准入

**SCOPED（baseline `2cb5d34e`）**：固定的 `UD-IQ4_XS` 与 `Q8_0` GGUF header 分别含
`blk.0.ffn_gate_exps.weight` 的 `[8192,2048,512]` IQ4_XS/Q8_0、独立 shared expert
及 F32 router；普通用户需以同一 exact artifact 画像提交权重、F32 activation 和 I32 本地
expert ID，经 CPU/CUDA/Apple/Vulkan 的 Direct `prepare` 或 Graph `compile` 固定成熟
`ggml_mul_mat_id`，变化输入重复执行。D5 允许缩小常驻 expert 容量，仍保留内宽、原生
block、top-10 与全局 ID 到本地槽的保序映射；不承诺整份 2.4T 权重常驻。
复核纠正：`GroupedMatMul::infer_output_descriptions` 原有 F32 分支已接受
`gguf_grouped_weight_description` 生成的合法 blocked `[E,N,K]`；`is_dense()` 也包含
canonical blocked-dense。因此此前“公共准入拒绝”的判断不成立；撤销 `b20c95bd` 新增的
格式准入实现，保留真实形状测试。本批是成熟路径接线验收，不是新增算子。owner 限于
`compute_ops` 与四个现有
`compute_provider_ggml*`；Graph/route 只消费其已登记的 grouped region，若实际编译暴露
独立缺陷再按精确 owner 定界。非目标为 HYLE 数学、通用量化器、其它模型的 FP4/FP8、
热路径重新选择及整模型加载。P0 审计准备期 canonical 格式检查和资源上界；D5-A6 的
适用 P1 仍随完整 profile 批末执行。

**AUDITED，有限 FIX**：固定 ggml CPU/CUDA/Metal/Vulkan 均有原生量化格式映射和
`ggml_mul_mat_id` 构图；Vulkan 已有 canonical grouped GGUF 权重域，Metal 已允许
rank-3 block-axis-2 权重；CPU/CUDA 的 logical F32 分支也已允许非 plain 权重，
并限制为 rank-3、block-axis-2。公共 schema 同样已经接受 F32 blocked 权重，
当前没有证实产品准入缺陷。
按同一份固定 header 反证，不改变 operation ID、F32 activation/output、I32
ID、资源/完成/失败语义。仅允许：

1. 保留现有 `compute_ops` 产品 schema；以公共测试确认 canonical GGUF `[E,N,K]`、
   logical F32 的既有准入，错误 dtype/ID shape 及不完整 K block 的拒绝。具体存储与
   stride 的准入继续由各 Provider 准备期处理，不在本批无依据收紧公共 schema。
2. 四个现有 ggml adapter 先保持原样；只在真实片段揭示具体准入或资源缺陷时按
   所属路线定向修复。四路线都使用各自 ggml 原生 program，不建 HYLE
   grouped kernel 或共享万能执行器。
3. 以 `UD-IQ4_XS` 的 IQ4_XS/Q5_K/Q6_K 和 `Q8_0` 的 Q8_0 角色先做同批
   storage/MatMul 入口测试，再用常驻 11 个 expert、512-way/top-10 的量化 routed
   gate/up/down、shared expert、近并列 ID 和变化输入两次 submit 检查 Direct/Graph
   及错误不发布；单个 expert 切片绝不代签 grouped chain。
4. 稳定候选一次运行 selector 所选公共语义、四路线 owner 与直接 Graph 消费者，
   随 D5-A5/A6 统一签三交付和适用性能。未运行格保持 `VALIDATION_NOT_RUN`。

停止条件：若固定上游在精确设备/版本/格式/形状拒绝原生 `mul_mat_id`，记录原生错误
及内存边界并按该路线 `BACKEND_REJECTED` 定界；不得用 F32 解包或自研数学伪装 GGUF
通过。`2cb5d34e` 仅批量写入六个真实权重角色的四路线 Direct/Graph 测试消费者，
已对照固定 header 6/6；该提交尚无编译或运行 verdict，不改变 48/176 ledger。

**本批接线更新**：公共 schema 已恢复至 `2cb5d34e` 的原有实现；新增语义测试 2/2 与
CPU IQ4_XS/Q8_0 grouped 定向执行确认原有准入有效。四路线 Direct/Graph 均已接入
共享 grouped 夹具及完整 MoE 画像夹具：F32 router/shared gate、IQ4_XS routed expert、
Q5_K/Q6_K shared expert，或 Q8_0 的对应权重角色。官方 BF16 与两种 GGUF 复用相同
recipe；保留 512-way/top-10、11 个常驻 expert、8192/2048 内宽、三次输入变化及近并列
ID 边界。原有独立解码 oracle 扩展到完整链，量化结果按既有 chain gate 归一化检查，
防止极小输出落在绝对容差内而错误接受全零结果。首轮旧 CPU 开发树关闭 oneDNN，
F32 router 因而选择 native，且误用 BF16 逐元素误差门；该轮不签通过。修正构建配置与
量化验收合同后集中检查，四路线完整批次尚待运行，不改变 ledger、三交付或 P1 状态。

**F21 定向补充审计（baseline `ea5cb859`，owner `compute_provider_ggml` 已 AUDITED）**：
开启 oneDNN 后同一 CPU Direct 批次 3/4 通过，GGUF 完整 MoE 的数值门通过，但两个 F32
router MatMul 仍报告 `hyle-native`。源码确认 oneDNN adapter 仅收 dense row-major 且主动
拒绝单线程 M<16；ggml CPU 的 plain MatMul 又只收 F16/BF16。这不是上游缺算子，也不能
把 `KEEP_RESIDUAL` 解释成已证明 HYLE 领先。有限 FIX 仅为 ggml CPU 既有 MatMul 准入补上
plain F32 rank-2 activation、native output-major 转置 RHS、F32 output 的域；保留原
`ggml_mul_mat`/F32 accumulation、资源/提交实现和所有其它准入条件。无需改变 oneDNN、
公共 API、native 实现或数学。用上述两个真实 router 及三次变化输入的完整 Direct/Graph
断言证明普通入口命中 ggml，已有 owner 测试随批执行；P1 仍只在稳定路线批末执行。

### D5-F22：grouped expert 动态 ID 的四路线检查边界

- baseline：`0e4bb7ed`；当前两个 2.4T GGUF 画像接线批，owner 绑定本卡。
- 新事实：CPU Direct 的真实 IQ4_XS `[11,2048,8192]` grouped fixture 在合法 ID 重放通过后，
  提交 `-1`/`11` 出现 `0xc0000005`。旧测试只有合法 ID，不能证明该失败边界。
- 有界源码审计：CPU `Program::index_checks` 和 CUDA 对应表只登记 Gather/Scatter/Gather；
  Metal `checked_index_port` 只为 Gather 设置；Vulkan 将 GroupedMatMul 直接合入 native region，
  未使用既有 `lower_index_check`。上游 MUL_MAT_ID 依赖调用方提供合法专家 ID。
- 固定 FIX：CPU/CUDA/Metal 三个已 AUDITED ggml owner 复用既有索引检查，在提交成熟程序前
  检查当前 I32 IDs 属于 `[0,E)`；CPU/CUDA recipe 的未检查内部 ID 不准越过该边界。
  Vulkan route 由 F8 CANDIDATE 退回 AUDITED，只在 `vulkan_program.cc`、`vulkan_region.cc`
  和私有声明处接入既有 GPU index check、safe IDs 和 checked publication；MUL_MAT_ID 仍由
  ggml Vulkan 执行，不新增 shader/数学，不做 host 模型调度，不改公共 API 或治理工具。
- 成本：CPU 读取当前 ID，CUDA 复用已有有界索引 staging/sync，Metal 使用原生共享缓冲；
  Vulkan 复用已有 GPU 校验与条件发布，禁止为了检查下载整块 activation/weight。
  三路线已有宿主检查机制不在本 FIX 重写；必要检查成本计入 D5-A6。
- 验证：同一批八入口验证两种 GGUF 格式的 `-1`、`E`、失败不覆盖已有输出、有效 ID 恢复；
  完整 MoE 和本批 attention/GDN/MTP/逐角色 projection 随同一稳定候选复核。
  失败 ID 不送入未修复的 GPU Provider 触发设备断言；源码已给出同类缺口。
- 停止：完成上述有限边界和已声明批次验收；不扩大为全库安全审查或新算子实现。

### D5-F23：CUDA 原生 F32 注意力准入

- baseline：`bdbede631`，owner `compute_provider_ggml_cuda` 已 AUDITED。
- 真实 consumer：两个 2.4T GGUF 画像的 F32 norm/RoPE、64/4×256 GQA、两位置 causal
  attention 和 sigmoid output gate。CPU 同一独立 oracle 已通过；CUDA Direct 报 SDPA
  无合格 Provider，Graph 报无完整路线。其它本批投影与完整 mixed MoE 均已通过。
- 原因：ggml CUDA adapter 的 `supports()` 只允许 F16/BF16，但其现有 BF16 attention
  lowering 已用原生 F32 QK MatMul/scale/causal mask/softmax/PV MatMul 执行，且保留 F32
  accumulation/source precision；输入 F32 无需新数学或降精度，当前仍被入口提前排除。
- 有限 FIX：准入加入同一 dense BSHD F32 dtype，其余 GQA、mask、shape、I32 workspace
  bounds 和 native capability 检查保持；复用同一既有图，不改 Provider 排序、FlashInfer、
  自研 CUDA 或公共 API，不把 F32 偷换成 F16。必要转换/临时 score 成本进入本轮统一验收。
- 验证：本批 CUDA Direct/Graph 的完整 attention 中间值和两位置 softmax oracle、重放；
  现有 F16/BF16 和 owner suite 在同一稳定批末复核，不单开逐算子 P1。

### 本批 Vulkan 数值与 Direct 生命周期的测试校正

固定 upstream `ggml-vulkan.cpp` 的 matrix path 在 coopmat2 下将 F32 两侧乘法输入转换 F16；
`ggml_prec_set_acc(F32)` 保证累加类型，不禁止输入舍入。这与本批两位置注意力首次采用的
8e-5 F32 点值门不一致。首个失败点：完整 F32 oracle `-0.1279833375`，独立 F16 乘法输入
舍入 oracle `-0.1280732523`，实际 `-0.1280732453`。没有修改 Provider、上游或图算法。
Vulkan attention 的 score/gated output 同时对完整 F32 参考检查 `2e-4 + 0.002*abs(ref)`，
并对独立舍入参考检查 `8e-5 + 0.00004*abs(ref)`；norm/RoPE 仍用原严格门，GDN state 用
`1e-7 + 0.00004*abs(ref)`。该数值预算只签本批真实归一化 attention 域，不声称任意 F32
指数范围都由 F32 乘法执行；CPU/CUDA/Apple 不使用这项 Vulkan 舍入参考。

Apple/Vulkan 的 Direct recipe 沿既有普通入口复用准备好的 Program，每次调用独立持有
invocation/result；下游绑定会保留上游结果，因此不能把尚有消费者引用的 invocation 当作可
覆盖槽位重放。Graph 则复用同一 invocation。此次校正不扩公共生命周期 API；D5-A6 必须分别
计入 Direct 的调用/资源成本与 Graph 重放成本，不能声称两者都已证明零 workspace 分配。

### Qwen3.8 Apple official Direct/Graph 完整格

不可变 tests-only candidate `6086dd721af664a1d759bb4eea3dfd645e02c99f` 在正式 MacBook Air M4
上以普通 Apple `Context::prepare` 和 Graph build/compile/bind/submit 覆盖官方 H=5120、
FFN=17408、GQA 24/4×256、partial interleaved RoPE 64 与 MTP 10240。每个 prepared artifact
使用变化输入重放，invalid Gather 不改写已发布结果；既有完整 ggml Metal R05 K16/V48×128 收据并入
两格。现有 MLX Metal 与固定 ggml Metal 直接接受这些形状，没有修改产品实现、Provider 顺序、公共
schema 或新增 HYLE 数学。

Release/WERROR selector 的 docs、closure、configure、Apple Direct 和 Apple Graph 五个 gate 全部
PASS，127 个无关 gate 跳过；完整 Direct factor 54.37 秒、Graph factor 13.05 秒。完整收据见
[Qwen3.8 official Apple complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen38-official-complete/README.md)。
ledger 现有 30 个 `RUN_PASS`、146 个 `VALIDATION_NOT_RUN`；Vulkan 两格以及 D5-A5 三交付和 D5-A6
P1 尚未由本批代签。

### Qwen3.8 Vulkan official Direct/Graph 完整格

不可变 tests-only candidate `5fc3c2f418795e6cd89b86e3f546d5775b96758a` 在 Windows RTX 4070
上以普通 Vulkan prepare/bind/submit 和 Graph build/compile/bind/submit 覆盖官方 H=5120、
FFN=17408、GQA 24/4×256、partial interleaved RoPE 64 与 MTP 10240。每个 prepared artifact
使用变化输入重放，invalid Gather 不改写已发布结果；既有完整 ggml Vulkan R05 K16/V48×128 收据
并入两格。现有固定 ggml Vulkan managed region 与限定 IREE dispatch 直接接受这些形状，没有修改
产品实现、Provider 顺序、公共 schema 或新增 HYLE 数学。

Release/WERROR selector 的 docs、closure、configure、Vulkan Direct 和 Vulkan Graph 五个 gate 全部
PASS，127 个无关 gate 跳过；完整 Direct factor 31.98 秒、Graph factor 16.73 秒。首两次 configure
分别因未绑定 Vulkan SDK，以及 CMD 提前展开 PATH 后丢失 Windows SDK `rc/mt` 而在产品编译前停止；
最终收据显式绑定 Vulkan SDK 1.4.357.0、`rc`、`mt` 与 `glslc`。完整收据见
[Qwen3.8 official Vulkan complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen38-official-complete/README.md)。
ledger 现有 32 个 `RUN_PASS`、144 个 `VALIDATION_NOT_RUN`；D5-A5 三交付和 D5-A6 P1 仍由稳定路线
批次统一执行。

### Qwen3.8 primary GGUF CPU Direct/Graph 完整格

不可变 tests-only candidate `051b3e061627661ac020e7a6217154677e69ddae` 已在普通 CPU
Direct/Graph 入口完成固定 `qwen38_27b:primary_gguf:UD-Q4_K_M` profile 的 R01-R05、R13-R15。
R01 保留 Q4_K embedding 的精确 5120 隐藏宽度；投影批次逐一运行 artifact 实际出现的 Q3_K、
IQ3_S、IQ4_NL、IQ4_XS、Q4_K、Q5_K、Q6_K、Q8_0，并额外固定 MTP 的 Q6_K
`10240 -> 5120` 权重。语义链保留 FFN=17408、GQA 24/4×256、partial interleaved RoPE 64、
Gated DeltaNet K16/V48×128 及 MTP 10240；每个 prepared artifact 以变化输入重放，越界 embedding
不发布结果。所有调用与 Graph report 只绑定固定 ggml CPU 或 oneDNN，没有增加 HYLE 数学、fallback、
热选择或产品实现。

Release/WERROR selector 的 docs、closure、模型输入工具、configure、CPU Direct 与 CPU Graph 六个 gate
全部 PASS，127 个无关 gate 跳过或复用；完整 Direct owner 63.21 秒，Graph 与 executor 63.67 秒。
完整收据见
[Qwen3.8 primary GGUF CPU complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen38-primary-gguf-complete/README.md)。
ledger 现有 34 个 `RUN_PASS`、142 个 `VALIDATION_NOT_RUN`；其余六条 Qwen3.8 GGUF 路线格以及
D5-A5 三交付和 D5-A6 P1 尚未由本批代签。

### Qwen3.8 primary GGUF CUDA Direct/Graph 完整格

不可变 tests-only candidate `02b58c8df5b3e281df8d6aea967c47d4eeea08c0` 已在普通 CUDA Direct/Graph 入口完成固定
`qwen38_27b:primary_gguf:UD-Q4_K_M` profile 的 R01-R05、R13-R15。R01 保留 Q4_K
embedding 的精确 5120 隐藏宽度；投影批次逐一运行 artifact 实际出现的 Q3_K、IQ3_S、IQ4_NL、
IQ4_XS、Q4_K、Q5_K、Q6_K、Q8_0，并额外固定 MTP 的 Q6_K `10240 -> 5120` 权重。语义链
保留 FFN=17408、GQA 24/4×256、partial interleaved RoPE 64、Gated DeltaNet K16/V48×128 及
MTP 10240；每个 prepared artifact 以变化输入重放，越界 embedding 不发布结果。所有调用与 Graph
report 只绑定固定 ggml CUDA、cuBLASLt、SGLang/CCCL 或 FlashInfer，没有增加 HYLE 数学、fallback、
热选择或产品实现。

MSVC 19.50 Release/WERROR selector 的 docs、closure、configure、CUDA Direct 与 CUDA Graph 五个 gate
全部 PASS，127 个无关 gate 跳过或复用；CUDA Toolkit 13.3.73、`CUDAARCHS=89-real`，完整 Direct
owner 63.09 秒、Graph owner 64.50 秒。完整收据见
[Qwen3.8 primary GGUF CUDA complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen38-primary-gguf-complete/README.md)。ledger 现有
36 个 `RUN_PASS`、140 个 `VALIDATION_NOT_RUN`；其余四条 Qwen3.8 GGUF Apple/Vulkan 路线格以及
D5-A5 三交付和 D5-A6 P1 尚未由本批代签。

### Qwen3.8 primary GGUF Apple Direct/Graph 完整格

不可变 tests-only candidate `dfa1423767d375154aa75cc717d327d7073d331b` 已在正式 MacBook Air M4
的普通 Apple Direct/Graph 入口完成固定 `qwen38_27b:primary_gguf:UD-Q4_K_M` profile 的
R01-R05、R13-R15。R01 保留 Q4_K embedding 的精确 5120 隐藏宽度；投影批次逐一运行 artifact
实际出现的 Q3_K、IQ3_S、IQ4_NL、IQ4_XS、Q4_K、Q5_K、Q6_K、Q8_0，并额外固定 MTP 的
Q6_K `10240 -> 5120` 权重。语义链保留 FFN=17408、GQA 24/4×256、partial interleaved RoPE 64、
Gated DeltaNet K16/V48×128 及 MTP 10240；每个 prepared artifact 以变化输入重放，越界 embedding
不发布结果。量化调用固定绑定 ggml Metal，其余语义片段固定绑定 MLX Metal；没有增加 HYLE 数学、
fallback、热选择或产品实现。

Q4_K `get_rows` 的独立 F32 oracle 首轮暴露了成熟 Provider 的数值边界。源码确认 ggml Metal
`dequantize_q4_K` 在高半字节分支以 half 精度执行 `xb->d / 16.h` 后发布 F32；实测最大绝对差
`4.506111145019531e-4`。仅该转换按 `5e-4` 验收，其它 embedding 格式仍保持 `1e-6`。
AppleClang 21 Release/WERROR selector 的 docs、closure、configure、Apple Direct 与 Apple Graph 五个
gate 全部 PASS，127 个无关 gate 跳过或复用；完整 Direct owner 80.02 秒、Graph owner 40.11 秒。
完整收据见
[Qwen3.8 primary GGUF Apple complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen38-primary-gguf-complete/README.md)。
ledger 现有 38 个 `RUN_PASS`、138 个 `VALIDATION_NOT_RUN`；剩余 Qwen3.8 GGUF Vulkan 两格以及
D5-A5 三交付和 D5-A6 P1 尚未由本批代签。

### Qwen3.8 primary GGUF Vulkan Direct/Graph 完整格

不可变 tests-only candidate `af6a99c450996250a3f557f1eda832ca91c7ef41` 已在正式 Windows
RTX 4070 的普通 Vulkan Direct/Graph 入口完成固定
`qwen38_27b:primary_gguf:UD-Q4_K_M` profile 的 R01-R05、R13-R15。R01 保留 Q4_K
embedding 的精确 5120 隐藏宽度；投影批次逐一运行 artifact 实际出现的 Q3_K、IQ3_S、IQ4_NL、
IQ4_XS、Q4_K、Q5_K、Q6_K、Q8_0，并额外固定 MTP 的 Q6_K `10240 -> 5120` 权重。语义链
保留 FFN=17408、GQA 24/4×256、partial interleaved RoPE 64、Gated DeltaNet K16/V48×128 及
MTP 10240；每个 prepared artifact 以变化输入重放，越界 embedding 不发布结果。所有调用与 Graph
report 只绑定固定 ggml Vulkan managed region 或限定 IREE Vulkan program，没有增加 HYLE 数学、
fallback、热选择或产品实现。

MSVC 19.50 Release/WERROR selector 的 docs、closure、configure、Vulkan Direct 与 Vulkan Graph 五个
gate 全部 PASS，127 个无关 gate 跳过或复用；Vulkan SDK 1.4.357.0、NVIDIA driver 610.62，完整
Direct owner 86.70 秒、Graph owner 70.37 秒。完整收据见
[Qwen3.8 primary GGUF Vulkan complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen38-primary-gguf-complete/README.md)。
ledger 现有 40 个 `RUN_PASS`、136 个 `VALIDATION_NOT_RUN`；Qwen3.8-27B primary GGUF 已完成四路线
八格，D5-A5 三交付和 D5-A6 P1 仍由稳定路线批次统一执行。

### Qwen3.8 fidelity GGUF CPU Direct/Graph 完整格

不可变 tests-only candidate `5bbecf49865ffbe638d9639ac0fb051ece539e8c` 已通过普通 CPU
Direct/Graph 入口完成固定 `qwen38_27b:fidelity_gguf:UD-Q8_K_XL` profile 的
R01-R05、R13-R15。固定 header 为 1 shard、866 tensors、27,320,697,856 logical parameters；
embedding、dense FFN、GDN projection 与 attention output 为 Q8_0，full-attention Q/K/V、LM head
与 MTP projection 为 BF16，norm、conv 与 recurrent state 为 F32。测试没有从制品档位推断全局 dtype。

R01 运行精确 5120 宽 Q8_0 embedding；投影批次覆盖 FFN up/down、融合 GDN QKV、attention output、
SSM alpha/output 的真实 Q8_0 角色和宽度；语义链保持 FFN=17408、GQA 24/4×256、partial
interleaved RoPE 64、Gated DeltaNet K16/V48×128 与 MTP 10240。每个 prepared artifact 使用变化输入
重放，越界 embedding 不发布结果；所有 Direct/Graph report 只接受固定 ggml CPU 或 oneDNN 成熟实现，
没有新增 HYLE 数学、fallback、热选择或产品代码。

MSVC 19.50 Release/WERROR selector 的 docs、closure、模型输入工具、configure、CPU Direct 与 CPU
Graph 六个 gate 全部 PASS，127 个无关 gate 跳过或复用；完整 Direct owner 66.90 秒，Graph 与 executor
67.12 秒。完整收据见
[Qwen3.8 fidelity GGUF CPU complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-qwen38-fidelity-gguf-complete/README.md)。
ledger 现有 42 个 `RUN_PASS`、134 个 `VALIDATION_NOT_RUN`；其余 CUDA、Apple、Vulkan 六格以及
D5-A5 三交付和 D5-A6 P1 仍待对应路线批次完成。

### Qwen3.8 fidelity GGUF CUDA Direct/Graph 完整格

不可变 tests-only candidate `e1ecacc3bd7c27c461b8cc46665180aeeacfd584` 已通过普通 CUDA
Direct/Graph 入口完成固定 `qwen38_27b:fidelity_gguf:UD-Q8_K_XL` profile 的
R01-R05、R13-R15。固定 header 为 1 shard、866 tensors、27,320,697,856 logical parameters；
embedding、dense FFN、GDN projection 与 attention output 为 Q8_0，full-attention Q/K/V、LM head
与 MTP projection 为 BF16，norm、conv 与 recurrent state 为 F32；测试按真实 tensor role 路由，
没有从制品档位推断全局 dtype。

R01 运行精确 5120 宽 Q8_0 embedding；投影批次覆盖 FFN up/down、融合 GDN QKV、attention output、
SSM alpha/output 的真实 Q8_0 角色和宽度；语义链保持 FFN=17408、GQA 24/4×256、partial
interleaved RoPE 64、Gated DeltaNet K16/V48×128 与 MTP 10240。每个 prepared artifact 使用变化输入
重放，越界 embedding 不发布结果；所有 Direct/Graph report 只接受固定 ggml CUDA、cuBLASLt、
SGLang/CCCL 或 FlashInfer 成熟实现，没有新增 HYLE 数学、fallback、热选择或产品代码。

MSVC 19.50 Release/WERROR selector 的 docs、closure、configure、CUDA Direct 与 CUDA Graph 五个 gate
全部 PASS，127 个无关 gate 跳过或复用；CUDA Toolkit 13.3.73、`CUDAARCHS=89-real`，完整 Direct
owner 66.53 秒、Graph owner 68.66 秒。完整收据见
[Qwen3.8 fidelity GGUF CUDA complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen38-fidelity-gguf-complete/README.md)。
ledger 现有 44 个 `RUN_PASS`、132 个 `VALIDATION_NOT_RUN`；其余 Apple、Vulkan 四格以及 D5-A5
三交付和 D5-A6 P1 仍待对应路线批次完成。

### Qwen3.8 fidelity GGUF Apple Direct/Graph 完整格

不可变 tests-only candidate `1544c98061f36a8a7e5ea67b5db00a5f47bd8872` 已在正式 MacBook Air M4
的普通 Apple Direct/Graph 入口完成固定 `qwen38_27b:fidelity_gguf:UD-Q8_K_XL` profile 的
R01-R05、R13-R15。固定 header 为 1 shard、866 tensors、27,320,697,856 logical parameters；
embedding、dense FFN、GDN projection 与 attention output 为 Q8_0，full-attention Q/K/V、LM head
与 MTP projection 为 BF16，norm、conv 与 recurrent state 为 F32；测试按真实 tensor role 路由，
没有从制品档位推断全局 dtype。

R01 运行精确 5120 宽 Q8_0 embedding；投影批次覆盖 FFN up/down、融合 GDN QKV、attention output、
SSM alpha/output 的真实 Q8_0 角色和宽度；语义链保持 FFN=17408、GQA 24/4×256、partial
interleaved RoPE 64、Gated DeltaNet K16/V48×128 与 MTP 10240。每个 prepared artifact 使用变化输入
重放，越界 embedding 不发布结果；量化调用固定绑定 ggml Metal，其余语义片段固定绑定 MLX Metal，
没有新增 HYLE 数学、fallback、热选择或产品代码。

AppleClang 21 Release/WERROR selector 的 docs、closure、configure、Apple Direct 与 Apple Graph 五个 gate
全部 PASS，127 个无关 gate 跳过或复用；MLX 0.32.2、ggml Metal 0.24.0、macOS SDK 27.0，完整
Direct owner 39.91 秒、Graph owner 37.79 秒。完整收据见
[Qwen3.8 fidelity GGUF Apple complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen38-fidelity-gguf-complete/README.md)。
ledger 现有 46 个 `RUN_PASS`、130 个 `VALIDATION_NOT_RUN`；剩余 Vulkan 两格以及 D5-A5 三交付和
D5-A6 P1 仍待对应路线批次完成。

### Qwen3.8 fidelity GGUF Vulkan Direct/Graph 完整格

不可变 tests-only candidate `a11d9667683795950960323627532629e2d7726e` 已在正式 Windows
RTX 4070 的普通 Vulkan Direct/Graph 入口完成固定
`qwen38_27b:fidelity_gguf:UD-Q8_K_XL` profile 的 R01-R05、R13-R15。固定 header 为 1 shard、
866 tensors、27,320,697,856 logical parameters；embedding、dense FFN、GDN projection 与
attention output 为 Q8_0，full-attention Q/K/V、LM head 与 MTP projection 为 BF16，norm、conv
与 recurrent state 为 F32；测试按真实 tensor role 路由，没有从制品档位推断全局 dtype。

R01 运行精确 5120 宽 Q8_0 embedding；投影批次覆盖 FFN up/down、融合 GDN QKV、attention
output、SSM alpha/output 的真实 Q8_0 角色和宽度；语义链保持 FFN=17408、GQA 24/4×256、
partial interleaved RoPE 64、Gated DeltaNet K16/V48×128 与 MTP 10240。每个 prepared artifact
使用变化输入重放，越界 embedding 不发布结果；量化调用固定绑定 ggml Vulkan，其余语义片段固定
绑定成熟 ggml Vulkan 或已准入 IREE Vulkan 程序，没有新增 HYLE 数学、fallback、热选择或产品代码。

MSVC 19.50 Release/WERROR selector 的 docs、closure、configure、Vulkan Direct 与 Vulkan Graph
五个 gate 全部 PASS，127 个无关 gate 跳过或复用；Vulkan SDK 1.4.357.0、NVIDIA driver 610.62，
完整 Direct owner 90.31 秒、Graph owner 75.23 秒。完整收据见
[Qwen3.8 fidelity GGUF Vulkan complete](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen38-fidelity-gguf-complete/README.md)。
ledger 现有 48 个 `RUN_PASS`、128 个 `VALIDATION_NOT_RUN`；该 profile 四路线八格已全部闭环；
D5-A5 三交付和 D5-A6 P1 仍随稳定路线批次执行。

### CUDA Qwen3.5 official Direct/Graph 完整格

不可变 candidate `9917438bb44ebc29f647f2b0ce2919a73342fa3e` 复用同一真实 checkpoint fixture，
通过普通 CUDA Direct `prepare`/submit 和 Graph compile/bind/replay 覆盖 R01-R05、R13-R15。每个
prepared call/graph 使用变化输入至少执行两次，成熟 Provider 在冷准备时固定；R02 SwiGLU 与 R04
RoPE 不再落入冻结的 `hyle-cuda`。CUDA 首次执行允许固定上游建立一次 graph metadata，稳定性从第一轮
warm replay 到第二轮 replay 验证；失败路径不发布输出。完整收据见
[CUDA Qwen3.5 official Direct](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-official-direct/README.md)
和
[CUDA Qwen3.5 official Graph](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-qwen35-official-graph/README.md)。
ledger 现有 4 个 `RUN_PASS`、172 个 `VALIDATION_NOT_RUN`。本批静态/共享 owner suite 不代签 D5-A5
要求的 relocated external source/static/shared consumer；D5-A5 与 D5-A6 P1 仍随稳定 CUDA 路线批次执行。

### Vulkan Qwen3.5 official Direct/Graph 完整格

不可变 candidate `1fa67a8facf76feb78f66b5576aee45ac7d16d99` 复用同一真实 checkpoint
fixture，通过普通 Vulkan Direct `prepare`/bind/submit 和 Graph build/compile/bind/submit 覆盖
R01-R05、R13-R15。embedding/head、FFN、GQA/RoPE/attention、causal conv/GDN 及 MTP/BF16
边界均使用精确官方 shape 和真实权重；每个 prepared program 以变化输入、token 或 state 建立独立
invocation 重放，失败 Gather 不发布输出。Direct 冷准备逐项断言固定 ggml Vulkan managed region 或限定
IREE dispatch，Graph 冷编译逐项断言成熟 managed region；没有回接冻结的 HYLE 自研数学。

不可变 W1-WIN/MSVC 19.50/WERROR selector 的 docs、closure、configure、完整
`compute_route_vulkan` 与 `graph_route_vulkan` factor 均为 PASS；定向 Direct 6/6、Graph 5/5 也通过。
完整收据见
[Vulkan Qwen3.5 official Direct](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-official-direct/README.md)
和
[Vulkan Qwen3.5 official Graph](../../../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-qwen35-official-graph/README.md)。
ledger 现有 6 个 `RUN_PASS`、170 个 `VALIDATION_NOT_RUN`。本批共享 WERROR owner suite 不代签
D5-A5 要求的 relocated external source/static/shared consumer；D5-A5 与 D5-A6 P1 仍随稳定 Vulkan
路线批次执行。

### Apple Qwen3.5 official Direct/Graph 完整格

不可变 tests-only candidate `e99c2bf929bbee747dbb8a392ab3ff813299be3f` 在 formal Air
`Mac16,12` 上复用同一真实 checkpoint fixture，通过普通 Apple Direct `prepare`/bind/submit 和
Graph build/compile/bind/submit 覆盖 R01-R05、R13-R15。MLX 0.32.2 保留原生 Metal region，
Gated DeltaNet 保留完整 ggml Metal region；Graph compile report 逐项确认成熟实现策略，Program 持有
捕获常量。变化输入/token/state、独立 invocation、失败不发布和 BF16/F32 边界均实际运行。

定向 Direct 5/5、Graph 5/5，完整 Direct owner 52/52、Graph owner 25/25 在 AppleClang 21、WERROR
下通过；Windows 不可变 selector 的 docs、closure 与 Apple-OFF exact-factor configure 也通过。完整收据见
[Apple Qwen3.5 official Direct](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-official-direct/README.md)
和
[Apple Qwen3.5 official Graph](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-qwen35-official-graph/README.md)。
ledger 现有 8 个 `RUN_PASS`、168 个 `VALIDATION_NOT_RUN`。已有 R03 三交付证明继续有效，但不提前代签
D5-A5 的最终稳定 Apple 路线批次；D5-A6 P1 也仍未运行。

## 验收合同

1. `RUN_PASS` 必须来自不可变 candidate 上的普通 Direct 或 Graph 入口，记录 exact profile、region、shape、
   provider/software/device identity 和输出 oracle；source symbol、target、枚举或 prepare-only 均不算。
2. 同一 prepared artifact 至少 submit 两次；第二次必须改变 activation 或 state，证明没有 stale output，且
   submit 不重新选路、计时、解码或分配持久 workspace。
3. `BACKEND_REJECTED` 需要精确 route/device/provider/version/format/shape 的 prepare/capability 失败；
   `UPSTREAM_MISSING` 需要固定成熟源码和 provider inventory 的具名缺口。不得保留模糊 `NOT_RUN`。
4. 本地 Qwen 实际 bytes 只作为测试输入复制到验证 scratch；不提交模型数据，不让远程副本覆盖 Windows
   产品源码。其它巨型模型使用固定 header/config 与格式合法的非零 payload，不冒充实际 checkpoint bytes。
5. region 可以由 Provider 融合承载，也可以由 Graph 组合成熟节点；HYLE 只检查边界、路由、状态和结果，
   不为“看见每个算子”拆坏上游融合。

## 预声明验证

- 文档/ledger 先运行 closure、docs 与 JSON self-check，不构建、不跑 P1。
- 测试 consumer 稳定后按 selector 选择 CPU、CUDA、Apple、Vulkan 的直接目标；重型路线串行。
- 候选固定后才运行三交付与 P1。未运行项保留 `VALIDATION_NOT_RUN`，绝不提前写 PASS。
- D6 只汇总红队、最终矩阵、限制、LOC 和状态，不在终审首次接线。

## AUDITED

八个模型批次、22 个 profile、真实宽度、176 个运行 cell、六项有限动作、机器、P1 与停止条件已经冻结。
D5 从 A1/A2 开始批量实现；新增模型、profile、Provider 或自研数学均须退回重新定界。
