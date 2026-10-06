# C4：CPU 新版主干、MTP 与 DSpark 组合接线

<!-- MODULE REVIEW: AUDITED -->

## 计划身份

- 状态：AUDITED，真实新版组合消费已完成有限源码/角色审计；compute_ops/ggml/native从C3交接，保持原状态。
  没有新数学核授权或执行PASS；C3原证据/未决画像资格保留，不假写整包CLOSED。
- baseline：`2d1d787396f1f2798fe14c7ff3cece9bbebbbbda`；Windows实际HEAD/worktree已核对，保留全部后续提交。
- owner范围：compute_provider_ggml、compute_ops；原ATen/借用计算涉及compute_provider_native；CPU Direct/Graph facade按实际产品变化绑定。
- 总合同：[CPU扩覆盖计划](cpu-quant-mtp-coverage-expansion-plan.md)；事实：[缺口清单](cpu-popular-quant-mtp-gap-list.md)。
- 本卡是一个批次，不是逐算子/逐模型队列。CLOSED按真实trigger新开；AUDITED范围变化重新定界。
- 性能：P0为主，唯一可选完整程序P1只按总计划条件固定；无自研竞赛或新数学。

## 范围、真实消费者与来源

- scope：DSP01输入/target hidden投影、DSP02带sink attention/cache、DSP03 Markov head、DSP04 F32confidence head。
- B01 Qwen Q4_0 MTP、B02/03 Pro-0813/Flash-0731、B07完整DSpark、B08 Q8_0/BF16sidecar属于同批组合/资格。
- 原数学来自固定官方forward/成熟库；目标hidden、shared embedding/head、cache/indices/state/精度端口显式。
- Pro/Flash不同shape/config分别验收；mtp.*前缀不能混同旧NextN；不能删confidence/activation quant以通过。
- DSpark数据输入和logits/confidence/状态输出属于计算；token选择、推测接受策略与服务不下沉。

## 待审计的有限输出

以下是计划输出，激活时须完成源码/生命周期审计并冻结实际FIX；不以本表假写AUDITED。

| ID | 本包动作 | 结束条件 |
|---|---|---|
| B1 | 复用成熟原语/完整region批量连接四片段和主干，共享参数在prepare绑定 | 源码公式与真实角色/混合格式、F32状态/信心边界正确；无新自研数学 |
| B2 | 完整MTP/DSpark、独立sidecar与主制品版本/共享权重关系 | 同公式参考、完整词表/内宽/原block，cache反馈/wrap/reset、changed replay符合原程序 |
| B3 | 生命周期、独立slots、OOM/失败/close及普通Direct/Graph入口 | Provider可保留融合，HYLE不分拆成逐kernel；已接纳失败不重新裁决 |
| B4 | P0完整成本与必要的代表性接入开销门，提前固定适用性 | 无重复热packing/解码/编译；真有未决风险才按总计划一份P1预算，不比较自研赢家 |

## 验证、预算与停止

- 四片段共用一份source/program/资格表，不按每个小node建候选或全量验收；最多一批定向纠正。
- 本地Qwen checkpoint复用；巨模型保持内宽/状态/block/词表，以受header约束的合法非零payload和缩小batch/token/驻留专家验算。
- 包末普通入口、资源与两平台三交付须实际完成；C5仅复用汇总，不最后首次接线。
- 新数学/Graph公共机制缺口立即定向定界；graph旧Vulkan BLOCKED不自动解除，不借本包重写Graph core。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。

## 用户最新性能口径

单一合格成熟carrier完成正确接入与P0；多家成熟库具备同合同重叠域时按总计划批量比较固定默认规则。
不比较自研，不为每个算子/档位重建或计时，最多两个代表消费族共用一份稳定候选P1。

## 本轮有限审计边界

- 依据C1完整header/index与固定官方`model.py`；本轮不再查其它版本或扩大模型范围。
- Pro H7168/R512、Flash H4096/R256，均为vocab129280、HC4、window128、draft block5。
  main_proj分别[7168,21504]/[4096,12288]，原FP8/E8M0 block128；confidence源BF16而F32执行。
- 实际checkpoint有mtp.0/1/2三阶段、head在mtp.2；config中旧num_nextn_predict_layers=1不能替代
  DSpark实际阶段。三条末尾compress ratio0和官方DSpark类与实际tensor互相对账。
- 固定有限输出：新版精确profile与实际角色绑定；target hidden归并/main projection/norm；
  target KV prefill/wrap与draft五token sink attention；Markov全词表与confidence；完整HC/MoE/head；
  Qwen sidecar实际Q3_K/Q4_K/Q6_K与Flash Q8_0/BF16 sidecar消费。
- 可复用既有成熟MatMul/Gather/Concat/Cast/Reduce/Scatter/RMSNorm/RoPE/sink attention、
  FP8/FP4原程序与共享Fragment消费形状；仅测试组合不伪造产品reopen。
  若发现公开语义或成熟承载真缺口，须先固定具名FIX并绑定owner，不能写新的数学核补洞。
- 缩小token/batch/驻留专家与头部常量分片；保留实际hidden/latent/head/rank/block/词表跨度。
  采样/接受策略留给consumer，head输入显式提供前一token，不省略Markov偏置或confidence。

## 固定 FIX 与承载裁决

| FIX | 真实差异 | 本批动作 | 结束证据 |
|---|---|---|---|
| F1 | 旧preview profile/NextN不能描述新版 | 固定新版revision/config/header身份、3-stage DSpark及sidecar实际角色；不扩公共路由器 | 精确身份和原dtype/shape对账 |
| F2 | 缺target三层hidden与draft输入消费 | 成熟Reduce/Concat、原FP8 projection/RMSNorm、BF16 embedding及HC扩展 | 原公式非零数值、实际内宽、共享参数 |
| F3 | 旧单token滑窗不能代签draft block | 原target KV/RoPE/group64 roundtrip、独立环cache、5-token稀疏sink attention与逆RoPE/分组output | prefill超过window、wrap/reset/实际反馈、完整attention输出 |
| F4 | 缺Markov/confidence及完整新版branch | BF16 embedding→F32全词表head、logit bias与BF16-source/F32 confidence；原HC/MoE/共享head；真实Qwen/Flash sidecar | 四片段与完整branch，两入口、状态/资源/三交付 |

已有成熟语义能表达上述程序；初批仅接组合消费者与资格，没有已证明必须新增的产品数学或公共API。
若实际执行暴露接缝缺陷，只在这些FIX所列边界内定向修复；超出范围先重新审计，不重开自研竞赛。
Provider可执行原完整operation/region，Graph负责边界；测试Direct展开仅为自然Compute消费，不规定产品逐kernel调度。
P0记录实际cold scale widening、prepare绑定、hot原量化/selected-expert和managed临时成本。
无第二家同合同承载的格不比赛；有具名重叠或接入成本风险才纳入总计划最多两个完整消费族的一份P1。
