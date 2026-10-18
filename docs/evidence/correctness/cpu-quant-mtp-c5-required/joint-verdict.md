# C5 联合合同裁决

产品候选 `c740abc74a5a62ffacf60b26798228ca484317df`。本裁决的对象是HYLE基础计算/图执行的
**全部C1准入CPU计算合同**，不是模型loader、token策略或104次整模型实权重推理。
没有从原范围删除制品、格式或必需region；没有把接口可表达、准备成功或metadata改成数学PASS。

## 复用必须同时成立的条件

1. 每个制品的固定revision/header、真实角色、dtype/physical block/group、实际K/N和state来自完整C1表。
2. 该制品全部不同numeric端口合同必须在普通Direct/Graph实际输出/changed replay通过；
   indexed端口另有越界/写保护/退役/恢复。589去重合同中的559 numeric、34 packed和45 plain已逐ID对齐XML。
3. 参数角色与模型数学分别核对：80 GGUF role/shape复核、非GGUF原低dtype/zero-centered norm/conv，
   不用文件Q标签、另一dtype、默认epsilon或交错RoPE代签。
4. 共享semantic region必须同模型族、操作序列、实际内宽/head/latent/state、算法、边界精度和布局；
   其普通两入口完整计算及实际state反馈已运行。它只继承不依赖矩阵编码的数学/状态合同。
5. 变化的矩阵编码必须继承其本身已经运行的原codec/activation/accumulation合同；
   不能用F32权重或另一format的胜负替代。边界匹配以TensorDesc和实际公开操作为准。
6. HYLE准备/Graph lowering对原库程序按operation/typed端口绑定，不按artifact字符串修改数学；
   当前ggml/ATen/region owning源码核对无模型名特化执行分支。旧Native187规则要求原peer，CPU-only新core不启用。
7. 产品/core/patch/runtime/lifetime与安装身份匹配。新增大table与ID9 conv各有重装SDK资格，
   其它程序原数学/ID/资源机制未变按Git精确差异复用。所有适用资源/交付失败都实际运行，不写未验PASS。

上述是按计划去重后逐合同验收的证明链：**实际端口 + 相同边界的完整片段 + 精确身份 + 生命周期 + 三交付**。
重复复制同族同state图、只替换format标签不能增强证明，也不能掩盖新format未经资格。
`LIBRARY_CONTRACT_PASS`表示这一证明链全部成立，区别于旧账本的整画像`RUN_PASS`。

## 准入对象的联合结果

| 对象 | 真实差异的运行证据 | 完整semantic/state与作用域 | 结果 |
|---|---|---|---|
| Qwen3.5-2B 22 GGUF档位 | 全部实际GGUF端口/原codec，F32参数 | 实际Q8/KV2/D256、HalfSplit64/theta1e7/eps1e-6、GDN/state；FFN原SiLU与残差 | LIBRARY_CONTRACT_PASS |
| Qwen3.8-27B 25 GGUF档位 | 全部实际端口及MTP独立角色 | Q24/KV4/GDN48真实state；完整Q4命名sidecar decoder/cache/carry/full V及各format端口合同 | LIBRARY_CONTRACT_PASS |
| Qwen3.8-2.4T 9 GGUF档位 | 全部实际端口，原IQ1/2等算法、F32 router/参数 | Q64/KV4/GDN128；完整NextN original混合角色、routed/shared MoE与全V、实际KV/carry | LIBRARY_CONTRACT_PASS |
| Kimi K3 9 GGUF档位 | 全部实际head-group/latent/expert/lookup端口 | KDA96/D128、SiTU、MLA、latent MoE、AttnRes bank/state，原norm epsilon和a/bias广播 | LIBRARY_CONTRACT_PASS |
| GLM-5.3 12 GGUF档位 | 全部实际MatMul/expert/typed参数 | MLA/rank2048、DSA selection/cache、shared index状态、完整NextN/full head，原LayerNorm | LIBRARY_CONTRACT_PASS |
| GLM-5.3-Flash 12 GGUF档位 | 全部实际编码/层不同format，F32/BF16原边界 | KDA64、MLA/rank1536、pooled indexer/ragged tail、mHC、完整NextN与index/carry/reset | LIBRARY_CONTRACT_PASS |
| Pro-0813 2 GGUF档位 | 真实MXFP4 experts、BF16 router、Q8/BF16 projections及全部端口 | 原F32 GGUF ratio0/4/128 attention/compressor/indexer、ordinary/hash MoE、mHC输出入全V head | LIBRARY_CONTRACT_PASS |
| Flash-0731 13 GGUF档位 | 全部实际量化端口含IQ1扩展、F32参数 | 同Flash实际H4096/rank1024/state，原GGUF程序；encoded格式差异各自原codec资格 | LIBRARY_CONTRACT_PASS |
| Pro-0813/Flash-0731 两官方制品 | 原FP4/FP8/E8M0/BF16/F32组合，A8及cold wo_a准备 | 新主干ratio0/4/128、四DSpark片段、完整三stage HC/attention/MoE/shared-head/Markov/confidence、cache wrap/reset/反馈 | LIBRARY_CONTRACT_PASS |
| Qwen27B Q4_0 MTP、Flash Q8/BF16 DSpark 三sidecar | 完整18/81/81 tensors及原Q3/Q4/Q6或MXFP4/BF16/Q8实际角色 | 产品embedding/prefill→完整decoder/stages→全head与实际cache/carry/Markov/confidence，已两系统两入口 | LIBRARY_CONTRACT_PASS |
| NF4 nested / AWQ / CT / MLX 四制品 | 34原packed合同＋45原BF16/F16/F32/I32未压缩text合同；原precision/group/packing | parent模型所需相同semantic/state数学与原低dtype边界联合；CT保留未量化MTP，MLX无分支权重不制造DSpark；explicit cold decode与selected-group成本保留 | LIBRARY_CONTRACT_PASS |

逐对象revision/digest、numeric/parameter/region关联和运行位置来自
[aggregate audit](aggregate-audit.json)、[family index](family-witness-index.json)、[80参数复核](../cpu-quant-mtp-c5-parameters/contract-review.md)。
表中档位数量必须与C1机器表一致，任何新release或不同g_idx/packing/quant-state不自动继承。

## 资源、三交付和成本

- 原ggml/ATen拥有compiled program、独立invocation、原workspace/线程/完成；HYLE边界
  保活readonly参数并管理accepted failure/slot退役/新slot恢复/close。真实OOM、并发和metadata回收已有两系统收据。
- 源码完整消费、安装static/shared自然C++、DLL邻exe、SDK搬移、锁定资产/现有cache offline均有资格。
  static HYLE仍有外部shared LibTorch与私有ggml CPU program，不宣称无依赖单文件SDK。
- P0查原codec、必要cold cast/decode、selected专家/K group、布局、同步、热临时与copy。
  ATen managed私有内存unknown，严格unknown拒绝/捕获域不承诺；图失败不回滚之前已完成输出。
  新接缝没有第二个合格同合同carrier或新winner，不启动逐format P1/自研竞赛，不签SOTA。

## 最终范围限制

Windows/Linux x86_64 CPU-only新core/锁定SDK域；非实权重payload保留真实内宽/格式/state，
允许缩小token/context/驻留expert。全V head可分片，text MRoPE三轴相同；不签视觉、采样/接受策略、
整模型驻留、性能吞吐或新GPU资格。所有原GPU拒绝/FAIL/旧Graph Vulkan P1 BLOCKED均保留。

结论：全部原准入计算合同的联合资格成立；技术反证没有未决产品缺陷。
本结论不将104对象改名为104次完整混合模型运行；最终模块状态仍须通过七门记录与状态协议裁决。
