# D5-F33：DeepSeek 原生 sink attention 接缝

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- baseline：`04b4eb3e`，Windows canonical；保留已提交、未运行的F32查表路由候选。
- owner：当前 AUDITED 的 `compute_ops`、`compute_provider_ggml`、
  `compute_provider_ggml_cuda`、`compute_provider_ggml_vulkan`、`compute_route_apple`、
  `graph_route_apple`。本卡由当前D5卡纳入；不重开其它CLOSED模块。
- trigger：[真实角色审计](llm-moe-d5-deepseek-remaining-audit.md)的每头F32 sink，
  以及 llama.cpp `4c9233c0` 的 raw/CSA/HCA attention，均需要显式mask和sink共同参与归一化。
  旧三输入SDPA没有sink端口，不能代签。
- scope：一个完整原生attention描述；CPU/CUDA/Vulkan绑定锁定ggml的矩阵乘与
  `ggml_soft_max_ext`/`ggml_soft_max_add_sinks`原生图，Apple绑定锁定MLX的完整
  `fast::scaled_dot_product_attention(..., mask, sinks, ...)`。不拆成HYLE调度的逐数学kernel。
- non-goals：新数学核、替换通用SDPA、扩大模型或dtype目录、变更第三方版本、
  FP8假转换、单独重建全路线、重写公共验证工具或先跑P1。
- dependencies：ggml `456172ec733a135778adcd32d00e576a58232e45`；
  MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85`。本地固定Git blob已确认接口和归一化语义。
- budget：一次共同接线和测试批次，随后与DeepSeek压缩状态/查表MoE合批冻结验证；
  具体失败最多一轮定向修正。不以单独attention通过增加完整画像格。
- stop：四路线普通Direct/Graph完整片段通过，或形成可复现的精确原生域拒绝。
  三交付和适用性能随D5-A5/A6；本卡不提前宣称它们PASS。

## 固定语义与边界

`model_ops::SinkAttention{scale}` 输入依次为F32 plain dense：
Q `[1,T,H,512]`，K/V `[1,S,1,512]`，additive mask `[T,S]`，sink `[H]`；
H=64或128，T/S非空，scale有限且为正，输出与Q同描述。K与V可引用同一只读输入；
输出不别名。显式mask已经包含窗口、因果关系和CSA/HCA选择；实现不额外叠加causal。

每个query/head计算 `scores = scale * QK + mask`，分母是
`sum(exp(scores)) + exp(sink[head])`，输出只累加真实V；sink没有V贡献，也不被mask屏蔽。
实际计算采用成熟实现的稳定归一化。有限Q/K/V、有限sink、有限或负无穷mask为数值域；
全部key被屏蔽时有限sink应使输出为零。NaN及正无穷输入不承诺有效数值结果。
mask和sink是普通运行输入，可在同一准备程序中改变，不能烘焙成准备期常量。

这是DeepSeek真实F32计算片段，不把官方FP8权重投影、F16 KV cache或完整模型算作已资格化。
压缩器、RoPE/YaRN、状态反馈和输出分组投影仍在共享consumer中接合。

| FIX | owner | 有限动作与适用验证 |
|---|---|---|
| F33.1 | compute_ops | 完整描述及shape/type/scale/端口/中间分数表示范围检查；schema正负测试 |
| F33.2 | CPU/CUDA/Vulkan ggml Providers | 原生sink/mask接线；复用prepare、arena、workspace、原生失败和Completion；保留其它SDPA域 |
| F33.3 | Apple Direct/Graph | 默认MLX完整调用与Graph冷编译准入；保留原生stream/资源及失败路径 |
| F33.4 | 共享consumer | 四画像的64/128头、D512、变化Q/K/V/mask/sink，多token及非整齐长度；独立稳定oracle、全部masked和大sink；接回实际CSA/HCA下游 |

P0：没有新增热搜索、host数学、KV往返或外部线程。ggml的QK分数和V转置由准备程序拥有，
成本含必要转换与中间资源，不能把F32原生分解宣称为FlashAttention。Apple由MLX决定其
原生fused/fallback程序，HYLE不拆解它。资源失败沿已有路线报告，不隐藏原生不支持。
批末selected owner/Direct/Graph/schema/头文件、资源失败/重放及三交付门统一消费本范围。

## 当前结论

AUDITED；产品和consumer尚未接线，native验收未运行。完整画像仍为128/176。
