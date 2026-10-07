# CUDA R1 当前官方 DSpark：第一批真实链路

当前范围：[R1卡](../../../docs/implementation/tasks/compute/cuda-final-u3-model-regions.md)。
Windows canonical，WSL仅前向验证；原输入来自当前Pro0813／Flash0731 common program，
没有以旧preview、单次MatMul或CPU PASS代签。

## 已实现与已验证

- 现成熟CUDA tensor runtime接原stable Softplus，替代这组program误落的旧自研Unary。
  SDK、external stream、metadata、异常与Completion复用，不写新数学。
- plain BF16 GroupedMatMul的原MMF会把F32 activation cast到BF16。现在lowering用原ggml cast
  把weights变F32再交原MUL_MAT_ID，保留公开F32乘数；quantized weight/A8路径不改。
  转换的O(E×N×K)临时空间与热搬运完整计费，不把F32 accumulation当成乘数precision证明。
- CUDA runner按真实prefill/decode phase绑定，并给既有source validator传实际output port。
- current target-hidden／F32 confidence、完整Markov vocabulary、prefill/ring-cache／五token
  attention、当前主干ratio0/4/128 attention/compression与MoE、完整三stage DSpark／共享embedding／
  Markov／confidence／完整head，都在普通Direct／Graph实际运行通过。
- Linux `3aff1ac2` 同候选一次批量：5 GTests×两入口，2 CTests通过，无skip／zero-filter；
  Windows `380c5dcc`同数学的前四类与完整three-stage通过，cache最终test glue `3aff1ac2`通过。
  复用只限test-only变化不影响的相同数学/输入/机器证据。测试仍保留原内宽，完整head分片，
  不称全部大模型权重常驻或提供推理engine。
- 全Compute+Graph非测试实现 **60,634行／253文件**，统一脚本自检通过。

## 精度反例、测试输入和原失败

完整原失败／构建／source诊断／纠正日志在 [批次归档](native-batch.zip)，精确范围见
[机器摘要](receipt.json)。首次Native Unary断言暴露成熟Softplus接缝；BF16 MMF源码证明
F32 RHS会转低精度，不能靠precision flag名字签保留F32 activation。

cache失败还包含独立问题：prefill和下一draft的参考要从相同actual cache开始。
修正后剩余projection相消反例仍存在，具名逐项诊断显示：

| 项 | 实际public输入 | 独立参考输入 |
|---|---:|---:|
| source weight 0.0625对应inverse | -0.2138671875 | -0.2138671875 |
| source weight 0.03125对应inverse | 0.47265625 | 0.470703125 |
| 独立source dot | 0.00140380859375 | 0.0013427734375 |

GPU输出为0.00140380859375，与独立方程对actual输入完全一致。inverse已受现有BF16上游gate
和原inverse equation validator约束；两个相消项使点值相对误差放大，单个末端relative gate
不能替代输入误差传播证明。现只扩展这组GPU consumer的既有validator，对grouped projection
逐完整row使用独立source weight bank与actual inverse计算，再核对原BF16边界。
cache／attention／state／最终输出全链gate保持，未修改全局tolerance、CPU fixture或产品误差合同。
打印诊断已从当前test移除，原日志保留；没有把test oracle问题假报成库性能FAIL。

## 未完成

三份quantized sidecar的原GPU codec oracle、其余49 family-region关联、全部fixed基础格式/
precision、最终U3 source/static/shared/CUDA-only/mixed/offline/relocation仍须集中验收。
U4条件default/best-of与全成本比较、U5自研清退、U6复核未由本批代签。
本批P0，无P1／SOTA赢家，当前owner仍AUDITED，goal active。
