# CPU成熟接线：Transformer与矩阵剩余域

<!-- MODULE REVIEW: SCOPED -->

- owner `compute_provider_native`；baseline `7a6d35353626e332d7d0a1f7d984c6cabebea7a9`，Windows canonical。
  CPU/Graph assembler、oneDNN、ggml产品不改；本轮只在已有ATen独立factory接入原库程序。
- trigger：全族审计仍登记Bool/log-softmax、I64/其它axis RoPE、mask SDPA与更高rank/stride MatMul
  成熟候选，当前ATen factory尚未准入这些family；不能据其未接给Native签唯一或WIN。
- 主消费：已有目标模型的带mask/LogSoftmax、长位置RoPE/GQA attention与rank/broadcast projection；
  正交消费：public rank0/8/empty、合法odd byte stride、±Inf/NaN/零、alias和资源/完成边界。
- scope：已公开MatMul plain F32/F16/BF16、显式F32 accumulation/现有合法Unknown档，任意合法rank
  与byte stride/dense output/broadcast/K0；Softmax三dtype任意axis/scale、Bool非零/浮点additive广播、
  log与无有效行NaN；RoPE F32 data、I32/I64 positions、F32 frequency、全部既有prefix/axis/layout与
  未旋转tail原bits；SDPA三dtype BSHD/BHSD、GQA/Dk-Dv、Bool/additive、bottom-right causal、空keys
  的既有数值合同。RMSNorm/LayerNorm未兼容布局候选同批核对，未证不扩大资格。
- 原程序：已锁LibTorch2.8 CPU的matmul/softmax/log_softmax、sin/cos/stack/cat、attention math/原
  Tensor composition候选。只允许已有语义packing/masks与原库kernel组合；不写HYLE payload数学。
  原SDPA不同all-masked/causal定义必须实证，不以函数同名冒充公共合同；需要原程序语义适配计全成本。
- 资源：复用私有native packing、checked mappings/alias/真实Completion/OOM/close，managed/unknown
  原库内存与线程不隐瞒；不承诺capture或零热分配。位/索引边界检查不是新数学核。
- 有限审计TM1–4：同一批原矩阵/softmax、位置与mask witness；固定精确准入/数值/资源与有限FIX；
  批量产品与meaningful tests；同一普通Direct/Graph两平台三交付，默认身份和真实最终输出。
- tier P0：包含全部F32/Double转换、packing、GQA展开/score/mask、axis/materialization与原临时
  Tensor；保留现有有效成熟region，不以新leaf拆更快region。无Native/SIMD优化/性能比较/清退。
- budget：一批原SDK witness覆盖上述四族，最多一轮具名原程序纠正；一次有限FIX后批量产品/编译，
  最多一次具体产品纠正。j2重型串行，Windows DLL邻exe/隐藏启动；只适用factor/消费者、header/三
  交付/selector/closure/docs/diff与统一LOC，不重跑未改模型/旧TC/NX/GPU/P1，不扩建仪器。
- stop：不能承载具体域保存具名DEFER/拒绝，不能写自研替代或签UNIQUE；预算后按协议裁决。
  当前SCOPED，原程序和产品/工程门均未验，固定AUDITED有限FIX前不写产品。
