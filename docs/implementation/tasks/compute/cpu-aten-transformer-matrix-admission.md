# CPU成熟接线：Transformer与矩阵剩余域

<!-- MODULE REVIEW: CANDIDATE -->

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


## 固定审计/FIX

[同一原库witness](../../../../evidence/correctness/cpu-aten-transformer-matrix/README.md)最终212断言通过，
不代签产品。首次oracle漏判visible +Inf已纠正，不归因vendor。原fused SDPA all-mask返回0的合同
差异已实际确认；不拆它冒充同合同程序，通用候选是原Tensor math完整attention，不新写HYLE kernel。

| FIX | 固定BORROW/接缝 | 产品验收 |
|---|---|---|
| TM1 | 原F32 matmul/Softmax/log-softmax与RMSNorm/LayerNorm程序，支持已有三floating dtype、rank/byte stride/axis/Bool-additive | 正交/empty/rank8/非末轴/special rows/parameter storage与最终结果 |
| TM2 | 原Double位置RoPE，I32/I64/两layout/prefix/frequency，clone保留tail；原attention math与bottom-right/zero keys | 长/负I64、未旋转NaN bits、mask/GQA/Dk-Dv/layout及最终storage舍入 |
| TM3 | 复用对齐native packing/private invocation/mapping/异常/OOM/close；增加Bool/long原载体，不重写生命周期 | alias只在公共允许且equal-dense input0；MatMul/SDPA仍无alias；两槽/线程/资源及managed完整成本 |
| TM4 | 同一普通Direct/Graph消费者默认绑定成熟域，与必要有效region组合；两平台三交付 | 用户不排序、不timing；缺验证/fixture skip不签PASS；Source/native数学冻结 |

固定精确域为plain已公开三浮点、F32 accumulation（现有合法Unknown匹配），输出inferred dense；
RoPE限已有F32 data/frequency域，I32/I64和全部公开axis/prefix，mask为Bool非零或可广播三floating。
未证dtype/bounded/严格native内存预算不签支持/唯一。RMS/LayerNorm增加非vendor-layout成熟候选；
合法省略gamma/beta、已声明F32阶段必须保持。AUDITED后只改现有ATen factory的TM1–4。


## 不可变产品候选与首批验证

CANDIDATE `3fc528dae5a25b7ac3f0bcc9606b112ee23b7c6d`：原matrix/Softmax/log、Double RoPE、完整
attention math及norm程序已经接到既有ATen factory；普通Byte packing、Bool非零与I32/I64原typed
carrier复用现有生命周期，限定公共equal-dense input0 alias；MatMul/SDPA/Cast继续无alias。
没有新Native payload算法。RMS/LayerNorm仍显式F32 accumulation和原浮点affine。

首批Windows Provider130/130、CPU12通过/1旧fixture skip、Graph33通过/6旧fixture skip，未重写skip。
新增6个批量case覆盖三storage rank8/broadcast/transposed/odd-byte MatMul/K0，Bool/log/additive/非末轴
Softmax与无效行NaN、I32/I64 batched prefix RoPE/两layout/长正负位置/未旋转signalling NaN bits，
两attention layout/GQA/Dk-Dv/Bool-additive/bottom-right causal/odd stride，norm浮点gamma/beta、
Softmax dense exact alias/独立slot变化输入/close/资源归零。后续empty/更高rank/并发/OOM与ordinary
Direct/Graph三交付资格仍需按本卡补齐，不能以这批unit代签完成。

fixture-only commits7ee7baf0、46e43573修正span prepare的显式resource、alias Result访问和mixed
special-value initializer；所有失败log保留，产品源码未改，未消耗产品算法纠正预算。Linux与
普通三交付尚未运行，状态不CLOSED，不签性能WIN。候选后的产品变更依法退回AUDITED。
