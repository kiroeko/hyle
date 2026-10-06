# CPU 冻结自研算子：有限精确域裁决

<!-- MODULE REVIEW: AUDITED -->

- owner `compute_provider_native`；baseline `3bf9aa99e18a96fe46ab07e9b84aba97ef7dec49`，Windows canonical。
- trigger：SI/MN/TC/TM/AS/CP已实际关闭，成熟普通域及资源声明固定；用户要求所有有成熟替代的
  CPU自研算子裁决，不能永久依靠Native1000 fallback保留。此前自研冻结现进入这一有界裁决批，
  不是恢复旧M5队列/比例/优化；SIMD基础、原Torch/ds4与standard Scan/TopK borrowing明确保留。
- scope：当前Native generic payload的12族：MatMul、两norm、13 Unary、6 Binary/Select、4 Reduce、
  3 packed/separate gate、RoPE、Softmax/log/masks、equal-head SDPA、Gather/Scatter、Concat/Cast/
  Copy、Fill；dtype/布局/alias/数值/资源分别划域，不把storage carrier视为宽math。View为metadata。
- 共同消费：已有普通CPU Direct/Graph/model port形状；主域F32/F16/BF16及已公开Cast/storage；
  反偏置包含NaN/Inf/signed-zero/稳定负尾部、odd/非末轴/重复index/empty与真实wait。成熟候选为
  普通CPU固定规则选择的oneDNN/优化ggml/ATen；必须记录id且拒绝把Native fallback当成熟peer。
- 资源独立：whole-domain只保留已证RMSNorm/Copy；prepared-only与strict invocation/private预算
  只按实际同合同候选划域。CP已消除cold compilation假缺口，不以旧unknown签唯一。
- 有限ND1–3：ND1固定源码/候选与同合同P0成本、已有证据可复用性；ND2一批不计时的数值/资源
  witness，失败实现不参加竞速；ND3对仍合格重叠域一份冻结P1 bundle，固定primary/holdout，
  workload A/A、A/B+B/A与完整packing/同步/分配/融合成本。先固定结果/有限删除或规则FIX再改产品。
- 预算：本轮准备/有界源码分析最多30分钟；最多一次具名fixture/测量缺陷纠正30分钟；分析总额
  2小时、采样总额2小时，n15起且不超过31有效配对/比较，算法优化预算0。没有需求的域不制造
  benchmark；既有可完整替代且无保留优势的平局/INCONCLUSIVE选择成熟实现，不伪造LOSS/WIN。
- stop：到预算按证据裁决而非无限重测。真实同合同缺口具名记录，不能丢必需能力或把整个
  重叠目录当唯一。测量失格不签赢家；适用未运行不签PASS。此卡当前SCOPED仅授权审计/fixture。
- 后续验证：只有冻结的有限产品FIX进入AUDITED才实现；一次所选factor/普通Direct/Graph与
  必要Win/Linux source/static/shared、closed SDK分支/源码身份复用、selector/closure/docs/diff与
  统一LOC。无GPU/模型级联合推理/新目录扩张/公共治理或性能仪器重写。

## 本批有限审计输出

| ID | 必需输出 | 当前 |
|---|---|---|
| ND1 | 12族当前具名成熟候选/资源边界/borrow vs self，valid旧证据与实际P0成本 | 待固定；历史宽泛KEEP_RESIDUAL不授权保留 |
| ND2 | 一批同端口/同reference/同checked执行数值与资源witness，不计时 | 数值322条实际完成；Native31未资格、成熟0失败；16个域固定退場，不能入竞速 |
| ND3 | 固定有限P1消费/peer/采样与严格方向；无优化，结果驱动有限FIX | ND2之后，不让错误实现赢性能 |

各普通域、strict资源与模型port条件的最终KEEP_WIN/KEEP_UNIQUE/RETIRE_OVERLAP须写明可复核
依据。所有私有算子数学与SIMD通用primitive分开，后者不因Provider输而删除。

## 已固定数值退场 FIX：ND-R1

[同合同实际witness](../../../../evidence/correctness/cpu-native-final-semantic-audit/README.md)固定下列
16个op/kind×dtype/position域为RETIRE_NUMERIC_INELIGIBLE，不签性能LOSS，不尝试优化：
Unary GELU/GELU-tanh三floating、F32 Unary SiLU；GELU/GELU-tanh gate三floating及F32/BF16 SiLU
gate；长I64公共RoPE原F32实现。generic descriptor无法低成本表达“输入值恰好避开负尾部/大位置”，
不靠随机输入或模型名掩盖通用问题。原ATen相同端口与既有资格可完整替代这些普通域。

允许的产品改动只为ND-R1：删除上述Native准入与其无其它Native用途的专用tensor执行分支/公式/
callback，保留F16 SiLU gate及I32 RoPE等待后续裁决；public operation、原ATen/ggml/oneDNN、SIMD
基础/函数、其它Native数学与原Torch/ds4 borrowing不改。不因为strict资源唯一保留错误数学；在
仅允许Native的strict域明确拒绝，不能偷偷调用managed路线或放宽预算/精度。源级删除和正交回归
批量完成后，再处理ND3合格重叠域；ND3 benchmark/rule/deletion集合未固定，当前不授权其产品改动。

ND1/ND2以此前全族/source/resource审计及本批反例固定这一实现子批，owner AUDITED。后续发现
新具体域或结果需要扩张产品FIX时须退回SCOPED/AUDITED，不无限扩大本次修正。生命周期/默认
Direct/Graph、受影响测试与两平台三交付仍适用；当前未运行，不签ND-R1产品完成。

### ND-R1实施进展（未终审）

`cde6fb3a`批量删除Native GELU/GELU-tanh公式/注册、F32 gate callback、BF16及非SiLU half gate
分支、I64 RoPE读载体/准入，原SIMD函数/模块与ATen不改。Windows owning binary低并行增量编译
通过。首次整批再执行发现退役Unary沿旧select失败返回InvalidArgument，导致普通context中止
成熟匹配；`f32d261a`定向改为Unimplemented冷拒绝，没有数学优化。原失败保留。

同一322条witness最终Native31明确拒绝、剩余已执行行0误差失败；普通成熟161全部执行通过，
没有把拒绝算PASS。只证明本批样例和冷迁移，不代签完整factor/Graph/两平台三交付；旧Native-only
测试对已退役域须按需求迁移并保留专家拒绝回归。ND3仍未计时/未裁决，模块仍AUDITED、goal活动。
统一脚本在`cde6fb3a`计Compute+Graph实现61,387/249，不含tests/vendor；`f32d261a`单行Status
替换不改代码行数。必要回归/交付完成前不关闭ND-R1，不把源码删除当完整CPU收口。
