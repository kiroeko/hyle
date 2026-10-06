# CPU 冻结自研算子：有限精确域裁决

<!-- MODULE REVIEW: SCOPED -->

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
| ND2 | 一批同端口/同reference/同checked执行数值与资源witness，不计时 | 待执行；prepare成功不代签数学 |
| ND3 | 固定有限P1消费/peer/采样与严格方向；无优化，结果驱动有限FIX | ND2之后，不让错误实现赢性能 |

各普通域、strict资源与模型port条件的最终KEEP_WIN/KEEP_UNIQUE/RETIRE_OVERLAP须写明可复核
依据。所有私有算子数学与SIMD通用primitive分开，后者不因Provider输而删除。
