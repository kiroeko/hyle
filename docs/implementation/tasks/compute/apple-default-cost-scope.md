# Apple A3：默认承载与有限成本裁决范围

<!-- MODULE REVIEW: AUDITED -->

父轮[Apple parity](apple-cpu-cuda-parity-closure.md)，baseline `bdc60f88`。
这是现有冷绑定的P0审计与后续有限比较范围；没有性能PASS或产品FIX。
本轮先补第三方承载，原FP4桥接待用户决定；API/算法完整固定前不启动P1。

## 现有合法默认

| 实际合同 | 默认完整承载 | 选择依据 |
|---|---|---|
| 普通浮点tensor/MatMul/norm/SDPA/SinkAttention/Hadamard | 原MLX GPU与compiled region | 所需dtype、accumulation、layout/shape通过公共schema与native准入 |
| MLX affine/MX4/MX8实际packed线性/grouped与affine embedding | 原MLX qmm/gather_qmm/dequantize/take | 原mode/group/nibble/scale与端口，非其它“四位”格式 |
| NF4/nested NF4、AWQ、CT冷权重准备 | 原来源公式的MLX GPU tensor程序 | 真实source packing/scale/zero/storage；结果由成熟MatMul消费 |
| GGUF原生MatMul/Grouped/GetRows | 原ggml Metal与原Anemll窄IQ1程序 | 原bytes/native type/所支持operation；无对应原kernel明确拒绝 |
| checked Q4_K精确所选行 | 原CPU row-decode公式在MLX GPU承载 | 原Metal half高nibble精度已FAIL；精确优先，不称性能WIN |
| block128 FP8线性/grouped、权重准备与FP8 roundtrip | 原MLX GPU转换/乘法/原有source组合 | 真实FP8 codes/scales、原F32顺序与BF16边界；不同物理格式不互换 |
| GDN/recurrent、mHC、模型原clamp/BF16-F32 linear/indexer | 原ggml Metal完整数学片段 | 各custom schema和native capability，不拆完整成熟片段 |
| 官方F32-scale FP4 roundtrip/依赖完整region | 明确未准入 | 原native tiny资格失败；等待有限保真桥接取舍，不能用其它FP4代替 |

当前普通入口先在prepare/compile做schema与上述冷匹配，固定MLX/ggml region；hot运行只
调用已准备程序。用户无需排序Provider或提供timing。未知模型身份不删除合法通用数学；
未知硬件/系统仍走成熟语义默认并让native capability给出明确拒绝，没有未测硬件SOTA承诺。
Environment实际记录device/architecture、OS含driver、SDK/deployment/Metal compiler与锁定
MLX/ggml版本；模型制品/phase/region和实际各端口格式已保留在Program/workload，新增
有证据条件规则可以冷匹配。编译依赖与manifest精确记录窄IQ1原来源，不把base版本替代
全部patch身份。现在不根据未测型号猜赢家。

## 自研与真实重叠

当前Apple owning source没有HYLE CUDA/Metal shader或独立GEMM/SDPA/GDN/Hadamard kernel；
Hadamard已经直接调用原`mx::hadamard_transform`。HYLE管理状态、失败/checked语义、原库
组合与存储适配；这些不是可通过删除“算子”去掉的资源能力。精确Q4_K、artifact与IQ1借入
保留原source/license，不作为HYLE优势算法。跨模块SIMD不在此轮清退范围。

不能只凭两个Provider名字造竞赛：原GGUF与MLX affine packing、FP8/block scale、NF4
各有不同物理合同。原准入集合里多数只有一个完整carrier，记录成熟ENGINEERING_DEFAULT，
不标未经测量的BEST_OF或SOTA。现有typed纯MLX与同数学直接原MLX region的比较仅检验
frontend/生命周期成本，不是HYLE数学胜利。

有实际重叠的两候选域须有界核对：plain BF16/F32输出linear与plain grouped，原ggml Metal
以及原MLX的必要widen/gather_mm组合。MLX若需整权重F32转换，cold常量/热动态转换、额外
native内存、同步与融合损失全部收费，不能把它视作无成本kernel peer。没有同合同原生
准入不进入排名，不为跑benchmark而新增算子或改变公开模型格式。本页不签这两域已裁决。

## 余下有限验收

1. 官方FP4方案决定并实现/资格化后，固定完整产品candidate和对应成本结构。
2. 只对上述两个真实重叠域做一次原生准入/完整成本候选，具名错误最多一次纠正；
   接近/不稳定快速选成熟合法默认，不升采样或无限优化。
3. 当前Air一次统一hyle-perf资格/适用完整region验证，n15/5% margin，先A/A；
   不合格项不导入WIN，不将不同服务的裸库计时当对手，不重跑无关矩阵。
4. 规则有适用硬件/driver-OS/toolchain/provider/模型制品/精度形状条件；其它组合继续合法
   成熟default。冗余自研只在真实存在、同合同替代与有限裁决后清退，不虚报删除数。

当前A3尚未完成。原已验区不反复重测；FP4未决时先完成独立交付/边界复核，不能用文档
或此P0表把whole goal签完成。
