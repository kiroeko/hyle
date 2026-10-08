# Apple A3：默认承载与有限成本裁决范围

<!-- MODULE REVIEW: AUDITED -->

父轮[Apple parity](apple-cpu-cuda-parity-closure.md)，baseline `bdc60f88`。
这是现有冷绑定的P0审计与后续有限比较范围；没有性能PASS或产品FIX。
FP4桥接已获用户授权并完成数值/资源/三交付；完整候选固定后启动一次P1。

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
| 官方F32-scale FP4 roundtrip/专家/完整region | 原DS4/DeepSeek数学的MLX GPU承载与已授权位域保真桥接 | 原packing/scale/RNE/BF16不变；全部finite原语与完整main/DSpark通过，不换格式 |

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

## 两成熟候选的正确性批次（产品 FIX 零）

只对linear的BF16输入/权重→F32输出，以及grouped的plain F32/BF16权重+F32 activation做
原生数学准入。原ggml Metal通过普通prepare，MLX候选通过普通Context.compile原astype/
matmul/gather_mm；两者有同一真实owning async/completion/lease服务。grouped的MLX候选
复用HYLE公共checked Gather语义生成安全IDs，仅复用检查服务，不调用产品乘法lower。
必要全权重F32转换按实际尺寸列cold/dynamic成本，不宣称零成本。

一批真实K4096/8192和N1024/专家router、单/多token、两个变输入/变权重replay及复用lease，
另有具名F32精度边界（BF16大指数与可表示的小值）和坏ID。独立source encoded BF16与
两term double方程先验证，再比较两候选。只构建新测试源，不重跑原矩阵；原结果失败不能
以宽容差或有限正常值样本覆盖。具名测试/接线失败最多一次纠正；产品缺陷需另固定FIX。
该准入不是性能结果，两候选没有共同正确性域就不排名；不启动P1或调全局设置。

### 正确性结果与成本事实

`12e52172`原两候选4/4测试通过，含完整F32输出范围与checked IDs/旧lease；
[原raw及结论](../../../../evidence/correctness/apple-parity-a2/README.md)保存。首个router
N768是多测的形状，不能冒称目标目录几何；改为共同Qwen24T fixture的N512/K8192后，
`fa53b31f`只定向复验该case，1/1通过。其余三个未变case复用，不重跑全部4项。
两term独立方程保持逐值门；两种候选普通Context均实际执行、变权重/输入，非mock。

| 实际数据域 | 原ggml Metal候选的必需转换 | 本MLX F32候选的必需转换 |
|---|---|---|
| BF16 [M,8192]与BF16 [512,8192]，F32输出 | activation widen，M1为32KiB、M16为512KiB；保留8MiB BF16权重 | 相同activation widen，加16MiB整权重F32数组 |
| BF16 [8,1024,4096]与F32 [5,8,4096] | 原native BF16权重，64MiB payload | 另有128MiB整权重F32数组、checked IDs和原gather_mm |
| 同形F32 grouped权重 | 原native F32权重 | dtype cast无转换，但仍有原gather_mm/必要布局/检查 |

这些是明确数组尺寸的P0事实，不是总RSS、峰值或性能结论；原库private scratch仍unknown。
动态weight的widen由hot input产生，不能当cold免费常量；immutable Graph常量必须另外
按完整cold setup/资源/完整region终态口径裁决。当前没有WIN/default切换，仍保留既有
合格成熟默认。正式P1等完整candidate固定，一次有预算，不因本批PASS扩成算子目录。

性能consumer固定四域：BF16 router N512/K8192的M1/M16、plain grouped E8/N1024/K4096/
T5/S8的BF16与F32权重。均为动态weight端口、两个ready resident输入/weight银行、两个
可复用slot；共享不可变原bytes，两侧独立Context/stream与同等completion/result lease。
两candidate原prepare/原MLX compiled数学、checked IDs、必要widen/布局/同步全部收费。
构造/compile/暖机冷成本单列，hot从bind到成功终态和lease释放；无热读模型/重选Provider。
仅本批consumer/contract与verify-only预检，暂不A/A或P1；immutable常量的cold方案不由
dynamic结果外推。正式采用primary/placement、正反/分配序与固定n15/5%规则，原FP4
决定并完整固定后才采样，不更改公共仪器。

`8d8263a0`primary/placement consumer增量构建通过，verify-only四域完整两银行输出均
逐值通过独立方程，actual owner QoS均21、caller user-interactive。固定
[合同](../../../../bench/contracts/compute-apple-mature-overlap-p1-v1.json)经统一validate通过，
n15、八strata、A/A/已知delta/timer-floor、5% margin；module-local A/A借用原parser/机器
合同/reference identity，不扩公共统计。原静态producer仅Metal factor导致目标不存在的
旧配置及一次Darwin QoS声明纠正均保存，未触发产品修改。raw见
[A2记录](../../../../evidence/correctness/apple-parity-a2/README.md)。
没有启动A/A、AB/P1或导入best-of，这四域仍只有correctness/cost结构资格，非性能PASS。

当前完整产品candidate固定为`326a40ee9b2f01a760d8bbdc66974685ff8cfdb0`，FP4数值/
资源/三交付均已通过。后续文档/状态不改变产品candidate。只启动上述四域一次当前Air
工程资格：同路A/A16process，合格才固定n15八strata，不升n31；原MLX参考pin1f8e74e3
和compiled窄patch逐文件核对。统一仪器要求fresh qualified producer，使用唯一新目录，
build2/采样串行，既有模型/交付不重跑。未资格不导入winner，近似/噪声保留合法成熟
工程default并明确方向未决；未运行的正式项不写PASS。

首次fresh qualified批次源码build完成58s，同路A/A16process全部PASS且无bias/instability。
正式首process被统一仪器拒绝：新样例marker用了execution/allocation/qos，而公共合同要求
execution_order/allocation_order/apple_qos。这是模块局部性能consumer的遥测格式错误，
不修改公共runner/统计核心或产品算法。唯一纠正改三字段名和对应A/A断言、更新本合同
source provenance hash；原失败保留，正式n15并未完成。新artifact需本批A/A资格，使用唯一
纠正producer后清前一producer；不重跑任何模型/交付，不增加n15预算或无限换测试方法。

## 最终裁决

`7365f0fb9`纠正批次资格与工程门PASS，当前Air16同路A/A、120产品、30controls完成，
停止n15；原正式入口拒绝保留。四域32分层NI PASS、四个合并严格WIN；GGML Metal相对
MLX分层吞吐比分别2.1995–2.2091、1.2465–1.2564、4.2981–4.3266、1.8420–1.8472。
[raw/manifest/verify](../../../../evidence/performance/apple-parity-overlap-p1/README.md)归档。
BF16原widen16MiB/128MiB成本支持部分差距，不将全部时间唯一归因于它。

现有cold GGML Metal默认就是赢家，不改产品/重复交付。优选只限该机器/OS-driver/
toolchain/原provider、真实dtype/shape/ready contiguous dynamic合同；未匹配或cold
constant/完整融合region继续合法成熟ENGINEERING_DEFAULT，非SOTA。多数其它packing域
只有一个完整carrier，不造竞赛。没有待清退HYLE shader/GEMM/GDN/Hadamard kernel；
独有被授权FP4保真桥接与资源/checked组合保留，SIMD不动。A3完成。
