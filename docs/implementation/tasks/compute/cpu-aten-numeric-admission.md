# CPU ATen：数值成熟程序批量接线

<!-- MODULE REVIEW: CLOSED -->

- owner：`compute_provider_native`，baseline `7c263f8631902b01b20b06007fc59db4fd6a2506`，Windows canonical。
  现有普通CPU/Graph已经组装独立ATen factory；本批不新增优先级或改公共规划器。若新的具名原库
  反例涉及已关闭的其它owner，先记录并重新绑定其有限接缝，不能借本卡跨owner改动。
- trigger：[全族审计](../../../../evidence/correctness/cpu-ggml-mature-batch/adapter-audit.md)的
  SiLU/GELU/GELU-tanh、narrow Softplus、两种GELU门控、number Min/Max与Min/Max Reduce仍被
  adapter挡住；已锁ATen有原版候选。原函数−Inf、尾部、signed-zero差异已实际观察，必须资格化
  完整原库程序而非按函数名宣称支持。
- scope：plain F32/F16/BF16，exact shape、rank0..8、任意合法byte stride、inferred dense output；
  Unary上述四kind；GatedActivation三kind的原程序候选（packed/separate，F32 activation与乘法后
  一次输出舍入）；Binary Min/Max；Reduce Sum/Mean/Min/Max、任意axes/keep-dim/empty及显式F32
  accumulation（F32允许Unknown）。已有GGML/oneDNN正确域仍按当前普通规则优先，ATen补成熟缺口。
- 主纵切：LLM激活/两GELU专家门控与聚合；正交纵切：广播number Min/Max及普通多轴统计/空identity。
  不特化模型名，不扩张公开operation，不改自研数学、SIMD、GPU或公共性能/构建治理。
- 接入：现有Native component、已锁LibTorch2.8.0 CPU SDK、`aten_cpu::provider()` identity与
  普通assembler；F32/narrow数值端口用原Byte copy打包到私有对齐native tensor，禁止把odd byte
  stride截断为element stride。输出由原Tensor程序转回原Byte copy，保留原公共资源合同。
- 数值审计：原版Double activation候选后显式F32中间值，先验证精度/尾部/±Inf/NaN/±0；
  必要的−Inf极限只用原库where修复。Min/Max用原fmin/fmax，零号用原signbit/where等组合，
  Reduce传播NaN、空identity和零号规则用原库程序完成；禁止新HYLE payload循环或近似多项式。
- 生命周期：复用checked mappings/CPU Queue/真实Completion。immutable prepared program、
  独立invocation native输入packing槽；原库分配/线程与热Tensor metadata明确managed replay，
  包含全部packing/cast/native scratch成本，unknown内存不冒充零；限定精确alias须实际资格化。
- 预算：一批原程序+完整数值边界诊断，固定有限FIX后一次产品接线；Windows/Linux受影响factor/
  普通Direct/Graph与三交付批量运行，最多一次具体产品失败纠正。无自研P1/优化或逐算子SDK重建。
- 预声明：四Unary/三gate/两Binary/四Reduce，各三dtype；独立oracle、普通变化输入及Graph下游，
  tail/large gate value/±Inf/NaN/±0、广播/strides/rank0/8/empty/allaxes/keepdim、alias/close/独立
  invocation/OOM，既有存储/index回归、SDK关闭adapter、公开头与两平台source/static/shared。
  change selector/closure/docs/diff，统一Compute+Graph LOC；不重跑未改模型/其它路线/P1。
- stop：原程序未承载完整同合同的域，具名DEFER并保存实际反例；不放宽数值门、不加自研math，
  不因为库原函数同名或adapter拒绝签UNIQUE。审计时产品gate均UNPROVEN，最终产品资格见下方。

## 有限 FIX（审计后冻结）

| ID | 审计与拟接缝 | 必需验收 |
|---|---|---|
| MN1 | 原版activation/packed-separate gate，明确中间精度和极限 | 精度、尾部、大value放大、输出一次舍入、alias |
| MN2 | 原版number Min/Max与Reduce程序，广播/任意axes/空identity/NaN与零号 | 不借原amin零号或ISA min偶然结果定义公共语义 |
| MN3 | 对齐native packing槽、真实完成/异常/OOM、私有lane与borrowed mapping次序 | 全byte stride、资源平衡与原SI存储回归 |
| MN4 | 一批普通Direct/Graph三交付、用法/成本/remaining ledger | 默认真实成熟identity；不把原probe代签产品PASS |

SCOPED只授权本轮有限审计。完成实际原程序witness并固定BORROW/FIX/DEFER后才进入AUDITED并写产品。


## 冻结审计结果

[原程序48,780断言与失效参考边界](../../../../evidence/correctness/cpu-aten-numeric/README.md)已完成。
MN1–4固定BORROW/FIX。erf-GELU采用原Double erfc Tensor等价程序，tanh-GELU采用原Double
sigmoid/pow/add/mul程序，SiLU采用原Double silu；原where仅处理−Inf极限，activation显式F32
后乘value的F32，再一次cast输出。直接raw gelu与消减参考失败保留，不放宽数值门。
number Min/Max与Reduce用原kernel及零号mask组合；empty/reduce axes由cold Program固定。
冷native typed input槽包含对齐Byte packing；旧SI借用slots与映射顺序保持。仅Unary/Binary
的允许dense equal-desc端口可精确alias，所有输入先native packing后才写输出。
公共trait允许与实际可行域分别匹配，广播/strided/所有gate/reduce外部alias拒绝；GatedActivation
公共trait为kNone，即使native private packing可保留旧值也不扩大别名合同。
P0为O(input+output)原Tensor程序，Double cast/原library多节点temporary/全部packing均收费；
不承诺热零分配/完整内部内存上界，不制造prepared-only资格。软件/资源合同不变，无SDK或架构重写。
审计时owner AUDITED、产品gate尚未运行；原program witness不代签产品成功。已经存在的GGML有限尾部
源码疑点在remaining ledger记录，不能据本批新ATen后备宣称所有现有默认数学已合格。


## 不可变候选与首批产品资格

`0763fd3e3163d16eb0b43e684f5881f8314bc1e9`：原Tensor程序与cold对齐packing槽、允许Unary/Binary
exact alias、实际两lane并发/冷OOM/资源平衡与managed成本已固定。Windows Provider120/120、
普通CPU12通过/1旧fixture跳过、Graph15/15，三dtype普通Direct和op→Copy Graph变化输入通过。
先行Provider119/120因qualification调用Axes::from(empty).value()；公共schema明确拒绝该列表。
仅修正测试用合法Axes::all/单轴/多轴，保留初始XML与命令，产品未变、不放宽数值门。
源码/installed static/shared与Linux全部通过；Windows证据按产品输入一致复用，详见最终收据。


## 逐能力覆盖与最终七门

[完整证据](../../../../evidence/correctness/cpu-aten-numeric/README.md)。PASS仅本批plain F32/F16/BF16
exact/rank<=8/dense-output域，已锁Windows/Linux x86_64 SDK；不代签其它CPU平台/模型/GPU。

| 能力/public journey | 正常与边界 | failure/OOM/overflow | ownership/teardown | concurrency/取消 | 主/正交消费 | 平台/交付 |
|---|---|---|---|---|---|---|
| 四Unary/三gate Direct→Graph下游 | PASS：三dtype/packed-separate/odd stride/tails/large-value/±Inf/NaN/±0、一次输出舍入 | PASS：checked容量与tracked冷OOM；其它非法alias拒绝；native物理OOM源码审计 | PASS：私有对齐输入packing、允许Unary alias先读后写、borrowed maps关闭与归零 | PASS：独立两queue/invocation两线程12轮；原kernel中途抢占NOT_APPLICABLE | PASS：LLM激活/专家gate及普通变化输入，不依赖模型名特化 | PASS：Win/Linux、公开头、同一consumer三交付 |
| number Min/Max与四Reduce | PASS：广播/odd stride/rank0/8/all-single-multi axes/keepdim/emptyidentity/NaN与signed-zero | PASS：half Unknown accumulation/非法alias拒绝；checked容量与tracked冷OOM，物理故障未注入 | PASS：Binary允许equal-desc alias，Reduce无alias；private slots/close/资源归零 | PASS：immutable Program及invocation隔离；同步原库不承诺抢占 | PASS：普通多轴统计与主gate融合边界消费，Graph最终结果核对 | PASS：两平台同一四交付journey |
| factory/默认组装/成本 | PASS：独立ATen identity、原成熟排序、feature关闭不绑定 | PASS：managed/private-thread/unknown真实属性，硬预算不以unknown冒充零 | PASS：既有CPU Context/Completion与存储index回归保持 | PASS：线程属性明确，无全局线程设置或新Queue | PASS：用户仅选CPU，准备绑定，执行不计时裁决 | PASS：既有SDK/package seam，stub仅声明实际边界 |

| Gate | 证据与限制 | Verdict |
|---|---|---|
| scope | MN1–4固定；十三kind及gate输入模式逐项回映；不扩张自研数学/模型loader/GPU | PASS |
| architecture | 既有Native binary/ATen独立factory、原Tensor程序和对齐packing、CPU/Graph成熟顺序；无新万能executor | PASS |
| correctness | 两平台120/120、CPU12通过/1旧fixture跳过、Graph15/15、头1/1；逐能力矩阵及故障/资源资格 | PASS |
| delivery | 两平台相同consumer source shared/static、installed shared/static；725产品输入hash复用证明/1333 forward identity | PASS |
| usability_docs | 普通入口/数值阶段/alias/managed成本/SDK平台限制说明；docs/closure通过 | PASS |
| red_team | root独立一次固定MN范围源码/收据复核；两fixture问题保留，原尾部参考及物理故障/其它owner边界明确 | PASS |
| engineering_performance | P0包含全部Byte packing/Double-F32 casts/原Tensor节点/temporary与metadata；无Native P1/SOTA/WIN | PASS |

sanitizer/codegen测量不在本卡预声明验证；未改intrinsic/layout ABI/source hint，不制造测量。
native物理OOM和非coherent映射故障仅源码检查，不写注入PASS；受供allocator冷OOM已实测。
旧Qwen fixture跳过不计PASS，主/正交ordinary旅程实际运行。测试警告修正不改变不可变产品，
不用测试维护commit制造新算法candidate或重做Windows交付。

最终 **CLOSED**：候选 `0763fd3e3163d16eb0b43e684f5881f8314bc1e9`，全部适用MN工程门完成；产品字节候选后未改。
有限重开条件：本批具名语义/资源/生命周期/组合/交付反例或相关依赖升级。
统一Compute+Graph LOC **61,950/249文件**，不含测试/vendor。CPU整体未完成，
其它成熟边界及冻结自研逐域裁决仍需后续有限批次。
