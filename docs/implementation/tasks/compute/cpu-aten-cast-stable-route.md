# CPU成熟接线：浮点Cast与稳定泛型路由

<!-- MODULE REVIEW: AUDITED -->

- owner：`compute_provider_native`、`compute_provider_ggml`；baseline `b881d2788e0848404c536342b806d6642f3aefa7`，Windows canonical。
  CPU/Graph现有assembler不变；只改原成熟factory的精确准入/数值程序。不修改其已关闭公共规划器。
- trigger：全族审计仍有FP8 numeric Cast成熟候选未接；新增普通Direct与whole Graph实际反例：
  gate=-90/value=1e38，ggml返回0、稳定参考约-7.37461；value=Inf返回NaN而参考-Inf。
  generic Sigmoid(-90)返回0而F32参考8.19400869e-40。见本轮[证据](../../../../evidence/correctness/cpu-aten-cast-stable-route/README.md)。
- 主消费：五种推理浮点storage转换（F32/F16/BF16/E4M3FN/E5M2）和稳定门控→MatMul；
  正交消费：Cast all-bits/rounding/empty/odd-byte及Sigmoid尾部→Multiply大值。
- scope：已有Cast，exact/rank0..8/合法byte stride/inferred dense output/out-of-place，同dtype原位存储bits
  materialization；不同dtype按公共RNE、FP8有限saturation、Inf和NaN mapping合同，不重新定义dtype。
  已锁ATen原Tensor转换/Byte packing与where/clamp/bit-view组合候选；没有自研payload循环/数学核。
  ATen增加stable Sigmoid原Double→F32→storage候选。泛型ggml Sigmoid/SiLU只有承载完整同合同后
  才可准入；原模型program保留各自独立已验收数值合同，不能以本轮generic反例直接推翻。
- 路由：只撤回已证明不合格的泛型候选，普通prepared/Graph匹配现有成熟ATen；用户不排序、不提供profile。
  若原ggml的F32程序不能保证stable-tail，则不能把错误整region保留为性能基线；此时必要边界由HYLE
  规划，ATen managed leaf和后续mature MatMul组合。其它ggml/oneDNN原region保留；不添万能执行器。
- 资源：既有冷private native packing槽、checked host mappings、真实Completion与异常/close；managed
  original Tensor metadata/scratch、private threads、unknown内存，不承诺capture或零热分配。
  FP8是实际原数值转换，不用上一轮Byte bit-copy冒充；Cast全部alias拒绝。
- 依赖：既有LibTorch2.8.0 CPU，已锁SDK仅Windows/Linux x86_64；不升级ggml/SDK、不改factor/package。
  不写Native自研math、不计时/淘汰SIMD算子；SIMD基础保留，其它路线暂停。
- 预算：一次原Cast程序与Sigmoid准入witness，固定TC1–4后批量产品修改；Win/Linux完整受影响
  Native/ggml/CPU/Graph factor与同一ordinary consumer三交付，最多一次具体产品纠正；重型串行j2。
  不逐dtype逐算子重建SDK，不重复旧ISA/loader、未改完整模型、P1或公共治理系统。
- 验证：独立FP8枚举/nearest-even参考、全部源storage bits及tie/边界/±Inf/NaN/±0、全25种dtype对、
  rank0/8/empty/zero与odd byte stride、out-of-place拒绝、重放/close/OOM/资源；稳定Sigmoid三dtype
  尾部及原大值gate下游；成熟identity与原有效其它region回归。模型fixture缺失如实跳过不签PASS。
  Windows DLL邻exe/隐藏启动；selector/closure/docs/diff、候选Compute+Graph统一实现LOC。
- tier：P0，包含所有packing/cast/native Tensor mask/临时数据及必要图边界/融合损失；不签SOTA/WIN。
- stop：原库程序不能匹配具体域即保存具名差异，不放宽数值、不写自研math或判UNIQUE；达到预算
  按协议裁决，未运行适用验证不签PASS。此轮完成不是CPU整体完成。

## 有限审计对象（尚未产品FIX）

| ID | 对象与拟接缝 | 必需证据 |
|---|---|---|
| TC1 | 原版五dtype Cast，RNE/有限sat/NaN mapping/同dtypebits、对齐Byte packing | 枚举与稳定独立参考；不把原to溢出NaN误当公共sat |
| TC2 | 原版Double Sigmoid；原ggml两个generic域资格 | 具名反例、stable-tail oracle；同模型program与generic合同分开 |
| TC3 | private invocation/managed成本/失败/close/alias边界 | 一批资源与普通重放资格，native物理OOM与实际冷OOM分开 |
| TC4 | ordinary Direct/Graph原最终结果、三交付与剩余ledger | 不用Native fallback代签接入；记录必要region边界，不以未资格核计时 |

当前SCOPED，Cast原程序尚未审计完成，产品/三交付/工程门UNPROVEN；完成原witness后固定有限FIX才写产品。


## 固定审计与有限 FIX

原版Cast/Sigmoid程序 **1,005,272独立断言**通过：25个容量断言，993,220个逐位Cast结果与
12,027个三dtype Sigmoid结果。F32有67,060特殊/random/tie/相邻值；F16/BF16源各65,536完整bits；
两FP8各256完整bits。枚举nearest-even参考及NaN widening/narrowing bits独立于原程序和SIMD实现。
这只证明原程序可运行，不代签产品stride/资源/默认/交付。

原native转换finite overflow/NaN mapping不是公共合同；采用原clamp/where/typed-bit view及原
bitwise kernels恢复已有Inf/NaN规范，finite数学仍由原to执行。相同dtype原typed storage直接Byte
copy，保持全部源bits；不同dtype先F32，FP8 finite sat与E5 Inf区分。没有新HYLE payload算术循环。
所有原输出大小先校验；早期raw fixture少grad-mode include、Byte from_blob错误元素数、Sigmoid
oracle忘记低精度storage舍入已修并保留失败日志，不把无效fixture读值写成原库失效。

原ggml generic SiLU/Sigmoid没有承载已实证stable tail；不升级跨平台pin，不新增HYLE稳定math
kernel或跨vendor回调/vtable。TC2固定撤回这两种generic冷准入及其死lowering，用已有ATen原程序
承担稳定generic叶。完整Graph在必要正确边界组合，MatMul和其它有效ggml区域保留；融合损失属于
此正确程序真实成本。ggml RecurrentGatedDeltaNet等具名模型原程序不继承generic严格合同，也不
以本反例重开它们；原有画像/误差资格保持，不能宣称全部原模型已重验。

| ID | 冻结动作 | 必需验收 |
|---|---|---|
| TC1 | BORROW/FIX：现有ATen factory接五dtype Cast与原NaN/saturation语义Tensor组合，对齐Byte packing；Cast禁止alias | 全25对、完整source bits/ties/empty/rank0/8/zero-odd stride与真实结果 |
| TC2 | BORROW/FIX：ATen原Double Sigmoid→F32→storage；ggml generic Sigmoid/SiLU拒绝，不留死callback | 大value/Inf gate下游不再错；其它成熟region与原模型程序不改；用户无需排Provider |
| TC3 | FIX：复用现有private槽与完成/异常/close；公开managed属性与所有转换/临时节点/packing费用 | tracked冷OOM/归零、不同invocation、alias拒绝、changed-input重放 |
| TC4 | FIX：迁移这两个generic资格测试及普通消费者的合法承载/图边界假设；Win/Linux完整受影响factor与三交付一批 | 有效旧准入保持、fixture跳过不计PASS；输出完整下游oracle，不用错误whole region作性能基线 |

其它Cast dtype、SDK平台、模型program、public API、性能仪器与Native自研payload不变。
owner已AUDITED，有限TC1–4已固定；全部产品工程门尚未运行，不能写CLOSED或新性能WIN。


### TC1有限接缝补充（仍AUDITED，产品未固定）

同一旧SDK实际默认ggml Cast对F32 quiet NaN 0x7fc12345/0xffc12345：F16给7e00/fe00，BF16给7fc1/ffc1；公共Cast的既有canonical为7fff/ffff和7fc0/ffc0。此诊断仍链接先前合格SDK产品0763fd3e，不是当前未编译草稿；收据HEAD只标原consumer生成时revision，命令/source见cast-nan-prior-*。

TC1补充固定GGML mixed浮点Cast的原位NaN storage-boundary：原kernel保留全部finite conversion与内存/融合，只检查原stored NaN bits并写已有规定bits；不做有限payload数学。额外O(n)检查和一个internal view node包含成本。新增全源bits/普通Graph下游回归，完整原转换与其它region保持；不因NaN编码差异撤掉有效成熟Cast。该FIX仍在已有五dtype Cast与同两个owner范围内，无新公开合同/SDK/常态成本等级；没有Native自研优化或P1。
