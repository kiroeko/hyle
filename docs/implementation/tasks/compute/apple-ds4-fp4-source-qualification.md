# Apple 官方 FP4：原 DS4 Metal pass 资格

<!-- MODULE REVIEW: AUDITED -->

父轮[Apple parity](apple-cpu-cuda-parity-closure.md)，baseline `d7c06ce74`；owner仍Apple三
AUDITED记录，不更改旧[F46](llm-moe-d5-apple-fp4-native-qualification.md)失败或预算。
这次不是重跑原MLX MXFP4 probe，而是新增具名成熟source候选：已经登记CPU来源的
antirez/ds4 `9139e2ae58a41503968a500f36f75895c1ba63fc`，原
`metal/dsv4_kv.metal::dsv4_e2m1fn_dequant`及indexer内完整group32 FP4 pass。
其原代码明确包含floor `7.052966104933725e-38f`、ceil power2 scale、饱和/RNE与乘scale。
另有原host `DS4_METAL_MATH_SAFE` 对应Metal Safe math模式的具名责任分支，测试只给本
候选library设置，不改变全局driver、系统或其它native库编译。

## 固定有限资格

仅原helper、原group32 reduction/量化pass与声明式IO wrapper；Hadamard输入是消费者的
另一步，本probe不借其旋转替换普通FP4 pass。原数学表达式保持，不能改任意codebook/
rounding/floor/scale算法。先用只读临时probe在当前Air验证全部finite BF16的原F32展开、
原CPU ds4公式参考、完整32组、normal/tiny边界；分别记录原默认及原Safe-mode source。
只做一批，不调整容差，不重复原F46命令，不跑P1；shader/ABI具名问题最多一次定向纠正。

结果通过后才重新固定有限产品FIX、公开Direct/Graph、完整expert/indexer片段、资源与三
交付；probe或floor存在本身不是路线PASS。若tiny仍不符，记录精确source/runtime边界并
裁决成熟缺口，停止同候选微调；不能悄悄用CPU模型计算或换BF16模型数学。此轮产品FIX零。

## 一批原source结果

当前Air/M4、AppleClang21，原helper与原group32 quant pass配声明式buffer wrapper。
[source/probe](../../../../evidence/correctness/apple-parity-a2/ds4-fp4-probe.mm)、
[原pass](../../../../evidence/correctness/apple-parity-a2/ds4-original-pass.metal)、
[身份](../../../../evidence/correctness/apple-parity-a2/ds4-source-identity.txt)、
[build](../../../../evidence/correctness/apple-parity-a2/ds4-fp4-probe-build.log)和
[完整结果](../../../../evidence/correctness/apple-parity-a2/ds4-fp4-probe-result.log)。
全部65280有限BF16的F32展开、完整32组，以原ds4 CPU公式作reference。原默认和原Safe
mode均222不符/190非零变零；其余32不符本批没有完整归因，不能擅自说都是符号零。
floor已经对齐，Safe模式未解决残余，原source没有完整合同资格；不重跑同候选，不修改容差。
此probe不是新产品能力，旧F46 FAIL与预算保留。GPU浮点路径的责任范围已定位，硬件/编译器
具体分摊未唯一证明，不据此宣称Metal全部FP4不可用。

## 待用户决定的有限方案

推荐允许必要的GPU数值保真桥接，目标仍为原FP4数学：输入/scale/饱和/RNE及最终BF16存储
定义都不变，正常域继续成熟原数学；针对原GPU不保真的次正规与零符号域，使用MLX原生
integer/view操作表达精确的源存储语义。没有CPU模型执行、新量化格式、模型改精度或算子大全。
这会新增一小段HYLE精度适配实现，而不再是原native程序直接借入；因此须按当前goal的冻结/
无完整成熟同合同时请求取舍规则明确授权，未授权前产品FIX仍零。

准入预算固定为：原source/原IEEE与已有storage转换复用审计；单个有限FIX；一次全部finite
BF16+mixed组/中点/符号与tiny独立oracle批次，具名实现失败仅一次纠正；原Direct/Graph完整
expert/indexer片段、资源与三交付仍必须通过。正常域不作无限性能优化；同原定义、完整转换/
同步/终态成本用父轮仪器。若仍失败或成本无合理界，列出真实缺口并停止，不继续微调。
替代方案是继续坚持纯成熟native source，把官方FP4明确BLOCKED；whole goal不能据此完成。

## 2026-10-09 用户授权与有限 FIX

用户明确答复“可以补一小段 GPU 数值保真适配”，后续pause/resume不撤销该授权。
baseline `0905a2f74`，Windows干净；Apple route/Graph owners仍父轮AUDITED。
旧MLX/原DS4 probe失败与预算保持。本次新增被授权的保真方案，不重跑旧候选。

唯一首批FIX：BF16 group32 roundtrip保持原max/floor、ceil-power2、饱和与E2M1 nearest-even。
原DS4八值/距离/偶数tie公式仅翻译为原MLX GPU tensor ops；用原BF16位视图取得tiny/sign。
当scale为最低两个normal power2时，只以位域/整数精确恢复可影响舍入的normalized值；
最低scale的半单位结果用整数存储保留次正规值，零符号也从原位恢复。正常域仍是原浮点
数学，输出保持BF16，无CPU模型计算/新shader/格式改动/新GEMM。

先一批两入口全部finite BF16/变化组max/所有中点邻域/正负零/tiny/normal/overflow与replay，
按raw BF16位比较。具名实现失败最多一次纠正；通过后才批量接原FP4 grouped承载及官方
expert/indexer/完整main/DSpark region、资源和三交付。它们尚未PASS，不由原语探针代签。
代码/API稳定的完整candidate才进入已准备的A3正式成本，build2/test1重型串行。

首批`80c01bdc`build与Direct/Graph各1/1通过：原65280有限BF16逐个覆盖、65280个
uniform group32、mixed groups、中点邻域、signed-zero/tiny/max变化和反向replay/close
后lease，全部raw BF16位相等。[raw](../../../../evidence/correctness/apple-parity-a2/README.md)
保留。这里只关闭roundtrip精确原语缺口，尚未签完整专家/main/DSpark或性能。

下一有限成熟接线不新增GEMM：沿已登记DeepSeek
`third_party/deepseek_quantization/upstream_kernel.py::fp4_gemm_kernel`的原sub-K32程序，
原FP4 nibble→F32→FP8精确转换、原activation group128量化/power2 floor、逐32 dot后
先activation scale再weight scale与顺序F32累加，由原MLX matmul/数组表达。只取checked
selected experts的原payload/scales，不全表decode或换Q8/BF16模型格式。原ATen承载已有
同样group32分解仅作来源证据，Apple不调用CPU。Canonical float4 carrier/InputSpec与
custom predicate两入口同接；热期没有Provider搜索。保持原资源status/索引/lease。
本批选现有官方FP4 projection的actual N/K、共享/独立activation、T1/T3、双replay，
随后对应完整官方expert/indexer/main/DSpark验证。具名失败最多一次纠正，未跑不PASS。
