# G01 CPU 驻留与 native ragged 有限轮终审

状态：CLOSED，2026-10-13；owner `compute_provider_ggml`。产品candidate
`973d3ffbf642aff06f6316fcec4b3cc99eba2e4d`，数学candidate642cc1e67；原单请求
d91a及全部证据保留。授权/有限FIX见[ragged卡](inference-g01-cpu-native-ragged.md)
与[resident卡](inference-g01-resident-sdpa.md)。仅关闭CPU已验有限域，整个G01尚未完成。

## 七问

1. 借锁定ggml ffa4e8b8的原FLASH_ATTN_EXT、MetadataArena/CpuExecution与CPU provider，
   无新第三方/数学kernel；原IQ1等不相关扩展/许可/算法保原身份。
2. packed NHD两request/GQA8/2、短decode＋长prefill，F16/BF16 D64/128/256与noncausal、
   empty、重排、独立slots/旧output、tail/readonly、foreign schema/小预算/逐分配点失败
   和accepted error/真实退休反证。分页、其它attention数学域及全模型保原goal。
3. BatchProviderVTable/BatchProgram/F1/F2/Queue/Completion/Graph region与实际退休全部
   复用，原leaf proposal不变，没有第二个数学、Queue/controller/scheduler。
4. 冷逐request构建原flash，K/V native roots实际资格核验；warm直接slice并写packed
   output，无KV复制/gather/按Q/head展开。query/output、causal mask和metadata仍计费；
   known cold plan保守合计与外部pool分账，native private上界UNKNOWN。两个独立原核
   生命周期下的[多请求P1](../../../../evidence/performance/inference-g01-cpu-ragged-w1-p1/README.md)
   QUALIFIED、5%非劣PASS，verify退出0；三点band PARITY、相对1方向均INCONCLUSIVE。
5. 原C++工厂与既有batch接口，无FFI、兼容wrapper/public vendor type；工厂声明/root
   交付修正保持原symbol/Options/layout，只补普通compute/graph targets的可发现性。
6. Windows四风险tests/default入口；WSL ASan+LSan/UBSan/TSan各13tests与source Graph
   双region三san PASS。973的source/实际relocated static/shared普通Direct＋Graph双
   dependent regions、原single resident回归、macro-independent逐头与CPU未选时header/
   library不安装反证PASS。见[原数学风险](../../../../evidence/correctness/inference-g01-cpu-native-ragged-candidate/README.md)、
   [Graph资源](../../../../evidence/correctness/inference-g01-cpu-graph-resource-candidate/README.md)、
   [公开三交付](../../../../evidence/correctness/inference-g01-cpu-ragged-factory-delivery/README.md)。
   工厂/header/root不改变产品TU算法、热成本或ABI，宿主风险按同输入identity复用；不
   代签GPU/driver/private池或未运行项。
7. 普通cpu::create_context→BatchProgram→prepare/update→run→output自然可用，原Graph
   工厂只链接公开compute/graph。guide说明旧output/token/completion释放、完整topology
   reuse/cold rebuild、dtype/shape/paged/capture/private预算限制。

## 实际性能与停止

独立clean source9cfa、contract8751dfe5，实际W1-WIN/MSVC19.50/CPU2/单native线程，
16 A/A、120产品process与30 controls，n15即达到原停止规则；累计2188.172s。
Q1/17 KV1024/768 D256 causal比0.9980–1.0030，Q64/128长prefill0.9983–1.0017，
Q1/17 KV31/47 D64 noncausal0.9992–1.0046。全部5%非劣/原控制门/原verify通过，
不以band替代方向，不签全模型token/s或所有CPU预算。原单请求P1独立保留，不重复采样。

compile/schema/shared link FAIL与旧single STOP保留。诊断min2 raw被helper误覆盖，
仅15.641s工具耗时可留存，不用于qualification；min1完整9.422s保留并固定16轮/4ms
floor/min1于正式前，n15/31/5%/公共controls不变。selector对公开root增加误归全部原
roots的BLOCKED原计划按独立工具卡保存，不重开无改动owner/跑全库/写虚构PASS；
实际三交付与逐头/CPU未选反证直接证明本轮factory seam，工具不在本owner扩建。

有限CR-01/02/03/H与适用门已裁决，CPU owner本轮CLOSED。reopen trigger为已验域
可复现数学、直接KV/root/成本、owner/实际退休/allocator失败错误，或新真实消费者
需要原合同缺的机制。G02分页、G03 producer、G06增长规划、G10总预算及其它路线仍
属于原十项goal，不以复看已签矩阵重新启动本轮。
