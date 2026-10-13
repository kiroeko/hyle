# G01 Apple 原生 page16 数学阶段证据

产品草稿 `afd2d8df7`，五 owner 仍 AUDITED，不签完整 G01/G02 或其它 G 项。
活动范围见[原生精确页卡](../../../docs/implementation/tasks/compute/inference-g01-native-paged-read.md)，
普通消费者见[公开接入](../../../docs/user/batch-attention.md)。

actual MacBook Air Mac16,12/M4/arm64/macOS27.0.1 26A434/AppleClang21.0.0，只有该 Air
用于本轮 Apple 数学。Windows canonical SHA 输入单向同步验证副本，无反向源码覆盖。
原 MLX1f8e74e3f12f source 与已有 cast/resource patches 保留；新 private inference-only
primitive 借原 dispatch/eval，原 one-pass、two-pass、GQA vector、NAX（含 D256 split）
与 older steel 的全部 K/V 源地址映射页表。普通 SDPA/AttnParams 不增页字段，原
MMA/softmax/reduction 不替换，无 CPU fallback/AD、全 KV gather 或新 executor。
NAX32-row K tile 分别从两页加载原16-row fragment，不假设页连续。

最终短域十二组：F16/BF16×D64/128/256×causal/noncausal，logical79>physical64。
长域十六组：上述十二组改 logical4111；另 F16/BF16×D64/128、H16/2、logical8207
命中原 GQA two-pass 特化。每组64次两请求 topology 演化，共1792最终 episodes，另有
独立 slot/旧结果保活。全部 independent double oracle PASS（F16≤0.002，BF16≤0.008）；
foreign schema/mixed generation拒绝、未引用页可写、ready页共享、metadata最终零用量、
单I32表/两mirror费用也通过。private primitive 只走原 Metal math；同 actual M4 上
q>8 沿原 NAX（包括D256 split），长 decode 沿原 two-pass/GQA dispatch 条件。Older
steel 的读点有源证据，未把 M4 数值外推为其它硬件上的 older steel 资格。

初次 build33.481s FAIL：route 缺直接 mlx/ops.h include；原日志/输入保留，未改工具。
修正后 build5.068s、初版 math4.449s/reference2.064s PASS。为覆盖长历史读法扩展
consumer，final build2.010s/short1.923s/long9.515s/reference0.928s 全部 PASS，累计
59.438s，不重置原失败。raw.zip 保全部阶段、三份 SHA 输入包、复现脚本、实际编译
command、全部 MLX patch after SHA 核验、selector 原 selected/skipped reasons。
receipt.json 绑定产品 commit、实际身份、结果与未决门。原 direct/Graph/fault/append
materialized reference 通过只签负空间，不代签新 page 故障路径。

新 page accepted-prefix/unknown/native error/逐cold OOM风险、宿主风险、三交付/逐头
和不可变同合同P1尚未完成；CPU+GPU联合分页仍由G02验收。未运行项不写PASS。
