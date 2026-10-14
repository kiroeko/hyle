# G03 Apple 原生只读来源固定 F2 slot 有限验收

2026-10-14；owner `compute_route_apple`；baseline `66b433cef`；按
[活动卡](inference-g03-apple-copy-source.md)实施的 AS01/AS02。
不可变 candidate `dab23ba92256cad21f2e83af6141eb8ad51264c5`。

原 native producer 的 readonly export 仍拥有独立的真实 MLX resource owner。原 F2
bind_source 增加私有 typed family 资格，使冷固定 program/slots 可以接每次新的导出，
不再为不同 resource token 重建。原可写端严格 resource identity，实际 backing范围/
alias/device、F1 admission、FIFO Job、成功 output/取消/放弃/actual retirement 不变。
没有新增公开 symbol/选项、对象字段、controller、依赖、兼容入口或模型/cache政策。

## 门与实际证据

| 门 | Verdict | 有限证据 |
|---|---|---|
| scope | PASS | 真实两输出反证，合法 SCOPED e96→AUDITED70a→CANDIDATE7b；有限来源缺口，完整G03保留 |
| architecture | PASS | 原 MLX/import/F1/F2/recording owner；固定三peer实际原档复核、Abs反偏置与页attention主旅程；无第二queue/controller |
| correctness | PASS | [source](../../../../evidence/correctness/inference-g03-apple-copy-source-stage/README.md)：来源64/two slots/精确bits/旧owner/partial/容量/foreign/domain spoof/原writer rejection；F16/BF16原生producer→dirty页→attention128；copy8/实际风险5/直接joint768 |
| delivery | PASS | [validation](../../../../evidence/correctness/inference-g03-apple-copy-source-validation/README.md)：source与relocated static/shared，原公共compute64/native128与graph joint768每包实跑；每包4 strict独立头，consumer/安装头匹配canonical |
| usability_docs | PASS | 现有 bind_source 声明和[Apple guide](../../../user/apple-compute.md)说明来源家族、借用/保活、容量、永久只读、显式等待和原reset/drain；两SDK不借内部target/hooks |
| red_team | PASS | typed ImportedArrayStorage而非公开domain/host accessible作为证明；failure不改binding；mutable检查未松动；原真实Metal device与范围/alias核验保；三hostsan actualPASS |
| engineering_performance | PASS | P0有界source资格，零新增字段/copy/wait；注入allocator实际64步bind/prepare无新增HYLE allocation，去逐导出program/slot构造；无新数学算法，不重跑原P1 |

所有产品源码/公共consumer对应候选字节。source记录31.444秒；三hostsan与两SDK记录
95.102秒，全部真实terminal0，heavy串行/最多4jobs。ASan无LSan；TSan复用已有独立
证明的final-counter与同Metal command-buffer模型、raw保字节，不用suppression，不改
产品/util/cache。GPU shader/driver/private pool不属于这些hostsan的资格声明。
Windows没有可适用Apple binary journey，不制造Windows GPU PASS。
最终[机械裁决与selected gate回映](../../../../evidence/correctness/inference-g03-apple-copy-source-validation/validation-result.json)
保本次terminal缺入口链接的初始docs FAIL与修正后PASS；registry转换以7b949b2b0作改前身份。

[统一 LOC self-check](../../../../evidence/correctness/inference-g03-apple-copy-source-validation/loc-report.json)
固定 dab23：全库 code2920606、自实现不含测试176461、本owner src4371。
后者仅compute_route_apple实现，不是整个Compute或引擎胶水估算。

## 限制与停止

readonly import 自身仍有metadata/opaque MLX descriptor成本；producer必须成功ready，
输入准备、producer-ready、copy及原attention观察的显式host等待保留。原schema版本/
native cold prepare/metadata费用仍事实；此修正不宣称全步零allocation/copy/wait。
旧[反证探针](../../../../evidence/architecture/inference-g03-native-output-copy-shape/README.md)
的四个fixture输入、compile/bind FAIL与累计42.421秒完整保留。

仅本来源资格范围结束。reopen只因已验身份/范围/ownership/失败/退休/费用/交付的
可复现错误或新的真实机制需求。完整pending producer→页→attention仍由
[compute_batch主卡](inference-g03-pending-page-consumption.md)推进；G01/G02未决门与
G04–G10、整个goal均保留，不能由此有限terminal代签完成。
