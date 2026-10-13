# G01 Vulkan native ragged 与固定记录有限轮终审

状态：CLOSED，2026-10-13；owner `compute_route_vulkan`、`compute_provider_ggml_vulkan`、
`compute_ggml_runtime`。产品candidate `bc0ba41f714f6529cb1c583bee59d482946e7589`，授权与
NR/RF-01..04见[原NR卡](inference-g01-native-ragged-layout.md)及[RF卡](inference-g01-vulkan-ragged-region-fusion.md)。
只关闭本有限F16 ragged/记录消费域；整个G01尚未完成，原单请求3d92/a400、其他数学/
runtime范围与全部失败保留。

## 七门与固定裁决

| Gate | 实际证据与边界 | Verdict |
|---|---|---|
| scope | F16 packed NHD、GQA8/2、D64/128/256，两/八requests、empty/reorder、capacity/tail/readonly及原slots/旧输出；固定NR/RF集合 | PASS |
| architecture | 全部外部View先行，原SDPA后相邻内部dense reshape留在一个原managed region，单Concat；原Work/Queue/权限/退休复用 | PASS |
| correctness | [RR证据](../../../../evidence/correctness/inference-g01-vulkan-ragged-recording/README.md)：五route＋五native共十source tests、KV根/byte origin、外部View alias负空间、实际suffix失败退休；三hostsan各十tests且含native Work的ggml-vulkan.cpp真实插桩 | PASS |
| delivery | 同上source/实际relocated static/shared各两public tests，SDK只用installed headers与hyle::compute；private结构/记录tests只签source。公开头输入不变，原逐头编译证据按相同输入复用；Windows四jobs定向编译9.734s | PASS |
| usability_docs | [Vulkan指南](../../../user/vulkan-compute.md)公开prepare_ragged_batch→原Program/slots/ready输入→Completion/owning output；cold layout变化、旧result/close及native/private限制可发现 | PASS |
| red_team | 相邻内部reshape与外部capacity alias分开；原CPU+Vulkan exact/control/state直接consumer增量build10.093s/run3.805s；候选记录生命周期反证与有限独立复检 | PASS |
| engineering_performance | P0实际3managed段→1、外部temporary blocks3→1，KV根不物化；[最终P1](../../../../evidence/performance/inference-g01-native-ragged-vulkan-rr-m1-p1/README.md)QUALIFIED/5%非劣PASS、原verify0 | PASS |

借固定ggml456172ec原FullF32 tagged flash/shader/tuning与native APIs，依赖pin/许可不变；
Provider只是借原reshape恢复，无新数学。冷region规则不吞并外部View，公共对象/API/
layout与原单请求domain不变，无新Queue/controller/线程/FFI/兼容层。Concat与query/
output O(QHD)、causal mask仍收费；native/private上界UNKNOWN，不签零总分配或总内存cap。

原fixed_attention_graph只允许一个flash且拒Concat，实际新native test首次wait后
recording_ready=false。RF-04仅允许多个同FullF32 tag flash及原Concat opcode，实际
record_safe、无memcpy/memset/event/semaphore与GC安全门仍权威。64步两flash/Concat
读取新payload、step32 invalidate重录、原四native风险tests通过；bindings/owner变化
仍失效，accepted failure与device lost沿原真实退休后cleanup，不推广一般capture。

## 性能、成本与停止

实际Air Mac16,12/M4、macOS27.0.1、AppleClang21，M1-MAC engineering interactive QoS；
clean source4f224878a、contractd9a54a85、productbc0。两边同原Program/Invocation/两slots、
F16 packed ready sources、checked completion及一个packed owning output，对照独立
原flash shader recipe，原oracle2e-3/readonly/tail/peer/5%/16paired rounds/min1/4ms/
n15/31/controls不改。16 A/A、n15产品120/controls30，931.800s、原verify退出0。

| 点 | candidate/peer吞吐比范围 | 5%非劣 / practical band | 相对1方向 |
|---|---|---|---|
| Q1/17 KV1024/768 D256 causal | 1.0545–1.0642 | PASS / INCONCLUSIVE | WIN |
| Q64/128同KV长prefill | 0.9993–1.0162 | PASS / PARITY | INCONCLUSIVE |
| Q1/17 KV31/47 D64 noncausal | 1.0014–1.0084 | PASS / PARITY | INCONCLUSIVE |

原280短D64 LOSS11–13%/qualified FAIL928.101s、融合0e9 LOSS17–20%/qualified FAIL
929.176s永久保[增量账](../../evidence/performance/inference-resident-g01-2026-10.md)。只有
真实机制修正后才验证新candidate，未在同candidate上反复采样寻找PASS。三次Vulkan
multi正式累计2789.077s，全部原jobs/raw及diagnostic/compile失败保留。Cold规则修正
与五route tests在最终bc0内，旧0e9源码身份检查不被冒充成功性能复用证书。

宿主ASan detect_leaks=0，原平台限制不签LSan/shader/driver/privateGPU内存；实际
Windows仅定向编译，不借Air性能签Windows/NVIDIA。selector净diff introducing-commit
归因BLOCKED独立保[工具记录](../build/validation-selector-shared-bench-paths.md)，合法
scope/audit提交和真实对应门替代取证，不修改公共工具/不重开无改动模块。其结果不
改名PASS，也不让模块无限扩仪器；普通docs/closure均实际通过。

统一LOC工具@bc0/self-check PASS：全库2,883,171，自实现不含测试175,077，三owner
实现目录联合11,857，联合数不冒充单factor。有限FIX及七门均裁决，三个owner本轮
CLOSED并停止重复已签性能。分页/联合准确权限G02、producerG03、metadata/growthG06、
private CPU预算G10及其它已准入G域继续属于原goal，不签全部模式对称、独立硬件
Formal或完整引擎token/s。公开MemoryResource default构造反例仍归G10/交付待修。

reopen trigger：本已验域可复现数学/直接KV/常态费用、owner或实际退休/失效错误、
allocator失败错误，或具名真实消费者需要原合同未有的机制；历史TODO或再看一遍不重开。
