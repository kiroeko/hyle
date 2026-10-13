# G01 原生 Vulkan attention 精度资格

状态：AUDITED，2026-10-13。新增owner：`compute_ggml_runtime`；baseline：
`d97126cfdfa3b73ebf385fb95be27d989ef5f683`。原Provider owner仍按
[resident SDPA卡](inference-g01-resident-sdpa.md)AUDITED执行。完整授权来自
[缺口goal](../project/inference-gaps-goal-execution-2026-10.md)；不是重新检查CLOSED来找问题。

新事实：[实际定向失败](../../../../evidence/correctness/inference-g01-resident-stage/README.md)
证明锁定Vulkan原FLASH_ATTN_EXT的F32 accumulator仍可能half缩窄Q/K/V/概率。
`flash_attn_cm2.comp`原decode/coopmat强制half，原scalar也按device.fp16选half shader；
只改Provider lower不能兑现原HYLE数值合同。原资源/CPU/CUDA/Metal有限CLOSED证据保留。

主consumer是公开GQA8/2 D256驻留cache decode/prefill，正交为D64、两个batch、非零offset、
causal/noncausal、source只读、独立slots及旧ResultLease/Tensor。数学oracle和容差不放宽，
不通过CPUfallback、环境变量改全device能力或新通用attention kernel绕过。

有限调查与拟FIX：借已有全F32 scalar flash shader，核原FA tuning/dispatch/pipeline cache
身份和scratch，提供只对该数学档生效的冷原生选择。与accumulator设置分开核对，默认
ggml调用和其他operation不得被隐式切换；私有固定源扩展若必要，要确认原op参数槽、
cache key与升级拒绝，不能复用旧half pipeline或运行期改全局features。

实际锁源码已核：FA只有前三个float参数和slot3 accumulator，slot4未使用；全F32 scalar
shader已生成嵌入，支持原F16/F32 KV解码。冻结私有slot4 tag由既有C++17 native_bridge
定义，不新增公共API/FFI。native仅在tag匹配且K/V为F16/F32、D64/128/256同D时准入；
原normal调用tag0与其他operation不变。三个有限FIX固定后进入AUDITED：

| FIX | 最终承载 | 验收 |
|---|---|---|
| NP-01 | 原FA slot4显式full-F32 tag；强制已有scalar tuning，float shared-memory核计；不改device features，禁用该档的KV dequant整历史scratch | 三个D、短/长Q、GQA、有效长度/offset；native范围/能力不符冷拒绝 |
| NP-02 | pipeline key追加full-F32位；只该位选择现有fp32 SPIR-V，保原mask/spec constants/stride/submit/retirement | 同Context交错strict/normal及独立slot，不混用half pipeline；原资源/失败/OOM路径 |
| NP-03 | Provider只给已准入D64/128/256同D设置tag；其余既有SDPA沿原明确传统lower，不自动CPUfallback | 公开原oracle/容差通过，源码/static/shared；scope外域不代签高效读取 |

改变的是私有固定源dispatch与cold cache身份，无新数学shader、线程/queue/runtime或
公共layout。strict每调用只有一次原参数位检查，cache首次准备才编译原SPIR-V；float
shared-memory和scalar成本明确计费，是否快由同合同P1裁决，不能以少score字节推断吞吐。
SCOPED提交`52e50fe6f`保授权顺序；三Provider/native owner实现前都需AUDITED。

原生源/metadata实际owner为compute_ggml_runtime；若修改锁定integration patch，使用既有
manifest before/after hash和依赖锁认证，原cache source不写。只修原承载精度选择和完整
源码交付，不顺手重写CMake、checker、性能统计、runtime/Queue或所有FA优化。中央DAG
和selector给影响，源码字节不变的CPU/CUDA/Metal数学不被重新打开或代签。

预声明验证：上述public小consumer与对应原生图/slot失败/OOM/lease/retirement；旧有效
Vulkan SDPA域和直接消费者按实际影响。source及relocated static/shared实际运行，同
源码/精度/位置/有效区；适用宿主ASan/UBSan/TSan，driver/private UNKNOWN仍明确。P0
记录full-F32 shader的mask/Q cast/scratch及native plan，排除Q/H展开完整KV；candidate后
借原full-F32 native shader同合同定向P1，沿原卡10%非劣margin/双顺序/A-A，不冒充SOTA。

调查检查点2小时、实现/验证检查点各4小时，正常推进自动延长并保原FAIL/TIMEOUT/elapsed；
本机编译4 jobs按内存下调，重型GPU测试继续避开桌面资源压力，Air实查身份后可远端跑。
有限问题真实裁决后才CLOSED；真实外部阻断按协议记录精确条件。当前尚未修改原生源码，
不因阶段性CPU成功把G01/G02或全部goal结束。
