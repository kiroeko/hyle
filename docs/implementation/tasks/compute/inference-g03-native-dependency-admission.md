# G03 原 native dependency 的取消接纳边界

状态：CLOSED，2026-10-14；owner compute_route_apple、compute_route_vulkan；
baseline `8512c09ebee5c1efd54d133b4a795df059517dca`。原 source
[terminal](inference-g03-copy-update-terminal.md)保持历史事实；本轮由
[真实 receiver 反例](../../../../evidence/architecture/inference-g03-pending-attention-receiver/README.md)
触发。[主卡](inference-g03-pending-page-consumption.md)仍 SCOPED，完整 pending
frame/late output/最终状态合并另按同一 goal 继续，不由本轮代签。

问题：原 native dependency qualifier 已核原 vtable、同 Runtime、terminal hint 和
已完成错误，却没有核仍在途但已请求取消的原 native state。于是 trusted attention
receiver 在 source.request_cancel 之后仍可通过资格检查并提交。同样 qualifier 是原
F2 bind_producer/prepare 的实际接纳边界，不只是测试辅助函数。正交消费者是普通数值
producer/copy prefix 到 readonly reduction/copy，取消合同不依赖 KV 或模型政策。

有限 scope 是原 Apple copy/native invocation 和 Vulkan Run 的资格检查。保持原
Completion/Runtime/Job/queue/F1/F2；不增加对象字段、锁、原子、分配、copy、native wait
或 export，不修改数学、Batch controller、构建和工具。依赖在检查后才被取消的竞态
仍属于已接纳工作：原实际退休/失败状态必须保留，最终 receiver 聚合仍在主卡。

API 保持原 private qualifier 和公开 F2 bind_producer 签名。已知 terminal 错误先返回
原错误；同原 endpoint 的 pending cancel 返回 CANCELLED，不消费 source Completion，
不录入后继 work、不把请求取消当 actual retirement。已成功退休后的原 cancel 请求
不能撤销成功；foreign endpoint/vtable 原诊断保持。ordinary ready/direct 行为不变。

预声明验证：两平台真实 copy→attention 的取消接纳负例和成功对照；原 copy source/
生命周期消费者，原 native invocation cancel dependency；failed/foreign/unknown/abandon
与 actual guards 保留；原相关 source/static/shared、相关独立头及适用宿主 ASan/
UBSan/TSan（沿已有显式模型，不能签 driver/shader/private native pools）。Windows
只选该 route 的适用编译/link，不运行本地 heavy GPU/fullmodel 或旧全局矩阵。改变
没有数学/normal algorithm，P0 检查原成本结构，不重跑已关闭的数学 P1。

调查/纠正/验证各 2h 检查点，用户常设授权有限追加、原 FAIL/TIMEOUT/累计保留；
heavy 串行、最多四 jobs，Air only 实查身份。冻结 FIX 完成并适用门裁决才关闭；一次
fixture/链接/观察失败先查原因，不重复旧工作或增加全局治理。三种交付公共合同必须
可复核；尚未运行的门不写 PASS。

## 能力与有限审计

| 能力族 | 现状/来源 | 动作与承载 |
|---|---|---|
| 原 endpoint 资格 | 原 vtable/Runtime/known terminal check 已有 | KEEP；不放松 foreign 检查 |
| 已请求取消的 pending dependency 接纳 | 两真实 native attention 反例；原 F2 实际使用同 qualifier | FIX/SELF，仅核原 cancel 状态 |
| 接纳之后的错误聚合 | 排序本身不合并原 source 与 suffix 状态 | 主卡保留；本有限接纳检查不能代签 |
| actual retirement/unknown/abandon | 原 Completion/F1/native owner 已有 | BORROW；取消不释放仍在途 guards |

不引入第三方或 FFI；现有原 route/native 依赖、许可证、升级和 private 成本不变。
基础 error/permission 合同由原 owner 承载，模型请求/evict/采样政策留 consumer。公开
F2 create/bind/prepare/submit/output/close/drain 全部仍回指原 source 能力；本轮仅修
已有 bind_producer/prepare 和 trusted receiver 共用的资格边界。

## 冻结 FIX 与成本

| ID | 固定修改 | 验收 |
|---|---|---|
| DA01 | Apple qualify_recording_dependency 与 qualify_copy_dependency：先保已退休 terminal，然后对原 pending cancel 原子 acquire 返回 Cancelled | gated 原 copy 与原 native producer：取消后不接纳后继；非消费、foreign、原 error 与 actual retirement 保留 |
| DA02 | Vulkan qualify_copy_dependency：在原 Runtime mutex 下先保原 failure/terminal，然后拒绝原 pending cancelled Run | 实际 native queue/fence 取消负例、成功对照、原 unknown/closed/abandon 生命周期；不新增锁或 native wait |
| DA03 | 修正 receiver 验证，把取消发生在资格检查前与接纳后分开；原反例证据不覆盖、不擦除 | 两实际 route，原 source/static/shared 与风险；最终聚合仍待主卡 |

Apple 在原 done acquire 之外新增一次 done acquire 与原 cancel 状态 acquire，只在
trusted dependency 资格检查中；Vulkan
复用已有 mutex、一次原 bool 检查。没有 default layout、热字段、分配、原子写、
线程、queue、结果 owner、buffer 或 native 同步变化。不以该成本检查代签完整 receiver。
只允许 apple_context.cc、apple_copy_recording.mm、vulkan_run.cc 与对应直接 tests/
guide/task/evidence；现有 endpoint/terminal precedence 和公共签名保持。

公开调用仍是原 F2 bind_producer(resource, std::move(completion))；失败保持该
Completion 和原 binding 不变。trusted batch adapter 在原 endpoint qualifier 成功
之后才 admit_read/submit。检查与请求取消由唯一观察 owner 外部串行；source worker
仅发布原 terminal/done。并发取消在检查之后发生，按原 accepted lifetime 处理，
不宣称原 GPU 抢占或最终聚合已经完成。

七问已裁决：无新增第三方/FFI；两个具名纵切与真实反例；复用原 native/F1/F2；
P0 原结构成本；三种交付/相关头适用；错误非消费与指南说明。工程/正确性/交付/
红队仍待实跑，AUDITED 不写 PASS。原 source terminal 的其它资格和失败仍保。

## Candidate 与实际 source 检查

不可变产品 candidate `9404af694`；只改三 qualifier 的原 cancel 检查，公共签名/
layout/数学/默认机制不变。两实际 native source 取消接纳与成功 attention 对照通过，
Apple 同时验证原 native invocation 与 copy recording 的 pending cancel 拒绝；
Vulkan 验证实际 Run/copy fence 的 pending cancel 拒绝。source 控制保留，最终退休后
仍无 mutable success。两 route 各原公共 65 steps、Apple 原 8/Vulkan 原 6 生命周期
亦通过，累计 13.760 秒。这里不宣称 Vulkan managed producer 的原 unsupported
prefix 已变为支持，不签最终 receiver 错误合并/late native 输出。

## 最终证据与裁决

[实际验证](../../../../evidence/correctness/inference-g03-native-dependency-admission/README.md)
封存 source/input 字节与累计、六宿主风险/四 SDK/12 独立头、边界检查、Windows
受影响 Vulkan TU 编译和原机械字段失败。产品 candidate 不变。两路线取消依赖被
拒绝而保原 source，原 source 实际退休与失败/unknown/abandon/foreign 行为保留。
已确认成功之后请求取消不撤销成功；Vulkan suffix 退休尚不等于原 source 状态已
被观察，最初误判的 source/ASan fixture FAIL 保留，修的是测试，不是产品。

source 累计 23.552s、宿主/四 SDK 171.656s、边界检查 22.343s、Windows 2.516s；
四 jobs/heavy 串行，无全模型/nvcc/旧数学 P1。宿主 TSan 原 final-acquire/Metal
显式模型 SHA 实查与已封存字节相同；没有 LSan/shader/driver/privatepool 内部声明。
Windows Apple .mm 不适用，Vulkan 只签编译/link，不伪造 Windows native GPU run。

| Gate | 实际依据 | Verdict |
|---|---|---|
| scope | 真实取消 source→attention 反例；仅 DA01–DA03 | PASS |
| architecture | 三原 qualifier，原 Runtime/Completion/F1/F2，无新执行 owner/FFI/依赖 | PASS |
| correctness | 实际取消拒绝/成功保持、原 public65/Apple8/Vulkan6、六 host 风险、原实际退休 | PASS |
| delivery | 原 source、relocated static/shared 四 SDK/12 头，受影响 Win TU | PASS |
| usability_docs | 原 bind_producer 非消费诊断与 compute guide 取消/退休限制 | PASS |
| red_team | 原 known error/foreign 优先与在途 guard；晚取消先观测成功；fixture FAIL 保留 | PASS |
| engineering_performance | P0 原常态 layout/锁/原子写/分配/queue/math不变，仅有限资格检查 | PASS |

影响 selector READY，保 skipped reasons；自动一跳 Graph 结构依赖按实际 qualifier
只由原 F2 调用的语义、原 native 生命周期/Batch 接纳/SDK 链接回映。没有 Graph
bridge/math/status 修改，不重跑已闭数学或代签旧 Graph Windows 未决门。
首 candidate registry 多带 performance 字段的 checker FAIL 在 8950153ac 修正；
最终 closure transition 与 docs/diff 使用当前实际状态运行，未修改公共检查工具。

LOC 固定 940 self-check：全库 2947154，自实现不含测试 176785，两 owner src
显式并集 13475。停止本有限轮：DA01–DA03完成。只有已验取消接纳/原错误/寿命/
成本/交付的可复现新缺陷或真实新的 dependency 机制触发新 scope。

完整 receiver 聚合/pending frame/late native output 仍[主卡](inference-g03-pending-page-consumption.md)
SCOPED；本卡 CLOSED 不授权改已关闭 source 或三 qualifier，不签完整 G03/G01/
G02 或 G04–G10，goal 继续 active。
