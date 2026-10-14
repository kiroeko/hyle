# G03 原 native dependency 的取消接纳边界

状态：SCOPED，2026-10-14；owner compute_route_apple、compute_route_vulkan；
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

有限 FIX 待在 AUDITED 固定；SCOPED 不授权产品修改。
