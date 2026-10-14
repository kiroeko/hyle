# G03 原 copy update Apple source 实跑

2026-10-14，按[活动卡](../../../docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
执行；AUDITED WORKTREE，尚未注册不可变 candidate，CU03其余路线/风险/SDK未签。
[公共与私有直接consumer](consumer.cc)、[实际日志](final.log)、[累计记录](receipt.json)、
[源码字节](source-identity.json)、[全部输入与原始编排](raw.zip)。
MacBook Air Mac16,12 / arm64 / macOS27.0.1 / AppleClang21；原cached source-parent/build，4jobs串行。

CU01公共CopyBatchUpdate只移入两个原controls，16bytes：foreign pair失败不消费，成功
同时移空caller；成功、取消、前缀失败、放弃、closed五实际pendingGPU场景通过。
原source guard保持至实际退休；成功回接原writer/generation/exact frame/cold slot，失败
无ready页。跨fixture线程只证明唯一观察owner交接，不创造产品执行owner。

CU02实际pending original writer借用保持geometry/root/last-write，源-only与坏ID拒绝；
最后independent empty logical phase已inline仍保前面真实native prefix。原writer public
view仍拒绝pending读。实际trusted successor另核同Runtime，foreign Runtime拒绝；复用
原ordered_read_unique与原CopyRecording/worker/preflight，最终native copy读取256bytes
全部31。成功/取消/前缀失败三场景均传原Status；失败native_calls=0，无成功页；最终
retirement释放原ordered read guard。只有suffix.wait，中间无producer/copy host wait。
这证明source seam与原native copy消费者，不代签pending attention/late native output。

三次input版本与全部configure/build/run记录保留，累计46.699秒，全部terminal0。
没有改原队列/Completion/worker，沒有额外公共对象常态字段/锁/原子/metadata分配。
原Apple每native phase worker的Metal同步成本仍存在；没有全模型/数学P1、shader/driver
sanitizer、其余平台/SDK资格或所有模式完全对称的主张。
