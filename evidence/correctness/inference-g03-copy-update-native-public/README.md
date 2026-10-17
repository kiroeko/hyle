# G03 Apple/Vulkan 公共 CopyBatchUpdate 旅程

2026-10-14，按[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
运行，AUDITED WORKTREE；[公共 consumer](consumer.cc)、[Apple](apple-final.log)、
[Vulkan](vulkan-final.log)、[累计记录](receipt.json)、[源码字节](source-identity.json)、
[所有输入/失败/编排](raw.zip)。MacBook Air Mac16,12 / arm64 / macOS27.0.1 / AppleClang21；
真实Apple与非software Vulkan，原source树cached builds，4jobs串行，仅hyle::compute。

同一public源码在两原namespace均通过：1次16384bytes初始化+64次1024bytes dirty更新；
foreign pair失败不消费，move同时移空caller，成功mutable返回保generation/untouched历史，
回收同一cold slot、metadata_bytes不变，重复返回writer拒绝。每步原F2 readonly回读至
staging再独立比较全部16384bytes；不把Vulkan device资源当host-readable，即使是UMA。
source/target原use实际退休。仅原copy字节资格，不签attention数学或全G03。

最初Vulkanfixture漏显式MemoryKind导致编译FAIL；修正后设备直接host读被原合同拒绝
(Vulkan visibility range requires staging)。两次FAIL和所有input保留，最终改用原stage
readonly copy，产品未因此更改。Apple先前PASS重复记录也保留，不重置累计预算。
Hostsan、relocated SDK、独立头、Windows与其余source生命周期仍待，不代签CLOSED。

实际累计111.774秒。
