# CPU ATen ScatterAdd：AS1–3 资格

产品candidate `e13edaebe0aac974fd2f7d2ff50246dfd61ad330`；fixture `7356a06f`。
Windows canonical 产品，Linux为前向验证副本。两平台1337个产品/test/构建/consumer输入SHA256匹配，
已锁Torch2.8 CPU、x86_64；没有macOS SDK资格、Native/SIMD性能比较或新数学kernel。

| 实际门 | Windows / Linux |
|---|---|
| Native factor，含独立ATen factory | 每平台140/140，无失败/skip |
| 普通CPU受影响测试 | 每平台12通过/1旧fixture skip |
| 普通Graph受影响测试 | 每平台33通过/6旧fixture skip |
| 公开头 | 每平台1/1 |
| 相同ordinary Direct/Graph | source shared/static、installed shared/static，两个平台全部实际通过 |

新增6个批量case覆盖F32/F16/BF16、I32/I64、rank3各轴/负轴、rank8、scalar/多维/empty indices、
odd-byte stride、重复updates、empty output；NaN/±Inf、empty Half base的原负零/signalling-NaN bits；
三dtype exact-dense input0 alias、拒绝updates alias、变化输入/真实wait、late越界输出不变与
失败槽重建、close、两线程独立queue/invocation及Stats平衡；每个被跟踪的cold prepare/invocation
OOM点live-bytes回滚与重试。原native物理OOM/noncoherent故障未注入，只审计异常/flush-close传播。

普通consumer使用F32 I64 duplicate多维轴0、非末轴scalar、odd strides与empty indices，op→Copy
Graph下游、变化输入和close。三storage数学在公共入口factor实测，不把F32 consumer宣称全部模型。
原`index_add_`的[156个SDK断言](../cpu-resource-domain-audit/README.md)证明借用候选可用；本批执行
与交付才证明实际产品接线。Add在F32累加base与全部updates，再一次storage舍入，符合公共更宽
累加/重排合同；empty indices走原Byte copy。Assign保持既有stable last-write，未改其数学。

P0完整成本：byte packing、原Float转换/clone/index-add、输出cast/copy、原Tensor metadata/临时
存储和线程；没有给Add重复分配legacy Gather/Assign的index scratch。managed replay；prepared
artifact retained bytes已知，private invocation与当前compilation声明unknown，不签零热分配/捕获。
严格资源政策与普通域分开；本批不据源码名称、优先级或cold prepare签Native WIN/UNIQUE。

首次MSVC `/WX`拒绝fixture vector构造的隐式I64→I32；显式转换后通过，产品未纠正。原失败log、
源码、消费者、drivers、XML、所有命令和身份/selector/LOC/cleanup在[qualification.zip](qualification.zip)，
结构摘要与hash见[receipt.json](receipt.json)。SDK关闭分支和公开头/assembler保持相同，仅复用该
未变接缝的既有证据，没有新跑完整feature-off SDK。统一实现LOC **61,516 / 249文件**，不含tests/vendor。
四个本批consumer目录已清理 **625,241,686 logical bytes**；SDK/producer/dependency cache保留。

[当前卡](../../../docs/implementation/tasks/compute/cpu-aten-scatter-add-admission.md)关闭仅为AS1–3；
CPU goal仍有精确资源声明与冻结自研的有限去留裁决，不表示Compute+Graph全路线完成。
