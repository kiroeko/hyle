# CPU最终F1：部分证据

当前任务为[F1](../../../docs/implementation/tasks/compute/cpu-final-f1-build-contract.md)，
总目标为[三平台CPU最终收口](../../../docs/implementation/tasks/compute/cpu-final-closure-plan.md)。
本页不是terminal，不能签F1或整个goal完成。

## B1构建资格与可用性

- 产品`adc794aa3384499d23b6c9b054570b017b236fae`：flags/IPO/linkage等不匹配撤旧WIN，成熟数学保留。
- 私有构建metadata合同20格实际通过；后续`688d28f15361ed33de1c528fd865377c1742749c`修正CMake编译器尾`.0`写法，
  同20格重新通过。该metadata检查不冒充真正compiler/CPU运行。
- Windows shared source/public-facade的24格Direct/Graph、24成熟region、两次变输入replay、9错配、
  独立并发槽/close及真实动态view物化实际通过，Native WIN标签0。
- [receipt](b1-receipt.json)、[原始configure/build/run与普通consumer](b1-build-default.zip)。
  没有新数学或新赢家P1，未签新的shared性能优势。

## B5 oneDNN升级

采用[官方稳定版v3.13.4](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13.4)，
固定commit`c531fe1a5cd631b7f60277cdf6675c588cadbc28`，source/archive/Git tree身份已核对并进入中央锁。
11个具名seam中C API/BRGEMM/threadpool/stream/primitive运行时及LICENSE相同，
C++ header增加verbose查询、options增加默认OFF的SYCL profiling，原资源overlay仍精确匹配。
两adapter的版本由真实header/同一CPU版本owner投影；旧Native17/187不自动沿用新版oneDNN。

产品`688d28f15361ed33de1c528fd865377c1742749c`编译成功，Windows shared owning四组最终通过。
初次四组中组合后备测试仍要求已退役Native MatMul，原失败保留；测试候选
`d2ffda17793cfef84eb9be2a29652aee8eb198f5`迁移为ATen真实转置矩阵两次变输入执行，
显式裁剪ATen时检查明确拒绝，不复活自研。修正后四组、24格普通Direct/Graph与动态/并发/close通过。
[升级receipt](b5-receipt.json)、[原失败/纠正/新源码身份及完整日志](b5-onednn-upgrade.zip)。
最终三平台三交付未完成，不签整包PASS，没有运行赢家P1。
其它库锁与GPU产品没有改变；中央lock路径产生的selector全库superset原计划保留，
无oneDNN依赖的目标有语义不适用记录，不把未运行全库suite写PASS。

Mac Air已读取当前真实身份且可达，尚未运行新版CPU产品；macOS覆盖仍未资格。
当前统一Compute+Graph实现**61,084行/249文件**，不含测试/vendor/bench；新增版本身份胶水，新增自研数学0。

[清理记录](cleanup.json)：已校验归档与锁定cache后，删除本轮重复oneDNN源码/下载约87.8 MiB；
保留当前producer、统一锁定source/archive cache与全部原始资格记录。

## B4逐合同工程默认

候选`6216e6d0651ad2684950abec036d8749ccc3c3a1`，Windows Direct16/Graph15无skip、无fail；
原24格/变输入重放、完整region、错配/并发/close与动态物化通过。
[receipt](b4-receipt.json)、[原失败/纠正与完整日志](b4-engineering-defaults.zip)。
原库能力/配置/生命周期及PreparedVTable保留，静态cold表不增加Context allocation；expert不带普通默认policy。
这是工程默认，不是新性能WIN；其它平台和最终交付未验，本包仍未关闭。
