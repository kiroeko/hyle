# AA-Q2 模块验证记录修复

baseline `d4b13426`，F1 产品 `ae270af9` 不变；
[有限 ledger](../../../docs/implementation/tasks/compute/llm-moe-c1-normalization-aa-qualification.md#q2-audited固定实现与验收)。
本节只证明记录修复和冷检查，不是性能资格 PASS；Q1 历史异常与原统计门保持。

## 冷检查

[结果](cold-result.json)锁定[cold-raw-bundle.zip](cold-raw-bundle.zip)的SHA-256，包含当前验证代码、
构建/执行命令、原始stdout/stderr、进程身份、32份vendor JIT、逐实例观测及11项Python测试日志。
两个真实verify-only进程分别为primary oneDNN/oneDNN和placement Native/Native：32个workload、
64个实例通过独立数值与完成检查，没有进入`Bench::compare`，没有性能样本。

增量构建只编译/链接两个benchmark，产品库未重编。两种EXE的.text差不超过原4096字节界；
真实EXE另在Windows任务scratch中按hash归档，可据receipt核对。DLL已邻置，PATH隔离、禁弹窗，
沿原W1 P核策略运行。可见的是opaque owner handles、线程/CPU及当前进程private executable
地址范围；它们不冒称已把每个JIT函数映射到role，也不能补回或解释历史21%异常。

源码核对证明两个`run()`、Problem数据和独立oracle未变，contract只增加模块记录目录及冷诊断
开关。原5%固定因素/总体中位数门与10%单实例gross guard保持；测试拒绝历史1.21435、倒数和
固定因素偏差，并检查独立记录、防覆盖、失败/timeout原字节保留、缺失/重复或错误实例事实。

冷检查后仅修正Python `main`的失败清理：首次调用新建的status也必须重新发现并写FAIL。
第11项测试证明这一点；AST核对所有冷检查使用的helper及C++制品均未变，按身份复用两个冷
进程，不为该错误路径改动重复运行数值或P1。原始冷时脚本与此复用核对均在ZIP中。

模块测试位于自身bench目录，沿既有selector映射；公共runner只接受受审HYLE诊断变量，vendor
dump环境由模块自己的child环境清除/设置，普通采样EXE拒绝未授权dump。公共selector/runner/
统计核心没有改变。P1须在候选及状态冻结后另行启动唯一实验，未运行项不写PASS。

## 唯一恢复实验失败

实际干净运行HEAD为`0e107a06dd12a549c8cf5354b0cff1dd6b9e97c6`，实现候选`07a69e60`。
[失败结果](failed-experiment.json)锁定[完整原始ZIP](failed-experiment-raw.zip)；ZIP经过CRC及
每项原字节核验，包含16个A/A child的stdout/stderr、退出、JIT、实例/代码范围/粗间隔、机器/
制品/源码身份、合同、公共validator日志、launcher和可用构建记录。未生成公共完整run manifest；
不能冒充完成一轮产品采样或deep verify。

- **记录接线FAIL**：公共validator只展开完整token，`{output}/normalization-aa-runs`保持
  字面值并在源树创建未跟踪目录。公共runner正确拒绝被validator改变的候选输入。
- **A/A资格FAIL**：16个进程均正常返回，256个比值/512个实例保留；其中oneDNN对自身的
  `direct-layer-17x257`，primary、peer先执行/candidate先分配，吞吐比0.906387684990405，
  倒数1.1032806563458395，超过原10%单实例门。修正记录路径不能消除这个失败。
- **产品性能INCONCLUSIVE**：产品采样进程0；不得判Native或oneDNN胜负、签默认路由/淘汰。
  唯一实验预算已结束，不追加原样重试、冷执行或诊断计时。原F1正确性/三交付证据保持。

失败实例实际均为oneDNN、共享输入、独立输出/workspace，观测CPU2。粗wall/线程CPU统计
覆盖整个比较，无法归因到某个arm；线程计时粒度还允许量化误差。JIT字节和private executable
范围不等于逐函数/role地址映射。本证据没有证明调度、SIMD、JIT地址或其他单一根因。

原始字面目录在核验归档后安全移到`.scratch/validation/onednn-aa/q2/failed-record-root`，
保留原argv/cwd；仓库根无此垃圾目录。Q2-D只修模块参数与目录命名并做不计时回归；公共
仪器、门限和产品C++不变。当前失败与原Q1异常均保留，不能用日志修复标记性能PASS。

## 接线修复收口

`7f625eb4`的[修复收据](path-fix-result.json)与[12项Python测试](path-fix-tests.txt)通过：
合同使用完整`{output}` token，由模块追加记录namespace；实际公共expander的带空格路径
回归覆盖到模块错误落盘。AST核对只改`main`，合同只改一个参数；不重建、不启动native
benchmark/冷检查/P1。原323份临时记录移位后再次与ZIP逐字节匹配。

资格仍FAIL、产品性能INCONCLUSIVE，两个owner按原预算进入BLOCKED；恢复要求见
[终局](../../../docs/implementation/tasks/compute/llm-moe-c1-normalization-aa-qualification.md#q2-终局blocked)。
[完整实现LOC](all-implementation-loc.json)为48,294行/195文件，自检PASS，不包含tests/bench/vendor。
