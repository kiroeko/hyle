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
