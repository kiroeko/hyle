# F5 CUDA native风险测试TIMEOUT

实际root `/root/hyle-scratch/compute-unified/foundation-f5-native-direct`，原配置、原binary
tree、原累计预算。新增fault consumer只装饰一次原CUDA finish_sequence的Completion
观察，保原GPU kernel/event/reaper；Gate用原cudaLaunchHostFunc阻塞native stream。意图
检验unknown observation后F5实际guard/drain，不以此fixture签native数学。

build-resume12 PASS；consumer-resume10在remaining167.929秒终端TIMEOUT/exit-9，stdout
为空。截至timeout累计configure16.403、build465.981、consumer180.030/180秒。runner
已终止，记录PID873的/proc不存在；没有live job待恢复。准确命令、原全部历史与log hash
见receipt/raw.zip；fixture-input.cc保存实际编译输入与hash。

没有运行到断言，因此不能声称已证明F5 drain产品缺陷，也不能把source审查推断签PASS。
进程曾实际live，主线程running、其余CUDA/reaper线程poll/futex；这不足以定位阻塞API。
原“drain只观察Completion可能先返回”的疑点仍INCONCLUSIVE。不得因观测超时直接重启/
换目录/转另一个gate预算，也不删除原TIMEOUT。

按活动F5 card预算停止规则，compute_batch进入BLOCKED from AUDITED，aggregate风险
INCONCLUSIVE；其余已通过正常/append证据保留，整个goal没有完成。恢复需明确批准同
root/config/tree一次有限追加，再SCOPED→AUDITED冻结只本风险的定位/修正；保全部旧失败
与计账。未批准前不运行本门或修改blocked owner产品。
