# G03 pending copy 原控制所有权交接反证

2026-10-14，SCOPED compute_batch，固定产品 dab23，未修改产品。
[原型](prototype.cc)、[实际结果](final.log)、[累计记录](receipt.json)、
[源码字节](source-identity.json)、[输入与原始编排](raw.zip)。MacBook Air Mac16,12 /
arm64 / macOS27.0.1 / Apple Clang21；source target hyle::graph，仅直接有限fixture。

同一原 pending CopyInvocation/CopySubmission 两个move-only控制一起移交给唯一观察者，
源两handle均失效；16-byte局部owner，无新queue/Completion/state/default field。
实测原GPU checkpoint阻塞时仍无output/returned writer，原F1 source use保持；跨fixture
线程交接后，成功、取消、原前缀失败、pending放弃、closed五场景全部通过。成功回传原
cold invocation，metadata bytes不变，原GP02 writer回接同generation/exact frame；
失败/放弃不伪造可重用页，abandon无隐藏wait但native实际退休前保原guard。

这改变设计下一步：组合owner应移动两个原控制并保持单一观察者，避免把public控制留
在caller、仅另持opaque别名，从而逼普通路径新增共享observer状态/锁/原子协议。
fixture线程仅证明所有权handoff；产品不创建线程/第二runtime。fixture显式copy wait仍
存在，本证据不签native dependency、pending读取/attention或四路线公共交付完整。
累计11.905秒（configure/build/run），全部terminal0，四jobs/heavy串行，无全模型/P1。
