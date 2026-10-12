# F5 原Graph region native batch首批

AUDITED实现checkpoint，未签candidate/全F5。实际WSL RTX4070/GCC14.2/NVCC13.3，原锁
FlashInfer0.6.18/69ff11fc；普通Compute/Graph产品入口、原Graph v19共享region执行器，
仅新batch/实际owner闭包，没有回放原算子测试。机器身份、精确命令、各轮源码hash、累计
budget和原日志见receipt/raw.zip；原source-build树复用，库按可用内存j4/j2/j1、test1。

paged F16 D64短KV与ragged BF16 D128长KV，各两节点串联，共享一个IR metadata输入。
同Context/backend session、同原region invocation的native slot准备及执行，外部F1输出直接
由原kernel写入；第二层Q消费第一层实际结果，对独立double dense oracle。两个独立Graph
invocation各有自己的计划/工作区；第三个超过冷max_invocations被拒绝。首布局双slot及
拓扑改变后重plan，共12次数值oracle PASS；长KV由原split-KV/merge承载。

共享输入只允许完全相同的owned BatchLayoutVersion及device资源域。每region仍规划和
上传自己的固定镜像，显式保活leader只读镜像供同输入消费；没有省略这额外copy计费。
一次prepare先预检完整node/layout集合，duplicate、foreign schema及同输入异版本在改动
前拒绝。caller替换library metadata拒绝；old output/原completion通过原LeasedInvocation
控制门阻止overwrite；更新先释放全部共享readonly引用，再回收writer，不绕过F1权限。
部分准备失败清掉所有batch token及Graph绑定，不能提交新旧计划拼接的批次。

metadata必须是显式host-known Graph输入，constant/device-produced metadata未准入。
原Graph vtable版本、普通Invocation布局/submit无新增规划或DAG扫描，执行时只消费已准备
token；原Graph Completion保region/slot/native资源到实际退休。最终metadata quota归零。

首次单节点四oracle PASS原日志保留；本轮扩双节点/共享输入/双slots后12oracle PASS。
旧LibTorch CUDA12与runtime13的构建提示保留，没有升级锁。不是capture/P1或Windows CUDA
native paged资格，CPU/Apple/Vulkan显式reference、append/F4完整闭环、其他故障及三交付/
san/P0/终审仍待执行。
