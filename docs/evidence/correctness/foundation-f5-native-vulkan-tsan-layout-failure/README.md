# F5 Air Vulkan TSan的IREE配置反证

固定F5 product25e、Mac16,12/M4/Clang21。配置/构建PASS，公开consumer先ragged F16
数学PASS，后IREE semaphore wait count=4560304736超过255而退出1；后native风险未跑。
原FAIL/命令/耗时/hash保留，不改错误码或suppress。

实际adapter用-fsanitize=thread，IREE C runtime未用IREE_ENABLE_TSAN。锁定operation.h
在IREE_SANITIZER_THREAD时有tsan_bridge字段；该宏由编译器instrumentation自动推导。
同一实际headers/配置的sizeof/offsetof探针：instrumented operation64/wait104/count80/
semaphores64；uninstrumented operation56/wait96/count72/semaphores56。
因而runtime把values指针位置读成count，符合实际巨值，不能以更改batch数学修该错误。

源owner的CMake只设置IREE_ENABLE_ASAN，未同步TSan。此为具体san配置ABI反证，触发
compute_provider_iree_vulkan独立窄reopen；普通san=off算子/API/SDK此前资格不因该证据
自动撤销。原TSan root configure14.050/300、build210.942/1800、consumer3.833/180。
继续验证必须同root/累计预算，不冒用新名字重新计时。原flags、探针源码/完整输出及
实际vendor header见raw.zip；本证据为FAIL，未签任何后续修复或最终closure。
