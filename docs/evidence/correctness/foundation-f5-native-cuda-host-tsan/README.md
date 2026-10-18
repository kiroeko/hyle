# F5 actual Linux CUDA FI宿主TSan数据竞争检查

固定product25e，actual RTX4070/GCC14.2/CUDA13.3/SM89，独立TSan binary tree。
实际commands核对公开D256、原native Direct/Graph、BatchInvocation/coordinator、
leased Graph与FI C++ adapter均有-fsanitize=thread；数学仍原FI真实GPU运行。
公开四域D256、Direct及Graph三个消费者的数据竞争门均PASS，具体风险episode见原日志。
这不代签GPU kernel/driver私有内存或其他平台。

首次公开消费者的数学全部PASS，退出返回66：libnvdiagclt.so.1创建的NVIDIA诊断线程
已结束但未join。完全不链接HYLE的CUDA初始化/分配/释放/primary-context对照精确复现
同一线程报告和返回66。原报告、对照源码及runtime help保存在raw.zip中，不删除原FAIL。

实际GCC libtsan帮助和[官方flags合同](https://github.com/google/sanitizers/wiki/ThreadSanitizerFlags)
确认report_thread_leaks只控制退出时线程泄漏检查；report_bugs仍默认true。
仅用TSAN_OPTIONS=halt_on_error=1:report_thread_leaks=0运行数据竞争门，先在无HYLE对照
确认正常退出，再沿同tree/累计预算运行三个真实消费者。没有suppression，没有改exitcode，
没有产品改动。线程生命周期facet仍INCONCLUSIVE_FOREIGN_NVIDIA_DIAGNOSTIC_THREAD，
不能称整套TSan全部通过。进程局部setarch -R沿已资格化的运行方式，不改系统设置。

原配置8.486/300、构建803.101/1800、消费者累计10.536/180；原失败计入。
[receipt](receipt.json)保存候选、身份、全部stage/hash、实际instrumented commands及精确scope；
raw.zip保存全部原日志/控制源码/帮助信息。此证据只签宿主数据竞争facet，不签完整F5 closure。
