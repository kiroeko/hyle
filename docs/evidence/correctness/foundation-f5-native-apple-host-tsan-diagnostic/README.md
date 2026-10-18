# F5 actual Air Apple TSan报告与无HYLE对照

固定product25e、Mac16,12/M4/macOS27.0.1/Clang21，原宿主instrumented配置构建PASS。
公开消费者两次均在ragged F16数学PASS后，进入后续cold prepare时返回-6：GCD worker读
MLX array::Data的std::function函数指针deleter，HYLE cold worker经MLX set_data构造时写。
两原FAIL/日志/elapsed保留；没有suppression、关闭race检查、改exitcode或产品改动。

第一个纯MLX square对照与第二个attention DAG对照未复现，不能藏掉这些差异。
第三个对照显式实例化相同function-pointer deleter模板，仍完全不链接HYLE，只用同一
MLX静态库/Clang/Metal，在原zeros→compile→async_eval→synchronize→wait的cold线程序列
复现相同读/写模板栈、64-byte Data分配、GCD worker、返回-6。对照源码、全部命令与
日志均在raw.zip；说明报告不依赖F5实现，但不自动证明这是假阳性或实际外库缺陷。

实际vendor CMake明确USE_TSAN在Darwin不支持；该库本配置没有vendor instrumentation。
因此本门仍INCONCLUSIVE_MLX_METAL_HOST_RACE_REPORT，实际consumer原结果FAIL，native
风险消费者尚未运行，不能写Apple TSan PASS/语义NA。已有ASan/UBSan和实际故障/权限
证据仍保各自facet，不能代签本并发门。后续须有真实同步/仪器或外库资格证据才能裁决。
预算同root configure300/build1800/两consumer180累计，不因对照/root/候选重置。
receipt/raw hashes和实际build commands保存，不签完整F5/goal完成。
