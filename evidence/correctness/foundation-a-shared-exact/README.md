# Foundation A 共同精确形状：首批实现证据

当前仅签共同 metadata/static 资格；A 的 native/hybrid、交付、sanitizer、effects 与红队没有由
本目录代签。活动卡是[四路线对称性](../../../docs/implementation/tasks/graph/foundation-a-route-symmetry.md)。

Windows MSVC 19.50 / Release，shipping Graph（HYLE_TESTING=OFF），复用原 CPU+CUDA source
树；source-parent 只增加本轮 fixture target。可用内存 42,210,340,864 bytes，库 j4 / test1。
共同 Graph 产品未增加数学/Queue/runtime，不构建全库算子目录。

- 首次 build PASS 209.187 秒；37 tests 中 33 PASS、4 FAIL，原日志完整保留。
- 原因是新测试夹具使用无显式输出 spec 的 symbolic Unary：原 GraphBuilder 正确拒绝。
  仅给测试补原合同要求的输出 spec，未放宽产品校验。
- 同树增量 build PASS 75.359 秒；修正后 37/37 PASS，含 6 个 exact/table tests 与
  31 个原 GraphBuilder/ViewSpec 语义/故障 tests。
- 累计 build 284.546 秒、test 0.109 秒；失败与活动预算不随新日志目录重置。

覆盖原身份/输出推导、有限表外点/外来 ID/缺失 binding、重复表/整表预检在 native factory
之前、derived quotient guard/绑定 ViewSpec、control-only 原身份，以及第二 native factory
失败后前缀 artifact/allocator 回收。尚未覆盖的 OOM、四路线实际运行和 package 风险不能
由测试数量推定 PASS。job receipts 保存实际输入 SHA256、命令、并行、PID、终局与耗时；
这是一轮 AUDITED 实施快照，尚无不可变 candidate。

追加的适用 OOM/control 补证：同树增量 build 11.203 秒、test 0.047 秒，39/39 PASS。
新增严格 metadata quota 不接纳 factory、失败回收/原 logical Graph 可重试；控制链只保留
数据祖先及 control 前驱，不复活无关 dead 节点，analysis OOM 归还全部 quota。
当前累计 build 295.749 秒、test 0.156 秒；原四失败继续保留。
