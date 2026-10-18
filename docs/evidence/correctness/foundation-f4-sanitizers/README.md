# F4 candidate 三sanitizer

固定29d3c7a3e，GCC14.2/Ubuntu24.04/shipping，维护factor仅compute_state及CPU copy backend。
ASan/UBSan/TSan各实际24项PASS，XML计数/失败数与每stage原始日志hash核对；
相同compiler/config树增量，库按实际内存j4/j2/j1，test1，不跑Graph/算子全集。
TSan首进程exit66 unexpected memory mapping在测试启动前失败，原日志/时间保留；
同树同flags仅setarch x86_64 -R进程级地址布局恢复，无suppression/全局sysctl/代码改动，
恢复后实际24项通过，未重置预算。原命令/输入hash/配置/累积时间见三个job与raw archives。
本证据不代签native GPU故障、普通SDK/loader/迁址、P0或最终F4 CLOSED。
