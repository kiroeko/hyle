# CUDA 原生包装成本夹具：数值资格

这两个独立 benchmark 只服务 R4 已定界的 `NATURAL_ROUTE` 成本说明，不能当作同合同 NI peer。
两侧使用相同 Marlin、HYLE precise SwiGLU 和 cuBLASLt FP16/FP32-accumulation 设置，实际运行
M1/M16 完整 FFN、各两个不同输入。原生侧分别为三段 native enqueue 或实际 CUDA Graph replay，
最后记录一个事件并等待；caller 承担 artifact/slot/错误及资源寿命，不具备完整 HYLE API 服务。

W1 MSVC/SM89 下两个 exe 编译和独立 double/FP16-stage oracle 全部通过。检查了 HYLE 与原生侧
的全部最终输出和两组输入变化；普通 Graph 与原生 Graph 均实际 capture。产品沿 a8c98249；
bench sources、exe hash 和逐程序结果在 [summary](summary.json)，[原日志](raw.zip)及
[哈希](sha256.json)保留。没有计时、A/A 或非劣结论。

冷准备复用同一具名 CUDA native kernel seam；cuBLAS 描述符/heuristic、32MiB workspace 上限、
16-byte alignment 和 FP32 reduction mask 与产品源逐项一致。原生 slot 的锁缓冲仅在冷阶段清零，
成功重放由实际 Marlin 程序归还锁；没有热 memset 或不同数学合同。
