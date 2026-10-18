# F45：CPU / Apple 原生低精度边界

baseline `55211caf1954f34a9322c08872ead3e0f3efd742`。只验证当前成熟原语，产品 FIX 尚未执行。

| 原生调用 | 已执行结果 | 可得结论 |
|---|---|---|
| MLX E4M3 decode | 254个有限code逐位一致 | 当前有限编码可用；没有验证NaN payload |
| MLX E4M3 encode | 1016个中点、相邻值、正负值及饱和值全部一致 | 原版直接RNE可用 |
| MLX FP8 group64/128 | 两种scale策略×F32/BF16，共24576元素；codes/output/scale均无差异 | 原版序列能由成熟MLX组合；未签完整GEMM |
| MLX MXFP4往返 | 普通256元素通过；极小值256元素中194个不符 | 不接纳为通用官方FP4 roundtrip；保留失败，不放宽门 |
| oneDNN `simple:any` E4M3 | 1008元素中253个encode、3个decode位模式不符 | 当前reference含F16中间舍入及零符号差异，不能代签直接RNE |
| oneDNN `simple:any` E2M1 | 56元素encode/decode全部一致 | 有成熟FP4转换；尚无完整CPU官方路径资格 |

原始Apple批次退出1，表示FP4极小值失败；不把整个批次写PASS。第一次编译误用了MLX
shift函数名，纠正公开API名称后运行；原日志保留。CPU程序退出0只表示执行结束，实际逐项
差异以上表和日志为准。测试oracle独立有限编码查表；没有新增产品数学。

原代码、日志、调用与版本见同目录文件。后续只推进已经验证的Apple FP8接线；CPU
F32量化和Apple FP4仍需沿成熟实现处理，不能用本原生结果增加176格ledger。
