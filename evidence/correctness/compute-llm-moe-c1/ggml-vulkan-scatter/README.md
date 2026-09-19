# C1-Y29：有序 Vulkan Scatter 局部验收

产品候选 `07e30f13c3677d6edf56f3f9d1c9c7157504a911`：普通 Direct/Graph 接通受检 assign/add，
复用锁定 ggml 的 stable Argsort 和原版 GetRows，适配任意轴、storage、布局、索引检查与完成状态。
Add 保持 base 先参与、重复索引按原顺序以及 F16/BF16 每次更新后舍入；不改 Provider pin/patch。

| 构建 | Direct | Graph | 结果 |
|---|---:|---:|---|
| W1 static | 83 | 24 | PASS |
| W1 shared | 83 | 23 | PASS |
| Air static | 83 | 24 | PASS |
| Mini static | 83 | 24 | PASS，correctness/portability |

共 **427 项**，先行 9 项不重复累加。W1 每种交付另通过 2 项 source/MASM 检查和 1 项公开头检查。
W1 DLL 邻置、PATH 清空并抑制系统错误弹窗；W1/Air validation layer 没有错误。Mini 不代签 Formal 性能。

W1 在产品候选运行；Air 首次编译发现 Direct 测试遗漏 `<algorithm>`。
`4c7d51c81e61060fffd4495af811f00f76cd4b2d` 只补该 include，产品文件完全相同；Air/Mini 在该
测试修正候选通过。原始失败在 `w1-raw.zip/initial-air-compile-failure/`；产品纠正次数为 0，
测试可移植性修正为 1。候选、文件/工具/可执行文件摘要与原始日志见 [结果](result.json) 及原始归档。

已证明六种 storage 的逐位 assign、三种浮点的顺序 add、I32/I64 索引、重复与乱序、空/尾块/
4097 长索引、标量/多维索引、rank8/负轴/奇数与零 stride/字节起点。非法索引不覆盖 output，
失败槽退役后新槽可用；双槽五次变化、旧结果、quota 失败回滚、关闭后保活均通过。
普通 Graph 执行 Scan→assign Scatter→add Scatter→Gather→norm，并保留独立结果。

[统一 LOC](implementation-loc.json)：Compute+Graph **48,255 行 / 195 实现文件**，自检 PASS，
不含测试、bench 或第三方。此次增加 300 行实现，包含私有 GLSL specialization。

本记录只签上述局部正确性；Scan/Scatter 的真实三交付、完整 C1 片段、固定优选/淘汰、global
package 门与 P1 尚未完成。CPU oneDNN 资格 BLOCKED 不变，64 行/256 格矩阵未据此签收。
Y29 对排序输入作整数 padding；普通 TopK 的上游浮点 padding 问题另作有限定向修复。
