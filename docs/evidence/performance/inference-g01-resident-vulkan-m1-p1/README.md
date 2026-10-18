# G01 resident Vulkan：合格P1未过性能门

不可变产品 `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`，性能source
`9087db46c5ce13038d53680c1d1ce23cb2453be5`。[receipt](receipt.json)、
[summary](summary.json)、[manifest](manifest.json)、[raw.zip](raw.zip)与
[回放收据](verify-receipt.json)保存原始输入/命令/产物/日志和真实裁决。

实际Air Mac16,12：M4、macOS27.0.1、AppleClang21.0.0，明确unbound engineering；
不是Formal或所有设备/模型证书。原v1仪器、严格F32 oracle、同完整提交/完成/输出归还，
原生split-K/mask-opt，16组A/A、八个strata×15进程和30个公共controls全部完成。
资格PASS、原verify退出0，**performance FAIL**，不得签G01/runtime CLOSED。

| 消费点 | candidate/peer ops/s 中位比范围 | 原5%门 | 方向 |
|---|---|---|---|
| causal Q1/KV1024/D256 | 0.687–0.710 | LOSE / FAIL | 全strata CI上界<1，稳定LOSS；吞吐低约29–31% |
| causal Q128/KV1024/D256 | 0.962–0.969 | PARITY / PASS | 全strata CI上界<1，仍是小幅方向LOSS，不改名为方向平手 |
| noncausal Q17/KV257/D64 | 0.921–0.977 | INCONCLUSIVE / FAIL | 全strata CI上界<1，方向LOSS；最后placement/order格落出5%带 |

活动卡10%非劣预算保留，现有仪器只准5%而采用更严格门；任何结果不回写改阈值。
decode也超出原10%预算，当前需要结构/热费用定位。其余差距不能仅由排名触发公共
API/layout重写。完整token/s仍须引擎整链；本记录是三个有限片段，不外推整体推理。

首个fresh构建的容量上界比较WERROR FAIL和100.28s保留；第二个profile只让指定Clang
tautological capacity警告不升级error，其余WERROR仍开。原失败与全部原日志均保留，
没有换目录抹掉预算。第二轮实际1221.57s。远程build/toolchain及二进制继续保留，以供
原路径回放；raw本地归档不是移动路径后伪造可移植性能证书。

初步源码费用定位：原NativeWork每次调用原ggml graph_compute，沿原native节点编码和
retire cleanup；独立original shader recipe可冷record后复用。两者都有相同公开输出
寿命，数学相同，实际managed replay费用不同。需用有限结构反例核对原native承载是否
有可合法复用记录，不修改公共仪器或用裸kernel计时替换完整费用；当前只是定位，非修复。
