# C1-Y30：原版 Argsort 大宽度遗漏修复

固定候选 `55fe0b8a505c0f725473f53e1a3eb875006b85fb`。
Windows [host 复现](host-reproducer.json)证实原版浮点 padding 会把2097153项补到2097152项。
修复前 `1c5f1df6` 的普通 Direct 测试实际返回0，遗漏末尾唯一最大值4；[原始失败](before/result.json)
及日志/JSON/命令完整保留。该失败不是性能结果或测试公平性判断。

原版 Argsort 只将 padding 指数改为整数计算和无符号移位；排序 shader、比较、pass、scratch
所有权与 upstream pin 不变。普通 managed TopK 及轴适配入口新增冷设备范围检查，区分 small
的 shared memory 与 large 的 padded scratch。无新排序算法、公共 API 或执行状态。

| 构建 | Provider | Direct | Graph | 结果 |
|---|---:|---:|---:|---|
| W1 static | 19 | 85 | 25 | PASS |
| W1 shared | 19 | 85 | 24 | PASS |
| Air static | 19 | 85 | 25 | PASS |
| Mini static | 18 | 85 | 25 | PASS，correctness/portability |

共 **514 项**；先行3项不重复计入。W1每种构建另通过source/MASM两项及公开头两项；DLL邻置、
PATH清空、禁系统错误弹窗。W1/Air无validation错误。Mini没有validation layer，未运行具名
`ExplicitValidationUsesItsOwnInstance`，由W1/Air提供该项证据，不将其记作Mini PASS。

新增回归覆盖2^21−1/2^21/2^21+1、升降序、末尾唯一胜者与stable ties；Graph在同一managed
region执行Abs→TopK并发布两个输出。受控limit反例覆盖scratch、grid、lane和两种axis入口，
同时确认small路径不被large scratch规则误拒。原有NaN/位级、布局、Scan、Scatter及生命周期
在同候选回归通过，见[汇总](result.json)与对应原始归档。

首次修复候选 `37fc82dd` 配置时遗漏NOTICE的独立license摘要，随后只更新该摘要；该构建纠正
记为 **1/2**，算法纠正0。首个配置失败保留在`w1-raw.zip/first-configure-failure/`。
精确patch仍覆盖11个文件，树身份为`641c36f67482a39f4540361598722ce0c2a2dab1d841d75b9e858552c6ab46cc`。

[统一 LOC](implementation-loc.json)：Compute+Graph **48,294行 / 195实现文件**，自检PASS，
不含测试/bench/第三方。旧验证输入与不再被active build引用的源码副本已按归档身份清理，
Windows/Air/Mini共517,759,160逻辑字节；当前源码、构建、SDK、原始cache与证据保留。

本记录不签C1完成：Scan/Scatter的真实三交付、完整C1片段/性能优选/淘汰及global门仍待完成，
CPU oneDNN资格BLOCKED不变。依赖锁变更使selector选中112门、保留8个skipped理由；本记录只
覆盖具名局部门，其余门继续登记为未完成，没有用514项测试替代完整selector裁决或P1。
