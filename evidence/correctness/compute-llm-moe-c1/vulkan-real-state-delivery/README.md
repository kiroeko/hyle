# C1-Y31：Vulkan 真实张量状态片段与三交付

候选 `a03f6786bfd9ecdcf0e158d5ea90f1a627f31236`，产品源码与已验收局部回归的
`55fe0b8a` 相同。范围与首次运行前固定的标准见
[C1-Y31](../../../../docs/implementation/tasks/compute/llm-moe-c1-tensor-state.md#c1-y31真实张量状态片段与三交付有限验证)。
本组局部裁决 **PASS**；没有产品或 consumer 纠正，没有 P1。C1 整包、全局 selector 余门和
64 行/256 格共同矩阵没有由此签收，CPU oneDNN 性能资格 BLOCKED 不变。

## 实际消费与结果

重新逐文件核验本地 Qwen3.5-2B：13 个文件、4,571,274,023 字节，source ID
`4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155`。仅提取实际 BF16
embedding 的 34×2048 子表及第 0 层 Gemma RMS 权重。consumer 只使用公开 C++ 头和
`hyle::graph`；文件读取、独立 FP64 参考与测试状态留在外部 consumer。

主链为 I32 索引差分→Scan→Gather→F32 Cast→Gemma RMS→SiLU gate/value→residual→BF16
Cast→Scatter assign→Scatter add→Gather。另验证 I32 exclusive Scan 和实际 scale 派生输入的
F32 inclusive/exclusive Scan。这是基础库的真实张量组合片段，不声称整条状态链就是 Qwen forward。

| 平台 | 交付 | 每条交付的实际执行 |
|---|---|---|
| Windows | source / static / shared | 14 组 Direct/Graph，140 次状态更新，14 个受检失败及 14 个新槽恢复 |
| Air `Mac16,12` | source / static / shared | 同上，启用 Vulkan validation layer |

合计 **6 条交付、84 组、840 次状态更新**；无 VUID、validation error 或 sync hazard。形状为
`[1,1]/[1,17]/[1,128]/[1,23]/[1,129]/[2,23]/[2,129]`，后四为 holdout；batch2保留三维payload。
每组两个独立状态流各五步，上一轮输出直接作为下一轮状态；共享不可变表、一个快照槽和两个
交替工作槽。两工作槽接收实际新输入以释放旧引用，不以额外payload复制规避寿命合同。

独立 FP64 norm/gate/residual 参考与预声明 BF16 相邻编码、数值误差及状态区间传播均通过；
以独立合格的实际 update 另行逐位核对 assign 最后写入与 add 逐项 BF16 舍入。I32 前缀精确，
F32 前缀遵守预声明累计绝对值误差界。旧输出逐位保持至 Context 关闭后，原表不变，失败槽
退役，新槽恢复；预热后 quota 用量不增长，所有 owner 析构后 quota 为零。

## 交付与证据身份

每个平台只做两次 producer 构建：source consumer 构建兼作 static producer；另建 shared。
安装后移动 SDK，并隐藏 producer、完整源码，再由独立 consumer 通过 `find_package` 构建。
Windows DLL 与 exe 相邻，测试时 PATH 隔离且关闭系统错误弹窗；两平台 WERROR、无插桩、
native 重型步骤串行且并行度 2。没有拿旧 Gather/norm 的 SDK 代签新增 Scan/Scatter。

- [汇总](summary.json)、[Windows 三交付](windows-delivery.json)、[Air 三交付](air-delivery.json)：
  原始日志、consumer 源码、SDK 文件清单和 EXE hash 在各自同名 ZIP，receipt 锁定 ZIP SHA-256。
- [命令与输入身份](commands.zip)：固定输入清单、fixture 提取/独立参考、构建/重定位/归档脚本；
  实际模型 payload 不入库。[原文件核验](source-current.json)只证明输入身份，不单独证明执行正确。
- Air 首次 launcher 指向不存在的 Homebrew Python，构建尚未开始；
  [原始失败及纠正](air-initial-launch.json)保留。改用该机 `/usr/bin/python3` 后执行；产品、
  consumer、oracle 和 verifier 均未改动，不能把这次启动失败写成首次即通过。
- [统一 LOC](implementation-loc.json)：Compute + Graph 实现 **48,294 行/195 个文件**，
  不含测试、bench、第三方，自检 PASS；本组没有增加产品实现代码。

入库 JSON 展示稿统一使用 LF 换行；[原始报告](original-reports.zip)保留归档前的原字节，
summary 锁定其 hash。两个平台的原始测试日志 ZIP 均未改变。

证据 SHA/CRC 核验后清理本轮临时 producer 与源码：Windows 974,692,165 字节，
Air 879,293,358 字节，共 **1,853,985,523 逻辑字节**。保留最新重定位 SDK、独立 consumer、
输入与证据；清理记录为 [Windows](windows-producer-cleanup.json)、[Air](air-producer-cleanup.json)。
本组停止重复运行。固定优选/淘汰、其余 C1 验收与全局 selector 余门继续按当前卡处理；
本证据不触发用户要求的 C1 完成通知。
