# CPU ds4 状态量化接入批次

产品候选 `dd3009f0ff2014de048e6a765d07746d5eba1bd4`；补验时 HEAD `223589e4` 只有
设备可用性文档变化。产品、原函数和测试语义保持不变。

## 实际完成的验证

| 入口 / 交付 | 环境与范围 | 结果 |
|---|---|---|
| 普通 Direct / Graph | Windows，oneDNN 启用的默认 CPU 组合 | 6 项 PASS |
| 源码消费者 | Windows MSVC 19.50，复用已核对源文件和消费者的静态构建 | 6 项 PASS |
| 搬迁静态 SDK | 独立消费者；命令检查没有 producer 源码/构建引用；MIT 许可随包 | 6 项 PASS |
| 搬迁动态 SDK | 独立消费者；DLL 邻置且逐一匹配 SDK SHA；运行 PATH 仅系统目录 | 6 项 PASS |
| 普通 Direct / Graph | WSL GNU 14.2，oneDNN OFF，原 Windows 产品源码的静态验证副本 | 6 项 PASS |

每个入口运行同一份三个测试，覆盖 FP8 group64、FP4 group32、全部 65,280 个有限 BF16、
5,760 个中点/低尺度值、变化输入的两次重放、非有限 payload 的 Completion 失败和 invocation
退役/重建、metadata OOM 后恢复及 group128 准备拒绝。Direct 还观察别名拒绝；两入口检查
metadata allocator 收支平衡。数值 oracle 使用既有独立格式表和 nearest-even 定义。

这证明两个成熟状态操作的普通 CPU 接线，不能代签完整官方 MoE、所有路线、所有 CPU 配置
或新增性能资格。新增完整画像格为零；原 154/20/2 账本保持。

## 修正和未运行项

- 最初的测试错误地要求复用已终态失败的 invocation；已有 Compute/Graph 合同要求重建。
  旧日志保留在 `prior-lifecycle-test.log.gz`，数值部分通过。修正测试后两个入口均通过。
- 源码交付原先已完成编译，但包装脚本把 Windows 环境字典当作不区分大小写，访问
  `SystemRoot` 失败。补验改用 `os.environ` 读取，并核对 968 个源文件和消费者输入后复用构建。
- WSL 旧副本同步后触发无关 oneDNN 重编译，该过程明确中止并保留日志，没有记 PASS。
  本批改用 oneDNN OFF 的普通 CPU 路线；Windows 的默认组合及三交付仍为 oneDNN ON。
- 原完整 CPU 测试目标包含与这两操作无关的 oneDNN 测试，在 OFF 配置下不能编译；
  后改为只编译同一公共测试文件的两个独立消费者，链接实际 CPU/Graph 产品目标。
- GNU 对旧 oracle 聚合初始化与 GoogleTest 单行 if 产生告警。验证 overlay 仅在两个
  消费者上关闭这两类告警；所有产品仍为 `HYLE_WERROR=ON`，没有修改产品或数学。
  该 overlay、实际 argv、SHA 和原失败日志均保存。不能把这里的通过写成完整 Linux 测试套件通过。
- 本批无 P1；完整官方状态消费者与剩余官方 MoE 的成本资格仍属 D5。没有扩大公共验证工具。

## 静态成本与复核

冷期沿已有自定义 Operation 身份/描述/资源资格绑定，运行时只有已选函数，不搜索或计时。
每组最多 256 字节栈转换，调用六个原版 ds4 函数；没有新增热分配、线程池、共享可变 scratch
或 NativePlan 字段。复杂度随元素数线性增长；每个 FP8 code 的查表成本和 FP4 八项匹配
均由原成熟函数承担。需要的 BF16/F32 转换、映射与关闭明确计入调用边界。

2026-10-05，维护者在实现后单独复核：原函数正文/许可/lock 一致；group128 继续拒绝；
metadata OOM、别名、accepted failure 和退役规则没有被省略；SDK 没有公开 ds4 私有头；
没有恢复 D4 清退的 HYLE 数学或改变精度门。并发状态沿已有每 invocation 所有权，转换暂存仅在栈上；
本轮不新增并发调度或取消语义。完整模型/性能资格保持未完成。

完整命令和结果见 `receipt.json`、各子目录 receipt、压缩原始日志及输入 SHA。
`loc.json` 由统一脚本统计：Compute+Graph **60,481 实现行 / 242 文件**，不含测试和第三方源码。
