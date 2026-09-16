# R4 CUDA：native helper 采集接线与恢复条件

## AUDITED：仅 benchmark / 资格接线

- baseline：`2c20308c`；六个产品 owner 暂时保持 BLOCKED，产品/默认算法未授权变化。
- trigger：[独立 CUDA 的停顿与干预证据](cuda-r4-context-causality.md)；
  [有限工具修正](../project/performance-native-runtime-topology-2026-09.md) 已通过 11 个 selected gates。
- 固定条件：W1-WIN、RTX 4070/SM89、driver 610.62、CUDA 13.3、原 priority 与电源策略；改为
  `native-runtime-pair`，caller 仍是登记 P 核、process 只开放实时选择的两颗不同物理 P 核。
  native helpers 自管，不降低正式计时 caller priority，不接管 SDK 的线程创建/销毁。
- 原两次失败资格日志继续排除；语义、完整 FFN、shape、精度、placement、执行/分配顺序、
  5% margin、n=15→31 及一次算法纠正预算不变。不以本卡扩大样本、改变产品 API 或掩盖原限制。

| ID | 有限 FIX | 验收 |
|---|---|---|
| Q1 | 模块 benchmark 增加实际 process mask/priority 核对与 telemetry；共同 helper 继续固定 caller | 错误/缺失 instrument 条件退出；准备/测量后条件一致，无 hot topology 查询 |
| Q2 | 两个模块合同与 A/A validator 同用 native-runtime-pair；从统一机器 probe 派生，而非自造 CPU 表 | DLL 邻置、GPU/工具链身份保留；同一 caller/process 预算，旧 receipt 不签新条件 |
| Q3 | 模块自身的冷 guard 阻止 A/A 失败后共享 runner 继续计时产品 | validator 先置 FAIL，完整 PASS 才放行；A/A 自身与非 instrumented 使用明确区分，不改公共 runner |

## 有限验证与停止

新 benchmark 接线冻结不可变 commit 后，只构建对应 Direct/Graph primary/placement/native-cost targets，
使用既有相同产品库与原 oracle 来源；没有算法或输入改变，不重跑完整产品 correctness/三交付。
普通 selector 与 docs/closure 仍运行并保留跳过原因。

先运行一次固定进度回归：32 个独立 CUDA prefix，caller/parent 条件与 instrument 一致，单进程
最多 10 秒；再运行 Direct 与 Graph 各 16 个原完整 A/A，单进程最多 20 秒。任何同类停顿/错误
立即停止，留现场而非继续换条件；进度回归与 A/A 不作排名，也不从有限次数声称所有驱动无故障。
这一阶段总预算最多 20 分钟，不自动追加启动；两个模块 A/A 的有效 receipt 可由随后同身份
正式 bundle 复用，无需重复 AA。

以上全部通过后，依已满足的恢复条件将所需 CUDA owner 按 BLOCKED → SCOPED → AUDITED 恢复，
再对冻结候选启动原预算的 P1。未通过不解除 BLOCKED；P1、默认规则终审与原生成本仍须实际完成，
本卡不提前授予路线 CLOSED。
