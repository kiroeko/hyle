# R4 CUDA primary-context 停顿：定向因果核对

## 独立诊断审计

- 当前阶段：诊断 `CLOSED`，E1–E4 已取得有限机制结论；六个产品 owner 仍由
  [R4 主卡](cuda-v1-four-route-foundation.md)保持 BLOCKED，修正采集条件的进度回归尚未完成。
  本卡不授权产品修改或 P1。
- baseline：`79ca39a3`；trigger：用户在上一轮已停后明确要求 CUDA 必须探查清楚，以及
  [实捕现场](../../../../evidence/correctness/cuda-r4-context-progress-diagnosis/README.md) 已把问题收窄到
  `cuDevicePrimaryCtxRetain`。不把旧单核采样自动恢复。
- 新的具体反例：原独立 CUDA probe 首次创建 context 用 `cudaMalloc`，没有按完整程序的调用顺序
  直接进入 `cuDevicePrimaryCtxRetain`。它的成功不能排除已捕获的 driver API 初始化路径。
- 范围：复用完整诊断 executable；另编译一个无 HYLE 依赖的 CUDA prefix executable，严格包含
  inventory、device-count、`cuInit`、`cuDeviceGet`、`cuDevicePrimaryCtxRetain`，保留每步标记。
  原产品、算法、GPU 配置、驱动、machine profile 和公共仪器不变。
- 从冻结起总预算 40 分钟；一次固定观察批最多 128 次串行启动或 8 分钟，完整程序与独立 prefix
  交替。捕获最多两个合格现场；单进程最多 20 秒，所有 DLL 邻置。此批不是性能采样。

## 有限动作

| ID | 动作 | 判定边界 |
|---|---|---|
| E1 | 相同 API 顺序的独立 CUDA prefix，原完整程序作直接消费者 | 独立失败才能排除 HYLE 为必需条件；单纯成功不排除间歇问题 |
| E2 | 首次 Context 阶段停留超过 5 秒时保留 dump、真实线程角色/动态与基础 priority/affinity、等待对象信息 | 栈、phase、调用点相互核对；不把 CPU 时间当作死锁证明 |
| E3 | 对实际停住的 owned child 做一次 CPU 可调度集合干预 | 第一个现场只扩 helper 到第二颗 P 核、caller 保持 CPU2；若有第二个现场，则只降 caller priority、仍保持整个进程 CPU2；每次最多观察 5 秒 |
| E4 | 沿实捕等待链核对 Windows 语义和 NVIDIA API/已知限制 | 归因需实际证据；没有公开 driver 内部源码时明确局限，不凭经验改 HYLE 锁 |

任一干预后完整程序恢复需保留完整数值输出；先记录恢复时刻/phase，再看退出，不能把停止观察当成功。
进程未达到目标阶段不干预；新失败位置单独记录。捕获前不更改目标线程优先级或 affinity。

停止时给出 API 调用方式、HYLE 必要性、CPU 采集条件和 SDK 内部未知项的分项结论；有证据的修正再
固定 owner/FIX/回归，按协议恢复。预算内未解则保留 blocker 并给出最小外部复现包，不增加 P1/n、
不自动更新驱动、不给隐式验收例外，也不转去填算子目录。

## E1–E4 结果

[完整原始证据](../../../../evidence/correctness/cuda-r4-context-causality/README.md)：实际执行 89 次
启动（45 native、44 完整 FFN），获得预声明两个现场后停止。两个停顿都来自**完全不链接 HYLE**
的独立程序，堆栈进入与完整夹具相同的 driver 零超时等待循环；直接 retain 是合法互操作入口。

两个现场的 caller 当前 priority 都为 24，SDK helpers 为 10–11，全部 affinity 都是 CPU2，
caller 已消耗约 4.45 / 5.09 秒 CPU，而 helpers 为 0。第一次只扩 helper 的可用核、caller 仍固定
CPU2，retain 恢复后正常释放/退出；第二次保持整个进程单核，仅降低 caller priority，约 62 毫秒内
完成。这支持单核限制与实际 priority 差造成 SDK 辅助线程调度饥饿；不需要修改 HYLE/Sched 才能复现。

尚未知是谁把 caller priority 提到 24，也不能把前两次没有现场的 120 秒停顿强行归为同因。
原始观测有明确注记：native basic-info 的 base 字段在本机是相对级别；第一次超时指针在恢复执行后
读取，值已陈旧，不参与结论。字段布局和 priority 查询以独立 Win32 对照自检通过。

本诊断停止，不再追加启动。后续固定可真实表达 native helper 的采集合同与有限进度回归；不把
两次干预直接当成产品修复、完整性能通过或所有驱动的结论。
