# 性能计时前完成边界与执行拓扑预审

## Task card

- trigger：Sched G02 的首次完整采样暴露两项在计时前即可由源码/短诊断识别的问题：peer 在 owner 返回后
  才清理 driver/stack；`structured-paired-process-v1` 产品对比受第一个/第二个进程热状态影响，而公共
  A/A 位于单个 control process，执行拓扑不同；
- scope：把完成边界、静态成本账单和同拓扑短 A/A 写成 P1/P2 preflight；解除
  `n-worker-physical-cores` 对 structured adapter 的错误绑定，使真实的同进程多 worker consumer 能使用
  既有 paired-row adapter；添加工具回归；
- non-goals：修改产品实现、统计估计器、5%/10% control、material margin、样本数、历史 manifest 或
  structured adapter 的 qualified 能力；
- stop：若需要新统计模型、改证据 schema、弱化 affinity/telemetry 或改历史合同，则停止并另立工具任务。

## 结果

sampling topology 只决定父进程 mask、worker singleton CPU 和验证 telemetry；measurement adapter 只决定
candidate/peer 如何成对。runner 的环境策略、Windows n-worker 选择、产品 telemetry 期望与 raw replay
本来已按此分层，唯一阻碍是 validator 中一条无实现依据的 adapter 等值检查。移除该检查后，同进程
paired-row benchmark 仍必须由进程退出状态和 telemetry 证明 exact worker placement 与工作量守恒。

方法文档现在要求先完成源码时序/静态成本账单，再做同执行拓扑短 A/A；任何固定位置效应或预计 MDE
不优于 material margin 都会在完整 n=15 前停止。公共 control process 的 A/A 不再被误写成独立 product
process 的拓扑证据。

G02 的同进程改造进一步证明“同拓扑”还不够：通用算术 control 没有 Scheduler、stack pool、allocator
和四 worker 热状态，不能校准这个 workload。paired executable 新增只用于 preflight 的 same-route 模式，
把两个标记行都执行 candidate 或都执行 peer。八个单次换序/placement 诊断中，完全相同路线的
first/second ratio 仍覆盖约 `0.917..1.052`，跨过 5% 门；因此产品 `A/B` 即使从 primary 的 `1.065/1.127`
翻到 placement 的 `0.970/0.934`，也没有资格解释为产品输赢。正式 n=15 没有启动。后续仪器必须让
workload-matched A/A 在同样的 runtime/allocator/线程拓扑下先通过，而不是增加产品样本。

同一 preflight 还发现既有 `/FUNCTIONPADMIN:64` replica 把 G02 benchmark `.text` 从 `0x32c00` 扩大到
`0x3f600`（约 +24.9%）。Microsoft 的合同是给每个函数入口增加 hot-patch padding；它同时改变代码
working set，并非单变量 placement control。G02 改用一个 `.text$AAA` 小型前置 contribution，只在 setup
前执行一次以防链接器丢弃；计时路线、产品库选项与函数体不变。其他历史合同不在本任务内批量改写；
以后被合法重开时先过新的 artifact-size preflight。

验证范围固定为 runner tooling tests、实际 G02 合同 validate、两个 opt-in benchmark target 编译、短
双顺序 probe、文档/closure checker 与 `git diff --check`。
