# R3 修正产品与验收环境的定向复核

产品投影 `d9712e68`；新 benchmark `2b613413`。本轮复核与实现步骤分开，只检查既定 C41–C46、
C45a/C46a 与 E1–E3。未修改产品，也不以源码审查替代未完成的性能或交付。

- 两个 `cpu_context.cc` 工厂分别使用固定短 Provider 数组，Context 仍复制并拥有状态。
  `DefaultContextOptions` 只借用 allocator/executor，头文件明确派生物的借用期限。没有全局
  Provider registry、隐建线程、用户 timing/profile 或重复执行时的重新选择。
- oneDNN leaf 的 select_route 先核对 dense row-major rank-2 f32 与矩阵维度，零维回 Native，
  单 task 且 rows<16 也回 Native；真实 executor 的 N tile 仍在 prepare 绑定。Graph region
  保留完整四节点融合，不套用 leaf 的行数阈值。其余 19-family Native 算法没有改动。
- C46a 的 full-column SIMD batch 只在 first_column=0 且 column_count=leading_columns 使用。
  workspace 规划先以 try_multiply 检查 rows×hidden 及 bytes，完整区间乘积来自同一已验证维度；
  分列 task 保持逐行，不能越入其他 worker 区间。Direct fused/natural comparator 做了相同变更。
- C45a 仅在 static 且 oneDNN 缺席时裁去两个本地 route 安装接口的 OPTIONAL_ABSENT 边，
  source graph 保留原有身份。Air/Mini 的 Native static/source/shared 已实际通过，x64 ON 边保留。
- `git diff d9712e68 2b613413 -- include src samples cmake CMakeLists.txt` 无差异。E1–E3
  只改变 benchmark setup、validator 和新实验合同；热 admission/dispatch/math/wait 没有改写。
- 新三个物理核合同由统一 CPU Set selector 证明 caller 与 worker 物理核不同且排除 SMT；
  benchmark worker hook 核对实际当前 CPU。两侧使用同一 kLowLatency executor，并计入三个核，
  没有把 caller 当作第三个 worker。serial 阶段 worker 尚未创建；two-worker 阶段固定两 worker。
- 两个 placement、执行/分配正反、实际 selected/peer A/A、全部 12 workload、8ms epoch、5% /
  10% preflight 和 n=15→31 保持。A/A cold guard 仍禁止失败后开始产品计时。

没有发现新的产品代码缺陷。Sched notification_complete 忙等仍只是具体疑点；共核诊断不能证明
独立因果，也不是 Sched 性能 PASS。Windows 验证中 DLL 缺失发生在测试 main 之前：这是外部
recipe 未为 GoogleTest/HYLE DLL 做相邻部署；按用户要求复制到 exe 目录后，23/23 source 检查
通过，无产品修改。错误弹窗已由非交互启动配置禁止；初始失败保留。

Windows 余下 SDK/CPU OFF 验收随后已通过；当前证据缺口为两个工程 P1 的实际裁决，尚不能关闭 R3。
