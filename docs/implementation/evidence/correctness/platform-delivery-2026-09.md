# 平台、Sanitizer 与交付正确性资格（2026-09）

本页是
[`CLOSED 组件全库正确性资格审计`](../../tasks/project/closed-components-correctness-qualification-2026-09.md)
Q09 的事实证据。它刷新平台、sanitizer 和交付风险，不创建第二套组件状态，也不把“一次全库通过”
外推成未执行路线的 `PROVEN`。

## 输入身份与产品投影

- 最终测试/工具候选：`aef74c1e976ff14c535851e2f5ad95358e95b798`；tree
  `524572feda74186770e2611ac237aa7f3a358f1d`；
- 完整 Windows 产品/二进制交付候选：`1a51d8324b3a7cc2daa43a65f58a508142beb835`；
  后续候选只改变三处测试、sanitizer-only GCC 诊断策略和用户说明，普通 source/static/shared
  产品字节及依赖图不变，因此这些交付证据仍可复用；
- `correctness-inventory.py --print-closed-products` 从中央 product/factor 图和
  `module-status.json` 投影 13 个公开根：`core,error,time,platform,mem,obs,simd,math,container,util,sched,io,process`；
  它们的 union 恰好是 29 个 `CLOSED` factor，`compute/graph` 没有混入；
- 测试均为 shipping layout（`HYLE_TESTING=OFF`）、WERROR、TLS/HTTP2 feature-off；feature-on 的
  public journey 已由 Q08 单独证明，不与本矩阵做笛卡尔积。

## 风险最小充分矩阵

| 风险路线 | 实际执行 | 结果 | 裁决 |
|---|---|---|---|
| Windows native | W1-WIN，MSVC 19.50，Release/shared；完整产品测试先得 102/105，纠正三条陈旧 OS 合同断言后定向 3/3；后续候选涉及的 Container/Sched/model-file 三项再验 3/3 | 全部适用测试通过；IOCP、Windows file/watch/process 路线被执行 | `PROVEN` |
| macOS native | MacBook Air ARM64，Darwin 25.6.0，AppleClang 21；完整非维护测试 57/58，纠正 model-file journey 的 capability 选择后定向 1/1 | 58 项语义全部通过；kqueue/watch/process/VM/NEON/fcontext 路线被执行 | `PROVEN` |
| Linux ASan | WSL2 原生 ext4，GCC 14.2，Debug/shared；全部 29 factor 与五条 journey 的非维护测试 | **58/58 PASS**，leak detection 与 fail-fast 开启 | `PROVEN` |
| Linux UBSan | 同一源码身份的独立 Debug/shared build | **58/58 PASS**，`halt_on_error` / `-fno-sanitize-recover=all` | `PROVEN` |
| Linux TSan | 独立 Debug/shared build，`setarch -R`；只选择 Mem-reclaim、Obs、Util-core、Sched、并发容器、IO/Process 和四条异步 journey | **30/30 PASS**；Sched 85.64 秒，全部单项低于 120 秒硬 timeout | `PROVEN`（standalone-fence 算法仍不由 TSan 证明） |
| complete source | 从干净最终候选生成 zip、校验 manifest/license/digest、在 checkout 外解压并以 13 个根重新配置/编译 | 158 个 build action 通过；SHA-256 `f948bd2415b71dce99793c13c787561f8a841196ba37a9641ac5bb46e21cdffe` | `PROVEN` |
| relocated shared/static SDK | W1-WIN 分别安装、移动 prefix，再运行通用与 15 个 module-owned focused C++ consumer | 两种 linkage 的通用 14 项检查及全部 focused consumer 通过 | `PROVEN` |
| hosted GitHub Actions | run `34590263514`，candidate `1a51d832` | 首个 job 在零 step 时被 GitHub billing/spending-limit 拒绝，其他 job 全跳过；本地 selector/YAML/PowerShell 合同通过 | hosted execution `UNPROVEN`，不是产品/工作流测试失败 |

每个 CTest 路线都有 120 秒单项 timeout，外层命令另有 20–45 分钟硬界；没有依赖无限等待或“多跑几次
直到绿”。ASan/UBSan 跑全部非维护正确性测试，TSan 只覆盖共享状态和异步生命周期，避免用纯值类型
数量稀释并发门。

## 本轮发现与归类

1. Windows 三条 IO 测试沿用了错误或 POSIX-only 假设：已 admission 的同步 IOCP failure 仍通过
   completion 交付；socket 与 IOCP 的关联持续到 native handle close；`UdpSocket::open` 不等于 bind。
   产品实现符合平台合同，测试及用户说明已纠正。
2. AppleClang 会在 WERROR 下拒绝两条字面量 overloaded self-assignment。测试改用运行时别名，仍直接
   验证 self-assignment，未回避所有权语义。
3. model-file journey 把 deadline 无条件强加给 macOS blocking-file fallback，而且吞掉 read error。
   现在经公开 `IoCapability::kAsyncFileData` 选择预算，并传播 error/DataLoss。用户指南同步说明：没有
   native async-file capability 时逐操作 deadline/cancel 会诚实返回 `Unimplemented`。
4. GCC PR 109224 会把 allocator-first coroutine placement-new 误判为 mismatched-new/delete。仅在 GNU
   sanitizer build 中把该诊断从 error 降为可见 warning；普通 shipping WERROR 不变。三类 sanitizer
   的实际执行没有报告对应 allocation/deallocation 缺陷。

这些均是测试、工具或文档缺陷，没有复现 correctness/security/UB/lifetime/concurrency 产品缺陷，
因此没有触发任何 `CLOSED` factor 重开。

## 明确保留的边界

- GitHub-hosted CI 与其中的 Linux AArch64 cross job 因账户额度不可用，保持 `UNPROVEN`。不得把本地
  workflow parser 通过写成 hosted execution `PROVEN`；额度恢复后可直接执行一次
  `workflow_dispatch`，无需重开产品模块。
- 第二台 Mac 没有运行：第一台 Apple Silicon 已完成同一 backend 风险，当前没有需要第二硬件反证的
  失败。TSan 不证明依赖 standalone fence 的算法，该风险继续由来源审计、stress 与真实 ARM 路线承担。
