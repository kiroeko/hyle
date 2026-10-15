# G01 原定驻留读取范围的整项裁决

状态：**G01 已完成，2026-10-15**。本记录聚合原 G01 的有限施工与既有完整路线，
不启动产品修改或重复采样。它是 goal 工作项裁决；当前模块状态仍由
[registry](../../module-status.json)唯一决定，G04/G05 的活动 CANDIDATE 不在此关闭。

原需求见[单机审计 G01](../project/inference-single-machine-foundation-audit-2026-10.md#驻留attention和参考档成本)：
直接消费驻留 cache/有效长度或精确页，消除按 Q/head 重复物化全 KV 的默认成本，
保自然 public 调用、权限、结果与原生退休，以及各路线适用的交付与成本资格。

## 逐项证据

| 原有限要求 | 已完成范围与裁决依据 |
|---|---|
| CPU 单请求与多请求 | [resident/ragged 终审](inference-g01-cpu-resident-ragged-terminal.md)：原 flash、默认公开工厂、Direct/Graph、源码/两 SDK/宿主风险与声明的 P1 已完成 |
| CPU 精确 page16 | [page16 终审](inference-g01-cpu-paged-terminal.md)：F16/BF16、独立有效区、乱序/共享/tail、64步、旧结果/错误/预算及原 P1 已完成 |
| Apple ragged | [原生 ragged 终审](inference-g01-apple-ragged-terminal.md)：借原 MLX 数学和原提交/完成对象，原声明交付、风险和 P1 已完成 |
| Vulkan 单请求与多请求 | [驻留端口终审](inference-g01-vulkan-resident-terminal.md)及[ragged/记录终审](inference-g01-vulkan-ragged-terminal.md)：原精度、真实驻留端口、固定记录/失效、交付/风险和最终 P1 已完成；早先失败保留 |
| Apple/Vulkan 精确 page16 | [原生分页终审](inference-g01-native-paged-terminal.md)：直接读原物理页与 owned 表，乱序/共享/tail/空域、两槽/旧结果/64步、四 SDK/宿主风险和两路线原 P1 已完成 |
| Linux CUDA 已有路线 | [原 F5 最终收据](../../../../evidence/correctness/foundation-f5-final/receipt.json)与[本次有限对账](../../../../evidence/correctness/inference-g01-aggregate-acceptance/cuda-baseline-reconciliation.json)：原 FI dense、ragged/page16 public source/static/shared、数学/权限/退休与 P0 已签；无新数学算法，不新增无变化 P1 |
| 真 CPU＋GPU 数学组合 | [G02 native 联合有限终审](../graph/inference-g02-native-paged-terminal.md)和[G03 七路线交付](inference-g03-final-closure.md)：原 native 驻留数学进入公开 CPU→GPU→CPU，exact frames 与实际退休有依据；G02 原 Windows 聚合门另保留 |

CPU/Apple/Vulkan 声明的八份最终 P1 收据本次逐份核对，qualification 为 PASS/QUALIFIED、
performance PASS、verify0，身份索引见[声明性能对账](../../../../evidence/correctness/inference-g01-aggregate-acceptance/declared-p1-reconciliation.json)。
各自负方向、INCONCLUSIVE、旧 FAIL/STOP/DIAGNOSTIC、原时间与累计均保留，不重新解释为 WIN。

Linux FI 保留原 F5 的 **P0**：原核直接读 K/V roots、indptr/page/tail，HYLE 不创建
按 Q/head 展开的全历史 KV。原 caller workspace、split/merge、metadata H2D、planner vectors、
occupancy、prepare/reuse 等待与页 guards 全部收费。原 dense native TU 与 F5 候选相同；
当前 plain coupled/causal/unbounded batch 仍走原分支。历史两 SDK 的 archive/log 身份已实核，
旧消费者 SHA 与当前新增 storage/LaneBinding 夹具的区别明确保留。
本记录不称 Linux FI 已测非劣、WIN、SOTA 或全域高性能通过。

## 聚合、停止与保留范围

[聚合收据](../../../../evidence/correctness/inference-g01-aggregate-acceptance/receipt.json)
把原要求映射到上述已关闭子批和既有 ROUTED 路线；没有找到仍欠兑现的原 G01 有限 FIX 或
其明确承诺的验收门。原子批曾写“整个 G01 未完成”是各自历史时点，不覆盖本次聚合裁决。
本次只更新 goal 跟踪，不修改 registry、公开 API、数学、编译系统或性能仪器。

停止原 plain G01 的日常施工与重复验证。只有已验有效区/地址/权限/结果退休/交付/已冻结
结构成本的真实缺陷或新的实际消费者机制，才按真实 owner 进入下一轮。

新编码 KV、window/非因果/不同 Dv 的新增域及其性能由 G04/G05 继续验收；active/delta、
topology/权限准入和 Q1 decode 成本归 G06。G02 原 Windows 中断及未跑门仍未证明；
G07–G10、capture、producer/growth 的性能、私有池总上界与整引擎 token/s 不由 G01 代签。
CUDA 仅原已准入 Linux/SM89，Vulkan BF16 等未准入域不因表格聚合而扩大。
