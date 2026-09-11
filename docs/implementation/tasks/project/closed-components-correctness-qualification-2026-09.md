# CLOSED 组件全库正确性资格审计（2026-09）

本记录是一项横向 correctness qualification，不是模块 closure card，也不创建第二套模块状态。
组件当前状态始终只看 [`module-status.json`](../../module-status.json)。本页只回答：当前 29 个
`CLOSED` 组件的公开能力，有哪些语义已被直接证据证明，哪些仍是 `UNPROVEN`。

## 任务卡

- baseline：`53dc0bd89105334f78ece6452fc287487d34fcbc`；Windows checkout 是产品改动唯一来源；
- trigger：模块局部测试数量不少，但仓库没有全库 line/branch discovery，也没有统一的逐能力语义
  覆盖账本；历史上已经由真实 deadlock 暴露过“测试通过”不等于完整生命周期正确；
- objective：按 [`closure-correctness-v1`](../../../developer/review/correctness-profile-v1.md) 对全部
  `CLOSED` 组件建立可复现的 public-API 正确性证据，补齐真正缺失的原子、组合和平台风险测试；
- scope：`module-status.json` 中除 `compute`、`graph` 外的 29 个 `CLOSED` factor，测试、测试专用
  fault seam、coverage discovery、sanitizer/backend/delivery qualification、跨模块 public journey、
  CI 与相应 developer/implementation 文档；
- non-goals：重新审能力范围、增加产品功能、执行 P1/P2、凭覆盖率百分比签署正确性、无 trigger
  重开 `CLOSED`、修改 `compute/graph`，或把 qualification 结果写成第二份生命周期 registry；
- verdict vocabulary：每个语义格只写 `PROVEN`、`UNPROVEN`、`NOT_APPLICABLE`；测试失败另记
  `DEFECT`。只有可复现 correctness/security/UB/lifetime/concurrency 产品缺陷，才为 owner 建立新
  closure card 并依法执行 `CLOSED -> SCOPED`；
- stop：发现产品缺陷时暂停该 factor 的资格签署并走定向 reopen；工具不能可靠观察所需事实时登记
  独立 tool gap；平台不可用或预算耗尽时保留 `UNPROVEN`，不能猜成 `PROVEN`。

## 当前基线事实

- 29 个 `CLOSED` factor 都有独立测试目录；仓库中共有 173 个非 `compute/graph` C/C++ 测试翻译单元、
  61,483 行翻译单元源码和约 2,057 个 GoogleTest declaration；另有 5 个测试支撑头/片段。table/property case 会让实际输入多于
  declaration；这些数字都不代表覆盖充分。
- 当前没有统一的仓库级 line/branch coverage 入口。首次 instrumentation 只用于发现 zero-hit
  产品路径和未触达分支，不设“达到某百分比即正确”的门。
- push/PR CI 使用 change-impact selector；三平台全 CPU 测试、package 与 product-shape smoke 只由
  `workflow_dispatch` 运行。这符合日常避免费时全测的原则。
- `HyleCiProductShapeSmoke.cmake` 仍只选择 `core,util,obs,time`，是旧阶段遗留的 shipping-layout
  覆盖缺口。本轮应由中央产品图投影全部已实现且 owner 为 `CLOSED` 的非 Compute/Graph 产品，避免
  再手写一份会漂移的产品名单。

## 固定语义矩阵

每个 capability family / public journey 必须逐格绑定测试、原始 receipt 或明确理由：

| 维度 | 必须观察的事实 |
|---|---|
| normal / boundary | 正常结果、empty/zero/min/max/capacity、有效格式和稳定复杂度前置条件 |
| failure / OOM / overflow | 错误类别、旧值/资源状态、OOM/系统失败、checked arithmetic 与恢复 |
| ownership / move / teardown | owner/view、move 后状态、borrow invalidation、幂等 close、最终资源释放 |
| cancel / deadline / concurrency | 适用时的 race、late completion、progress、backpressure、cancel/timeout |
| main consumer | 只经 public C++ API 完成最相关的真实产品任务并观察最终状态 |
| orthogonal consumer | API 有领域特化风险时，用第二产品纵切反证；确实不适用时说明原因 |
| platform / backend / delivery | 会改变语义风险的 OS/backend，以及 source/static/shared 的适用旅程 |

line/branch coverage、测试数、sample 成功或“以前跑过全库”都不能替整行签字。旧证据只有在候选
产品字节、相关依赖/toolchain/build 语义和原始 identity 仍匹配时才能复用。

## 有界执行波次

1. Wave A：`core`、`error`、`math`、`container_intrusive`、`util_core`、`time`；
2. Wave B：`timer_queue`、`platform`、`platform_thread`、`platform_fs`、`mem_reclaim`、`mem`、
   `mem_vm`；
3. Wave C：`obs`、`simd`、`container`、`container_concurrent`、`container_queue`、
   `container_text`、`util_runtime`；
4. Wave D：`sched`；
5. Wave E：`io_core`、`io_tls`、`io_http`、`io_http2`、`io_transfer`、`io_watch`、
   `io_websocket`、`process`。

每波先做静态 public surface → capability → test/evidence 回映，再只补 `UNPROVEN` 高风险格。普通
提交只运行受影响测试；每波结束运行一次该波 qualification，终局才运行一次完整矩阵。所有 stress、
fuzz、网络和并发测试必须有硬 timeout；超时是确定失败，不允许无限等待。

## 跨模块 public journeys

| Journey | 公开组合 | 终局可观察结果 |
|---|---|---|
| 数据准备 | Container/Text → SIMD/Math → Mem | 输入解码/变换正确，scratch/owner 全部释放 |
| 异步运行时 | Sched → Queue/Timer/Cancel → Error/Obs | result/error、deadline、cancel、drain 与指标一致 |
| 模型文件生命周期 | Watch/Transfer/File/Mapping → Mem | replace/resume/map/close 后文件与映射状态一致 |
| 服务连接 | TCP/TLS → HTTP/1/HTTP/2/WebSocket | framing、stream、backpressure、cancel 与 close 完整 |
| 工作进程 | Process → LocalStream → Sched cancel | stdio、timeout/tree termination、reap 与 teardown 完整 |

这些 journey 只验证 HYLE 已公开的通用机制组合，不下沉 router、JSON schema、模型格式、batching 或
部署策略，也不能替代各 factor 的 checked arithmetic、OOM、race 等原子反证。

## 有限工作集与预算

| ID | 工作 | 完成判据 | 当前状态 |
|---|---|---|---|
| Q01 | 生成公开头、factor、测试和现有证据 inventory | 29 factor 无遗漏；孤儿公开头/测试显式报错 | `COMPLETE` |
| Q02 | 建立 Linux line/branch discovery profile | 可复现、隔离 build、输出 zero-hit 清单；百分比不作 gate | `COMPLETE` |
| Q03 | Wave A 语义矩阵与缺口测试 | 每个公开能力格为 PROVEN/NA 或诚实 UNPROVEN | `COMPLETE` |
| Q04 | Wave B 语义矩阵与缺口测试 | 同上 | `COMPLETE` |
| Q05 | Wave C 语义矩阵与缺口测试 | 同上 | `COMPLETE` |
| Q06 | Wave D 语义矩阵与缺口测试 | 同上 | `COMPLETE` |
| Q07 | Wave E 语义矩阵与缺口测试 | 同上 | `COMPLETE` |
| Q08 | 五条跨模块 journey | public API、失败/取消/teardown 均可观察且有 timeout | `QUEUED` |
| Q09 | 平台、sanitizer、三交付与 CI 收束 | 风险最小充分矩阵通过；无笛卡尔积式重复 | `QUEUED` |
| Q10 | 终局红队与裁决 | 无伪 PROVEN；缺陷均定向处理；文档/工具合同一致 | `QUEUED` |

每个 factor 的静态审计以一个工作批次为预算：一旦所有公开能力已经映射，停止继续枚举。每个新发现
测试缺口最多先选择一种最能反证风险的技术，不为同一事实机械叠加 table/property/fuzz/stress。若同一
缺陷经过三轮修正仍未解决，或必须改变产品合同，立即按模块协议裁决，不在横向任务内无限施工。

## 预声明验证

- 新增 inventory/coverage 工具的 focused unit tests 与 `--help`/错误输入；
- 每波受影响 factor tests 与公开头；
- WSL/Linux：coverage discovery，ASan/UBSan；TSan 只跑真实共享状态与异步生命周期；
- Windows：原生 IO/process/platform backend 与 MSVC shipping layout；
- macOS：真实 kqueue/watch/process/VM/backend 风险；
- 受影响产品的 source、relocated static、relocated shared consumer；
- `python scripts/check-module-closure.py`；
- `python scripts/check-docs.py`；
- `git diff --check`。

本轮不改变 `module-status.json`，因此项目审计本身没有 candidate commit，也不重签旧 P1 证据。

## Q01 结果

`python scripts/correctness-inventory.py --revision HEAD --check` 在 baseline 上得到 29 个 `CLOSED`
factor、185 个 factor-owned public headers、1 个有意的跨切 Trace 头、173 个测试翻译单元、61,483 行
翻译单元源码和 2,057 个 GoogleTest declaration，ownership/registration issue 为 0。两个未由普通
factor CMake fragment 列出的 SIMD 源，分别被 SVE/RVV QEMU driver 直接登记；工具会保存这些引用，
不把它们误报成孤儿。输出明确标为 discovery，不能转换成语义 `PROVEN`。

工具接入的 change-impact 反证还发现 selector focused test 停留在 Process/WebSocket 加入前的 29 factor、
14 product 和旧 reverse-consumer 集。断言已按中央图精确更新为 31 factor、15 product，并补入
`process`/`io_websocket` 的真实传播边；41 项 selector 测试通过。inventory 及其测试只选择两个 focused
Python suite、docs 与 closure gate，选择 factor 数为 0；它没有借维护改动触发全库 native build。
首个 Core coverage probe 还发现 closure focused test 硬编码了治理迁移时的 26 CLOSED / 2 QUEUED，
当前 registry 合法增长到 29 CLOSED 后因此让 full CTest 失败。该重复状态真相已删除；测试继续验证 schema、
候选 identity、退役术语和完整状态迁移，而当前分布只由 registry 决定。

## Q02 结果

`correctness-coverage.py` 已提供 `run`（全新 configure/build/factor-only ctest/collect）和 `collect`（已有
GCC coverage build）两条路线。它固定 GCC 14 JSON、line+branch、独占非空目录拒绝、默认并行度 2、
per-test 120 秒与 phase 1,800 秒硬 timeout；shipping 与 fault instrumentation 必须分开构建。公开 CMake
没有新增 coverage option，普通用户和发布 artifact 不受影响。configure 前会绑定 source tree、factor
与逐产品文件 hash；`collect` 拒绝没有该身份的旧 build，不能把当前工作树冒充构建输入。

WSL Core 端到端探针从零配置、构建并只运行 6 个 `core` label 测试，ctest 实耗 1.67 秒；报告绑定
source tree 与 gcov 版本，发现 54/110 executable lines、4/29 branches、6 个本 build 未观察文件。该低点
不形成 Core 缺陷或状态结论：其中包含纯编译期头和 abort 前无法写回 `.gcda` 的 fail-fast 路径，正是
下一波逐能力语义回映要区分的仪器负空间。focused 聚合测试验证过滤第三方、跨 TU 合并、zero-hit range、
function/branch 与 unobserved file 语义。

Q06 首次覆盖 Sched 时，GCC 14.2 在 `-O0 --coverage` 对标准 coroutine allocator 形状触发了已知
[GCC PR 109224](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109224)：templated placement-like
`operator new(size_t, Args...)` 与 coroutine 实际要求的 usual sized delete 被误报为 mismatch。coverage
profile 只把这一条 C++ warning 从 error 降为可见 warning；普通 WERROR/shipping build、其他 warning 和
产品代码均不变。这样既不伪修 promise allocation，也不让 discovery 仪器阻塞合法模块。

## Q03 结果

Wave A 的六个 factor 已完成逐能力回映，详见
[`correctness/wave-a-2026-09`](../../evidence/correctness/wave-a-2026-09.md)。干净 HEAD tree 上的
shipping run 为 19 个测试组、18 PASS/0 FAIL/1 个只适用于优化构建的合法 SKIP；独立 fault run 的
Error/Time 为 5/5 PASS。旧 candidate 到当前 HEAD 的产品路径均无变化，Time 只增加边界测试，故历史
三平台/结构证据可复用；当前 build/package 三交付、OS/ISA 专属路线和 sanitizer 仍明确留给 Q09，未
写成伪 PASS。zero-hit 路径逐项归因为编译期头、平台/工具链分支、abort 无法 flush 或已覆盖能力的等价
模板实例，没有发现产品 correctness defect，也没有触发任何模块 reopen。

## Q04 结果

Wave B 的七个 factor 已完成逐能力回映，详见
[`correctness/wave-b-2026-09`](../../evidence/correctness/wave-b-2026-09.md)。当前公开头与实现相对各自
registry candidate 均无差异；shipping discovery 的 14/14 CTest group 全部通过。独立 fault build
暴露出单体 Mem executable 在 gcov 下不能在 120 秒内作为一个测试原子运行；该部分结果被作废，清空
计数后以相同硬 timeout 运行完整 Platform/Platform-fs 与 17 个 Mem 故障/诊断用例，全部通过。

本波未发现产品缺陷或 reopen trigger。12 个 unobserved product file 都是 façade、declaration、
visibility 或 private include；Windows/macOS/ARM/多 NUMA backend、当前 build/package 三交付和适用
sanitizer 仍诚实留给 Q09。覆盖率只作路径发现，没有被转换成 gate、API 改动或状态结论。

## Q05 结果

Wave C 的七个 factor 已完成逐能力回映，详见
[`correctness/wave-c-2026-09`](../../evidence/correctness/wave-c-2026-09.md)。WSL shipping 整波
21/21 group 通过，新增高风险测试的 Windows MSVC Release 回放 5/5 通过。审计补齐 assignment/OOM、
构造回滚、并发异常解锁、确定性的 StringInterner generation、UTF 六方向容量和 metrics overflow 等
直接证据；没有修改产品代码。

本波 `DEFECT=0`，不触发模块 reopen。最终 discovery 观察 11,696/12,697 executable lines 与
5,508/8,278 branches；49 个未观察 product file 已归因为 façade/template manifest 或非本机 ISA/backend，
没有把覆盖率写成正确性分数。当前 sanitizer、non-host ISA、Trace ON、三平台最小 backend 和三种交付
仍明确留给 Q09。

## Q06 结果

Sched 已完成逐公开能力回映，详见
[`correctness/wave-d-2026-09`](../../evidence/correctness/wave-d-2026-09.md)。本轮补齐 Scheduler/BlockingPool/
Cancel/Task/Join/TaskScope 的分阶段失败与 ownership、消息原语 close/blocked wake/retained-value teardown、
同步计数边界、priority/bulk/deadline 等高风险证据，没有修改产品代码。

清空旧 profile 后，WSL 公开头与运行时 2/2 group、运行时 253/253 GoogleTest 通过；Windows MSVC 19.50
Release shared/WERROR 的同两组也通过。最终 discovery 为 3,880/4,163 executable lines、
1,805/2,457 branches；唯二零命中执行函数是私有 Sched↔IO attach/detach，依法由 Q07 的 `IoContext`
公共组合闭合，不用私有测试伪造证据。本波 `DEFECT=0`，Sched 不触发 reopen；sanitizer、macOS 与完整
三交付矩阵仍诚实保留给 Q09。

## Q07 结果

IO/Process 八个 factor 已完成逐公开能力回映，详见
[`correctness/wave-e-2026-09`](../../evidence/correctness/wave-e-2026-09.md)。最终 WSL feature-off
shipping discovery 的 31/31 group 通过；feature-on 的 TLS/HTTP/HTTP2/Transfer/WebSocket 16/16 group
通过。后续只按 change impact 回放新增 Transfer HTTP status taxonomy 1/1，没有为单项测试改动重复运行整波。

审计补齐 DNS、TCP/UDP/local、File awaitable、HTTP transport/body/source/writer/Upgrade、Transfer
Plan/Checkpoint/HTTP failure、Watch、WebSocket 和 Process 的 move/ownership/failure/teardown 直接证据。
最终 discovery 为 6,950/8,825 executable lines、3,552/6,514 branches；8 个未观察文件均路由到
feature-on 或 Q09 native backend，没有用比例签署正确性。

本波发现一项 GCC 14.2 source/shared WERROR 产品交付缺陷，已依法定向 reopen、修复并重新关闭
`io_websocket`；当前未修复 `DEFECT=0`，其余 factor 不触发 reopen。Windows/macOS native backend、
Linux fallback、适用 sanitizer、三交付及 CI product-shape 保留为 `UNPROVEN(Q09)`；五条跨模块组合留给
Q08。
