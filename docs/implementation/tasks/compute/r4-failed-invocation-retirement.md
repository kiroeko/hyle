# R4 依赖修复：失败终态的执行槽退役

## Closure card

- 状态：AUDITED；owner 为 `compute` 与 `graph`。`8f3cffdc` 完成合法 CLOSED → SCOPED。
- baseline：`ab6d87b349345342258b5b3090953119c06167a0`。
- trigger：R4 Marlin 的 per-invocation workspace 含 native locks；成功完成会复位，失败完成不能
  保证复位。`src/compute/operation.cc::mark_operation_terminal` 忽略 terminal Status，Graph 的
  `mark_invocation_terminal` 也无条件清除 in-flight，导致相同槽可再次进入 native launcher。
  这是 R3 terminal record 允许重开的错误/寿命缺陷，不重开 CPU 性能例外或算法选择。
- 主 consumer：Marlin group128 decode/短 prefill，经 Direct 或 Graph/captured replay 使用同一槽。
  正交 consumer：任意 accepted-prefix 后失败的 shared/delegated Provider，不能假设它已重置 scratch。
- 正常旅程：准备不可变 artifact → 创建槽 → 重复成功运行；失败 Completion 到终态后槽退役，
  调用者销毁并重新创建槽。新槽仍必须遵守 Provider/device 的恢复限制；不承诺修复丢失的设备。
- 范围：执行槽的 idle/running/failed 准入、真实终态发布、失败后的 reset/bind/reprepare 诊断，
  以及相应 public comments、用户文档和故障回归。未接受工作的拒绝仍可修正输入后重试。
- 直接依赖：Error/Completion/Provider 原有协议；直接消费者为 CPU/CUDA Provider 和 Graph。
  不修改 Mem/Sched、Queue/event/capture 算法、算子、规划规则、构建治理或性能仪器。
- 性能：P0。复用现有一字节原子状态，保持成功提交的原子次数和槽大小；不新增热 memset、
  provider callback/vtable、allocation、搜索或同步。检查生成代码；R4 最终完整 P1 包含修复后的代码。
  CPU 原 P1 数据保持其原候选身份，不声称新产品字节与它相同，也不重测已经批准的例外。
- 调查预算：20 分钟定向源码审计，一组可控失败回归；不得扩成全模块审计。若发现需要新增终态协议
  或扩大 Provider 生命周期，先停止本修复并重新定界。
- 预声明验证：W1/WSL/Air/Mini 上受影响 Compute/Graph 与直接 CPU consumer；W1 上 cuBLASLt/Marlin
  真实 GPU 用例；适用 WSL ASan/UBSan/TSan；四环境 source/static/shared 的公共执行槽旅程。
  使用已有 source/package harness，重型串行、并行度 2，Windows DLL 邻置。未运行项保留 UNPROVEN。
- stop：有限修复和适用正确性/交付/P0 成立后冻结并结案；不能用受控协议 fixture 代替 CUDA 路线验收。

## 当前审计输入

1. Compute shared submit 和 delegated submit 都使用同一个 InvocationState 的 `atomic<bool>`；
   异步 attachment callback 与同步 completed 分支均把失败槽恢复为空闲。
2. Graph commands/capture/delegated 最终都会经 `mark_invocation_terminal`；该处已按 Status 发布
   outputs_ready，但仍允许重用失败槽。reset 只清 tensor binding，不重建 Provider invocation。
3. `tests/compute/operation_test.cc` 的 mapping/unmap failure 现有用例会复用前一失败 invocation，
   需要按当前真实消费者的失败语义迁移，不能反向约束新路线。
4. 所有外部 payload/workspace 仍必须保活到真实终态；retirement 不提早释放资源，不隐藏 wait，
   不吞晚到错误，不自动重新执行，不影响其它独立槽。

## 审计冻结的有限修复

| ID | 精确问题 | 允许动作 / 验收 |
|---|---|---|
| F1 | shared/delegated terminal 只记录 busy=false | 复用原子字节表示 idle/running/failed；同步和异步失败均不可重提；拒绝前回滚仍 idle |
| F2 | Graph reset/bind/replay 可复用失效 scratch | 复用原子字节，失败终态阻止重提/重绑定/重准备；成功 replay、独立槽和真实完成寿命不变 |
| F3 | 普通入口未说明失败槽退役 | 注释/指南与回归覆盖 accepted failure、late failure、abandon、未接受拒绝、重建槽与成功复用 |

F1–F3 已固定。源码定位还覆盖 terminal attachment 失败分支：只有真实工作已排空才可向调用者
返回无 Completion 的错误；退役状态不能被 admission rollback 再覆盖为空闲。保持既有 Completion
协议，使用现成 wait/attachment 行为，不新增回调或后端终态接口。

### 形状与生命周期冻结

- 原子机制为真实 Completion/lifetime attachment（ROUTED / KEEP）；缺口为 run-slot 的终态准入
  （SELF / FIX）。不改变不可变 artifact 或 Provider 实际 native owner。
- 共享与 delegated Direct 均经同一个私有 InvocationState。Graph commands、capture、delegated
  均经 Graph 的最终 Completion；外层 Graph 独占内部 Provider run-state，因此在该边界统一退役。
- 内部状态为 idle/running/failed，复用原 `atomic<bool>` 的一个字节；成功 admission 仍一次 CAS，
  terminal 仍一次状态 store。失败终态不再返回 idle；未接受的拒绝仍由原 rollback 返回 idle。
- `in_flight()` 在失败已终结时返回 false；`valid()` 仍表示句柄存在。后续 submit 返回
  FailedPrecondition。Graph 的 bind/reset/prepare_replay 不能重新启用失败槽，必须由 Executable
  新建 Invocation；Direct 由 PreparedOperation/PreparedDelegatedOperation 新建。
- Completion 的真实错误仍由 wait/poll 报告。release/abandon 依旧保留执行状态到终态，取消请求
  不能提前复用槽；独立槽和已知成功槽继续运行。资源借用与设备失效限制保持各 owner 合同。
- 所有外部 C++ 签名、目标/交付身份保持现有形状；仅私有状态编码和明确失败后的可用性改变。

### 七问裁决

1. 没有新第三方依赖；Marlin 来源仍由 R4 C2 的 pin/lock/许可证负责。
2. 主 consumer 的 scratch 恢复与正交 accepted/delegated failure 由相同真实终态约束；成功复用、
   拒绝后重试、失败后重建三种旅程分别验证，避免把所有错误都当已接受失败。
3. 复用原 Completion attachment、Mem 所有权、Provider artifact 和 Graph slot；不复制资源管理。
4. P0 检查同大小状态、CAS/store 数、失败清理及生成代码；不给旧 P1 更换 candidate 身份。
5. 仅 C++，无新增 C/FFI 或动态 plugin ABI。
6. shared/source/static 仍通过原 public 类型消费。公共失败合同变更按上面四环境交付范围验证。
7. 在 Direct/delegated/Graph public declaration 与普通指南写明退役/重建，错误无需用户识别
   Provider 私有锁。测试用受控失败协议，不制造 GPU hang/device loss。

## 验收

| Gate | 当前裁决 |
|---|---|
| scope | AUDITED，F1–F3 固定 |
| architecture | AUDITED；复用现有状态字节与 Completion 协议 |
| correctness | UNPROVEN |
| delivery | UNPROVEN |
| usability_docs | UNPROVEN |
| red_team | UNPROVEN |
| engineering_performance | UNPROVEN；仅 P0，禁止本卡启动 CPU 追加 P1 |

当前尚无新产品改动或 candidate。历史 R3 terminal 仍作为旧候选的验收证据保留；其余 CPU owner 的
当前状态不改变。本轮不解除旧 ggml BLOCKED 或 compute_sched QUEUED。

Compute + Graph 当前全实现 **32,183 行**，不含测试/第三方（`ab6d87b3`，统一脚本 self-check PASS）。
