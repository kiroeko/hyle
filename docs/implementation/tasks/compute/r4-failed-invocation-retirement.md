# R4 依赖修复：失败终态的执行槽退役

## Closure card

- 状态：CANDIDATE；owner 为 `compute` 与 `graph`。`8f3cffdc` 完成合法 CLOSED → SCOPED。
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

1. Compute shared submit 与内部 delegated submit 使用 InvocationState 的 `atomic<bool>`；
   普通 DelegatedInvocation 为保留 Provider 自带 completion owner，使用 DirectDelegatedState 的
   pending 原子与 retention callback。这两种 owner 的同步/异步完成都把失败槽恢复为空闲。
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
- 共享 Direct 在 InvocationState 退役；普通 delegated Direct 在其原有 DirectDelegatedState 的
  retention callback 退役，保留 Provider 自带的 completion owner。内部 delegated 入口也保持同一
  失败约束。Graph commands、capture、delegated 均经 Graph 的最终 Completion；外层 Graph 独占
  内部 Provider run-state，因此在该边界统一退役。
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
| scope | PASS；F1–F3 固定且完成 |
| architecture | PASS；复用现有状态字节与 Completion 协议 |
| correctness | PASS；四环境普通回归、真实 CUDA consumers、WSL 三种 sanitizer |
| delivery | PASS；四环境 source/static/shared 公开执行槽旅程 |
| usability_docs | PASS；Direct/delegated/Graph 注释及普通指南说明退役与重建 |
| red_team | PASS；失败后入口封闭、拒绝回滚、真实终态保活和独立槽复核 |
| engineering_performance | PASS；P0 源码与生成代码；未追加 CPU P1 |

### 实现检查点

F1–F3 已实现，当前仍为 AUDITED，跨平台验收完成后再冻结 CANDIDATE。
[初始 W1 证据](../../../../evidence/correctness/r4-failed-invocation-retirement/README.md)：
共同 Compute/Graph 与选定 CPU consumer 最终 9/9 CTest 通过；其中一个旧测试按取消后重建槽的
合同迁移，再仅重跑对应 backend。cuBLASLt 与 Marlin 的实际 GPU 回归各 2/2 通过。
覆盖 shared/internal delegated/public delegated、Graph command/capture 的失败、取消、丢弃完成、
独立槽和重建槽；未接纳拒绝仍可重试。异常 assembly/attachment 失败会先排空已接纳工作再返回错误。

P0 检查确认状态字节大小/对齐和原子 RMW 数不增加，终态增加 Status 解码；不假报指令完全相同。
适用 WSL/Air/Mini、sanitizer、其余直接 CPU consumer 和四环境 source/static/shared 尚未完成，
聚合 correctness/delivery 仍未裁决。只补这些预声明门，不追加 CPU P1。

历史 R3 terminal 保留旧候选身份；其余 CPU owner 状态不改变。本轮不解除旧 ggml BLOCKED 或
compute_sched QUEUED。统一脚本 Compute + Graph 当前全实现 **32,252 行**，不含测试/第三方，
self-check PASS。

### 普通验证补齐

`a116280b` 的受影响产品和测试已冻结，后续 C3 没有更改这些路径。W1 剩余直接 CPU/source
消费者 14/14 通过，与初始九项共同覆盖 23 项；CUDA backend 三项（含 late failure）通过。
WSL 23/23、Air/Mini 各 19/19；四环境 source 和 relocated static/shared 公开旅程全部完成，
每种安装包的 Direct、Graph、CPU executor 均 3/3。原始证据与 SHA 见上述证据目录。

Windows harness 两个问题已定向处理：tests/install 分开配置；同 exe 目录 DLL staging 串行。
保留失败日志，不修改产品/公共治理。CUDA Provider 本身仍由 R4 各自的 card 范围继续验收。
剩余适用项仅 WSL ASan/UBSan/TSan，正在按原声明串行执行；聚合 correctness 尚未最终裁决，
不重跑已通过的普通门、不追加 CPU P1。当前全实现 LOC 随 C3 为 **32,562 行**，不含测试/第三方。

### CANDIDATE 冻结与最终验证

产品 candidate：`a116280bd73678db2790de6ea1d2758ea34b1bf0`。没有在跨平台验证期间改变本卡
产品/测试；C3/C4 的独立 CUDA 工作不属于该 candidate。WSL ASan、UBSan、TSan 各 **12/12**
受影响 runtime CTest 通过；TSan 按环境合同使用 `setarch -R`。原始 configure/build/test/LastTest/
CMakeCache 与完整 archive 身份保存在 `wsl-sanitizers.tar.gz`。外部 oneDNN/JIT 没有随 HYLE
插桩，不据此声称其内部已 sanitizer 验证。W1 CUDA backend 的三个回归单独归档。

独立于实现的一遍红队确认：所有 shared/internal delegated/public delegated 与 Graph 终态都
发布 idle/failed；未受理错误由原 rollback 恢复 idle，受理后的 assembly/attachment 错误先等待
已有前缀，不在返回时提前释放 native scratch。失败槽的 submit/bind/reset/prepare_replay 无绕过，
Provider 自有 completion owner 未被覆盖，独立槽与重建槽通过真实 consumer 验证。测试 fixture
只证明受控协议；不声称修复 device loss 或失败 native runtime。

性能裁决限定 P0：一字节状态的大小/对齐与成功原子次数保持，增加必要终态 Status 解码，没有新增
热分配、搜索、memset、Provider callback 或正常路径等待。旧 CPU P1 仍绑定原 candidate，用户
批准的唯一验收例外不变。七门均 PASS，进入 CANDIDATE；下一状态转换仅记录终审，不重建或重跑 P1。

重开条件：可复现的失败槽重入、过早释放、终态/取消竞争、拒绝后无法重试或与真实 Provider 的
恢复合同冲突。新算法、其它路线的性能或无具体缺陷的再审均不属于本卡。candidate 全实现 LOC
**32,252 行**，当前已提交 HEAD 为 **32,562 行**，不含测试/第三方；不是新架构完成百分比。
