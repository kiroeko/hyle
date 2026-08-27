# util_core v1 严格收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是当前 `util_core-v1` 轮的冻结卡与施工事实。2026-07 的 util-core 任务、点值和绿色结论只作
> 审计输入；当前状态只看 [`module-status.json`](../../module-status.json)。

## Closure card（`AUDITED`）

- component / owner / round ID：`util_core` / util-core maintainers / `util-core-v1`。
- task kind：`REFACTOR`；baseline commit：
  `c5e26e71b1b5d11d8d36913ab5c3930c849495fb`。
- 进入原因与 demand/defect ID：
  - `UC-V1-BOUNDARY-01`：Core 重构后，五个 header-only primitive 仍只继承旧 util 总任务的结论，
    尚未按当前能力/消费者/依赖方法裁决；
  - `UC-V1-EVIDENCE-01`：历史专用 benchmark 没有 current consumer provenance、统一资格合同或当前
    peer identity，不能直接成为 P1 证书；
  - `UC-V1-ORDER-01`：roadmap 曾把 `platform` 排在其真实 `PUBLIC time` 依赖之前，必须恢复依赖优先
    的施工顺序。
- 真实 consumer task / normal workload：
  - mem `FiberStackPool`、sched join/sync 与 graph first-error 的几个指令级受锁状态转移；
  - util-runtime Registry factory 返回的 intrusive owning handle 与编译期类型标签；
  - 本地推理采样、随机化算法和工程计算需要的可复现非密码随机流、无偏 bounded draw 与 shuffle；
  - C++ 资源 acquisition 后的词法 rollback/cleanup。
- 范围：`random.h`、`ref_counted.h`、`scope_guard.h`、`spin_lock.h`、`type_name.h` 的功能面、公开/internal
  边界、ownership/concurrency/failure/复杂度、C++ 易用性、直接依赖与消费者采用、source/package
  交付、architecture/user/task/performance 文档。
- 非目标：util-runtime Registry/demangle；blocking/fiber mutex；密码学随机数；weak/aliasing owner；
  runtime-stable 或跨编译器 type ID；Native FFI/C API；替 mem/sched/graph 收口；Formal 发布证书。
- 直接依赖：`core`。直接消费者：mem、sched、graph、util-runtime，以及独立 C++ `util_core` 用户；
  没有产品 consumer 的公开能力必须由成熟 peer/语言合同和普通用户任务证明，不能由 benchmark 自证。
- tier 与理由：`P1`。SpinLock、Rng 和 Ref copy/drop 可进入高频路径；ScopeGuard 与 type-name 在同轮按
  P0 结构门处理。默认 material margin 为 **5%**。
- 冻结 workload / 资源预算：
  - Rng：同算法 next、同一 URBG 的 varying bounded draw、1024 元素 shuffle；32B state、零分配；
  - Ref：预建对象的自然 copy/drop 与跨函数 owning handoff；一指针 owner、对象内 32-bit count；
  - SpinLock：无争用和两方短状态 handoff；八线程持续争用只作选择边界诊断，不是必胜 workload；
  - ScopeGuard/type-name：无 runtime allocation/indirection，release codegen/constexpr/layout 门。
- 总调查预算：两小时；性能预算 40 分钟，独立红队 20 分钟。P1 异常最多一次复现/反序、一次
  codegen/profile 机制诊断、一个产品修复批次和一次受影响重跑；不在本轮改中央统计仪器。
- machine profile：`W1-WIN` 为 P1 主裁决机；`M1-MAC` 是 SpinLock 已知 ARM holdout；`W1-WSL`
  承担 GCC、sanitizer 与可移植性。远端只验证 canonical Windows 提交。
- instrument profile：P1 `closure-perf-v1`；结构/correctness 为 `closure-correctness-v1`。
- strong peers（固定本地源码）：
  - SpinLock：oneTBB `e349c15263bbb1d6b48f2b514c06201e1b90eea3` `spin_mutex`；Folly
    `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` `MicroSpinLock` 作形状/退避查漏；
  - Rng：同一 Folly commit 的 `xoshiro256pp`；C++23 UniformRandomBitGenerator/distribution/shuffle
    是组合合同；
  - ScopeGuard：Abseil `81c993af4fb88b3a51829a796df02251fa27102f` `Cleanup` 与同一 Folly commit
    的 `ScopeGuard`；
  - Ref：同一 Folly commit 的 channels `IntrusivePtrBase`/Boost intrusive pointer 形状；
    `std::shared_ptr` 只作更宽 ownership 的成本下界，不冒充同能力 peer；
  - type-name：Boost `df8eeb90846648464e3c8e378e312f0cec772e83` `boost::core::type_name` 与受支持
    compiler 的 pretty-function 合同；runtime demangle 不属于同交付形状。
- 适用 correctness / delivery / performance gates：normal、zero/full bounds、invalid state/contract、
  move/destroy、reference overflow/underflow contract、concurrency/stress、deterministic/property、
  ASan/UBSan/TSan、MSVC/GCC/AppleClang、header self-containment、独立 source 与 relocated header-only
  static/shared SDK consumer、直接消费者编译/测试、P1 qualification、P0 codegen、文档与红队。
- 预期输出：有限 capability/gap ledger、冻结 FIX、candidate commit/tree、逐风险 verdict、P1 manifest、
  user decision table/pitfalls、依赖回审、限制/reopen triggers、terminal receipt 与四项 LOC。
- 停止条件 / 可接受限制：全部适用门通过则 `CLOSED`；任一 correctness/delivery blocker 或 required
  P1 workload 无法通过资格/5% 非劣门则精确 `BLOCKED`。可接受限制包括非密码 RNG、caller-owned
  Ref 生存期、unfair thread spin、compiler-local type spelling，以及没有 weak/aliasing/runtime-stable ID。

## Capability / gap ledger

本轮只把真实 consumer、标准/成熟实现的共同形状和可复现缺陷提升为 `MUST`；单个 peer 独有能力
保留为 `GENERAL` 查漏项，不因“别的库有”自动扩张产品面。

| ID | requirement class | demand kind | exact consumer / peer / standard source | normal shape | semantic / artifact / delivery owner | ownership/failure/concurrency | complexity/resource target | disposition | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|
| UC-RNG-01 | MUST | consumer + peer + standard | inference sampling/randomized algorithms；Folly xoshiro；C++ URBG | deterministic stream、bounded draw、shuffle、explicit/TLS stream | util_core / header / build-package | instance/thread owns mutable state；not crypto | scalar O(1)、bounded expected O(1)、shuffle O(n)、32B、零分配 | KEEP | P1 | 固定序列/state/jump/property/同引擎 bounded+shuffle/codegen；不承诺 secure entropy |
| UC-REF-01 | MUST | direct consumer + peer + standard | util Registry；Folly/Boost intrusive pointer；`shared_ptr` ownership surface | explicit adopt/retain、copy/move/cast、custom teardown、identity compare/hash | util_core / header / build-package | one strong count；borrow 必须有 live owner；失败 dynamic rvalue cast 不消费 owner | copy/drop/cast O(1)、一 pointer owner、对象内 32-bit count | KEEP | P1 | lifetime/concurrency/layout/codegen；无 weak/aliasing；普通单份计数以前置条件换最短原子路 |
| UC-GUARD-01 | MUST | defect + general C++ task + peer/standard | acquisition rollback；libstdc++ scope_exit；Abseil/Folly guard | move-only once cleanup、dismiss/explicit invoke；构造/转移失败不丢 cleanup | util_core / header / build-package | callable value owned；析构 throw 终止；失败 move 后 source 仍拥有动作 | O(1)、栈内、零分配；empty callable 不付无谓 byte | FIX | P0 | 用标准 copy fallback/failsafe 语义修复可抛 callable；move/throw/size/codegen |
| UC-SPIN-01 | MUST | direct consumer + peer | mem/sched/graph；oneTBB/Folly | compact unfair TTAS for a few-instruction critical state | util_core / header / build-package | thread-blocking；external owner lifetime | O(wait)、1B、零分配；短低争用 | KEEP | P1 | mutual exclusion/TSan/同合同 handoff/ARM holdout；八线程持续争用只诊断选择边界 |
| UC-TYPE-01 | MUST | direct consumer + peer/compiler | util Registry；Boost/compiler | constexpr borrowed compiler-local spelling/hash | util_core / header / build-package | static storage borrow；collision possible | compile-time O(name length)、零 runtime state | KEEP | P0 | compiler/qualifier/array/function/incomplete type；非 ABI/serialization ID |
| UC-ADOPT-01 | MUST | dependency audit | mem/sched/graph/util-runtime source + component graph | 每个真实使用点有合法直接边；专用下游状态机不强改成通用 primitive | owning downstream / util_core / build-package | 各 consumer 保留自身状态机 | 当前 component closure 不扩大 | KEEP | P0 | mem/sched PUBLIC、graph PRIVATE、util PUBLIC util_core 均正确；相邻专用实现见 backlog |
| UC-RNG-G1 | GENERAL | peer | Folly SeedSeq/batched xoshiro | richer seed material / bulk generation | future consumer / util or algorithm owner | caller-owned output/state | bulk O(n)，可能 SIMD | DEFER | P2 | 当前 consumer 没有 bulk/SeedSeq demand；显式 state + jump 已覆盖可复现拆流，不能为行数扩 API |
| UC-REF-G1 | GENERAL | peer + standard | Boost/std reinterpret/weak/aliasing surfaces | unrelated cast 或 control-block ownership | different ownership type | 需要额外 lifetime identity/metadata | 不得破坏一指针 Ref 合同 | REJECT | P0 | reinterpret 会让 CRTP decref/destroy 落到无关静态类型；weak/aliasing 应选 shared_ptr 或另立模型 |
| UC-GUARD-G1 | GENERAL | peer + TS standard | Folly/libstdc++ scope_fail/scope_success/unique_resource | exception-count conditional action / generic resource pair | future named consumer | exception-only outcome 与 Status return 不同 | O(1) | REJECT | P0 | HYLE 主失败流是 Status/Result；typed RAII 优先，不能让 exception-only guard 冒充 error-return rollback |
| UC-PERF-01 | MUST | evidence defect | `closure-perf-v1`；当前 util_core bench | current consumer-shaped P1 evidence | bench contract / perf instrument | candidate/peer 同算法、同计费边界、反序与 qualification controls | 5% non-inferiority；Scope/type 用 P0 structural | FIX | P1 | RNG 不再用不同 engine 冒充同合同；Ref 不让 shared_ptr 冒充 exact peer；SpinLock 不把线程创建混进 handoff |
| UC-DOC-01 | MUST | user + architecture contract | util architecture/user guide/current registry | 选择表、坑、边界、当前状态和证据入口一致 | docs / util_core / build-package | 明示 borrow/TLS/spin/destructor 限制 | 无运行时成本 | FIX | P0 | 去掉 current 页 provisional 误导；历史页仍保持 HISTORICAL |

- 公开入口反向追踪：`SpinLock/cpu_relax` 落到 mem、sched、graph；`Ref/type_name` 落到
  util-runtime Registry；`Rng` 落到推理采样/随机算法普通任务；`ScopeGuard` 落到 acquisition 后多早退
  rollback。没有入口只由 benchmark 命名。
- 语法消费者 / 语义重复扫描：真实直连边均与中央 component graph 一致。container 的两个 adaptive
  lock 私有复制了 pause hint，但它们还拥有 park/wait 策略且 container 尚未收口；登记
  `CONT-UTIL-CPU-RELAX-01`，不在 util_core 轮越界改变其依赖/API。sched 锁内构造/移动泛型 `T` 是
  `SCHED-SPIN-HOLD-01`，由 sched 判断 blocking/exception contract，不能据此污染 SpinLock 本身。
- public/internal/higher-layer 裁决：五个公开 header 与 `hyle::util_core` target 保留；secure random、
  distributions/bulk algorithms、weak ownership、blocking/fiber lock、runtime-stable type identity 留在上层或
  独立能力，不塞入本 target。
- 本轮冻结 `FIX` 集：
  1. `ScopeGuard` 对齐 scope_exit/Folly 的构造失败 failsafe、throwing-move copy fallback 与成功后才移交
     ownership，并用 `[[no_unique_address]]` 消除 empty callable 的无谓布局字节；
  2. 补齐相关异常/布局、Ref 排序、RNG jump、type-name 复杂类型和 direct-include 回归测试；产品头只做
     被测试证明需要的直接 include 修正；
  3. 将 util_core P1 harness/contract 改成冻结的同引擎/同算法 normal shapes，并使用现有中央
     `closure-perf-v1` controls；旧不同 PRNG 和 thread-construction 点值只可作历史诊断；
  4. 更新 util architecture/user/current task，明确选择、坑、负空间、证据与当前状态。
- 相邻模块 backlog：`CONT-UTIL-CPU-RELAX-01`、`SCHED-SPIN-HOLD-01`。二者都没有证明当前 util_core
  correctness/delivery blocker，因此不扩大本轮产品修改。

## Candidate

`AUDITED` 冻结有限 FIX 并实现后填写。

## Verification manifest

候选冻结后逐风险填写；未运行项不得写 `PASS`。

## 独立红队

候选冻结后在 20 分钟预算内执行。

## Closure decision

当前为 `SCOPED`，尚无 terminal verdict。
