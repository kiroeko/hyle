# util_core v1 严格收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是当前 `util_core-v1` 轮的冻结卡与施工事实。2026-07 的 util-core 任务、点值和绿色结论只作
> 审计输入；当前状态只看 [`module-status.json`](../../module-status.json)。

## Closure card（`SCOPED`）

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

`AUDITED` 前逐项填写；当前不得因历史实现或 benchmark 名称预判 `KEEP/FIX`。

| ID | requirement class | demand kind | exact consumer / peer / standard source | normal shape | semantic / artifact / delivery owner | ownership/failure/concurrency | complexity/resource target | disposition | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|
| UC-RNG-01 | MUST | consumer + peer + standard | inference sampling/randomized algorithms；Folly xoshiro；C++ URBG | deterministic stream、bounded draw、shuffle、explicit/TLS stream | util_core / header / build-package | instance/thread owns mutable state；not crypto | scalar O(1)、shuffle O(n)、32B、零分配 | audit pending | P1 | state/property/peer/codegen；不承诺 secure entropy |
| UC-REF-01 | MUST | direct consumer + peer | util Registry；Folly intrusive pointer shape | explicit adopt/retain、copy/move/cast、custom teardown | util_core / header / build-package | one strong count；borrow 必须有 live owner | copy/drop O(1)、一 pointer owner | audit pending | P1 | lifetime/concurrency/codegen；无 weak/aliasing |
| UC-GUARD-01 | MUST | general C++ task + peer | acquisition rollback；Abseil/Folly guard | move-only once cleanup、dismiss/explicit invoke | util_core / header / build-package | callable value owned；destructor throw terminates | O(1)、栈内、零分配 | audit pending | P0 | exception/move/codegen；优先资源 RAII |
| UC-SPIN-01 | MUST | direct consumer + peer | mem/sched/graph；oneTBB/Folly | compact unfair TTAS for a few-instruction critical state | util_core / header / build-package | thread-blocking；external owner lifetime | O(wait)、1B、零分配；短低争用 | audit pending | P1 | mutual exclusion/TSan/peer/ARM holdout；非长临界区 |
| UC-TYPE-01 | MUST | direct consumer + peer/compiler | util Registry；Boost/compiler | constexpr borrowed compiler-local spelling/hash | util_core / header / build-package | static storage borrow；collision possible | compile-time O(name length)、零 runtime state | audit pending | P0 | compiler/qualifier/codegen；非 ABI/serialization ID |
| UC-ADOPT-01 | MUST | dependency audit | mem/sched/graph/util-runtime source | 合法直连且无同义复制 | owning downstream / util_core / build-package | 各 consumer 保留自身状态机 | 不扩大依赖闭包 | audit pending | P0 | syntax/semantic duplicate/build graph |

## Candidate

`AUDITED` 冻结有限 FIX 并实现后填写。

## Verification manifest

候选冻结后逐风险填写；未运行项不得写 `PASS`。

## 独立红队

候选冻结后在 20 分钟预算内执行。

## Closure decision

当前为 `SCOPED`，尚无 terminal verdict。
