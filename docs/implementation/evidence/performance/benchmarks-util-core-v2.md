# util_core v2 性能证据

> **当前工程 verdict：`PASS`。**candidate
> `d42d992cf99232669f60422f30fb3b50ec9ec4eb` / tree
> `9944f099679503df632b6335e69154656164b30b` 在 W1-WIN 完成 fresh n=15 qualification；
> performance 与 qualification gate、artifact/build identity、A/A、5%/10% known-delta、timer-floor、
> placement 和 consumer provenance 全部 `PASS`。本页只签两个冻结的正常消费者 workload，不声称
> util_core 全操作、全平台或 Formal SOTA。

本页是 [performance 总账](README.md) 的 util_core v2 详节。设计边界见
[util architecture](../../../architecture/util-design.md)，用户选型见
[util guide](../../../user/util.md)，本轮事实见
[util-core-v2 task](../../tasks/util/util-core-v2-capability-completion.md)。旧 util_core/runtime 证据保留在
[benchmarks-util](benchmarks-util.md)，但不覆盖本候选。

## 冻结合同

- candidate：上述 clean source commit/tree；MSVC 19.50.35728，Release/static，portable ISA，LTO off；
- peers：Boost `78505eab9b203e3e26e83893d0685f160133fa0f` 的 `intrusive_ptr` 与 oneTBB
  `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` 的 `spin_mutex`，两份 checkout 均 clean；
- contract：`util-core-p1-v2`，SHA-256
  `8771c52b0d26cb240653625e5d0640a09f3ecef039c34e8a54b3a853e02d2a7a`；
- workload：正常 intrusive owning-handle handoff，以及 `FiberStackPool` 形状的极短锁状态转移；
- strata：primary/placement × candidate-first/peer-first；每项 15 个独立进程样本，99.26% distribution-free
  CI，5% non-inferiority limit 为 `1/1.05 = 0.95238`；ratio 大于 1 有利于 HYLE。

| consumer workload | stratum | median advantage ratio | 99.26% CI | exact / 5% NI |
|---|---|---:|---:|---|
| Ref owning handoff | primary / candidate-first | 1.1003 | [1.0976, 1.1022] | `WIN / PASS` |
| Ref owning handoff | primary / peer-first | 1.1009 | [1.0981, 1.1040] | `WIN / PASS` |
| Ref owning handoff | placement / candidate-first | 1.0991 | [1.0781, 1.1029] | `WIN / PASS` |
| Ref owning handoff | placement / peer-first | 1.1011 | [1.0988, 1.1217] | `WIN / PASS` |
| short state transition | primary / candidate-first | 1.0000 | [0.9741, 1.0037] | `PARITY / PASS` |
| short state transition | primary / peer-first | 0.9986 | [0.9641, 1.0027] | `PARITY / PASS` |
| short state transition | placement / candidate-first | 0.9986 | [0.9627, 1.0033] | `PARITY / PASS` |
| short state transition | placement / peer-first | 0.9990 | [0.9622, 1.0016] | `PARITY / PASS` |

Ref handoff 的四个 strata 都在约 1.10×，所以 combined exact verdict 是 `WIN`。短状态转移的点估计在
0.9986--1.0000，四个 CI 均完整落入 5% equivalence band，combined exact verdict 是 `PARITY`。这里不把
0.1% 左右的点值解释成产品缺陷，也不从 Ref 的结果外推到未测 ownership 拓扑。

仪器 A/A 在两种执行顺序的 median ratio 为 0.9983 / 0.9999，CI 均位于 5% equivalence band；注入的
5% 差异测得 1.0528 / 1.0431，10% 差异测得 1.0996 / 1.1015，且保持单调；两个 workload 的 timer-floor
multiple 均远高于 20×。因此本轮在 n=15 已取得资格，没有触发 n=31 扩展。

## 新能力的结构成本

v2 新增能力不建立新的亚纳秒性能表：当前尚无 material consumer contribution，benchmark 不能反向定义
API。它们按冻结的 P0 结构合同验收：

- `FunctionRef<R(Args...)>` 是两个机器字、trivially copyable、零分配且不复制 callable；大或非平凡
  by-value 参数以同一对象穿过 erased invoker；
- `NotNull<T*>` 是一个 pointer、trivially copyable/standard-layout，动态 null 只在构造时 fail-fast，
  optimized unwrap 为 pointer identity；
- 默认 `RefCounted` 的 optimized probe 保留 atomic RMW；`ThreadUnsafeRefCounted` 为 plain integer，
  没有 runtime mode branch；
- RISC-V rv64gc 发出标准 `0x0100000f` 编码，启用 Zihintpause 时同一编码反汇编为 `pause`；这只证明
  ISA correctness，不冒充 RISC-V 真机 timing。

## Retained bundle 与限制

retained engineering bundle 位于 `C:/hyle-scratch/util-core-v2-perf-evidence`。manifest、summary、
observations 与 contract-lock SHA-256 分别为：

- `d42e407a730bd65d7afd84a6e5904eba9f110b4462fc0cffaf5031fadde48dd2`；
- `9dd623ed2457d9b3277be435ef7aa5c81fc2af2c10c473df3875d5a2e53ffe67`；
- `6aeb3d8bb25e6d3153171d57f4eaf835a1cedb56e6c06d051444d7254c212d01`；
- `ecf4f4760de7d2134cd0694b13931dd3a519ba101f24cc24905e140cab0892b8`。

独立 verifier 返回 `qualification=QUALIFIED`、`paths_verified=true`、
`provenance_compatibility=EXACT`、`environment_policy_compatibility=CURRENT` 与
`current_certification_policy_eligible=true`。该 bundle 是可深验的本机工程证据，不是发布 artifact 或
Formal certificate。

未签范围包括持续/高争用锁、blocking/fiber synchronization、RNG throughput、FunctionRef 的任意
callable/ABI 组合、跨平台性能和端到端推理 KPI。出现同合同 material consumer 失败时回到 AUDITED；
不能通过换顺序、加 benchmark-only metadata 或新增第二个“fast API”解释失败。
