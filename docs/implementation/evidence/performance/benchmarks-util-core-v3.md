# Util-core v3 性能证据

> **当前工程 verdict：`PASS`。** 候选
> `763574765d22f8b3f6dbd6046aee124d1117eea3` / tree
> `9ccc0e7a932f7ec4a7345fe56f3525e747587606` 在 W1-WIN 完成 fresh、n=15、同合同 Abseil
> qualification；三个 FunctionRef consumer workload 的 non-inferiority gate 全部 `PASS`。本页不声明
> Util-core 全操作、全平台、任意 callable 或推理端到端 SOTA。

设计见 [util architecture](../../../architecture/util-design.md)，使用选择见
[util guide](../../../user/util.md)，本轮事实见
[Util-core v3 task](../../tasks/util/util-core-v3-foundation-utility-superset.md)。v2 对 Ref/SpinLock 的旧候选
结论仍是历史输入，不能冒充 v3 FunctionRef 的当前证据。

## 比较合同

- peer：Abseil `989c696cd131fdaba300c44eb86e1c00fd3a4b56`，clean source checkout，header-only；
- contract：`util-core-function-ref-p1-v1`，SHA-256
  `b6d0c63217540f630bdd3a422d8938cb151cc48281f006e94c84104ca1702b37`；
- semantic matrix：`bench/util_core/function_ref_consumer_shapes_v1.json`，SHA-256
  `f3ce0ece50863825dad54120b535baa88e03a6aa066df74ec1a6afaa3a305031`；
- build：MSVC 19.50.35728，Release/static，portable ISA，LTO off；两侧从 clean source 编译；
- strata：primary/placement × candidate-first/peer-first；每项 15 个独立进程样本，99.26%
  distribution-free CI，5% non-inferiority limit；ratio 大于 1 有利于 HYLE；
- setup 在计时外；两侧均借用 caller-owned callable/object，执行同一 4096-record traversal、同一函数体和
  checksum，不把 ownership、allocation、thread-safety 或 teardown 的差异藏进实现比较。

三个 workload 是 `SAME_CONTRACT`，而不是只因输出 checksum 相同就硬比：const callable、compile-time
free function 和 compile-time bound member 的 ownership/lifetime、result、charged phases、异常合同、
同步 call-only progress 与交付形状逐格相同。它们是由成熟 peer 与北极星调用方式形成的
`PROVISIONAL_ARCHETYPE`，不冒充已有 HYLE 推理引擎 trace。

## 资格化结果

| workload | stratum | median advantage ratio | 99.26% CI | exact / 5% NI |
|---|---|---:|---:|---|
| const borrowed | primary / candidate-first | 0.99830 | [0.98897, 1.02318] | `PARITY / PASS` |
| const borrowed | primary / peer-first | 1.00236 | [0.99292, 1.00778] | `PARITY / PASS` |
| const borrowed | placement / candidate-first | 1.00252 | [0.99617, 1.01629] | `PARITY / PASS` |
| const borrowed | placement / peer-first | 0.99972 | [0.99690, 1.02979] | `PARITY / PASS` |
| nontype free | primary / candidate-first | 0.99878 | [0.98804, 1.00452] | `PARITY / PASS` |
| nontype free | primary / peer-first | 0.99608 | [0.96790, 1.00959] | `PARITY / PASS` |
| nontype free | placement / candidate-first | 1.00485 | [0.99751, 1.03024] | `PARITY / PASS` |
| nontype free | placement / peer-first | 0.99948 | [0.96625, 1.01540] | `PARITY / PASS` |
| nontype bound member | primary / candidate-first | 1.24769 | [1.22235, 1.25192] | `WIN / PASS` |
| nontype bound member | primary / peer-first | 1.25504 | [1.24606, 1.26701] | `WIN / PASS` |
| nontype bound member | placement / candidate-first | 1.24965 | [1.24405, 1.26121] | `WIN / PASS` |
| nontype bound member | placement / peer-first | 1.24872 | [1.21502, 1.26778] | `WIN / PASS` |

const borrow 与 free-function 的 combined exact verdict 均为 `PARITY`；bound-member 四层均在约
1.25×，combined exact verdict 为 `WIN`。前两项少数低于 1 的点值没有被隐藏：方向随执行顺序/placement
变化，全部 CI 跨过 1 且完整通过 equivalence/non-inferiority gate；不存在“全部 strata 固定输”的额外
复核触发条件。结果不外推到 owning callback、异步 storage 或其他 signature。

A/A、5%/10% known-delta、timer-floor、placement、artifact/build identity 与 consumer provenance
全部 `PASS`，n=15 已满足资格，未触发 n=31。独立 verifier 返回
`qualification=QUALIFIED`、`paths_verified=true`、`provenance_compatibility=EXACT`、
`environment_policy_compatibility=CURRENT` 与 `current_certification_policy_eligible=true`。

## P0 结构与三平台旁证

FunctionRef runtime const route 保持两个机器字和一次 erased tail invocation；nontype free route 在 GCC 与
AppleClang 的 observable caller 中直接降为目标 arithmetic，未保存 runtime target。ScopeFail/
ScopeSuccess 与 UniqueResource 都是栈内 O(1)、零分配、零锁、零 dispatch 的 value owner；conditional
guard 只保存 callable、construction-time exception count 与 active bit，resource owner 只保存 resource、
deleter 与 ownership bit。

probe source `tests/util_core/codegen_probe.cc` SHA-256 为
`47C129E485F1403887EE2BC28B1C5E432B67CCB77DE3E30323B471AD5EE0CE7D`；对应 MSVC/GCC/AppleClang object
SHA-256 分别为
`E193DA5492750EC521503D221F0A5566AE0C7B9446BE809AF514D69822EC824D`、
`47BAB2AE51D26C37D7E079FB16B9D4C88FF58B1DB6235CB3142C56FCAF78B9C6`、
`4B6FAA92970070C3F6A1A76E1A5A6C9EA177812FFE06C91CE97A82806665B5CC`。

候选精确 worktree 在 W1-WIN/MSVC、W1-WSL/GCC 14 + UBSan、M1-MAC/AppleClang 通过 Util-core unit、
逐头 self-contained 与 no-exceptions compile。三平台 source embedding、搬迁后 shared/static Util
consumer 通过；显式使用 const/nontype FunctionRef、conditional guards 与 UniqueResource 的新 installed
sample 也在三平台的 relocated shared/static SDK 上运行通过。

## Retained bundle 与限制

最终 bundle 位于
`C:/public/project-hyle/_validation/util-core-v3-refresh-perf-evidence`。manifest 与 summary SHA-256 为：

- `30D9C54FABBE68545C01DB8BBB1DDB491C4A02AC385B7258BAECF7BFA9230091`；
- `A6B5F42249FCC02C37710D96D361F7610E2C4559D2D648BC2D5E071F2DEC7DFB`。

早先针对 `ff203194...` 的 PASS bundle 因 AppleClang no-exceptions warning 修复刷新 candidate identity，
只保留为 superseded diagnostic；本页只使用 `76357476...` 的完整重放。Formal 仍为 `PENDING`。

未签范围包括 owning/nullable/async callback、任意 signature、跨平台 timing、Scope/UniqueResource 纳秒排名、
deleter 失败恢复、端到端推理 KPI 和 universal SOTA。FunctionRef 生命周期、异常条件、sentinel 或交付形状
变化时必须重新审计，不能通过 benchmark-only overload、layout metadata 或换执行顺序维持绿色结果。
