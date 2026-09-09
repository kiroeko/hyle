# Util-core v4 性能证据


> **当前工程 verdict：`PASS`。** 最终候选
> `439b8915e2c63a6a15545324444b594a25c62a62` / tree
> `a7891b8d6a3dab26599e443f9e03150a195b3ba9` 保持 Util-core 的 O(1)、零分配基础形状；固定的
> FunctionRef 同合同 P1 重放中，两项为 `PARITY`，bound-member 一项为 `WIN`，三项 5% 非劣门
> 全部 `PASS`。本页不把亚纳秒点值外推成全操作、全平台或端到端推理 SOTA。

设计见 [util architecture](../../../architecture/util-design.md)，使用选择见
[util guide](../../../user/util.md)，产品与红队事实见
[Util-core v4 task](../../tasks/util/util-core-v4-resource-move-ctad.md)。

## 为什么这些比较可比

- peer：Abseil `989c696cd131fdaba300c44eb86e1c00fd3a4b56`，clean header-only source；
- contract：`util-core-function-ref-p1-v1`，SHA-256
  `b6d0c63217540f630bdd3a422d8938cb151cc48281f006e94c84104ca1702b37`；
- consumer matrix：`bench/util_core/function_ref_consumer_shapes_v1.json`，SHA-256
  `f3ce0ece50863825dad54120b535baa88e03a6aa066df74ec1a6afaa3a305031`；
- build：W1-WIN，MSVC 19.50.35728，Release/static、portable ISA、LTO off；candidate 与 peer 均从
  clean source 构建；
- 每个 workload 都是 caller-owned synchronous call-only view，执行同一 4096-record traversal、同一
  callable body、异常合同、checksum 和 charged phases；setup/teardown 均在计时外；
- strata 为 primary/placement × candidate-first/peer-first；每层 15 个独立进程样本、99.26%
  distribution-free CI、5% material/non-inferiority margin。ratio 大于 1 有利于 HYLE。

因此这不是把 owning callback、异步生命周期或不同工作量硬比成同一功能。4096 只属于冻结的正常
consumer workload，没有进入 public API、对象 layout、算法分支或常态 metadata。

## P1 资格化结果

| workload | stratum | median advantage ratio | 99.26% CI | exact / 5% NI |
|---|---|---:|---:|---|
| const borrowed | primary / candidate-first | 1.00209 | [0.97777, 1.01202] | `PARITY / PASS` |
| const borrowed | primary / peer-first | 1.00310 | [0.99386, 1.02503] | `PARITY / PASS` |
| const borrowed | placement / candidate-first | 1.00594 | [0.99796, 1.01966] | `PARITY / PASS` |
| const borrowed | placement / peer-first | 1.00315 | [0.98967, 1.00856] | `PARITY / PASS` |
| nontype free | primary / candidate-first | 0.99493 | [0.98901, 1.01363] | `PARITY / PASS` |
| nontype free | primary / peer-first | 1.00237 | [0.99767, 1.01680] | `PARITY / PASS` |
| nontype free | placement / candidate-first | 0.99156 | [0.97622, 1.00983] | `PARITY / PASS` |
| nontype free | placement / peer-first | 0.99924 | [0.97713, 1.00578] | `PARITY / PASS` |
| nontype bound member | primary / candidate-first | 1.24636 | [1.23358, 1.25666] | `WIN / PASS` |
| nontype bound member | primary / peer-first | 1.26437 | [1.22898, 1.28901] | `WIN / PASS` |
| nontype bound member | placement / candidate-first | 1.24523 | [1.23400, 1.26088] | `WIN / PASS` |
| nontype bound member | placement / peer-first | 1.25765 | [1.23220, 1.26494] | `WIN / PASS` |

free-function 的两个低于 1 点值完整保留；其 CI 跨过 1、方向不形成四层固定劣势，且全部通过 5%
非劣门，因此按预先冻结规则是 `PARITY`，不能据此修改实现。bound-member 四层均稳定在约
1.245--1.264 倍，按同一规则为 `WIN`。A/A、known-delta、timer-floor、placement、artifact/build
identity 和 consumer provenance 均通过；独立 verifier 返回 `QUALIFIED`、路径已验证、provenance
`EXACT`、environment policy `CURRENT`。

## P0 结构、红队与三平台

- `UniqueResource` 仍只保存 resource、deleter 和 ownership bit；deleter 的 nothrow move-or-copy
  路由在编译期完成，没有 allocation、lock、动态分发或新增 runtime mode；
- checked factory 把 equality 结果到 `bool` 的完整转换也纳入 nothrow constraint。红队构造的 throwing
  conversion proxy 现在 compile-negative，不再可能进入错误 `noexcept` 后 terminate；
- FunctionRef 仍为两个机器字；新 member CTAD 只读取成员函数类型中已有的 signature/cv/`noexcept`，
  继续复用原 nontype invoker，不增加运行时路线；ref-qualified、volatile 和 overload ambiguity 不猜测；
- probe source `tests/util_core/codegen_probe.cc` SHA-256：
  `F6463A779BE725EA517372574F026CABA896471DB7F6A94BB69E7D85B8232841`；对应 MSVC、GCC 14、
  AppleClang 21 object SHA-256 分别为
  `AFAD22627C95579D5868751B3628BA84DDC2375622917ADD98921F7410D1B5A4`、
  `005F0505968655B7C84270056D85117C8F5167EE9ACBD4C6686442B99A910E7F`、
  `20D2FD6061A113328278170F4FD7AD99F5B9196607AB6763D6FA32AA3E2276D0`；
- exact candidate 在 W1-WIN/MSVC 20/20、W1-WSL/GCC 14 + UBSan 19/19、M1-MAC/AppleClang 21 +
  UBSan 18/18 applicable factor tests 通过；逐头、no-exceptions 与 code-shape 门通过；三平台 Util
  source embedding、搬迁后 shared/static installed consumer 全部通过；中央 component contract 通过。

## Retained bundle 与边界

原始工程 bundle 位于
`C:/public/project-hyle/_validation/util-core-v4-439b8915-perf-evidence`。关键 SHA-256：

- manifest：`E9A618A9F97281F43F755569EE6403EF9C4115C79B51FEF89C9AC00891C7F4FC`；
- summary：`07082A8654CC7AD317C79415CF4E4133E7B142E6C461FEEA5E20A79BC019157C`；
- observations：`2553E980A6D142F0DB9DF7D0A6150A3F735F62DCC5D5D5370686770E89458CBE`。

未签范围包括 owning/nullable/async callback、任意 signature、跨平台 timing、Scope/UniqueResource
纳秒排名、端到端推理 KPI、universal SOTA 与 Formal。资源类型若只有 potentially-throwing copy，且
deleter 只能 move，则 owner move 失败会由已转移 deleter 回收 token 并 disarm source；它保证不泄漏和
不重复释放，但不承诺 source 在异常后继续持有资源。
