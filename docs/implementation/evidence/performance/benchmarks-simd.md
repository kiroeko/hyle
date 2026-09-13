# SIMD 工程性能证据


> 当前产品状态只看 [`module-status.json`](../../module-status.json)。本页首先记录 SIMD v7
> Compute-real row primitive 和 v6 推理整行合同的当前有限工程证据，随后保留 v3 路由基线；它们都不承诺“全部操作、全部 CPU、
> 全部编译器均为 SOTA”。架构与停止规则见
> [`simd-design`](../../../architecture/simd-design.md) 和
> [`SIMD v7 closure card`](../../tasks/simd/simd-v7-compute-row-primitives.md)。

## SIMD v7：Compute-real row primitive（`QUALIFIED/PASS`）

Compute Native 的 dense f32 consumer 触发了四项可复用连续 row 合同：full-precision reciprocal-sqrt、IEEE
number-min/max、byte-bool value select，以及传播 NaN 并确定 signed zero 的 extrema reduction。普通 fat
artifact 每个完整 row 只有一次缓存后的 target function-pointer hop，exact artifact 直接调用；无逐元素
dispatch、临时 buffer、allocation、lock 或 syscall。

W1-WIN、MSVC 19.50、Release exact AVX2 static、LTO off。唯一正式 P1 使用 31 个独立进程样本、双顺序与双
placement；这是合同预声明 `[15,31]` schedule 在一次 qualification 内的自动扩样。比较对象是相同语义、ISA、
storage/output 与 noinline whole-row 边界的 direct locked-Highway lowering；scalar 只作正确性 oracle。
`advantage ratio > 1` 有利于 HYLE：

| workload | 四个 median ratio 范围 | exact | 5% non-inferiority |
|---|---:|---|---|
| exact reciprocal-sqrt f32 / 4096 | `0.99699–1.00098` | `PARITY` | `PASS` |
| number-min f32 / 11008 | `1.00744–1.00880` | `PARITY` | `PASS` |
| byte-bool select f32 / 4096 | `1.02182–1.02624` | `PARITY` | `PASS` |
| propagating-min f32 / 4096 | `0.99640–1.00962` | `PARITY` | `PASS` |

A/A、5%/10% known-delta、timer-floor、placement、artifact/build/host/input/provenance gates 全部通过；deep
verifier 为 `QUALIFIED/CURRENT/EXACT`。外部 bundle 路径与 SHA-256、三平台 correctness/sanitizer/交付以及
AppleClang 本地常量池标签误报的独立修复，均记录在
[`SIMD v7 closure card`](../../tasks/simd/simd-v7-compute-row-primitives.md)。结论是新增自然入口没有 material
中间层税；不能据此宣称未测 SIMD operation 或所有 ISA 均为 SOTA。

## SIMD v6：修正后推理整行回放（`QUALIFIED/PASS`）

F06b 已把 HYLE 与 direct peer 的 release 参数检查、一次 noinline 调用边界、输入、输出、scratch、
ISA 和计费范围逐项对齐。W1-WIN、MSVC 19.50、Release exact AVX2 static、LTO off；每个预冻结
order/placement stratum 为 15 个独立进程样本。`advantage ratio > 1` 有利于 HYLE：

| workload | 四个 median advantage ratios | exact | 5% non-inferiority |
|---|---|---|---|
| split SwiGLU f32 / 11008 | 1.0025 / 0.9980 / 0.9974 / 1.0002 | 4/4 `PARITY` | 4/4 `PASS` |
| weighted RMSNorm f32 / 4096 | 1.0093 / 1.0014 / 1.0020 / 1.0142 | 4/4 `PARITY` | 4/4 `PASS` |
| expert top-k f32 / 256/6 | 0.9905 / 0.9891 / 0.9842 / 0.9870 | 4/4 `PARITY` | 4/4 `PASS` |

A/A、5%/10% known-delta、timer-floor、placement、artifact/build/host/input/provenance gates 全部
通过；A/A 在 15 样本即得到 `PARITY`，所以合同允许的 31 样本扩样没有触发。deep verifier 为
`QUALIFIED/CURRENT/EXACT`。bundle 位于
`C:\public\project-hyle\_validation\simd-v6-provider-consumer-replay\perf-evidence-v5`；contract-lock、
manifest、summary SHA-256 分别为
`0F88BA6B2C0233AE69EB1003747C088545B9E34560E04D070601039BE89B5EE8`、
`F755A646EF0EF90FA1AD2F871B42C6E184335CF257D940EDF6148933114A7D3E`、
`F59DE0DE45944A2AE18C985310B1B4D6E2DA333E4890CDF3C5C9DF4FDF8B054A`；contract SHA-256 为
`6eb9fe307f839ae50997f6f142a7b2fe5dc1c61ea5733dc6005aa39c6f330e33`。

Top-k 在四个固定方向都朝 HYLE 负向，但实际中位差只有约 0.95%--1.58%，不是 5%；完整 row 的
绝对差约 4.3--7.9 ns。源代码和 MSVC codegen 定向审计确认，`k=6` 时两侧核心都是同合同的有界
插入；HYLE 的自然通用入口还同时承载 `k==1` 和大 `k` heap/full-sort 路由，生成函数体约
`0x738` bytes，而只含有界插入 body 的 peer 约 `0x529` bytes。因此固定方向有合理的前端/分支与
代码体积机制，不能全部归为仪器噪声；但它没有达到冻结的 5% material threshold，也没有真实引擎
端到端 profile 证明值得拆分公共 API 或增加常态元数据。本轮保持一个自然的通用 API。若具名引擎
profile 证明该 4--8 ns 成为端到端 material bottleneck，才以 consumer trigger 重开并比较 out-of-line
cold general path、编译期 small-k overload 等自然实现方案。

## SIMD v6：初始推理整行回放（`DIAGNOSTIC/REJECTED`）

本节只保留初始候选的原始统计事实。独立红队发现三条 direct peer 没有承担 HYLE public route 的
release 参数检查，因而不满足 `SAME_CONTRACT`；narrow/quick 特殊值用例也未保证特殊值进入实际向量
lane。统计 verifier 的 `QUALIFIED/CURRENT/EXACT` 不能弥补语义等价失败，所以下列结果不得用于
non-inferiority、SOTA、`PASS` 或模块冻结。修正后的合同和新证据会以新 identity 追加，不改写本节。

候选固定为 `5d7349502311b622d6263ca68071d62dff125b67` / tree
`9a995bbaa0ea43e67f3889682c647cf7907a7bfb`。冻结合同
[`simd-inference-row-p1-v1.json`](../../../../bench/contracts/simd-inference-row-p1-v1.json)
原计划比较本轮新增的三个完整 row：

- split f32 SwiGLU，`n=11008`，HYLE 与 direct Highway 都计算 `silu(gate[i])*value[i]`；
- weighted f32 RMSNorm，`n=4096`，两边都是 f32 accumulation、相同 epsilon、weight、输出和 alias
  边界；
- 非破坏 expert score top-k，`n=256, k=6`，两边都使用 ds4 消费形状的 bounded insertion、同一
  score+`uint32_t` index、NaN-last 与 lower-index tie 语义，且只使用 caller 提供的 `O(k)` 结果槽。

一次 measurement operation 是一次完整 row；元素数不冒充独立 callback batch，构造、allocation 和
数据生成都在计时外。但 HYLE 一侧的 release 参数检查没有在 peer adapter 中对齐，所以初始命题仍是
苹果对橘子，不能因其余 storage/output/scratch/ISA 条目相同而升级为实现级比较。

W1-WIN、MSVC 19.50、Release exact AVX2 static、LTO off；31 个独立进程样本，primary/placement 与
candidate/peer-first 共四个预冻结 strata。`advantage ratio > 1` 有利于 HYLE：

| workload | 四个 median advantage ratios | 98.9% CI envelope | exact | 5% NI |
|---|---|---|---|---|
| split SwiGLU f32 / 11008 | 0.9967 / 1.0067 / 1.0022 / 0.9922 | [0.9681, 1.0341] | 4/4 `PARITY` | 4/4 `PASS` |
| weighted RMSNorm f32 / 4096 | 1.0056 / 0.9849 / 0.9988 / 0.9884 | [0.9573, 1.0305] | 4/4 `PARITY` | 4/4 `PASS` |
| expert top-k f32 / 256/6 | 1.0025 / 1.0022 / 1.0104 / 1.0005 | [0.9691, 1.0342] | 4/4 `PARITY` | 4/4 `PASS` |

统计层曾给出三行 `PARITY/PASS`，A/A、5%/10% known-delta、timer-floor、placement 与执行顺序控制也
全部 `PASS`；语义红队已经撤销该聚合 verdict。点值仍可说明没有明显自然路线异常，但不能据此签
same-contract、改 route、API、layout 或常态 metadata。

被撤销的 bundle 位于
`C:\public\project-hyle\_validation\simd-v6-provider-consumer-replay\perf-evidence-v2`。contract-lock、
manifest、summary SHA-256 分别为
`660BBE3B35850599E1C878A3F70841D858859D36E177B9F1112438AA8B08CBC8`、
`9A909FBB4405FE900CF88BEE1216095EB298ED624A8085159272FD572912A4D3`、
`807C99BEA45CACF6ED17798128A183D9EC08655233B76430FC044C7291250438`；deep verifier 为
`QUALIFIED/CURRENT/EXACT`，contract SHA-256 为
`d5745aff4a0f7183310a1d4671cd05a472aef5f279cece78046feaf406a1c216`。

## SIMD v3 历史路由基线

以下内容绑定 v3 候选和当时的 provider lock，用于说明 387-operation/422-scope 路由为何存在；
它不覆盖 v6 新增 row 或改写旧 evidence identity。

### 签署范围

SIMD v3 的目标不是维护第三套完整寄存器库，而是在自然的 whole-buffer 与 source-kernel 合同下：

- 用锁定 Highway 承载同合同标准 primitive；
- 只在有限 `type × target × compiler/platform` 作用域保留有 material 证据的 HYLE 实现；
- 对 `tanh_f32_batch` 与 `gelu_tanh_f32_batch` 使用已证明更快的 xsimd body；
- ordinary fat library 每个整批调用只做一次缓存后的 ISA target 调用；fixed/exact source TU 的 provider
  路由必须在编译期消失；
- 所有 O(n) 批量操作保持 O(n) 时间、O(1) 辅助状态，除显式 scratch/allocating convenience 外没有隐藏
  allocation、copy、lock 或 syscall。

冻结 inventory 为 387 个旧 operation、1,797 个定义位置。终态账本现有 422 个 route scope：

| 决策 | scope 数 | 含义 |
|---|---:|---|
| `ROUTE_HIGHWAY` | 366 | 同合同默认、相同 codegen、持平、反转或有限实验后不可分 |
| `KEEP_HYLE` | 34 | 仅在机器可表达的有限赢家或 Highway 缺少同合同时保留 |
| `REMOVE_REDUNDANT` | 22 | alias/selector/转发 spelling 已由现有路线完整承载 |
| `PENDING` | 0 | 不允许隐含 fallback |

xsimd 的两个 whole-buffer family 例外记录在 `scripts/simd-family-routing.json`，不冒充 387-operation
旧寄存器账本的一部分。机械复验使用：

```text
python scripts/simd-op-coverage.py --check-provider-manifest --check-ledger \
  --require-route-decisions --check-provider-routes
```

### P0：复杂度与零 provider 路由成本

`scripts/simd-provider-codegen-audit.py` 对 direct provider body 与通过 HYLE route 类型实例化的 body 做
最终 object 比较。MSVC 19.50、GCC 14.2 与 AppleClang 21 均通过；当前覆盖的 15 个 provider/kernel
组合没有额外有效 instruction、call、branch、stack reference、copy 或 spill。GCC objdump 的显式
alignment NOP 只在收据中计数后忽略；所有有效指令仍要求逐项相同。

主要收据：

- [`MSVC provider route codegen`](../../../../evidence/performance/simd-14-provider-route-codegen-w1-win-msvc-avx2.json)
- [`MSVC algorithm/provider codegen`](../../../../evidence/performance/simd-15-algorithm-codegen-w1-win-msvc-avx2.json)

ordinary fat dispatch 不被伪称为零成本：未知 CPU 的公开入口允许每个 whole-buffer 一次缓存函数指针调用。
它不在 element loop 内选择 provider，也不会再调用第二套 Highway dynamic dispatch。fixed/exact
source TU 则直接绑定唯一 symbol。

### P1：W1-WIN 普通 row/dtype 路线

当前有限复测使用 W1-WIN、MSVC 19.50、Release/AVX2、`closure-perf-v1`、5% material margin；规模为
4,096 与 1,048,576。每侧四个 code-placement replica，正反序各两次，CPU 2 P-core、高优先级；A/A、
layout、输入唯一性、pin 与 priority 资格均通过。

报告给出 3 个 `TIE`、4 个 `INCONCLUSIVE`、0 个 `LOSE`。`INCONCLUSIVE` 表示仪器不把该行提升为
“相等/更好”，不是把未知写成平局；但六个产品行的 95% CI 下界全部高于 0.95，因此冻结的 5%
non-inferiority 命题仍成立：

| workload | candidate/Highway | 95% CI | 5% NI |
|---|---:|---:|---|
| row add f32, 4K | 1.014 | [1.005, 1.024] | PASS |
| row add f32, 1M | 1.006 | [0.997, 1.034] | PASS |
| f16 → f32, 4K | 1.008 | [0.992, 1.019] | PASS |
| f16 → f32, 1M | 1.010 | [0.995, 1.022] | PASS |
| mixed bf16 dot, 4K | 1.007 | [0.997, 1.021] | PASS |
| mixed bf16 dot, 1M | 1.010 | [0.995, 1.022] | PASS |

机器收据和原始日志见
[`simd-17-w1-win-p1.json`](../../../../evidence/performance/simd-17-w1-win-p1.json) 与相邻的
`simd-17-w1-win-row-*.txt`。这只签 W1-WIN 的冻结 source-primary normal workload，不外推为跨平台
逐 operation SOTA。

### P1：xsimd 激活 family 例外

同一资格化仪器比较锁定 Highway 与锁定 xsimd 的完整 f32 whole-buffer body。四行都超过 5% 门，且方向
一致；下表小于 1 表示 Highway 比 xsimd 慢：

| workload | Highway/xsimd | 95% CI | 产品路线 |
|---|---:|---:|---|
| tanh, 4K | 0.702 | [0.648, 0.775] | xsimd |
| tanh, 1M | 0.714 | [0.664, 0.798] | xsimd |
| tanh-GELU, 4K | 0.751 | [0.694, 0.835] | xsimd |
| tanh-GELU, 1M | 0.759 | [0.714, 0.850] | xsimd |

因此 reporter 的四个 `LOSE` 是 Highway candidate 输给 peer；shipping route 正是 xsimd，所以产品
verdict 为 `PASS/ROUTE_XSIMD`。accuracy preflight 同时验证 tail、in-place、NaN/Inf/signed-zero；旧轮
观察为 xsimd 最大 1 ULP、Highway 最大 2 ULP。当前复测报告与原始日志同样归档在上述 SIMD-17 收据旁。

### 保留算法与平台例外

这些结论不要求把所有 HYLE 实现删除：

- VQSort adaptation 保留在 `src/simd/sort/vqsort_hyle/`。同合同 W1-WIN f32 workload 中，partial
  4K/k10 为 `1.147× [1.108, 1.185]`，full 4K 为 `5.950× [5.883, 6.052]`，full 1M 为
  `7.463× [7.374, 7.556]`；资源合同为 in-place/no-hidden-allocation。完整边界见
  [`SIMD-11 ordering evidence`](simd-11-ordering.md)。
- `Stream` 在 Windows AVX2 最终 codegen 相同且计时反转，故默认 Highway；AppleClang 21/NEON 上，
  Highway 的逐向量 store-prefetch 使大工作集明显变慢，只有该精确 scope 保留 HYLE plain-store。
  收据见
  [`simd-15-stream-workload.json`](../../../../evidence/performance/simd-15-stream-workload.json)。
- 其余 HYLE scope 只按 `scripts/simd-op-routing.json` 中的精确 selector 和 evidence 生效。不得从一个 ISA、
  type 或 compiler 外推到全局。

### Correctness、sanitizer 与交付

最终 SIMD suite 在 Windows/MSVC、WSL/GCC 与 macOS/AppleClang 均为 5/5 PASS，覆盖 ordinary API、route
static assertions、有限 winner、provider codegen 和实际 ordering target。WSL GCC 14.2 的 ASan 与
UBSan 也分别为 5/5 PASS。

ASan 曾发现 VQSort adaptation 的真实 debug-shape 缺陷：Highway 在 ASan 下启用 debug 网络并移除
256-key slot，而移植层错误地从 `NDEBUG` 选择 release 阈值，导致空函数调用。修复改用 Highway 的
`HWY_IS_DEBUG_BUILD` 作为同一 authority；release 热路径不变。finding 与五组回归见
[`simd-17-vqsort-sanitizer-fix.json`](../../../../evidence/simd-17-vqsort-sanitizer-fix.json)。

source embedding、搬迁后的 installed static 与 installed shared consumer 已在三平台通过。v3 候选
当时只安装 ordinary HYLE 头与库；后续 v4 已把 `target.h`、`fat_kernel.h`、`hyle::simd_expert` 与锁定
Highway/xsimd 开发头扩展到 static/shared SDK，同时继续禁止安装内部 `provider/*` route。这里保留
v3 历史事实，不再把旧 source-only 交付形状冒充当前产品。

### 结论边界

本页允许的结论是：冻结的 v3 作用域内，路由复杂度与中间层成本结构正确；W1-WIN 的普通 row/dtype
source-primary workload 通过 5% 工程非劣门；两个激活 family 的 xsimd 例外得到 material 复现；已保留
VQSort 与 Apple NEON Stream 例外有具名证据。

以下结论仍被明确禁止：整个 SIMD 在所有操作/类型/CPU/编译器上绝对最快；SVE/RVV 真机性能已认证；
static/shared 与 source 在任意工具链 ABI 兼容；Formal release SOTA 已签署。新硬件、vendor 升级或真实
推理 consumer 暴露 material gap 时，只重签受影响 route group。
