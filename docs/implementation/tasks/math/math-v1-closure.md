# math-v1 边界重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `math` 的当前 closure card。旧
> [`math-01-scalar.md`](math-01-scalar.md) 只记录 clay 迁移史，不证明当前四个公开函数的需求、
> 边界或正确性。机械状态只看
> [`module-status.json`](../../module-status.json)。本轮先裁决 scalar math 与 SIMD 的边界；
> capability map 和有限 `FIX` 集进入 `AUDITED` 前不改产品代码。

## Closure card（`SCOPED`）

- component / owner / round ID：`math` / scalar numeric primitive owner / `math-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`8652aa53c46ee28ff002bf301906c5ab4a148765` /
  `1d95449146887a73931e21d1687e2500feb436e6`
- 进入原因与 demand/defect ID：
  - `MATH-BOUNDARY-01`：现有 target 是 clay math 迁移剩余，没有从用户任务正向定义 scalar
    numeric 与 `hyle::simd` 的分工；
  - `MATH-INT-01`：本地 CCCL 与推理 shape/tile 任务都证明非负整数 ceil-division / round-up
    是通用 primitive，但现有 `round_up_to_multiple` 在结果不可表示时无提示回绕；
  - `MATH-FLOAT-01`：`sgn` 与 `approx_eq` 没有产品消费者，NaN/无穷/近零语义与
    Eigen 等成熟形状不能等价，不能因为已有测试就保留。
- 真实 consumer task / normal workload：
  - 当前 HYLE 产品源对 `hyle::math` 的语法消费为 0；这是需求分类事实，不是自动删除理由；
  - 北极星推理任务使用 element count / lane width / block size 计算 chunk 数与可表示的
    padded extent；工程留出是二维 tiling / chunk partition；
  - 单值标准数学函数直接路由 C++23，lane/vector 与 whole-buffer 数值计算路由
    `hyle::simd`。
- 范围：`include/hyle/math/**`、`src/math/**`、`tests/math/**`、math component/package journey、
  scalar-vs-SIMD 边界与 architecture/user 文档。
- 非目标：SIMD lane/backend/batch 实现、线性代数、FFT、几何、随机、特殊函数库、模型算子、
  C++23 标准库镜像、Native FFI/C API、稳定 C++ ABI 与 Formal 证书。
- 直接依赖 / 直接消费者：当前产品依赖为空，产品消费者为空；测试是唯一语法消费者。
  本轮只对边界做决定，不倒逼 SIMD 依赖 math，也不为保留 target 制造假边。
- tier 与理由：`P0`。当前函数是 header-only `constexpr` scalar primitive，未被 profile 证明为
  独立热点；本轮守复杂度、可表示性与 release codegen，不跑纳秒胜负。
- material margin / resource budget：P1 margin 不适用。保留的 primitive 必须零分配、零常态
  metadata/lock/syscall；ceil division 不得比同语义成熟形状多于一次整数除法。
- 总调查预算 / 性能预算：90 分钟 / 15 分钟 P0 codegen。若删除整个组件的交付半径超出预算，
  以具名 build/package blocker 终止，不扩大到 SIMD 本体。
- 最多复现、机制诊断、产品修改次数：一次边界/溢出复现，一次 release codegen 检查，
  一个有限产品修改批次及一次重跑。
- machine profile：`W1-WIN`、`W1-WSL`、`M1-MAC`。
- instrument profile：`closure-correctness-v1`。
- strong peers（每族最多两个）：
  - 标准可替代面：C++23 `<bit>/<cmath>/<numbers>/<limits>` 和 HYLE 当前所支持标准；
  - integer ceil/round：CCCL `132670eba92ca5bbddb2d479e76045184ed885a5` 的
    `cuda::ceil_div` / `cuda::round_up`；
  - fuzzy floating comparison：Eigen `26f009dbbe1f8520439e6b1b6c564d897e80ace6` 的
    scalar `isApprox` / `isMuchSmallerThan`；只用于语义对照，不借用 Eigen 为产品依赖。
- 无同合同 peer 时的预算及禁止声明：只做复杂度、分配与 codegen 结构门；不声称 scalar
  math 或 SIMD 全面 SOTA，不为无消费者的浮点近似语义制造 benchmark。
- 适用 gates：public-header self-contained；normal/zero/max/divisor/overflow 与保留浮点的
  NaN/infinity/signed-zero 语义；`constexpr` oracle/property；MSVC/GCC/AppleClang；source 与 installed
  static/shared SDK 中真实的 INTERFACE target journey；P0 release codegen；用户选型文档；独立红队。
- 预期输出：capability-family map；逐 symbol public trace；保留/路由/删除裁决；有限 `FIX`
  集；candidate 与三平台/交付/P0 证据；architecture/user 指南；红队、LOC 与 terminal receipt。
- 停止条件 / 可接受限制：每个公开符号有准入来源、完整语义或明确路由；不存在
  `PARTIAL + KEEP`；SIMD/math 选型不再要求用户猜测。无产品消费者、header-only 无独立
  binary 及不提供 Native FFI 都可接受，但不得伪造 journey 或证书。

## Capability-family coverage map

`SCOPED`：下一阶段从用户任务与上述 strong peers 自顶向下建图，再将现有四个公开符号
逐一反向映射。本节尚未给出 `KEEP/FIX` 授权。

## Capability / gap ledger

`SCOPED`：未冻结。已复现的 overflow 与 exceptional-value 问题是审计输入，只有 ledger 完成后
才能决定是修复、重塑、路由还是删除。

## Candidate

尚未形成。

## Verification manifest

只使用 closure card 预声明的 gates；未运行项不记 `PASS`。

## 独立红队

在 candidate 冻结后执行，30 分钟上限。

## Closure decision

尚未裁决。
