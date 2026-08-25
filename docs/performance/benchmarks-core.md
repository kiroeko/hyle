# core 性能证据

> **当前状态（2026-08-25）：黄色。**本页旧点值只作历史输入；当前 core 正在按
> [`benchmark-evidence-standard`](./benchmark-evidence-standard.md) 重新取得资格。candidate、
> 三平台原始日志、最终 verdict 和证书 SHA 尚未冻结，因而本页目前不签任何新的
> `WIN/EQUIVALENT/LOSE`，也不据旧 `TIE` 宣称同档。模块状态只看
> [`pre-release-readiness`](../milestones/pre-release-readiness.md)。

> 本页是 core 性能结论的唯一模块信源。设计与使用分别见
> [`core-design`](../architecture/core-design.md) 和 [`guide/core`](../guide/core.md)；施工与剩余门见
> [`core-module-closure`](../milestones/core-module-closure.md)。

## 当前证据身份

| 项 | 当前事实 |
|---|---|
| 产品/API 变更 | `6b21140`、`87587dd` 是本轮 compiler/list 施工点；cache layout 又在当前工作树重裁决 |
| benchmark harness | `217c8be` 是五类 workload 的当前起点，不是最终证书 identity |
| 强 peer | Boost.Intrusive `auto_unlink + constant_time_size<false>`；最终日志必须记录 exact clean tree |
| runner | 当前五态 runner；最终日志必须记录 runner hash、可执行文件 hash、fresh build log 与预声明 margin |
| candidate | **尚未冻结**；最终由本轮 milestone 填写 commit/tree |

旧 IntrusiveList 数字绑定 `71dabf7`。2026-07-28 后 compiler/cache primitive 又有变化，2026-08
依赖、harness、C/C++ surface 和 cache layout 继续变化，所以旧表不能借“链表算法看起来没改”
自动成为当前证书。

## 当前 workload 资格

`bench/core/core_bench.cc` 使用 epoch 外预分配、stable-address 的 member-hook 节点。Boost peer
具有相同的 non-owning、auto-unlink 和无常量时间 size 合同；`std::list` 只在 FIFO 行作为
owning/allocating 成本下界，绝不是同合同 SOTA peer。

当前 harness 有五类操作家族；四类各测 N=32/4096，teardown 只测 N=4096。其中短
aggregate 可以是 placement 机制诊断，不自动成为产品声明：

| workload | 计费边界 | 证据层 | 当前资格/结论 |
|---|---|---|---|
| FIFO batch | refill + 全部 pop，ns/node | 产品候选 | N=32 仍需等价 code-shape/placement 扰动；N=4096 待 exact candidate 三平台重跑 |
| refill + arbitrary cancel | refill + 按 permutation 全部取消，ns/node | N=4096 产品候选；N=32 机制诊断 | N=32 已确认 `INCONCLUSIVE/placement-sensitive`，不单独阻塞；小批产品风险由 time 的真实 arm/cancel 消费者承重；N=4096 待重跑 |
| read-only traversal | 预建链表后求和，ns/node | 产品候选 | N=32 需 placement 扰动；N=4096 待重跑 |
| whole-batch handoff | 两次 whole-list splice，ns/transfer | **机制诊断** | N 不增加计时操作量，不能独立签产品 SOTA；由 time/scheduler 的真实 batch handoff 承重 |
| non-owning list teardown | refill + list 析构摘链，ns/node | 生命周期产品候选 | 不能把合并点值写成“纯析构成本”；待 exact candidate 三平台重跑 |

这些分类先于数值。只有同语义、ownership、检查、计费边界、layout、compiler/link shape 的
产品候选，完成默认序、peer-first、适用的 placement 扰动和三平台合成后，才取得五态 verdict。
whole-batch microbench 即使数字稳定，也只能解释 O(1) 重链机制。

### N=32 cancel 为什么已经无胜负资格

clean `cdc34e6` 的旧 harness 让 MSVC 看见 compile-time `Count=32`，编译器完全展开 refill，结果
表现为 Boost 小幅领先。把两边同时改成语义等价的 runtime bound 后，方向反转为 HYLE 领先；
该改动归档在 `217c8be`。后一个方向不是“修正后 HYLE WIN”，前一个方向也不是 Boost WIN：
合同等价的代码形状能翻转结果，按当前标准只能记
`INCONCLUSIVE/placement-sensitive`。禁止挑 runtime/compile-time 中有利的一版、改变产品字段，
或用旧 `TIE` 掩盖不确定性。这一 N=32 microbench 因此降为非阻塞机制诊断：它不签产品
胜负，也不被删掉；小批 cancel 的产品门由 time 的真实 arm/cancel/expire 形状覆盖。

`HYLE_BENCH_RIVAL_FIRST` 只交换同一可执行文件内的执行顺序，不交换 lambda 定义、heap allocation
或函数 placement；它是必要门，但不能替代等价 code-shape 扰动。FIFO N=32 仍含 compile-time
bound，同样不得在完成该门前签胜负。

## 产品消费者守门

- IntrusiveList 的真实直接消费者是 `time::TimerQueue`、`CommonTimeoutQueue` 及 scheduler 的
  ready-list drain。arm/cancel/expire、prefix move、whole-batch handoff 和 consume-all 的正常
  workload 才能证明 core primitive 没给产品路径造成回归。
- cache layout/prefetch 必须在实际消费者中承重。队列、map/interner、SharedPool、scheduler 与
  SIMD/vqsort 分别验证 footprint、false sharing、访存和 codegen；单测一条 hint 或 `alignas`
  没有产品意义。
- 当前 target-family cache layout 仍在复证中；历史 128B consumer 点值不能自动代表当前 x86
  64B dense layout 或 Apple 128B layout。

## 2026-07 历史归档（旧统计规则）

下表仅保存回归来路。表中的 `TIE` 意味着当时没有按旧 sign-test 门签出 WIN/LOSE，**不等于**
当前五态中的 `EQUIVALENT`；原始 paired ratios、当前 harness 和 placement 门不全时只能作历史输入。

### Windows x64 / MSVC 19.50 / Release（rival-first，2026-07-25）

| 场景 | HYLE | Boost 同形 | std floor | 当时记录 |
|---|---:|---:|---:|---|
| push 4096 + pop 全部 | 2.164 | 2.296 | 23.880 | 旧 TIE；HYLE 9/15 较快，p=.61 |
| refill + 乱序 remove | 2.206 | 2.355 | — | 旧 TIE；HYLE 12/15 较快，p=.035 |
| refill + 非空 list 析构 | 2.005 | 1.976 | — | 旧 TIE；HYLE 6/15 较快，p=.61 |

### WSL2 x64 / GCC 14.2 / Release（rival-first，2026-07-25）

| 场景 | HYLE | Boost 同形 | std floor | 当时记录 |
|---|---:|---:|---:|---|
| push 4096 + pop 全部 | 2.068 | 1.951 | 11.643 | 旧 TIE；HYLE 3/15 较快，p=.035 |
| refill + 乱序 remove | 2.097 | 2.124 | — | 旧 TIE；HYLE 8/15 较快，p=1.0 |
| refill + 非空 list 析构 | 1.990 | 2.000 | — | 旧 TIE；HYLE 7/15 较快，p=1.0 |

WSL 当时使用 Linux ext4 上的独立 Release build 与 pinned benchmark rivals；这些运行早于当前
runner/harness，不能只靠重新解释旧 p 值恢复资格。

### Apple M4 / AppleClang 21 / Release（2026-07-25）

| 顺序 | 场景 | HYLE | Boost 同形 | std floor | 当时记录 |
|---|---|---:|---:|---:|---|
| 默认 | push 4096 + pop 全部 | 1.352 | 2.464 | 15.192 | 旧规则 WIN；15/15，p=6.1e-5 |
| 默认 | refill + 乱序 remove | 1.028 | 1.029 | — | 旧 TIE；HYLE 5/15 较快，p=.30 |
| 默认 | refill + 非空 list 析构 | 0.929 | 0.924 | — | 旧 TIE；HYLE 7/15 较快，p=1.0 |
| rival-first | push 4096 + pop 全部 | 1.370 | 2.464 | 15.768 | 旧规则 WIN；15/15，p=6.1e-5 |
| rival-first | refill + 乱序 remove | 1.023 | 1.018 | — | 旧 TIE；HYLE 7/15 较快，p=1.0 |
| rival-first | refill + 非空 list 析构 | 0.918 | 0.939 | — | 旧 TIE；HYLE 12/15 较快，p=.035 |

这三张表只能说明旧实现/旧工具下的观测方向；没有预声明 equivalence margin、当前 source
identity、五类 workload 和 placement 证据，不能写成“当前三平台无 LOSE”或“当前 SOTA”。

## compiler/prefetch 历史机制证据

2026-07/08 的 object 检查曾确认：MSVC x64 的 C/C++ local/NTA/write 映射为
`prefetcht0/prefetchnta/prefetcht0`（无 ET0 时 write 退为读 hint）；GCC14/Clang20 baseline x86
不越过 target 强发 PREFETCHW，`-march=znver3` 才选择 `prefetchw`；AArch64 capable targets 生成
对应 `prfm`。pure-C compiler-contract loop 当时也保持 TU-local、无 HYLE helper/DLL hop。

这些是机制/codegen 历史输入，不是产品 SOTA 表。当前 candidate 必须重新绑定 MSVC/GCC/
AppleClang、Debug/Release/sanitizer 退化和真实 SIMD/vqsort consumer；结果写入本轮 milestone，
产品点值仍归相应消费模块。

## 正式复跑骨架

先在合同卡中预声明每个 claim-bearing workload 的 equivalence margin，再 fresh configure/build。
下面的 `<m>` 不能在看到结果后填写：

```bash
cmake --preset bench -DHYLE_BENCH_BOOST=ON \
  -DHYLE_BOOST_DIR=/path/to/clean/boost
cmake --build --preset bench --target hyle_core_bench -j2
python scripts/bench-stat.py build-bench/bin/hyle_core_bench -n 15 \
  --require-clean --peer boost=/path/to/clean/boost \
  --equivalence-margin-pct <m> \
  --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink" \
  --pair "HYLE IntrusiveList::remove" "boost::intrusive::erase" \
  --pair "HYLE IntrusiveList iteration" "boost::intrusive iteration" \
  --pair "HYLE IntrusiveList::append" "boost::intrusive whole-list splice" \
  --pair "HYLE IntrusiveList auto detach" "boost::intrusive auto_unlink detach"

# 再在独立 n=15 采样中交换执行顺序；PowerShell 使用环境变量语法。
HYLE_BENCH_RIVAL_FIRST=1 python scripts/bench-stat.py \
  build-bench/bin/hyle_core_bench -n 15 \
  --require-clean --peer boost=/path/to/clean/boost \
  --equivalence-margin-pct <m> \
  --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink" \
  --pair "HYLE IntrusiveList::remove" "boost::intrusive::erase" \
  --pair "HYLE IntrusiveList iteration" "boost::intrusive iteration" \
  --pair "HYLE IntrusiveList::append" "boost::intrusive whole-list splice" \
  --pair "HYLE IntrusiveList auto detach" "boost::intrusive auto_unlink detach"
```

正式证书还必须附 fresh build log，并对 N=32/短 aggregate 执行预先登记的等价 placement 变体；
runner 的单顺序输出只是证据输入，不自动成为模块 verdict。
