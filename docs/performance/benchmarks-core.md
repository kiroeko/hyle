# core 性能证据

> **当前状态（2026-08-25）：绿色 / `SIGNED`。**core 的同合同 N=4096 正常路径在 Windows、
> Linux 和 Apple M4 两档 QoS 均通过预声明 3% 的 SOTA non-inferiority gate；精确五态仍逐格
> 保留 `WIN/EQUIVALENT/INCONCLUSIVE`，没有把“排除实质落后”冒充“已经证明平局”。模块状态
> 只看 [`pre-release-readiness`](../milestones/pre-release-readiness.md)。

> 本页是 core 性能结论的唯一模块信源。设计与使用分别见
> [`core-design`](../architecture/core-design.md) 和 [`guide/core`](../guide/core.md)；施工与剩余门见
> [`core-module-closure`](../milestones/core-module-closure.md)。

## 2026-08-25 当前证书身份

| 项 | 绑定事实 |
|---|---|
| product candidate | `a951f0a436336366df9729e663c2fccfd357cff2` / tree `714d660299bb67d2c855452e8cbdb14cd89ec841` |
| formal harness source | `2707583f2fdd0d38f77f3302480be662dfade2f6` / tree `ac85153f7d3241472ace9fc8556afdb88311082a` |
| product-diff gate | a951→270 对 `CMakeLists.txt/cmake/include/src/samples/package` 零差异 |
| strong peer | Boost.Intrusive clean `df8eeb90846648464e3c8e378e312f0cec772e83` |
| nanobench | v4.6.0 `b709728dba4df0356d467abfb0ba20781a8031c0`，header SHA-256 `2cfcdd57044fd0ee11d1be2b5dc545781b05c36049613c0089a73b332afd8c4d` |
| runner | SHA-256 `22b285cb469479522f719158e2b9c1757aefb604d4a781e801ebbd9d76dca26d` |
| ratio / margin | Boost op/s ÷ HYLE op/s；采样前固定 3%，band `[0.971,1.030]` |
| raw contract | 每个 process/每对手保留 baseline/rival op/s 与 ratio 的 schema-1 JSON；default 与 peer-first 独立采样 |

当前 core 运行时行为基线为 `9459d3f`，最终 source/public-contract/验证 harness 为 `c0b9a84`。
性能原始证据仍按上表绑定 `a951f0a`/`2707583`，因为 `a951f0a`→`9459d3f` 没有改变
IntrusiveList、prefetch 或 compiler contract 的 claim-bearing 热实现；`9459d3f`→`c0b9a84`
也只改变合同文字、test/evidence/probe/sample 分类。新增的 runtime identity 是 O(1) 冷启动路径。
因此这是对同一热代码的证书复用，不是没有重跑却声称得到一组新性能数据。

`2707583` 修正了正式 FIFO 的最后一个合同不对称：两边现在都取回并消费 popped node payload。
旧 Boost 路径只 `pop_front()`，GCC 可把它折成不恢复 payload 的六指令循环；修正后两边都承担
真实 FIFO consumer 的节点取回工作，旧 WSL 5--8% apparent LOSE 消失。该修正只改 harness，
没有按结果修改产品类型或元数据。

## 正式 claim 与结果

正式产品 claim 只覆盖 N=4096 的 FIFO、arbitrary remove、read-only traversal 和 auto-detach。
N=32 在采样前已分类为 code-shape/placement 机制诊断；whole-list append 每次永远只做两次 O(1)
重链，无法让 N 增加计费操作，固定为 `NOT_COMPARABLE/N/A`。以下 `PASS` 只表示相同 3% margin 下
已排除 material loss；exact `INCONCLUSIVE` 没有被改名。

| 平台 / QoS | FIFO | arbitrary remove | traversal | auto-detach | composite SOTA gate |
|---|---|---|---|---|---|
| Windows x64 / MSVC 19.50 | `INCONCLUSIVE` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 `PASS` |
| Linux x64 / GCC 14 | `INCONCLUSIVE` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 `PASS` |
| Apple M4 / interactive | `WIN` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 `PASS` |
| Apple M4 / utility | `WIN` | `EQUIVALENT` | `EQUIVALENT` | `EQUIVALENT` | 4/4 `PASS` |

Windows default n=31 的 remove CI `[0.936,1.032]` 仍跨 non-inferiority 上界，按预登记升级规则
保留原日志并用同一 binary/environment/margin 跑完整 default n=63；最终 default/peer-first：

| Windows workload | default CI / exact / gate | peer-first CI / exact / gate |
|---|---|---|
| FIFO | `[0.959,0.992]` / `INCONCLUSIVE/PASS` | `[0.962,1.020]` / `INCONCLUSIVE/PASS` |
| remove | `[0.950,0.992]` / `INCONCLUSIVE/PASS` | `[0.924,0.995]` / `INCONCLUSIVE/PASS` |
| traversal | `[0.995,1.008]` / `EQUIVALENT/PASS` | `[0.990,1.009]` / `EQUIVALENT/PASS` |
| auto-detach | `[1.008,1.026]` / `EQUIVALENT/PASS` | `[0.992,1.022]` / `EQUIVALENT/PASS` |

Linux default/peer-first 均为 n=31：FIFO `[0.955,0.998]` / `[0.958,0.995]`，remove
`[0.939,1.004]` / `[0.956,1.011]`，traversal `[0.994,1.015]` / `[0.987,1.014]`，
auto-detach `[0.997,1.023]` / `[1.000,1.029]`；四项两序 gate 全 `PASS`。

M4 每格 n=31，interactive/utility 都显式设置 QoS 且 `qos_ok=1`（macOS 不支持硬 P/E affinity）：

| M4 workload | interactive default / peer-first | utility default / peer-first |
|---|---|---|
| FIFO | `.399 [.394,.402] WIN` / `.395 [.391,.397] WIN` | `.393 [.390,.395] WIN` / `.391 [.389,.393] WIN` |
| remove | `.998 [.980,1.013] EQ` / `1.001 [.969,1.016] INC` | `1.006 [.999,1.011] EQ` / `1.000 [.994,1.004] EQ` |
| traversal | `1.002 [.999,1.003] EQ` / `1.000 [.999,1.002] EQ` | `1.000 [.997,1.009] EQ` / `1.000 [.996,1.004] EQ` |
| auto-detach | `1.003 [.987,1.017] EQ` / `.988 [.976,1.003] EQ` | `1.005 [.998,1.012] EQ` / `.995 [.991,1.002] EQ` |

四组 M4 的每个 claim-bearing gate 都为 `PASS`。FIFO 在这份同合同 workload 中观测为约 2.5×，
但证书不把该现象解释成未经隔离验证的单一原因，也不外推到其它平台或其它 list 操作。

## 原始证据与消费者归因

- Windows：default/peer-first n=31 各 279/279 raw，升级 default n=63 为 567/567；exe SHA-256
  `5e64fae17e0d60906ee1a545c731bb7e670ad5c29b923eebf74a41d95891c085`。三份结果日志 SHA-256
  分别为 `974ca2eeddfbafa321bd05e534d6103df7db40ee9f38e6670cc5d24cfa5dfed6`、
  `207a74d73b99dee79ed7ec23c307440761e6cb6e0022b4f6871c7c6f7861fa23`、
  `a86062096b15387f2e85b46c9638b3e52500644bd0e804d8f457927c2e0a282c`。
- Linux：default/peer-first 各 279/279 raw，clean native-ext4 GCC 14；exe SHA-256
  `53e636e7acebc6b7cffe568a50a3e47fb5a6169c8ba6ad339a2bc45091709572`，日志 SHA-256
  `1a93bfd73898000cf8766e4e9548c1db18690c381760ac430d2643d4e2bc6461` /
  `096a00ee1e400f5573b9a55ac0392d4eda3ac419550531709629ad7d676aad4f`。
- macOS：interactive/utility × default/peer-first 各 279/279 raw；exe SHA-256
  `f6d423c5b5f779d911696890f985f5ca649d4488d42b815318cf0c03542fc03b`；
  `.scratch/evidence/core-2707583/macos/evidence-sha256.txt` 对 99 个归档文件校验为
  0 missing/0 mismatch。

time::TimerQueue 是 core list 的真实消费者，但 N=8 cancel 同时计入 time 自己的 exact `size()`、
empty-transition `next_deadline` 和 owner/link 合同；dependency-free Seastar-shaped peer 没有等强
语义。M4 新顺序诊断中 HYLE 稳定 2.636--2.637 ns/node，peer 随注册顺序从约 2.584--2.588
变为 2.399--2.405；runner 数值从 `EQUIVALENT/PASS` 变为 `LOSE/FAIL`。因此产品资格是
`NOT_COMPARABLE/N/A`，只登记为 time 的 semantic-cost/placement 债务；core 的同合同 primitive
结论由上面的 Boost workload 承担。N=1/N=64 cancel 与 N=1/8/64 expire 诊断没有发现 gross
consumer regression，all-expire 的 unstable 行不签 SOTA。

## 签署前证据身份（归档）

> 以下直到 2026-07 历史归档的“当前/待重跑”是本轮签署前现场，已由上面的正式证书取代。

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

## 当前复跑骨架

本证书已在采样前声明 3% margin 与 default n=31；若任一 claim-bearing gate 为 `UNRESOLVED`，
保留该日志并用同 binary/environment/margin 把对应完整顺序升级到 n=63。fresh configure/build：

```bash
cmake --preset bench -DHYLE_BENCH_BOOST=ON \
  -DHYLE_BOOST_DIR=/path/to/clean/boost
cmake --build --preset bench --target hyle_core_bench -j2
python scripts/bench-stat.py build-bench/bin/hyle_core_bench -n 31 \
  --require-clean --peer boost=/path/to/clean/boost \
  --equivalence-margin-pct 3 \
  --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink" \
  --pair "HYLE IntrusiveList::remove" "boost::intrusive::erase" \
  --pair "HYLE IntrusiveList iteration" "boost::intrusive iteration" \
  --pair "HYLE IntrusiveList::append" "boost::intrusive whole-list splice" \
  --pair "HYLE IntrusiveList auto detach" "boost::intrusive auto_unlink detach"

# 再在独立 n=31 采样中交换执行顺序；PowerShell 使用环境变量语法。
HYLE_BENCH_RIVAL_FIRST=1 python scripts/bench-stat.py \
  build-bench/bin/hyle_core_bench -n 31 \
  --require-clean --peer boost=/path/to/clean/boost \
  --equivalence-margin-pct 3 \
  --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink" \
  --pair "HYLE IntrusiveList::remove" "boost::intrusive::erase" \
  --pair "HYLE IntrusiveList iteration" "boost::intrusive iteration" \
  --pair "HYLE IntrusiveList::append" "boost::intrusive whole-list splice" \
  --pair "HYLE IntrusiveList auto detach" "boost::intrusive auto_unlink detach"
```

正式证书还必须附 fresh build log。N=32/短 aggregate 的等价 code-shape 已证明会改变资格，故
固定保留为 placement-sensitive 机制诊断，不从中选择有利变体；runner 的单顺序输出仍只是证据
输入，必须同 peer-first 和平台门合成。
