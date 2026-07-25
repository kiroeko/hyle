# core 性能证书

> 本页是 core 性能结论的唯一信源。设计与使用分别见
> [`core-design`](../architecture/core-design.md) 和 [`guide/core`](../guide/core.md)。
> 实现证据绑定 `71dabf7`；本页记录的是该实现的最终 API，而不是中途为了定位差异保留的
> 实验形状。

## 测什么、不测什么

唯一有独立算法热路的公开能力是 `IntrusiveList`。`bench/core/core_bench.cc` 使用 4096 个
epoch 外预分配、stable-address 的 member-hook 节点，对拍 Boost.Intrusive `auto_unlink` +
`constant_time_size<false>`；它与 HYLE 具有相同 ownership、生命周期和常态元数据。`std::list`
只作 owning/allocating floor，不能用来宣称同形 SOTA。

三个正常场景分别是 FIFO batch、乱序取消、直接丢弃未 drain 的 batch。每个样本为独立进程，
用 `scripts/bench-stat.py -n 15` 的 paired sign test 判 win/loss；单位 ns/node，越低越好。

为排除同进程先后顺序影响，最终代码至少运行一次 `HYLE_BENCH_RIVAL_FIRST=1`；Apple M4
同时归档默认顺序。`bench-stat.py` 的显著阈值为 `p < 0.01`，未达到就记 TIE，不能把中位数
较小写成胜出。

## Windows x64 / MSVC 19.50 / Release（rival-first，2026-07-25）

| 场景 | HYLE | Boost 同形 | std floor | paired 结论 |
|---|---:|---:|---:|---|
| push 4096 + pop 全部 | **2.164** | 2.296 | 23.880 | TIE；HYLE 9/15 较快，p=.61 |
| 乱序 O(1) remove | **2.206** | 2.355 | — | TIE；HYLE 12/15 较快，p=.035 |
| 非空 list 析构自动摘链 | 2.005 | **1.976** | — | TIE；HYLE 6/15 较快，p=.61 |

绝对值 spread 较大，所以点值不承重。三个正常场景对同合同 Boost 都没有显著落后；结果说明
auto-unlink/list cleanup 没把真实生命周期合同做成慢路径，也没有靠 size/owner 常态元数据换分。

## WSL2 x64 / GCC 14.2 / Release（rival-first，2026-07-25）

| 场景 | HYLE | Boost 同形 | std floor | paired 结论 |
|---|---:|---:|---:|---|
| push 4096 + pop 全部 | 2.068 | **1.951** | 11.643 | TIE；HYLE 3/15 较快，p=.035 |
| 乱序 O(1) remove | **2.097** | 2.124 | — | TIE；HYLE 8/15 较快，p=1.0 |
| 非空 list 析构自动摘链 | **1.990** | 2.000 | — | TIE；HYLE 7/15 较快，p=1.0 |

WSL 使用 Linux ext4 上的独立 Release build 与 `/root/vendor` pinned benchmark rivals；源码从
Windows 主工作树同步。配置 core benchmark 仍触发全项目可选后端/第三方探测，这不是 core
运行时成本，而是下一轮 build/package-system 审查必须修正的组装边界问题。

## Apple M4 / AppleClang 21 / Release（2026-07-25）

| 顺序 | 场景 | HYLE | Boost 同形 | std floor | paired 结论 |
|---|---|---:|---:|---:|---|
| 默认 | push 4096 + pop 全部 | **1.352** | 2.464 | 15.192 | HYLE 赢；15/15，p=6.1e-5 |
| 默认 | 乱序 O(1) remove | **1.028** | 1.029 | — | TIE；HYLE 5/15 较快，p=.30 |
| 默认 | 非空 list 析构自动摘链 | 0.929 | **0.924** | — | TIE；HYLE 7/15 较快，p=1.0 |
| rival-first | push 4096 + pop 全部 | **1.370** | 2.464 | 15.768 | HYLE 赢；15/15，p=6.1e-5 |
| rival-first | 乱序 O(1) remove | 1.023 | **1.018** | — | TIE；HYLE 7/15 较快，p=1.0 |
| rival-first | 非空 list 析构自动摘链 | **0.918** | 0.939 | — | TIE；HYLE 12/15 较快，p=.035 |

双顺序都得到同一产品结论：FIFO 明确领先，乱序删除与析构摘链同档。最终三平台没有任何
同合同场景被判 LOSE；这就是当前 core 热路的 SOTA 证据边界，不外推到未测 workload。

## 其他 core 能力

- `kVersion*`、缓存常量和 C inline list 都是编译期/header-inline，不存在 DLL 热路。
- `version()` 是 plugin/诊断冷路；只返回常量。它不值得用纳秒 benchmark 塑造 API。
- `prefetch_read` 与 cache spacing 的价值只能在真实消费者（SwissTable、并发队列、SharedPool、
  scheduler）中测；单独测一条 hint 或 `alignas` 没有产品含义，结果归各消费者性能页。
- C intrusive list 与 C++ 使用相同 circular-sentinel 指针算法；不为同一机器码制造第二张表。

## 复跑

```bash
cmake --preset bench -DHYLE_BENCH_BOOST=ON
cmake --build build-bench --target hyle_core_bench -j2
python scripts/bench-stat.py build-bench/bin/hyle_core_bench -n 15 \
  --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink" \
  --pair "HYLE IntrusiveList::remove" "boost::intrusive::erase" \
  --pair "HYLE IntrusiveList auto detach" "boost::intrusive auto_unlink detach"

# 再交换同进程运行顺序；PowerShell 用 $env:HYLE_BENCH_RIVAL_FIRST='1'
HYLE_BENCH_RIVAL_FIRST=1 python scripts/bench-stat.py \
  build-bench/bin/hyle_core_bench -n 15 \
  --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink" \
  --pair "HYLE IntrusiveList::remove" "boost::intrusive::erase" \
  --pair "HYLE IntrusiveList auto detach" "boost::intrusive auto_unlink detach"
```
