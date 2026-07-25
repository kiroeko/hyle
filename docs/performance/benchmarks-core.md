# core 性能证书

> 本页是 core 性能结论的唯一信源。设计与使用分别见
> [`core-design`](../architecture/core-design.md) 和 [`guide/core`](../guide/core.md)。
> 当前实现证据绑定本轮 core 收口提交；最终 commit 在提交后回填。

## 测什么、不测什么

唯一有独立算法热路的公开能力是 `IntrusiveList`。`bench/core/core_bench.cc` 使用 4096 个
epoch 外预分配、stable-address 的 member-hook 节点，对拍 Boost.Intrusive `auto_unlink` +
`constant_time_size<false>`；它与 HYLE 具有相同 ownership、生命周期和常态元数据。`std::list`
只作 owning/allocating floor，不能用来宣称同形 SOTA。

三个正常场景分别是 FIFO batch、乱序取消、直接丢弃未 drain 的 batch。每个样本为独立进程，
用 `scripts/bench-stat.py -n 15` 的 paired sign test 判 win/loss；单位 ns/node，越低越好。

## Windows x64 / MSVC 19.50 / Release（2026-07-25）

| 场景 | HYLE | Boost 同形 | std floor | paired 结论 |
|---|---:|---:|---:|---|
| push 4096 + pop 全部 | **1.794** | 2.735 | 23.978 | HYLE 赢 Boost 1.52×；15/15，p=6.1e-5 |
| 乱序 O(1) remove | **1.779** | 2.286 | — | HYLE 赢 Boost 1.29×；14/15，p=9.8e-4 |
| 非空 list 析构自动摘链 | **1.662** | 2.188 | — | HYLE 赢 Boost 1.32×；14/15，p=9.8e-4 |

绝对值 spread 为 22–50%，所以不使用“gap 大于两侧 spread”规则硬判；同进程 paired ratio
方向高度一致，符号检验才是承重证据。结果说明新增 auto-unlink/list cleanup 没把真实生命周期
合同做成慢路径；没有添加 size/owner 等常态元数据。

## WSL2 x64 / GCC 14.2 / Release（2026-07-25）

| 场景 | HYLE | Boost 同形 | std floor | paired 结论 |
|---|---:|---:|---:|---|
| push 4096 + pop 全部 | **1.475** | 1.988 | 11.583 | HYLE 赢 Boost 1.35×；14/15，p=9.8e-4 |
| 乱序 O(1) remove | **1.430** | 2.106 | — | HYLE 赢 Boost 1.47×；15/15，p=6.1e-5 |
| 非空 list 析构自动摘链 | **1.353** | 1.981 | — | HYLE 赢 Boost 1.46×；15/15，p=6.1e-5 |

WSL 使用 Linux ext4 上的独立 Release build 与 `/root/vendor` pinned benchmark rivals；源码从
Windows 主工作树同步。配置 core benchmark 仍触发全项目可选后端/第三方探测，这不是 core
运行时成本，而是下一轮 build/package-system 审查必须修正的组装边界问题。

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
```
