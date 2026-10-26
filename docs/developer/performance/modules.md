# 基础模块性能资料

这里汇总已有模块对照与性能排查的入口，帮助判断哪些机制值得复用、哪些结论仍有边界。
**不是对当前 HEAD 的全库重测，也不是支持矩阵或全库胜率。** 完整推理的最新窗口和主机／设备归因在
[decode 性能资料](decode.md)；比较方法和 HTTP/2 排查经过在[性能说明](README.md)。

## 2026-09-17 的基础模块对照

下表摘自仍保留的[原始汇总 JSON](../../implementation/evidence/performance/full-library-scorecard.json)。
其中每个 claim 保存对应候选、参考版本、workload、比例及证据路径。
它绑定当时源码，不因文档恢复就成为当前源码的性能结论；部分旧 Markdown 报告和流程文件已经删除，
原 JSON 内的旧路径也不保证仍在工作树。复查当前实现时，先核对实际源码和参考版本，再复用对应原始记录。

倍率均大于 1 有利于 HYLE：吞吐用 HYLE／参考，耗时用参考／HYLE。
以下范围是原记录中的分层或点估计范围，**不是统一置信区间**；不同平台、规模和合同不能合成平均加速比。

| 模块／真实调用形状 | 当时结果 | 参考及不能外推的范围 |
|---|---|---|
| Core、标量 Math 与多数 Time 原语 | 以 constexpr／内联代码、数值和布局检查为主，未给整体速度排名 | 编译器／标准库；没有为无额外机器码的包装强造纳秒胜负 |
| Error：owning failure／借用，成功路径 | 失败路线原保守倍率约 1.21–4.78；成功热路径机器码等价 | Abseil Status／StatusOr；不代表任意错误消息和保留历史都零成本 |
| Platform：host-planning snapshot | 1.0157–1.0372，部分区间跨 1，方向未定 | Windows 的 oneTBB 衍生 OS adapter；不覆盖其它系统或所有探测 |
| Mem：逻辑范围分配 | 24.759–25.054 对 23.861–24.122 M op/s，倍率 1.0310–1.0432 | 对齐、合并、owner 合同相同的 OffsetAllocator adapter；不是裸 handle 比较 |
| Mem reclaim：受保护只读快照 | 倍率 1.365–1.381 | Folly；不是所有分配器、回收器或真实会话成本 |
| Container：4096 项 LRU 混合查找／插入／权重淘汰 | 148.99–152.57 对 119.54–123.52 M op/s，倍率 1.2223–1.2728 | Abseil adapter；保留相同 OOM／回滚／限额，不覆盖所有 hash 分布 |
| Container queue：4 producer／4 consumer 无界 typed MPMC | 33.70–35.00 对 25.05–25.92 M op/s，倍率 1.2614–1.2888 | oneTBB 无界队列；不能外推 bounded、SPSC 或不同等待政策 |
| Container text：UTF 与 Base64 | UTF／Base64 decode 方向未定；小记录 Base64 encode 倍率 1.0329–1.0449 | 锁定 simdutf 的相同 caller-buffer 路线；不是每种文本规模都更快 |
| Intrusive：4096 节点的稳定排序／合并等 | 原报告保留部分优势；counted／forward 的方向证据不完整 | Boost.Intrusive；个别历史报告缺逐格区间，不能补签严格持平 |
| Util：4096 记录的 FunctionRef 调用 | 普通间接调用方向未定；non-type bound-member 倍率 1.2452–1.2644 | Abseil FunctionRef；优势限该 callable 形状 |
| Time：64 timer 的完整调度生命周期 | 62.80–64.63 对 53.42–55.92 M op/s，倍率 1.129–1.190 | Seastar timer queue；限该分布，不能代表所有取消或空闲轮询 |
| Sched：四 worker 请求 fan-out／collect | 6.940–6.943 对 4.191–4.197 M op/s，倍率 1.6462–1.6625 | Marl 固定版本，普通优先级、同工作和完成边界；不覆盖所有 TaskScope／取消 |
| Obs：owner-thread／shared metrics | owner 倍率 1.1053–1.1452；shared 方向未定 | 直接 C++ 机制基线；没有宣称胜过完整外部观测系统 |
| IO：HTTP/1 fragmented streaming | 17.655–17.835 对 5.830–5.881 K op/s，倍率 3.030–3.065 | Boost.Beast，完整 transport／协议生命周期；不是完整 HTTPS 推理服务 |
| IO：HTTP/2 small multiplex | 201.87–204.51 对 139.54–140.93 K op/s，倍率 1.4658–1.4780 | 当时 nghttp2 h2c 单 worker 小响应；不能代替后续大 body 强背压对照 |
| IO：HTTP Range resume | 33.63–34.90 对 28.03–29.37 op/s，倍率 1.2063–1.2474 | libcurl，localhost、校验及原子发布；不预测公网下载 |
| Compute／Graph | 当时只有有限 Direct／Graph、FFN 与资源旅程 | 各自后端、dtype、设备及调用责任不同；旧片段不能代替当前完整模型速度 |

记录中成功路径、Codegen、局部吞吐与完整生命周期属于不同证据，不能相互补足。
大多数文件系统、容器、内存、日志、优先级、取消和 Process 形状没有这轮同合同外部排名。
旧记录中的非劣容忍值不抹掉已测到的小幅落后，也不表示后来所有组合已经优化。

## 2026-10-18 的复杂度与资源排查

后续[全模块排查记录](../../evidence/performance/library-review-2026-10-18/README.md)按当时构建表审查了 64 个 factor，
**不是每个模块、算子和 shape 都重新测速**。记录包含 31 个处理条目：26 项性能／复杂度机制、
4 项接线／正确性修正，另 1 项 HTTP/2 归因与基准校正。条目数不是严格去重算法数或当前胜负数。

| 机制 | 当轮改进与实际成本边界 | 原记录 |
|---|---|---|
| StateStore 统计 | 去掉重复 root 扫描；1024 独立 root 的热统计约 172–174 → 1.25 µs。冷更新／排序另计 | [独立 root](../../evidence/performance/library-review-2026-10-18/state-distinct-roots/summary-final.json) |
| CUDA BF16 SDPA 与有界 attention 临时量 | 适用融合或 Q 分块避免内部全长 scores；8K F32 点的显式 private 请求 8368 → 48.5 MiB | [排查与空间口径](../../evidence/performance/library-review-2026-10-18/README.md)；不等于进程总显存或模型吞吐 |
| CUDA Concat／Hadamard | 平衡拼接消除累积前缀二次项；Hadamard 4096 的 host／device 常量各从 64 MiB 降到 16 KiB | [Concat](../../evidence/performance/library-review-2026-10-18/concat/summary.json)、[Hadamard](../../evidence/performance/library-review-2026-10-18/hadamard/summary.json)；平衡拼接仍不是一次写完的线性最优算法 |
| CPU 中间存储与 batch workspace | 原多节点 arena 改复用，固定 64×1 MiB Neg 链冷 private 请求约 63.94 → 2.94 MiB；未使用 scratch 少 144 MiB／slot | [存储复用](../../evidence/performance/library-review-2026-10-18/cpu-region-storage/summary.json)、[workspace](../../evidence/performance/library-review-2026-10-18/cpu-batch-workspace/summary.json)；不等于全部 RSS |
| io_uring | CQE 分类不再逐个扫描全部 pending 特殊操作；另修 direct slot 合同及重复扫描 | [CQE](../../evidence/performance/library-review-2026-10-18/io-cqe/summary.json)、[direct slots](../../evidence/performance/library-review-2026-10-18/io-direct-slots/summary.json)；访问计数与延迟分别看 |
| Container／Sched／TimerQueue | erase 少一次 hash；密集 churn 避免频繁同容量重建；BlockingPool 跳过无退休扫描；空闲 timer 避免重复遍历 | [erase](../../evidence/performance/library-review-2026-10-18/concurrent-erase/summary.json)、[churn](../../evidence/performance/library-review-2026-10-18/flat-churn/summary.json)、[blocking](../../evidence/performance/library-review-2026-10-18/blocking-pool/summary.json)、[timer](../../evidence/performance/library-review-2026-10-18/timer-idle/summary.json)；各有限 workload，不代表外部普遍胜出 |
| Process 环境合并 | 排序索引将重复比较改为 M log M＋E log M，保留平台大小写语义 | [同工计数与回归](../../evidence/performance/library-review-2026-10-18/process-env/summary.json)；不是完整 spawn 吞吐排名 |
| HTTP/2 大 body | 参考绕过 TCP 检查是双 Task 小残差的重要来源；修正归因对照后，三轮中位差 −0.92%／−1.49%／−0.43% | [TCP 干预](../../evidence/performance/library-review-2026-10-18/http2/tcp-transport/summary.json)；产品本轮未因此变快，原自然路线差距仍需分别看 |

Apple／Vulkan 的 attention 分块、mask、dtype 与显存改进有各自实际受理域和失败记录，
在[同一排查记录](../../evidence/performance/library-review-2026-10-18/README.md)按实验顺序保留。
不能只取前段方案：后续曾撤回使峰值增加或显著变慢的候选，并修正过早期范围判断。
同样，G04／G05 的[双 K 加载筛选](../../evidence/performance/g04-g05-same-process-ilp-2026-10-17/README.md)
没有证明稳定完整流程收益，未采纳；它与后续制品精度／计时边界校正是不同问题。

## 从哪里复查或重新运行

先选实际问题和现成构建，不按本表重跑全库。源码、设备、输入、数值、权限、准备与完成责任对齐后，
再判断旧数据能否解释当前实现。证据目录里的临时源码和冻结镜像用来复现原结论，不是现行 API 或开发入口。

当前 benchmark 源码及目标分别在 [bench](../../../bench/) 各模块的 `CMakeLists.txt`。
初始化可通过现有高级 CMake 参数启用 benchmark，例如只准备 CPU 容器比较：

```text
python scripts/init.py --name bench-container --components container --backends cpu --config Release -D HYLE_BUILD_BENCH=ON
python scripts/build.py --name bench-container --target hyle_container_benches --jobs 2
```

这构建该模块已登记的 benchmark，不自动启用或获取每个可选参考库，也不自动产生旧表的对照数据。
按对应 benchmark 的源码、参数及原记录选择具体程序；统计工具和可选参考库入口见
[脚本说明](../../../scripts/README.md)。不要把缺失参考的单臂结果写成胜出。
HTTP/2 的 Linux 固定配对入口另见[复现方法](README.md#http2-配对测量)。

复核历史文本可用 `git show 59e481ae8^:docs/implementation/evidence/performance/full-library-scorecard.md`；
它只用于解释旧数字；其中的状态登记、审批和资格流程不是现行开发约定。
