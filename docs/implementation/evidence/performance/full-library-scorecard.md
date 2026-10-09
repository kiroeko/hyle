# HYLE 全库性能账本（2026-09-17）

> 历史快照：记录2026-09-17基础架构终审的候选与范围，不是当前模块清单或未完成队列。
> 最新状态以[当前登记](../../module-status.json)为准；旧名称及排除项仅描述当轮事实。

本页是 [`full-library-scorecard.json`](full-library-scorecard.json) 的人读结论。机器账本绑定
当轮 48 个 `CLOSED` factor，覆盖 R2 共同语义、
Mem 定向尾项、R3–R6 四路线及 [R7 终局](../../tasks/compute/four-route-foundation-final-review.md)。
基础架构重构完成；六个 R7 owner 绑定 `87ae5834`，以 P0 源码及关闭跟踪的代表性 codegen 证明继承
未变执行路径。没有新增 P1、没有宣称整个 binary placement 不变，TRACE ON 不继承 OFF 性能资格。

CPU 小片段当时获准接受的是 Graph 接口成本预算；
[比较边界校准](../../tasks/compute/cpu-r3-comparison-boundary-clarification.md)确认裸 direct 与完整
Graph 服务不等价，八项为历史自然路线诊断。R7 已退役这些活动配对入口及旧 Sched 不等责比较；
原统计和负向方向保留。ggml CPU 与 compute_sched 未收口身份不计入数量，旧 copy/跨域 Graph 数据
仍在[历史证据](benchmarks-compute-graph.md)，不作为新路线完成证明。

十三个已有 Compute/Graph roots 的 R6 变化仅属
[Vulkan 交付轮](../../tasks/build/vulkan-r6-product-delivery.md)：
[baseline](../../../../evidence/correctness/vulkan-r6-product-delivery/inherited-baseline.json) 与
[逐树身份](../../../../evidence/correctness/vulkan-r6-product-delivery/p0-unchanged.json)证明原 C++/API
未变，继承 CPU/CUDA/Apple 的测量候选、统计值和 CPU 例外。
[两个 CUDA vendor owner](../../tasks/build/cuda-vendor-content-audit-v1.md)只完成冷内容准入修正，
原 runtime/算法 blobs 与限定性能同样继承。新增交付/规则 P0 claim 不计作性能胜项。
账本不授予跨平台、跨能力的永久 SOTA 标签，也不重新审计无关模块；其它 peer freshness 仍是
2026-09-14 原审计时的事实。

## 结论先行

HYLE 目前可以证明一批有价值的局部最优路线，但还不能证明“全库 SOTA”。当前 98 个去重合同组的严格
方向为：

| 方向 | 合同组 | 正确读法 |
|---|---:|---|
| `WIN` | 25 | 指定合同、peer、平台和规模下，合格区间固定有利于 shipping HYLE route |
| `PARITY` | 3 | 没有 material 固定方向，且有双边 practical equivalence 或相同热路 codegen 依据 |
| `INCONCLUSIVE` | 33 | 区间跨 1、跨平台/顺序方向不同，或当前数据不能解析方向 |
| `N/A` | 36 | 结构/规则证据或不同责任边界的自然路线诊断；没有计时排名 |
| `LOSS` | 1 | CPU Graph holdout 对完整 Native Graph 的负向方向，通过原 5% 非劣门 |

这些数量只用于导航，禁止计算胜率或平均加速比：一个合同组可能覆盖多个规模或同机制工作负载；SIMD
的 34 个 HYLE winner scope 被去重为一个 route-policy claim，而不是人为记 34 次胜利。证据质量也不相同：
35 个 claim 在仓库中保留完整合格可比机器 bundle，20 个绑定外部 hashed bundle，7 个只能从当前报告和审计数据
恢复，22 个是结构/规则证明，14 个为明确排除同合同排名的自然路线诊断。诊断也保留原始数据；
排除的是比较资格，不是否认测量发生。后三类不能承担最强性能宣传声明。

Peer freshness 也属于资格：比较对象采用当前锁定的稳定 release/snapshot，而不是无条件追逐参考仓库
未发布 HEAD。本轮确认 Highway/xsimd 直接产品 lock 与当前参考身份一致；Error 使用的 Abseil 旧 identity
到当前 HEAD 之间没有 Status/FunctionRef 产品源码变化；simdutf 超前 6 个提交只改 Ice Lake validation，
Base64 未变；nghttp2 超前 17 个未发布提交增加 queue/CONTINUATION 防护。后两项的公开结论仍严格限定在
HYLE 实际锁定的 simdutf v9.1.0 与 nghttp2 v1.70.0，不能宣传为战胜未来 HEAD。

## 已经能回答“为什么选 HYLE”的路线

以下是当前最有辨识度、且没有把较弱语义 peer 当对手的代表：

| 消费路线 | 当前可证明结果 | 强 peer / 约束 |
|---|---|---|
| SIMD provider selection | 366 scope 默认 Highway；34 个有限 HYLE winner scope；22 个冗余 spelling 删除；exact TU 零有效指令增量 | Highway/xsimd，按 op/type/ISA/compiler 路由 |
| HYLE VQSort adaptation | partial 4K/k10 `1.147x`；full 4K `5.950x`；full 1M `7.463x` | Highway，同排序与 in-place/no-hidden-allocation 合同 |
| tanh / tanh-GELU | shipping route 选择 xsimd；相对 Highway 约 `1/0.759` 到 `1/0.702` 倍 | MSVC AVX2，tail/in-place/special-value 对齐 |
| Error owning failure / borrow | 保守 `1.21x` 到 `4.78x`；成功热路 codegen 等价 | Abseil Status/StatusOr，同 ownership 与完成边界 |
| reclamation snapshot | `1.365x` 到 `1.381x` | Folly，同 protected immutable read |
| model-cache LRU | `1.222x` 到 `1.273x` | Abseil，同 OOM/rollback/weight/limit 合同 |
| typed unbounded MPMC | `1.261x` 到 `1.289x` | oneTBB，同 allocator/OOM/FIFO/drain 合同 |
| scheduler request fan-out | `1.646x` 到 `1.663x` | Marl，同 worker/input/compute/completion 边界 |
| SIMD Compute rows | exact rsqrt、number-min、byte-select、propagating-min 的 16 个 strata 全部 `PARITY/PASS`；median `0.9964–1.0262` | 同一 locked-Highway lowering、精确数值/NaN/signed-zero、ISA、storage 与 whole-row 边界 |
| HTTP/1 fragmented stream | 约 `3.03x` | Boost.Beast，完整 public transport + HTTP lifecycle；不是裸 parser |
| HTTP/2 small multiplex | `1.466x` 到 `1.478x` | nghttp2，同 session/body/window/flow-control 生命周期 |
| WebSocket masking | `2.005x` 到 `3.354x`，覆盖 124 B 到 1 MiB | Boost.Beast，同 RFC mask state 与完整 buffer |
| TLS verified handshake | 约 `1.10x` | Asio，双方使用同一 OpenSSL provider |
| HTTP Range resume | `1.206x` 到 `1.247x` | libcurl，同断点、验证和原子发布 |
| logical range allocator | `1.031x` 到 `1.043x`，八个 CI 均在 1 上方 | OffsetAllocator 同合同 adapter；裸 handle 快路语义较弱而不排名 |

因此 HYLE 当前真正可辩护的产品价值不是“重新写了所有成熟库”，而是：

1. 以 Highway 等成熟 provider 承担广度和持续维护；
2. 在有限、机器可表达的 scope 上保留 HYLE 或另一 provider 的已证赢家；
3. 在 Error、Mem、Container、Sched 与 IO 之间提供一致的 allocator、ownership、failure、cancel 和
   lifecycle 组合，同时已有若干完整 public-route 工作负载没有出现 material 税。

如果消费者只需要一个 HYLE 尚未证明优势的单独 primitive，并且成熟 peer 的 API 已完全满足需求，直接
使用 peer 仍然是合理选择。HYLE 不能用“功能较少”换来的局部数字要求用户迁移。

## 不能被代表性 workload 遮住的覆盖空白

“factor 有一条 P1”不等于“factor 的全部能力都已测”。能力族审计覆盖了公开功能，性能账本则必须把未
排名部分继续写成未知：

| 区域 | 已有计时证据 | 仍只有 P0/正确性或没有当前同合同排名 |
|---|---|---|
| Core / Math / Time / Platform | host-planning、warm-file、TimerQueue | 大多数 compiler/value helper、clock、CPU/cache/NUMA/frequency、thread/Parker、完整 FS 操作 |
| Mem | reclamation、logical range、VM lazy release | mimalloc/system path、arena/stack/pool、PMR、Buffer/BufferPool、quota、NUMA placement、mapping/warm/residency 的其余形状 |
| Container / Util | LRU、concurrent membership、MPMC、UTF、Base64、FunctionRef | small/static sequence、ring、hash/B-tree、bitset/sparse set、interner、SPSC/work-stealing/sharded queue、Registry/demangle |
| SIMD | 422-scope provider route、有限 winner、三个 inference row、部分 sort/activation/dtype | 不是每个 op/type/ISA/规模都有 current Formal timing；Highway default 表示维护选择，不表示 HYLE 独立赢过它 |
| Sched | fan-out、completion wake | context switch、steal、priority、TaskScope、channel、locks、timers、blocking、cancel/shutdown 的其它真实混合 |
| IO / Process | HTTP/1、HTTP/2、TLS、Transfer、Watch、WebSocket mask | TCP/UDP/local/DNS/file/signal、完整 WebSocket message/server、recursive watch、process lifecycle、跨网络与多 worker |
| Obs | owner/shared metrics | log、trace、sink fan-out、structured model-load 和 exporter 的同合同外部 P1 |
| Compute / Graph | CPU Direct 原四项记录；CPU Graph 对完整 Native 的 4/4 NI PASS，另八项为裸 direct 成本诊断，decode 保留用户预算取舍；CUDA/Apple/Vulkan 已声明完整 FFN 门通过，各 workload 整体严格方向未决 | 四路线普通 C++ 与各自三交付/寿命已验收；范围限已测设备/版本/格式/shape，未覆盖算子目录与完整模型；R7 最小观测/终审尚待完成 |

这张表是当前最大的不确定性：没有证据的能力不得默认为 PARITY，更不能因模块已 `CLOSED` 就继承 SOTA。
新增计时也不按表机械铺开；只有真实 consumer profile、成熟 peer 的明确复杂度优势或跨模块 P2 将某路线
提升为热点，才冻结一条同合同任务。

Build/package 是横向交付单元而非 runtime factor。source embed、installed static 与 shared 三种旅程已经
成立；exact-target SIMD source/static 可以让 provider 路由在编译期消失，普通 fat artifact 每个整批最多
一次缓存函数指针调用。当前没有全库 compile-time、link-time、binary-size 或 shared-vs-static runtime
排名，因此也不能把构建便利或“直接源码使用”写成性能胜项。

## 所有当前负向事实

CPU Graph v3 原四项负向观察完整保留：其中三项是完整 Graph 对裸 direct 的成本差额，不能计作
同等服务 LOSS；decode 的成本目标未决由用户接受。另一项 holdout 对完整 Native Graph 仍为
同合同 LOSS，原 5% NI 通过。原正向 two-worker natural 也同样退出排名，不能只排除负数。
v2 的五项保留为历史事实，不重复计数，完整列于下方。CUDA Graph M1 一个
分层、Air Vulkan Graph M16 四个分层也有严格小幅 LOSS，均通过原 5% NI；它们所属 workload 因
其余分层方向/区间不一致而整体 INCONCLUSIVE，未被上述合同组计数掩盖。其它负向事实也继续保留：

| 路线 | 事实 | 根因 / 行动 |
|---|---|---|
| WSL exact-file watcher | 四个点均慢约 `1.4%–3.2%`；其中两个 strata 的 CI upper `<1` | HYLE 支付 UTF-8 validation、generic multi-target route、bounded coalescing 和 fiber wake，libuv adapter 比较 basename 后直接 callback。低频 reload 控制路接受该取舍；真实 profile 变成 material 才重开 |
| SIMD expert top-k 256/k6 | 四个 median 慢 `0.95%–1.58%`，完整 row 差 `4.3–7.9 ns`；四个 CI 都跨 1 | 通用入口还承载 `k==1` 与大 k 路线，函数体约 `0x738` 对 peer `0x529`。机制存在，但没有引擎 profile 授权拆 API 或加常态 metadata |
| concurrent membership | 两个 median 慢约 `1.2%/2.2%`，所有 CI 跨 1 | 当前只证明 5% 非劣；HYLE 产品面还承担安全 concurrent erase。后续必须用真实 read/write/erase mix 决定，不以 read-only 点值扭曲类型 |
| Error success | median 慢 `0.14%–0.77%`，所有 CI 跨 1 | noinline success node 与 64-node hot path 逐指令相同，历史轮方向也跨 1；归因于布局/测量分辨率，PARITY |
| HTTP/1 small、UTF、Base64 decode 等 | 个别 order/placement median 小于 1，但 CI 跨 1 或方向翻转 | 保持 `INCONCLUSIVE`，不得按单次点值改产品 |

另有三个容易误导的历史数字：HTTP/2 v1 大块 flow-control 的固定 LOSS 已由 v2 同合同修复并替代；
SIMD NEON KV64 曾有约 `27%` LOSS，后续以 `DupOdd` 修正，当前同合同证据为 parity；Sched
`structured-spawn-join` 的 HYLE 绝对吞吐约低 `6%–8%`，但双方 ownership/lifecycle 不同，合同明确为
`NOT_COMPARABLE`，不能计为 LOSS、PARITY 或 WIN。这正是“不得苹果对橘子”的实际应用。

## 真实纵切证据与缺口

当前最接近真实纵切的合格证据是完整 public transport + HTTP/1 request/streaming response + body
validation + connection retirement：128×1 KiB 为 `INCONCLUSIVE` 且 5% NI 通过，16×64 KiB fragmented
为约 `3.03x WIN`。TLS、HTTP/2、Range resume、Sched fan-out、model-cache LRU、Mem range/VM，以及 SIMD
推理/Compute row 也使用真实消费者形状，而非单指令纳秒循环。

四路线目前各自已有真实 Direct/Graph FFN、变化输入、资源寿命及普通三交付证据；CPU、CUDA、
Apple 和 Vulkan 的默认实现、数值格式及完成责任分别按自己的有限合同验收，不能跨合同拼出排名。
项目仍未把 model open/map、metadata、tensor preparation、scheduler、多 provider 计划、kernel、
token stream、HTTP/TLS 一次性计入同一 P2；没有完整模型的 tokens/s、time-to-first-token、p99、
峰值 RSS、内存带宽和 energy/token，也没有完整路由、handler、JSON/SSE 应用层服务器比较。
任何“HYLE 推理端到端更快”都没有证据资格。

后续 P2 的正确触发点不是再跑全部模块，而是第一个真实 HYLE inference/service consumer 可执行时冻结：

- 模型加载：wall time、peak RSS、major faults、warm/cold cache，按文件/映射/量化元数据完整计费；
- prefill/decode：tokens/s、TTFT、p50/p99 inter-token latency、memory bandwidth、energy/token；
- 服务：固定模型与请求分布下的 HTTP(S)/SSE 并发、backpressure、cancel 和 shutdown；
- ablation：只切换一个 HYLE route 或 provider，证明收益归因而非组合偶然。

在此之前，允许的表述只能是“这些明确路线在这些条件下胜出或非劣”，不能是“全库/全平台 SOTA”。

## 怎样复核

机器账本逐 claim 保存 consumer shape、合同、peer、规模、绝对值、ratio/CI、平台敏感性、机制、行动、
限制和证据路径。执行：

```powershell
python scripts/check-performance-scorecard.py
python scripts/check-module-closure.py
python scripts/check-docs.py
```

任何模块从 `CLOSED` 合法重开并签发新 candidate 后，账本 candidate 会故意失配，迫使维护者重新判断旧
证据是 exact、可证明继承，还是已经失效；这不是第二套状态机。

## 2026-09-16：R3 Graph v2（历史候选）

该历史候选记录时 owner 尚未 CLOSED；以下完整方向不重复并入当前 v3 合同组数量。
[原始工程证据](../../../../evidence/performance/graph-cpu-route-p1-v2-2b613413-w1-win/README.md)
资格及 controls PASS，n=31 已到上限；工程门 11/12 PASS，聚合 INCONCLUSIVE。
严格方向 5 LOSS / 2 WIN / 5 INCONCLUSIVE；LOSS 未被 5% 非劣 PASS 改名。

| Workload | median ratio 范围 | 全部分层 CI 外界 | 严格方向 | 非劣门 |
|---|---|---|---|---|
| decode-fused-boundary-1x256x512 | 0.970136–0.974529 | 0.962630–0.981172 | LOSS | PASS |
| decode-natural-sequence-1x256x512 | 0.981288–0.989639 | 0.973949–0.999395 | LOSS | PASS |
| short-prefill-fused-boundary-16x512x1024 | 0.994235–1.000967 | 0.981550–1.009335 | INCONCLUSIVE | PASS |
| short-prefill-natural-sequence-16x512x1024 | 0.990111–1.005592 | 0.981816–1.017910 | INCONCLUSIVE | PASS |
| holdout-fused-boundary-17x129x257 | 0.984310–0.989053 | 0.979737–0.998289 | LOSS | PASS |
| holdout-natural-sequence-17x129x257 | 0.987358–0.993463 | 0.982557–0.997530 | LOSS | PASS |
| decode-native-sequence-1x256x512 | 0.999597–1.008232 | 0.992717–1.015219 | INCONCLUSIVE | PASS |
| short-prefill-native-sequence-16x512x1024 | 0.978004–0.987949 | 0.966585–1.001661 | INCONCLUSIVE | PASS |
| holdout-native-sequence-17x129x257 | 0.959773–0.964887 | 0.954126–0.970966 | LOSS | PASS |
| two-worker-fused-boundary-17x129x257 | 0.970361–0.982654 | 0.950646–1.002706 | INCONCLUSIVE | UNRESOLVED |
| two-worker-natural-sequence-17x129x257 | 1.157200–1.173222 | 1.137193–1.202477 | WIN | PASS |
| two-worker-native-sequence-17x129x257 | 1.268071–1.317464 | 1.242161–1.330614 | WIN | PASS |

该候选已停止采样；定向产品修正后的 v3 完整结果见下节，历史固定负向与未决事实保留。

## 2026-09-16：R3 CPU Direct 默认入口

[原始工程证据](../../../../evidence/performance/compute-cpu-route-p1-v3-d9712e68-w1-win/README.md)
资格、全部 controls 与 4/4 非劣门 PASS，n=15 即停止；严格方向 1 WIN / 3 INCONCLUSIVE。
R3 已结案，这四项完整纳入上文当前 CLOSED 计数。

| Workload | median ratio 范围 | 全部分层 CI 外界 | 严格方向 | 非劣门 |
|---|---|---|---|---|
| decode-row-major-f32-matmul-m1-k1024-n1024 | 0.996999–1.001596 | 0.970106–1.021232 | INCONCLUSIVE | PASS |
| decode-row-major-f32-matmul-m4-k1024-n1024 | 0.997917–1.003004 | 0.993853–1.006296 | INCONCLUSIVE | PASS |
| short-prefill-row-major-f32-matmul-m32-k512-n1024 | 1.047825–1.063461 | 1.027823–1.076037 | WIN | PASS |
| decode-transposed-weight-f32-matmul-m1-k1024-n1024 | 0.997157–1.003399 | 0.981472–1.011034 | INCONCLUSIVE | PASS |

## 2026-09-16：R3 Graph v3 原统计未决，用户接受具名例外

[完整原始包](../../../../evidence/performance/graph-cpu-route-p1-v3-7fa6dc0a-w1-win/README.md)：
当时的仪器资格、实际 workload A/A、五项 controls 与身份深检 PASS，n=31 已到上限；原程序输出
11/12 门 PASS。这些仪器结果不能替代 API 服务等价性；其中八项现在按自然路线成本解释。
原双 worker 同实现项已过门。唯一未决为约 12 微秒 decode-fused 的一个 CI，下界
0.9517370300 仍低于 1/1.05；中位额外耗时约 2.6%–3.5%，其 CI 允许的退化上界约 5.07%。
不能把该项写为统计 PASS，也没有足够结论写成 material FAIL。用户已明确批准
[具名验收例外](../../tasks/compute/cpu-r3-acceptance-exception.md)，三个 Graph owner 按协议结案。
工程动作是 ACCEPT_TRADEOFF，原统计聚合仍 INCONCLUSIVE；本候选不再采样。
下表保留当时 4 LOSS / 2 WIN / 6 INCONCLUSIVE 的原输出，不把它当作当前同合同计数。
八个 fused/natural 裸 direct 对照均已降为诊断；四个完整 Native Graph 对照保持 1 LOSS /
1 WIN / 2 INCONCLUSIVE。原绝对值、比值、CI 与样本均未修改。

| Workload | median ratio 范围 | 全部分层 CI 外界 | 严格方向 | 非劣门 |
|---|---|---|---|---|
| decode-fused-boundary-1x256x512 | 0.966131–0.974263 | 0.951737–0.984423 | LOSS | UNRESOLVED |
| decode-natural-sequence-1x256x512 | 0.987477–0.995959 | 0.977373–1.005240 | INCONCLUSIVE | PASS |
| short-prefill-fused-boundary-16x512x1024 | 0.990410–1.002805 | 0.985584–1.014731 | INCONCLUSIVE | PASS |
| short-prefill-natural-sequence-16x512x1024 | 0.993661–1.003142 | 0.985273–1.015241 | INCONCLUSIVE | PASS |
| holdout-fused-boundary-17x129x257 | 0.987678–0.995114 | 0.981875–0.999792 | LOSS | PASS |
| holdout-natural-sequence-17x129x257 | 0.991623–0.994412 | 0.983759–1.001743 | INCONCLUSIVE | PASS |
| decode-native-sequence-1x256x512 | 1.005643–1.012878 | 0.999680–1.021756 | INCONCLUSIVE | PASS |
| short-prefill-native-sequence-16x512x1024 | 0.978415–0.993894 | 0.970855–1.000433 | INCONCLUSIVE | PASS |
| holdout-native-sequence-17x129x257 | 0.962867–0.967719 | 0.955836–0.971654 | LOSS | PASS |
| two-worker-fused-boundary-17x129x257 | 0.972652–0.977621 | 0.961814–0.993454 | LOSS | PASS |
| two-worker-natural-sequence-17x129x257 | 1.162959–1.175137 | 1.144884–1.187616 | WIN | PASS |
| two-worker-native-sequence-17x129x257 | 1.300473–1.320614 | 1.277643–1.333040 | WIN | PASS |

## 2026-09-17：R4 CUDA 最终默认与边界成本

六个 CUDA owner 按755fb7c7产品候选 CLOSED，新增一个冷规则/资源 claim、四个完整 FFN 接入 claim
和两个原生边界诊断。原 R3 CPU 数字与接受例外全部保留，没有重跑 CPU 性能。

[Direct v3](../../../../evidence/performance/compute-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md) 与
[Graph v3](../../../../evidence/performance/graph-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md)均完成 n=15、
qualification、5% NI、controls/validators 与原始重放。四个 workload 的整体严格方向均为
INCONCLUSIVE；Graph M1 单个分层的小幅 LOSS 在证据中保留。默认与独立参考使用相同成熟精确数学，
这四项不是 HYLE 算法胜利。v2 的原 HYLE 比较仍是采用成熟默认的规则依据。

[原生边界](../../../../evidence/performance/cuda-r4-native-boundary-v1/README.md)的两项诊断因为完整
责任不同，classification 为 NATURAL_ROUTE、direction=N/A，不能签 NI/算法胜负；
`REJECTED_DIAGNOSTIC` 在此表示排除可比性排名，并非数值检查失败。差值、冷准备范围、opaque SDK
存储限制与所有原始数据保留。性能资格仅签 RTX4070/SM89、driver610.62、Toolkit13.3.73 的两个
resident FFN 原型，不签其它 GPU、版本、shape 或模型吞吐，也不代替 R5/R6/R7。

## 2026-09-17：R5 Apple 与 R6 Vulkan

[Apple 最终原始证据](../../../../evidence/performance/apple-r5-final-1636365b-m1-mac/README.md)
对应产品 `1636365b`，由 R6 逐树身份继承到当前交付候选。Direct/Graph × M1/M16 四个完整量化
FFN workload 的原 5% NI 全过，n15 停止；严格区间均跨 1，各组 INCONCLUSIVE。默认采用成熟
MLX full-region，独立参考拥有相同原生 owner/lifecycle；caller-thread 裸 MLX 两项成本诊断
属于 NATURAL_ROUTE / NOT_COMPARABLE。资格限 M4 与所记录的 MLX/macOS/SDK 域。

Vulkan 产品 `8e45b104` 的 Windows
[Direct](../../../../evidence/performance/vulkan-r6-w1-direct-8e45b104/README.md) /
[Graph](../../../../evidence/performance/vulkan-r6-w1-graph-8e45b104/README.md) 与 Air
[Direct](../../../../evidence/performance/vulkan-r6-m1-direct-8e45b104/README.md) /
[Graph](../../../../evidence/performance/vulkan-r6-m1-graph-8e45b104/README.md) 全部完成资格、controls、
独立 full-output oracle 及原 5% NI；每入口/profile 的 16 门全过，n15 停止。四个去重 workload
各自纳入两个 profile，整体方向均 INCONCLUSIVE；Air Graph M16 四个 peer-allocation-first
分层的 median 额外延迟约 1.2%–1.5%，严格 LOSS 保留。采用原样 ggml strict-F32 shader 的
同合同接入比较不构成 HYLE 算法胜利或整个 ggml runtime 排名。

Vulkan [Windows](../../../../evidence/performance/vulkan-r6-w1-native-boundary-8e45b104/README.md)
与 [Air](../../../../evidence/performance/vulkan-r6-m1-native-boundary-8e45b104/README.md) 的
Direct/Graph 原生成本均完成四顺序分层 × n15；裸 submit/fence 不承担完整取消、quarantine
及生成结果 lease，继续排除 NI/排名。正负差额不隔离 host 包装成本，不以缓存猜测解释方向。
资格限已记录的 RTX 4070 / M4-MoltenVK、strict F32 和固定 shape；没有 Linux 原生 GPU、
AMD/Intel 或其它数值格式的性能资格。两路线的普通三交付、host 插桩和失败寿命分别由
[R5 terminal](../../tasks/compute/apple-v1-four-route-foundation.md) 与
[R6 terminal](../../tasks/compute/vulkan-v1-four-route-foundation.md)持有。

本次补齐四个 Apple 与四个 Vulkan workload、两项规则、四项不可排名诊断、两项冷交付和一项
CUDA vendor 内容准入，共 17 个 claim。原 81 个合同组的统计值、方向与工程结果没有改写，
只按真实源码身份更新当前 owner 绑定。R7 尚未完成，不由账本登记签署整个基础架构结束。

## 2026-09-17：CPU Graph 比较资格校准

上述 R6 登记之后，按用户要求复核先前的独立调查并修正落账：四个 fused-boundary 和四个
natural-sequence 对照没有 Graph 的接纳/排他/关闭/Completion 服务，不能继续标作同合同。
八项的原有利与不利统计对称保留，当前比较均为 NATURAL_ROUTE，排名/同合同 NI 为 N/A。
[逐项前后记录](../../../../evidence/review/cpu-r3-comparison-boundary/identity.json)证明全部 98 项
绝对值、ratio/CI、采样事实不变，其他 90 个 claim 完全不变。没有新的计时、产品修正或性能 PASS。

## 2026-10-05：CPU 官方低精度完整 MoE 增量

本节登记D5新增两个合格工作负载；上述98项与机器JSON是2026-09-17的基础架构历史快照，
没有将旧算子claim重新绑定到新产品，也未重签无关模块。当前owner状态以module-status.json为准。

产品与SDK固定`3c22b7b9`、消费者`a8ec029a`、同数学Direct来源`a4f56461`，CPU-only
LibTorch2.8.0+cpu。比较普通managed Graph/Direct一次完整变化输入周期，包括原A8、权重codec、
matmul、复制/转换、同步与Completion释放；不比较不同数学或裸库API。

| 工作负载 | Direct/Graph速度ratio范围（大于1有利于Graph） | 四顺序CI外包络 | 严格方向 | 固定5%非劣门 | 工程动作 |
|---|---|---|---|---|---|
| GLM5.3 Flash官方FP8 MoE，两个完整片段/周期 | 0.991525–1.011748 | 0.980660–1.025591 | INCONCLUSIVE | PASS | KEEP成熟组合与普通入口 |
| DeepSeekV4 Flash官方FP4/FP8 MoE，三个完整片段/周期 | 0.994669–1.002478 | 0.970165–1.023398 | INCONCLUSIVE | PASS | KEEP成熟组合与普通入口 |

每条四种执行/分配顺序各n15，共60产品进程、8工作负载A/A和30公共控制，全部完成；
W1-WIN/i9-13900K、MSVC19.50、caller P核/native-runtime-pair、OMP/MKL1。
全部逐顺序CI跨1，不能把实用带PARITY/PASS说成数学胜利。没有Pro独立P1、其它硬件或整模型声明。
[完整证据与限制](../../../../evidence/performance/cpu-official-fragments-p1/README.md)、
[原值summary](../../../../evidence/performance/cpu-official-fragments-p1/qualified-run/run/summary.json)
及[严格重放](../../../../evidence/performance/cpu-official-fragments-p1/qualified-run/run/strict-replay.json)。
