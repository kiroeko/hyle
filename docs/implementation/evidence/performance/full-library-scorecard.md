# HYLE 全库性能账本（2026-09-16）

本页是 [`full-library-scorecard.json`](full-library-scorecard.json) 的人读结论。机器账本只绑定
[`module-status.json`](../../module-status.json) 中当前 38 个 `CLOSED` factor，包含 R2 已收口的
compute_ops schema、Mem reservation 尾项和 R3 CPU。CPU 唯一已量化的 Graph 小片段未决项经用户
批准接受，原 LOSS/INCONCLUSIVE 保留；GPU 与 ggml 等未收口 owner 不计入数量。旧 CUDA copy 与跨域 Graph 数字保留在
[历史证据](benchmarks-compute-graph.md)，不作为四路线基础架构完成证明。账本不授予跨平台、跨能力的
永久 SOTA 标签，也不重新审计无关模块；其它 peer freshness 仍是 2026-09-14 原审计时的事实。

## 结论先行

HYLE 目前可以证明一批有价值的局部最优路线，但还不能证明“全库 SOTA”。当前 72 个去重合同组的严格
方向为：

| 方向 | 合同组 | 正确读法 |
|---|---:|---|
| `WIN` | 26 | 指定合同、peer、平台和规模下，合格区间固定有利于 shipping HYLE route |
| `PARITY` | 3 | 没有 material 固定方向，且有双边 practical equivalence 或相同热路 codegen 依据 |
| `INCONCLUSIVE` | 25 | 区间跨 1、跨平台/顺序方向不同，或当前数据不能解析方向 |
| `N/A` | 14 | P0 结构成本或没有完整同合同 peer；没有计时排名 |
| `LOSS` | 4 | CPU Graph 的固定负向方向保留；其中一项工程例外不改写为统计 PASS |

这些数量只用于导航，禁止计算胜率或平均加速比：一个合同组可能覆盖多个规模或同机制工作负载；SIMD
的 34 个 HYLE winner scope 被去重为一个 route-policy claim，而不是人为记 34 次胜利。证据质量也不相同：
31 个 claim 在仓库中保留完整机器 bundle，20 个绑定外部 hashed bundle，7 个只能从当前报告和审计数据
恢复，14 个是 P0 结构证明。后两类不能承担最强宣传声明。

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
| Compute / Graph | compute_ops 为 P0 schema；CPU Direct 4/4 非劣 PASS，Graph v3 11/12 非劣 PASS + 1 项用户验收例外，原统计聚合未决 | R3 已收口，GPU 新路线未验收；历史 copy/跨域结果不继承为新路线完成 |

这张表是当前最大的不确定性：没有证据的能力不得默认为 PARITY，更不能因模块已 `CLOSED` 就继承 SOTA。
新增计时也不按表机械铺开；只有真实 consumer profile、成熟 peer 的明确复杂度优势或跨模块 P2 将某路线
提升为热点，才冻结一条同合同任务。

Build/package 是横向交付单元而非 runtime factor。source embed、installed static 与 shared 三种旅程已经
成立；exact-target SIMD source/static 可以让 provider 路由在编译期消失，普通 fat artifact 每个整批最多
一次缓存函数指针调用。当前没有全库 compile-time、link-time、binary-size 或 shared-vs-static runtime
排名，因此也不能把构建便利或“直接源码使用”写成性能胜项。

## 所有当前负向事实

已收口机器账本中的 CPU Graph v3 有四项固定 LOSS：三项通过原 5% 非劣门，一项保留统计未决并经
用户明确接受工程取舍。v2 的五项保留为历史事实，不重复计数，完整列于下方。其它负向事实也继续保留：

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

但这些仍不是完整推理引擎：上一轮曾验证 CPU RMSNorm → MatMul → SiluAndMul 多算子纵切；Compute 与
Graph 已因真实能力缺口重开，所以该结果现在只是历史基线，不代表当前候选。项目仍未把 model open/map、
metadata、tensor preparation、scheduler、多 provider 计划、kernel、token stream、HTTP/TLS 一次性计入
同一 P2；没有 tokens/s、time-to-first-token、p99、峰值 RSS、内存带宽和 energy/token；也没有完整路由、
handler、JSON/SSE 应用层服务器比较。任何“HYLE 推理端到端更快”都没有证据资格。

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

以下 owner 尚未 CLOSED，单独列出完整性能方向，不并入上文当前 CLOSED 数量。
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
资格、实际 workload A/A、五项 controls 与身份深检 PASS，n=31 已到上限；11/12 非劣门 PASS。
原双 worker 同实现项已过门。唯一未决为约 12 微秒 decode-fused 的一个 CI，下界
0.9517370300 仍低于 1/1.05；中位额外耗时约 2.6%–3.5%，其 CI 允许的退化上界约 5.07%。
不能把该项写为统计 PASS，也没有足够结论写成 material FAIL。用户已明确批准
[具名验收例外](../../tasks/compute/cpu-r3-acceptance-exception.md)，三个 Graph owner 按协议结案。
工程动作是 ACCEPT_TRADEOFF，原统计聚合仍 INCONCLUSIVE；本候选不再采样。
当前 4 LOSS / 2 WIN / 6 INCONCLUSIVE 全部纳入机器账本，没有删去负向 workload。

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
