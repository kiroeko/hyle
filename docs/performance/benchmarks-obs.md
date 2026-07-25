# obs 性能证书（2026-07-25）

本页只测 obs 的真实生产形状：运行期 level gate、同步结构化 fan-out、C ABI seam 和多
producer。文件 I/O、异步队列、滚动与上传属于 sink/应用；把它们混进 producer
microbenchmark 既不能解释 registry 复杂度，也会让 OS I/O 淹没这里要守的成本。

## 1. 公开合同与复杂度

| 操作 | 时间复杂度 | 分配 / 同步 |
|---|---|---|
| `should_log` / 被 gate 的宏 | O(1) | relaxed atomic load；参数不求值 |
| `log_record` / `log_message` | O(S)，S 为当前 sink 数 | 不分配；不拿 mutation mutex；短 reader epoch + per-slot active ref |
| `add_sink` / `remove_sink` | O(S)，且 S ≤ 64 | 冷路径 mutation mutex；remove 另等待旧 reader/callback 排空 |
| `flush_sinks` | O(S) | 同步调用支持 flush 的 sink；同 emit 的安全引用合同 |
| C record / sink trampoline | 与 native 同阶 | owning `hyle_status` 只在失败时拥有错误对象 |

registry 使用两份进程期、原地 immutable snapshot。mutation 串行发布新代；producer
只复制已发布快照并取得 active reference，因此注册/注销不把共享 mutex 塞进所有正常
日志。64 槽是显式能力上限，不存在随日志量增长的 retired list 或热路径 heap traffic。

## 2. 方法

- runner：`hyle_obs_bench` / nanobench，Release，15 个独立运行的中位数；sink 只做
  relaxed atomic counter，不做 I/O；
- normal：单 sink preformatted、4-field structured、`std::format` 宏与 C Status seam；
- fan-out：1/4/16 sink，表中按每次 callback 归一化；
- concurrent：2/8 producer 各发 100,000 条，按每条 emit 归一化；
- Windows x64 MSVC、WSL2 GCC 14、Apple M4 AppleClang；
- M4 不宣称物理 P/E 核绑定。interactive 与 utility 都测；utility 组波动更低，故表中用
  utility 作稳定证书，interactive 只作异构调度敏感性审计。

所有点值单位均为 ns/op；并发行为 ns/emit，fan-out 行为 ns/sink callback。

## 3. 三平台 n=15

| 场景 | Windows | WSL2 | M4 utility | 结论 |
|---|---:|---:|---:|---|
| `HYLE_LOG_INFO` gated | 2.007 | 0.916 | 0.906 | O(1) relaxed gate；不格式化 |
| native preformatted，1 sink | 39.328 | 34.488 | 25.001 | immutable snapshot + 安全 active ref |
| structured，4 fields | 38.690 | 34.464 | 24.267 | 与 preformatted 同档；字段为 borrowed view |
| `std::format` 宏，1 sink | 92.150 | 85.761 | 64.356 | 增量主要来自调用方格式化，不是 registry 复杂度 |
| C `hyle_obs_log` + Status | 41.406 | 37.020 | 27.237 | 相对 native seam 约 +2.08 / +2.53 / +2.24 ns |
| fan-out 1 sink | 39.538 | 34.938 | 25.564 | 与 normal 交叉核对 |
| fan-out 4 sinks，per sink | 18.316 | 17.837 | 8.312 | 固定 snapshot 成本被摊薄 |
| fan-out 16 sinks，per sink | 13.120 | 13.726 | 4.459 | 总工作仍严格 O(S) |
| 2 producer，per emit | 105.034 | 88.882 | 79.512 | 无 producer mutex；sink atomic counter 有共享争用 |
| 8 producer，per emit | 95.171 | 64.011 | 123.902 | 平台 cache/调度不同；无复杂度退化 |

最终单原子 wait-state 修正后的 Windows gated/normal spread 为 4.5%/4.3%；WSL normal
为 17.0%，M4 utility normal 为 7.8%，而 fan-out 4/16 的 M4 spread 为 9.0%/8.1%。
短 normal workload 仍会随 WSL 宿主负载或 M4 异构调度漂移；QoS 只是偏好而非
硬核绑定，不能把一次落到 P-core 的点值当产品结论。表保留独立进程中位数，不挑最好点。

## 4. 架构 A/B，而非 benchmark API

审查中实现过一个 `shared_mutex`/exclusive snapshot 版本，再用相同 normal workload 对拍。
Windows 代表值：mutex 版单 producer约 43.9 ns、2 producer约 108 ns、8 producer约
205 ns；最终 immutable snapshot 单 producer约 40 ns、2 producer约 92–109 ns、8
producer约 93–100 ns。最终形状不是为了增加一条易赢的 API，而是删除正常 emit 的
registry lock，在保持 quiescent unregister 的前提下把 8 producer 成本约减半。

## 5. 成熟库对照与 SOTA 判定

本地 Abseil 可选行在 Windows 同一 runner 测得 gated 约 439 ns、stream +
`ToSinkOnly` 约 892 ns。**它们不是同合同排名**：Abseil 同时构造完整 `LogEntry`、类别/
配置元数据并承载更重的全功能日志语义；HYLE 是基础加速库的低层 structured producer
seam。这里引用 Abseil 的价值是 recursion suppression、Flush 和 process-lifetime registry
形状，而不是拿 payload 较轻的 HYLE 宣称“快几十倍”。Folly 同样用于同步 handler、flush
与终端转义的设计核对，Tracy用于 compile-time trace/backend 边界。

因此当前严格结论是：

- gate O(1)，emit/flush O(S)，mutation O(S≤64)，复杂度与成熟同步 sink registry 同档；
- 正常 emit 无分配、无 mutation mutex，多 producer A/B 已关闭旧架构争用；
- C ABI 没有改变复杂度，普通路径只增加约 2–3 ns trampoline/Status seam；
- 暂无一个与 HYLE **相同 metadata、ownership、quiescent-unregister** 合同的成熟 peer 数字，
  所以不伪造全胜表；当前可以认证“没有已知复杂度或正常用法性能缺口”，不能认证跨合同
  的倍数胜利。

若真实消费者以后要求 category routing、per-sink filtering 或异步队列，必须先定义同合同
workload 再重开；不能为了 benchmark 再向每条 record 添加常态元数据。
