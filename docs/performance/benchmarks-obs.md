# obs-runtime 历史性能证据（2026-07-29，`06cec5a`）

> **2026-08-25 freshness gate：**本页只覆盖旧 gate/record/fan-out 热路。Tracy 0.14.1、
> 当前 29-symbol C 面与 caller-storage sink 不在该次采样合同内；obs 已重新打开。相同 source
> identity、peer pin、编译合同与三平台重跑完成前，不是当前发布证书。现行状态只看
> [`pre-release-readiness`](../milestones/pre-release-readiness.md)。

本页只测 obs 的真实生产形状：运行期 level gate、同步结构化 fan-out、C ABI seam 和多
producer。文件 I/O、异步队列、滚动与上传属于 sink/应用；把它们混进 producer
microbenchmark 既不能解释 registry 复杂度，也会让 OS I/O 淹没这里要守的成本。

## 1. 公开合同与复杂度

| 操作 | 时间复杂度 | 分配 / 同步 |
|---|---|---|
| 编译期 floor 以下的 `HYLE_LOG_*` / `HYLE_OBS_EMIT_*` | O(0) | 参数不求值；目标文件没有 obs 未定义符号 |
| `should_log` / 运行期被 gate 的宏 | O(1) | 一次 relaxed atomic load；参数不求值 |
| `log_record` / 长度已知的 view、`hyle_obs_log_n` | O(S)，S 为当前 sink 数 | 不分配、不扫描 message；不拿 mutation mutex；短 reader epoch + per-slot active ref |
| NUL convenience `log_message(const char*)` / `hyle_obs_log` | O(M + S) | M 为消息首个 NUL 前字节数；之后与 exact-length 路径相同 |
| `add_sink` / `remove_sink` | O(S)，且 S ≤ 64 | 冷路径 mutation mutex；remove 另等待旧 reader/callback 排空 |
| `SinkRegistration` create/reset/destruct | 与 add/remove 同阶 | 只在控制面；不给 record/sink 增加热路径字段 |
| `flush_sinks` | O(S) | 同步调用支持 flush 的 sink；同 emit 的安全引用合同 |
| C record / sink trampoline | 与 native 同阶 | owning `hyle_status` 只在失败时拥有错误对象 |
| 默认 sink | O(M + fields + I/O) | 控制字节流式转义；Windows debugger mirror 为固定 4 KiB，无 heap |

registry 使用两份进程期、原地 immutable snapshot。mutation 串行发布新代；producer
只复制已发布快照并取得 active reference，因此注册/注销不把共享 mutex 塞进所有正常
日志。64 槽是显式能力上限，不存在随日志量增长的 retired list 或热路径 heap traffic。

## 2. 方法

- 实现：runtime/API 为 `4d31bcf`，采样稳定性调整后证书绑定 `06cec5a`；
- runner：`hyle_obs_bench` / nanobench，Release，15 个独立进程运行的中位数；sink 只做
  relaxed atomic counter，不做 I/O；
- normal：单 sink preformatted、4-field structured、`std::format` 宏与 C Status seam；
- fan-out：1/4/16 sink，表中按每次 callback 归一化；
- concurrent：2/8 producer 各发 100,000 条，按每条 emit 归一化；
- Windows x64 MSVC 19.50、WSL2 GCC 14.2、Apple M4 AppleClang 21；
- M4 不宣称物理 P/E 核绑定。interactive 与 utility 都测；utility 组波动更低，故表中用
  utility 作该轮稳定证据，interactive 只作异构调度敏感性审计。

所有点值单位均为 ns/op；并发行为 ns/emit，fan-out 行为 ns/sink callback。

## 3. 三平台 n=15

| 场景 | Windows | WSL2 | M4 utility | 结论 |
|---|---:|---:|---:|---|
| `HYLE_LOG_INFO` runtime gated | 2.160 | 0.910 | 1.100 | O(1) 单次 relaxed gate；不格式化 |
| native preformatted view，1 sink | 42.070 | 38.970 | 23.910 | exact byte view + immutable snapshot + 安全 active ref |
| structured，4 fields | 38.450 | 34.500 | 23.560 | 直接 record 与旧证书同档；字段仍 borrowed |
| `std::format` 宏，1 sink | 89.190 | 87.260 | 63.210 | 宏已从两次 runtime gate 减为一次 |
| C NUL `hyle_obs_log` + Status | 44.610 | 40.230 | 26.790 | 含 `strlen` convenience 与 checked Status seam |
| C exact `hyle_obs_log_n` + Status | 44.430 | 39.790 | 27.090 | slice 不扫描；短 literal 点值与 NUL 行同档 |
| fan-out 1 sink | 41.680 | 38.200 | 24.340 | 与 normal view 交叉核对 |
| fan-out 4 sinks，per sink | 18.790 | 18.600 | 8.170 | 固定 snapshot 成本被摊薄 |
| fan-out 16 sinks，per sink | 13.140 | 13.640 | 4.280 | 总工作仍严格 O(S) |
| 2 producer，per emit | 108.800 | 89.040 | 79.390 | 无 producer mutex；sink atomic counter 有共享争用 |
| 8 producer，per emit | 93.690 | 62.950 | 124.390 | 平台 cache/调度不同；无复杂度退化 |

Windows normal/structured/format spread 为 9.5%/8.6%/6.3%；M4 加长 epoch 后为
0.5%/5.1%/4.3%，fan-out 1/4/16 为 3.3%/6.4%/4.9%。WSL 受宿主负载影响仍较明显，normal
spread 23.6%，所以它只支持“同历史档/无 material regression”，不拿单个最好点作胜负。
M4 runtime gate 的绝对值极小，1.04–1.74 ns 对应 63.6% spread；复杂度/codegen 成立，点值
只记中位数。QoS 不是物理 P/E 核绑定，表保留全部独立进程样本的中位数，不挑最好点。

编译期 floor 另由 consumer test 证明参数不求值；Windows `dumpbin /imports`、WSL
`nm -uC`、M4 `nm -u` 均未找到被裁剪 translation unit 的 obs 未定义符号。它是 O(0)
source/codegen 合同，不需要制造一行“0.00 ns”benchmark。

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

因此该轮严格结论是：

- gate O(1)，emit/flush O(S)，mutation O(S≤64)，复杂度与成熟同步 sink registry 同档；
- 正常 emit 无分配、无 mutation mutex，多 producer A/B 已关闭旧架构争用；
- exact-length message 将 slice 从 O(M+S) 收回 O(S)；短 NUL convenience 与 `_n` 点值同档，
  没有为 benchmark 删除边界检查；
- structured 直接 record 与 2026-07-25 历史证据三平台同档，fan-out/multi-producer 未见 material
  regression；view convenience 的额外 pointer+length 是保留内嵌 NUL/避免 slice copy 的产品合同；
- 暂无一个与 HYLE **相同 metadata、ownership、quiescent-unregister** 合同的成熟 peer 数字，
  所以不伪造全胜表；该轮只支持“没有已知复杂度或正常用法性能缺口”，不能认证跨合同
  的倍数胜利。

若真实消费者以后要求 category routing、per-sink filtering 或异步队列，必须先定义同合同
workload 再重开；不能为了 benchmark 再向每条 record 添加常态元数据。
