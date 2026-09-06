# Platform v9 process-memory / mapped-range 工程性能证据

> 当前机械状态只见 [`module-status.json`](../../module-status.json)。本页只绑定候选
> `9604bf4a33f13bd5f84ef78fedce55f305ac9d50` / tree
> `57adb69f185cf02d42ab7f220ebeccbb91170b1b` 和 Platform v9 的定向重开；其它 Platform
> factor 的当前证据仍见 [`benchmarks-platform.md`](benchmarks-platform.md)。

新增 `process_memory_info()` 是显式冷查询：Windows/macOS 为有界 OS query，Linux 线性读取一次
`/proc/self/status`；它不缓存，不给 allocation/mapping 热路增加字段、锁、allocation 或 syscall。
`vm_advise/vm_lock/vm_unlock` 的实现和签名未改变，只把合同修正为可用于 owner 已验证的任意合法
mapped page range。因此这两项按 P0 系统事实、复杂度和 syscall route 验收，不发明 snapshot 纳秒排名。

## 未受影响 host holdout

P1 只复跑既有 host-planning normal journey，确认该候选没有旁路 material regression。candidate 与
oneTBB adapter 都完成同样的动态 available-CPU/topology 与 host-memory planning record，包含相同字段、
失败处理、setup 与 teardown；合同是
[`platform-host-p1-v3.json`](../../../../bench/contracts/platform-host-p1-v3.json)。

W1-WIN、MSVC 19.50、Release static、LTO off；15 个独立进程、双执行顺序与 placement：

| Stratum | median advantage ratio | 99.26% CI | exact | 5% NI |
|---|---:|---:|---|---|
| primary / candidate-first | 1.0272 | [1.0172, 1.0354] | `PARITY` | `PASS` |
| primary / peer-first | 1.0249 | [0.9999, 1.0407] | `PARITY` | `PASS` |
| perturbed / candidate-first | 1.0266 | [1.0086, 1.0554] | `INCONCLUSIVE` | `PASS` |
| perturbed / peer-first | 1.0212 | [1.0031, 1.0483] | `PARITY` | `PASS` |

4/4 的 5% NI 与全部 controls 为 `PASS`；三个 exact `PARITY`，一个因置信区间越过 5% 胜出边界而为
`INCONCLUSIVE`。点估计均有利于 HYLE约 2.12%--2.72%，但本页不据此签发 material win。

## 资格与边界

- evidence：`C:\public\project-hyle\_validation\platform-v9-host-perf-evidence-r1`；
- contract-lock / manifest / summary SHA-256：
  `2714905D56EA9CCED7132A5EA55FE44C2925AACEF45B14A774EEDE4EF987C420` /
  `03856C2DC1CE2C1E1B8292881611A40567E11EFDCA3FD00BD696E25B7D95878D` /
  `E22B8F15BAE064EC3E0EB23809A77027956C80F6E870F65D5C90864A8386A09A`；
- deep verifier：`QUALIFIED`，environment `CURRENT`，paths verified，source/provenance `EXACT`；
- 本证据不重签 CPU/cache/NUMA/fs/thread 全模块性能，也不把 OS query latency、文件映射 owner 或
  Mem fragment algorithm 归给 Platform；Formal 为 `PENDING`。

完整收口事实见
[`platform-v9-process-memory-mapping.md`](../../tasks/platform/platform-v9-process-memory-mapping.md)。
