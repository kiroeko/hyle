# Platform 当前工程性能证据


> 当前状态与精确 closure record 只看 [`module-status.json`](../../module-status.json)。本页同时服务
> `platform`、`platform_thread` 和 `platform_fs` 三个内部 factor；共享报告不把它们合成一个分数，
> 也不改变普通用户只选择 Platform 产品 façade 的心智。

## 当前裁决

| factor | tier / 当前 candidate | 当前工程结论 |
|---|---|---|
| `platform` | P1；`3c1ef2392d1f05c79edfe1c5d4ea5558bd917e3c` / tree `63b58d1051d86c4aecb1bf736ae84265e359a5f5` | host-planning 4/4 strata 通过 5% non-inferiority；`PASS` |
| `platform_thread` | P0；Parker 产品 bytes、状态机与 memory order 未变 | integrated-source/provenance、复杂度、code shape、三平台和三交付结构 `PASS`；没有新 timing claim |
| `platform_fs` | P1；`c39df6f94c636ab8fa94cb700dd263763a42d680` / tree `37889346564f3a5cdc895f26c9d610883076ba02` | warm 4 MiB file lifecycle 8/8 strata 为 `PARITY`，5% NI 全过；`PASS` |

三者 Formal 均仍为 `PENDING`。这些结论不表示全部 OS query、所有拓扑/文件系统、所有平台或端到端
推理 SOTA。

## `platform`：host-planning P1

冻结合同 [`platform-host-p1-v3.json`](../../../../bench/contracts/platform-host-p1-v3.json) 比较完整的
process-visible CPU/package/NUMA/cache 与 host-memory planning snapshot。candidate 和 direct Win32 peer
都返回同一 HYLE value record 与 `Result` 失败形状；CPU-set、topology、physical/commit memory query、
scratch setup/teardown 和组合均计费。oneTBB
`e00c68dcac0e83947c9b54f6d5304c0616c9639f` 锁定成熟 lineage，计时 peer 是等价 OS adapter，因而不把
更少字段或不处理失败的路线当同合同。

唯一 fresh W1-WIN qualification 在 n=15 通过 A/A、5%/10% known-delta、timer-floor、placement、
artifact/build/host/input/provenance controls。四个 median advantage ratio 为
`1.015725 / 1.025350 / 1.024989 / 1.037168`，全部 5% NI `PASS`；exact CI 没有形成 material WIN，
所以只签 non-inferiority。

bundle 位于
`C:\public\project-hyle\_validation\platform-v11-perf-control-resume\perf-evidence`；contract-lock、
manifest、summary SHA-256 分别为
`2714905D56EA9CCED7132A5EA55FE44C2925AACEF45B14A774EEDE4EF987C420`、
`3BA2589D00AC079B6A9366F7D12F49CAF1DC95E6199CB33895533EE5A9E8F0F8`、
`353ED5BFF2C576DC6E35B019ABAB863C4992509781DA8729F79473B5E93D8764`；deep verify 为
`QUALIFIED / EXACT / CURRENT / paths_verified`。完整终局事实见
[`platform-v11 task`](../../tasks/platform/platform-v11-perf-control-resume.md)。

## `platform_thread`：P0 结构门

Parker 是 Rust std 1.94.1 指定 revision 的 integrated source；HYLE 拥有公开 thread/wait 语义和三个 OS
backend，锁文件、MIT notice、安装许可证与 source hash 共同约束所借机制。本轮没有修改 Parker 算法、
layout 或正常路径，因此 timing peer comparison 为 `NOT_APPLICABLE`。当前证据只签：状态转移与
memory-order 同源、无新增 allocation/lock/syscall/metadata，三平台 correctness 与 source/static/shared
消费通过。完整事实见
[`platform v8 task`](../../tasks/platform/platform-v8-os-resource-contract.md)。

## `platform_fs`：warm-file P1

冻结合同 [`platform-fs-p1-v2.json`](../../../../bench/contracts/platform-fs-p1-v2.json) 对一个 warmed
4 MiB regular file 执行相同 open、完整 read/checksum 和 close；两侧各自的 caller buffer allocation order
在计时外受控，failure、ownership、sharing flags、完整读取循环与 teardown 相同。Folly
`65749da412d92fe27d87ae45a8ef670720712858` 锁定 `File` lineage，计时 peer 是同一 OS ABI 的 faithful
adapter。

唯一 fresh W1-WIN bundle 按预声明规则扩到 31 个独立进程；A/A、5%/10% known-delta、timer-floor、
placement 和全部 artifact/build/semantic controls 通过。八个 ratio 为
`0.996402 / 0.999517 / 1.012026 / 0.998621 / 0.987503 / 1.005656 / 0.994822 / 1.002090`；
方向随执行、allocation 和 placement 翻转，所有 CI 包含 1，故裁决为 8/8 `PARITY` 与 5% NI `PASS`，
没有固定方向产品劣化。

bundle 位于
`C:\public\project-hyle\_validation\platform-fs-v2-resume-evidence-r1`；
contract-lock、manifest、summary、observations、controls SHA-256 分别为
`4324C5FD0A97E9B63D2868290B9988A175DBFA344254A52F0F2A79C40F50BE06`、
`6BC5AE79776C0A4B561326297F0910E00BB3C44CD9D405368369A1C18E655C14`、
`6AE27668D8DC940535E7F1495525D21F7AC67346E14DC26BFCDFAE4E551E9A19`、
`C71F9EFD171F99092FF71EB97602579FAC076E4BF5DFADC6B0C5CE5EEE0B35A1`、
`D994AF76428673F48EDD2E92610E478265B4AB9264725882E22B54D1594FC4F4`。完整终局事实见
[`platform-fs v2 task`](../../tasks/platform_fs/platform-fs-v2-control-resume.md)。

## P0 能力与声明边界

CPU/cache/frequency/NUMA discovery、node-memory query、cgroup/job/process limits、process control、
VM/dynlib、opened-file metadata/duplicate/random access 等未进入上述两个 P1 workload 的能力，只签各自
复杂度、系统调用数、allocation/lock/metadata、correctness 和三交付结构门。OS 没有公开可靠事实时
返回 absent/unsupported，不能用猜测、synthetic 数据或另一个平台的接口补绿。

旧 v3--v10 receipt、失格 controls 和被替换 candidate 只从各自 task 与 Git 历史追溯；它们不再展开在
当前页，也不能与上述两个 current bundle 拼样或续签。

复验当前 bundle：

```powershell
python scripts/hyle-perf.py validate --contract bench/contracts/platform-host-p1-v3.json
python scripts/hyle-perf.py verify --contract bench/contracts/platform-host-p1-v3.json --manifest C:\public\project-hyle\_validation\platform-v11-perf-control-resume\perf-evidence\manifest.json
python scripts/hyle-perf.py validate --contract bench/contracts/platform-fs-p1-v2.json
python scripts/hyle-perf.py verify --contract bench/contracts/platform-fs-p1-v2.json --manifest C:\public\project-hyle\_validation\platform-fs-v2-resume-evidence-r1\manifest.json
```
