# sched 性能证据


> 当前产品状态只见 [`module-status.json`](../../module-status.json)。当前工程结论以紧接的 sched-v5
> 小节为准；后续各轮数据只解释对应候选，不能拼成全操作、跨平台或 Formal SOTA。

## sched-v5 IO completion 工程证据（2026-09-08，`PASS`）

产品 candidate 为 `2f9ecc8168b4d303a17ae4fcad385b8150cdd072` / tree
`212a5c0305d9a756cdea91b2ed1270319264db8a`。本轮只改变 attached IO driver 报告 in-flight、run queue
为空时的 idle transition：worker 不再先执行不 poll completion 的固定 spin/yield search，而直接进入
已有的 sleeper publication、queue recheck 与可被 foreign wake 打断的 proactor wait。无 IO hook 或无
in-flight IO 时仍走既有 searcher 路线；公共 API、layout、依赖、allocation、lock、metadata 与 syscall
种类均未变化。

冻结的真实 consumer gate 是 [`io-tls-p1-v1`](../../../../bench/contracts/io-tls-p1-v1.json)。W1-WSL、
GCC 14.2、Release/static、portable ISA、OpenSSL 3.0.13；clean verification source 为
`b38aa34cf75cd89b179f0ee32449bfeb4ff0cb59`，产品目录相对 candidate 无差异。对 Asio
`8806a6803cde7054c3049d3666d3ec36786568c5` 的双顺序 5% non-inferiority 四门全部 `PASS`：verified
handshake advantage `1.081/1.100`，512x256 B encrypted-stream advantage `1.069/1.075`；四行
superiority 都因 CI 跨 5% practical margin 而为 `INCONCLUSIVE`，不宣称确定胜出。精确表、合同边界、
bundle 与 hash 见 [`benchmarks-io`](benchmarks-io.md)。

这相对修复前合法暴露的 `0.448/0.446` 与 `0.0265/0.0259` 不是调 spin 常数或 benchmark 特化，而是
移除了每个低并发 IO 依赖链中约 160 us 的错误 completion 等待。A/A、known-delta、timer floor、
artifact/build/provenance 与 source/peer/input/host stability 全部通过。三平台 Sched/IO/TLS correctness、
WSL ASan/UBSan/TSan 和 Windows installed shared/static consumer 也通过。该 `PASS` 只裁决本轮受影响的
IO completion 路线；无 IO 批执行的既有证据仍由 sched-v4 小节承载，不借本次 TLS workload 宣称全
Scheduler SOTA。

## sched-v4 最终工程证据（2026-09-07，`PASS`）

产品 candidate 为 `dbc9634455176450fb86d68b373c6cc4a4156f24` / tree
`5e39d8ab76b6da83e7dd267584f9592c59930532`。资格化使用 clean source
`e91fc36b52075f8d36a5d4cb2f375e4c967082b1`；candidate 后只增加直接消费者/benchmark fixture 修正，
`src/sched` 与 `include/hyle/sched` 相对 candidate 无差异。W1-WIN、MSVC 19.50、Release/static、四个固定
physical cores，统一仪器的 A/A、5%/10% known-delta、timer floor、artifact/build/provenance 与独立
verifier 全部通过。

同合同产品门使用 `sched-batch-p1-v4`（SHA-256
`FC5DBFA551BFE8F6DA778E7AAF836D6ED0CE19092BD65C970CFF519F1339AD58`）：16 批 × 128 个
non-suspending 1024-element tile，四个 provider workers、caller-owned slots、一个 batch completion；
外部角色只提交/等待。advantage 为 HYLE/peer，31 个独立进程 paired samples，置信水平 98.93%：

| peer / 顺序 | HYLE / peer 中位吞吐（M item/s） | advantage / CI | exact | 5% NI |
|---|---:|---:|---|---|
| Marl / HYLE first | 2.055 / 1.800 | 1.153 / [1.093, 1.200] | `WIN` | `PASS` |
| Marl / peer first | 2.071 / 1.810 | 1.129 / [1.112, 1.196] | `WIN` | `PASS` |
| oneTBB / HYLE first | 2.040 / 1.938 | 1.049 / [0.985, 1.128] | `INCONCLUSIVE` | `PASS` |
| oneTBB / peer first | 2.062 / 1.945 | 1.050 / [1.017, 1.092] | `INCONCLUSIVE` | `PASS` |

holdout manifest/summary/controls SHA-256 分别为
`012A7B0569C73823351A811F912233ACF051C3BDC5FFA0045A671E78EF1C23CE`、
`78815254D646D68CCA41441452A956331C2F2F354ED5B72806DAF099B5DE5FD8`、
`EB19961E902A1A566231BB6D5989502EF4A98C723366DBDF5B90A7FA97310E2A`；原始 bundle 位于
`C:\hyle-scratch\sched-v4\p1-holdout-evidence-e91fc36b`。Marl/oneTBB revision 分别为
`b8406ab0a82524170ead6dd6c371b579bbc28b3e` / `a8216b95202f3c55f9749870f4f8cfa4df0b32e3`。

控制合同修正还回放了旧 `sched-p1-v1`。其锁定副本 SHA-256 为
`DAE15DB50195B0906D741BB8B35DF623E2D0F5D18E129351B60C3BCFE132CA52`：

| workload | 当前结果 | 语义裁决 |
|---|---|---|
| request fan-out/collect | advantage 1.617 / 1.632，双顺序 `WIN`、5% NI `PASS` | `SAME_CONTRACT` |
| completion wake | 1.024 / 1.015，双顺序 `PARITY`、5% NI `PASS` | `SAME_CONTRACT` |
| typed structured spawn/join | 0.927 / 0.927，旧机器 aggregate 报 `FAIL` | `NATURAL_ROUTE / NOT_COMPARABLE`：HYLE 每项拥有 typed result、Status、cancel flag、JoinHandle 与 lifecycle cell；Marl 只写 caller-owned slot 并共享一个 batch WaitGroup |

最后一行是 sched-v1 红队已经记录过的苹果/橘子，不是同合同 SOTA 门。原始 manifest 的 `FAIL` 不被
抹除或改写；manifest/summary SHA-256 为
`1B2A6FF939BF7F2D9A8CDAE4E1A50822F2F44C3604D7BBB919FFF667BB4A4214` /
`C477D9180384A9F920BD5F75F9AC700958D4B2373610C9DC4A38E96C4E7D271C`，bundle 位于
`C:\hyle-scratch\sched-v4\p1-primary-evidence-e91fc36b`。当前 contract/consumer manifest 已把该行改为
non-gating diagnostic，防止未来再次把不同 ownership 语义聚合成产品失败。

作为同产品方向复核，当前三条 HYLE absolute median 相对同机、同工具链、同 workload 的 sched-v1
qualified bundle，在两个执行顺序上的比值依次为：request `1.132/1.088`、typed structured
`0.987/1.074`、completion `1.015/1.009`；没有固定方向的 5% material regression。这是跨 bundle
diagnostic，不冒充新的 paired certificate；工程 `PASS` 由两条当前同合同 replay、P0 成本检查和上面的
Marl/oneTBB batch holdout 共同给出。结论只覆盖这些真实 source-build 形状，不声明全操作、跨平台或
Formal SOTA。

## 被替代证据

v1--v3 与更早操作级数表、失败候选和调参记录已被 v4 的普通 scheduler 合同及 v5 的 IO completion 增量取代。精确历史由 Git/raw evidence 保存；不得跨候选拼表或把不同 ownership 的旧行重新升级为同合同门。
