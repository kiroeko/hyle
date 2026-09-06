# Mem v6 host-storage 工程性能证据

<!-- PERFORMANCE EVIDENCE: CURRENT -->

> 当前机械状态只见 [`module-status.json`](../../module-status.json)。本页只绑定候选
> `9604bf4a33f13bd5f84ef78fedce55f305ac9d50` / tree
> `57adb69f185cf02d42ab7f220ebeccbb91170b1b` 与下述冻结任务；不把一个 host
> allocator 路线外推成 Mem 全能力、全尺寸、全线程或端到端推理的“SOTA”。

能力与 ownership 见 [Mem architecture](../../../architecture/mem-design.md)，用户选型见
[Mem guide](../../../user/mem.md)，完整收口事实见
[Mem v6 closure card](../../tasks/mem/mem-v6-comprehensive-rebuild.md)。

## 性能命题与苹果对橘子前置门

本轮 P1 正常任务是一个隔离 subsystem `Heap`：同一 owner 通过 runtime-erased handle 完成 64 个
24 B--3 KiB、16/32/64-byte alignment 的 allocate、端点 touch 和 reverse sized-free。peer 是同一
静态二进制、同一 mimalloc 3.5.1 上拥有独立 heap 的 `std::pmr::memory_resource`，并强制通过不可
去虚化的 base pointer 调用。两边 owner construct/warm-up 在计时外，所有成功 allocation、touch 与
free 都计费；failure、thread-safety、owner lifetime、alignment 和 allocator backend 相同。

因此该行是 `SAME_CONTRACT`，回答的是 HYLE runtime-erased owner/adapter 是否带来 material wrapper
损失。它不拿 system heap 对 arena、cached pool 对 general allocator、bulk release 对 individual free，
也不拿更少检查/更弱失败语义的路线来排名。冻结合同是
[`mem-v6-host-allocation-p1-v1.json`](../../../../bench/contracts/mem-v6-host-allocation-p1-v1.json)。

`SystemMemorySnapshot`、large-page arena reserve、BufferPool policy cleanup 和 LifetimePlanner 没有被
伪造为纳秒任务：它们分别按公开 OS/vendor 事实、side-effect/resource contract 与既有复杂度作 P0
验收。新增冷查询不进入 allocation/free 热路，BufferPool 对象反而删除了一个 benchmark-shaped
公开 knob。

## W1-WIN 结果

W1-WIN、MSVC 19.50、Release、static source consumption、LTO off；15 个独立进程样本，双 execution
order、双 heap allocation order、primary/perturbed placement 共八个 strata。`advantage ratio > 1`
有利于 HYLE：

| Stratum | median advantage ratio | 99.26% CI | exact | 5% NI |
|---|---:|---:|---|---|
| primary / candidate-first / candidate-allocation-first | 1.0241 | [1.0087, 1.0365] | `PARITY` | `PASS` |
| primary / candidate-first / peer-allocation-first | 1.0110 | [0.9891, 1.0379] | `PARITY` | `PASS` |
| primary / peer-first / candidate-allocation-first | 1.0112 | [0.9867, 1.0373] | `PARITY` | `PASS` |
| primary / peer-first / peer-allocation-first | 1.0139 | [0.9930, 1.0346] | `PARITY` | `PASS` |
| perturbed / candidate-first / candidate-allocation-first | 1.0202 | [1.0002, 1.0411] | `PARITY` | `PASS` |
| perturbed / candidate-first / peer-allocation-first | 1.0068 | [0.9769, 1.0394] | `PARITY` | `PASS` |
| perturbed / peer-first / candidate-allocation-first | 1.0176 | [0.9955, 1.0406] | `PARITY` | `PASS` |
| perturbed / peer-first / peer-allocation-first | 1.0206 | [0.9811, 1.0312] | `PARITY` | `PASS` |

聚合 exact verdict 为 `PARITY`，8/8 的 5% non-inferiority 为 `PASS`，全部 controls 为 `PASS`。八个
点估计均朝 HYLE 方向，但只有约 0.68%--2.41%，所以结论止于：**该冻结正常 owner lifecycle 没有
material regression；没有签发普遍性能胜出。**

第一次允许的 evidence bundle 因 A/A 的一个执行顺序和 5% known-delta control 未收敛而是
`DIAGNOSTIC`；产品点值不能越过失格仪器形成 PASS。按 closure card 唯一一次重试得到上述 qualified
bundle，不修改产品、合同、样本数或 margin。

## 资格与不可变身份

- qualified bundle：
  `C:\public\project-hyle\_validation\mem-v6-host-perf-evidence-r2`；
- contract-lock / manifest / summary SHA-256：
  `297958D9E2F9F98D12370D99DAFA76B8D1E89D50D04F55F630F8563193093144` /
  `A7749296A3A772DB71F2534D01C8E38F2506830FCCDCE61F4014F6151C3B8EC0` /
  `5054BED366D60C2D7F1B8926D50FBD9BBEC7B2A6C4B7A6AFB3BD389144A2FA0F`；
- deep verifier：`QUALIFIED`，environment `CURRENT`，paths verified，source/provenance `EXACT`；
- rejected diagnostic r1 的 manifest / summary SHA-256：
  `EDFB52D1BF5714169DE897B3C65146059D1B8D04DC6846A62887BFA2BF8B42A6` /
  `03B46535A012AF9757C2BEAB8BDE20466A2E858A30838B69BB9796531F48C697`。

## 结构性能与边界

| family | frozen bound | 本轮结果 |
|---|---|---|
| system/isolated allocation | direct template 或一层显式 runtime erasure；无默认 stats/leak metadata | `PASS`；P1 如上 |
| arena/stack/pool/fixed heap | 各自声明 hard/soft、individual/bulk 与 concurrency；不按 benchmark 混为一种 owner | `PASS`；复杂度与资源测试 |
| Buffer/View/Composite | owner/view 分离，range 操作 O(1) 或 O(segments)，无 hidden staging | `PASS`；property/OOM/move tests |
| BufferPool | 有界 retention/best-fit；策略字段只表达用户资源预算 | `PASS`；删除内部 scan knob |
| LifetimePlanner | build `O((N+E) log N)`、metadata `O(N)`、query O(1) | `PASS`；stress/property/provider-count tests |
| pressure/large-page control | 显式冷查询/启动控制；unknown 与 partial side effect 可见 | `PASS`；不作 syscall 纳秒排名 |

本页不声称 allocator 比 mimalloc 更快——mimalloc 正是 HYLE 锁定的 private mechanism；也不覆盖远端
NUMA、显式 OS 大页成功率、设备 resource、端到端 tokens/s 或 Formal certificate。Formal 状态保持
`PENDING`。

## 复现

```powershell
python scripts/hyle-perf.py validate `
  --contract bench/contracts/mem-v6-host-allocation-p1-v1.json
python scripts/hyle-perf.py verify `
  --manifest C:\public\project-hyle\_validation\mem-v6-host-perf-evidence-r2\manifest.json `
  --contract bench/contracts/mem-v6-host-allocation-p1-v1.json
```
