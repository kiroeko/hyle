# container_intrusive 性能资格


> 本页是工程性能证据，不声明模块状态；唯一权威状态见
> [`module-status.json`](../../module-status.json)。当前 v4 candidate 3 在 W1-WIN 的四个冻结正常
> consumer shape 上取得 5% material margin 的 P1 engineering `PASS`。这不是跨平台发布证书，也
> 不是对所有 intrusive container、规模或编译器的普遍 SOTA 声明；`formal-release-v1` 仍为
> `PENDING`。

设计与消费边界分别见
[container-intrusive design](../../../architecture/container-intrusive-design.md)、
[用户指南](../../../user/container-intrusive.md) 和
[v4 closure task](../../tasks/container_intrusive/container-intrusive-v4-capability-extension.md)。统一仪器规则见
[instrument v1](../../../developer/performance/instrument-v1.md)。

## v4 当前工程命题

v4 没有通过给默认双链增加常态元数据来追 benchmark，而是把四种不同成本合同拆为可选择的类型：

- `CountedIntrusiveList` 只为需要 mutation 后 O(1) `size()` 的 owner 增加一个 list 级 count word；
- `IntrusiveForwardList` 为 arena/deferred-work chain 保持一 pointer node，并用 list 级 cached tail 支持
  O(1) append；
- `IntrusiveHList` / `SafeIntrusiveHList` 用一 pointer head 与 `next + prev_next` node 支持已知节点 O(1)
  unlink/replace；
- 双链与 forward-list 的稳定 `sort/merge` 等 sequence algorithm 保持 caller-owned、零分配，排序复杂度
  O(n log n)。

默认 auto/safe 双链仍保持 v3 的两 pointer hook/list、零分配、无 count/owner/lock/dispatch 合同；它的
旧 workload 不因本轮新增能力而失效，但不能替新增能力签性能。

### v4 冻结 workload 与 peer

全部 workload 使用 4096 个预分配、layout-matched 的 stable-address node；分配、输入生成和 permutation
均在 epoch 外。每个 workload 都由 public package sample 或真实 Sched consumer 反向追踪：

| Workload | 正常消费形状 | 同合同 peer / 裁决边界 |
|---|---|---|
| `counted-membership-4096` | 每次插入/任意已知节点删除后读取 O(1) count | Boost.Intrusive `constant_time_size` safe list |
| `forward-arena-4096` | cached-tail append、遍历、显式 detach-and-consume | Boost.Intrusive cached-tail safe slist |
| `hlist-callbacks-4096` | callback register、已知节点 unregister、detach-before-fire | v4 前 Sched 手写 hlist；只证明真实 consumer 回归，不作外部 SOTA 声明 |
| `sequence-sort-merge-4096` | 两个 non-owning stable sequence 排序、合并、遍历 | Boost.Intrusive safe list `sort + merge` |

Boost 固定为 commit `78505eab9b203e3e26e83893d0685f160133fa0f`；Sched baseline 固定为
commit `b4915574020cc7da857d2871c545a3489c75288b`。两侧的 ownership、node/list footprint、操作序列、
callback/checksum 与 charged phase 对齐；`std::list` 等 owning/allocation 形状不进入比较。

### v4 W1-WIN P1 结果

候选和合同 identity：

- HYLE commit `58431ddbef6f54f9d4275b0fea2d317cba00b9f3`，tree
  `8df2b7d7806918d03b5bb5c31611adb196f30c7f`，clean；
- contract `bench/contracts/container-intrusive-p1-v2.json`，SHA-256
  `c10b5e062cb944f06ccb880a7ac1ff454f810cf28906d2b3afad84a0a3efe415`；
- W1-WIN，MSVC `19.50.35728`，Windows SDK `10.0.26100`，x64 Release/static、portable ISA、LTO off，
  fresh build；
- primary/placement × candidate/peer execution order × candidate/peer allocation order，共 8 strata；
- n=15 只因全部 control 通过且至少一个产品门仅因统计功效未决，按冻结规则升级到一轮 fresh n=31；
  最终表只使用 n=31 的独立进程 paired samples。

`advantage_ratio > 1` 有利于 HYLE。ratio 范围仅显示 placement/order 敏感度；裁决使用各 stratum 的
distribution-free median CI。

| Workload | Combined exact | 5% non-inferiority | 8-strata median ratio range |
|---|---|---|---|
| `counted-membership-4096` | `INCONCLUSIVE` | `PASS` | 1.0118–1.0539 |
| `forward-arena-4096` | `PARITY` | `PASS` | 0.9823–1.0089 |
| `hlist-callbacks-4096` | `WIN` | `PASS` | 1.2206–1.2770 |
| `sequence-sort-merge-4096` | `WIN` | `PASS` | 1.0881–1.1188 |

四项 workload 的全部 8 strata 均通过 5% non-inferiority gate，因此 v4 W1-WIN P1 engineering verdict
为 `PASS`。counted 的点估计通常有利于 HYLE，但 combined exact CI 仍跨 decision boundary，所以只能写
`INCONCLUSIVE`；forward 是 `PARITY`。hlist 的 22.1%–27.7% 胜出只相对旧 Sched 手写实现，sequence
的 8.8%–11.9% 胜出只相对本合同的 Boost shape；两者都不能外推为普遍 SOTA。

A/A 双顺序、5%/10% known-delta、timer floor、placement、artifact identity 和 build contract 全部
`PASS`。最终 bundle 含 992 observations、460 raw logs 和 6 个 build artifacts；verifier 返回
`qualification=QUALIFIED`、`paths_verified=true`、`product_gate_policy=consumer-evidence-v1`、
`provenance_compatibility=EXACT`、`current_certification_policy_eligible=true`。

### v4 可重放证据

当前本机 retained bundle 位于：

```text
C:\public\project-hyle\_validation\container-intrusive-v4-perf-evidence-58431ddb\manifest.json
```

manifest SHA-256 为
`EB3BF0A4BFED3CB39BF61A239487963FE6D889B13721CC5D996E7057F8DD4283`；summary SHA-256 为
`b9ca233cf0a3be1d50c469e545997c7b871487a63a5b593a1fa277a14294e369`；observations SHA-256 为
`5fa6d4cad79557815cb6ee9a9ad4ccf9524a13d9d23ce556b347e19b5c249077`。验证命令：

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/container-intrusive-p1-v2.json `
  --manifest C:\public\project-hyle\_validation\container-intrusive-v4-perf-evidence-58431ddb\manifest.json
```

本机绝对路径是 retained engineering receipt，不是发布接口；bundle 丢失时不能只凭本文数表续签新
candidate 或 Formal。

## 边界与历史

P1 点值不替代 layout、复杂度、生命周期、三平台 correctness 和三交付门；这些事实见当前 v4 record。v3 及更早专用 runner、旧 margin 与旧候选已从当前页移除，可由 Git 恢复，不能续签当前证据。只有真实 consumer 的 material 回归、算法/API/layout/交付变化，或成熟同合同 peer 的更好复杂度才触发新轮。
