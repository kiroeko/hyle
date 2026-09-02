# container_intrusive 性能资格

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

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

## v3 历史产品边界

公开组件提供两种 caller-owned、stable-address、member-hook、non-owning 双向链表：

- `IntrusiveListHook` / `IntrusiveList<T, &T::hook>`：auto-unlink，linked node 可以先于 list retirement；
- `SafeIntrusiveListHook` / `SafeIntrusiveList<T, &T::hook>`：safe-link，membership 必须由 list 显式改变。

两种 hook 与 list 各为两个 pointer，不保存 count、owner、allocator、lock 或 dispatch；常态操作零分配。
两端插入/弹出、已知 node remove、`iterator_to`、单项/整链/range splice、append、list move/swap 和
typed auto membership transfer 均为 O(1)；遍历、clear、callback drain 与 owner teardown 为 O(n)。

强 peer 固定为 Boost.Intrusive commit `df8eeb90846648464e3c8e378e312f0cec772e83`、tree
`09d7782250845a1a2b6d2da7c6b136ce0a625b67`：

```text
boost::intrusive::list<
  Node,
  member_hook<Node, list_member_hook<link_mode<auto_unlink|safe_link>>, &Node::hook>,
  constant_time_size<false>>
```

`std::list` 的 ownership、allocation 与 footprint 不同，不进入胜负。Boost 的 base/function/offset hook、
完整 policy matrix 也不属于本模块合同，不能用 API 数量制造输赢。

## v3 历史 workload

全部 workload 使用 N=4096 的预分配稳定节点；allocation 和输入构造在 epoch 外。它们是正常消费者
形状的 deterministic replay，不冒充应用 trace。

| Policy / workload | 两侧对称计费 | 复杂度 |
|---|---|---|
| auto + safe FIFO batch drain | refill；HYLE `consume_all(noexcept callback)` 对 Boost `clear_and_dispose(noexcept callback)`；逐 node callback、observer、checksum | O(n) |
| auto + safe arbitrary cancel | refill；固定非顺序的已知 node remove/erase；checksum | 每 node O(1)，整批 O(n) |
| auto + safe traversal | 已链接节点只读扫描与 checksum | O(n) |
| auto + safe list-owner teardown | refill；list retirement 逐项 reset membership；前后 opaque storage observation | O(n) |
| auto linked-node retirement | raw storage construct/link；固定顺序 node 析构与 auto-unlink；前后 opaque storage observation | 每 node O(1)，整批 O(n) |

owner teardown 和 node retirement 是不同 lifetime 合同，不能合并。lifetime workload 的实际 benchmark
object 还经过 no-IPO/object-code gate，防止中间 relink/reset loop 被 dead-store elimination 消掉。

## v3 W1-WIN P1 结果

候选和合同 identity：

- HYLE commit `b5816b810d536f2041520e4b370c1e17d3ac7a4a`，tree
  `3bf043c40dbb5e4b49aba563f23f71b3c2d7caf3`，clean；
- contract `bench/contracts/container-intrusive-p1-v1.json`，SHA-256
  `5a0f4fb64811616a472988fa6480ff2b61cd5a9593e88cfaa522e345f4e0f2a8`；
- W1-WIN，MSVC `19.50.35728.0`，Windows SDK `10.0.26100.0`，x64 Release/static、portable ISA、
  LTO off；fresh build；
- `closure-perf-v1`，5% non-inferiority margin；primary/placement × candidate/peer execution order ×
  candidate/peer allocation order，共 8 strata；
- n=15 只因冻结停止规则允许的统计功效扩展而触发一轮完整 fresh n=31；两阶段不合并，最终表只用
  n=31 的独立进程 paired samples。

`advantage_ratio > 1` 始终有利于 HYLE。末列是 8 strata 的 process-median ratio 范围，只用于展示
位置/顺序敏感度；裁决使用每个 stratum 的 distribution-free median CI，而不是该范围本身。

| Workload | Combined exact | 5% non-inferiority | Strata median ratio range |
|---|---|---|---|
| auto-fifo-4096 | PARITY | PASS | 1.0088–1.0220 |
| auto-cancel-4096 | INCONCLUSIVE | PASS | 0.9903–1.0965 |
| auto-traversal-4096 | PARITY | PASS | 0.9976–1.0034 |
| auto-owner-teardown-4096 | PARITY | PASS | 0.9949–1.0022 |
| auto-node-retirement-4096 | INCONCLUSIVE | PASS | 1.0311–1.0709 |
| safe-fifo-4096 | PARITY | PASS | 1.0024–1.0222 |
| safe-cancel-4096 | INCONCLUSIVE | PASS | 0.9954–1.1011 |
| safe-traversal-4096 | PARITY | PASS | 0.9991–1.0045 |
| safe-owner-teardown-4096 | PARITY | PASS | 0.9968–1.0009 |

结论是 9/9 required workload 的全部 8 strata 均通过独立 5% non-inferiority gate，因此本候选的
W1-WIN P1 engineering performance 为 `PASS`。六项 exact 为 `PARITY`；cancel 与 auto-retirement 的
不同 strata 同时出现 parity、win 或跨 exact decision boundary，故三项 combined exact 诚实写
`INCONCLUSIVE`。`INCONCLUSIVE` 不是输，也不能包装成赢；这里成立的命题只是冻结合同上的非劣。

全部资格控制通过：A/A 两顺序落入 equivalence band；5% 与 10% known-delta 均被正确识别且单调；
timer floor、primary/placement、artifact identity 和 build contract 均为 `PASS`。最高 efficiency-class
logical CPU 2、above-normal child policy、single-thread environment、采样 mutex、源/peer/host/build/
artifact hashes 在采样前后保持稳定。最终 manifest 记录 2,232 observations、460 raw child logs 和
6 个构建 artifacts。

## v3 历史可重放证据

当前 engineering evidence 位于：

```text
C:\hyle-scratch\container-intrusive-v3\perf-qualify-b5816b8-output\manifest.json
```

manifest SHA-256 是
`3f081670022973a1b07ea6b4c04ab93b8d8fcbad61feef8bc6c584f30a10bd70`；summary SHA-256 是
`f7c0b557fe81009c286f0f97b56d7b1dab7d274bd546efe08fd46d18e9db6fb4`；observations SHA-256 是
`a6d1af6b796f586d89aded4afd89d792b29e66769bd15daa7728758cc55480d0`。当前兼容性检查命令：

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/container-intrusive-p1-v1.json `
  --manifest C:\hyle-scratch\container-intrusive-v3\perf-qualify-b5816b8-output\manifest.json
```

该包在 v3 轮当时的机械结果为 `qualification=QUALIFIED`、`paths_verified=true`；后来加入
consumer-key policy 的 CLI 还曾把它解释为 `product_gate_policy=null`、
`current_certification_policy_eligible=false`、`product_interpretation=INCONCLUSIVE`。

> **2026-08-29 回放兼容性：**提交 `2995286c` 为 Trace-enabled Obs 资格运行把
> `TRACY_ONLY_LOCALHOST` 加入全局 sanitizer，曾因原地扩展 v1 让 verifier 停止。冻结合同明确
> `HYLE_TRACE=OFF`，且 strata/controlled environment 未使用 Tracy 控制。环境策略版本化修复后，当前
> CLI 对原 manifest 返回 `environment_policy_compatibility=LEGACY_COMPATIBLE_TRACE_DISABLED`、
> `paths_verified=true`、`qualification=QUALIFIED`。旧 manifest 与 hash 均未修改；v3 verdict 仍只绑定
> 原 receipt，且旧合同仍因 `product_gate_policy=null` 不能续签当前 Formal。规则见
> [统一性能仪器](../../../developer/performance/instrument-v1.md)。

这不撤销 v3 当时签署的 engineering closure，也不能把旧合同按当前 consumer-key policy 续签或送入
Formal。若 live bundle 丢失或 owner 要求 refresh，必须冻结带显式 `gate_role`、consumer provenance、
ownership 和 material contribution 的新 contract ID，并在兼容工具就位后重跑。Formal 还必须保留可
独立获取的签名 artifact bundle，不能把这些绝对路径或 Windows 一机证据改名为发布证书。

## 正确性与结构证据边界

纳秒测量不证明复杂度或 lifetime。对应的独立门包括：

- compile-time 两 pointer layout/align contract；
- auto/safe hot operation、safe normal destructor、owner teardown 与 node retirement 的 codegen/object gate；
- deterministic edge matrix、same-list range splice 和 fixed-seed randomized differential；
- Windows MSVC/clang-cl、WSL GCC 14 ASan/UBSan、macOS AppleClang correctness；
- 独立 intrusive 与直接 `time` 消费者的 source、relocated static/shared package smoke。

这些结果登记在 v3 closure task；它们不能由上表点值替代，上表也不反向改变产品 API、metadata 或
算法形状。

## 历史证据为何不再展开

v2 使用 `front/pop_front` FIFO loop、旧 3% margin、旧 ratio direction、专用 runner、dirty candidate
和不完整 controls。其数表、旧本机命令和 Formal 草案已从当前页面删除，避免误读；可追溯事实仍在
[v2 closure task](../../tasks/container_intrusive/container-intrusive-v2-closure.md)。
`scripts/run-container-intrusive-bench-evidence.py` 仅保留为 legacy module diagnostic；统一
`hyle-perf.py` 仍是唯一仪器，但 frozen `container-intrusive-p1-v1.json` 现在也只是 legacy verify 入口，
不是新 qualification/Formal 入口。任何续签使用新 contract ID；任何 v2 点值都不得与 v3 样本合并或
用于当前结论。

## 消费者边界与重开

`time` 的 timer queue 直接消费 safe 双链；Sched cancellation 直接消费 safe hlist。scheduler waiter、
IOCP completion/inflight 与 pool free-list 的 storage reuse、publication 或 state-machine 合同不同，
继续保留特化实现。TimerQueue 或 scheduler 的整体性能属于各自模块，不能用来塑造本链表 API，也不能
用本页替它们签署性能。

只有以下事实重开本轮：真实 consumer 的 material regression；成熟同合同 peer 证明更优复杂度；
算法/API/layout/compiler 或 delivery shape 变化；bundle 丢失或 owner 需要用当前身份续签工程证据。
只有 verifier 策略版本漂移、但尚无显式兼容规则证明 frozen artifact 不受影响时，才属于方法兼容缺口，
也不单独重开产品。LOC、API
数量、单次点值、稳定亚纳秒差或某一个 placement/order 翻转都不能单独驱动产品形状。
