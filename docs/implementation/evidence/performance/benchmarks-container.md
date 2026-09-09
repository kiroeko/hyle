# Container 工程性能证据


## Container v6 终态候选

当前证据由两个预冻结合同组成，二者都从干净不可变候选运行，没有在看到结果后改变 workload、peer、
margin、顺序或样本数：

| factor / workload | 同合同 peer | 当前结果 | 可声明边界 |
|---|---|---|---|
| `container` / entry+weight LRU mixed access | Abseil `38f8c1535de9ebc12c6340e5c1c8ebd72afde560` adapter；同 hash、allocator、recoverable OOM、rollback、typed result、per-entry/total weight 与 entry/weight limits | 8/8 strata `WIN`；median advantage `1.2223–1.2728`；8/8 NI `PASS` | 冻结 4096-entry model-cache 形状胜出，不外推所有 cache 或 hash workload |
| `container_text` / safe UTF-8→UTF-16 caller buffer | simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b`；同 provider、validation-with-errors、exact sizing、endian-correct valid conversion、typed status/offset/count 与 caller buffer | 8/8 `PARITY`、8/8 NI `PASS`；median `0.9863–1.0102`；7/8 CI 跨 1，余下一项只证明 HYLE 小幅有利 | 同语义正常路径未证明固定劣化，不声明普遍更快 |
| `container_concurrent` / read-mostly membership | oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f`；同 hash/equality、system allocator、8 worker、分布、查询量、生命周期与 checksum | 双顺序 `PARITY`、NI `PASS`；median `0.9882/0.9777`，CI 均跨 1 | 冻结 membership 形状 5% 非劣，不声明 exact 胜出 |
| `container_queue` / unbounded typed MPMC handoff | 同一 oneTBB revision；同 allocator、recoverable OOM/retry、producer/consumer participation、setup/drain、payload 与 FIFO conservation | 双顺序 `WIN`、NI `PASS`；median `1.2888/1.2614` | 冻结 4P4C handoff 形状胜出，不外推其他 queue policy |

单线程/Unicode 终态 manifest 位于
`C:\public\project-hyle\_validation\container-v6\qual-single-v7-terminal-evidence\manifest.json`，SHA-256
`c4793256dfe376d82feb4beeb3b5d9789f90666bd9c981bd4a9323d108f4ea5a`；contract SHA-256
`3009a1294fdc7004b365e8af79c7a5d13f237e9ffaa540c722f105cc7b71d5ef`。它绑定
`container_text` commit `3ba50461ec7c8f387f4d953f1e499a1f6d449870`，qualification 与 performance gate 均
`PASS`，n=15 已给出终局裁决而未升级样本。

并发/queue manifest 位于
`C:\public\project-hyle\_validation\container-v6\qual-concurrent-v3-evidence\manifest.json`，SHA-256
`f16b135ff6a1b0974055ae63940e88df44f2ffe44c71841d6417930eb06eb196`；contract SHA-256
`0af5617ca1c4b978a8ad65ace62764efc2e792642939465110796c333f4e6389`，绑定 commit
`111c28eab99daff6257427473a2a1b11ac1cfd0c`。`container`、`container_concurrent` 与
`container_queue` 在 `111c28ea..3ba50461` 间没有产品字节变化；因此单线程 LRU 与并发结果可按 factor
identity 复用，不把 `container_text` 的后续修正错误归给其他 factor。

两合同的 A/A、5%/10% known delta、timer floor、placement/执行顺序、artifact identity 与未计时
semantic/resource oracle 均通过。这里的 `PASS/WIN` 只回答表中冻结的同合同 consumer 形状；不同
allocator、失败语义、错误结果或所有权的默认路线必须另立 `NATURAL_ROUTE` 合同，不能混入这些结论。

## 被替代候选

本轮更早候选的错误比较、撤销结论和纠正过程只属于历史任务；当前树不再维护其数表。需要审计时从 Git 恢复对应 revision，不能把旧结果与本页终态样本合并。
