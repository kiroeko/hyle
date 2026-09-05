# Frozen foundation 当前 consumer 来源复核（2026-09）

<!-- PERFORMANCE PROVENANCE REVIEW: NON_MATERIAL -->

本记录只裁决旧工程性能包引用的 consumer 文件，从 retained candidate 到联合复验源码
`27aa252fd66c6ae51e7a6b1dab5030bce6183436` / tree
`57e6be3c2cd3c42cbd6838e4d51b1e311a649661` 的整文件 SHA-256 漂移。它不修改旧 contract、manifest、
raw observation 或产品代码，也不把人工复核冒充 fresh qualification 或 Formal 证书。

复核逐项比较了 frozen semantic contract、ownership/lifetime、thread-safety/progress、material
contribution、charged phases、算法、工作量和 delivery。以下变化均未触及对应 workload；因此只允许
`ENGINEERING_REPLAY_ONLY / REVIEWED_NON_MATERIAL` 续签。后续若任一合同事实变化，必须建立新
contract 并重新 qualification，不能沿用本记录。

## 精确迁移

| contract / workload | path | frozen SHA-256 | current SHA-256 | 字节变化与裁决 |
|---|---|---|---|---|
| `error-source-consumer-v2` / `taskgraph-source-success-64` | `src/graph/taskgraph.cc` | `9adb3f42e9cf9fd11c4edf961f2d121fe06341f6b6417b79ee4c8ecb9a2078f5` | `81ac9b6a6637521343f2e387fb1f6afa3388feb9a3af4226c3e364d91be10b59` | 只把 allocator 公共词汇从 `Allocator` 改为同一轻量借用句柄 `AllocatorRef`；64-node evaluate 循环、Status 成功 carrier、检查和 trace 工作不变。`NON_MATERIAL`。 |
| `error-source-consumer-v2` / `taskgraph-source-owning-last-failure-64` | `src/graph/taskgraph.cc` | `9adb3f42e9cf9fd11c4edf961f2d121fe06341f6b6417b79ee4c8ecb9a2078f5` | `81ac9b6a6637521343f2e387fb1f6afa3388feb9a3af4226c3e364d91be10b59` | 同上；最后一个 owning error 的构造、移动、提前返回与销毁不变。`NON_MATERIAL`。 |
| `error-source-consumer-v2` / `parallel-source-first-error-handoff` | `src/graph/parallel_eval.cc` | `5a4888eab4a46f670f81f8a112587cc5c2f5b0872592b5b132f1a2e8ee0f7fb4` | `bdd1536724f5d29d2aa68e8b9f38cd3895a4c7d683f701605b6d9ca78288a92f` | allocator/arena 词汇迁移为 `AllocatorRef`/`Arena`；冻结的 Status outcome、record、first-error 与 waiter handoff 未改变。`NON_MATERIAL`。 |
| `error-source-consumer-v2` / `join-source-failed-owning-handoff` | `include/hyle/sched/join.h` | `be10fceda9516727094ef030454d95d3934eb13ebdf07aa6db2af6b60862e8b9` | `5abe87d1a4fb42c62a256c71fe05371064b2ed9f60d7c0373dc7347f992f9f7b` | Join storage 的 allocator 字段和参数仅改名为等价借用句柄；锁内 Status copy、Result move、所有权和析构序列不变。`NON_MATERIAL`。 |
| `container-intrusive-p1-v2` / `counted-membership-4096` | `samples/package/container/cpp/main.cc` | `12af08374ca2b304e160a9b1865ef188ce5341d92d5c78ff314579206cc9b1dd` | `47522da4f537d07f75dc82cfba42050f946126fcab1c4b84e4afcb00c1a0b1cd` | 文件末尾新增独立 `ConcurrentNodeHashMap` 示例；先前 intrusive counted journey 的节点、操作、顺序和退出码均未改。`NON_MATERIAL`。 |
| `container-intrusive-p1-v2` / `forward-arena-4096` | `samples/package/container/cpp/main.cc` | `12af08374ca2b304e160a9b1865ef188ce5341d92d5c78ff314579206cc9b1dd` | `47522da4f537d07f75dc82cfba42050f946126fcab1c4b84e4afcb00c1a0b1cd` | 同一文件只追加后续无关示例；cached-tail append/traverse/drain journey 不变。`NON_MATERIAL`。 |
| `container-intrusive-p1-v2` / `sequence-sort-merge-4096` | `samples/package/container/cpp/main.cc` | `12af08374ca2b304e160a9b1865ef188ce5341d92d5c78ff314579206cc9b1dd` | `47522da4f537d07f75dc82cfba42050f946126fcab1c4b84e4afcb00c1a0b1cd` | 同一文件只追加后续无关示例；两表 stable sort/merge/traverse journey 不变。`NON_MATERIAL`。 |
| `container-intrusive-p1-v2` / `hlist-callbacks-4096` | `src/sched/cancel.cc` | `9d3138f5fcf8efb0e5790c202e77b2877055f081c56f435e3edbdc0730f25bcb` | `634465e979b8978dc2c54a5d8b2c2e93ec302d2652f406d0aaf88aa98f559b8e` | 只迁移 allocator 公共词汇；callback hlist 的 register、known-node unlink、detach-before-fire 和节点布局不变。`NON_MATERIAL`。 |
| `time-host-consumer-v1` / `host-phase-deadline-64` | `src/sched/scheduler.cc` | `9fc1b77d51b70b701962418ca5b3003244b9828ac676fd5e9e33a49716842ba3` | `32ec50841311ca5db0a67f735582bcf1a49c4abdc605ae460b4d636808700ef8` | scheduler 只迁移 allocator 借用词汇；64-phase precise clock/deadline/remaining 工作、次数和计费边界不变。`NON_MATERIAL`。 |
| `mem-reclaim-p1-v2` / `immutable-snapshot-read-4096` | `samples/package/mem/cpp/main.cc` | `010e2780457acae9059e0685a7587b81bee0fa97c12217da4cd286602c3b9588` | `0bc93989ac5bb5ee30e44d3a5ab0e3a8b61ba13b881130802280ea74e9572e20` | reclamation 示例之后的 LifetimePlanner 示例扩展了 resource plan 检查；warmed guard/acquire-load/payload-read journey 不变。`NON_MATERIAL`。 |
| `platform-host-p1-v1` / `inference-host-planning-snapshot` | `docs/implementation/tasks/platform/platform-v3-capability-factorization.md` | `664ad1c653d02a19bde33bc3055a263dc8f7b6241c84f7b8e0c7789956a9ffcb` | `48c06e3b49c92f626bfde51bea07db758651d830f9ac6d1d39c7b238c9f21161` | task 后附最终验证、限制与 receipt；冻结的 topology/memory workload、实现、contract 和 benchmark source 未改。`NON_MATERIAL`。 |
| `platform-fs-p1-v1` / `model-file-open-read-close-4mib` | `docs/implementation/tasks/platform_fs/platform-fs-v1-factor.md` | `2c48209881ed05494e5ac300dfedbd43bbe54960abeedc3227511d0fe447960a` | `c05ecd52805a727b9212a00c4ef31e1ddeb0e655462ab0daec9685864679a9f8` | task 后附最终验证、限制与 receipt；4 MiB open/read/checksum/close workload、FileHandle 实现和计费边界未改。`NON_MATERIAL`。 |

`bench/error/borrow_consumer_shapes_v1.json`、三份 Obs consumer shape、Platform thread scheduler
consumer 与其 retained hash 当前逐字一致，不需要 renewal。TimerQueue 已由独立
`timer-queue-consumer-v1` 精确迁移记录续签到同一 scheduler 当前 hash；本页不重复登记该边。

## 结论边界

- 以上 12 条边只恢复旧包对**当前同一消费命题**的可验证性；旧统计点值、peer、toolchain、machine、
  candidate 和适用范围均不扩大。
- `NON_MATERIAL` 不等于文件无变化，也不等于当前上层 Graph/Sched/Mem 产品已收口；只说明被引用的
  frozen workload 子路径没有变。
- 本轮另外发现 Obs ordinary/structured 的历史 `SAME_CONTRACT` 分类不能由相同输出次数推出；其
  修正写回 Obs 性能说明，保留 natural-route 事实但撤销不成立的 peer non-inferiority 外推。
