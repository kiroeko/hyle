# Mem v4：reclamation 产品聚合与交付

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（SCOPED）

- component / owner / round：`mem` / Mem / `mem-v4-reclamation-facade`；
- task kind：`REOPEN`；baseline：`444b02c01afe00f4cd3b44433204e58db3c24712`；
- trigger ID：`MEM-RECLAIM-FACADE-2026-09-03`；
- demand：项目 owner 已准入读极热、低频替换的推理 metadata/snapshot publication；完成恢复轮的
  `mem_reclaim` 必须从普通 `hyle::mem` 产品自然取得，而非只存在于 maintainer factor；
- 范围：Mem façade roots、selected header/package/source delivery、Mem package sample、user guide 与
  Container direct-consumer replay；
- 非目标：allocator、PMR、pool、arena、buffer、NUMA、mapping/VM 算法，mimalloc 升级，Native FFI/
  C API、Compute/Graph、公共仪器或 Formal；
- dependencies/consumers：不改变 base `mem` factor 的 `core/error/platform` 边；产品 façade 增加
  `mem_reclaim` root，Container 同轮声明自己的直接 factor edge；
- tier：本轮自身为 `P1`，但不另造 Mem timing workload；工程性能继承同 candidate 的
  `mem_reclaim` read-side P1 与 Container holdout，Mem 本轮只对 delivery/resource structure 作门；
- machines：W1-WIN、W1-WSL、M1-MAC；instrument `closure-perf-v1`；margin 5%；
- 调查预算：协调轮 4 小时中的 30 分钟；不允许独立性能实验或 allocator 修改；
- provenance：`NORTH_STAR_JOURNEY`，当前不是已发布推理引擎 trace；
- required gates：Mem source/static/shared + relocation、selected headers、package sample、docs/选型、
  FFI-clean scan、direct Container consumer；
- stop：交付闭包或文档无法在一次修复内成立则 `BLOCKED`；不得重开 Mem v3 其余能力。

## Capability / gap ledger（待 AUDITED 冻结）

| ID | requirement | current | action | verification |
|---|---|---|---|---|
| `MEM4-01` | `hyle::mem` 用户可取得 reclamation | factor 只可 maintainer build | `FIX` façade/root/header install | 三种 C++ delivery |
| `MEM4-02` | 普通 snapshot recipe 与选型 | current user guide 不宣传未交付 factor | `FIX` | package sample/docs |
| `MEM4-03` | allocator/VM 等既有能力 | Mem v3 已关闭 | `REJECT` 重开 | 实现 identity 不变 |

## SCOPED verdict

本轮只承担 semantic/delivery owner 的窄复检。`mem_reclaim` 自己承担算法、correctness、性能和限制，
Container 自己承担默认 table-lifetime 改动；任何一方失败都不能由 Mem v3 历史 PASS 代签。
