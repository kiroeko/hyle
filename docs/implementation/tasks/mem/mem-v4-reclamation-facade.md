# Mem v4：reclamation 产品聚合与交付

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

## Closure card（VERIFIED）

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

## Audit 与交付 freeze（AUDITED）

Mem façade 当前 roots 为 `mem,mem_vm`，因此 `<hyle/mem/reclamation.h>` 不会随普通 Mem package 交付；
这正是唯一 public/delivery gap。冻结 FIX 为：加入 `mem_reclaim` root、更新中央 component contract、在
Mem installed sample 执行一次 typed snapshot retire/synchronize、更新 Mem user/architecture 指南，并
回放 Container product closure。base `mem` 与 `mem_vm` 的源码、mimalloc identity、allocator/VM API 和
性能证据均不得改变。

六问直答：产品闭包中既有 mimalloc 仍由 base Mem 直接拥有且本轮不变；新 reclamation factor 无第三方
产品依赖，Folly 只是 benchmark peer。llama.cpp/ggml 身份和推理 journey 记录在 resume card。性能只
聚合同 candidate 的 reclamation P1 与 Container natural-route verdict，不另造 Mem benchmark。Native
FFI 为 `N/A/CLEAN`。source/static/shared 与 relocation 均为本轮硬门。用户 API 从 `hyle::mem` 正常发现，
指南按 allocator/storage/reclamation 任务选型，不能把 internal factor target 暴露成产品菜单。

Audit verdict：唯一能力缺口和所有交付投影已冻结，允许进入实现。

## Candidate freeze

产品候选为 `0f6ef8e83ee290ab565c0d53081a0acb75a46675`，tree
`8cb5ba0024cdb20af65347144f09e16f26147632`。Mem product roots 已加入 `mem_reclaim`，installed sample
执行真实 snapshot publication/retirement，用户指南给出选型与陷阱；base allocator/VM 实现未改。

## 候选验证结果

W1-WIN、W1-WSL GCC 14 与 M1-MAC AppleClang 21 均通过 Mem source embedding、可重定位 shared SDK、
可重定位 static SDK 的 installed consumer；sample 实际完成 snapshot protect、publish、typed retire 与
synchronize。selected roots 同时验证 `mem,mem_vm,mem_reclaim`，公开头和传递依赖没有由源码树或测试
target 遮蔽。base allocator/VM 相对 candidate 前未发生实现漂移。

算法/correctness/sanitizer 与 P1 由同一 candidate 的
[mem_reclaim 记录](mem-reclaim-v2-inference-resume.md)承担；性能结果集中在
[reclamation evidence](../../evidence/performance/benchmarks-mem-reclamation.md)。用户文档已把
reclamation 放入 allocator/storage/lifetime 的任务选型，解释 short guard、显式 domain、callback
线程、record/allocator lifetime、`try_reclaim/synchronize/drain` 和 lock/shared_ptr/hazard 的边界。
FFI 负向扫描没有发现伪 C ABI。scope、architecture、correctness、delivery、usability/docs、red-team、
engineering performance 七项均可判 `PASS`，可以进入 `VERIFIED`。

## Closure decision 与 terminal receipt

本窄 façade round **`CLOSED`**：`hyle::mem` 产品已自然聚合 `mem_reclaim`，用户无需知道内部 factor
target；source/static/shared、relocation、sample、指南、FFI-clean 与同 candidate P1 均通过。base Mem
allocator/resource/pool/NUMA 和 `mem_vm` 本轮没有实现、API、依赖或性能漂移。

限制继承 underlying reclamation factor 的短 guard、borrowed domain/record、同步 callback 与 stalled
reader 合同；Mem 仍没有 Native FFI、稳定跨工具链 C++ ABI或 Formal 性能证书。只有 façade/header/install
不再交付该能力，真实 Mem consumer 暴露选型或 lifetime 缺陷，underlying factor/API/direct dependency
material 改变，或本轮未触碰的 allocator/VM 族满足自己的独立 trigger 时重开对应边界。

immutable candidate 四项 LOC 为 `436,638 / 111,204 / 70,416 / 7,403`，由
`loc-report.py --module mem --commit 0f6ef8e83ee290ab565c0d53081a0acb75a46675 --self-check` 重算通过。
该 conventional boundary 包含 `include/hyle/mem` 与 `src/mem`；registry 将在下一提交绑定 terminal
receipt 与首个实际保存 `CLOSED` state/candidate 的 anchor。
