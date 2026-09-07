# 已冻结模块的真实推理引擎能力缺口审计（2026-09）

<!-- PROJECT AUDIT: CLOSED -->

## 有限审计卡

- owner / kind / demand：HYLE maintainers / `READ_ONLY_PROJECT_AUDIT` /
  `FROZEN-INFERENCE-CAPABILITY-2026-09-07`。
- baseline：`9864c187d4e6caff8606a176e78ba40d234fafae`，canonical checkout 为
  `C:\public\project-hyle\hyle`。开始时工作树干净，main、origin/main 与远端 main 一致。
- 前置：[selector v2](../build/validation-selector-v2-factor-context.md) 为 `PROJECT UNIT REVIEW: CLOSED`，
  实现提交 `66897618aeb2b0c043a4904d205f11abf8b5b892`、终态/证据提交
  `1df0edb0854ae4995f883e7136c7baa76fcd505d` 及行尾修正已推送；
  [性能审计](frozen-performance-opportunity-audit-2026-09.md) 为 `PROJECT AUDIT: CLOSED`，提交
  `be52a7abaa789554281189a4609d5cf79a237711` 已推送。本轮完整读取最终任务，不重新执行两轮工作，
  不复议微性能结论。
- 权威阅读：AGENTS、README、完整 registry、module-closure-methodology、capability-completeness、
  module-review-template、engineering-charter、coding-conventions、roadmap、build-system。
  模块当前状态仅由 [registry](../../module-status.json) 决定，历史 task 和 architecture WATCH
  只作事实/边界输入，不作为当前能力、状态或修改授权。
- 范围：自动筛选 `product_state == CLOSED` 且 `freeze_confirmation.marker == FROZEN` 的 19 因子，
  从其当前公开 C++ 头、用户指南、architecture 与当前 closure task 对照五个真实参考仓库。
- 排除：没有 FROZEN 的旧 CLOSED、util-runtime、sched、io 及其保留因子、compute、graph、Native FFI/
  C API；build/package 只在已有 C++ 能力被交付阻断时记缺口，不单独审计其系统。
- 预算：一次源码/公开面取证、一次疑点反证，最多三小时；至多 80 个去重语义需求、10 个有限候选。
  三个只读子 agent 分别扫描 llama.cpp/ggml、SGLang、vLLM；主 agent 扫描 ds4、核对 HYLE 并唯一编辑
  文档。每个需求有来源、route、owner、verdict 与负空间后停止调查，不按 symbol 数复制引擎。
- 变更预算：产品/API/CMake factor/性能合同/registry 修改、构建、native 测试、P1、性能样本均为零。
  不新建 worktree、clone 或依赖；所有参考 checkout 保持原状。远端只查询默认分支与 HEAD，不移动本地。
- 裁决：`COMPLETE / ROUTED / MISSING_MUST / WATCH / OUT_OF_SCOPE / API_GAP`。缺口按语义需求去重，
  必须有准确源码、foundation ownership、组合不足、跨引擎覆盖、依赖/环分析、当前 reopen trigger
  和最小自然 C++ 草图；本轮草图仅是下一 session 的审计候选，不修改 API。
- 验收：19 因子均有 capability × consumer evidence 矩阵，五仓库版本/状态与准确源码引用齐全，
  所有缺口和未来 owner 输入有去向，无缺口清单明确；最终只运行 selector 选出的 docs/closure，
  `check-module-closure.py`、`check-docs.py`、`git diff --check`，记录不可变快照 LOC，提交推送。
- 停止：完成有限分类与验证后关闭本 audit 并停止；候选可为零，不直接重开任何模块，不把 WATCH
  改写成实施 FIX。不确定能力保留具体证据门，不通过扩大构建或采样预算消除未知。

## 参考身份

2026-09-07 使用 `git rev-parse HEAD`、`git status --porcelain=v1`、`git remote -v`、
`git symbolic-ref refs/remotes/origin/HEAD` 与只读 `git ls-remote --symref origin HEAD` 记录。
五个工作树均 clean。四个本地 HEAD 与远端默认 HEAD 不同，本轮准确审计本地快照，不宣称已审远端新代码。

| 仓库 / origin | 本地 commit | 默认分支 | 查询到的远端默认 HEAD |
|---|---|---|---|
| llama.cpp / https://github.com/ggml-org/llama.cpp.git | `3ad1ba7336986d98592d3e28cafd1a406715351f` | master | `5202104b59ada9005db079eea43882a2b7bf5802` |
| ggml / https://github.com/ggml-org/ggml.git | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | master | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` |
| SGLang / https://github.com/sgl-project/sglang.git | `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110` | main | `a8edafff7cf4a8096437141bd43c00a42faa230b` |
| vLLM / https://github.com/vllm-project/vllm.git | `52358e6e192aeae73dd3764046c98fc4b156ea83` | main | `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b` |
| ds4 / https://github.com/antirez/ds4.git | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` | main | `b6af0adf8ca97c89145c9f9c15be70c9fd6c4507` |

每个 checkout 根为 `C:\public\project-hyle\<仓库名>`，本地分支与默认分支同名。
后续源码表的行号绑定上面的本地 commit；仓库源码是需求证据，不声称已经由 HYLE 集成消费。

## 裁决摘要

本轮 19 个 FROZEN factor：`MISSING_MUST = 0`、`API_GAP = 0`、去重 `WATCH = 1`。
目前没有满足当前 registry reopen trigger 的候选，推荐重开清单为空。唯一观察项是 W01 的
进程既有页迁移及调用线程后续分配策略；两者不能被现有 per-span NUMA placement 冒充，但真实
HYLE C++ 消费者是否必须接管它们尚未得到证明。

下列 68 条源码观察记录不是 68 个不同缺口。相同语义归并到逐模块矩阵；SG09/V18 合并为 W01，
跨进程通信、后台服务关闭等已确定属于上层的需求统一裁决 `OUT_OF_SCOPE`，不再重复记 WATCH。
`COMPLETE` 只表示对应基础能力有自然公开 C++ 路线；`ROUTED` 表示现有组合、标准库或消费者已用的
第三方能够自然表达。两者都不证明参考引擎已集成 HYLE、能够直接替换或数值逐位等价。
本轮没有端到端集成、性能或上游实现正确性判决。

## 自动范围与当前能力基线

范围提取使用以下只读表达式，没有根据目录、façade 或旧 CLOSED 文案补入模块：

```python
registry = json.loads(Path("docs/implementation/module-status.json").read_text(encoding="utf-8"))
frozen = [m for m in registry["modules"]
          if m.get("product_state") == "CLOSED"
          and m.get("freeze_confirmation", {}).get("marker") == "FROZEN"]
```

所有下表记录均为 `CLOSED/FROZEN`。closure task 的候选前 `MISSING/FIX` 只说明当时问题，当前能力
必须同时核对其 terminal receipt 和 baseline 公开头。特别是 Platform-thread 的当前卡虽然联合标题
为 BLOCKED，卡内该 factor 已 CLOSED；Platform/base 与 FS 以各自当前 resume 卡为准。

| factor | registry 当前 closure task | 当前公开能力与源码入口 | 本轮消费对照 / verdict |
|---|---|---|---|
| core | [core v5](../core/core-v5-assert-macro-contract.md) | [assert.h](../../../../include/hyle/core/assert.h)、compiler/environment/cache：fatal/debug、失败时有界 typed context、编译事实与 hint | LG19/D03：基础 invariant `COMPLETE`；任意 backtrace/crash policy 归宿主，`ROUTED` |
| container_intrusive | [intrusive v4](../container_intrusive/container-intrusive-v4-capability-extension.md) | [intrusive_list.h](../../../../include/hyle/container/intrusive_list.h)、forward_list/hlist：双向 auto/safe/count、cached-tail 单链、known-node unlink、稳定排序/merge | D05/LG04：非 owning membership `COMPLETE`；LG13 的 index chain 可保留原形，`ROUTED`；没有 atomic publication 或共享内存指针需求被硬迁入 |
| util_core | [util-core v4](../util/util-core-v4-resource-move-ctad.md) | [function_ref.h](../../../../include/hyle/util/function_ref.h)、unique_resource/scope_guard/ref_counted/borrow/random/spin_lock/type_name：同步借用、清理、共享 owner、URBG 与短锁 | LG19/D04 清理和同步借用 `COMPLETE`；LG16/D10 精确 RNG 序列保留标准库/engine，`ROUTED`；异步 callable 不得借 FunctionRef 延寿 |
| error | [error v5](../error/error-v5-capability-completion.md) | [result.h](../../../../include/hyle/error/result.h)、status/format/system_error：owning Result、非空 Result<T&>、native/standard error、bounded no-throw 文本 | LG01/LG19/SG10/D03：可恢复失败与传播 `COMPLETE`；多任务错误汇合由 owner 用 vector/首错组合，`ROUTED` |
| obs | [obs v5](../obs/obs-v5-metrics-primitives.md) | [log.h](../../../../include/hyle/obs/log.h)、log_control/metrics/trace：同步 typed record、quiescent sink、Counter/Gauge/FixedHistogram 与显式 Concurrent 档、merge/桶分位估计 | LG18/LG20/SG04/SG07/V14/V15 `COMPLETE`；SG05/SG06 名称标签、动态 schema、导出与 rolling writer `ROUTED` |
| math | [math v4](../math/math-v4-checked-discrete-sizing.md) | [scalar.h](../../../../include/hyle/math/scalar.h)：checked/saturating 整数、cast、sum/product/multiply-add、ceil/round、LCM/bit-ceil | LG19/SG10/SG11/V05/D03/D11：标量 sizing `COMPLETE`；模型/tensor layout policy 不属于 Math |
| time | [time v5](../time/time-v5-interoperability.md) | [clock.h](../../../../include/hyle/time/clock.h)、unix_timestamp/native_interop：单调/墙钟/CPU clocks、typed duration/deadline、manual/provider、checked Unix/OS interop | LG20/SG07/D03 elapsed `COMPLETE`；calendar/协议文本按 std/宿主 `ROUTED`；不混用 device event clock |
| timer_queue | [timer queue v1](../timer_queue/timer-queue-v1-factor.md) | [timer_queue.h](../../../../include/hyle/time/timer_queue.h)、timer_entry/common_timeout_queue：caller-owned radix/common deadline、cancel/drain、clock-source overload；不执行 callback | SG15 是周期 watchdog 服务，没有证明缺另一种 timer storage；现有 queue 能力确认，服务为 `OUT_OF_SCOPE`。未伪造参考引擎已直接消费 TimerQueue |
| platform | [platform v11](../platform/platform-v11-perf-control-resume.md) | [cpu.h](../../../../include/hyle/platform/cpu.h)、numa/host_memory/virtual_memory/dynlib/environment/random：OS/ISA/topology、quota/容量事实、per-span placement、VM、library owner、entropy；owning convenience 与 buffer-fill 两层 | LG07/LG10/SG08/SG10/V01–V03 `COMPLETE`；W01 `WATCH`。未知容量/权限失败不是空缺被隐藏为成功 |
| platform_thread | [platform v8：thread terminal](../platform/platform-v8-os-resource-contract.md) | [thread.h](../../../../include/hyle/platform/thread.h)、wait/process：owning callable Thread、required/best-effort 配置、affinity、线程 ISA 状态/AMX 许可、address wait/Parker | LG08/SG08/V01/V04/D04 原语 `COMPLETE`；固定参与者 barrier 用 std::barrier `ROUTED`；pool/IPC 协议 `OUT_OF_SCOPE` |
| platform_fs | [platform-fs v2](../platform_fs/platform-fs-v2-control-resume.md) | [fs.h](../../../../include/hyle/platform/fs.h)：FileHandle、RandomAccessFile、exact/all、identity/size/preallocation、advisory lock、temp/namespace 与 owning path convenience | LG01/V09/D01/D11 同步 bytes/control `COMPLETE`；async/direct/concurrent-offset 服务与事务协议 `OUT_OF_SCOPE` |
| mem_reclaim | [reclaim v2](../mem/mem-reclaim-v2-inference-resume.md) | [reclamation.h](../../../../include/hyle/mem/reclamation.h)：短 read guard/protect、caller-owned retire record、typed callback、try_reclaim/synchronize/quiescent drain | LG10/SG02 的 registry/refcount 不证明必须 RCU/hazard；现有短 snapshot 路线确认，普通 shared_ptr/锁 `ROUTED`。没有证明 stalled-reader-independent 或 GPU/executor 回收缺口 |
| mem | [mem v6](../mem/mem-v6-comprehensive-rebuild.md) | [allocator.h](../../../../include/hyle/mem/allocator.h)、region/pool/heap/buffer/buffer_pool/lifetime_plan：对齐 allocator/PMR、arena/typed pool/quota/stats、resource owner/view、composite、lease、tensor-free interval/alias planner | LG04/LG05/SG11/V05/V08/D03 `COMPLETE`；标准 PMR pool `ROUTED`；foreign device provider/completion/graph legality `OUT_OF_SCOPE` |
| mem_vm | [mem-vm v2](../mem_vm/mem-vm-v2-inference-mapping.md) | [mapping.h](../../../../include/hyle/mem/mapping.h)、large_mapping/fiber_stack：FileMapping RO/RW/COW、checked holes、prefetch/populate/lock_prefix/release_ranges、无私有 header 的 SharedMapping、LargeMapping、guarded stack | LG02/LG03/SG12/V06/V09/D01/D02 `COMPLETE`；共享协议不由 mapping 伪装覆盖；Windows partial-unmap 诚实 unsupported |
| simd | [simd v6](../simd/simd-v6-provider-consumer-replay.md) | [math.h](../../../../include/hyle/simd/math.h)、ops/sort/scan/search/dtype_ops/random/bitpack；ordinary row、narrow activation/norm、非破坏 score-index top-k、expert fixed/fat vendor seam | LG14/LG15/SG17/V17/D08/D09 `COMPLETE`；AoS/custom comparator/std 算法、目标 TU vendor primitive `ROUTED`；tensor/quant/fusion/sampling policy `OUT_OF_SCOPE` |
| container | [container v6](../container/container-v6-identity-p1-resume.md) | [lru_cache.h](../../../../include/hyle/container/lru_cache.h)、small/static/pod vector、ring/deque、flat/BTree/hash、bitset/sparse set：普通 storage 与 entry/weight LRU | LG13/LG17/SG01/V11/V13 `COMPLETE/ROUTED` 按任务选型；pinned set + 可逐出 LRU 可组合；KV tree/policy `OUT_OF_SCOPE` |
| container_concurrent | [container v6](../container/container-v6-identity-p1-resume.md) | [concurrent_hash_map.h](../../../../include/hyle/container/concurrent_hash_map.h)、node map/set/vector：guarded shard reference、显式 retained/reclaiming table、append storage | LG10/SG01 registry/cache 可继续 owner lock；没有证明缺新的 shared-index 合同，`ROUTED`；既有 guarded API 直接可用，未声称 Python dict 本身等于并发需求 |
| container_queue | [container v6](../container/container-v6-identity-p1-resume.md) | [unbounded_queue.h](../../../../include/hyle/container/unbounded_queue.h) 与 SPSC/MPMC/MPSC/SPMC/work-stealing 等具名队列：typed role/progress/capacity storage | SG14/V12 typed FIFO handoff `COMPLETE`；D05 外部锁稳定节点也可用 intrusive；wait/close/cancel/backpressure `OUT_OF_SCOPE` |
| container_text | [container v6](../container/container-v6-identity-p1-resume.md) | [string_view.h](../../../../include/hyle/container/string_view.h)、string_algo/path/interner：borrowed slice、ASCII/hex、UTF8/16/32 strict validation/六方向转换、generation-safe interning | LG11/SG16/V16/D06/D07 基础文本 `COMPLETE/ROUTED`；长度单位和 owner 明确；tokenizer/regex/normalization/stop policy `OUT_OF_SCOPE` |

能力边界同时核对了下列当前 guide / architecture，而非沿用旧 watchlist 的实施计划：

| 产品族 | 当前用法 | 长期边界 |
|---|---|---|
| Core / intrusive | [Core](../../../user/core.md)、[intrusive](../../../user/container-intrusive.md) | [Core](../../../architecture/core-design.md)、[intrusive](../../../architecture/container-intrusive-design.md) |
| Util-core / Error | [Util foundation 部分](../../../user/util.md)、[Error](../../../user/error.md) | [Util 两层](../../../architecture/util-design.md)、[Error](../../../architecture/error-design.md) |
| Obs / Math / Time | [Obs](../../../user/obs.md)、[Math](../../../user/math.md)、[Time](../../../user/time.md) | [Obs](../../../architecture/obs-design.md)、[Math](../../../architecture/math-design.md)、[Time](../../../architecture/time-design.md) |
| Platform / Mem | [Platform](../../../user/platform.md)、[Mem](../../../user/mem.md) | [Platform](../../../architecture/platform-design.md)、[Mem](../../../architecture/mem-design.md) |
| SIMD / Container | [SIMD](../../../user/simd-usage.md)、[Container](../../../user/container.md) | [SIMD](../../../architecture/simd-design.md)、[Container](../../../architecture/container-design.md) |

## 准确源码观察矩阵

以下链接均固定到开头记录的本地 commit；文件内其余行号也属于同一快照。热路径标记只说明消费位置，
不产生性能 verdict。`N` 为项数/字节数，`B` 为桶数，`P` 为 worker/rank 数，`K` 为选出项数。
未冻结模块在这里仅接收需求，没有检查其 HYLE 产品实现。

### llama.cpp 与独立 ggml：20 条

llama 随仓 ggml tree 为 `7674bdaa9279cfc46abde8ce489652f508c0db09`，独立 ggml root tree 为
`95e7421c814b177dccab55d655fc1c91bdf7709b`。两份 `src/ggml-alloc.c` 同 blob
`a71838eafc69d1cab66dfb87885690a6e1e464d0`，`src/ggml-cpu/ggml-cpu.c` 同 blob
`87a329f269752bd398380a12ab91d4b0c116018b`，不是两个独立机制证据。下表 `L` 表示 llama，`G` 表示独立 ggml。

| ID / verdict | 文件、symbol、真实调用 | 语义需求、合同与自然路线 |
|---|---|---|
| LG01 COMPLETE | [L src/llama-model-loader.cpp:1464](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-model-loader.cpp#L1464) `load_data_range`，:1475–1476 非 mmap seek/read；`src/llama-mmap.cpp:264` `read_raw_unsafe` | 模型加载同步 caller-buffer read；owner 活过调用，短读/EOF/OS error 可区分，O(bytes)。FileHandle::read_exact / RandomAccessFile::read_exact_at；共享 owner 由 caller 同步 |
| LG02 COMPLETE | [L src/llama-model-loader.cpp:1403](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-model-loader.cpp#L1403) `init_mappings` :1424 创建；:1470 借地址 | 长寿命零拷贝权重 range。FileMapping::open/map/bytes/view；无 payload copy，view 不活过 reset/release，失败 Result；禁止并发 truncate |
| LG03 COMPLETE | [L src/llama-mmap.cpp:468](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-mmap.cpp#L468) mapping ctor :499–505 advice；:529 `unmap_fragment`；:789 mlock `grow_to`；loader :1776/1778 释放 | loader 预取、逐步锁前缀、丢弃已复制整页。FileMapping 的 hint/populate/lock_prefix/release_ranges 区分保证；control 串行，释放前终止 borrow，成本按页/range/fragments；Windows unsupported 有明确失败 |
| LG04 COMPLETE | [G src/ggml.c:332](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml.c#L332) `ggml_aligned_malloc`；:1632 `ggml_init` 外部/自有 buffer；:1668 free；:1708 object bump | 对齐、owner 与 borrowed workspace、phase bulk lifetime。AllocatorRef/Buffer/FixedArena/Arena；OOM 明确，外部 backing 长寿，bump 分摊 O(1)，clear 作废旧对象；不为每个 tensor 造独立 allocator |
| LG05 COMPLETE | [G examples/gpt-2/main-backend.cpp:832](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/examples/gpt-2/main-backend.cpp#L832) worst-case graph reserve；`src/ggml-alloc.c:718/799–817` last-consumer reuse | 擦除 tensor 后的 interval/resource/size/alignment/alias placement。LifetimePlanner build/instantiate/views；alias 无环、同 resource、output 禁后续复用，OOM rollback，O((N+E) log N)/O(N)；不声称全局最优 packing |
| LG06 OUT_OF_SCOPE | [G src/ggml-alloc.c:623](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml-alloc.c#L623) `ggml_gallocr_allocate_node` :632/645–657 检 op/layout/view；L loader :1699/1720/1747 upload/event | Graph/Compute/loader 决定 inplace 合法性、backend、staging event completion；最后异步访问完成才复用，失败/取消仍须排空或转移 owner。Mem 只接已经证明的 facts；direct read 给 IO |
| LG07 COMPLETE | [G src/ggml-cpu/ggml-cpu.c:637](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml-cpu/ggml-cpu.c#L637) `ggml_numa_init`；:2173 affinity，:3246 worker 配置 | 启动/重配 CPU、node、allowed affinity。Platform facts + Thread 配置；稀疏 id/unknown/error 明确，冷查询 O(CPU+node)；不能用在线核数代替配额，具体分区归 sched |
| LG08 COMPLETE | [G src/ggml-cpu/ggml-cpu.c:3369](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml-cpu/ggml-cpu.c#L3369) `ggml_threadpool_new_impl`；:3231 谓词等待，:576 barrier，:2737–2765 stop/join | OS Thread、原子字等待/Parker 已有；固定 barrier 组合 std::barrier。state 活过 waiter/waker 并在 join 后释放，伪唤醒重查；Parker 单 token 不冒充 semaphore，spawn/config error 可见 |
| LG09 OUT_OF_SCOPE | [G src/ggml-cpu/ggml-cpu.c:3208](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml-cpu/ggml-cpu.c#L3208) poll/worker loop/kickoff；:618 atomic chunk claim | Sched/Compute 轮次、active worker、spin/sleep 与 chunk policy。generation 不得重执行旧图、取消收敛、metadata O(P)；atomic chunk 不证明缺 work-stealing queue |
| LG10 COMPLETE | [G src/ggml-backend-reg.cpp:220](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml-backend-reg.cpp#L220) `load_backend` :229/237 symbols、:245 version/init、:261 handle 移交 | DynLib owning open/get/symbol/close，冷路径失败 Result，owner 活过所有 symbol/callback；registry 同步、版本、优先级与卸载协议归 util-runtime/engine |
| LG11 COMPLETE | [L src/unicode.cpp:1130](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/unicode.cpp#L1130) `unicode_cpts_from_utf8` :1140 消耗一 byte、:1141 U+FFFD；vocab :1235 同类策略 | UTF 基础用 strict validation/六方向 caller-buffer 转换；无分配、全写或不写、错误 code-unit offset、O(input+output)。容错在 engine 按错误前缀组合；llama 宽松 decoder 不等于 HYLE strict 合同 |
| LG12 OUT_OF_SCOPE | [L src/llama-vocab.cpp:827](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-vocab.cpp#L827) WPM preprocess/category/lowercase；BPE :614 regex；UGM :1164 normalize_prefix | tokenizer 拥有 Unicode 数据版本、模型映射、normalization/token identity。词表长寿、session 独占，成本随输入和匹配表；`unicode.cpp:1117` 的单 codepoint 映射不能仅凭函数名认定为 full NFD |
| LG13 ROUTED | [L src/llama-vocab.cpp:96](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-vocab.cpp#L96) SPM priority_queue，:139–166 skip stale/merge/reinsert；:1836/:1847 vocab/ranks | std::priority_queue/vector/unordered_map 自然表达 O(log N) push/pop、expected O(1) lookup；session-owned O(N) storage。stale skip 没有 decrease-key 需求，index chain 无需强换指针 hook |
| LG14 COMPLETE | [L src/llama-sampler.cpp:293](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-sampler.cpp#L293) `softmax_impl` :301–317；:1556 top-p 调用 | 连续 row 的 max/exp/sum/normalize 用 softmax_f32；source/output 独占、exact alias 或不重叠、O(N) 无 row allocation。必须保留 AoS 时 std reduction/transform 自然 ROUTED，不强改 layout；:250 log_softmax 是注释，不计消费 |
| LG15 COMPLETE | [L src/llama-sampler.cpp:321](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-sampler.cpp#L321) top-k/partial_sort；:1549/1577 top-p prefix accumulation | score+index top_k_f32、partial_sort/select/scan；caller O(K) result/workspace，输入可保持，tie/NaN/index 合同显式。AoS 用 std comparator；min_keep/阈值/采样次序属 engine |
| LG16 ROUTED | [L src/llama-sampler.cpp:217](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-sampler.cpp#L217) `llama_sample_dist` :244 discrete_distribution、:246 mt19937；:1122 持有 RNG | 标准库已自然支持 owned/clone/reset 随机状态和离散抽样。sampler 单 owner、权重/空候选预条件由 engine 保证；不为替换而更改精确随机序列或宣称跨 STL 位一致 |
| LG17 COMPLETE | [L src/llama-sampler.cpp:2919](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-sampler.cpp#L2919) penalties accept :2925–2937 增计数/减最旧/erase/push | 最近 N history + frequency map：CircularBuffer + FlatHashMap/std map；覆盖前取最旧值，capacity=0 明说，owner 同步两份状态；steady ring O(1)/零分配，penalty 数学归 sampler |
| LG18 COMPLETE | [G src/ggml.c:287](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml.c#L287) sync log callback；L `common/log.cpp:227` 复制入队、:295 drain、:319 pause/join | LogRecord/LogSink/SinkRegistration；borrow 到 callback 返回，异步前复制、注销 quiescent、noexcept 并发 callback。queue/drop/block/worker/durability 是 host sink/runtime，emit 不等于落盘 |
| LG19 COMPLETE | [G src/ggml.c:1717](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml.c#L1717) object sizing/capacity；:253 abort；L loader :1625 cancel、:1758 验证错误汇合 | Math checked arithmetic、Error value/context、Core invariant、typed owner/UniqueResource/scope guard。成功不造诊断 allocation；OOM/cancel 与 invariant 分离，多错误保存顺序由 loader 负责 |
| LG20 COMPLETE | [L src/llama-context.cpp:728](https://github.com/ggml-org/llama.cpp/blob/3ad1ba7336986d98592d3e28cafd1a406715351f/src/llama-context.cpp#L728) eval/prefill 计时计数；:3338 perf_get_data；:3353 reset | Time monotonic + Obs Counter/Gauge，owner state 到采集/重置前有效，O(1) 更新；并发 snapshot 非跨字段事务，quiescent 后 exact。阶段定义由 engine 拥有 |

### SGLang：18 条

| ID / verdict | 文件、symbol、实际调用 | 语义需求、合同与自然路线 |
|---|---|---|
| SG01 COMPLETE | [multimodal_cache.py:91](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/mem_cache/multimodal_cache.py#L91) MultiModalStaticCache get/set/free，`managers/mm_schedule.py:254/284`；`buffer_mode/storage_existence_cache.py:49/57/78`，pipeline :324/718 接线 | item/weight 预算与 recency：LruCache get/peek/try_insert_or_assign；hit 不分配，插入按 victim 数付费，oversize/OOM 明确，borrow 不跨 mutation。tensor/rank/stale-positive policy 归 owner |
| SG02 OUT_OF_SCOPE | [tree_v2_impl.h:83](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/mem_cache/cpp_radix_tree/tree_v2_impl.h#L83) tree_walk、:172 lock_ref；tree_v2.cpp :55 evict；radix_cache_cpp.py :111 match_prefix | KV token-prefix split/ancestor pin/leaf eviction，key walk O(tokens)、pin O(height)，stable nodes + tensor/page policy。std map/vector/unique_ptr/heap 已供 storage；业务 tree 属 engine，普通 refcount 不是 RCU 需求 |
| SG03 ROUTED | [hash_binding.cpp:62](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp#L62) hash_page/SHA256；utils.py :114/146 hash chain/cache salt；:25 checked_u32 | O(bytes) 内容链摘要，token/prior-digest 长度检查、独占 hash state。现用 OpenSSL/hashlib + engine wire protocol；HYLE hex 可复用，hash_bytes 不承诺持久化，不能冒充 SHA256 |
| SG04 COMPLETE | [metrics_collector.py:279](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/observability/metrics_collector.py#L279) running gauge；:2166 histogram、:2176 token counter、:2250 更新 | scheduler 生产指标：Counter/Gauge/FixedHistogram、显式 Concurrent 档；O(1)/O(log B) 更新、O(B) snapshot/merge，无 per-record allocation。overflow/非有限值及 quiescent reset 明确 |
| SG05 ROUTED | [metrics_collector.py:258](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/observability/metrics_collector.py#L258) PROMETHEUS_MULTIPROC_DIR、:283 mostrecent、:2198 labels；request_metrics_exporter.py :34 schema | 名称/labels/cardinality、跨进程聚合与动态 schema 由已用 Prometheus/host exporter 管理，拥有 strings/registry/environment。固定数值 primitive 不应重复实现后端或默默强加全局 registry |
| SG06 ROUTED | [request_metrics_exporter.py:128](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/observability/request_metrics_exporter.py#L128) write_record async lock、:149 JSON/newline/flush；utils/log_utils.py :35 rotation | Obs 同步 record + host writer；异步前复制，lock 覆盖选文件/整条写入，切换先关闭旧 handler，失败可重试。FileHandle write_all/sync 提供 bytes，rotation/retention/JSON 与异步生命周期由宿主负责；flush 不等于 durability |
| SG07 COMPLETE | [scheduler_stage_metrics.py:71](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/observability/scheduler_stage_metrics.py#L71) start/enter/exit/drain、:85/93 monotonic_ns、:126 sample | 单 owner exclusive stage 记账，嵌套 finally 恢复、disabled 不读钟，O(1) 切换。Time + 固定字段/数组 + scope guard 自然组合；stage enum/profiler 标签归 scheduler |
| SG08 COMPLETE | [numa_utils.py:162](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/utils/numa_utils.py#L162) node∩allowed affinity；native numa_utils.cpp :69 OMP pin | CPU facts/Thread affinity：冷查询、稀疏 id、空集合/权限错误，mask 活过 spawn 复制；OMP/PyTorch 数量和 rank placement 属 runtime |
| SG09 WATCH W01 | [native numa_utils.cpp:31](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/kernels/aot/csrc/cpu/numa_utils.cpp#L31) :48 migrate_pages、:54 membind；`srt/distributed/bootstrap.py:95/207/212` 接线 | 区分已有进程页迁移与调用线程后续分配策略；不能以 per-span bind 签同义 COMPLETE。现由 libnuma 启动 adapter 完成，HYLE 通用接管必要性未证明，见 W01 |
| SG10 COMPLETE | [pool_host/base.py:51](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/mem_cache/pool_host/base.py#L51) host budget；memory_pool_host.py :214–220 申请前校验 | Platform host/node available + Math checked multiply/add；snapshot 非 reservation、OOM 仍要处理。rank 均分/10 GiB headroom 属 owner，不把 psutil 全机公式称为 cgroup 保证 |
| SG11 COMPLETE | [cpu/topk.cpp:95](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/kernels/aot/csrc/cpu/topk.cpp#L95) 每 worker 对齐 scores/pairs；flash_attn.cpp :412/424 resize buffer/indices | CPU scratch：alignas array 或 Buffer/PMR + checked sizing，owner 活过 kernel/所有切片，不同 worker 不共享写，复用避免每次分配。tile/stride/attention layout 留 Compute |
| SG12 COMPLETE | [host_shared_memory.py:27](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/utils/host_shared_memory.py#L27) rank0 create/peers open；:42–59 保持 owner；offloader.py :397 消费 | SharedMapping create/open/access/unlink/reset：page-aligned、size/protocol 外协商、view 不长于 owner。:45–49 CUDA host registration 是 Compute/provider 的额外需求；storage 不等于 GPU 可见保证 |
| SG13 OUT_OF_SCOPE | [cpu/shm.cpp:249](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/kernels/aot/csrc/cpu/shm.cpp#L249) all_reduce :276–292 copy/fence/state/双 buffer；parallel_state.py :674 调用 | 跨进程 publication/代际/peer completion；buffer 复用不能越过 peer 读取，需声明 timeout/crash/ABA/并发 collective。属于 IO/Compute/engine 协议，不把 volatile busy-wait 认证为通用 C++ 同步 |
| SG14 COMPLETE | [cache_controller.py:436](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/managers/cache_controller.py#L436) storage threads/queues；:1126 prefetch_io_aux_func 发布 ack | typed FIFO storage：UnboundedQueue 或明确角色的 SPSC。发布后 ownership handoff，empty/OOM 分开，payload nothrow，allocator 可并发，析构需 quiescence；不以 approximate length 判 drain |
| SG15 OUT_OF_SCOPE | [cache_controller.py:470](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/managers/cache_controller.py#L470) stop/sentinel/join；:633 detach；watchdog.py :189/197/203 Event/monitor loop | 后台服务 start→wait→stop/wake→drain/cancel→join→release。join timeout 后保持 live resources，不得先释放再 reattach；归 util-runtime/sched/IO，queue 不新增常态关闭状态 |
| SG16 ROUTED | [detokenizer_manager.py:384](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/managers/detokenizer_manager.py#L384) sent_offset/printable prefix/finalize；:75/90 DecodeStatus/chunks，:187 trim | std::string/vector + UTF/view 原语；byte/codepoint/token offset 分开，尾部状态避免重复发送/每 token 复制全历史。replacement、batch grouping、stop token、finalize 为 engine，不以 byte substring 冒充 Python str offset |
| SG17 COMPLETE | [cpu/topk.cpp:7](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/kernels/aot/csrc/cpu/topk.cpp#L7) softmax；:85 grouped_topk、:116/130 partial_sort | 连续 row normalize 和 score/index 选择已有 softmax_f32/top_k_f32；caller owns O(K) output/scratch，alias/tie/NaN 选择明确。group/expert mask/renormalize 属 Compute/engine，不能声称 bitwise 等价 |
| SG18 OUT_OF_SCOPE | [sampler.py:717](https://github.com/sgl-project/sglang/blob/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/layers/sampler.py#L717) top-k/top-p/min-p；:850 multinomial_with_seed、:876 float64 Gumbel | request seed×position×column、重排不变性、阈值与精度是完整 sampling 协议；:747 拒绝一种 seed/min-p 组合。mutable bulk RNG 不等于 stateless coordinate hash；通用算术可复用，策略归 engine/Compute |

`buffer_mode/buffer_page_cache.py:3` 明写 NOT WIRED YET，没有作为现用 consumer。
`radix_cache_cpp.py:43/78–88` 标记 experimental 且 HiCache 未接线，tree_v2.cpp:107–119 load/commit 未实现；
SG02 只认已经接线的 prefix/ref/evict 部分。

### vLLM：18 条

| ID / verdict | 文件、symbol、实际调用 | 语义需求、合同与自然路线 |
|---|---|---|
| V01 COMPLETE | [cpu_resource_utils.py:176](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/utils/cpu_resource_utils.py#L176) allowed CPU；ompmultiprocessing.py :71 configure_omp_envs；cpu_worker.py :152 绑定后不改线程数 | Platform topology/capacity + Thread affinity；冷路径、稀疏 CPU、required/unsupported 明确。OpenMP/PyTorch 初始化顺序与 worker 数归 runtime |
| V02 COMPLETE | [cpu_worker.py:73](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/v1/worker/cpu_worker.py#L73) node memory budget，:84 失败；cpu_resource_utils.py :122/164 node/cgroup | host/node fallible snapshot + checked sizing；全局限制只合并一次、total unknown 保持 optional、0 available 有效；分配竞争仍需 OOM 路径 |
| V03 COMPLETE | [csrc/cpu/utils.cpp:45](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/csrc/cpu/utils.cpp#L45) memory locality 子需求 | 仅“caller 已知 arena 的 bind/interleave”可用 vm_bind_numa/LargeMapping/selected nodes。policy accepted 不等于 residency；整个进程环境改写另列 V18，不能混算同合同 |
| V04 COMPLETE | [cpu_attn.py:481](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/v1/attention/backends/cpu_attn.py#L481) _get_attn_isa :494 torch probe、:505 AMX；:401 CPU attention；cpu_attn.cpp :274；cpu_attn_amx.hpp :203 tile_loadconfig/:414 tile_release | cpu_features + this_thread::isa_state/request_amx_tile_permission 已公开。每执行线程请求失败不得进 kernel；tile state/模型 dtype/block-size selection 属 kernel。没有从 torch probe 名称推断其 OS 许可实现 |
| V05 COMPLETE | [csrc/cpu/utils.cpp:96](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/csrc/cpu/utils.cpp#L96) ScratchPadManager grow；cpu_fused_moe.cpp :138/199 大尺寸 scratch | Buffer::create 成功后 move 替换，Math checked add/multiply/cast/alignment；OOM 保留旧 owner，旧切片不得跨换块；workspace 分区/串行责任在 kernel，不从 singleton 推出线程安全 |
| V06 COMPLETE | [csrc/cpu/shm.cpp:241](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/csrc/cpu/shm.cpp#L241) SHMManager :260 join、:326 shm_open/ftruncate/mmap、:278 cleanup | SharedMapping move-only storage owner；rank 间协商 size/access/readiness，OS page alignment、错误 Result、unlink/reset；MAP_POPULATE 的 touch 不能被默认 lazy open 冒充 |
| V07 OUT_OF_SCOPE | [shm_broadcast.py:648](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/distributed/device_communicators/shm_broadcast.py#L648) acquire_write；:761 acquire_read/finally；:250 written/readers；:640 shutdown | 广播 ring 的多 reader publication、backpressure、cancel/timeout、notification。writer 扫 reader 状态 O(P)，复用需所有 reader 完成；IO/runtime/engine 协议，不是进程内 queue 或 SharedMapping 的完整替身 |
| V08 ROUTED | [shm_broadcast.py:389](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/distributed/device_communicators/shm_broadcast.py#L389) _rebuild_tensor；:410 ring 必须复制、:415 空 payload、:428 prompt_embeds | std::shared_ptr<Owner> + span 或 Mem Buffer/view/adopt 自然表达 owner/borrow；复用 chunk 离开 scope 前复制或协议延迟复用，不能为 zero-copy 延长悬空 view；tensor adapter 留 Compute |
| V09 COMPLETE | [weight_utils.py:866](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/model_executor/model_loader/weight_utils.py#L866) safetensors iterator，:958 eager、:995 safe_open；:774 bounded prefetch | FileHandle/RandomAccessFile/FileMapping 提供 bytes 与显式 warm-up，O(file bytes)/O(block size) scratch；格式、名字与 shard 由已用 safetensors/loader 管理，不复制 parser 到 Mem |
| V10 OUT_OF_SCOPE | [weight_utils.py:791](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/model_executor/model_loader/weight_utils.py#L791) parallel prefetch；:841 executor；:1003 multithread loader、:1027 future.result | Sched/IO/loader 拥有并发上限、warning vs fatal error、取消/关闭、结果驻留量与返回顺序；不以 Python executor 类型名判基础模块缺线程池 |
| V11 COMPLETE / ROUTED | [vllm/utils/cache.py:51](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/utils/cache.py#L51) LRU，:202 skip pinned；multimodal/cache.py :181 size budget；lora/model_manager.py :1241 pin_adapter | entry/weight budget 为 COMPLETE；pin eligibility 可拆 pinned owner registry + 可逐出 LRU，或 hash+intrusive list，ROUTED。hot hit 不分配、oversize/OOM/eviction callback 明确；没有给常规 LRU 加常态 pin bit 的必要证明 |
| V12 COMPLETE | [v1/engine/core.py:1057](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/v1/engine/core.py#L1057) EngineCoreProc queues；:1128/1140 threads、:1806 producer、:1464/:1844 consumers | typed FIFO handoff storage 用 UnboundedQueue，block 分配分摊、OOM 独立、payload nothrow、析构 quiescent；blocking/shutdown/deadline/backpressure 属 service |
| V13 ROUTED | [request_queue.py:131](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/v1/core/sched/request_queue.py#L131) PriorityRequestQueue :146/152 heap、:175 heapify、:194 ordered copy；scheduler.py :203/2158 消费 | std heap + erase_if/make_heap 自然实现 priority/arrival waiting；push/pop O(log N)、批量过滤 O(N)、有序快照 O(N log N)/O(N)。公平性/优先级属于 scheduler，没有 handle heap 必要性 |
| V14 COMPLETE | [vllm/logger.py:75](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/logger.py#L75) print_once memoization、:96 rank scope；weight_utils.py :895 调用 | Obs synchronous record/sink；动态消息去重可组合已有 LRU/hash，不能把 first-N state 当消息字典。callback 借用/quiescence 明确，rank scope 与记录保留归 host |
| V15 COMPLETE | [v1/metrics/loggers.py:493](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/v1/metrics/loggers.py#L493) PrometheusStatLogger；:1212 token count、:1226 TTFT、:1235 latency | Obs Counter/Gauge/FixedHistogram，local-first 或显式 Concurrent，O(1)/O(log B) 更新、弱 running snapshot、停写后 exact；labels/export/multiprocess schema 继续宿主 ROUTED |
| V16 ROUTED | [detokenizer.py:310](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/vllm/v1/engine/detokenizer.py#L310) check_stop_strings :340 suffix find；tokenizers/detokenizer_utils.py :260 incomplete byte fallback | owning std::string + borrowed StringView/UTF primitives；搜索新增尾部，保留 byte/codepoint 索引映射。stop-list precedence、tokenizer cleanup 与 byte fallback 状态机归 engine |
| V17 COMPLETE | [csrc/cpu/layernorm.cpp:5](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/csrc/cpu/layernorm.cpp#L5) rms_norm_impl；activation.cpp :6 kernel/:37 silu | 普通连续 f32/f16/bf16 row norm/activation 已有 rms_norm/swiglu，O(N)/O(1)、不分配/锁、窄输入用 f32 compute，epsilon/alias 明确；:48 fused residual、OMP batch/stride 为 Compute，不判位一致 |
| V18 WATCH W01 | [csrc/cpu/utils.cpp:18](https://github.com/vllm-project/vllm/blob/52358e6e192aeae73dd3764046c98fc4b156ea83/csrc/cpu/utils.cpp#L18) init_cpu_memory_env :39 migrate_pages、:49/63 allocation policy、:77 strict；cpu_worker.py :71 调用 | 当前代码证明真实启动需求，但没有证明 HYLE 普通 C++ 无法通过 per-arena/启动配置自然完成。部分失败/权限/继承范围未形成共同 foundation 合同，保留一个去重 WATCH |

### ds4：12 条

| ID / verdict | 文件、symbol、实际调用 | 语义需求、合同与自然路线 |
|---|---|---|
| D01 COMPLETE | [ds4.c:2531](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4.c#L2531) model_open：open/fstat/mmap；:2401 model_close | 模型载荷 owner 与 zero-copy borrowed bytes。FileHandle→FileMapping，RAII/Result 表达失败、checked range；模型 metadata/tensor 表活过使用，owner 在读者退出后释放；GGUF parser 属 loader |
| D02 COMPLETE | [ds4.c:2411](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4.c#L2411) model_prefetch_cpu_mapping :2423 WILLNEED；:2576 loader 调用 | 加载期 best-effort hint，不等于触页完成/驻留。FileMapping advise/prefetch/populate/lock 可明确选择；失败警告或致命由 loader 决定。Metal no-copy buffer 与映射共同寿命给 Compute |
| D03 COMPLETE | [ds4.c:1264](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4.c#L1264) xmalloc_zeroed checked product/explicit memset；:1282 now_sec；:2018 cursor_error；:2491 tensor product guard | checked bytes、显式初始化、elapsed、首错上下文：Math + allocator/Buffer + fill + Time/Error。O(bytes) 初始化是消费者明确需求，不偷换 calloc lazy 合同；parser/model 错误策略仍由 ds4 决定 |
| D04 OUT_OF_SCOPE | [ds4.c:1922](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4.c#L1922) threads_init；:1959 shutdown；:1980 parallel_for_min_rows；:2014 threshold | 同步 bulk、caller 参与、nested serial fallback、generation 与 stop/broadcast/join。Thread/std mutex/CV/FunctionRef 是可用基础；任务阈值、分块、pool 生命周期属于 Sched，state 活到所有 worker join |
| D05 COMPLETE | [ds4_server.c:13276](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_server.c#L13276) enqueue、:13294 dequeue；:13266 known-predecessor detach | 外部 mutex 下 stable request next/head/tail：IntrusiveForwardList 可提供 O(1) push/pop/known-predecessor unlink、无 owning node allocation。stop、wait、batch dispatch 属 service；普通 hook 不承诺并发 publication |
| D06 ROUTED | [ds4_server.c:1257](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_server.c#L1257) utf8_stream_safe_len，:6897/7157/7890/8719 SSE 消费 | 跨 token 的 trailing incomplete UTF-8 暂存，防重复/坏前缀发送；最多保留一个 codepoint 的未完成尾部。strict UtfResult + owner carry 或明确宽松协议可自然组合；final=true 放行的 ds4 策略不是 strict Unicode 保证 |
| D07 ROUTED | [ds4.c:39765](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4.c#L39765) tokenize_span 临时 NUL copy→bpe；:39721 tokenize_text；:39725 special_token_at | pointer+length/StringView/std::string 可自然承担 borrowed/owning 文本；旧 BPE 接受 NUL-string 不证明缺一种 vector/string。special token、chat template、BPE merge 为 tokenizer owner，无基础 API 修改 |
| D08 COMPLETE | [ds4.c:7581](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4.c#L7581) unweighted/weighted/head RMSNorm :7590/7599 | SIMD rms_norm_f32 的连续 row 语义完整，caller buffer、epsilon/alias、O(N)/O(1)。ds4 此处 double sum，HYLE 数值合同独立；模型若必须该累加顺序用 std/expert，不把名称相同写成 bitwise substitute |
| D09 COMPLETE | [ds4.c:11583](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4.c#L11583) topk_desc；:11618 layer_topk_selected_experts_from_probs、:11626 bias、:11633 调用 | 非破坏 score+index top-k 已有自然 O(K) output/workspace，明确 tie/index；bias 排名与无 bias 权重归一化留 engine。只判功能，不重议既有 top-k 微性能结论 |
| D10 OUT_OF_SCOPE | [ds4.c:40257](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4.c#L40257) sample_build_probabilities；:40189 RNG、:40224 heap、:40349 必要时全候选 fallback | seed 消耗、非有限 logit、temperature/top-k/top-p/min-p、speculative p/q 协议归 sampler；基础 heap/std exp/sort/URBG 路线存在。保持精确序列可保留 owner 算法，不因 HYLE 有另一个 RNG 自动替换 |
| D11 ROUTED | [ds4_kvstore.c:923](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_kvstore.c#L923) store_live_prefix_text；:771/784 checked file budget、:1084 write/fflush、:1092 close、:1107 rename；server :10175 调用 | 同步临时文件/完整写/close/rename + checked sizing 可组合 Platform-fs/Math；model KV serialization、空间准入/eviction、错误恢复是 engine/IO 协议。现代码未请求 fsync，不把 fflush+rename 签为断电 durable |
| D12 OUT_OF_SCOPE | [ds4_linux_memory.h:9](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_linux_memory.h#L9) nonmovable memory parse：MemAvailable−CmaFree；ds4.c :41507/41525 glm_graph_memory_guard_budget 消费 | ROCm pinned/nonmovable 预算不是普通 host available；provider 必须说明可 pin、设备工作集、共享物理池和重复计费。给 Compute/provider+engine 的输入，不能声称 Platform snapshot 保证 GPU 可分配或照抄公式为跨 OS fact |

## 自然 C++ 使用形状复核

| 疑似缺口 | 对实际任务的反证 | 结论 |
|---|---|---|
| loader 只能整文件 map，无法渐进锁定/释放 | mapping.h:93–113 已有 lock_prefix/locked_size/release_ranges；open/map 接现有 FileHandle，view/holes/lifetime 公开。LG03 有真实需求而非 benchmark 假形状 | COMPLETE；OS 不支持的 partial-unmap 不能通过 API 命名变成支持 |
| graph temporary 只能逐块分配 | lifetime_plan.h:44/95/128/152 的 requests→plan→instantiate→view 是 tensor-free consumer 路线，LG05 擦除后的 facts 可直接传入 | COMPLETE；op/alias 合法性在 Graph，不倒置依赖 |
| weighted LRU 缺失，或者所有 LRU 必须 pin | lru_cache.h:20/54/168/264 已有 item/weight budget；V11 pin 可由 owner 分离 pinned registry 与 evictable LRU，或组合 hash+hook | COMPLETE/ROUTED；没有不能自然组合的证明，不加常态 pin metadata |
| thread API 只能 fn+void*，或没有 AMX OS 权限入口 | thread.h:110/121 已有 owning capturing/move-only callable，:181–182 显式线程状态/请求权限；普通 CPU/path 查询还有 owning convenience | COMPLETE；低层 buffer-fill/fn+ctx 不是唯一普通 C++ 面 |
| UTF 只给 bool，流式输出必须新加 decoder owner | string_algo.h:279/292 有 typed status/byte offset；src/container_text/string_algo.cc:180/242 走产品 simdutf detailed validator，src/container_text/CMakeLists.txt:42 无条件配置该 private mechanism | 基础 COMPLETE，carry/replacement ROUTED；对有效前缀保留最多 3 个未完成尾 bytes，final/非法恢复由 engine 决定；不重扫全部输出历史 |
| top-k 必须修改 source 或分配 N 个 index | sort.h:51–69 的 ScoredIndexF32/top_k_f32 只需 K 槽；LG15/SG17/D09 的语义可自然表达 | COMPLETE；AoS/custom comparator 继续 std::partial_sort，不能为连续 row API 强制改模型 layout |
| 固定 histogram 数量/显式 scratch 是 benchmark 参数 | bucket 个数决定静态 storage，边界值 runtime、merge/quantile 查询公开；sort workspace 长度由实际输入规模推导，无固定样本 N | COMPLETE；动态 metric schema 由既有 exporter ROUTED，固定 API 不承担其所有 policy |
| shared bytes 等于共享通信服务 | mapping.h:134–171 与 user/mem 明确 page-aligned storage、外部 size/readiness；wait.h 是线程等待；SG13/V07 的 peer 协议更强 | storage COMPLETE，IPC service OUT_OF_SCOPE；不声称 std::atomic/thread wait 天然保证任意跨进程协议 |

当前未发现现有 C++ 能力被 source/static/shared delivery 阻止消费的新证据；本轮不重跑 package，
也不把只读声明核对写成新 native 集成 PASS。旧 architecture WATCH 中已经由 Math v4、Obs v5、SIMD v6、
Mem/Container 当前 terminal 落地的条目不重复计缺口。

## 唯一有限观察项 W01：NUMA 环境接管

- 源码：SG09 与 V18；两个仓库确有实际启动调用。它们是两个消费位置，不据相似 libnuma 代码推断
  两种独立算法。CPU worker/model load 初始化属于真实消费，但不是 token 热路径采样证据。
- 精确差别：`numa_migrate_pages(pid, ...)` 面向既有进程页；`numa_set_membind`/interleave 设置调用
  线程的后续分配策略及其继承效果。后者不能写成“原子修改所有既有线程的 allocator”。两个操作
  不同寿命、权限和部分失败面；源码的 warning/strict 也不证明全部迁移成功。
- 现有能力：Platform 的 numa_nodes/node_memory/vm_bind_numa/selected interleave 与 Mem 的
  LargeMapping 已自然覆盖 owned arena placement。startup numactl/libnuma 或宿主 adapter 可以处理更宽
  policy；本轮没有证明 HYLE 的 native loader 必须在外部 allocator 已分配之后接管整个进程。
- verdict：`WATCH`，不是 `MISSING_MUST/API_GAP`。未证明 foundation ownership 与组合不足，所以不提出
  一个笼统 set_numa_mode API，也不把观察项写成待实施 FIX。
- 当前 registry 对照：Platform 第 1 条为 “A named consumer reproduces a Platform node-memory, CPU or
  host-fact correctness, security, ownership, concurrency, error or natural C++ API-usability defect.”；
  第 3 条为 “A named inference journey proves that deferred node distance, bandwidth, residency or pressure
  capability belongs in Platform.”。目前没有复现前者，也未证明后者。不能先发明 API 再用“API changed”
  触发重开。
- 晋级所需的有限输入：一个明确 native C++ consumer，列出无法改归 owned arena 的 foreign allocations，
  证明启动配置不足；分别冻结 migration 和 calling-thread default policy 的 scope、inheritance、restore、
  partial failure、supported-OS 行为；再判断是否为通用 Platform mechanism，或仍属 launcher/provider。
  没有这些事实就停止，不靠新 benchmark/P1 推动晋级。
- 若未来获准：预计只影响 Platform 的窄 OS mechanism 与该真实直接 owner 的 adapter；owned mapping
  若需组合可沿既有 mem_vm→platform 边。是否引入 libnuma 尚未决定，不能将参考依赖自动列为 HYLE 依赖。
  不需要引入 Compute/Graph/模型类型，不允许 platform→mem/sched 的反向边。

## 缺口、重开排序与无缺口清单

按“真实消费者覆盖度→基础性→不可自然组合的证据→修改范围/成本”准入后，合法 reopen candidate
为 **0 个**，因此没有排名第一的 MISSING_MUST/API_GAP，也没有可冻结的下一轮产品 FIX 集。
MISSING_MUST/API_GAP 所需的 ownership、同合同、依赖、trigger 与最小 API 草图条目为空，原因是没有
任何记录同时达到这些证据门，而非漏填。W01 是观察优先级第 1 项，不能混入实施候选排名。

无确认缺口的 19 因子为：core、container_intrusive、util_core、error、obs、math、time、timer_queue、
platform、platform_thread、platform_fs、mem_reclaim、mem、mem_vm、simd、container、
container_concurrent、container_queue、container_text。其中 platform 保留 W01；其余 18 个没有本轮
残留 WATCH。timer_queue、mem_reclaim、container_concurrent 等缺少新的特定上游 storage 需求时，
如实记录负空间，不能因“未被五仓直接调用”判定无用或为覆盖率制造迁移。

这不是未来所有推理引擎需求都完整的宣言；它只覆盖准确快照与上列真实场景。下一轮应恢复原依赖顺序的
util-runtime/sched/io 独立任务，使用下列需求输入；尚不应为 W01 重开 Platform。

## 给未来 owner 的有限输入与依赖图影响

这些条目统一为 `OUT_OF_SCOPE`，是未来审计输入，不是对当前未冻结实现的缺陷判决或施工授权。

| ID / owner | 真实来源与必要行为 | 依赖/生命周期约束 |
|---|---|---|
| U01 util-runtime / engine | LG10 backend registry：按名/版本/priority 决定 factory；成功注册持有 library；失败 rollback；若未来需要 unregister/unload，应先排空全部实例/callback | 借 Platform DynLib、Error 和现有容器；不要将动态 registry/unload 塞入 util_core/Core，不给底层反向依赖 runtime |
| S01 sched | LG09/D04 CPU bulk：caller participation、nested policy、固定 worker scratch、generation publication、NUMA placement、spin/sleep、错误/取消汇合 | 使用 Thread/wait/Time/intrusive/queue 等向下边；pool 策略不得下沉 platform_thread；thread state 活过 join |
| S02 sched / util-runtime / IO | SG15/SG14/V12 后台 prefetch/ack/watchdog：start→wait→stop/wake→drain/cancel→join→释放；超时保留活资源，防 stale sentinel/generation | queue 只 storage，close/deadline/backpressure 在 service owner；不能 queue→sched 形成环；取消和 empty/OOM 不是同一状态 |
| I01 IO / model loader | LG06/V09/V10 的 direct/parallel/async load：offset/alignment、bounded inflight、short-transfer、取消后 drain、结果驻留上限与错误级别 | 同步 FileHandle/RandomAccessFile 已供普通读取；并发 offset、completion 与 direct 服务留 IO；owner 必须覆盖 in-flight |
| I02 IO / distributed runtime | SG13/V07 shared-memory publication：多 reader acknowledgment、membership/代际、backpressure、notification、timeout/peer crash、复用与 shutdown | storage 用 SharedMapping；不能把进程内 queue/Parker 当 IPC certification；独立通用 doorbell/barrier 若日后有完整跨 OS 消费合同再准入 |
| I03 IO / engine | D11 cache snapshot：临时文件、预算、完整写、close/rename、失败清理与恢复；若要求断电 durability，另行定义文件和目录 sync 边界 | Platform-fs 提供同步 control，模型 KV serialization/eviction 属 engine；不要把 fflush/rename 自动升级为事务保证 |
| C01 Compute / provider | LG06/SG12/V08/D02/D12 的 imported/pinned/managed/设备 storage：registration、coherence、native identity、event completion、共享物理池/可 pin 预算 | provider 注入 Mem resource/Buffer；completion 在 Compute，Mem 不反向引用 Device/Stream/Event。共享 host/device budget 不能相加或重复扣款 |
| C02 Compute / engine | LG12/LG15/SG02/SG17/SG18/V17/D08–D10：tensor stride/batch、quant block、fused activation/residual/norm、attention/KV/MoE、sampler policy | SIMD 留模型无关 row/expert primitive；真实数值/格式/seed 协议归对应 owner，不能据 GPU/Triton operator 名字新增 scalar API |
| G01 Graph | LG05/LG06 的 tensor-free planner 上游：推导 interval/alias/output、op inplace legality、动态形状变化后的重规划、multi-resource backing | Graph→Mem LifetimePlanner，不能 Mem→Graph；KV radix/slot/rank batching 不自动等于通用 Graph；本轮不评重做中的实现 |
| E01 tokenizer / serving / observability host | LG11–LG18/SG03/SG05/SG06/SG16/SG18/V14–V16/D06/D07：Unicode/version/regex/chat/stop、exact RNG、内容摘要、labels/export/rolling writer | std/既有 tokenizer/OpenSSL/Prometheus + HYLE bytes/view/metrics/sinks 自然组合；不引入 engine/Python/tensor 类型到冻结 foundation |

当前直接边由 [HyleComponents.cmake](../../../../cmake/HyleComponents.cmake) 核对：core 无依赖；
util_core/container_intrusive/error/math/time/mem_reclaim 的 foundation 边保持原状；timer_queue 公开
core/time/container_intrusive；platform 公开 core/error，thread/fs 公开 core/error/time、私有 platform；
mem 公开 core/error/platform，mem_vm 公开 core/error/mem/platform/platform_fs/util_core；simd 公开
core/platform；container 公开 core/mem/container_intrusive、私有 platform；concurrent 公开
core/mem/mem_reclaim/container；queue 公开 core/mem；text 公开 core/mem/container、私有 platform；
obs 公开 core/error/time、私有 mem_reclaim。

本轮新增 component/依赖边/第三方/产品 API 均为 **0**，没有形成依赖环。表中的未来组合描述的是
ownership 方向，未修改 CMake，也不代表未来 owner 已实现或已签 PASS。

## 验证与终态记录

正文候选 tree `71fd3c2330b698844e7a92f97fd10319c0ab4cfd` 已通过以下实际验证；终态记录只增加本节和
audit marker，提交前同样只重验适用的轻量门，不重签任何 native/P1 证据。

```text
python scripts/validation-select.py plan --base 9864c187d4e6caff8606a176e78ba40d234fafae --candidate WORKTREE --output C:/public/project-hyle/_validation/frozen-inference-capability-audit/plan.json
python scripts/validation-select.py run --base 9864c187d4e6caff8606a176e78ba40d234fafae --candidate WORKTREE --scratch C:/public/project-hyle/_validation/frozen-inference-capability-audit/scratch --output C:/public/project-hyle/_validation/frozen-inference-capability-audit/result-body.json
git diff --check
git diff --cached --check
```

selector 实际执行 `python scripts/check-docs.py` 和 `python scripts/check-module-closure.py`，两者退出码
均为 0：前者检查 456 Markdown / 1298 relative links；后者检查 28 components / schema 1。
正文和终态使用的 Python 为 3.12.10，Windows 11 AMD64，主机 `kiro-workstation`。WORKTREE maintenance
receipt 绑定精确 tree/contract/toolchain/machine/selector 身份，但不复用 immutable native receipts。

| gate 组 | 数量 / selected-skipped | 实际理由与结果 |
|---|---|---|
| docs | 1 selected | mandatory documentation contract；PASS |
| closure | 1 selected | mandatory registry contract；PASS；registry 未改，无需 --previous |
| configure | 1 skipped | no product or factor test impact |
| factor correctness | 24 skipped | outside the change and its required consumer boundary |
| headers | 24 skipped | no public-header seam change for this factor |
| component-contract | 1 skipped | no global component/package seam change |
| delivery | 14 skipped | no delivery boundary change for this product |
| P1 | 1 skipped | no explicit frozen P1 card; sampling is never automatic；本轮预算本身禁止 P1 |

总计 2 selected / 65 skipped；selected 内无 build、native、performance。完整逐 gate JSON 与终态
`result-final.json` 留在上面的 checkout 外目录；底层 command logs/receipts 在 `.git/hyle-validation`。
正文 docs log SHA-256 为 `fb2749dc65de9e8733ffe281114c86b03bfa48e0946d12924d8dfa3396d5a238`，closure log
为 `bb153d7b8aba3df1b2cbc195c40607a192fef31b16b9ef46d9db3e64ab797bf7`。这些是文档/registry 证据，
不续签前两轮 selector 或性能审计。

只读终稿复核没有 material finding。另逐项核对：自动范围 19/19；68 个 source IDs 唯一；68 个 GitHub
commit-bound 源码链接与本地 HEAD、文件及行范围匹配；本地相对链接存在；五个参考 HEAD/branch/status
与起点完全一致，仍 clean。产品、API、测试、CMake、性能合同、registry 对 baseline 的 diff 为空。
唯一交付修改是本 task 与 tasks 索引。

LOC 在最终 immutable commit 用现有 `loc-report.py --module all-product-code --module-path include/hyle
--module-path src --commit <audit-commit> --self-check --json` 输出。此 explicit aggregate scope 只是
取得全库/自实现规模，不是新增 component，也不把全部产品代码冒称本轮工程单元。**本轮产品模块实现
LOC 为 0**；全库代码、自实现含/不含测试的数值随提交号在最终交付报告给出。

终态：`PROJECT AUDIT: CLOSED`。19 因子审计完成，确认缺口 0、有限 WATCH 1、合法 reopen 候选 0；
未处置本轮 blocker 0。按预算停止，不重开模块，不执行未来输入清单，不启动新性能采样。
