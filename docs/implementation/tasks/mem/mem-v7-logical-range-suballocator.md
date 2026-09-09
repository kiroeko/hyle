# Mem v7：动态逻辑区间子分配

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

## Closure card（`CLOSED`）

- component / owner：`mem` / payload-free logical range ownership；round：`mem-v7-logical-range-suballocator`；
  baseline：`67893204aad888f8e801c59ec6c149432a8e779e`。
- trigger：`MEM-V7-CROSS-DOMAIN-LOGICAL-RANGE`。2026-09 多消费者纵切审计确认，现有
  `FixedHeap` 需要真实 host pointer 并把 block metadata 放在 payload backing 中，`LifetimePlanner`
  只处理预先已知的 interval；两者都不能管理 opaque backing 中运行时变化的 `[offset,size)`。
- 主纵切：异构渲染/资源流送。Cycles device sub-pointer/delayed-free、AMD VMA/D3D12MA virtual block
  都要求先在大块 device backing 中取得 aligned logical range，再由 Compute/provider 建资源或提交 copy。
- 正交纵切：推理 serving。SGLang `RangePageAllocator` 与 pinned-buffer buddy allocator、vLLM block pool
  都重复使用动态 page/byte offset 分配；KV hash、eviction 和 request policy 不进入 Mem。
- 当前锁定来源：SGLang `dcebe8c4733a`、vLLM `51da0ca66c80`、Cycles 无 Git identity 的本地快照仅作
  辅助；OffsetAllocator `3610a7377088b1e8c8f1525f458c96038a4e6fc0`、VMA
  `3aa921224c154a0d2c43912bc88e1c42ce1f7607`、D3D12MA
  `1d86c1130f61453634b1df85782e1fecfd59a525` 为成熟算法/API 参考。正式证据位置见
  [跨产品审计](../project/public-api-consumer-portfolio-audit-2026-09.md#f01--mem-动态逻辑区间子分配确认)。
- 正常任务：创建一个固定 logical capacity 的 metadata owner；按不同 power-of-two alignment 动态取得
  contiguous range；以不可伪造的 allocation identity 归还；观察常数时间摘要；整块 reset/reuse。
- scope：payload-free owner、64-bit capacity/offset/size、显式 metadata allocator 与容量、aligned
  allocate、individual free、bulk reset、stale/double/wrong-owner 诊断、O(1) summary stats、move/destroy、
  单 owner/caller-synchronized 合同，以及 Mem 产品/三种 C++ 交付和用户选型文档。
- non-goals：分配真实 host/device memory、device/stream/fence、buffer-image granularity、defrag/move
  callback、KV/texture/descriptor policy、并发 wrapper、linear/ring/double-stack strategy、C/Native FFI、
  Compute/Graph、公共性能仪器或现有 `FixedHeap/LifetimePlanner` 重写。
- 直接依赖：现有 Mem 的 Core/Error/Platform 与 `AllocatorRef`；不得为 metadata 反向依赖 Container。
  直接消费者：未来 Compute device heap、推理 cache/backing owner、渲染 resource allocator；本轮用独立
  public C++ probe 证明，不修改仍为 `QUEUED` 的 Compute/Graph。
- 性能 tier：`P1`。P0 先裁决复杂度、metadata bytes、allocation/copy/lock；P1 primary 是固定种子的
  mixed resource churn，在 alignment=1/32-bit 能力交集上对 OffsetAllocator 的同合同 adapter：peer 侧必须
  同样计费 fallible result、move-only RAII、owner/lease 独立寿命、generation、固定 metadata 与精确常态
  counters。raw peer 自然路线只能作较弱语义的诊断地板。holdout 是变化 alignment 与大于 4 GiB capacity
  的 HYLE correctness/resource trace。默认 5% non-inferiority margin；VMA/
  D3D12MA 只在能隔离相同 virtual-block 合同和依赖身份时才进入计时，否则仅作算法/API 参考。
- 仪器/机器：`closure-correctness-v1`；P1 使用 `closure-perf-v1` 的 W1-WIN 锁定身份。正确性先覆盖
  W1-WIN；产品候选再覆盖 W1-WSL 与两台可用 macOS 中至少一台，另一台只在平台差异或前者不可用时
  使用。文档/状态改动不触发 P1。
- 调查预算：能力/API/算法最多 3 小时，性能调查最多 90 分钟；至多两次产品纠正、一次反序复现和一次
  codegen/profile 归因。范围变化退回 `SCOPED/AUDITED`，不能增加 benchmark-only 策略开关。
- 预声明验证：public-header self-contained；zero/invalid/overflow/OOM；alignment；fragment/coalesce/reset；
  stale/double/wrong-owner；metadata 上限；随机 model/property；move/destroy；ASan/UBSan，TSan 仅在实现
  引入共享状态时适用；Mem focused tests；source/static/shared public consumer；direct consumer probe；
  七问、独立红队、P1 和固定治理命令。
- stop / blocker：有限 `FIX` 全部完成且七门通过则重新 `CLOSED`。若 64-bit aligned range 的正确性、
  metadata bound 或同合同 P1 在预算内不能裁决，进入 `BLOCKED`；不能用弱化语义、无限 benchmark
  姿势或修改现有已验收 allocator 来绕过。

## 能力与形状审计

| 能力族 | 准入旅程/来源 | 语义与生命周期 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|
| payload-free virtual range owner | VMA/D3D12MA virtual block；Cycles device sub-pointer | owner 只持 metadata，不碰 opaque backing | 无；`FixedHeap` 把 metadata 放进 caller storage | `MISSING` | `FIX` |
| 64-bit aligned dynamic allocation | GPU buffer/heap、pinned staging、KV/page backing | 返回 `[offset,size)`；alignment 为非零 2 的幂 | 无通用动态路线 | `MISSING` | `FIX` |
| individual release / coalesce | resource completion、cache eviction | move-only lease 析构/`reset()` 归还并合并相邻 free range | 无 | `MISSING` | `FIX` |
| phase-wide invalidation | model/device teardown、heap generation replacement | owner `reset()` 一次失效全部逻辑 allocation；旧 lease 后续销毁安全 no-op | `LifetimePlanner` 只覆盖静态 interval | `MISSING` | `FIX` |
| bounded metadata / pressure facts | 长寿 server 和 device heap | create 时显式 live-allocation 上限；热路不增长、不锁、不分配 | 无 | `MISSING` | `FIX` |
| fragmentation observation | heap sizing/debug | O(1) summary；显式 O(N) detailed scan | 无 | `MISSING` | `FIX` |
| concurrent allocation | 多 worker 直接争抢同一 heap | 需要同步与进度合同 | caller-synchronized owner + 上层 pinned owner | `OUT_OF_SCOPE` | `DEFER` |
| linear/ring/stack strategy | frame upload、FIFO fence、phase scratch | 生命周期与默认 free-list owner 不同 | `FixedArena`/`StackArena` 或 future concrete consumer | `ROUTED` | `KEEP` |
| defragment/move callback | movable resource heap | 需要 device copy、descriptor 与 fence owner | Compute/provider | `OUT_OF_SCOPE` | `REJECT` |

新公开入口全部归入 `range_allocator.h`：`RangeAllocatorOptions`、两份 stats、move-only
`RangeAllocation` 与 move-only `RangeAllocator`。`pool.h` 只作为 Mem pool/heap 产品 façade 聚合该头；
用户也可直接 include 精确头。没有保留旧兼容入口，也没有 benchmark-only strategy、bin、search-limit 或
placement knob。

### 成熟 peer 的取舍

- OffsetAllocator 以外置 node metadata 和 TLSF-like 两级 bitmap 提供极低常数的 32-bit、无 alignment
  offset 分配；它证明 metadata 与 payload 分离和 O(1) common lookup 的价值，但其固定 32-bit offset、
  allocation count、显式裸 `free` 与无对齐参数不能直接成为 HYLE 公共合同。初始开发夹具直接把它与
  HYLE RAII 路线相排，raw peer 约快 14--20%；生命周期/安全检查/统计矩阵复核后该行被降为
  `NATURAL_ROUTE`，不得据此改坏公开 API。正式 P1 由完整计费 adapter 对齐这些语义。
- VMA/D3D12MA virtual block 提供 64-bit offset、alignment、individual free、clear 与统计；这构成能力/API
  下界。VMA 的 aligned TLSF 搜索在碎片化候选中允许遍历，因此不能据其名称宣称所有 aligned request
  都有硬 O(1) 最坏上界。
- HYLE 采用 64-bit 两级 size-class bitmap：保守 `size + alignment - 1` 能命中时直接 O(1) 取得保证可放置
  的 range；只有保守尺寸溢出或不存在、但地址相关的更小 range 仍可能适配时，才扫描 free ranges 保持
  完整语义。free 与相邻 coalesce 为 O(1)。这比把 alignment padding 永久计入 allocation 更节省 backing，
  也比为了 benchmark 隐瞒可用小洞更诚实；worst-case allocate 明确为 O(F)。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 | 结果 |
|---|---|---|---|---|
| `MEM7-F01` | 缺 payload-free 64-bit logical-range owner | 新增 range allocator/lease/stats；固定 metadata | 边界、alignment、OOM、随机 model | `PASS` |
| `MEM7-F02` | individual free、bulk reset 与 stale lifetime 缺少自然 C++ 形状 | RAII lease、generation invalidation、owner/lease 独立控制块寿命 | free/coalesce/reset/move/destroy | `PASS` |
| `MEM7-F03` | Mem 产品、安装和三交付尚不可发现 | façade、CMake source、focused test、package consumer | header/source/static/shared | `PASS` |
| `MEM7-F04` | 用户不知道它与 FixedHeap/LifetimePlanner/device policy 的边界 | architecture/user/usage 文档及最短 recipe | docs checker + usability probe | `PASS` |
| `MEM7-F05` | 无同合同工程性能证据 | 固定 mixed-churn contract；OffsetAllocator 算法只在 alignment=1/32-bit 交集并经完整 lifecycle adapter 对拍 | controls + 双顺序 5% NI；raw route 不签排名 | `PASS`；8/8 strata 的 5% NI 通过 |

本表冻结后不扩张到 F02 Sched、F03 IO、Compute/Graph、并发 wrapper、device policy 或 benchmark-only
策略。验证发现上述语义内部缺陷可在原合同内修；能力范围变化必须退回 `SCOPED/AUDITED`。

## API / usability freeze

普通 C++ 调用固定为：

```cpp
HYLE_ASSIGN_OR_RETURN(
    auto ranges,
    hyle::mem::RangeAllocator::create(
        8ull << 30,
        {.max_allocations = 4096,
         .metadata_allocator = hyle::mem::system_allocator()}));
HYLE_ASSIGN_OR_RETURN(auto weights, ranges.allocate(768ull << 20, 256));
create_device_view(backing, weights.offset(), weights.size());

deferred_release.push(std::move(weights)); // completion owner 最终析构 lease
ranges.reset();                            // phase-wide logical invalidation
```

- `RangeAllocatorOptions::max_allocations` 是必填且必须非零；不提供会悄悄保留数百 KiB metadata 的任意默认。
  `metadata_allocator` 默认 system allocator，但若替换为借用 allocator，它必须长寿于 owner 及所有仍存在的
  lease handle。
- `RangeAllocator::create(capacity, options)` 拒绝零 capacity、零/不可表示的 live 上限和 invalid metadata
  allocator；metadata 在 create 时一次性建立，后续 allocate/free/reset 不向 backing 申请内存。
- `allocate(size, alignment=1)` 拒绝零 size 与非 2 次幂 alignment；空间或 live-slot 耗尽返回
  message-less `ResourceExhausted`，避免压力热路为错误文本分配。它不分配或映射真实 payload。
- `RangeAllocation` 不可复制、可移动，公开 `valid/offset/size/reset`；析构自动 individual free，`reset()`
  幂等。没有接受任意 owner/token 的公共 `free`，所以 wrong-owner 与 double-free 由类型形状消除。
- owner `reset()` 将 live ranges 一次失效并开始新 generation；旧 lease 的 offset 立即不再拥有 backing，
  其后 `reset`/析构只释放 control reference，不会误归还新 range。owner 可在 lease 前析构；控制 metadata
  延迟到最后一个 lease 释放，因而没有悬挂 pimpl。
- 整个 family 是 caller-synchronized：同一 control 上的 allocate、reset、stats、lease destroy 必须外部串行。
  普通路径没有 mutex/atomic/syscall；异步 device completion 应把 lease 移交到 pinned owner，而不是让任意
  worker 并发析构。
- `stats()` 为 O(1)，报告 capacity/allocated/free/peak/live/max/metadata/outstanding handles；
  `detailed_stats()` 显式 O(N)，另报 free-range count 与 largest free range。peak 不因 reset 清零。
- `RangeAllocator` 是一指针公开对象；`RangeAllocation` 的 `offset/size` 是以 `valid()` 为前置条件的
  O(1) observer，正常读取不重复执行整套 generation 验证。source/static/shared 均使用
  同一个公开 C++ 头和 `hyle::mem` target，不创建 C/Native FFI。

### P0 成本上界

- create：O(M) 初始化最多 `2 * max_allocations + 1` 个 node slot，三次 metadata allocation；失败完整回滚；
- allocate：常见两级 bitmap O(1)，最多拆成 prefix/allocation/suffix；地址相关 fallback 最坏 O(F)，不拷贝
  payload、不锁、不 syscall、不热路分配 metadata；
- individual free：O(1) physical-neighbor coalesce + bitmap/list 更新；bulk reset O(M) 重新建立空闲 node inventory，
  不逐个访问外部 payload；
- summary O(1)，detailed stats O(N)；metadata 上界为 control + `(2M+1) * (node + free-index)`，由 API
  精确报告，不能伪称只与 capacity 有关；
- lease 生命周期只有 caller-synchronized 非原子引用计数与 node/generation 检查；没有 hidden shared_ptr。
  已证明的私有索引不变量在 Debug 中诊断、优化构建中作为 optimizer precondition，不让对手关闭的
  内部 assert 变成 HYLE 独有的常态分支；可恢复输入仍始终返回 `Status/Result`。

## 七问预裁决

1. **第三方：**产品实现不依赖第三方；OffsetAllocator/VMA/D3D12MA 只作锁定 peer/算法参考。其许可证与
   commit 记录在 card，未来升级 peer 不自动改变产品或继承性能 verdict。
2. **功能/消费：**渲染 resource heap 为主纵切、推理 page/backing 为反偏置纵切；六个 MUST family 经
   `MEM7-F01..F04` 后完整。device/fence/KV policy 在其 owner 路由，不向 Mem 下沉。
3. **下层复用：**使用 Core compiler/assert、Error Status/Result 和 Mem `AllocatorRef`；`platform` 是 Mem
   既有产品边但本能力无 OS 调用。固定 metadata 不反向依赖 Container，也不复制上层 queue/scheduler。
4. **性能：**复杂度和成本如 P0；P1 只对 alignment=1/32-bit same-contract mixed churn 与完整计费的
   OffsetAllocator adapter 做 5% NI。raw peer API 的较弱语义只作 natural-route 诊断；aligned/64-bit 是
   HYLE correctness holdout，不能拿 peer 缺失语义排名。
5. **FFI：**`N/A/CLEAN`；没有旧 C header/schema/bridge 或未修饰动态符号，本轮不为假想语言绑定擦除
   move-only C++ lifetime。
6. **交付：**`mem` 现有 source/static/shared 产品承载新 compiled implementation；新增 public header 和
   source 由中央组件机制发现，不增加组件边或第三方 package dependency。
7. **易用性/文档：**显式 owner + RAII lease 对应普通资源生命周期；guide 必须写清 range 不拥有 bytes、
   reset 后 offset 失效、caller synchronization、metadata allocator 寿命及与 FixedHeap/LifetimePlanner 的选型。

Audit verdict：全部 MUST 已成为五项有限 FIX，API、成本、peer、normal/holdout shape 和停止条件均已冻结，
允许进入产品实现。

### 撤回的 candidate 1

`f13ee33bb0ecc2a882f32a7a7403197f78c2055a` 在 W1-WIN、W1-WSL 与 Windows 三交付通过，但
M1-MAC/AppleClang 21 `-Werror` 编译在四处 `HYLE_ASSUME(is_free(...))` 报 `-Wassume`：Clang 不接受
assumption expression 内潜在有副作用的 helper call。它没有生成 macOS 产品 binary，不能成为终局候选，
也不继承开发性能结果。registry 已退回 `AUDITED`；唯一追加修复是先物化纯布尔值再交给 optimizer
precondition，范围、公开 API、算法、layout、workload 和 peer 均不得变化。

### Candidate 2

不可变产品候选为 `d3f732979f396aa5651e5eb81c228fdc73b7c166`。它相对 candidate 1 只把四处
helper-call assumption 展开成相同的纯字段表达式；W1-WIN/MSVC 与 W1-WSL/GCC 14 定向测试及开发
性能重放均未回归。后续只允许补精确候选证据；任何产品变化都再次退回 `AUDITED`。

## Pre-candidate implementation receipt

- 新实现以 32-byte partition node 和两级 bitmap 管理 64-bit range；free-state 复用空闲链身份，不增加
  常态 flag/padding。private structural invariants 在 Debug 中检查、优化构建中成为 optimizer
  precondition；可恢复输入、容量和 OOM 仍走 `Result`。
- W1-WIN/MSVC 19.50 Release 全 Mem 226/226（加入额外 lifetime 用例后为 228/228 的下一次候选重放）、
  新增 RangeAllocator 14/14 Debug、public-header self-contained，以及 selected Mem source/relocated
  shared/relocated static 三交付均通过。
- W1-WSL/GCC 14.2 `-Werror` + UBSan 与 ASan 分别通过完整 Mem 和 public-header 两项；GCC 首轮发现的
  未使用 const helper 已删除。实现没有共享状态，TSan 按 caller-synchronized 合同不适用。
- 开发 P1 八种 execution/allocation/placement 反转均显示完整语义 HYLE 路线不劣于完整计费 adapter；
  这不是正式样本。初始 raw Offset 路线的明显领先已按语义矩阵正确降级，未用较弱 API 反向塑造产品。

## 预声明验证命令

Windows 先在独立 build directory 配置 selected Mem Release，构建并运行 `hyle_mem_test`；新增 test filter
覆盖 create/OOM/overflow/alignment/split/coalesce/reset/stale/move/destroy/model property。随后运行现有 Mem
suite、public-header self-contained、selected source/static/shared package journey 与直接 standalone
consumer。候选阶段再以 archive 验证 WSL GCC ASan/UBSan 和一台 macOS AppleClang；实现没有共享状态，
TSan 记为 `NOT_APPLICABLE`，不能写成 `PASS`。

P1 只在 API/correctness 候选固定后启动，使用独立 opt-in target、锁定 peer identity、固定 seed/trace、
相同成功/失败处理和 setup/teardown charging；peer adapter 必须实际执行与 HYLE 相同的 Result、RAII、
owner/lease、generation、stats 和 metadata-bound 工作，而不是只把返回类型包一层；记录 A/B、B/A、A/A、
known-delta、timer floor、placement 与资源 telemetry。OffsetAllocator 不支持的 alignment 与 >4 GiB 行只作
correctness/resource holdout。

每个状态迁移和终局至少执行：

```text
python scripts/check-module-closure.py --previous-ref <registry-change-before>
python scripts/check-module-closure.py --write-status
python scripts/check-module-closure.py
python scripts/check-docs.py
git diff --check
```

## 终局候选验证

候选 2 的产品 tree 保持 `16ac17e2fb501abeaeac623785ccf9daca1b29de`。状态和证据提交没有改变
公开头、实现、测试 workload 或 benchmark contract。

| 风险面 | 精确候选或可继承证据 | 结果 |
|---|---|---|
| W1-WIN correctness | MSVC 19.50 Release/WERROR；`hyle_mem_header_self_contained_test` + 完整 `hyle_mem_test` | `2/2 PASS`；RangeAllocator 14 个具名用例包含在完整 suite |
| W1-WIN 三交付 | detached candidate 的 source、relocated shared、relocated static Mem package consumer | `3/3 PASS`；安装树包含 `hyle/mem/range_allocator.h` |
| W1-WSL sanitizer | candidate 1 完整 Mem/header 的 GCC 14.2 ASan 与 UBSan；candidate 2 对受影响 RangeAllocator 14/14 作 UBSan 重放 | `PASS`；candidate 2 只改变四个等价 optimizer assumption 表达式 |
| M1-MAC portability | MacBook Air arm64、AppleClang 21、Release/WERROR，从 candidate 2 独立 worktree 构建完整 Mem/header | `2/2 PASS`；candidate 1 的 `-Wassume` 阻塞已消失 |
| TSan | 产品明确 caller-synchronized，未引入共享可并发状态 | `NOT_APPLICABLE` |
| P1 | W1-WIN、15 独立进程、8 个 execution/allocation/placement strata 与全部 controls | qualification/performance `PASS`；8/8 的 5% NI `PASS` |

P1 绑定
[`benchmarks-mem-v7.md`](../../evidence/performance/benchmarks-mem-v7.md)：HYLE 的八个 median advantage
为 `1.0310–1.0432`，全部点估计朝 HYLE 方向；因多数置信区间跨 parity/WIN 的 5% 边界，exact 聚合
保留 `INCONCLUSIVE`，没有夸大为普遍胜出。raw OffsetAllocator 的弱语义路线没有进入当前性能声明。

## 终局红队

与实现过程分离的只读检查再次攻击公开 API、metadata 证明、溢出、生命周期、依赖、交付和性能比较：

- `2*M+1` nodes 覆盖 M 个 live allocation 下每次最多 prefix/suffix 两个 split；create 的三处分配逐点
  OOM 回滚，metadata byte 加法/乘法和 32-bit node sentinel 均有表示范围检查；
- alignment rounding、`offset + size` 与最大 64-bit capacity 不越界；快速 size-class 搜索不能证明可放置
  时走完整 physical fallback，不会为保持 O(1) 宣传而错误拒绝可用 range；
- move-only lease 消除了 wrong-owner/double-free 入口；generation 阻止 stale lease 释放复用 node，owner
  销毁后 control 由最后一个 lease 回收，reset 跨 generation 仍准确保留 outstanding handle references；
- caller-synchronized、borrowed metadata allocator lifetime、observer 前置条件、payload-free/device-policy
  边界均在公开头和 guide 显式可见；没有 benchmark-only knob、常态锁/原子/syscall 或隐藏增长；
- `mem` 的 Core/Error/Platform 公共边、compiled source、header ownership、安装导出与三种 C++ journey
  一致；没有新增或遗留 C/Native FFI；
- 正式 peer adapter 对双方计入相同正常生命周期；32-bit/alignment 能力差异只作 correctness holdout，
  没有苹果对橘子排名。未发现未映射公开入口或 medium/high 风险遗留。

## 七门与裁决

| Gate | 证据 | Verdict |
|---|---|---|
| scope | 六个 MUST family、两个真实产品纵切、有限 `MEM7-F01..F05` | `PASS` |
| architecture | payload-free owner 与 device/KV policy 分离；依赖向下且无 Container 反向边 | `PASS` |
| correctness | 边界/OOM/property/lifetime、Windows、WSL sanitizer、macOS portability | `PASS` |
| delivery | public header + source/shared/static relocated consumer | `PASS` |
| usability_docs | 自然 owner/RAII recipe、选型、同步/reset/allocator-lifetime 陷阱 | `PASS` |
| red_team | 上述分离红队，无未处理 medium/high finding | `PASS` |
| engineering_performance | P0 成本结构 + qualified W1-WIN 8/8 5% NI | `PASS` |

LOC 由 `python scripts/loc-report.py --module mem --commit d3f732979f396aa5651e5eb81c228fdc73b7c166
--self-check` 计算并自检通过：全库 503,751 行，自实现不含测试 90,524 行，Mem 实现 8,266 行。

最终裁决为 `CLOSED/FROZEN`：本轮真实缺口已实现，七门全部通过。Formal 独立硬件声明仍为
`PENDING`；当前只签发上述 W1-WIN 同合同工程非劣结论。只有可复现 correctness/security/UB/lifetime
缺陷、具名消费者证明的新通用能力缺口、公开 API/layout/算法/依赖/交付变化，或合格同合同证据显示
超过 5% 的 material regression/更优复杂度档，才允许创建新 card 重开；无新事实的“再看一次”和旧
benchmark 点值不构成 trigger。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
