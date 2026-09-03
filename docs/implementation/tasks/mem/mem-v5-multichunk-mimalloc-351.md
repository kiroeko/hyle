# Mem v5：multi-chunk lifetime 与 mimalloc 3.5.1 定向完成

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`mem` / Mem / `mem-v5-multichunk-mimalloc-351` / `REOPEN`；
- baseline：`ae01203b35d7f6d0b9c76117a1a50fc837c07772`；前轮为
  `mem-v4-reclamation-facade`；
- trigger IDs：`MEM-LIFETIME-MULTI-CHUNK-01`、`MEM-MIMALLOC-351-01`；
- trigger evidence：[`current product replay`](mem-current-product-replay-2026-09.md)；
- 具体新事实：
  1. `LifetimePlan::instantiate()` 对每个 resource 只创建一个 `peak_bytes` Buffer，无法表达已经由
     `MemoryResource::maximum_allocation` 声明、且当前 ggml graph allocator 实际支持的多 physical
     allocation placement；
  2. shipping host allocator 仍硬锁 mimalloc v3.5.0，而 stable v3.5.1 改变实际 allocator 源码、
     hot-path codegen、平台适配与 correctness，命中 direct-dependency trigger；
- 最小受影响面：base `mem` factor 的 lifetime-plan API/实现、mimalloc vendored/lock/custom build，及由
  两者改变的 tests、三种 C++ delivery、architecture/user/performance/task 事实；
- 明确不受影响：`mem_vm`、`mem_reclaim` 的算法与 API，普通 Buffer layout，Tensor/Graph/Compute 策略，
  backend usage/reset/host-pointer import、async completion、hazard pointer、Native FFI/C API 和公共性能仪器；
- 这一段记录当时 `REOPENED` 的授权边界；后续合法状态与施工授权由下方 closure card、audit 和 registry
  共同给出，不能回头把 trigger 登记阶段当成产品实现授权。

## Closure card（`SCOPED`）

- component / owner / round ID：`mem` / Mem / `mem-v5-multichunk-mimalloc-351`；
- task kind：`REOPEN`；baseline commit：
  `ae01203b35d7f6d0b9c76117a1a50fc837c07772`；
- 进入原因与 demand/defect：`MEM-LIFETIME-MULTI-CHUNK-01`、`MEM-MIMALLOC-351-01`；证据见
  [current product replay](mem-current-product-replay-2026-09.md)；
- 真实 consumer task / normal workload：
  1. 推理 graph 的静态 lifetime 集合需要先得到可检查的 physical-backing/offset placement，再在具有有限
     `maximum_allocation` 的 host/device-like `MemoryResource` 上实例化；同一 alias root 必须保持在一个
     backing 内，多个 resource slot、半开 lifetime、no-reuse/output 和零大小请求仍保持原合同；这是由
     ggml graph allocator 的真实 multi-chunk 路线和 HYLE package journey 共同准入的
     `NORTH_STAR_JOURNEY`，planner 是 cold P0 路径；
  2. source、static SDK 与 shared SDK 消费者通过 `SystemAllocator`、`AllocatorRef`、PMR 或独立 `Heap`
     完成 64 个 24 B--3 KiB、16/32/64-byte alignment 的成功 allocate/touch/reverse-free lifecycle；这是
     `samples/05_installed_sdk`、Mem package 与 Container/Sched 默认分配路线的普通 shape；
- 范围：base `mem` factor 的 lifetime-plan public C++ API、O((N+E) log N) planner/instantiate 实现与
  tests；private mimalloc vendored snapshot/lock/custom target/明确平台配置；只重放受影响的 host allocator
  P1、三种 C++ delivery、依赖/license/symbol、architecture/user/task 事实；
- 非目标：`mem_vm` 或 `mem_reclaim` 算法/API；普通 Buffer layout；Tensor/Graph/Compute backend policy；
  reset/host-pointer import、async completion、hazard pointer；Native FFI/C API、稳定 C++ ABI、公共性能仪器、
  全 Mem benchmark 重做或 Formal certification；
- 直接依赖：`PUBLIC core error platform`，private locked mimalloc；直接产品消费者：`mem_vm`、Container、
  Util-runtime、Sched、IO、Compute、Graph、Mem product façade 与 source/static/shared package journey；只有
  package sample 当前直接调用 `LifetimePlanner`，ggml 是 capability/shape peer 而非 HYLE 产品依赖；
- tier：聚合为 `P1`，因为 mimalloc v3.5.1 material allocator/codegen change 进入正常 host-allocation 热路；
  multi-backing planner 只签 P0 complexity/resource verdict，不为 cold plan 构造纳秒 benchmark；
- material margin / resource budget：host allocator 对同一 v3.5.1 binary 的 direct mimalloc same-contract
  route 为 5% non-inferiority；planner 保持 build O((N+E) log N)、O(N) scratch，query/view O(1)，不产生
  byte/page-proportional table，每个 physical backing 不超过 provider max，root/alias group 不跨 backing，
  backing/placement metadata O(N)；不承诺全局最优 bin packing；
- 总调查预算：一个工程日（上限 8 小时）；其中 API/算法/测试 4.5 小时，vendor/build/correctness 1.5 小时，
  P1 45 分钟，delivery/docs/red-team/receipt 1.25 小时；超时必须按协议裁决，不能以“继续研究”续期；
- 最多实验/修改：P1 一次资格化采样；若异常，只允许一次顺序反转复现、一次 codegen/profile 机制诊断、
  一个由性能证据触发的内部产品修复和一次候选重跑。上述次数不把已冻结的两个 correctness/capability
  FIX 冒充性能实验；
- machine profile：W1-WIN 是 P1 工程签名机；W1-WIN、W1-WSL、M1-MAC 都执行适用 correctness、
  sanitizer/build/delivery；编译用已登记工具链，WSL 使用 GCC 14、`-j2`；
- instrument profile：P1 用 `closure-perf-v1`；planner 的 P0 structural/resource 检查随 correctness 证据；
- strong peers：
  1. standalone ggml `d4716378882593333721eb33f153144b6885caf2` 的 graph allocator multi-chunk
     `{chunk, offset}` / backend maximum-buffer-size shape，只作为能力、边界与复杂度 peer；
  2. mimalloc v3.5.1 tag peeled commit `34fbd7e7cd4627424490afe19b20f8066bfc537d`（2026-09-01），
     作为 private mechanism 与 host-allocation lower bound；v3.5.0 commit
     `18b08671c9302247bfb682286e6bf3cc1773f801` 只作 upgrade provenance/方向性诊断，不与不同 binary
     placement 结果混成产品 SOTA gate；
- 无同合同 peer：multi-backing plan 没有统一 API 的外部 SOTA 排名；以 provider max、alias/lifetime
  correctness、上述复杂度/metadata 预算和 deterministic replay 为 gate。禁止声明最少 backing、最小 peak、
  device allocation latency、end-to-end token/s、所有 allocation size/thread/platform 或 blanket Mem SOTA；
- API 语义档位与选择前提：单一 `resource-aware static lifetime plan` 档；caller 在 build 时借用 resource
  properties，plan 不持有 provider，可先查询 placement/backing plan，再用兼容 resources 实例化；实时动态
  图、runtime grow/release 使用既有 Arena/Pool/Buffer/CompositeBuffer，不塞入本 planner；
- 常态成本合同：build 冷路径 O(N) metadata/scratch、无 provider allocation；instantiate 每个 planned
  backing 一次 provider allocation，失败用 `Result<...>` 回滚全部已建 owner；plan query 与 instantiated
  view/backing lookup O(1)、无 allocation/lock/atomic/syscall；host allocator 不增加 per-allocation metadata、
  copy、lock、queue、wake 或 syscall，只保留既有 compile-time/runtime erasure seam；
- correctness gates：ordinary/single/multiple backing、exact max/alignment、reuse、alias/no-reuse/output、dense
  multi-resource、zero/empty、overflow、oversized root、incompatible instantiate resource、partial OOM cleanup、
  move/reset/destroy/property replay；mimalloc version/cross-thread free/realloc/alignment/PMR/stats/collect/NUMA
  reserve；Release + ASan/UBSan/TSan 的适用 Mem tests；
- delivery/build gates：selected Mem source embedding、relocated static SDK、relocated shared SDK；dependency lock、
  vendored tree/archive identity、offline source bundle、license install、private symbol/visibility 与 only-selected
  C-language enablement；direct consumers 至少 package Mem、Container allocator compile/run 和 mem_vm API surface；
- performance gate：冻结的 `isolated-heap-lifecycle-64` normal aggregate，HYLE `Heap::allocator()` 对同一
  mimalloc v3.5.1 binary 上的 runtime-erased PMR heap route，双方相同 heap ownership、size/alignment/touch/
  reverse-free、setup/warmup/teardown charging；15/31 process samples、candidate/peer execution 与 allocation
  order、placement、A/A、known 5%/10% delta、timer floor 资格化，5% non-inferiority；
- 预期输出：自然 multi-backing API/实现/测试/recipe；精确 v3.5.1 vendor+lock+配置决策；候选与三平台、三
  delivery、P0/P1、红队证据；只更新当前事实，不重写历史 task/evidence；
- 停止条件 / 可接受限制：两个 FIX、任一 correctness/sanitizer/delivery/lock gate 或一次允许的性能修复后
  的 5% P1 仍失败即 `BLOCKED`；若 v3.5.1 在支持平台无法保持明确 baseline/THP/ISA 交付合同也
  `BLOCKED`，不静默回退 3.5.0 或 OS allocator。允许 deterministic next-fit backing boundary + best-fit
  lifetime-slot reuse，不要求 NP-hard 全局 bin packing；Formal 与端到端推理排名保持 `PENDING`。

## SCOPED verdict

两个 trigger 已压缩为上述有限范围、预算、normal workload、peer、适用 gate 与停止条件。允许进入只读
capability/API/实现审计；以下 audit 冻结以前未改产品代码、依赖锁或 benchmark。

## Capability-family coverage map（`AUDITED`）

| Family | admitted journey / evidence | required semantic variants | lifecycle / ownership / failure | current route | coverage | frozen action / negative space |
|---|---|---|---|---|---|---|
| static lifetime placement | ggml graph reserve；HYLE package | half-open、alias、no-reuse/output、dense resources | plan 不持有 tensor/provider；错误为 `Result` | indexed best-fit reusable slot | `PARTIAL` | `FIX` resource-aware multi-backing；不做动态 graph allocator |
| physical backing inspection | provider max；先 measure 后 allocate | resource/backing/local offset 三层 | plan 可独立于实例存在 | 每 resource 只有 peak/alignment | `MISSING` | `FIX` backing count/plan 与 placement backing ID |
| transactional instantiation | host/device-like open provider | same/equivalent provider、partial OOM | owner 持有全部 Buffer；view 借用 | 每 resource 一次 Buffer | `PARTIAL` | `FIX` 多 Buffer、兼容性检查与全回滚 |
| host system/isolated allocation | SDK、Container/Sched、mimalloc | compile-time、runtime-erased、PMR/Heap | cross-thread free；exact sized free；nullptr OOM | v3.5.0 + `mi_free` | `PARTIAL` | `FIX` 3.5.1，并把已知 size 交给 `mi_free_size` |
| allocator policy/retention | `system_collect`、HugeMapping、env | ordinary THP、explicit huge/NUMA | 进程启动前配置；无隐式 owner | vendor default + HYLE explicit routes | `PARTIAL` | `FIX` 保留旧 THP=1 baseline；不继承 FULL retention |
| package/dependency identity | source/static/shared C++ | selected closure、offline、license、symbols | mimalloc 是 private mechanism | v3.5.0 lock/vendor | `PARTIAL` | `FIX` 3.5.1 全身份与三交付重签 |
| 其余 Mem 累积产品 | [current replay](mem-current-product-replay-2026-09.md) | allocator/pool/VM/resource/buffer/reclaim 等 | 各自既有 owner | 28 个 public headers / 3 factors | `COMPLETE/ROUTED` | `KEEP`；本轮不得顺手重开 |

## Capability / gap ledger

| ID | class | exact source / normal shape | owner / failure / concurrency | complexity/resource target | current | action | verification / negative space |
|---|---|---|---|---|---|---|---|
| `MLP-01` | `MUST` | ggml multi-chunk；bounded `MemoryResource` | build 只借 resource facts；invalid/dense/max 失败可见 | O((N+E) log N)、O(N) | `MISSING` | `FIX` | bounded single/multi-resource、oversized root |
| `MLP-02` | `MUST` | inspect then allocate graph storage | placement 指向 resource-local backing | query O(1)，metadata O(N) | `MISSING` | `FIX` | exact boundary/alignment/alias/replay |
| `MLP-03` | `MUST` | instantiate with equivalent provider | `PlannedBuffers` 独占；partial OOM 全回滚 | one provider allocation/backing | `PARTIAL` | `FIX` | compatible/stricter/smaller max、OOM/move/reset |
| `MLP-04` | `MUST` | zero、empty、10k interval stress | zero view 仍绑定 resource；无 recursion | 同上，无 byte/page table | `PARTIAL` | `FIX` | zero-only backing、overflow、determinism |
| `MIM-01` | `MUST` | upstream v3.5.1 exact source | Mem 语义，mimalloc mechanism，Build artifact | 无额外常态 metadata | `PARTIAL` | `FIX` | commit/tree/archive/vendor/license/version |
| `MIM-02` | `MUST` | HYLE distributable baseline | 启动前 env override；跨线程 free | 不暗升 ISA/RSS policy | `PARTIAL` | `FIX` | explicit CMake definition、三平台/build metadata |
| `MIM-03` | `MUST` | exact-sized System/Heap/PMR free | 调用方带回原 size/alignment | 无 per-block HYLE metadata；直接 sized free | `PARTIAL` | `FIX` | API compile test、alloc/realloc/free、P1 pair |
| `MIM-04` | `MUST` | source/static/shared normal consumer | vendor type 不进入 ordinary API | shared 一次 import，static direct vendor call | `PARTIAL` | `FIX` | selected/source/static/shared/symbol/offline |
| `MLP-G1` | `GENERAL` | globally optimal bin packing | NP-hard policy，不属于 API correctness | 不承诺最少 backing/peak | `OUT_OF_SCOPE` | `REJECT` | deterministic next-fit boundary 明示 |
| `MIM-G1` | `GENERAL` | vendor option registry/secure heap | vendor-specific policy | 不扩大 HYLE public surface | `ROUTED` | `KEEP` | 启动 env 或独立未来 consumer trigger |

### Public-surface trace 与有限 `FIX`

- `LifetimeFlag`、`LifetimeRequest` 及其 half-open/alias/no-reuse 语义保留；
- `MemoryPlacement` 增加 resource-local `backing_index`，在 64-bit ABI 中复用原 padding，不为正常 request
  另建 side table；`offset` 始终是该 backing 内偏移；
- `LifetimeResourcePlan` 冻结为 `{peak_bytes, backing_count}`：`peak_bytes` 是本 resource 所有 physical
  backing 实际计划 size 之和；原来跨 backing 已失去意义的单一 alignment 字段移除；
- 新增 `LifetimeBackingPlan {size, alignment}` 与
  `LifetimePlan::backing_plan(resource_slot, backing_index)`；alignment 是计划要求的最小 alignment，实例
  provider 可以提供更强 alignment；
- `LifetimePlanner::build(requests, resources)` 是唯一普通 build：resource span 与 dense slot 一一对应，
  build 期间读取稳定 properties 但不保留 ref/provider；删除无约束的单参数入口，避免先产出无法实例化的
  “成功”计划；
- `LifetimePlan::instantiate(resources)` 可接受同一个或等价 provider；不要求 domain/对象 identity 相同，
  但逐 backing 重新验证 ref、minimum alignment 与 maximum allocation。更强 alignment/更大 max 合法，
  更小 max 返回 `OutOfRange`；
- `PlannedBuffers::backing_count(resource_slot)` 与双索引
  `backing(resource_slot, backing_index)` 取代含糊的单索引 backing；request `view()` 仍为 O(1)；
- zero-only dense resource 具有一个 size=0 backing，使它的 placement/view 继续绑定显式 resource；空 request
  只接受空 resources 并产生零 resource/backing；
- `SystemAllocator::deallocate(ptr, size, alignment=...)` 取消 `size=0` 默认值，使公开形状与既有 exact
  sized-free 合同一致；System runtime erasure、Heap、PMR、system resource 与 static/shared seam 在已知 size
  时调用 `mi_free_size`，不新增用户可见 allocator 类型；
- mimalloc 固定 v3.5.1；HYLE 的自定义 `static.c` target 不执行 upstream CMake，故明确在 non-Android
  保留 `MI_DEFAULT_ALLOW_THP=1`（旧 HYLE 行为，用户仍可在启动前设 0/2），不私自复制 v3.5.1
  `MI_OPT_ARCH` flags，继续服从 HYLE/toolchain 的 distributable ISA baseline；MSVC 继续用 C++ 编译该 TU；
- frozen `FIX` 只有 `MLP-01..04` 与 `MIM-01..04`。不增加 Tensor/backend hints、动态 graph owner、
  allocator option API、benchmark-only field 或其他 Mem capability。

## API / usability freeze

- 状态：`FROZEN`；freeze identity：本 task 的 `AUDITED` 状态提交；reviewer：Codex；
- 普通 C++ journey：

```cpp
std::array resources{host_scratch, device_scratch};
HYLE_ASSIGN_OR_RETURN(
    auto plan, hyle::mem::LifetimePlanner::build(requests, resources));
for (std::size_t r = 0; r != plan.resource_count(); ++r) {
  HYLE_ASSIGN_OR_RETURN(auto summary, plan.resource_plan(r));
  for (std::size_t b = 0; b != summary.backing_count; ++b) {
    HYLE_ASSIGN_OR_RETURN(auto physical, plan.backing_plan(r, b));
    observe_before_allocating(r, b, physical.size);
  }
}
HYLE_ASSIGN_OR_RETURN(auto buffers, plan.instantiate(resources));
HYLE_ASSIGN_OR_RETURN(auto temporary, buffers.view(request_id));
```

- ownership/lifetime：resources/providers 长寿于 `PlannedBuffers`、views 与 in-flight use；plan 不持有 provider，
  buffers 独占 physical Buffer；owner move 不使 view 的 provider allocation token 失效，reset/destruction 会；
- failure/default/concurrency：全部构造/lookup 边界用 `Result`；没有 hidden fallback；plan 是 immutable，独立
  query 可并发，instantiate 和 buffers mutation 由 caller 同步；no-reuse/output 不自动推断；
- 场景选择：已知整图 lifetime 且 provider 有单 allocation 上限时用 planner；已有多个独立 Buffer 只需
  一个逻辑 byte range 时用 `CompositeBuffer`；动态 acquire/return 用 `BufferPool`；phase scratch 用 Arena；
- 误用诊断：非 dense slot、无效 resource/property、oversized alias root、过小 instantiate max、跨 resource
  alias 与越界 query 均有明确 Status；guide 必须说明 plan 可迁移但不会因更大 provider 自动重新合并 backing；
- public dependency / delivery：头只暴露 Error + Mem 类型；mimalloc type/header 不进入 ordinary API。
  source/static/shared 都编译同一 sketch；static 只在内部 inline seam 增加 link-only `mi_free_size` 声明；
- 冻结后只允许 pimpl arrays、slot/backing packing、rollback 和 mimalloc private call/config 的实现修正；任何
  public field/语义档、normal workload 或常态 metadata 变化都退回 `AUDITED` 并要求具名 evidence。

## Owner 六问直答（audit verdict）

1. **第三方产品路径：**唯一直接第三方是 in-tree mimalloc；v3.5.1 commit
   `34fbd7e7cd4627424490afe19b20f8066bfc537d`、tree
   `e69ccc36d80961459810aed044247c6b69c549cd`。它拥有 heap mechanism，Mem 拥有 HYLE alignment、
   sized-free、OOM、ownership、THP baseline 和 API 语义，Build 拥有 lock/vendor/artifact/delivery。ggml 只作
   capability peer，不进产品 link；标准库、compiler builtin、OS API 不记第三方模块。
2. **推理能力：**锁定 llama.cpp `c7bda030e7faee594dbe7550185e857351ad405d` 与 standalone ggml
   `d4716378882593333721eb33f153144b6885caf2`。本轮完成后 whole-graph bounded backing journey 为
   `COMPLETE`；host-pointer device registration、buffer usage/reset、async completion 继续 `ROUTED` 到
   Compute/Graph，非 Mem 缺口。当前 ledger 无未处置 `MUST`。
3. **性能：**planner 以复杂度/metadata/provider-allocation 结构门验收；host hot path 用公平、双方 sized-free
   的 64-lifecycle P1，5% margin。没有 blanket SOTA、所有尺寸/线程/平台或端到端推理声明；正式结论等待
   candidate 采样，异常只能按 card 的一次诊断/修复预算处理。
4. **旧 C ABI / Native FFI：**`N/A/CLEAN`；无 HYLE C header/schema/bridge/managed sample 或未修饰 HYLE
   export。`extern "C" mi_*` 是 static SDK 的 vendor link seam，不是承诺给 Java/C# 的 HYLE FFI。
5. **构建与导入：**component edge 仍为 `PUBLIC core error platform` + private mimalloc，无新 component、
   环或公开第三方 type。source/static/shared、selected root、offline/license/symbol 必须在新 candidate 重签，
   旧 receipt 不继承。
6. **C++ API / 文档：**上述 build→inspect→instantiate→view 路线、显式 owner 和统一 `Result` 是自然形状；
   删除无约束 build 与默认 deallocation size 修掉两个误用入口。architecture 解释算法/owner/第三方决策，
   user guide 给场景表、完整 recipe、provider compatibility、zero/alias/OOM/THP/ISA 陷阱；当前文档更新属于
   冻结 FIX，未完成前不能签 gate。

## Same-contract performance equivalence freeze

| Dimension | HYLE candidate | mimalloc PMR heap peer | verdict |
|---|---|---|---|
| work/result | 64 个相同 size/alignment allocate、端点 touch、reverse sized-free | 相同 | `EQUAL` |
| ownership/lifetime | 一个 move-only Heap + `AllocatorRef`，外置构造/销毁 | 一个 `mi_heap_t` + 非去虚 PMR owner，外置构造/销毁 | `ADAPTER_EQUALIZED` |
| failure | 成功 steady state；nullptr OOM 不计时 | 成功 steady state；bad_alloc 不计时 | `EQUAL` |
| concurrency/progress | 单 owner thread；mimalloc cross-thread 能力不在本行 | 相同 | `EQUAL` |
| setup/charging | owner/order/warmup 外置；全部 alloc/touch/free 计入 | 相同 | `EQUAL` |
| delivery/codegen | source/static HYLE；`mi_free_size`；trace off/LTO off | 同 artifact、同 v3.5.1；PMR dispatch + `mi_free_size` | `ADAPTER_EQUALIZED` |

## AUDITED verdict

全部受影响 `MUST` 已是 `COMPLETE` 或上述八个有限 `FIX`；public trace、API/owner/failure/concurrency、
complexity/resource、dependency/config、normal workload 与停止条件已冻结。允许进入实现；不得借本轮重开
`mem_vm/mem_reclaim`、其他 Mem family、公共仪器或 Compute/Graph。
