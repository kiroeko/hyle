# Graph v8：显式 allocator 下的确定性 placement 排序

## Closure card

- component / owner：`graph` / Graph；round=`graph-v8-explicit-allocator-sort`；
- baseline commit：`65454f60776deec433ba90ed65472c3e66b3c117`；前一轮 terminal record 为
  [Graph v7](graph-v7-execution-lanes-and-islands.md)；
- trigger 与证据：Compute/Graph 下层复用回审发现 Graph compile 的三处 target ordering 使用
  `std::stable_sort`。每个 comparator 已用 target ordinal 消除等价项，因此稳定性没有产品语义；但
  C++23 `stable_sort` 允许取得临时 buffer，MSVC 19.50 在超过 insertion-sort 阈值后明确调用
  `_Optimistic_temporary_buffer` / `_Get_temporary_buffer`，绕过 `CompileOptions::allocator` 及已有 OOM
  故障注入。这是 v7“compile 分配显式归调用方 allocator”的可复现所有权缺陷，不是重新泛查；
- 正常 consumer 任务：用户在同一显式 allocator 下编译含多个 execution target/provider fallback 的
  Graph；排序必须确定，allocator failure 必须可控，且不得因标准库实现或 target 数量悄悄访问全局 heap；
- 主产品纵切 / 具名来源：多 target placement 与 region-decline fallback；三个现有 comparator 的最终
  ordinal tie-break 已给出严格全序；
- 正交反偏置纵切：单 CPU target 仍走同一公开 compile API；修正不得改变 target/provider 选择、Graph
  steady submit、workspace 或 report；
- 跨模块 public journey 与每段 owner：Container 继续拥有 allocator-aware metadata，Mem/`AllocatorRef`
  拥有 compile allocation；Graph 只在已有 target ordinal range 上做冷路径排序；
- scope：消除三处无语义的 `stable_sort` 临时分配可能性，证明选择序不变，并补足大于标准库小范围阈值的
  allocator/selection 回归；
- non-goals：重做 placement policy、增加 Container 排序 API、改 Graph public API/layout、改 steady
  execution、横向扩算子或 provider；
- 直接依赖 / 直接消费者：依赖既有 Core/Error/Container/Mem/Compute；消费者为 Graph compile、Native
  与 region provider fallback；
- 性能 tier 与原因：`P0`；这是冷路径 allocation/复杂度所有权修正，候选仍为 O(T log T)，不以纳秒
  benchmark 决策；
- 适用机器 / 仪器 / material margin：Windows/MSVC 源码锚点与 Graph 定向测试；Linux/macOS 逐头和
  source/static/shared 在 candidate 后按受影响格验证；P1/P2 不适用；
- 调查与纠正预算：一个 comparator 等价性审计轮、一个实现轮、一个红队轮；
- 预声明验证：三个排序点的 total-order 证明；target 数量超过 32 的 deterministic placement/fallback；
  failing `AllocatorRef` 不被隐藏成功路径绕过；既有 Graph/region/native SDPA 纵切；三平台、sanitizer、
  source/static/shared、closure/docs checker；
- stop / blocker 条件：如果任一 comparator 存在真实 equivalent target 且依赖原顺序语义，则停止并退回
  `SCOPED` 明确定义 tie-break；不得用另一个隐藏 scratch allocation 替代。

`SCOPED` 阶段只允许核对三处 comparator 与测试设计；完成有限改动表并进入 `AUDITED` 前不改 Graph
产品代码。

## AUDITED 结论（2026-09-14）

三处排序分别服务初次 operation placement、region 选择后的 operation fallback，以及 region decline 后的
fallback。逐项 comparator 都先排 required/original/preferred/locality，最后以 `left < right` 打破全部平局；
因此任意不同 target ordinal 都不等价，stable insertion order 不可观察。现有 Container 决策也明确普通
heap algorithm 直接使用标准库；本轮不新增泛化算法或依赖。

| 能力族 | 层级 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|
| target 候选确定全序 | 原子机制 | policy key + ordinal tie-break | COMPLETE | KEEP |
| 显式 compile allocator | 原子机制 | SmallVector/plan metadata 使用 `AllocatorRef`，但 stable-sort scratch 可越界 | PARTIAL | FIX |
| provider/region fallback | 复合功能 | 冷路径按同一全序尝试，steady path 固定 | COMPLETE | KEEP |

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 | 结果 |
|---|---|---|---|---|
| G82 | 三处 total-order comparator 仍调用可取临时 buffer 的 `stable_sort` | 改为 in-place `sort`；不改 comparator | 大于 32 targets 仍选同一 required/preferred/original target；源码无 `stable_sort` | 待实现 |
| G83 | v7 allocator sweep 未跨过实现的临时-buffer 阈值 | 增加多 target 回归并复用 allocator sweep；不接管全局 `new` | selection/失败/重试确定，Graph allocator 账单闭合 | 待实现 |

### API / usability freeze

公开 API、默认 placement、report、对象布局和调用草图全部不变。compile 仍允许显式 allocator 的有界冷
分配；Invocation/submit 仍零 metadata allocation。失败继续用 `ResourceExhausted` 回到调用方，不增加
“fast sort”开关、兼容层或 benchmark seam。

## 七问

1. 第三方：只使用 C++23 `sort`；无新增依赖、版本或许可证面。
2. 功能：多 target/region fallback 功能不变；修复的是现有显式 allocator journey。
3. 复用：继续使用 Container SmallVector、Mem allocator、Compute provider；标准 heap algorithm 按既有
   Container 边界直接使用，不复制新模块。
4. 性能：compile 排序保持 O(T log T)、O(log T) stack，去掉可选 O(T) hidden scratch；steady path 不变，
   所以只需 P0。
5. Native FFI：Graph 仅面向 C++，本轮无历史 C ABI。
6. 交付：没有 header/target/dependency 变化；仍需三种 C++ 交付回放。
7. 易用性：用户调用不变，但 allocator ownership 从“通常成立”变为结构上成立；guide 无新增选项。
