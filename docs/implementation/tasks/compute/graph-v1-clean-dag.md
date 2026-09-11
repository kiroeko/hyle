# Graph v1：清洁计算 DAG 与 one-op Executable 纵切

## Closure card

- component / owner：`graph` / Graph；round=`graph-v1-clean-dag`；
- baseline commit：`601b25ad4fe54baa1f8dd19153bb1230085b44d9`；
- trigger 与证据：长期产品已裁决为逻辑计算 DAG、编译与可复用执行物；现有 `TaskGraph`、
  parallel/data-parallel API 属不同问题域，继续留在公开头、target、tests 和 benchmark 会反向污染新 IR；
- 正常 consumer 任务：推理引擎以 Compute RMSNorm operation 建一个不可变 Graph，编译为可共享
  Executable，为每个并发请求创建独占 Invocation、绑定 caller storage 并取得 Compute Completion；
- 主产品纵切 / 具名来源：ggml graph plan、oneDNN compiled partition、ONNX Runtime provider partition、
  TensorRT engine/context 与 MLX compiled function 的生命周期；
- 正交反偏置纵切：同一 one-op 合同允许工程计算/custom operation，不包含模型、token、KV 或 batching；
- 跨模块 public journey：Compute 拥有 operation/provider/backend execution；Mem 拥有 storage/workspace；
  Graph 拥有 logical DAG、compile report、Executable/Invocation；Error 表达 node/region context；
- scope：删除旧 Graph 产品面；实现 GraphBuilder、不可变 Graph、最小 value/op schema、结构/shape/layout
  验证、CPU provider compile、CompileReport、不可变可共享 Executable、独占可复用 Invocation 和 one-op
  submit；内部 plan 不成为第三个公开产品；
- non-goals：兼容 TaskGraph/parallel_for/reduce/scan/sort/data-parallel；多节点 fusion/partition/liveness；
  arbitrary control flow；模型循环；GPU capture；完整 serialization；Native FFI；
- 直接依赖：公开边为新 Compute operation/execution 合同、`core`、`error`；实现边仅为 `util_core` 的
  intrusive plan ownership。Mem storage 由 Compute 公开合同承载，Graph 不重复登记或解释；首轮不依赖
  Sched/Container；
- 直接消费者：独立 C++ one-op graph consumer；后续 multi-op CPU round；
- 性能 tier 与原因：`P1`；steady Invocation submit 必须对同一 PreparedOperation 做 one-op 税基线，
  compile/cold 与 replay/warm 分开；
- 适用机器 / 仪器 / material margin：W1-WIN CPU + 统一 instrument；同 buffer、layout、queue、warmup
  与 completion 边界；5% 是 action 门而非统计同义词；
- 调查与纠正预算：资产/能力各一遍，API 纠正最多两轮，性能纠正最多两轮，每 shape 最多 15 个 paired
  samples；不以未来 multi-op 摊薄解释本轮税；
- 预声明验证：旧入口/target 负扫描；DAG cycle、invalid ID、shape/layout、binding、OOM、partial compile、
  reset/reuse、并发 Invocation、迟到 Completion 与销毁顺序；Windows/Linux/macOS CPU；ASan/UBSan，
  适用时 TSan；source/static/shared consumer；P0/P1、red-team、closure/docs/diff；
- stop / blocker 条件：Compute one-op 合同未稳定、Graph 必须读取第三方 owner 内部字段、Invocation 无法
  隔离并发状态、或 steady submit 出现无法解释的 material 税时停止。

## 旧资产处置

| 旧资产族 | 新边界 | 默认处置 |
|---|---|---|
| incremental dirty/cached `TaskGraph` | 未来 `graph_incremental` 候选 | 从当前源码和公开面删除，只留 Git 历史 |
| `parallel_for/reduce/scan/sort` | 未来独立 `parallel` 候选或 Sched consumer | 删除，不迁入计算 Graph |
| data-parallel partition helper | compiler 私有启发式只有被新纵切重新证明才可实现 | 删除旧版本 |
| tests/bench/CMake/sample | 只证明旧 callable-DAG 形状 | 随旧产品面删除 |

不建立 legacy target、兼容 alias、deprecated header 或源码参考副本。

## 能力与形状审计

| 能力族 | 层级 | 准入旅程/来源 | 语义与生命周期 | 原子分解/应用边界 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|---|---|
| typed value/node identity | 原子机制 | one-op RMSNorm graph | generation-aware typed ID；错误/跨 builder ID 可诊断 | 不暴露下标、地址或第三方 node | 全新实现 | `MISSING` | `FIX` |
| GraphBuilder / immutable Graph | 复合功能 | 正常构图与 publish | builder 单 owner；`finish()` 原子地产生不可变、自含 logical graph | 不持有 tensor payload 或 runtime | 全新实现 | `MISSING` | `FIX` |
| validate/canonicalize | 复合功能 | invalid edge/shape/layout/alias | O(V+E)，错误带 value/node context；失败不留下半图 | shape inference 只做 RMSNorm identity shape | 全新实现 | `MISSING` | `FIX` |
| CPU compile / CompileReport | 复合功能 | Graph 到可解释 execution artifact | 冷路径 prepare Compute op；报告 provider、workspace、copy/sync/fallback | 首轮无 partition/fusion/liveness | 全新实现 | `MISSING` | `FIX` |
| Executable / Invocation | 复合功能 | 并发请求重复执行 | Executable 不可变共享；Invocation 独占可变 binding，完成后方可 reset/reuse | 调度/cancel/admission 属后续组合 | 全新实现 | `MISSING` | `FIX` |
| one-op submit | 复合功能 | 与 direct PreparedOperation 同合同 | submit 只填 slots 并调用预绑定 artifact；borrow 活到 Completion | 不遍历原 Graph、不重新 prepare | 全新实现 | `MISSING` | `FIX` |
| multi-op compile/plan | 复合功能 | 后续 transformer slice | partition/liveness/workspace/fusion 需真实多节点卡 | 不能用 one-op 冒充 | 后续轮 | `OUT_OF_SCOPE` | `DEFER` |
| incremental task DAG / parallel algorithms | 应用策略/另一产品 | 旧 CAD/task/sched 形状，无当前 consumer | 不属于 tensor compute DAG | 未来有消费者再独立审计 | Git 历史 | `OUT_OF_SCOPE` | `REJECT` |

旧 `TaskGraph::add_node/mark_dirty/evaluate`、`evaluate_parallel*`、`parallel_for/reduce` 和
`BlockedRange` 全部是未映射公开入口：它们不能回指 Compute operation、Tensor、Executable 或
Invocation，反而把 Sched 和 Container 强制泄漏给 Graph。因此没有原样 `KEEP/MOVE` 项。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 | 结果 |
|---|---|---|---|---|
| G01 | 3 个旧公开头、3 个实现 TU 与 tests 证明的是 callable task graph | 全量删除，不保留 legacy/reference/compat | 全树负扫描无 TaskGraph/parallel/data-parallel 入口 | 完成：旧字节随 `f5581e82` 删除，当前 Graph 树负扫描为零 |
| G02 | 中央组件图把旧 Graph 公开依赖 sched/container，且 product 与状态冲突 | 重建为直接依赖 Compute/Error/Mem/Core 的新 factor/product | 精确选择无 sched/container 闭包 | 完成实现：实际公开边 `core/error/compute`、私有 `util_core`；Windows exact-factor 闭包不含 sched/container |
| G03 | 没有逻辑计算 IR | 实现 typed IDs、GraphBuilder、immutable Graph、one-op structure/shape validation | invalid ID/cycle/shape/output/finish 原子性测试 | 完成实现：generation-aware IDs、原子 mutator/finish、shape/output/cross-builder 测试通过；cycle 在 one-op typed API 中不可表示 |
| G04 | 没有 compile artifact 或诊断 | 实现 CPU compile、CompileReport、immutable Executable、独占 Invocation | provider/workspace/copy/sync 可枚举，partial failure 可清理 | 完成实现：report 明示 CPU/HYLE-native/零 workspace-copy-sync，copyable artifact、OOM/lifetime 测试通过 |
| G05 | 没有与 Compute 共用的运行合同 | one-op plan 直接持有 `PreparedOperation` 并提交 bindings | steady 不读 Graph、不 search/allocate/prepare | 完成实现：plan 直接拥有 PreparedOperation，Invocation 仅固定 slot bind 后转交 submit |
| G06 | 旧测试无法迁移为证据 | 重新建立 correctness/OOM/lifetime/concurrency/reset 测试 | 预声明 CPU matrix 全部有 verdict | 完成：Windows/MSVC、macOS ARM64/AppleClang 21 和 WSL/GCC 14 的 unit/header 全过；WSL ASan、UBSan、TSan 各 2/2 Graph 门通过，two-thread 独立 Invocation 无 TSan 报告 |
| G07 | 旧文档/sample/package 没有新用户 journey | 新 Graph guide 与 source/static/shared consumer | 三交付同 API，误用有诊断 | 完成：独立用户页与 package consumer 覆盖 builder/compile/report/bind/submit；Windows source、可搬迁 shared、可搬迁 static 均编译运行通过，shared export hygiene 通过（6 库、691 exports） |
| G08 | Graph 税未量化 | candidate 后做 P0、one-op A/A preflight 与一次 P1 | 无无法解释的 material fixed loss | 完成：同一 PreparedOperation/Queue/TensorView/Completion 的四 strata paired ratio 0.9879--0.9971，全部 CI 在 5% 等价带，combined `PARITY/PASS`；controls 全 PASS |

### API / usability freeze

```cpp
hyle::graph::GraphBuilder builder;
auto input = HYLE_TRY(builder.input(desc));
auto output = HYLE_TRY(builder.add(hyle::compute::RmsNorm{1e-5F}, input));
HYLE_TRY(builder.output(output));
auto graph = HYLE_TRY(std::move(builder).finish());
auto compiled = HYLE_TRY(hyle::graph::compile(graph, context));
auto invocation = HYLE_TRY(compiled.executable.create_invocation());
HYLE_TRY(invocation.bind(input, input_view));
HYLE_TRY(invocation.bind(output, output_view));
auto completion = HYLE_TRY(invocation.submit(queue));
```

- ID 带 builder identity/generation 并只能由其 origin 使用；错误 ID 返回 `Status`，不以 fatal assert 代替。
- Graph 拥有 desc/op/edge 元数据但不拥有 payload；Executable 拥有 prepared artifact；Invocation 借用
  caller TensorView，并在 Completion 终态前禁止 reset/rebind/resubmit。
- one-op v1 只允许一个 RMSNorm node、一个或两个输入（可选 weight）和一个输出；有限不是隐式功能，
  compile 对其它结构明确返回 `Unimplemented`，GraphBuilder 仍保持可扩展 typed DAG 形状。
- compile 可做 allocation 和验证；固定 specialization 的 create-invocation/submit 正常路线使用 caller
  注入 allocator 或预留 inline slots，steady submit 不分配、不遍历、不做 provider route。
- Graph/Executable 可跨线程只读共享；Builder/Invocation 默认单 owner、非线程安全；并发运行使用不同
  Invocation。CPU Queue 本轮同步终态，但 API 仍遵守异步 borrow 规则。
- source/static/shared 统一通过 `hyle::graph` product；Native FFI 与序列化为 N/A。

## 七问

1. 第三方路径：本轮无第三方 Graph runtime；只消费本轮 Compute 合同。provider 依赖在 Compute owner，
   Graph 不直接链接 ggml/MLX/CUDA 等。
2. 功能完整：完整覆盖 one-op 推理纵切与生命周期，不声称 multi-op compiler 完成；旧通用 task graph
   与 parallel algorithm 已按产品边界拒绝。
3. 内部复用：Error、Mem 与 Compute 分别拥有失败、元数据 allocation/storage、operation execution；
   首轮不复制 Sched、Container 或私有 allocator。
4. 性能：validate 为 O(V+E)，one-op compile 为 O(1) node；steady submit 为 O(bindings)，无 heap/search/JIT。
   candidate 后与直接 PreparedOperation 做同合同 P1。
5. FFI：旧 Graph 没有合法 Native FFI；旧 C++ API 无兼容承诺并删除。
6. 构建：G02/G07 证明 source/static/shared 和精确 product closure；header-only 假 journey 不适用。
7. 易用：公开四对象生命周期与用户任务一致；guide 明示 one-op 限制、binding owner、并发 Invocation、
   compile report 和错误恢复。

## 当前状态

候选 `5c9dc1604e6daaec1d73062296f2a71d019e7b35` 的 G01--G08 已全部完成。Windows、macOS、WSL、
sanitizer、source/static/shared、红队与 P1 事实见本卡及
[`benchmarks-compute-graph`](../../evidence/performance/benchmarks-compute-graph.md)。本卡只终结清洁 one-op
Graph 生命周期；multi-op compiler/plan 必须由新的有限卡合法重开，不能让 one-op 状态冒充完整 Graph。
