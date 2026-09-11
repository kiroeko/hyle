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
- 直接依赖：新 Compute operation/execution 合同、`core`、`error`、`mem`；只有公开类型或实现确需时才
  登记其它边，首轮不公开依赖 Sched/Container；
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

## 当前状态

当前为 `SCOPED`。Graph 可与 Compute 共同审计，但产品修改必须等本卡自身进入 `AUDITED`；Compute
未形成可消费 one-op 合同时，Graph 不提前制造伪 adapter。
