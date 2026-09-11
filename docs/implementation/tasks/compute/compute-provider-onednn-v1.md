# Compute oneDNN provider v1：显式 CPU provider 与 prepared MatMul

## Closure card

- component / owner：`compute_provider_onednn` / Compute；round=`compute-provider-onednn-v1`；
- baseline commit：`5828c649675e88c7fb7230c4df4ba928984ccfab`；
- trigger 与证据：[`compute-v2`](compute-v2-cpu-operator-catalog.md) 的源码 P0 预检确认普通 row-major
  MatMul 只有单线程朴素 accumulation，缺少成熟实现的 JIT/blocking/packing/threading；该卡的 stop 条件明确要求
  形成 provider/packing owner，而不是增加 benchmark-only fast API；
- 正常 consumer 任务：推理或工程计算用户为固定 f32 `A[M,K] * B[K,N]` 冷准备一次 provider artifact，
  为每个并发调用取得独占 run state/workspace，随后以 caller-owned TensorView 重复提交；
- 主产品纵切 / 具名来源：ggml/llama.cpp、SGLang、vLLM 与 ds4 的 decode `M=1/4`、短 prefill
  `M=32` MatMul；oneDNN `6b7f1fbbb18e`（3.14.0）primitive/descriptor/user-scratchpad；
- 正交反偏置纵切：普通 row-major 工程 GEMM，不使用模型名、量化格式或转置布尔捷径；
- 跨模块 public journey 与每段 owner：Mem 拥有 tensor/workspace storage；Compute 拥有 provider descriptor、
  Context provider state、PreparedOperation 与每次运行状态；Graph 只组合同一 artifact/run-state 合同并把
  liveness 交给 Mem；oneDNN 类型不得泄漏到普通 Compute/Graph API；
- scope：审计并实现显式、无全局注册副作用的 provider SPI；锁定 oneDNN 源码、许可证和最小 CPU 构建；
  首轮只承载当前 dense f32 MatMul；prepare/Graph compile 冷路径完成 capability 与 route 选择；使用
  caller/Invocation-owned scratchpad，steady submit 不分配、不 packing、不搜索；报告真实 provider/version、
  blocking/threading/layout/workspace 与 fallback；source/static/shared 均保持同一 C++ 语义；
- non-goals：oneDNN Graph partition、完整 oneDNN primitive 暴露、量化/f16/bf16、异步 queue、GPU engine、
  process-global provider registry、Native FFI，以及用 oneDNN 类型替代 HYLE Tensor/Operation；
- 直接依赖 / 直接消费者：公开/实现依赖待 AUDITED 裁决，预期只依赖 `compute` 及其公开闭包；外部 Apache-2.0
  oneDNN 为本因子私有 provider；直接消费者为 direct Compute MatMul 与 Graph v2；
- 性能 tier 与原因：`P1`；只有同一 API/shape/layout/precision/threading/storage/workspace/completion 的最终
  shipping route 才比较，wrapper tax 与 kernel/provider 收益分层记录；
- 适用机器 / 仪器 / material margin：Windows W1-WIN 做一次 immutable-candidate qualification，固定 5%；
  Windows、WSL/GCC 14、两台 Apple 做 correctness/header/package，非 W1 不签工程性能；
- 调查与纠正预算：oneDNN C/C++ primitive、thread runtime、scratchpad、cache、CMake/package 各扫描一遍；
  provider SPI/API 最多两轮，性能根因最多两轮，每个正式 workload 最多一次 `n=15`；
- 预声明验证：shape/stride/zero/overflow/alias、unsupported/provider failure/OOM、artifact/run-state lifetime、
  并发独立 invocation、context/queue close、差分、逐头、三平台、sanitizer、source/static/shared、license/offline
  cache、P0 静态成本和合格 P1；
- stop / blocker 条件：若 oneDNN 要求 submit 时隐藏分配、不可隔离的全局线程/缓存、无法表达 caller
  scratchpad、无法通过 relocatable source/static/shared，或 provider SPI 必须形成 Compute 反向依赖/静态初始化，
  停止并记录精确 blocker；不得降低 HYLE owner/lifetime 合同或复活旧 ggml bridge 来绕过。

## SCOPED 待审计项

1. provider 是传给 Context、prepare 还是 Graph compile；必须让 direct 与 Graph 共用同一显式对象，且普通
   用户不理解内部 factor target；
2. immutable artifact 与 per-run mutable oneDNN memory/stream/scratchpad 如何分离，不能把共享
   PreparedOperation 做成数据竞争；
3. 首轮采用 `SEQ` 还是 threadpool runtime；线程语义必须先于性能 benchmark 冻结，不把单线程和多线程
   当同合同；
4. oneDNN primitive cache、JIT、weight packing 与 scratchpad 分别由谁拥有，哪些计入 prepare、哪些必须
   显式暴露；
5. 可选依赖如何进入 `hyle::compute` product façade，关闭时不得下载、不得导出恒失败 stub，开启时
   source/static/shared 与离线 bundle 必须一致；
6. HYLE-native 和 oneDNN 无法形成稳定 material 区分的 shape 默认借 oneDNN；明显胜出的原生 route 才保留。

AUDITED 前不得添加 oneDNN 产品代码、下载门或公开 provider API。

## AUDITED 能力、所有权与交付裁决（2026-09-12）

### 依赖与成熟实现事实

- shipping pin 采用 oneDNN `v3.13.2` / `71094df451909e95ceeefc8f64c275cdcbbcdff5`，而不是本地
  `master` 的 3.14 development commit；许可证为 Apache-2.0；源码进入统一 dependency lock/source
  cache，不能从 `C:/public/project-hyle/oneDNN` 直接构建；
- 最小构建为 static CPU inference、MatMul-only、Graph/GPU/tests/examples/docs OFF、primitive cache OFF、
  user scratchpad ON；CPU runtime 使用 THREADPOOL，使串行和显式外部 executor 共用同一 artifact，拒绝
  OMP/TBB 的 process-global pool policy；
- oneDNN C API 允许 `DNNL_MEMORY_NONE` memory object 后设置 caller data handle，并可从 primitive
  descriptor 查询 scratchpad descriptor。primitive 可跨线程共享，但 stream、memory handle 和 scratchpad
  必须属于一次独立 invocation；library scratchpad 与 global primitive cache 都不满足 HYLE owner 合同；
- oneDNN 没有接受 HYLE allocator 的 engine/JIT 内部 allocation seam；这些只允许发生在 context/prepare/
  invocation 冷路径，并由 provider 拥有。任何 steady execute allocation 都由 allocation probe 判 blocker，
  不能在文档里冒充 caller-owned。

### 能力与有限施工 ledger

| ID | 类型 | 裁决 | 冻结合同 |
|---|---|---|---|
| OD01 | explicit provider value | `FIX` | `Provider` 是可复制、小型、类型擦除值；无 registry/static initializer；`ContextOptions` 可给有序 provider span，并提供单 provider convenience |
| OD02 | context provider state | `FIX` | Context 冷创建 provider engine/state；provider 配置及其借用 executor 必须活到 Context 及全部 retained artifact 结束 |
| OD03 | immutable prepared artifact | `FIX` | `PreparedOperation` 改为可复制共享的 allocator-owned artifact，固定 provider/layout/workspace；只读对象可跨线程共享 |
| OD04 | exclusive run state | `FIX` | 新增 move-only `OperationInvocation`；每个并发运行独占 stream/memory handles/scratchpad view；submit 只改本 invocation |
| OD05 | caller-owned scratchpad | `FIX` | prepared 报告 bytes/alignment；caller 或 Graph Invocation 提供 Mem `BufferView`；不足、错域、错对齐在 create-invocation 失败 |
| OD06 | explicit threading | `FIX` | `ParallelExecutorRef` 是借用、同步、infallible 的 Compute 原子；空值为 caller-thread serial；oneDNN adapter 不创建线程、不改变全局线程数 |
| OD07 | row-major f32 MatMul route | `BORROW` | oneDNN 首轮只接受 dense row-major `A/B/C` rank-2 f32；转置连续-K 权重继续走已有 HYLE SIMD dot route |
| OD08 | cold ordered fallback | `FIX` | Context 声明顺序逐 provider prepare；只有 `Unimplemented` 继续，OOM/runtime/corruption 立即返回；native 最终 fallback；submit 不再裁决 |
| OD09 | Graph run-state reuse | `FIX` | Graph plan 保存共享 prepared；每次 Graph Invocation 为每个 node 创建 OperationInvocation，provider scratchpad 进入同一 Mem lifetime plan |
| OD10 | truthful report | `FIX` | direct observer/Graph report 区分 native、oneDNN、mixed 与 fallback；版本、线程数、workspace 可发现；不声称 weight 已持久 packing |
| OD11 | constant-weight prepack | `DEFER` | 当前 `prepare(desc)` 没有 weight payload lifetime；不得隐藏复制。由未来 constant binding/weight artifact 卡承载 |
| OD12 | quantized/f16/bf16/epilogue | `DEFER` | 另开 op/provider 卡；本卡不把 oneDNN 大 API 面伪装成 HYLE 已支持能力 |
| OD13 | automatic link-time activation | `REJECT` | 链接 provider 不改变 `Context::create(cpu)`；自然入口是显式 provider。没有进程全局注册、弱符号或初始化顺序 |

这是本轮全部产品改动。新 operation、Graph partition、persistent weight、第二种 provider 或 Sched adapter
都必须另立有限卡。

### API 与生命周期冻结

自然 direct journey 为：

```cpp
auto provider = hyle::compute::onednn::provider({.executor = executor});
auto context = hyle::compute::Context::create(
    hyle::compute::Device::cpu(), provider).value();
auto prepared = hyle::compute::prepare(context, hyle::compute::MatMul{},
                                       a_desc, b_desc, c_desc).value();
auto scratch = allocator.allocate(prepared.workspace_requirement()).value();
auto invocation = prepared.create_invocation(scratch.view()).value();
auto done = invocation.submit(queue, a, b, c).value();
```

空 executor 表示同步串行，且无需用户理解 oneDNN threadpool 类型。自定义 executor 是借用句柄，合同为：
`parallel_for(n, fn)` 返回时全部 `fn(i,n)` 已完成，不能失败/抛异常，嵌套调用必须保证进展；若应用已有
可失败或可关闭 scheduler，必须由拥有它的上层 adapter 保证在 Context lifetime 内可用，不能把失败吞进
oneDNN `void` callback。

`PreparedOperation` 保留 Context；`OperationInvocation` 保留 prepared 并借用 scratch storage 到析构。
tensor views 只借用到同步 Completion 终态。provider 内部 C handle 销毁顺序固定为 invocation memory/stream
→ prepared primitive/descriptor → context engine。provider shared library 必须比这些 C++ 对象存活更久，和
任何 installed C++ shared dependency 相同。

### 构建与产品形状

- factor 真实边为 `compute_provider_onednn -> compute`（PUBLIC）；Compute/Graph semantic factor 不反向依赖
  provider。用户产品 `hyle::compute` 聚合 provider factor，`hyle::graph` 也聚合它以提供相同显式选择；
- `HYLE_COMPUTE_ONEDNN=AUTO|ON|OFF` 只在相关产品 closure 中生效。OFF 不探测/下载且不定义 factory；
  AUTO 在 oneDNN 支持的平台启用，ON 遇到不支持环境失败。factor identity/target 仍存在，方便依赖图稳定；
- provider public header 不暴露 `dnnl_*`。shared provider 私链静态 oneDNN 并隐藏 vendor symbols；static SDK
  导出所需静态 target/header metadata；source、static、shared 使用同一 HYLE C++ provider API；
- oneDNN archive/hash、展开树、许可证、online/offline cache 由中央 lock/package registry 唯一登记。

### P0/P1 preflight 与停止线

- cold：Context O(provider count)，prepare 一次 oneDNN primitive selection/JIT，create-invocation 创建固定
  stream/memory wrappers并映射 scratch；这些 allocation/JIT 不计入 submit，但必须单独报告；
- steady：binding validation/mapping O(bindings)，一次 `dnnl_primitive_execute`，零 HYLE allocation、零 route
  search、零 descriptor creation、零 packing API。allocation probe 若发现 oneDNN execute 隐藏 allocation，
  本卡 BLOCKED，不用 timing 掩盖；
- correctness 对 native/reference 做 M={0,1,4,32}、K/N 含 0/1/非整块、大值、NaN/Inf 容忍策略、wrong
  workspace/layout/alias、context close、并发 invocation、provider failure/OOM 差分；
- P1 分 direct wrapper tax 与 kernel 两层。只比较相同 f32、layout、线程数、预热/prepare、scratch、完成
  边界；row-major oneDNN 对成熟 oneDNN direct C route，transposed native 不拿来冒充 oneDNN 胜负。Windows
  W1 immutable candidate 固定 5% material 门；换序/A-A/地址效应不合格即记 INCONCLUSIVE，不调 API 迎合。

### 七问直答

1. 第三方：pin、许可证、cache、最小 build、升级 owner 和 native fallback 均已冻结；
2. 功能：本卡 dense row-major f32 MatMul journey 完整；constant/quantized/epilogue 明确不冒充完成；
3. 复用：Tensor/Mem/Status/Context/Graph lifetime plan 全部复用已有 owner，不复制 allocator/scheduler；
4. 性能：route 在 prepare 冻结，steady 无 HYLE 搜索/分配；最终仍须 allocation probe 与公平 P1；
5. 交付：只有 C++ provider；无 C/Native FFI、无 oneDNN 类型泄漏；
6. 构建：一个可选 factor、一份 lock、一份 package registry 覆盖 source/static/shared；
7. 易用：单 provider Context convenience、显式 threading/workspace 与 provider observer；坑写入 Compute guide。

## Audit verdict

`OD01--OD10` 是有限且可实施的 provider seam；`OD11--OD13` 划清负空间。允许
`compute_provider_onednn` 进入 `AUDITED`，并依法修改 Compute/Graph 的 artifact/run-state 合同。任何
submit 时隐藏分配、全局线程/registry、provider 反向依赖或 vendor 类型泄漏立即触发本卡停止线。
