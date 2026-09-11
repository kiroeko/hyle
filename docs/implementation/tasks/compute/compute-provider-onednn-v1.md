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
