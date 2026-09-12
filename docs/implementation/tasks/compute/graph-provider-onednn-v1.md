# Graph oneDNN provider v1：真实 CPU gated-MLP region 审计

## Closure card

- component / owner：`graph_provider_onednn` / Graph；round=`graph-provider-onednn-v1`；
- baseline commit：`e6963d7416a86ebada0299c65a6b10b6aeafd6ac`；
- trigger 与证据：Graph v3 的 provider-neutral region、workspace、逐 step report 与 ordered
  completion 已实现，但只有测试 provider，不能证明合同适合成熟编译器。锁定 oneDNN 3.13.2 的
  `examples/graph/gated_mlp.cpp` 会把 gate/up MatMul、分解 SwiGLU 与 down MatMul 交给一个
  compiled partition；当前 HYLE `compute_provider_onednn` 只承载轻量 x64 BRGEMM 单算子，且明确以
  `ONEDNN_BUILD_GRAPH=OFF` 构建，不能冒充这条 journey；
- 正常 consumer 任务：推理引擎把固定拓扑和 immutable f32 权重冷编译一次，每个请求只绑定动态
  activation、输出和 caller-visible workspace，随后可建立多个独立 Invocation 重复执行；steady
  submit 不重新分区、建图、JIT、pack、搜索 route、创建线程或隐藏分配/复制；
- 主产品纵切 / 具名来源：Transformer MLP 的 `RMSNorm -> gate MatMul + up MatMul -> separate
  SiLU gate -> down MatMul`。RMSNorm 可作为相邻 HYLE-native step；本 factor 候选 region 只覆盖后四个
  HYLE node，并降低为 oneDNN 官方 gated-MLP 的 MatMul/Sigmoid/Multiply/Multiply/MatMul 图。来源为
  ggml/llama.cpp、ds4、vLLM、SGLang 的 gated MLP 重复执行形状和 oneDNN 官方 Graph 示例；
- 正交反偏置纵切：无模型、GGUF 或量化命名的 caller-owned dense f32 branched MLP，使用小批量与
  非典型矩形维度，证明入口表达的是普通 Graph region，而不是某个模型或 benchmark shape；
- 跨模块 public journey 与每段 owner：Compute 拥有 Device/Context/Queue/Completion、Tensor storage
  bridge 与 provider identity；Graph 拥有 DAG、partition/selection、constant binding、liveness/workspace、
  Invocation 与 report；本 factor 只拥有 oneDNN engine、compiled partition 和每 Invocation stream/tensor
  handles。权重/activation storage 仍由调用方或 Graph owner 持有，模型加载、KV、batching、sampling
  与 serving 留在消费方；
- scope：审计锁定 oneDNN Graph C API 的 CPU partition/compile/execute、HYLE operation/layout 映射、
  engine/stream/tensor/scratchpad 生命周期、并发 Invocation、失败/OOM、threading、copy/sync、版本/许可、
  可选构建成本和三交付；只有审计证明 exact region 可执行且符合热路径合同后才冻结有限实现集；
- non-goals：系统扩充 CPU 算子目录、量化/稀疏 MLP、Attention、GPU oneDNN/SYCL、训练、动态 shape、
  跨设备图、公开 oneDNN 类型、让 oneDNN 拥有 HYLE Graph、模型 runtime、Native FFI，以及为了得到一个
  partition 私有 fork oneDNN 或伪报 dispatch/copy/savings；
- 直接依赖 / 直接消费者：SCOPED 阶段不声明依赖。候选为 public `compute`、private `graph` 与 private
  locked oneDNN；禁止 Compute/Graph 反向依赖本叶子。直接消费者是 Graph region selection 和上述公开
  Graph 纵切；是否需要公开 `hyle/graph/providers/onednn.h` 由审计裁决；
- 性能 tier 与原因：`P1` 资格化 Graph/provider 边界及 steady allocation，`P2` 比较完整 gated-MLP
  admission-to-terminal-completion；单 MatMul 点值不得决定 region API 或产品选择；
- 适用机器 / 仪器 / material margin：Windows W1-WIN x64 与 WSL Linux x64 做正能力、correctness 和
  必要 P1/P2，固定 5%；两台 Apple 验证 factor 未选择/不支持时的负能力、Graph fallback 和三交付隔离，
  不伪造 oneDNN x64 性能；
- 调查与纠正预算：锁定/当前 oneDNN Graph API、pattern/kernel/execute、allocator/threading 与 CMake/
  package 各一次定向源码审计；一个最小 upstream-only CPU 可执行探针；一个 HYLE region 原型；correctness
  最多两轮、ownership/concurrency 最多两轮、构建交付最多两轮，P1/P2 各一次预检和一次根因纠正；
- 预声明验证：CPU 能形成并执行有意义的多节点 region；f32 数值与 scalar/native reference；常量借用和
  Executable/Invocation/Completion 析构顺序；两个并发 Invocation；user scratchpad；compile/create/submit
  失败 rollback；unsupported shape/fallback；无 per-submit HYLE/oneDNN metadata allocation；report 的
  dispatch/copy/sync/thread facts；逐头、Windows/WSL/Apple 适用格、sanitizers、source/static/shared、
  exact-factor selection、离线依赖/许可证、P0 静态账单、必要 P1/P2 与终局红队；
- stop / blocker 条件：若 CPU 无法形成并正确执行 region、公开 API 只能经 abort/异常越过 `noexcept`、
  warm execute 有不可消除的隐藏动态分配/线程生命周期、并发必须共享不安全 scratchpad，或 Graph build
  无法作为显式可选成本与轻量 BRGEMM factor 共存，则精确 BLOCKED；不得退化成 synthetic provider、
  多次 HYLE 单-op submit 或私有 oneDNN ABI 后声称完成。

## SCOPED 待审计项

1. 官方 gated-MLP 示例在 CPU 是否稳定得到一个可执行 partition；GPU-only gated-MLP primitive 与 CPU
   `larger_partition_kernel_t` fallback 的真实 dispatch/execute 含义是什么；
2. HYLE `GatedActivation{kSilu,kSeparate}` 降为 Sigmoid/Multiply/Multiply 后，node range、boundary、
   dtype/shape/stride 和 constant role 是否保持同一语义；
3. C Graph API 的 engine、graph、partition、compiled partition、stream、logical tensor、tensor handle
   和 user scratchpad 的唯一 owner、fallibility 与销毁顺序；
4. compile 冷路径允许的 JIT/packing/allocation 与 steady execute 的 allocation/lock/thread/copy/sync 静态
   账单，oneDNN 内部 cache 或 scratchpad 是否破坏合同；
5. `ONEDNN_BUILD_GRAPH=ON` 与 primitive subset、concurrent execution、SEQ/thread runtime 的兼容性，
   以及轻量 single-op factor 与重型 Graph factor能否只在选中时付费；
6. provider factory 应属于 Compute provider 命名空间还是 Graph provider 命名空间、是否需要公开头，
   以及 source/static/shared 安装时 oneDNN 的静态/动态依赖和许可证投影；
7. 用官方 C/C++ 示例、HYLE region 路线和等价 direct route 三方证明同合同；若 oneDNN 只产生多个内部
   primitives，report 必须如实记录 dispatch，不得把一个 API 调用写成一次硬件 dispatch。

本卡只授权审计。`AUDITED` 前不得添加 provider 产品实现、依赖、公开头、feature、benchmark 或 Graph
产品改动；空 factor target 只使 component identity 和 exact-factor selection 可执行。

## Source audit 与可执行探针（2026-09-13）

- 锁定 oneDNN 3.13.2 源码与 `C:/public/project-hyle/oneDNN` 当前 main
  `6b7f1fbbb18e868b687994ac3c1739530f7a68a6` 保持相同执行形状。公开
  `dnnl_graph_compiled_partition_execute_v2()` 在
  `src/graph/interface/partition.cpp` 每次创建 `ins`、`outs` 两个 `std::vector<tensor_t>` 并分别
  `reserve()`；公开 API 没有 caller-owned/precreated metadata seam；
- f32 CPU gated-MLP pattern 会选择 `gated_mlp_base_t`，CPU 不走 GPU-only gated-MLP primitive，而回落
  `larger_partition_kernel_t`。其 `execute_impl()` 在每次执行都
  `std::make_shared<scratchpad_t>`；即使调用方传入 user scratchpad，shared control block 仍已分配，随后
  `prolong_scratchpad_lifetime()` 才因 user-managed 直接返回。首线程的 execution-args TLS 初始化还另有
  warmable allocation；
- 以 `ONEDNN_BUILD_GRAPH=ON`、`ENABLE_PRIMITIVE=ALL`、CPU `SEQ`、GPU `NONE`、user concurrent
  execution 构建官方 `graph-gated-mlp-cpp`，Windows x64 的 `f32 mb=1, ic=64, oc=128` 成功形成一个
  partition 并运行；bf16/f16 在该机器如实返回 unsupported。由此排除“CPU 没有真实 region”，但不能
  排除固定的 steady metadata allocation；
- 该定向构建的 `dnnl.lib` archive 为 1,249,833,896 bytes，官方示例可执行文件为 30,519,296 bytes。
  数字不等同最终 HYLE 安装体积，却证明 Graph 全量构建不能偷偷替换现有轻量 BRGEMM factor；
- oneDNN C API 的 status/fallible 形状、显式 user scratchpad 和同步 CPU stream 可以映射 HYLE
  failure/lifetime；阻塞只来自公开 execute 的固定 allocation 与由此产生的热路径/维护合同。借用 private
  `compiled_partition_t` 仍不能绕过 `larger_partition_kernel_t` 的 `make_shared`；直接修改共享 `.deps`
  源码也违反依赖资源与可重复构建边界。

上述事实命中预声明 stop 条件。P1/P2 不启动：计时快慢不能推翻源码已证明的常态分配，也不能授权私有
ABI 或临时修改 cached source。

## Audit verdict

v1 的“直接适配 oneDNN Graph 公共 API”路线为 `BLOCKED`，不是 Graph 通用 runtime 的 blocker。恢复只有
两类合法入口：未来精确 oneDNN 版本提供 precreated execute metadata；或新的有限卡先审计 HYLE-owned、
build-local 且可升级的 source patch / 现有 oneDNN BRGEMM ukernel compiled-region 路线。后一条必须如实
报告内部 dispatch，不得把一次 HYLE submit 写成一次硬件 dispatch，也不得让 Graph/Compute 反向依赖叶子。

## Candidate

- commit：尚无；
- public API、layout、算法和常态成本变化：尚无；
- 直接消费者迁移：尚无；
- candidate 后仅证据/文档变化：尚未进入候选。

## 红队与裁决

- 红队反证与处理：以公开 API 足够、user scratchpad 等于零分配、一个 partition 等于一个硬件 dispatch
  三个反命题检查源码；前两者被 `partition.cpp`/`large_partition.cpp` 反证，第三者由 CPU
  `larger_partition_kernel_t` 的多 primitive loop 反证，均已写入 BLOCKED 边界；
- 未决限制：公开执行 API 的固定 metadata allocation；未授权 source patch 或 private execute ABI；
- reopen triggers：按 registry 的唯一 resume condition：新 oneDNN seam，或先有独立审计的 build-local
  source-patch / BRGEMM ukernel region 路线；
- LOC：终局统计；
- 最终状态与理由：`BLOCKED`；真实 CPU region 功能成立，但公开路径命中 steady allocation 停止线，
  当前空 factor 不声明依赖、API、feature 或产品能力。
