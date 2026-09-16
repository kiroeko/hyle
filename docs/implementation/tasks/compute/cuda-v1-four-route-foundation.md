# R4 CUDA：原生计算、压缩权重与 Graph replay

## Closure card

- 状态：`SCOPED`。本卡承接[四路线实施合同](four-route-foundation-refactor.md) R4。
- owner：`compute_provider_cublas`、`compute_provider_marlin`、`compute_provider_cuda`、
  `compute_route_cuda`、`graph_route_cuda`。五个身份先在 `2e8eb24a` 登记为 `QUEUED`。
- baseline：`2e8eb24a`；已有产品继承 `d57fa720`，不覆盖其 CPU 修复与验收证据。
- trigger：用户已授权四路线实际实施；CUDA 现有 stream/resource/event/capture 可复用，但没有本轮
  普通默认计算入口、现代压缩 carrier 与完整 Graph 计算旅程。旧 copy/custom-kernel probe 不签此门。
- 包间边界：R3 的 Graph 小片段性能门继续由[原卡](cpu-graph-replay-cost-audit.md)保持 `BLOCKED`。
  本卡不改变其 margin、样本、算法或状态。按 roadmap 的真实依赖及默认包顺序，先推进独立 CUDA owner；
  R3、整个 goal 都不因此宣告完成。若需要修改现有 Compute/Graph/CUDA backend，先登记具体反例和精确
  owner 的新 scope，不以本卡越过它们的状态。
- 主 consumer：vLLM/SGLang 的持久 compressed W4A16 权重、变化 activation、decode/短 prefill FFN，
  去除模型 batching/KV/Torch 运行时策略后，普通 C++ Direct 和 Graph 使用同一 native carrier。
- 正交 consumer：caller-owned GPU dense matrix transform，显式 stream、两槽不同输入、异步完成与回收。
- 直接依赖：既有 Compute schema/runtime、CUDA backend、Mem/Error；Graph 路线额外消费 Graph。
  native Provider 不依赖 Graph，CUDA 路线不取得 CPU/oneDNN。
- 范围：cuBLASLt FP16/FP32 GEMM、锁定现代 Marlin 的 symmetric group128 W4A16、必要 HYLE gated
  activation；冷配置/packing、压缩驻留、两槽与变化输入、有限 shape、普通默认绑定、CUDA capture、
  source/static/shared 与 Windows/Linux NVIDIA 验证。
- non-goals：算子大全、模型执行器、CPU fallback、异卡/跨路线模型调度、Torch runtime、旧 API 兼容。
- 性能：P0 + P1；W1-WIN RTX 4070/SM89 工程机器，WSL correctness/portability；不借 CPU Formal
  资格签 GPU。5% 非劣门与相对 1.0 方向分别报告，实际 CUDA control/driver/device 身份写入冻结合同。
- 调查预算：最多 50 分钟接入源码审计、一个现代 Marlin 无 Torch 编译/正确性 preflight，最多两次
  有明确错误原因的 portability 修正。不是复制旧探针。到预算固定有限 FIX，或记录具体 carrier blocker。
- 实施/性能纠正预算：每个 Direct/Graph 固定候选各一次适用 P1，n=15，资格成立且区间未决时最多到
  n=31；失败只允许一次源码归因的机制修正并冻结新候选。不得改 margin 或无限追加形状/样本。
- stop：依赖的真实合同缺口先依法处理；modern carrier 无法满足错误/资源边界时具体裁决；有硬件
  支持的适用门未跑保持未证明。无设备的其它 SM 不外推为已资格化。

## 能力与接入审计

| 能力族 | 层级 / 准入来源 | 当前能力与边界 | 承载 / 动作 |
|---|---|---|---|
| CUDA resource/stream/event/capture | 原子机制；CUDA SDK、已有真实 GPU consumer | 已有 typed native API 与 accepted-prefix/完成寿命 | ROUTED / KEEP，既有 backend 不重写 |
| dense GEMM | 原子计算；cuBLASLt、数值管线 | vendor 具备算法/描述符/workspace，HYLE Provider 缺失 | BORROW，冷绑定、独立 invocation scratch |
| compressed Q4 linear | 原子计算；当前 vLLM/SGLang Marlin | kernel 可引用；Torch host 包装不能直接作为 C++ SDK | BORROW，版本锁定的窄 native adapter |
| gated activation | 组合所需原子片段；FFN | 现有 CPU 算子语义可复用，CUDA 实现缺失 | SELF，仅真实链路所需 elementwise kernel |
| 普通 Direct 入口 | 复合功能；上述两种 consumer | 需要默认 composition、packing owner、清楚的 prepare/run/teardown | SELF，复用既有 Context/PreparedOperation/Completion |
| CUDA Graph 程序 | 复合功能；持久 FFN 与两槽 replay | 既有 Graph 规划/资源复用和 native capture 可消费真实 Providers | SELF/ROUTED，不重新造通用执行器 |
| 模型 batch/KV/权重文件加载 | 应用策略 | 不属于基础计算合同 | APPLICATION / REJECT 本卡实现 |

具名源码基线：vLLM `51918e252af4dfff381001f42b7ebe6a0454501b`，
SGLang `06992c92ad4494fe9f9636e75be3e35040512447`，见
[已登记 carrier 调查](cuda-compressed-provider-admission.md)。本轮只读核对这些 pin，不重新扫全部历史。
`vllm/csrc/libtorch_stable/quantization/marlin/marlin.cu` 的 `marlin_mm` 在热调用查询 device、选择
config、设置函数属性；`kernel.h` / `marlin_template.h` 承载真实 kernel。`core/scalar_type.hpp`
仍 include Torch exception header；native 接入需显式隔离并记录其维护责任，不能伪装为原样链接库。

### 待审计后冻结的有限 FIX

| ID | 问题 / 允许边界 | 验收 |
|---|---|---|
| C1 | cuBLASLt descriptor/algo/workspace 的冷准备、正常 GEMM Provider | 同语义 oracle、错误映射、不同槽并发、重复执行不查 tactic |
| C2 | 现代 Marlin kernel 的窄 adapter、group128 packing/格式 owner、压缩常驻 | pinned provenance、无 Torch 依赖、冷/热分离、变化输入及数值/尾部限制 |
| C3 | HYLE gated activation 与同 GPU 多 Provider 组合 | 完整 FFN，非 copy/mock，保持各实现真实数值边界 |
| C4 | 普通 CUDA Direct/Graph 入口、固定规则与 native capture | 用户不排序 Provider、不交 profile；正确默认选择、两槽与资源复用 |
| C5 | 实际新增组件的构建隔离、三交付、依赖/许可证与用法 | Win/Linux source/static/shared；CUDA OFF 不探测/泄漏依赖；Windows DLL 邻置 |

表内为有限候选动作；进入 `AUDITED` 时补齐具体 API/packing/tactic/cost 决定后才写产品实现。
新事实改变 scope 时按协议退回，不能顺手改已有治理或性能仪器。

## API / usability 边界

- 普通入口拟为 `compute::cuda::create_context` 与 `graph::cuda::create_context`；用户仅选择 CUDA
  device/resource。已登记 CPU 入口保持原样。底层 typed stream interop 仍在 backend 头中。
- Direct 使用 `prepare`、`create_invocation`、`submit`、真实 `Completion`；Graph 使用 compile、
  bind、冷 prepare_replay、固定 submit。正式冻结前核对普通 capture recipe，不能以专家 callback 替代。
- 压缩权重需要有具名 RAII owner 和公开可发现的冷 packing 路线，描述真实物理格式；不把 FP32 展开
  权重伪装为 compressed 常驻。不让每个普通用户重写 Marlin permutation。
- cuBLASLt 选择算法只在 cold prepare；每槽 scratch、可变状态和 captured pointer 独立。Marlin
  config/device 属性/函数属性只在冷阶段解析，hot 只用已绑定参数与 launcher。
- 默认规则按已支持 dtype/格式/device/shape 静态匹配；无同域收益证据时借成熟实现；HYLE 只补实际
  必要片段。执行时不试跑、不重新计时、无隐式 CPU fallback。
- 外部 tensor/resource borrow 要活到终态；artifact/invocation/code owner 由已有 runtime 保活。
  enqueue 成功不等于设备成功；受理部分 prefix 后的失败必须保存可等待终态。取消不提前释放资源。
- 常态账单在审计冻结时给出；不能凭 capture 支持宣称 vendor 所有内部资源都已计费或零分配。

## 七问与预声明验证

1. 第三方：cuBLASLt 属系统 CUDA Toolkit；现代 Marlin 保留 Apache-2.0 provenance、精确文件/hash/
   适配差异。正式接入进入唯一 dependency lock 与许可投影，未选路线不取得源码或 SDK。
2. 完整旅程：Direct dense/Q4 + Graph FFN，decode/短 prefill、变化输入、两槽，以及正交 dense transform。
   shape/数值与完整 peer 成本先冻结；没有实际运行的组合保持未证明。
3. 内部复用：Compute schema/资源 admission/Completion、Mem、CUDA native backend 与 Graph planner；
   不复制 allocator、event pool、stream manager 或开新 Sched 审计。
4. 性能：与相同选定实现的直接 native 调用/完整 captured FFN 比较包装成本，再给出最强适用完整
   peer。包含必要转换、同步和融合损失；不同格式路线单列，不能签同合同 WIN。
5. FFI：自然 C++；仅 adapter 内部隔离 CUDA TU/第三方细节，不新增 C 产品或动态 plugin ABI。
6. 交付：Windows/WSL 上 source/static/shared 正常 consumer；backend-only/CUDA OFF 的裁剪合同。
   Windows 每次新 exe 启动前将实际依赖 DLL 放在同目录，非交互 CTest + 有限 timeout。
7. 易用：普通代码、source/package target、资源借用、支持的 dtype/SM/shape、capture 失效条件和错误，
   随本包用户指南及样例一起交付。

| Gate | 预声明范围 | 当前 verdict |
|---|---|---|
| scope | 有限 C1–C5、具名 owner、现有 R3 blocker 不改变 | SCOPED |
| architecture | pinned source / 生命周期 / 原生 cold-hot 审计与有限 preflight | UNPROVEN |
| correctness | oracle；非法 dtype/format/shape/资源；OOM；变化输入/双槽；close/abandon/accepted prefix | UNPROVEN |
| delivery | W1-WIN + WSL NVIDIA source/static/shared；CPU/CUDA 隔离；直接用户样例 | UNPROVEN |
| usability_docs | 普通默认入口、packing、capture、资源/完成边界与规则说明 | UNPROVEN |
| red_team | artifact/Queue/resource 错配、失败寿命、未支持域拒绝、消费 API 复核 | UNPROVEN |
| engineering_performance | P0 账单；冻结 API/算法后按统一仪器做实际 workload A/A、双执行/分配顺序、P1 | UNPROVEN |

重型验证串行、编译并行度 2。GPU 设备错误用现有有限隔离 fixture，不主动制造 hang/device loss；
host sanitizer 与 CUDA 适用检查分别报告，外部 vendor/JIT 不冒称已插桩。

## 身份登记验证

`2e8eb24a` 只添加五个 `QUEUED` 声明及状态投影。CMake 自身投影与 baseline 对比：已有 41 个 factor
完全相同、全部 façade root 相同，新增项没有 header/依赖/target；未构建、未跑 P1、无产品完成增量。
closure（带 previous ref）、docs、scorecard、diff 检查通过。selector 原始计划保留在
`.scratch/validation/r4-queued.json`；它把任何中央 manifest 变化扩大为 native gates。上述 native
gates **未执行、未记 PASS**；本次仅身份登记的执行图等价核对保留在 `r4-identity-proof.json`。
真实接线 C5 时再以实际 component/package 差异执行适用门，不为没有 target 的身份登记重建全库。

## Candidate / 裁决

尚无 R4 产品 candidate；没有 R4 correctness、delivery 或 performance PASS。审计完成后记录有限 FIX
和 API 冻结；本卡不能解除 CPU 未决项，也不把旧 CUDA CLOSED/probe 当作本轮完成。
