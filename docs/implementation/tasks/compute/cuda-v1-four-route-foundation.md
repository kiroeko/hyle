# R4 CUDA：原生计算、压缩权重与 Graph replay

## Closure card

- 状态：`AUDITED`。本卡承接[四路线实施合同](four-route-foundation-refactor.md) R4。
- owner：`compute_provider_cublas`、`compute_provider_marlin`、`compute_provider_cuda`、
  `compute_route_cuda`、`graph_route_cuda`。五个身份先在 `2e8eb24a` 登记为 `QUEUED`，
  `d1d137e8` 进入 `SCOPED`；另绑定 `compute_backend_cuda` 的下述 C6 私有互操作边。
- baseline：`2e8eb24a`；已有产品继承 `d57fa720`，不覆盖其 CPU 修复与验收证据。
- trigger：用户已授权四路线实际实施；CUDA 现有 stream/resource/event/capture 可复用，但没有本轮
  普通默认计算入口、现代压缩 carrier 与完整 Graph 计算旅程。旧 copy/custom-kernel probe 不签此门。
- 包间边界：本卡启动时 R3 Graph 小片段性能门由[原卡](cpu-graph-replay-cost-audit.md)保持 `BLOCKED`。
  后续用户批准的例外已由[独立验收卡](cpu-r3-acceptance-exception.md)完成 R3 结案，原始 margin、
  样本与统计结果不变；整个 goal 仍未完成。若需要修改现有 Compute/Graph/CUDA backend，先登记具体反例和精确
  owner 的新 scope，不以本卡越过它们的状态。审计已确认 C6 需要复用 backend 私有 native scope，
  为该 owner 从 R3 `CANDIDATE` 定向退回 `AUDITED`；其旧交付证据保留，不修改 CPU/Graph owner。
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

### 审计冻结的有限 FIX

| ID | 问题 / 允许边界 | 验收 |
|---|---|---|
| C1 | cuBLASLt descriptor/algo/workspace 的冷准备、正常 GEMM Provider | 同语义 oracle、错误映射、不同槽并发、重复执行不查 tactic |
| C2 | 现代 Marlin kernel 的窄 adapter、group128 packing/格式 owner、压缩常驻 | pinned provenance、无 Torch 依赖、冷/热分离、变化输入及数值/尾部限制 |
| C3 | HYLE gated activation 与同 GPU 多 Provider 组合 | 完整 FFN，非 copy/mock，保持各实现真实数值边界 |
| C4 | 普通 CUDA Direct/Graph 入口、固定规则与 native capture | 用户不排序 Provider、不交 profile；正确默认选择、两槽与资源复用 |
| C5 | 实际新增组件的构建隔离、三交付、依赖/许可证与用法 | Win/Linux source/static/shared；CUDA OFF 不探测/泄漏依赖；Windows DLL 邻置 |
| C6 | 复用 backend 已有 primary-context scope；提供同构建 Provider 的窄私有声明/导出 | 各 Provider 不复制 context manager；保留准确 context、push/pop 与错误合同；shared 真实链接 |

以上 C1–C6 冻结。C6 的具体证据是 `ProviderVTable::create_context` 传入精确 backend state，但 CUDA
现有 `ScopedPrimaryContext` 定义只在私有 `cuda_internal.h`，跨 Provider DSO 没有可链接的声明/导出。
将其已有实现原样投影到小型私有 bridge，并借取 exact primary handle；不扩大公共 API、不新造第二套
context owner、不修改 event/capture 算法。新事实改变 scope 时按协议退回，不顺手改治理或性能仪器。

### 现代 carrier 预检结论

[原始来源、构建和运行证据](../../../../evidence/research/cuda-r4-modern-marlin-preflight/README.md)：
当前 vLLM pin 的 kernel，在 Windows 无 Torch runtime 下通过 12 次真实变化输入，M=1/7/17/64、
K=N=1024、group128，最大 double-oracle 绝对误差 0.001953125；权重 524,288 bytes、scale
16,384 bytes。设备 kernel 未改算术。仅隔离 scalar descriptor 的 Torch 检查、修正 LLP64 模板 ID
类型、取出现有 repack kernel；一次编译纠正恢复上游使用的 relaxed-constexpr flag。
这证明 C2 的接入机制成立，不签 HYLE 产品、SOTA、Graph、并发、三交付或 P1。

## API / usability 边界

- 普通入口为 `compute::cuda::create_context` 与 `graph::cuda::create_context`；用户仅选择 CUDA
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

冻结的实现边界：cuBLASLt FP16/FP32、FP32 accumulation、plain rank-2 row/column layout；不偷偷
切成 TF32。冷 `cublasLtMatmulAlgoGetHeuristic` 绑定合格 algo，hot `cublasLtMatmul` 显式传 stream/
workspace，参照 [NVIDIA 13.3 heuristic/capture 合同](https://docs.nvidia.com/cuda/archive/13.3.0/cublas/index.html)。
每 invocation 保有自己的可变 vendor handle 与 scratch，prepared 描述和 algo 固定；vendor opaque
storage 未知就报告 unknown，不假报严格配额可计费。Direct 提交不创建新的 handle/描述符或选 tactic。

Marlin 冻结为 FP16 activation/output、FP32 reduction、signed symmetric group128；首轮支持 SM80+
可编译 native kernel（实际资格只给测试过的 SM89），K 为 128 倍数、N 为 64 倍数、M=1..64。超过域
明确拒绝，不解压为浮点冒充同格式 fallback。cold packing 使用既有 GPTQ repack/permutation，typed
`QuantizedWeights` owner 提供 weights/scales view 给普通 QuantizedMatMul 与 Graph constant；物理
format ID 由 CUDA adapter 拥有，用户不手写 permutation。所有 input/resource borrow 的寿命明确。
kernel 的 tile/config/SM/共享内存需求在 prepare 固定，每槽 FP32 reduce scratch 和 locks 独立；
前一次成功完成使 locks 回到可重用状态，失败槽不未经 reset/重准备继续使用。

HYLE 中间 kernel 仅 FP16/FP32 separate/packed SwiGLU，复用现有 GatedActivation 语义；相同 CUDA
stream 上与 cuBLASLt/Marlin 组合。普通 Graph 可通过既有 `prepare_replay(kPreferCapture)` 形成
native captured 程序；捕获失败按已声明的模式错误/报告，不假报 captured。Graph 的规划/绑定保持
共同层，CUDA kernel 的配置/launch 保持 CUDA owner，无需新造通用执行器。

P0 静态成本预算：Context/prepare/packing/每槽创建允许显式 metadata/SDK 分配、冷上传与同步；
成功 replay 的 HYLE adapter 不分配、不重打包、不查设备、不查规则。Direct 保留现有 admission +
一次最终 Completion；Graph capture 只在冷 prepare_replay 逐节点调用 launcher，hot 使用原生 replay。
图/Direct 的完整 native peer 必须包含相同 resource/数值/完成边界，vendor 内部成本全纳入计时。

### C4 普通入口实现定界

沿上述已审计边界实现两个路线头 `hyle/compute/cuda.h`、`hyle/graph/cuda.h`，本身不包含 CUDA SDK
类型。`DefaultContextOptions` 只接受设备 ordinal 与借用 metadata allocator；factory 内部组成三个
适用域互不重叠的 Provider：plain dense MatMul → cuBLASLt，既定 Q4 carrier → Marlin，SwiGLU →
HYLE。普通 `cuda::QuantizedWeights` 引用已实现的 packing owner，不复制格式或资源管理。

Graph 路线提供 `compile(graph, context, device_resource, options)` 与
`prepare_replay(invocation, queue, options)`，在冷路径内构造单 GPU 的 storage/target/lane 绑定。
默认 replay 为 PreferCapture + RejectHostWait，真实路线仍可查询；RequireCapture 可显式要求失败时
返回错误。公共共同层不改变；原生 MemoryResource/stream interop 继续由 backend 拥有。绑定地址变化
沿既有合同使 capture 失效，改变原地址内的输入内容则可重放。高级多 lane 继续使用既有显式环境。

产品测试走普通 factory 的 dense/Q4 FFN、固定常量、两个独立槽/stream、变化输入、实际 captured
报告、无热 HYLE metadata 分配及完整 oracle；不以 factory 或单算子通过代替完整 CUDA 路线。

## 七问与预声明验证

### 当前用户确认的环境条件

2026-09-16 用户明确确认保留依据 GPU 型号、架构/SM、驱动及 CUDA Toolkit/Provider 版本设计
优选规则的能力，详见[路线适用域](../../../architecture/compute-routes.md#11-优选规则的硬件与软件适用域)。
C4 的普通入口不把规则限制为只看 op/dtype；需要的设备/软件事实由路线在 cold create/prepare
读取，Provider 的 native prepare 可匹配有证据的适用条件。执行只调用已经绑定的程序，普通用户
不提供排序或 profile。当前只有 SM89 实际资格，不能把 SM80+ 可编译边界写成全架构 SOTA；未知
环境不自动命中性能 winner。没有具体 winner 条件时不为未来假设新造公共硬件 schema 或规则引擎。

### C1/C5 适用尾项的有限审计

以下两项属于已冻结 C1/C5，相关 CUDA owners 仍为 AUDITED；不改旧 CPU/Graph runtime 或公共工具。

- C1 的 cuBLAS metadata OOM 检查尚缺：沿已有 Marlin 测试方式，逐项注入 Context、prepare、
  invocation 的 HYLE metadata 分配失败，检查 ResourceExhausted、回滚和最终零存活分配；预留
  Completion 后用真实 GEMM 验证重复执行不增加 HYLE metadata allocation。SDK opaque 内存不冒称覆盖。
- C5 的 Linux CUDA host 插桩存在已复现配置错误：nvcc 13.3 拒绝直接传入 `-fsanitize=address`；
  同一 TU 的 dry-run 加 `--forward-unknown-to-host-compiler` 后，host compile/link 均收到该参数。
  [原始预检](../../../../evidence/correctness/cuda-r4-host-instrumentation-preflight/README.md)
  只证明参数接受，不是 sanitizer PASS。
  只在 `compute_provider_marlin`、`compute_provider_cuda` 的非 MSVC sanitizer 构建中加此转发。
  `sanitizer=off` 的 shipping compile/link 命令不变；不改 HyleSanitizers、统计工具或 Windows ASan 合同。

应用后完成 cuBLAS 实际测试、WSL 六个 CUDA factor 的普通回归与适用 host ASan/UBSan/TSan。
重型串行、Ninja 两 jobs；编译失败只按具体诊断作有限适配纠正。外部 CUDA/SDK 与 device code 不签
host 插桩覆盖，任何未运行项保持未证明。原 C5 frozen candidate 的交付证据与本尾项分开记录。

1. 第三方：cuBLASLt 属系统 CUDA Toolkit；现代 Marlin 保留 Apache-2.0 provenance、精确文件/hash/
   适配差异。正式接入进入唯一 dependency lock 与许可投影，未选路线不取得源码或 SDK。
2. 完整旅程：Direct dense/Q4 + Graph FFN，decode/短 prefill、变化输入、两槽，以及正交 dense transform。
   shape/数值与完整 peer 成本先冻结；没有实际运行的组合保持未证明。
3. 内部复用：Compute schema/资源 admission/Completion、Mem、CUDA native backend 与 Graph planner；
   不复制 allocator、event pool、stream manager 或开新 Sched 审计。
4. 性能：与相同选定实现的直接 native 调用/完整 captured FFN 比较包装成本，再给出最强适用完整
   peer。包含必要转换、同步和融合损失；不同格式路线单列，不能签同合同 WIN。
   [P1 源码预审与有限合同](cuda-r4-performance-contract.md) 已将裸 API 的不同生命周期服务明确
   归为 NATURAL_ROUTE；产品 NI 使用相同公共生命周期的完整 Provider 组合，不能把更弱裸调用改名
   为 SAME_CONTRACT。原 5% 门和有限采样/纠正预算不变。
5. FFI：自然 C++；仅 adapter 内部隔离 CUDA TU/第三方细节，不新增 C 产品或动态 plugin ABI。
6. 交付：Windows/WSL 上 source/static/shared 正常 consumer；backend-only/CUDA OFF 的裁剪合同。
   Windows 每次新 exe 启动前将实际依赖 DLL 放在同目录，非交互 CTest + 有限 timeout。
7. 易用：普通代码、source/package target、资源借用、支持的 dtype/SM/shape、capture 失效条件和错误，
   随本包用户指南及样例一起交付。

| Gate | 预声明范围 | 当前 verdict |
|---|---|---|
| scope | 有限 C1–C6、具名 owner；R3 用户例外由独立卡结案 | AUDITED |
| architecture | pinned source / 生命周期 / 原生 cold-hot 审计与有限 preflight | AUDITED，待产品消费复核 |
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

尚无完整 R4 产品 candidate；C1–C5 未完成，完整路线 correctness、delivery 或 performance 未验收。
有限审计在预算内停止，进入 C1–C6 实施；不把旧 CUDA CLOSED/probe 当作完成。
CPU 例外由独立卡结案，本卡不改变其原统计。

### C6 产品检查点

`42ae3005` 将既有 ScopedPrimaryContext 声明移到私有 `native_context.h`，导出原构造/析构，
增加只读 primary-context borrow。原 push/pop 算法未变，同构建 Provider 可以复用该 scope。
[W1 检查证据](../../../../evidence/correctness/cuda-r4-native-context/README.md)：shared/Werror、
Ninja 并行度 2、CUDA backend 独立 factor，三个适用 source 测试通过，DLL 实际邻置。
selector 只选 docs/closure/configure/backend CUDA；没有 P1，没有签完整 R4 三交付或 Provider 集成。
统一脚本 Compute + Graph 全实现 **31,034 行**（不含测试，self-check PASS）；源码规模包含旧资产。

### C1 产品检查点

cuBLASLt 的实际 Provider 已实现并通过 [W1 shared 定向检查](../../../../evidence/correctness/cuda-r4-cublas/README.md)：
2/2 CTest（公共头与四个 GPU 用例），FP32 四种输入布局、FP16 decode/短 prefill、转置权重、
三次变化输入、双槽及 double oracle；受控 pending Completion 验证释放 artifact/invocation、
关闭 Context 后仍保留真实终态。C6 bridge 已被独立 Provider DSO 实际消费。

本次有限域明确为非零 rank-2、匹配 FP16/FP32；输入 row/column-major，普通 MatMul output 为
dense row-major；tensor 16-byte alignment，workspace 256-byte alignment、最多 32 MiB。
F32 使用 pedantic compute；FP16 split-K 禁止 output-type partial sums，保持 FP32 accumulation。
每槽独占 vendor handle/workspace，prepare 后不再查 heuristic；SDK private bytes 仍报告 unknown。

selector 的 cuBLAS factor 配置只增加 CUDA=ON，现有 45 个工具单测通过，没有重写仪器。
中央 manifest 变化产生的其它 native/package 选择项尚未执行、不记 PASS；待 C5 最终闭包集中裁决。
仍为 AUDITED 实施检查点，未进入完整 R4 CANDIDATE；C2–C5、OOM 专项、完整 capture/三交付/P1 仍待完成。
统一脚本 Compute + Graph 当前全实现 **31,486 行**，不含测试，self-check PASS。

### C2 产品检查点

Marlin 已成为真实独立 factor：[W1 shared 定向证据](../../../../evidence/correctness/cuda-r4-marlin/README.md)
通过公开头与五个 GPU 测试，包含 11 组矩阵形状、双槽三次变化输入、独立 double oracle、压缩常驻、
move/close/Completion 寿命、非法域和 host/metadata OOM 回滚。预留 Completion 后 replay 不增加
HYLE metadata allocation；未把这一事实外推为 CUDA driver/JIT 无隐藏内存。

pin、Git blob/hash、真实 archive hash、改造后的 vendor tree hash 与 Apache LICENSE/NOTICE 均已
登记。普通用户可调用冷 `QuantizedWeights::pack`，只保留 Q4 与 FP16 scales。native 配置和 device
查询只在冷阶段完成；编译含 SM80 PTX 与 SM89 native image，实际 GPU 正确性仅验证 SM89。
现有 kernel 的 32-bit 索引进一步要求 N*K <= INT32_MAX；冷边界拒绝越域，不让大尺寸静默溢出。
scale 及反量化权重采用 FP16，累加/reduction 采用 FP32。

源码确认一个真实依赖缺口：共同 Compute/Graph 在失败 terminal 后也把槽设为可再次提交，可能
复用未清理的 Marlin locks。当前 header 要求失败后重建槽；下一步先按精确 owner 定向定界、审计并
修复准入，不能对 CLOSED Compute/Graph 直接施工，也不能以每次热 memset 掩盖生命周期缺口。
这不重开 CPU 算法或已批准的 CPU 性能例外。

该缺口已交由[失败执行槽退役卡](r4-failed-invocation-retirement.md)，精确绑定 Compute/Graph owner
并冻结 F1–F3；R4 其它 C1–C6 范围保持原样。

C2 仍为实施检查点；C3–C5、普通默认入口、真实 Graph/capture、最终三交付及 P1 未完成。
selector 扩大的其它 native/package gates 保持未运行，不记 PASS。统一脚本 Compute + Graph
全实现 **32,183 行**，不含测试及第三方，self-check PASS。

### C3 产品检查点

HYLE CUDA SwiGLU 已接入独立 factor 和无 SDK 类型的公共专家头，
[W1 shared 初始证据](../../../../evidence/correctness/cuda-r4-kernels/README.md)验证 FP16/FP32、
packed/separate、48 组 type/layout/shape、双槽及三次变化输入；非法域、NaN/Inf、实际 alias、
pending/关闭寿命和 metadata OOM 回滚均通过。一个测试按实际 submit binding 检查时机修正后
定向通过；产品代码和公开合同没有为此改动。selector 的既有 45 个工具测试通过。

冷阶段固定原生 kernel/configuration，热阶段只提交单 kernel。没有 caller workspace 或额外
native invocation 状态，不把 driver/code image 全局存储记为零。不支持空 shape、非 dense、
非 FP16/FP32、非 SiLU 或输出元素数超过 INT32_MAX。两种输入布局在最后一维解释 gate/value。

C3 中间片段已经可真实执行；同 GPU 多 Provider 的完整 FFN 仍随 C4 验证。本卡仍 AUDITED，
普通默认入口、Graph capture、跨平台三交付和 P1 未完成；中央 seam 扩大的其余 native/package
选择项尚未运行、不记 PASS。统一脚本 Compute + Graph 全实现 **32,562 行**，不含测试及第三方，
self-check PASS。

### C4 产品检查点

[W1 实际链路证据](../../../../evidence/correctness/cuda-r4-routes/README.md)：普通 Direct/Graph
factory 已组成三个原生 Provider，用户不提交排序或 profile。dense/Q4 FFN 与混合三 Provider
链经过 double oracle、双槽变化输入、实际 native capture/command、重绑定和真实 pending 寿命
验证；冷准备后重复运行没有新增 HYLE metadata 分配。两个公开头和 Direct/Graph 共五个 GPU
用例最终通过，含一次仅测试夹具的阻塞顺序纠正，原始日志保留。selector 工具 45 项通过。

尚未完成 C5 的公开产品 roots/用户样例、Windows+WSL 三交付、最终规则性能资格。C4 的两个 factor
已真实实现，当前 `hyle::compute/graph` 根列表仍保持原样；接线需要按既有 selector 将其涉及的旧
root owner 定向绑定交付范围，不扩大到 CPU 算法重审、不重写工具。完整 R4 尚无 terminal candidate，
不把 factory 或本 W1 checkpoint 当作路线完成。当前全实现 LOC **32,662 行**（不含测试/第三方）。

### C5 交付收口

[C5 交付卡](../build/cuda-r4-product-delivery.md) 已完成公开 roots、实际 FFN 样例与 Windows/WSL
源码、静态、动态消费；CUDA OFF/CPU-only/backend-only/core 隔离及 106 项 selected gates 全部通过。
八个旧 owner 按交付范围 CLOSED，原 CPU 数值与唯一用户例外不变；六个 CUDA owner 仍由本卡 AUDITED。
主卡当前剩余 host 插桩、cuBLAS OOM 新测试及已冻结比较合同的 CUDA P1，不用 C5 结案替代完整 R4。

### GPU 设备内存检查的有限执行范围

host 插桩不覆盖 device 指令。沿本卡既定 CUDA 适用检查，使用本机 CUDA 13.3 Compute Sanitizer
2026.2.1 的 memcheck，对普通 Direct 与实际 captured Graph 的完整 FFN 测试各运行一次，保留
真实错误退出和 leak-check；单程序最多 180 秒，串行。使用 f1cf1112 的普通 shipping 验证副本：
后续 a8c98249 只添加 TSan-only host 投影与测试同步，GPU 与 sanitizer=off 产品源码不变。
不运行故意触发非法设备访问的 late-failure 测试，不把该排除伪装成 device 错误处理未验证；其
正常隔离进程旅程已有 ordinary/ASan/UBSan/TSan 证据。本检查不扩算子、peer 或 profiling 仪器。
若工具/驱动实际不能提供 verdict，记录具体缺口，不伪写 PASS 或调整系统驱动配置。

### 正确性 / 插桩尾项完成

[本轮完整证据](../../../../evidence/correctness/cuda-r4-host-qualification/README.md)：Windows 与 WSL
普通六因子检查各 13/13 PASS；WSL ASan 与 UBSan 适用项完成。ASan 只排除独立最小 CUDA 程序也会
产生的精确 2 笔/280-byte 驱动内部分配，逐程序核对数量，并保留全部未抑制报告。
TSan 原 11 项 PASS、两项同步可见性告警经 [有限重审](cuda-r4-host-validation-visibility.md)修正后
2/2 PASS；产品只加 TSan 条件下的最终 acquire 观察，shipping source bytes 等价。实际 device
memcheck 对 Direct/Graph 完整链路均为 0 errors / 0 bytes leaked。新 cuBLAS OOM 回滚和无热 metadata
分配测试在 Windows/WSL 与各插桩模式均实际通过。

六个 CUDA owner 继续 AUDITED。剩余工作是已定界的 A/A、固定候选 P1、原生包装成本诊断和由完整
收益证据约束的默认规则终审；没有重跑 CPU P1、已通过的无关 gate 或把 host 插桩外推成 vendor
全部源码已检查。
