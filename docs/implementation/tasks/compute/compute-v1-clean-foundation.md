# Compute v1：清洁异构执行基础与 CPU one-op 纵切

> **历史终局记录。** 本卡只记录已发生的 v1 纵切和删除裁决，不授权恢复任何旧源码、API、测试或
> benchmark，也不是当前 Compute 的实现模板。当前状态与工作范围只查
> [`module-status.json`](../../module-status.json)及其所指向的活动记录；旧产品字节只存在于 Git 历史。

## Closure card

- component / owner：`compute` / Compute；round=`compute-v1-clean-foundation`；
- baseline commit：`601b25ad4fe54baa1f8dd19153bb1230085b44d9`；
- trigger 与证据：Compute/Graph 长期需求已由
  [`compute-design`](../../../architecture/compute-design.md)与
  [`graph-design`](../../../architecture/graph-design.md)接受；用户明确要求旧实现不兼容、不得继续污染
  新设计。registry 把 `compute` 记为 `QUEUED`，中央组件图却仍构造旧 target/product，且旧公开头、源码、
  tests、bench、sample 与可选 gate 合计进入约 12.5k 行待审资产；
- 正常 consumer 任务：C++ 推理引擎在 CPU 上创建显式 Context/Queue，把 caller-owned Mem storage 绑定为
  TensorView，准备一次 RMSNorm 并重复提交，以 Completion 管理输出与 workspace 生命周期；
- 主产品纵切 / 具名来源：ggml/llama.cpp、ds4、vLLM、SGLang 的 normalization/低层 operator 调用；
- 正交反偏置纵切：工程计算消费者使用 typed host custom kernel 完成一个 SAXPY 类直接任务，不构造
  Graph，不取得模型语义；
- 跨模块 public journey：Platform 提供 host/ISA 事实，Mem 拥有 storage，SIMD 提供叶子向量核，Error
  表达失败，Compute 拥有 Device/Context/Queue/Completion、tensor binding、prepare 与 submit；
- scope：逐入口裁决并删除旧 Compute 产品面；建立最小 CPU Device/Context/Queue/Completion、
  TensorDesc/TensorView、Operation/PreparedOperation、CPU reference 与 HYLE-native RMSNorm、typed host
  custom-kernel direct path；为 one-op Graph 暴露稳定的窄编译/提交合同；
- non-goals：兼容旧 Device/Stream/Event/DevicePool/GgmlBackend/Kernel API；完整算子目录；CUDA、Metal、
  MLX、MPSGraph、Vulkan、HIP/ROCm；Graph 多节点编译；Native FFI；重开已关闭依赖；
- 直接依赖：公开边为 `core`、`error`、`mem`；实现边为 `math`、`simd`、`util_core`，分别复用 checked
  shape 算术、叶子核和 intrusive runtime ownership；不因 coroutine convenience 公开依赖 Sched；
- 直接消费者：本轮 `graph` one-op 纵切及独立 source/static/shared C++ consumer；
- 性能 tier 与原因：`P1`；RMSNorm 叶子与 prepared-submit dispatch 是真实重复热路，先做 P0 静态成本，
  再与相同 storage/layout/precision 的直接 SIMD/native route 对拍；
- 适用机器 / 仪器 / material margin：W1-WIN CPU 为工程性能机，统一 instrument；抽象税同时报告绝对
  ns 和百分比，5% 只作 material action 门，不把稳定方向改写为 PARITY；GPU 不适用本轮；
- 调查与纠正预算：资产/能力各一遍，API 纠正最多两轮，性能根因纠正最多两轮，每个合格 shape 最多
  15 个 paired samples；达到上限按协议裁决，不扩 benchmark；
- 预声明验证：旧入口/target/gate 负扫描；新 header isolation；正常、OOM/unsupported、move/destroy、
  submit/completion、并发 Context 与关闭测试；Windows/Linux/macOS CPU；ASan/UBSan，适用时 TSan；
  exact factor 与 Compute product 的 source/installed static/installed shared consumer；P0/P1、red-team、
  closure/docs/diff；
- stop / blocker 条件：Tensor/storage owner 无法与 Mem 合法组合、Completion 需要隐藏 process-global
  runtime、one-op 税有无法解释的 material loss、或中央 factor 无法表达独立交付时停止在明确状态。

## 资产隔离原则

旧源码不会复制到新的 `legacy/`、`reference/` 或兼容目录。审计只把仍有价值的事实浓缩到本卡：

| 旧资产族 | 审计问题 | 默认处置 |
|---|---|---|
| Device/Stream/Event 与 global GPU poller | 是否符合显式 Context/Queue/Completion 和有界 shutdown | `DELETE`，除非某个无依赖算法可原样证明 |
| DevicePool 与 compute memory factories | 是否错误复制 Mem storage owner | `DELETE`，需求回到新的 storage bridge |
| generic Kernel/host ops | 是否能成为 typed custom-kernel 路线，而非裸参数袋 | 只保留语义教训，重新实现 |
| ggml bridge/FFN façade | 是否泄漏 provider/model/runtime owner | `DELETE`；未来按 region provider 独立审计 |
| CUDA/Vulkan backend | 是否满足新 backend 合同 | 源码删除；能力分别在后续 factor round 重建 |
| tests/bench/sample/CMake gates | 是否只验证旧形状 | 与旧入口一起删除，不迁移成新验收证据 |

Git 历史负责恢复旧字节；历史 ggml storage 缺陷只通过架构页和既有调查记录保留。

## 能力与形状审计

| 能力族 | 层级 | 准入旅程/来源 | 语义与生命周期 | 原子分解/应用边界 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|---|---|
| CPU device discovery | 原子机制 | direct RMSNorm 与工程 custom kernel | `Device` 是无 owner 的稳定描述；本轮恰有一个 CPU device，枚举不得静默截断 | Platform 提供 host facts；Compute 只形成执行设备身份 | 全新实现 | `MISSING` | `FIX` |
| Context / Queue / Completion | 复合功能 | 重复 prepare/submit 与显式 teardown | `Context`/`Queue` 拥有 runtime；`Completion` 保住 in-flight 状态；close/drain 可观察 | 不含 fiber/coroutine 调度策略 | 全新实现 | `MISSING` | `FIX` |
| tensor schema | 原子机制 | 推理 normalization 与工程数组 kernel | 固定上限 rank、显式 dtype/shape/element-stride；desc 拥有元数据，view 借用 storage | 不包含模型 tensor、KV policy 或第三方对象 | 全新实现 | `MISSING` | `FIX` |
| Mem storage binding | 原子机制 | caller-owned input/output | `TensorView` 组合 desc 与 `mem::BufferView`；不复制、不接管、不伪造 native pointer | allocation/map/coherence 仍由 Mem 拥有 | 全新实现 | `MISSING` | `FIX` |
| operation schema / capability | 原子机制 | ggml/llama.cpp 等反复使用的 RMSNorm | operation 是有类型值；unsupported 在 prepare 冷路径报告 | 本轮只接纳 f32 dense row RMSNorm；不是完整 catalog | 全新实现 | `MISSING` | `FIX` |
| prepared RMSNorm | 复合功能 | 同一 shape 重复执行 | prepare 固定 route/shape/epsilon；submit 只检查 bindings 并调用预绑定 route | CPU reference 用于差分；HYLE SIMD 为正常 route | 全新实现 | `MISSING` | `FIX` |
| typed host custom kernel | 原子机制 | SAXPY 类正交工程计算 | 函数和参数类型在模板边界成立；本轮 CPU queue 同步完成，不持有 caller callable | 不建立字符串 registry、裸参数袋或 Graph | 全新实现 | `MISSING` | `FIX` |
| diagnostics | 复合功能 | 用户解释 selected route | device/backend/provider、workspace 与同步属性可查询 | tracing 属 Obs；CompileReport 属 Graph | 最小结构化属性 | `MISSING` | `FIX` |
| GPU/backend/provider islands | 复合功能 | CUDA、Apple、Vulkan、HIP 后续纵切 | 各自独立 owner、依赖、完成和失败合同 | 不可借旧宏或旧 vtable 冒充 | 后续独立轮 | `OUT_OF_SCOPE` | `DEFER` |

旧公开入口已全部回映：`Device/Stream/Event` 只有设备与完成意图可进入新能力族；`DevicePool`、compute
memory factories、`GgmlBackend`/`FfnBlock` 重复或越过 owner；`KernelArgs`、`BufferBinding`、
`SpecConstant`、`SpirvBlob` 把多后端细节焊在无类型参数袋中；`gpu_poll_*` 是 process-global runtime。
它们没有一个可以原样 `KEEP`。旧 host ops 的算术结果只可作为独立差分输入，不是实现来源。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 | 结果 |
|---|---|---|---|---|
| C01 | 9 个旧公开头、8 个实现 TU、旧 tests/bench/sample 反向定义产品 | 删除全部旧字节；不建 legacy/reference/compat 目录 | 全树负扫描无旧 API/target/test/sample | 完成：`f5581e82` 删除 14,756 行，Git 历史是唯一字节来源 |
| C02 | registry 为待建，中央 CMake 却构造旧 product，并下载 ggml/Vulkan/CUDA 依赖 | 把 manifest、root dependency gate、package/CI/setup 合同改为新 CPU factor；后续 backend 不提前下载 | `compute` 精确选择只取得声明依赖，旧 option 不再是产品入口 | 完成：CPU factor/product 已登记；Windows exact-factor 与精确 product 的 source/static/shared 闭包通过，不存在旧 backend/provider gate |
| C03 | 旧全局 stream/poller 无显式 owner 与有界 teardown | 全新实现 CPU `Device/Context/Queue/Completion` 和 close/drain | 生命周期、move、并发 Context、迟到 completion 测试 | 完成实现；Windows move/close/retained-handle/OOM 测试通过，跨平台矩阵待 C07 |
| C04 | 旧 kernel 参数把 storage 退化为 `void*` | 全新实现 `DataType/TensorDesc/ConstTensorView/TensorView`，直接组合 Mem views | rank/overflow/stride/size/domain/alias 边界测试 | 完成实现；Windows rank/zero/scalar/overflow/stride/range/alias 测试通过 |
| C05 | 热路混合 route、compile、launch | 全新实现 typed RMSNorm descriptor、capability、`PreparedOperation` 与 SIMD/reference routes | prepare 后零 heap、零 route search；完整数值/失败差分 | 完成实现：prepare 缓存 route/count，submit 复用 SIMD；数值、weighted、in-place、coherence/provider failure 通过 |
| C06 | 无模型偏置的 custom compute 只有裸参数袋 | 提供 CPU-only typed host submit；语义明确为 return 前完成 | compile-time misuse 与 SAXPY journey | 完成实现；nothrow void/Status callable、同步完成和 closed-queue 测试通过 |
| C07 | 旧测试只证明旧 API | 从需求重新建立 header/correctness/OOM/lifetime/concurrency 测试 | 预声明 CPU matrix 全部有 verdict | 完成：Windows/MSVC、macOS ARM64/AppleClang 21 和 WSL/GCC 14 的 unit/header 全过；WSL ASan、UBSan、TSan 各 2/2 Compute 门通过且无报告 |
| C08 | 旧 sample、setup、package smoke 和 guide 宣称旧功能 | 重写 Compute 用户页、package consumer 和选择/限制说明 | source/static/shared consumer 使用同一 API | 完成：独立用户页与 package consumer 使用同一 prepared RMSNorm/typed-host API；Windows source、可搬迁 shared、可搬迁 static 均编译运行通过，shared export hygiene 通过（5 库、659 exports） |
| C09 | 没有同合同 one-op 抽象成本结论 | API 稳定后做 P0 静态账单、短 preflight 与一次 P1 | 不存在无法解释的 material 固定输 | 完成：消除 view/数组复制、重复 properties 读取和固定 8-rank 比较后，正式仪器全 PASS；相对语义更窄的 direct Mem+SIMD 固定安全合同成本为 22.2--29.8 ns，明确降为非等价 diagnostic，不伪造胜负 |

本轮不保留任何旧实现以等待未来比较；若后续 GPU/provider 需要某段算法，只能从 Git 提取事实并在新
backend closure card 中重新审计，不能把旧 TU 恢复进产品树。

### API / usability freeze

普通 C++ 路线固定为下列语义；精确 overload 可在 C01--C06 内作至多两轮纠正，但不得改回旧对象：

```cpp
auto context = HYLE_TRY(hyle::compute::Context::create(
    hyle::compute::Device::cpu()));
auto queue = HYLE_TRY(context.create_queue());
auto desc = HYLE_TRY(hyle::compute::TensorDesc::dense(
    hyle::compute::DataType::kFloat32, shape));
auto prepared = HYLE_TRY(hyle::compute::prepare(
    context, hyle::compute::RmsNorm{.epsilon = 1e-5F}, desc, desc));
auto completion = HYLE_TRY(prepared.submit(
    queue, input_views, output_views));
HYLE_TRY(completion.wait());
```

- `Device` 可复制且不拥有 runtime；`Context`、`Queue`、`PreparedOperation`、`Completion` 为显式对象，
  owner 被销毁或 close 后的新操作失败，析构不做无界隐藏等待。
- `TensorDesc` 自含固定 rank 元数据；`TensorView` 借用 Mem storage，Buffer/resource/context 必须活到
  Completion 终态。错误 dtype、rank、shape、stride、byte range、storage domain 和 alias 在提交前返回
  `Status`，不依赖 assert。
- 普通重复路线是 prepare-once/submit-many；一次性 convenience 若以后加入，名称必须披露冷路径。
- CPU v1 queue 是 caller-thread、in-order、同步完成：无 heap、lock、atomic、thread handoff 或隐藏 copy；
  `Completion` 仍作为统一生命周期结果。typed host submit 同样在返回前运行，因此不捕获 callable。
- prepared submit 允许一次预绑定函数调用；不允许 provider 搜索、JIT、元数据分配、字符串 lookup、隐藏
  staging 或 copy。CPU 路线仍通过 Mem 的显式 host map/coherence 合同取得存储。RMSNorm 接受 dense
  contiguous f32 row，允许 input/output 完全同址，不允许其它 overlap。
- 默认 device 只可由显式 `Device::cpu()` 取得；本轮不提供环境相关 `default_device()`。缺能力返回
  `Unimplemented/FailedPrecondition`，OOM 保留 `ResourceExhausted`。
- source/static/shared 都以 `hyle::compute` product 和公开头发现；Native FFI 为 N/A。

## 七问

1. 第三方路径：首轮没有新第三方；正常 RMSNorm 复用已关闭 HYLE SIMD，reference route 为 HYLE 测试
   代码。ggml/CUDA/Vulkan 旧依赖从 active gate 移除，未来各自锁版本、许可证和 fallback。
2. 功能完整：首轮覆盖一个真实推理 op 与一个正交 custom-kernel journey，并建立以后扩 catalog/backend
   所需生命周期；完整 op 集和 GPU 明确不冒充完成。
3. 内部复用：Error 表达失败，Mem 拥有 storage，SIMD 提供叶子核，Platform 只在设备事实确需时私有
   使用；不再复制 allocator、poller、scheduler 或 status。
4. 性能：设备/desc 检查为 O(rank)，RMSNorm 为 O(n)/O(1) workspace；prepared steady 路径零常态分配、
   零搜索、零 handoff。P1 在 candidate 后按同一 SIMD route 做 one-op dispatch 税。
5. FFI：旧 C/Native FFI 不存在；旧 C++ ABI 不兼容且完全删除。跨语言 runtime 本轮 N/A。
6. 构建：C02/C08 必须证明精确 source、installed static/shared；旧可选 backend 宏不进入产物 identity。
7. 易用：对象顺序对应用户任务，错误与 owner 明示；guide 必须同时给 prepared 正常路线、typed host
   路线、in-place 规则、storage lifetime 与同步 CPU 限制。

## 当前状态

候选 `5c9dc1604e6daaec1d73062296f2a71d019e7b35` 的 C01--C09 已全部完成。Windows、macOS、WSL、
sanitizer、source/static/shared、红队与 P1 事实见本卡及
[`benchmarks-compute-graph`](../../evidence/performance/benchmarks-compute-graph.md)。本卡只终结清洁 CPU
one-op foundation；完整算子目录和异构 provider 必须由新的有限卡合法重开，不能把本轮称为完整
Compute 产品。

## Terminal record（2026-09-12）

- candidate：`5c9dc1604e6daaec1d73062296f2a71d019e7b35`；tree：
  `add3629707ab890ca7ee2c76e9dc3d773b20f9b3`；候选之后只有 benchmark peer 身份、证据和状态文档变化；
- 终局范围：只签发 CPU one-op foundation，不签发完整算子目录或任一异构 backend/provider；
- LOC：Compute 实现 `1,122` 行 / `10` files；候选全库 `554,435` 行，自实现不含测试 `96,718` 行，
  均由 `loc-report.py --commit 5c9dc160 --self-check` 取得。

| 关闭门 | 结论 | 终局事实 |
|---|---|---|
| scope | `PASS` | C01--C09 全部完成；旧资产只留 Git 历史，本轮边界没有扩成完整 Compute |
| architecture | `PASS` | Device/Context/Queue/Completion、显式 Mem binding、typed Operation/PreparedOperation 与 host escape hatch 符合接受的 Compute 架构 |
| correctness | `PASS` | Windows/MSVC、WSL/GCC 14、macOS/AppleClang 的 unit/header 全过；当前候选 WSL ASan/UBSan 全过，TSan 在禁用 ASLR 后 23/23 全过且无报告 |
| delivery | `PASS` | `compute` 精确 product 的 source、relocated installed shared、relocated installed static consumer 全过；无 Native FFI |
| usability_docs | `PASS` | package sample 与 guide 覆盖 prepare-once/submit-many、typed host、in-place、borrow lifetime、同步 CPU 限制和错误恢复 |
| red_team | `PASS` | 终局负扫描无旧 DevicePool/GgmlBackend/KernelArgs/gpu poller；steady 路无隐藏 allocation/search/JIT/copy；没有 benchmark-only 公开入口 |
| engineering_performance | `PASS` | P0 成本账单成立；同合同 Graph/Compute P1 非劣门通过。Compute/direct 只作不同安全语义的 diagnostic，不伪造胜负 |

终局验证使用当前候选产品字节：Windows 的 Compute/Graph 精确 unit/header，以及 Compute selected-product
source/shared/static relocated smoke 全过；MacBook Air 四项定向测试 4/4；WSL ASan 与 UBSan 各 4/4。
WSL TSan 的 PIE 在默认 ASLR 下曾在 `main` 前报 instrument 自身 `unexpected memory mapping`，同一二进制以
`setarch x86_64 -R` 运行后 Compute 23/23、Graph 14/14 全过；这不是产品竞态报告。

P1 合同、原始 bundle、hash 与精确重放见
[`benchmarks-compute-graph`](../../evidence/performance/benchmarks-compute-graph.md)：Graph/Compute 四个
99.26% CI 全跨 `1.0` 且落在 5% 等价带，方向为 `PARITY`、工程门为 `PASS`。终局红队没有发现需要把
候选退回 `AUDITED` 的产品问题。

合法重开条件：真实 multi-op/算子目录卡；任一具名 provider/backend 卡；可复现 correctness、lifetime、
delivery 缺陷；或同合同合格证据显示 material 回归。one-op `CLOSED` 不能被解释为完整 Compute 已完成。
