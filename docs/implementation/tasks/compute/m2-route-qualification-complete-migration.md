# M2 组合路由资格完整迁移

<!-- MODULE REVIEW: CLOSED -->

## Closure card

- component / owner：`compute_backend_cuda`、`compute_provider_cuda`、`compute_provider_flashinfer`、
  `compute_provider_ggml_cuda`、
  `compute_provider_marlin`、`compute_route_apple`、`compute_route_vulkan`、
  `compute_provider_ggml_vulkan`；`/root`。
- baseline commit：`35e488a2074f22a73505bd32ed0800600f378309`。
- trigger 与证据：用户明确要求路由系统“完全重构”。前一张
  [组合资格卡](m2-route-qualification.md)只要求“至少一条”真实迁移并只迁移 CCCL，错误地把共同
  matcher 完成当成全部现存路由完成。只读盘点仍发现 FlashInfer/ggml CUDA/Marlin 的 CUDA 架构
  资格、Marlin/Apple/ggml Vulkan 的量化格式资格，以及 Vulkan 按设备/subgroup 选择 shader 的手写
  静态路由。
- 正常 consumer 任务：用户选择 CPU/CUDA/Apple/Vulkan 路线；路线 Context 捕获原生事实；每个
  Provider/原生编译器在冷 prepare/compile 用共同 `RouteRule` 判定已验证的静态实现域，再运行本地
  shape、资源、SDK capability 和生命周期检查，绑定最终程序。
- 主产品纵切：CUDA 默认序列中的 FlashInfer、ggml CUDA、Marlin、CCCL 对同一真实 CUDA Environment
  采用共同硬件/Provider/operation/quantization 合取；规则不命中时继续既有成熟 fallback，执行不重新
  匹配。
- 正交反偏置纵切：Apple affine Q4 与 Vulkan NVIDIA/Apple-MoltenVK matvec、ggml Vulkan MXFP4 用同一
  合取语义，但继续由各自 native owner 构造事实、资源和程序。CPU 当前没有手写硬件 allowlist；SIMD
  自己的 ISA dispatch 保持原 owner，不为了形式制造第二次探测。
- 跨模块 public journey 与每段 owner：共同 `RouteRule` 已由 `compute` CLOSED 交付；本卡只在各 native
  owner 接入静态规则。构建因子删除不可用 Provider，Context 固定设备/版本事实，prepare/compile 合取
  workload/operation/format，prepared artifact 执行已绑定程序。
- scope：完整枚举当前 `src/compute_provider*`、`src/compute_route*`、`src/graph_provider*`、
  `src/graph_route*` 中的平台、硬件/架构、驱动/runtime/Toolkit/SDK/API、Provider 版本、模型、phase/
  region、量化 mode/group/物理格式条件；迁移所有真正选择 Provider 或实现变体的手写静态条件；由
  `compute_backend_cuda` 从不可变 Environment 投影一次共同 CUDA 事实，Apple/Vulkan 由各自 native owner
  投影，消除同一事实的重复路由表达。
- non-goals：不把 rank/shape/stride、容量/overflow、workspace、occupancy、设备 limit、同步、资源
  accessibility、SDK 原生 `supports_op` 或错误诊断搬进 matcher；不改变 Provider 次序、数学、支持域、
  fallback、模型 loader 或 B2 算子范围；不重写 selector/交付脚本。
- 直接依赖 / 直接消费者：依赖既有 `compute::RouteRule`、CUDA/Apple/Vulkan Environment 和各 Provider
  身份；直接消费者是 CUDA 默认 Context、Apple Direct/Graph lowering、Vulkan Direct/Graph compiler。
- 性能 tier 与原因：`P0`；仅冷 Context/prepare/compile 条件表达迁移，submit/execute 必须零 matcher。
- 适用机器 / 仪器 / material margin：Windows canonical；CUDA/Vulkan 真实测试与三交付，Apple 在 Air
  或 Mini 做 native correctness/交付；公共 matcher 已有 Windows header/correctness。没有热算法变化，
  P1 不适用。
- 调查与纠正预算：60 分钟完成全目录资格分类并冻结有限 FIX；每个迁移点最多两次产品纠正；适用验证
  另计。新发现只在满足“静态事实选择实现”定义时加入，不把所有 `Unimplemented` 变成路由规则。
- 预声明验证：自动文本审计证明受管轴不再在产品中手写比较；每条迁移规则的命中/逐轴拒绝与原 fallback；
  Direct/Graph cold-only 计数；CUDA/Vulkan 实机 correctness；Apple native correctness；受影响公开头、
  source/static/shared、selector、docs/closure/diff。
- stop / blocker 条件：若迁移要求把 shape/资源值塞进通用硬件对象、在 submit 重新解释规则、改变数学或
  Provider 优先级，则停止并重新定界；远端 Apple 不可达或真实 SDK 缺失时按实际门 BLOCKED，不由
  Windows 冒充。

## SCOPED 审计清单

1. 对四组目录做受管轴扫描，并逐项区分静态路由、动态语义验证、资源计算、原生 capability probe。
2. 核对每条手写路由的权威事实来源、Provider/version identity、fallback 和 Direct/Graph 消费链。
3. 冻结有限迁移表；现有 B2 算子接线、自研数学和性能比较继续暂停。
4. 先以现有 matcher 表达；只有具体反例无法表达且属于受管轴时才允许修改共同 API。

## 受管轴完整审计

| 位置 | 手写静态选择 | 归类与动作 |
|---|---|---|
| CUDA backend / CCCL | CCCL 已用 rule，但独立重复编码 SM/runtime/toolkit facts | FIX：backend 从 immutable `Environment` 统一投影 device/SM/driver/runtime/toolkit；CCCL 改为复用 |
| FlashInfer | SM 8.9 硬门；attention operation 与 FP16/BF16 实现域 | FIX：硬件/Provider 与 operation/dtype 分别用冷 rule；causal/layout/shape/residency 留本地 |
| ggml CUDA | SM 8.9 硬门 | FIX：用 backend facts 的冷 rule；`ggml_backend_supports_op` 与图语义留原生 owner |
| Marlin | SM80+；Q4 signed/group128/dtype/weight+scale format | FIX：pack/provider 共用硬件 rule，proposal/prepare 共用量化 rule；shape、容量、resource 留本地 |
| Apple | affine Q4 unsigned/group64/dtype/packed format | FIX：在已构造真实 public descriptions 后用 Apple rule；MLX rank/native dtype 与语义推导留本地 |
| Vulkan native | subgroup arithmetic/size + NVIDIA 0x2786 或 Apple M4 决定 matvec shader | FIX：Vulkan Environment 投影事实，两条有序 rule 选 shader；shape 阈值和 device limits 留本地 |
| ggml Vulkan | MatMul F16 + MXFP4 physical format 进入成熟 region | FIX：format 单 tensor 与完整 MatMul 域均用 rule；canonical layout、rank 和 ggml native support 留本地 |
| CPU / Graph 边界 | CPU 无手写硬件 allowlist；SIMD 拥有 ISA dispatch；Apple Graph 解码 public Q4 storage | KEEP：不是实现赢家选择，不复制 ISA 探测，不把边界合法性伪装成路由 |

全目录没有现存 model family/name/revision/configuration、phase 或 region 特例；共同 matcher 已能表达这些
轴，故本轮不虚构模型规则。Vulkan descriptor indexing、IREE feature、reduction lane、CUDA occupancy/
shared-memory 等均是原生 capability/resource 计算，不属于静态 best-of 资格。

## 有限 FIX（AUDITED）

| ID | 有限产品动作 | 必须证明 |
|---|---|---|
| CM1 | `compute_backend_cuda` 提供固定大小、借用式 route facts；CCCL 复用 | 六项事实来自同一 snapshot；无分配；原 CCCL 命中/拒绝不变 |
| CM2 | FlashInfer 与 ggml CUDA 迁移 SM89，FlashInfer 迁移 operation/dtype | 非 SM89 安全 decline；FP16/BF16 原域命中；submit 无 matcher |
| CM3 | Marlin 迁移 SM80+ 与 Q4-S128/weight/scale format | pack、proposal、prepare 一致；非法 shape/resource 仍由原检查裁决 |
| CM4 | Apple affine Q4 迁移 route/operation/dtype/mode/group/format | Direct/Graph 原 Q4 正反例；MLX 程序与输出不变 |
| CM5 | Vulkan 两个设备/subgroup shader 资格迁移为有序 rules | NVIDIA、Apple M4、未知设备三路不变；large shape 只在 NVIDIA 命中 |
| CM6 | ggml Vulkan MXFP4 单 tensor 与 MatMul region 资格迁移 | canonical MXFP4 正例与伪格式/布局反例；Direct/Graph region 不变 |
| CM7 | 受管轴扫描与行为测试；文档列出保留的本地检查 | 产品不再手写上述静态比较；三交付与真实路线 correctness 通过 |
| CM8 | 修正普通 Compute package consumer 对零 workspace 的调用 | requirement 为零时传空 view，非零时才分配；不改变 Compute API、Provider 或算法 |

API 冻结为既有公开 `RouteRule`；没有具体反例要求修改 `compute`。只新增 CUDA backend 私有事实投影和
Vulkan owner 私有 boolean attribute。所有 matcher 调用只在 Context 创建、proposal、prepare/compile；
不进入 bind/submit/wait。P0 检查对象布局、调用位置和成本结构，P1 不适用。

状态进入 `AUDITED`；允许且只允许 CM1--CM7。B2、新算子、自研数学、性能比较继续暂停。

## CANDIDATE freeze

- product candidate：`f15204826ba001524b6a66b6fa2291400226a160`；tree
  `5ef86a1a2a90aad6f480515a5fdbb92b74edff76`。
- CM1--CM3：CUDA backend 统一投影 device/SM/driver/runtime/toolkit 六项冷事实；CCCL、FlashInfer、
  ggml CUDA 和 Marlin 的现存硬件、Provider、operation/dtype 与 Q4-S128 资格均改用共同 rule。
- CM4--CM6：Apple affine Q4、Vulkan NVIDIA/Apple M4 subgroup shader 与 ggml Vulkan MXFP4/MatMul
  资格均改用共同 rule；原生 shape、layout、设备 limit、SDK capability、资源和同步检查留在原 owner。
- CM7：受管轴扫描只保留 Apple Graph 的公开 Q4 storage 解码和 Vulkan reduction 的资源计算；它们不
  选择 Provider 或实现。`match_route_rule` 不在 submit/enqueue/execute/bind/wait 中出现。
- Windows MSVC 19.50、CUDA 13.3、SM89、Vulkan：六个受影响产品/测试目标完整构建；CUDA backend、
  CUDA Provider、ggml CUDA、Marlin、Vulkan route、ggml Vulkan 六个真实套件 `6/6 PASS`。格式化后的
  四个受影响目标增量重建并 `4/4 PASS`；DLL 已邻置 staging，没有交互弹窗。
- P0：facts 是固定大小借用数组；匹配仅发生在 Context 创建、proposal、prepare/compile，prepared
  artifact 和提交路径不保留规则或重新裁决。没有改变数学、Provider 顺序、支持域或 fallback。

产品与 API 在该提交冻结，八个 owner 进入 `CANDIDATE`。尚待 Linux FlashInfer、Apple 原生路线、
三交付、selector 与终局 docs/closure；这些门未运行前不写 PASS。若跨平台验证暴露产品缺陷，按有限
CM1--CM7 退回 `AUDITED` 修正，不借机恢复 B2 或扩算子。

## 三交付反例与退回 AUDITED

Windows CUDA source consumer 在完整编译、链接和 DLL staging 后由 CTest 返回失败。定向诊断证明
普通样例把 `workspace_requirement().size == 0` 的 provider-owned RMSNorm 仍包装成有效的零字节
`BufferView`，`create_invocation` 因而按合同拒绝并报告
`provider-owned operation does not accept caller workspace`。这不是 route matcher、CUDA 数学或
运行库缺失；scratch 中按 requirement 为零传默认空 view 后，同一 consumer 在 RTX 4070 / SM89
实际运行并返回 0。

候选 `f15204826` 的路线产品实现保持有效，但交付样例不足以通过预声明三交付门，故八个 owner 明确
退回 `AUDITED`。新增且只新增 CM8：统一修正样例中 RMSNorm 与可选 oneDNN 两处同形 workspace
准备；不改公共 API、Provider 优先级、算法或其它样例。修正后生成新不可变候选并只重跑受影响的
三交付、原路线定向门及 docs/closure；既有跨平台 PASS 可在产品树身份未变的范围内复用。

## CM8 修正候选

- 新 product candidate：`3d7e212972e27f8f532cb084f5eb5c4656b45016`；tree
  `836df752e2a158f8f3b4d00bbeef9f916f1ac559`。
- 两处普通 Compute package workspace 准备现在只在 requirement 非零时分配；零 requirement 传默认
  空 view，符合 provider-owned invocation 合同。公共 API、路线实现、数学与 Provider 顺序均未改。
- Windows scratch 先稳定复现 `size=0`、有效零字节 view 被拒绝；替换为当前 canonical 样例后只增量
  编译 consumer，实际识别 RTX 4070 / SM89、完成 CUDA FFN 并返回 0。

八个 owner 恢复 `CANDIDATE`。原失败和诊断保留；尚待对新候选完成三交付、Apple 回放、受影响
selector/组件合同、docs/closure。未运行项仍不写 PASS。

## Terminal record

八个 owner 在不可变产品候选 `3d7e212972e27f8f532cb084f5eb5c4656b45016`（tree
`836df752e2a158f8f3b4d00bbeef9f916f1ac559`）关闭。路线产品本身冻结于 `f15204826`；其后 CM8 只修正
普通 Compute package consumer 的零 workspace 调用，未改变路线规则、原生程序、Provider 次序、数学
或 fallback。

### 实现与七门

| 门 | 结论与证据 |
|---|---|
| scope | PASS：全目录受管轴已分类，CM1--CM8 全部完成；shape/resource/capability/sync 与 B2 算子继续在范围外 |
| architecture | PASS：CUDA 统一投影六项事实；Apple/Vulkan 保留原生事实与程序 owner；共同 matcher 只表达静态资格合取 |
| correctness | PASS：Windows 六个定向真实套件 `6/6`；WSL FlashInfer 独立头与四项 FP16/BF16 provider 测试；Air Apple 独立头与 39 项 native 测试 |
| delivery | PASS：新候选的 CUDA、Apple、Vulkan source/static/shared 均实际运行；Vulkan 同时覆盖 Compute+Graph 与隐藏 producer 后的重定位 SDK consumer |
| usability_docs | PASS：普通用户仍只选路线；[Compute routes](../../../architecture/compute-routes.md)说明平台/硬件/软件栈/模型/量化组合与冷绑定，交付入口没有新增排序或 timing profile |
| red_team | PASS：受管轴扫描不再发现剩余手写静态实现选择；未知事实安全 decline；submit/execute/bind/wait 无 matcher；没有把语义和资源检查伪装成路由 |
| engineering_performance | PASS（P0）：固定大小借用 facts；冷路径有限线性匹配；无新增热路径字段、分配、锁、原子、syscall、设备查询或计时，P1 不适用 |

Windows selector 在路线候选上先通过 Compute/Graph gates 003--062，随后被无关
`factor:io_core` 的两个固定 UDP 端口测试阻断；原始 verdict 与未运行 gate 保持 `BLOCKED`，没有写成
PASS。Windows excluded range `54319--54418` 覆盖测试端口 `54355/54394`，精确 baseline
`c810475d` 独立复现相同失败，且 IO 源码/测试无候选差异，因此按工程证据裁决为主机环境限制，不构成
本候选回归。新候选另行完成全部受影响三交付和组件合同。

主要命令与结果：

```text
Windows targeted CUDA/Vulkan route suites                    6/6 PASS
WSL FlashInfer header + native provider                      2/2 PASS
Air Apple route header + native provider                     2/2 PASS (39 native cases)
CUDA Compute source/shared/static delivery                   PASS
Apple Compute source/shared/static delivery, APPLE=ON        PASS
Vulkan Compute+Graph source/shared/static delivery           PASS
cmake -D... -P cmake/HyleCiComponentContract.cmake           PASS
python scripts/loc-report.py ... --commit 3d7e2129 --self-check --json
                                                                58,666 / 251; pass
```

完整身份、命令摘要和原始边界保存在
[证据目录](../../../../evidence/correctness/compute-m2-route-complete/README.md)。统一 Compute+Graph LOC 为
**58,666 行 / 251 个实现文件**，排除 tests、bench、docs、third_party、samples 与构建脚本；这是全部
存量，不作为本轮新增完成量。

### 限制与重开

本轮没有模型特例，因为现有产品中没有已验证的 model family/name/revision 静态赢家规则；共同 matcher
已保留这些轴，后续只能由真实模型与硬件/驱动证据增加。资格限已记录的 Windows RTX 4070/SM89/CUDA
13.3、Windows Vulkan SDK 1.4.357.0、Air Apple M4/MLX 0.32.2 和 WSL GCC 14 消费形状；未知设备或版本
不声明 SOTA。CPU SIMD ISA dispatch、Apple Graph Q4 public storage 解码以及 Vulkan shape/device limit/
资源计算保持原 owner。

重开只接受可复现的错误命中/拒绝、UB/寿命/并发/交付缺陷；真实消费者要求现有 matcher 无法表达的
静态资格；公开路线合同、Provider 优先级或 adopted dependency 相关升级；或合格证据显示热路径新增
matcher/常态成本。普通 B2 算子接线、未知硬件和文档修改不重开本卡。M2-B2 可恢复，但本卡停止施工。
