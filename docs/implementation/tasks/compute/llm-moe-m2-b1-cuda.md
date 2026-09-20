# M2-B1 CUDA：成熟 Provider 的张量、归一化与状态接入

## Closure card

- component / owner：`compute_provider_ggml_cuda`、`graph_provider_ggml_cuda`；`/root`。
- baseline：`c9744a45b9f6aa2fab27d857af70410f04685518`；Windows canonical，工作树干净。
- 当前阶段：AUDITED；`be44efa5` 已独立完成 SCOPED，以下 C1–C4 及本节精确域冻结后实施。
- trigger：[M0 接入清单](llm-moe-provider-integration-ledger.md)的 B1 新批量片段需求，符合
  [M1-C terminal](llm-moe-m1-cuda-admission.md)的 reopen trigger。现有原版 CUDA adapter
  只接有限 F32 rank2 RMSNorm/Add 及 attention，不能承载张量、门控、索引和状态的完整需求。
- 需求真相：[64 行矩阵](llm-moe-coverage-matrix.json)的 F01.1–3、F02.1–4、F03.1–4、
  F04.1–6、F19.1–4，共21行；本卡不缩小需求，不把部分映射或旧 CLOSED 计为完成。
- 主纵切：目标模型 embedding→norm/gate→residual/index/state；同 GPU 原版成熟程序，
  两请求独立、只读常量共享、状态与旧结果保活，payload 改变后重复执行。
- 正交纵切：无模型注解的广播/归约/转换图，带非连续布局、动态错误、关闭与多输出。
- scope：成熟原版 operation/组合的批量映射、普通 CUDA Direct/Graph、原生资源与失败、
  固定冷绑定、适用交付和完整接入成本。Graph 只译 region 边界，Compute 不依赖 Graph。
- non-goals：所有冻结自研算子的扩张、调查、优化、比较或删除；新数学 kernel/shader；
  模型 loader/服务器；公共 selector/性能仪器或全库资源系统改写；oneDNN 旧 Q2 恢复。
- 直接依赖：复用已验收的 Compute/schema、CUDA backend、Graph region、Mem/Error 及
  `compute_ggml_runtime` 的原版 ggml CUDA vendor/资源桥。既有普通 CUDA route 为直接消费者。
  这些 CLOSED owner 不因本卡自动重开；具名新缺陷才另行定向审计。
- 来源：沿用 M0/M1 锁定 ggml `456172ec733a135778adcd32d00e576a58232e45`、MIT、
  已验收资源补丁；每个新增映射查原版后端真实域，不能只看同名公开函数。
- 预算：本批 CUDA 源码/设计总120分钟，自2026-09-20 20:10 UTC起，前置核对保守计10分钟，
  余110分钟。每具名根因最多两次源码驱动纠正；实现细节调查合并计费，不按子卡续杯。
  构建及预声明验证单独记录。CPU 的原120分钟预算和 oneDNN 待决项保持原状。
- 性能：先 P0；API/算法固定的不可变候选才运行一次必要 P1，与同一原版成熟完整调用相比，
  非劣门5%，含转换、同步、融合和实际完成。A/A/双顺序先资格化，15对起，合法总上限31对。
  不比较冻结自研；具体 workload/账单在候选前固定，不在实现阶段计时找答案。
- 机器：Windows/WSL RTX4070 SM89，重型串行、parallel2、原生编译 `89-real`；Air/Mini CUDA-OFF。
  不声称其它架构已有资格。Windows DLL 邻置，不出现 UAC、可见 helper 或电源策略修改。
- stop：全部21行与适用门完成后关闭；未支持/数值不相容先记录原版反例和成熟后备责任，
  不能变为自研授权。预算耗尽须作明确裁决。M5 前按用户指示暂停。

## 待审有限改动集

| ID | 允许边界（AUDITED 前不得实施） | 必须验证 |
|---|---|---|
| C1 | 复用既有私有 Recipe，成批接原版 Copy/Cast/Concat、浮点广播/Unary、norm/归约及合格 indexed 域；只作描述符/端口与原版组合翻译 | 原版真实 dtype/layout/容量/数值域，独立 oracle，拒绝上游不接受或语义不等价的域 |
| C2 | 每 invocation 的既有原生图/stream/buffer/event、必要中间值、绑定与检查 | 不执行准备期用户状态，不漏检查，失败不伪造完成；两 slot、旧结果、OOM/部分构造、回收 |
| C3 | Graph 共用 C1 的能力判断和同一程序；成熟融合、发布中间值及错误/effect 边界 | 普通 Graph 真实组合，资源/生命周期与 Direct 一致；无模型标签也可用 |
| C4 | 测试、普通消费者、指南、三交付与成本记录；仅所属 factor 的必要源文件描述 | 不新造 CMake 依赖机制；Windows/WSL 及适用插桩、OFF、交付、候选复核/P1 |

## 冻结前必须回答

1. rank0..6、正 stride/空值、广播、动态索引和两输出如何映射；ggml rank4 不能当作公共语义上限。
2. F32/F16/BF16 转换舍入与特殊值、stable unary/gate、L2 epsilon、norm-before-gate、全遮罩
   Softmax、重复 Scatter、checked I32 Scan、TopK ties/NaN 的确切成熟承载与拒绝域。
3. 原生操作能否冷验证；动态索引/溢出检查不能以性能为由省略，也不能退回隐藏 CPU 模型计算。
4. 同一 invocation 的外来 buffer/view 与原版内部 storage 所有权、异步失败/销毁、常量上传及
   managed replay 成本；只复用已验收 bridge，不擅自改变 stream/capture 或 backend 合同。
5. 已有默认工厂能否直接消费新增域；模型/硬件/版本条件准备时绑定，执行不搜索或竞速。

## 预声明验证与状态

- 全21行能力账本、主/正交 public journey、独立数值 oracle；Direct 与 Graph 使用同合同输入。
- 边界/类型/shape/stride/alias/索引/溢出拒绝、两 slot、旧结果、OOM/部分构造、失败/关闭；
  不变资源桥的既有身份匹配证据可复用，新操作风险须补实际验证。
- Windows/WSL 真 GPU 的受影响 factor/消费者；host ASan/UBSan 不代签 GPU device 安全；
  Air/Mini 执行 selector 适用 CUDA-OFF，不制造无 CUDA 的 native 旅程。
- 最终稳定程序用一份真实组合 consumer 覆盖 Windows/WSL source/static/shared、prefix 搬移、
  producer 隐藏、依赖/许可证/导出与 DLL 邻置；不逐算子重跑同一 package 接缝。
- selector 选中门及 docs/closure，七门与一次独立候选复核；文档/状态不构建或跑 P1。
- 当前 scope/architecture/correctness/delivery/usability_docs/red_team/engineering_performance
  均未最终裁决；不存在新增完整覆盖 PASS，矩阵仍0/256。

本轮沿用已有公开 CUDA 工厂；pending oneDNN 决策不影响本卡独立工作，也不由本卡代为批准。

## AUDITED：批量承载与精确域

2026-09-20 UTC 核对原 pin 的 `ggml-cuda/{ggml-cuda,unary,binbcast,cpy,concat,norm,sumrows,softmax}.cu`
及已验收 HYLE bridge/Recipe。源码预算累计25分钟、余95分钟；这是同一120分钟预算。

| 需求族 | 原版事实与当前覆盖 | 有限裁决 |
|---|---|---|
| F01 布局/复制/转换 | CUDA CPY 支持 F32/F16/BF16 相互转换；Concat 按 storage 位宽复制；公开 rank 高于原生4维 | PARTIAL/BORROW：dense 高 rank 冷折叠，原生可表示的 aligned stride 用 cont；最终 dense 输出。非浮点与 checked index Cast 仍未接 |
| F03 广播/Unary | binbcast 原版 Add/Sub/Mul/Div/Repeat；Unary F32/F16，BF16 不直接进入该 kernel | PARTIAL/BORROW：半精度先原版转F32、最终一次cast；用原版 reshape/repeat 译完整合法广播；首批 Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh |
| F04 norm/归约 | RMSNorm/F32 sum_rows/mean、reshape/view/cont 已有后端；CUDA norm 使用二阶矩相减 | PARTIAL/BORROW：RMS 原版+affine；LayerNorm 用 mean→sub→sqr→mean→epsilon→sqrt/div 原版两遍组合，再affine和最终cast；任意归约轴先冷布局翻译 |
| F02/部分 F03–F04 | 动态索引、重复Scatter、checked Scan、stable TopK、Select/min/max、stable gate、masked Softmax尚无本轮同合同资格 | PARTIAL；不冒用CPU原版结论，也不以已有自研填新覆盖；下一成熟承载必须具名核对，任何新owner先合法定界 |
| F19 生命周期 | 已有不可变Recipe、每slot原生backend/stream、非owning boundary buffer、两event、真实Completion | PARTIAL/FIX：新增scalar constants在冷期上传；输出reshape的实际root也要绑定外来buffer并标记output，防止view重绑覆盖或融合吞掉发布中间值 |

首批输出/输入须 plain F32/F16/BF16，dense 支持公共 rank0..8；非dense先限原生rank≤4且stride
按元素对齐。空张量、不可折叠高rank stride、奇数字节stride、其它storage与剩余算子仍属于21行
的未完成部分，不削分母。公开 schema 先推导并核对输出，检查元素/byte/原版索引容量；原生每个
中间节点再次核对后端和实际launch维度，不能把 `supports_op` 泛化为所有内部数值域正确。

普通 Direct/Graph 工厂已有 ggml 优先绑定，不需修改其 CLOSED owner。保持 SM89 资格与 M1-R
冷模型/环境事实；未知条件不扩大证书。新算子没有自己的公开factory、runtime或queue。

资源/API freeze：沿用现有public工厂和私有Recipe；每个slot创建一次图/metadata/backend/scratch，
提交只更新外来地址、view依赖及既有两event桥。新增常量由slot独立持有并冷上传，不在prepare执行
用户数据；FP32转换、repeat/cont及原版managed维护如实收费，无法确知的copy/repack/私有字节不写0。
Graph合并合法pure节点，发布的view及其root都标output；attention仍保留已验收完整leaf，动态检查
节点未证明可融合前保留边界。没有新增数学kernel、CPU计算fallback、依赖pin或公共owner变化。

C1–C4 的首批实现由上述精确域约束；未绑定域和全部适用验证继续留在本卡，不按一批提交签收B1。
预声明差分增加：rank6/8双向广播、非相邻轴归约、scalar/strided输入、三种浮点的转换/特殊值，
LayerNorm大偏移常量/低方差行，以及一张发布中间值的多节点图在两个slot重放。
