# M2-B1 Vulkan：成熟张量、归一化与状态的批量接入

## Closure card

- component / owner：`compute_provider_ggml_vulkan`、`compute_route_vulkan`、
  `graph_route_vulkan`；`/root`。当前阶段 SCOPED，尚不授权产品修改。
- baseline：`09d2cf99f1298e988ccda65ca76ec1b4907bd0e7`，Windows main，工作树干净。
- trigger：[总计划](llm-moe-coverage-completion-plan.md) M2-Vulkan 和
  [M0 清单](llm-moe-provider-integration-ledger.md) B1 的真实新消费域，符合
  [M1-V terminal](llm-moe-m1-vulkan-admission.md)的 M2 新 shape/数值/片段重开条件。
  当前 `region_supports` 仅接受有限 dense rank≤4、主要 F32 域；不能承载完整 B1。
- 需求：矩阵 F01.1–3、F02.1–4、F03.1–4、F04.1–6、F19.1–4，共21行。
  首批实现检查点不是缩小分母；尚未接通、语义不相容和未验收分别记录。
- 主纵切：模型 embedding→norm/gate→residual/index/state；正交纵切：无模型标签的
  多维广播、非连续输入、指定轴归约与发布中间结果。两请求独立、输入更新后重放、旧结果保活。
- scope：原版 ggml Vulkan 算子/合法组合的批量 schema 映射、普通 Direct/Graph、
  原生资源与失败、冷绑定和成本、适用三交付。复用已验收 managed region/native bridge。
- non-goals：冻结自研 CPU/CUDA/Metal/Vulkan 数学的调查、扩张、优化、比较和删除；
  SIMD 基础；新 shader/算子算法；模型 loader/engine；公共 selector/性能仪器改写。
- 依赖/消费者：Compute 公共 schema、既有 ggml native device/graph/work、Mem/Error；
  Graph 单向消费 Compute 的同一 region。共同 runtime、IREE 和其他 CLOSED owner 保持不变；
  只有具名新缺陷才另行定向审计。IREE 已接 Scan 的证据按实际身份与域复用。
- 第三方：ggml `456172ec733a135778adcd32d00e576a58232e45`，MIT；沿用锁和资源 overlay。
  原始缓存只读。核对实际 Vulkan backend 与 shader 域，不从 CPU/CUDA 或同名 API 推断资格。
- 预算：本批源码/设计合计120分钟，自2026-09-21 05:50 UTC起，前置核对计入；
  每具名根因至多两次源码驱动纠正。实现细节调查合并计费，构建和预声明验证单列。
  不按子卡重置预算；遇到成熟覆盖未证明的域先具名裁决，不转换为自研授权。
- 性能：P0 查原版分配/编码、转换/cont/repeat、边界copy与同步；固定候选后一次必要 P1，
  对照同一成熟完整调用，相同输入输出/资源/完成语义，非劣门5%，A/A与双顺序先资格化，
  15对起、合法总上限31对。候选前冻结真实工作负载和成本账单，不比较冻结自研。
- 机器：Windows RTX4070、Air/MoltenVK；WSL/Mini 按实际公共/OFF/host风险。
  parallel2、重型串行、Windows DLL邻置、无UAC/电源改动；同一批四路线推进，
  Vulkan 不再等其它三路线全部完成后启动。
- stop：有限 FIX、21行成熟承载及适用门完成才 CLOSED；预算到期须裁决，未运行不签 PASS。
  M5开始前按用户要求暂停；旧C1/M3-01专门暂停已由用户取消。

## SCOPED 审计边界

| ID | 要裁决的有限改动 | 验收边界 |
|---|---|---|
| V1 | Copy/Cast/Concat、浮点 Binary/Unary、norm/归约、indexed/mask/stable gate 的原版承载；高rank折叠、广播/布局与精度翻译 | 与原版真实后端能力及公共语义一致；失败域明确，不能只以名称接线 |
| V2 | 新片段的metadata、外来buffer/view、对齐/范围/dispatch、暂存和输出root | 准备期不执行用户payload；发布中间值、旧结果、两slot和accepted失败/close；成本透明 |
| V3 | 普通 Direct/Graph 共用能力判断和原版region；默认冷规则与报告 | 用户只选Vulkan，无模型标签同样可用；完整region保留成熟融合，热期不搜索/计时 |
| V4 | 具名测试、真实consumer、指南、source/static/shared与最终P0/P1 | 每批复用相同接缝证据，不逐枚举重复全库或三交付；不造第二套构建机制 |

AUDITED前冻结公开形状、精确首批承载域、资源/错误/成本合同与有限 FIX；本表尚非实施授权。

## 预声明验证

- 全21行能力账本与主/正交 Direct/Graph 旅程；独立 scalar/bit oracle、特殊值、
  scalar/rank6/8、正stride、广播、tail/empty、非相邻归约轴、低方差归一化。
- 按实际新增域验证 dtype/layout/overflow/alias拒绝，动态index/scan失败不发布结果；
  两slot、变更payload、旧结果、部分构造/OOM/accepted失败/drain/close。
- Windows/Air 真GPU的 owning factor与直接消费者；适用host ASan/UBSan及WSL/Mini/OFF。
  host插桩不代签driver；不因未装可选validation layer伪造PASS。
- 稳定批次的一份组合consumer覆盖Windows/Air source/static/shared、prefix搬移、
  producer隐藏、原生资源/许可与DLL邻置。M1已有资源/交付事实只按实际未变输入复用。
- 运行change-impact selector选中门并保留skip原因；文档/状态只运行轻量门，
  closure明确previous-ref、docs和diff检查。候选后单次与实施分离的复核。
- 当前七门均未最终裁决；没有新增完整矩阵PASS。启动LOC为56,908行/249实现文件，
  使用统一脚本，排除tests/vendor；该数字只描述规模。
