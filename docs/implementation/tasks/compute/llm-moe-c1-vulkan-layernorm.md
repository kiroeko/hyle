# C1-Y33：Vulkan LayerNorm 成熟实现接入

## 当前合同与审计

- owner：`/root`；baseline `5f61546cd0f09ff735b2c01f3f28e33fa3044a2c`。
- 当前 owner：`compute_provider_ggml_vulkan`、`compute_route_vulkan`、
  `graph_route_vulkan`；三者现有 C1 AUDITED 重新定界为下面有限 FIX。
- 需求：[C1](llm-moe-c1-tensor-state.md) 的 F04.2，最后一轴 LayerNorm、F32 累加、
  可选 scale/bias、F32/F16/BF16、zero/one/tails、普通 Direct/Graph 与状态/结果寿命。
  本地2B张量可作为输入；LayerNorm 组合是已冻结的正交基础库消费，不冒称 Qwen 原始 forward。
- 触发事实：当前 `region_supports` 只接 RMSNorm，LayerNorm 仍走 HYLE shader；
  [Provider 复核](llm-moe-c1-provider-reuse-audit.md) 已登记此接入余项。整包性能候选尚未稳定，
  本节先完成成熟接入及数值资格，不提前启动 P1。
- 来源：干净 ggml `456172ec733a135778adcd32d00e576a58232e45` 的 `ggml_group_norm`、
  Vulkan 分派和原版 `group_norm.comp`；HYLE 既有 schema、Region/Plan、Cast/Copy/alias。
  [源码身份](../../../../evidence/correctness/compute-llm-moe-c1/ggml-vulkan-layernorm/source-audit.json)
  仅证明审计输入，不是 GPU 验收。

原版 `norm.comp` 的 E[x²]−E[x]² 不满足已要求的大均值小方差资格。原版 GroupNorm 已有先均值、
再 centered variance 的两遍算法；把连续每行映射成 `[width,1,1,rows]`、group=1 即为逐行
LayerNorm，保持 F32 数学及原厂 shader。其固定512线程、X维每行一个工作组、descriptor 对齐
与 metadata/输出 owner 约束必须在接入层处理。未看到结果前不声明性能优势。

## SCOPED / AUDITED：有限 FIX

| ID | 固定改动与边界 |
|---|---|
| Y33-A | Provider 的 F32 dense region 增加 LayerNorm。冷期 reshape → 原版 GroupNorm → 恢复 shape → 可选原版 Mul/Add；无 affine 时用原版 Cont 建立实际输出 owner，不能把 externally bound 输出直接设为内部 view。每个 LayerNorm 预留至多5个 tensor/node 的 arena 容量。pin、shader、vendor patch 和原生运行机制不变。 |
| Y33-B | route 准备期匹配512线程、行数工作组和 F32 暂存容量；保留设备/容量不适用时的已有合法 native 路径。非连续、rank>4、F16/BF16 用既有 Cast/Copy 与无复制 dense alias 接成 F32 二维 region；scale/bias 分别转换，affine 后仅一次 RNE 写回 half。空域仍无 GPU 工作；不新增归一化算法或公共 API。 |
| Y33-C | GroupNorm 的输入必须 descriptor 对齐。沿现有 region 冷适配把动态/偏移常量输入变成已规划的对齐存储；同时覆盖普通 Direct、Graph、Recorder、带前后缀的组合与混合参数。已有 region/native Run、结果槽、失败和 quota owner 继续共用，不新增运行时选择器。 |
| Y33-D | 先用独立两遍 FP64/明确 half RNE 对真实成熟路径作资格回归；新增零方差/epsilon0、NaN/Inf、大均值小方差、宽1/127/129/257/2048、rank8、odd byte stride/origin、scale/bias 类型组合、变输入/双槽/旧结果及冷限制测试。测试必须证明 managed 路径被执行，不能只证明 fallback 数值正确。 |

本节只接 F04.2；归约/Softmax 和其余已登记 Provider 接入余项仍待各自有限审计。现有自实现
保留为必要未覆盖域和同合同比较候选；删除必须等实际覆盖及完整成本裁决，不凭本节提前淘汰。
不改公共构建系统、selector、性能统计核心、CPU 资格 blocker 或其他模块。

## 成本与适用验证

P0：F32 dense region 使用原版两遍中心化核，affine 为至多两个原版逐元素节点；无 affine 时
输出拥有独立端口的 Cont 成本显式保留。half/非连续适配计入必要的格式转换、连续化、暂存及
最终舍入；rank flatten 本身只有冷 metadata。512线程/行数/容量在准备期固定，执行不搜索、
计时或更换 Provider。region 的原厂编码/私有分配与 HYLE 边界同步仍按既有真实机制计费。

- 首先执行 W1 的新增资格、provider/Direct/Graph 适用回归；同候选 W1 static/shared、Air/Mini
  对应 factors 和 source contract。使用已有短路径/cache，build2/CTest1，重型步骤串行。
- Windows/Air 的 source/static/shared 普通消费加入本实际 LayerNorm 片段；覆盖不同参数、
  batch/尾宽、重绑定、输出保留及关闭。按 selector 执行所选门，未变且身份匹配的证据复用；
  Y32 已完成的无关工程门不因文档或本私有适配原样重跑。
- Windows DLL 邻 exe、PATH 隔离、禁错误弹窗；Air validation，Mini 只签 correctness/portability。
- 不运行 P1；API/算法与全部 C1 成熟接入固定后执行原 C1 bundle，n15/上限31、5%门及等责原则不变。
- 此具名接入来源调查到此停止；产品纠正最多两轮。首次失败完整保留；预算到点按反例裁决，
  不改门限、切换成自写归一化或扩泛算子调查。

当前裁决：审计完成，以上 FIX 已固定，产品尚未实施，适用验收未运行。C1 整包与共同矩阵
未签收；C1 全部适用验收完成后通知用户，在 C2 前等待提问。

## 首个实现候选

审计提交`5f94bed4`之后，Y33-A–D已落代码：原版GroupNorm及输出owner恢复、F32 region、
half/stride/rank8适配、冷设备/暂存限制与descriptor对齐；新增真实managed路径、偏移常量和
独立参考回归，更新混合图的实际F32暂存账单。产品/测试首次构建和GPU验证尚未运行，
不能签数值资格、性能或三交付。后续先执行W1三个受影响factor，再沿本卡完成其余适用门。
