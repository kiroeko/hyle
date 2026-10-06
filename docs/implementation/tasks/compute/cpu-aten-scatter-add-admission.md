# CPU 成熟接线：ScatterAdd 索引域

<!-- MODULE REVIEW: CLOSED -->

- owner `compute_provider_native`；baseline `214cb3d48cb39416e8540f7188249105cde58b10`，Windows canonical。
- trigger：[资源域审计](../../../../evidence/correctness/cpu-resource-domain-audit/README.md)实证：
  I64 ScatterAdd 三种浮点 Native 准入、成熟 factory 拒绝；原同版 ATen `index_add_` 的156个断言
  已通过。SI明确未纳入 Add，不能把 adapter 拒绝当成熟库不存在。
- 主消费：模型 MoE expert 输出按 token index 加回、状态 indexed accumulation；正交消费：
  public 各轴/多维和 scalar indices、odd byte stride、重复/empty/非有限、alias与checked失败。
- scope：原已公开 plain F32/F16/BF16 ScatterAdd、I32/I64、rank<=8、dense inferred output；
  ATen原 typed numeric packing + 原 `index_add_`，不新增数学kernel/公开语义/factor/SDK。
  Assign/其它已关闭数学只复用生命周期；Native/SIMD优化、比较、清退和其它路线保持冻结。
- 语义：Add公共合同允许storage或更宽累加/重排，不要求逐更新storage舍入；先validate所有索引再写
  output。多维 indices 按逻辑flatten，scalar用unsqueeze。原 typed kernel不能把Byte carrier当数学。
- 生命周期：复用 private typed lanes/所有输入先packing/checked host maps/真实Completion/异常/OOM/
  flush-close传播；每 invocation 私有状态。only equal-dense input0 alias；其余拒绝。managed replay、
  私有native线程/临时Tensor成本公开，native invocation bytes unknown，不签prepared/capture/零热分配。
- 普通入口与规则：已有 CPU/Graph 默认 assembler 自动匹配 ATen350，ggml200保留合格I32域；
  用户不注册、排序或timing。只有新增同合同I64/布局域走ATen；不得以self后备算接线完成。
- 依赖/消费者：已有Torch2.8 CPU SDK与Native factor、Compute CPU/Graph CPU；无公共API/package seam
  变化。Windows/Linux x86_64现有Torch资格；不伪造macOS SDK资格。
- tier P0：O(index count + base + updates)，包括typed packing、I32/I64转换、原index-add/temporary、
  output cast和copy。不跑自研P1，不引入stable sort给Add，不删除仍有效whole-region。
- 预算/停止：一次原库诊断（已完成）；固定AS1–3后批量产品/test/普通consumer；一次两平台受影响
  factor/consumer与三交付，最多一次具体产品纠正。具名失败先查源码，不扩建仪器/无界重测。
- 预声明验证：三dtype/I32-I64/任意axis/scalar-multidim/odd strides/duplicates/empty/NaN-Inf；
  exact input0 alias、independent lanes/replay/close、late越界不写输出与重建；代表cold HYLE OOM回滚。
  同普通Direct/Graph变化输入与下游结果、source/static/shared Win/Linux；公开头沿既有边界检查，
  selector/closure/docs/diff与统一LOC。native物理OOM和noncoherent故障只审计传播，不虚构注入PASS。

## 有限 FIX（审计固定）

| ID | 允许接缝 | 必需验收 |
|---|---|---|
| AS1 | ATen Add准入/typed packing/原index_add，复用checked索引和flatten公共shape | 数值、storage舍入/重复、布局/empty/indices宽度 |
| AS2 | private lifecycle与equal-dense input0 alias，checked失败终态 | 变化输入、独立invocation、越界无写/恢复、close/OOM/资源平衡 |
| AS3 | 已有普通默认候选与Direct/Graph普通三交付 | mature身份、最终结果、完整managed成本/限制文档 |

原库156断言与公共合同/既有生命周期源码已核对。AS1–3固定BORROW/FIX，owner进入AUDITED；
准入仅三floating数值，索引I32/I64。完整public rank/byte stride由已有packing承载，输出保持dense。
Add使用F32 base/update与原index_add之后一次输出storage舍入，符合公共允许的更宽累加；没有排序、
逐更新HYLE数学或alias扩大。empty indices直接原Byte copy，避免无计算时破坏base特殊值存储bits。
现有legacy Gather/Assign index scratch不为Add重复分配；Add已有typed index lane足够，所有检查和
原kernel/metadata成本仍归managed。本卡不代签全CPU goal或性能WIN。

## 不可变候选与已完成验收

产品candidate `e13edaebe0aac974fd2f7d2ff50246dfd61ad330`，fixture-only `7356a06f`
显式声明测试I64→I32 narrowing，修复`/WX`；产品/API/成本未改。原compile failure保留。
Windows Provider140/140、CPU12通过/1旧fixture skip、Graph33通过/6旧fixture skip、header1/1，
相同ordinary Direct/Graph的I64 duplicate多维/非末轴scalar/odd-byte与empty、变化输入及close实际通过。
Linux与两平台installed/static仍按本卡验收，当前未运行不签PASS；无Native/SIMD比较或新WIN。
1337个固定前向产品/test/构建/consumer输入绑定候选，副本不作为产品改动来源。

## 终审：AS1–3 CLOSED

产品candidate固定后未改；所有预声明适用门已完成，之前CANDIDATE段的待运行项现以
[完整资格收据](../../../../evidence/correctness/cpu-aten-scatter-add/README.md)为准。
两平台各Provider140/140、CPU12通过/1skip、Graph33通过/6skip、header1/1；same ordinary Direct/
Graph source shared/static与installed shared/static全部实际运行通过。七个旧fixture skip不计PASS。
三storage/I32-I64/各轴/多维-scalar/rank8/odd stride/duplicate/empty/special值与alias实测；
late越界无写与重建、并发两个私有lane、close、逐点cold HYLE OOM live-byte回滚及Stats平衡通过。
native物理OOM/noncoherent故障只审计传播；不虚构注入、capture、zero-allocation或整模型推理资格。

| 七门 | Verdict与边界 |
|---|---|
| 第三方路径 | PASS：原已锁Torch2.8 CPU与156 SDK断言；无新SDK/自研数学/许可证变化 |
| 功能完整 | PASS，AS精确域：已有public Add、三floating/I32-I64/布局/duplicate/empty及最终storage结果 |
| 内部复用 | PASS：原typed/index-add、现有packing/mappings/Completion/alias/异常与cold allocator |
| 性能 | PASS P0：F32 packing/clone/index-add/cast、原Tensor/线程/临时存储完整计入managed；无自研P1/WIN |
| 交付边界 | PASS：无公开API/新factor/FFI/中央依赖改动；不扩大Assign或其它原库数学 |
| 构建交付 | PASS：Win/Linux同consumer source/static/shared，header与1337前向hash；关闭分支仅复用未变接缝证据 |
| 易用文档 | PASS：CPU路线自动匹配，普通Direct/Graph下游/变化输入/close；资源/舍入/SDK/alias限制公开 |

统一Compute+Graph实现 **61,516行 / 249文件**，不含tests/vendor；本批四个consumer目录清理
**625,241,686 logical bytes**，保留SDK/producer/cache。所有失败、源码/consumer、原输出/XML、
driver、身份/selector/LOC/cleanup存档，不把旧CLOSED、probe或文档算新架构完成。

重开限AS具名数值/资源/失败/交付反例、已锁依赖升级或真实新公共消费域；不因“再看一遍”重开。
CPU goal仍需精确资源声明与冻结自研的有限域裁决，SIMD基础保留、其它路线暂停，本卡不关闭该goal。
