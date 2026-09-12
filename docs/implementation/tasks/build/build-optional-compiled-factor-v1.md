# Build/package：可选 compiled factor 的缺席投影

<!-- PROJECT UNIT REVIEW: AUDITED -->

> Build/package 是横向开发单元，不进入产品模块 registry。本任务由
> `compute_provider_onednn` / `graph_provider_onednn` 的 feature-OFF 交付触发；
> 不改变 provider API、算法、性能路线或模块状态。

## Closure card

- baseline：`bcf25f2c6c813483743dd0f4a4c5279531d0576c`；
- trigger：选择 `graph` product 且 `HYLE_COMPUTE_ONEDNN=OFF` 时，已有
  `compute_provider_onednn` 的中央 private 依赖与空 INTERFACE target 冲突，fresh configure
  被 `hyle_add_module()` 拒绝。强制下载 oneDNN 或制造空 DLL 都会改变 OFF 语义；
- normal consumer：x64/ARM64 source、static 或 shared 用户选择 Compute/Graph，但保持可选 oneDNN
  明确关闭；SDK 仍发布 provider header 的条件接口和正确 product/factor metadata；
- scope：为 `hyle_add_module()` 增加显式 `OPTIONAL_ABSENT` 形状；缺席 target 只投影中央 PUBLIC
  依赖，忽略只在实现存在时才有意义的 PRIVATE/PRIVATE_BUILD 依赖；最终 graph validator 仍验证
  public 边精确且不允许额外真实边；两个 oneDNN factor 的 OFF 分支采用该形状；
- non-goals：条件化中央 component graph、增加第二套依赖清单、制造 stub 源/DLL、改变 oneDNN
  默认值、下载/lock/package owner、修改 provider 产品代码或一般 header-only 语义；
- cost：configure-time 常数分支；不增加运行时 target、符号、文件或元数据；
- finite FIX：`BUILD-OPTIONAL-01` 新增显式参数与非法组合诊断；
  `BUILD-OPTIONAL-02` 精确验证缺席 target 的 public 投影；`BUILD-OPTIONAL-03` 迁移两个 OFF
  分支；`BUILD-OPTIONAL-04` fresh OFF/ON 与 source/static/shared 回归；
- stop：若缺席形状需要隐藏真实 public 依赖、让未选 factor 被构建，或使 installed product
  引用不存在 target，则 `BLOCKED`；否则四项 FIX 通过即 `CLOSED`。

## Candidate

- 尚未实现。

## Verification 与裁决

- 待实现后填写；当前终态为 `AUDITED`。
