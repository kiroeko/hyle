# G04/G05 有限能力与资源取舍终审

状态：**SCOPED，2026-10-16**。用户明确选择先收口本批，并将完整 Qwen3.5-2B CUDA
命令行消费者与 llama.cpp 的 prefill/decode tokens/s 作为后续引擎验收。旧 goal 已清除；
本轮不恢复 goal，不开始 G07–G10 或引擎实现。

## 范围与裁决合同

- baseline：`68565dfc94f39ab610d1a2c8ad55181b26c6a8fc`。
- 完整产品：`669121418669b31f6849c74f64362aaf8d6eb919`；本轮产品 FIX 集为空。
- 主消费者：既有编码 append→原 F2 页更新→驻留页→Direct/Graph/native attention→
  checked completion→结果/关闭；正交消费者保留 window、noncausal、Dqk≠Dv、多请求、
  shared/tail/旧 reader、数值溢出与消去反例。
- 选定 CUDA 编码路线：FP8 E4M3FN 驻留、tile 内转换、F32 partial/merge、最终缩放和
  输出窄化；不改为完整 BF16 常驻副本，不采用已否决的 C1 转换。
- CPU/Apple/Vulkan 的原 Q8_0 有限域、普通 F16/BF16 路线与原资格保持。模型是否选
  FP8 KV 属于引擎配置；首个 CUDA 引擎对照优先匹配双方普通 BF16 配置。
- 不改 API、算法、布局、依赖、构建系统、公共性能仪器或原 P1 contract/receipt。

本轮 owner 为仍在本批的五个组件。已 CLOSED 的本批其他 owner 复用各自 terminal，
Apple 与 Torch CUDA 当前 G07 SCOPED 保持，不为聚合收口重开或代签 G07。

| owner | 冻结产品 candidate |
|---|---|
| compute_ops | `982676cb05a6e6244e66e0a596347aa96043617b` |
| compute_backend_cuda | `982676cb05a6e6244e66e0a596347aa96043617b` |
| graph_route_vulkan | `982676cb05a6e6244e66e0a596347aa96043617b` |
| compute_ggml_runtime | `42a4c7cdcc0822a7b162efe575909622a906de1d` |
| compute_provider_flashinfer | `0a29d119426b2975d48c238c37a2994675e0ba9f` |

## 工程性能与旧门的关系

这是一轮经用户选择的**验收范围重定界**，不是原 5% 门通过。依据是原 comparator
具有额外完整 BF16 驻留和 BF16 partial，与选定压缩/数值合同不同；按
[工程章程](../../../developer/standards/engineering-charter.md)及
[收口方法](../../../developer/review/module-closure-methodology.md)，这种比较只能说明
自然方案取舍，不能签同约束实现非劣或据此追改产品。

本轮预声明工程性能 **P0**：检查已冻结的压缩驻留、tile 解码、F32 中间结果、dirty
更新与既有生命周期成本，并复用未变产品的已运行正确性、设备/宿主风险和交付证据。
CPU/Apple/Vulkan 原合格 P1 与方向继续保留；CUDA 原 attempt14 仍为 QUALIFIED、
prefill LOSS、5% UNRESOLVED，不回填 PASS、不提高阈值、不增加样本。其结果继续作为
披露资源/精度差异的不可变历史 warm 方案对照。

本轮不证明“算法总体更快”，不证明 HYLE 比 llama.cpp 慢或快。原 comparator 是本仓库
构造的 FlashInfer BF16 peer，**不是 llama.cpp 引擎**。F32 累加也不是 HYLE 独有优势。
已知 CUDA warm 场景约 4.0–4.9% 耗时劣势保留，完整引擎性能尚未验证。

## 有限检查与停止条件

只做一批只读终审：核五 owner 当前产品身份、既有普通门/后续 G06 证据的输入与覆盖、
源码成本结构、公共用法、独立阶段红队和统一 LOC。普通验证仅 docs、closure、diff；
registry 每次合法转移用明确 previous revision。复用匹配输入，不编译、不重跑
sanitizer/SDK/P1，不运行模型。预算为一个约 45 分钟的终审块；到点仍有真实未运行
适用门则明确记录并裁决，不把适用风险改成 NOT_APPLICABLE。

FI 按 BLOCKED→SCOPED→AUDITED→CANDIDATE→CLOSED 恢复；其他四 owner 从既有
CANDIDATE 终审。阶段提交只维护文档/registry，不产生新产品 candidate。

## 七门

| Gate | 当前裁决 | 终审依据 |
|---|---|---|
| scope | 待终审 | 本卡有限旅程；原 CE01–07/后续有限修正不扩域 |
| architecture | 待终审 | 原 F1/F2、唯一 native owner 与 actual retirement |
| correctness | 待终审 | 原独立字节/数学、OOM/并发/退休与适用风险；G06 后续同输入证据 |
| delivery | 待终审 | 原 source/static/shared 与后续四路线迁址 SDK |
| usability_docs | 待终审 | batch 指南及编码/精度/资源/限制 |
| red_team | 待终审 | 与实现分离的一遍自审；不冒充第二审查人 |
| engineering_performance | 待终审 | 上述 P0 结构合同；旧 P1 结果/方向原样保留 |

真实引擎目标与精度策略写入
[Qwen 命令行消费者范围](../project/qwen35-2b-command-consumer-scope-2026-10.md)。
全链消费者能暴露新的实际缺陷，届时带最小反例合法 reopen；不能以尚未开发的引擎
或“可能再快一点”为由继续无限优化本批。
