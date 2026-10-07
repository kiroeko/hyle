# Compute + Graph CPU 最终收口

2026-10-07用户明确授权新的goal处理构建/规则/预编译、全面best-of/自研清退、三平台真实模型计算覆盖。
启动基线 `167cb5ff6f9edac8a9aaa77c41e6720bd3c377d2`，Windows工作树干净；后续提交与用户修改保留。
当前模块状态只读[registry](../../module-status.json)，旧goal已完成，不恢复旧百分比或施工队列。

## 目标与固定范围

- 用户只选CPU路线；同一普通Direct/Graph接口在Windows x86_64、Linux x86_64与MacBook Air ARM64有完整成熟计算承载。
- 标准`compute-cpu`在三个支持平台正常构建后必须可执行固定目标集合的全部必需CPU合同。
  无硬件特化只影响性能资格，不影响可用性；所需依赖缺失在configure明确失败，不允许AUTO静默减去必需算子。
  高级显式裁剪可以缩小能力，但不能冒充标准配置。Mac当前差距作为MUST补齐，不因可编译就签完成。
- 固定需求来自[104 GGUF及原版/非GGUF/分支覆盖](../../../user/cpu-model-coverage.md)与
  [逐制品联合合同](../../../../evidence/correctness/cpu-quant-mtp-c5-required/joint-verdict.md)：
  Qwen3.5-2B、Qwen3.8-27B/2.4T、Kimi K3、GLM5.3/Flash、DeepSeek V4 Pro-0813/Flash-0731及已发布MTP/DSpark。
- 原tensor角色、物理格式、精度、shape、模型族/制品身份与完整state/region一起决定需求，不能从文件Q标签推断。
- 全部实际可比重复域有当前版本裁决及条件化默认规则；只有一家合格承载的域不造性能竞争。
- 用户追加：先升级固定oneDNN版本，再与已升级ggml/其它合格成熟候选做F4；
  自研CPU在同合同落后且可完整替代时物理删除，优化挽留预算0，不仅降低优先级。
- 没有优势且有完整成熟替代的CPU自研算子物理清退；SIMD/Mem基础不删，具名独有动态/资源消费另行完整替代。
- source/static/shared规则、可用程序、runtime与软件身份一致；开启其它路线不能静默改变所承诺的CPU能力。
- 用户追加：未知硬件/工具链/线程没有已测特化时仍须合理默认，不以全局Provider固定顺序代替逐合同决策。
  优先级为已测优选→按op/原格式/精度/资源/融合条件的成熟通用默认→有需求的合法兜底/明确拒绝；
  通用默认复用成熟库原生ISA选择，具名说明依据与成本，不携带旧WIN/SOTA标签，不运行在线竞速。

范围是上述固定制品的text LLM/MoE计算及分支，不包含任意新release、视觉、模型loader/tokenizer/采样/服务引擎、
模型级CPU+GPU联合推理或算子大全。大模型验证允许分片驻留与较小token/context/expert驻留，
必须保留实际内宽、原编码、状态及全部必需分支；不冒称全部实权重整模型常驻推理。

## 六包与交付物

| 包 | 权重 | 施工卡 | 完成条件 |
|---|---:|---|---|
| F1 构建与预编译合同 | 10% | [F1](cpu-final-f1-build-contract.md) | CPU配置/依赖/规则软件与优化身份、安全ISA、混合SDK能力、三交付投影有明确实现与验收 |
| F2 需求与重叠域归并 | 10% | [F2](cpu-final-f2-demand-overlap.md) | 固定需求逐端口/region→候选→平台→验证；列出真正重复域、唯一承载和所有缺口 |
| F3 三平台成熟承载 | 25% | [F3](cpu-final-f3-platform-carriers.md) | Mac CPU及其它实际缺口以成熟库接齐，普通两入口与资源/三交付有本包资格 |
| F4 批量best-of | 25% | [F4](cpu-final-f4-best-of.md) | 当前合格候选同合同批量裁决，固定特化规则与逐合同合理默认/未知身份回退，不留未知冒充最优 |
| F5 自研清退 | 20% | [F5](cpu-final-f5-retirement.md) | 逐源码族去留可追溯，落后重复实现删除，唯一需求有明确完成路线 |
| F6 完整旅程与终审 | 10% | [F6](cpu-final-f6-final-qualification.md) | 三平台两入口三交付、完整计算图/状态/失败、文档/LOC/清理全部适用验收完成 |

权重用于新goal估算，不把doc、旧CLOSED、单独probe计为实现完成。每包实际实现/验证/未完成/阻塞分别报告。
包按顺序冻结合同；F1中依赖F2/F3的最终CPU能力实施明确交接，不用一条有限flags修复冒充整个F1完成。
F6只做汇总与差异补验，不能首次接线或首次给前三平台签资格。

## 施工与性能边界

每包开始绑定真实owner，合法SCOPED→AUDITED固定有限FIX，再改产品；只读需求/证据盘点不伪造产品reopen。
`graph`的旧Vulkan BLOCKED及GPU owner旧状态不自动解除；CPU实现只改CPU route/provider，公共机制需具体反例再定界。
旧187规则与当前17格只相对当时默认peer，不自动当全Provider最优，也不移植到未测硬件/软件。

先批量接线，稳定候选后集中构建验证；复用producer/锁定cache与同身份证据，j2/test j1，重型串行。
F4从真实合同去重，固定normal/holdout及串行/常用并发档位；必要转换、布局、同步、线程、融合损失、准备与重放成本完整记录。
每批只一轮preflight/适用P1与至多一批具名接线纠正；不优化自研、不无限n升级/换顺序重测。
噪声/同实现对照不合格不记LOSS/PASS；稳定loss如实记，未裁决有实际阻塞/用户取舍，不猜best-of。
只有一个合格Provider完成正确性/资源/P0即可。性能资格按已测硬件和工具链，未知身份成熟合法基线且不声称最优。

三平台指Windows、Linux、macOS；WSL是Linux验证环境而非独立物理性能机。
Mac先读取真实硬件/工具链，macOS CPU不以Metal执行替代。远端只前向同步不可变Windows输入。
不自行停止用户程序、改电源或弹出交互窗口；Windows DLL邻exe，后台工具hidden。
按change-impact selector/七门执行适用验证，docs/state不build/P1，不改公共治理或统计核心。
按`[skip ci]`提交推送；临时产物有owner/路径/活动进程校验，归档校验后及时清理。

## 停止

所有固定需求与适用验收通过，明确通知“Compute + Graph CPU 最终收口完成”，列限制并停止。
真实缺成熟承载、需要新数学、无法保持SDK ABI/既有GPU能力或性能无法公平裁决时，说明具体问题与有限取舍，
不能自行删范围后签完整，也不能无限研究。每包以当前卡状态裁决，goal工具状态不替代registry。

启动只激活F1审计；基线实现Compute+Graph **61,073行/249文件**（统一脚本、不含测试/vendor/bench）。
