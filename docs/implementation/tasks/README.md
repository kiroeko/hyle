# 任务记录

这里保存当前施工卡和当前 terminal record。不要按目录中“最新文件”猜状态：模块唯一入口是
[`module-status.json`](../module-status.json) 及其生成的[状态页](../status.md)，其中每个非 `QUEUED`
组件都直接链接当前 record。

当前治理 schema 的迁移事实：

- [治理执行性与文档压缩 v4](project/governance-documentation-v4.md)。

当前跨产品能力审计事实：

- [CLOSED 组件全库正确性资格审计](project/closed-components-correctness-qualification-2026-09.md)：
  当前 29 个已关闭 factor 的逐能力语义证据、coverage discovery、跨模块 public journey 与平台/
  sanitizer/交付资格；不创建第二套模块状态；
- [HYLE 能力族深度复审](project/hyle-capability-family-deep-audit-2026-09.md)：当前 27 个已实现 factor 的
  能力族、已交付头反向追溯、四纵切与缺口/路由结论；
- [HYLE 范围重定标与全组件能力扫描](project/hyle-scope-capability-audit-2026-09.md)：此前 factor 粒度的
  边界扫描；当前结论以能力族复审为准；
- [Public C++ 消费者纵切组合审计](project/public-api-consumer-portfolio-audit-2026-09.md)：此前 F01–F08
  纵切事实；已完成项和当前边界以新扫描为准。

当前公共工具修复事实：

- [Provider 批次治理 v1](project/provider-batch-governance-v1.md)：为 M2 B2--B6 固定 Provider 能力
  超集、稳定路线 profile、Git 增量同步、证据复用与单一终局收据；不重写 selector/closure/P1 核心；
- [Native route 平台适用性](build/validation-native-route-platforms-v1.md)：Apple/CUDA 的真实 ON 测试与跨平台 OFF 边界；
- [Objective-C++ 实现 LOC 计入](build/loc-objective-cpp-accounting-v1.md)：修正 `.mm` 正式实现的语言范围遗漏；
- [GLSL 实现 LOC 计入](build/loc-glsl-accounting-v1.md)：修正正式 compute shader 的语言范围遗漏；
- [CUDA compiled factor 验证上下文](project/validation-selector-cuda-factor-context-2026-09.md)：
  显式验证或反向选中 CUDA factor 时同步启用其真实 feature，不把空 target 当成有效测试；
- [Structured pair placement validator 修复](project/performance-structured-placement-validator-2026-09.md)；
- [性能计时前完成边界与执行拓扑预审](project/performance-preflight-topology-2026-09.md)；
- [真实 workload paired interleaving](project/performance-workload-interleaving-2026-09.md)；
- [AppleClang 本地常量池标签归一化](project/simd-codegen-local-pool-normalization-2026-09.md)：
  codegen 比较忽略函数局部 `LCPI` 序号，但继续核对 relocation 类型、槽号与外部边。

当前构建/交付审查：

- [exact-factor 测试依赖闭包](project/build-factor-test-closure-2026-09.md)：产品依赖保持纯净，测试开启时
  从中央组件声明取得 factor 测试所需的夹具闭包；
- [共享库导出边界 v13](project/build-system-v13-shared-export-hygiene.md)：清除意外内部 C/汇编导出，
  并把显式 C++ export-only 变为 shared artifact 机械合同。
- [可选 compiled factor 的缺席投影](build/build-optional-compiled-factor-v1.md)：让 OFF 状态保留组件身份与
  公开闭包，但不制造空 DLL、私有实现边或依赖下载。
- [可选 compiled factor 的验证上下文](build/validation-selector-v3-optional-factor-context.md)：显式验证
  可选 factor 时激活其真实 feature，不把缺席测试误写成 PASS。
- [CUDA system SDK 依赖投影](build/build-cuda-system-sdk-projection-v1.md)：让 CUDA 专用公开头与 target 在
  source/static/shared 三种交付中由同一 system-SDK `find_dependency` 合同承载。

Compute/Graph 研究与追溯：

- [Apple MLX/Metal 当前收口](compute/apple-cpu-cuda-parity-closure.md)：沿共同模型目录核对接线、成熟承载、默认优选、冗余自研与当前 Air 三交付。
- [CUDA 最终收口 goal](compute/cuda-final-closure-plan.md)：继承 CPU 已固定的新模型／混合量化／MTP 共同需求，
  六包处理 CUDA 接线、构建交付、条件规则与冗余自研；[U1 当前只读审计](compute/cuda-final-u1-demand-wiring.md)。
  这份协调计划不替代 registry 的产品 owner／状态。
- [LLM/MoE 成熟 Provider 完成计划](compute/llm-moe-coverage-completion-plan.md)：以七个大模型与
  本地 Qwen3.5-2B 确定需求、Provider 和接入方式，按 M0–M6 成批接四路线；
  [M0 当前卡](compute/llm-moe-m0-provider-scope.md)；旧 C1 队列撤回，
  [所有自实现冻结至最后速裁](compute/llm-moe-self-operator-freeze.md)。
- [四路线基础重构执行计划](compute/four-route-foundation-refactor.md)：七包已完成；
  [验收索引](compute/four-route-foundation-acceptance.md)列明实际证据与资格限制，产品状态仍以 registry 为准。
- [六项架构复核](compute/six-point-architecture-review.md)：本轮路线隔离、Apple/多 Provider 接入、
  渐进路由、有限实验与文档清理的已完成研究；不自动恢复旧 goal。
- [历史决策与证据索引](compute/compute-graph-route-unification-audit.md)：不再保存第二份百分比/执行队列。
- 前轮基础施工记录（其中部分已被后继卡替代，不能按本列表启动）：
  [ComputeOps identity](compute/compute-ops-v2-stable-custom-identity.md)、
  [Compute proposal](compute/compute-v8-route-proposal-contract.md)、
  [Graph foundation](compute/graph-v9-foundation-routing.md)、
  [Native route](compute/compute-provider-native-v2-foundation-route.md)、
  [oneDNN route](compute/compute-provider-onednn-v5-foundation-route.md) 与
  [oneDNN node-set region](compute/graph-provider-onednn-v4-node-set-foundation.md)。
  当前 owner、阶段与有效 record 只按 registry，不按历史导航推断。

当前全库性能工作：

- [全库性能定位与差距消除](project/full-library-performance-positioning-2026-09.md)：区分方向、实用
  materiality 与产品行动，清点全部 CLOSED factor 的去重性能事实，并只定向处理真实固定劣势。

任务结束后只保留当前状态链仍引用、或被长期 architecture/developer/user 文档引用的记录。被替代的
完整 Jira 式历史由 Git 保存，不用导航页和兼容跳转页维持第二套当前事实：

```text
git log --all --name-status -- docs/implementation/tasks/<area>
git show <revision>:docs/implementation/tasks/<area>/<task>.md
```

新记录从[模块任务模板](../../developer/review/module-review-template.md)开始。横向 build/tool/docs 工作
使用 `project/` 任务；它们不伪装成拥有稳定 candidate 的产品 component。
