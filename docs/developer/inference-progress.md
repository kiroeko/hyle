# 基础推理、多会话与多任务进度

更新：2026-10-25 07:03（北京时间）。整体暂估 **18%**，当前阶段：无多余采样的分块 prefill，以及 NF4 完整激活路径。
百分比只表示本轮工作完成度，不是现有引擎的功能覆盖率；盘点后按实际缺项调整。
本轮基线为 `c950466ec`，已提交并推送；上一轮 MTP 范围与结果保留在[工程说明](../implementation/inference-roadmap.md)及所属证据。
最近完成批次已推送：`29a746911`（开放续轮／等待窗口／交付游标、DeepSeek 短历史、NF4 原值组件）。

范围：单机、多系统，CPU／单 GPU／CPU＋GPU；基础普通推理、多会话、多请求／多任务。
实现重心在 Compute／Graph、状态与缓存、内存／资源寿命、调度／I/O 和共用推理原语。
真实模型入口只作验证消费者，按需要修正接线；不扩服务器、聊天应用或通用产品管理。
功能不重开 MTP／多模态穷尽；性能明确包括 MTP 对 llama.cpp 约 10% 的差距，允许按根因调整 MTP 执行与接线，保持数值、接受、完整状态及退休合同。
运行中接纳、输出背压和取消是请求层能力，不以开发 HTTP 服务作为前置。
模型范围沿用[模型与算子清单](../implementation/model-operators.md)；完整模型无法运行时继续检查本机可做的原组件。
功能完成标准是当前硬件能实现、能检查的相关底层功能全部完成；新增缺项继续纳入。
首批清单、代表样例或盘点完成都不作为功能穷尽完成。本轮性能查明普通／MTP 落后原因，寻找反超机会和更好的基础设计，CUDA 重点参考 TensorRT-LLM；不以只整理旧对照作为完成。

- 功能盘点与补齐：约 25%
  - 基础普通推理：约 30%
    - 模型／量化／算子与融合缺项盘点：约 10%
    - BNB NF4 普通角色与计算接线：约 25%
    - MLX affine 普通投影／embedding／head：0%
    - CT asymmetric INT4 原投影角色与零点接线：100%
    - 解码／激活／计算／缓存精度合同：约 5%
    - DeepSeek 原 target 空／短历史与 SWA 阶段：100%
    - GLM Pro 原组件短历史与真实 index 供给：100%
    - Kimi 普通异构状态统一提交与恢复：0%
    - Kimi 混合量化专家工作集组合：0%
    - prefill／decode／上下文与状态：约 5%
    - 采样／停止／预算与错误边界：约 5%
    - 验证消费者输入边界与仅 pending 续轮：100%
  - 多会话：约 3%
    - 完整状态／前缀复用／隔离与恢复：约 5%
    - 生成后前缀命中的 Text R2 suffix 接纳：100%
    - 容量／淘汰／驻留与实际退休：约 5%
    - 已确认缺项的实现与有限检查：0%
  - 多请求与多任务：约 32%
    - 有界槽／世代／FIFO 与等待／唤醒原语：100%
    - 运行中有界接纳与增量续轮：约 65%
    - 请求级取消与保留同批未提交成员：约 80%
    - 请求局部失败与设备失败区分：约 25%
    - 输出消费／背压与完成观察：约 60%
    - prefill／decode 调度量与公平性：约 5%
    - 多 Model 的资源寿命与设备接纳：约 5%
- 架构复审、设计与调整：约 12%
  - 全链职责、所有权与配置合同复审：约 5%
  - 按真实缺项调整共同机制：约 15%
  - 新模型／新后端／新请求政策扩展检查：0%
- 性能设施复用与参考引擎狩猎：约 5%
  - Hyle 设施实际消费与成本盘点：约 5%
    - 内存／容器／调度与 I/O：约 5%
    - 冷图／融合／重放与缓存：约 5%
    - 普通输出与整数页表暂存复用：约 10%
  - 本地参考引擎调研与候选取舍：约 10%
    - llama.cpp／ik_llama.cpp：约 15%
    - vLLM／SGLang：约 15%
    - TensorRT-LLM／Strata：约 10%
    - ds4／DeepInfer／deepseek-recipe：约 15%
  - 已确认热点实现与完整消费者对照：0%
  - MTP 对 llama.cpp 差距归因与执行优化：约 8%
  - prefill／decode／内存显存／多轮多请求综合对照：0%
- 完成前遗漏复审与记录收尾：0%

## 正在做

| 任务 | 所属 | 简述 | 进度 |
|---|---|---|---|
| NF4 完整激活／导入设计 | 功能／基础推理 | 按原 F16 policy 收敛单一精度配置和共同冷导入，已缓存原组件可继续用，不下载整模型 | 约 10% |
| 有界 prefill 与 token 调度量 | 功能／架构 | state-only 图族、ledger 和输入游标已接；正在原 P154／Q64 CUDA 消费者检查，尚未通过 | 约 55% |
| Kimi 普通异构状态盘点 | 功能／架构 | 查原 KDA／MLA 状态组合与角色缓存，为同一 StateHead 的提交／恢复设计最小切口 | 约 10% |

先形成可落实的缺项清单，再补功能、调整对应架构、验证性能候选。
不为统一接口新增热分配、搬运、动态路由或第二套执行器；不强制替换本来合适的容器。
只运行改动相关的必要检查，复用工具链和依赖缓存，不重复全库构建。

本轮已有依据（组件通过不代表整模型、在线请求全链或跨平台通过）：

| 项目 | 已测结果／边界 |
|---|---|
| [普通 R2 前缀后缀接纳](../evidence/correctness/ordinary-text-r2-prefix-suffix-2026-10-25.json) | 原失败已修；真实 R2/T3、四请求退休及缓存预算通过 |
| [输入与有界窗口](../evidence/correctness/ordinary-input-and-request-window-2026-10-25.json) | pending-only 两轮、四请求 R2 CUDA 通过；窗口 224 次占位／672 turn、热零分配 |
| [运行中追加轮次原语](../evidence/correctness/request-incremental-turns-2026-10-25.json) | host 8 组通过；包括等待输入、pending 一次消费、实际提前停止额度复用；实际模型结果另见生命周期用例 |
| [等待／唤醒窗口](../evidence/correctness/request-window-suspension-2026-10-25.json)／[输出交付游标](../evidence/correctness/output-delivery-credit-2026-10-25.json) | host 检查通过；同一 FIFO、暂停仍占 payload、成功前缀确认幂等 |
| [Qwen 请求生命周期](../evidence/correctness/qwen-live-request-lifecycle-2026-10-25.json) | 原 BF16 CUDA 普通随机 R2 用例通过：取消 A 后 B 保旧状态续跑、新请求接纳、错误输入隔离、credit=1 背压及增量续轮；已修 B 残留取消状态。正式回归已严格编译／链接，另 7 项主机检查核取消不遮蔽后来真实错误。不是在线 CLI、设备错误注入、MTP credit 窄化或其它平台通过 |
| [Linux](../evidence/correctness/request-host-linux-portability-2026-10-25.json)／[macOS 主机原语检查](../evidence/correctness/request-host-macos-portability-2026-10-25.json) | WSL／GCC13、指定 Air／Apple Clang21 三套严格编译及运行通过；测试分配钩子修正后 Windows 两套复验通过；不是 Linux／Apple GPU 推理检查 |
| [NF4 冷 metadata](../evidence/correctness/nf4-original-cold-metadata-2026-10-25.json)／[原值 F16 组件](../evidence/correctness/nf4-original-f16-component-2026-10-25.json) | 原 15,360 个解码值及 CPU／CUDA M1、M2 投影通过；完整 F16 激活／导入及其余 351 份原 state 未完成 |
| [CT asymmetric INT4 metadata](../evidence/correctness/ct-asymmetric-qwen-cold-metadata-2026-10-25.json)／[原投影](../evidence/correctness/ct-asymmetric-qwen-original-projections-2026-10-25.json) | 同一原 loader／Recipe 的 28 角色、七组投影及 178,176 值通过 CPU 独立原值检查；M1 七组 packed，M2 按既有策略分解。不是完整 27B 或性能／其它原权重后端结果 |
| [DeepSeek 普通短历史](../evidence/correctness/deepseek-ordinary-early-history-2026-10-25.json) | 原三层 target CUDA 组件通过空启动、ratio4／ratio128 首压缩及 SWA 回绕；给定 126 前缀不是实际执行前 126 行；旧 Flash／Pro 消费者仅语法检查 |
| [GLM Pro 真实 index 短历史](../evidence/correctness/glm-pro-ordinary-short-history-real-index-2026-10-25.json) | 原 layer74 的 7 个角色供给 target77，CUDA 连续可见 1／2／3／4 行、完整状态提交、4 次 stale 及 11 个非法索引拒绝通过；74 上游输入及 76 hidden 是明确 fixture，不代表完整 78 层生成 |
| [控制传输归因](../evidence/performance/inference-control-transfer-2026-10-25.json) | 既有 trace 的直接 H2D＋显式同步 API 约占普通／MTP warm 窗 0.528%／0.531%，尚未确认约 10% 差距主因 |

## 最新完整性能对照

本轮尚未测量。上一轮当前产品的普通 CUDA decode：单轮／两轮 **97.43／96.67**，
llama.cpp **101.83／101.78** tok/s；原 prefill、空间、综合与公平性边界见
[性能资料](performance/decode.md)。同一历史窗口 MTP 单轮／两轮 **120.15／124.14**，参考 **134.88／140.10** tok/s。这些是标明日期与工况的已测版本，不代签本轮的新功能、合批或多用户吞吐。
每次完整对照后更新这里的 prefill、decode、内存／显存及多轮／多请求综合速度。
