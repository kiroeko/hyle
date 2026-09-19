# M0：模型需求、成熟 Provider 与成批接入方式

## Closure card

- round / owner：新 LLM/MoE 完成 goal，M0，`/root`。
- baseline：`ebe5af5109afb01b029c1ba85484d03be61d6fe6`；Windows 唯一产品真相。
- trigger：用户明确撤销逐算子自写顺序，要求先模型需求、Provider、接入方式，再四路线找库接入。
- 上位合同：[唯一总计划 M0–M6](llm-moe-coverage-completion-plan.md)；
  [冻结与退场](llm-moe-self-operator-freeze.md)优先于旧 C1 任意 FIX。
- consumer：八个冻结模型输入的语言主干；大模型只要求计算能力，本地 Qwen3.5-2B 做完整旅程。
- 主纵切：ggml 模型图中 tensor→线性/attention/递归或专家片段→状态更新；HYLE 普通 Direct/Graph。
- 正交纵切：带 stride/索引的数值片段、独立 invocation、保留旧结果和具名错误。
- owner：compute、compute_ops、graph、compute_backend_cuda、compute_provider_native/cuda、
  compute_route_apple/vulkan、graph_route_apple/vulkan、compute_provider_ggml_vulkan。
  这些 owner 通过冻结移交后仅 SCOPED；**本卡产品 FIX 为空**。未来 M1/M2 必须另审有限接入 FIX。
- 依赖：已固定模型/llama.cpp/ggml/SGLang/MLX/oneDNN 源码与旧 receipt。旧 ggml CPU 与
  oneDNN BLOCKED 不变，CLOSED 下层无新反例不重开，不注册空 Provider 充数。
- scope：主计划 M0-01..04；覆盖所有 F01–F19，非仅旧 C1 基础张量。
- non-goals：产品实现、自研算子调研/扩张/优化/性能比较/删除、模型重新下载、重做已有参考、构建与 P1、公共治理修改。
- tier：P0 的源码/成本审计；没有新产品性能结论。
- 预算：全局新增调查最多两个工作小时，每个未决 seam 最多45分钟；优先复用已登记证据。
  到点记录具名缺口/阻塞及下一有限动作，禁止转成“库不支持，自己写”。
- 预声明验证：现有源文件/pin/hash 与需求 ID 的静态对应、目录/冻结清单完整性、JSON 解析、
  selector docs/closure、明确前序 revision 的状态迁移、docs 与 diff check。无 native/P1。
- stop：64需求域均明确四路线候选与接入方式，所有候选的准入处置与下一 owner/FIX 可审阅。
  未核对后端不签产品覆盖；未决项不能静默排除。M0 完成后实际推进 M1/M2，不扩研究目录。

## 有限清单与当前事实

| ID | 必需产物 | 当前事实 |
|---|---|---|
| M0-01 | 八个输入的模型能力→64需求→实际图/算子符号与状态/格式 | 原来源、64域与七大模型入口可复用；逐域后端映射待完成 |
| M0-02 | 四后端支持/限制与 HYLE 已绑定域；成组 lowering/region 边界 | ggml Vulkan/MLX 已有基础，CPU runtime 仍阻塞；CUDA ggml 未接；不能记为全量接入 |
| M0-03 | 全部候选的有限准入决定、版本/许可、C++ seam | 目录已重写为广覆盖优先；新增专用库的需求价值待逐项定案 |
| M0-04 | 各 owner 接入 FIX、模型/配置/设备/驱动条件路由、适用验证、冻结基线和历史移交 | 冻结基线与旧授权撤回先落盘；产品 FIX 尚未授权 |

## 七问的本阶段边界

1. 来源/许可：先复用已固定 pins，区分 llama.cpp 自带 ggml 与独立 ggml，不默默互换或升级。
2. 完整消费：模型图决定原子和融合边界，真实状态/量化不可由 F32 smoke 代签。
3. 基础设施：已关闭 Mem/Sched/backend 只按必要 seam 复用，不重审；ggml abort 有独立阻塞责任。
4. 成本：原生融合、持久计划/线程池、常量 packing、转换/同步明确；M0 不测竞速。
5. FFI：HYLE 公开 C++ 接口不携带模型加载器/Python/Torch，vendor 类型按路线 owner 隔离。
6. 三交付：每个准入库绑定现有 CMake lock/feature/target/resource/package，后续成批验证。
7. 易用性：用户选择路线，普通 Direct/Graph 匹配固定规则，SDK 不让用户排序 Provider。

## 首次移交

旧轮未完成且不复活。新计划重写不算产品进度；现有64行/256格全门验收状态保持 UNPROVEN。
本卡只授权有限源码/文档工作，已有自实现全部冻结至 M5；预计新增自研为0。


## 新 scope 移交依据

`4de4ddbd` 已单独提交旧授权撤回、新总计划、Provider 目录与冻结合同；docs/closure 及
前序状态检查通过，满足冻结记录的恢复条件。本次只将对应11个旧活动 owner 从 BLOCKED
转为 SCOPED 并绑定本卡，产品 FIX 仍为空。ggml CPU、两个 oneDNN 的原 BLOCKED 及
所有 CLOSED/QUEUED 均未变；无产品改动、native 构建、自研调查或 P1。
