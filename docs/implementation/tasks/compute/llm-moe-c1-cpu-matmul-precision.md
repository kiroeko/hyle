# C1-Z：CPU MatMul 累加精度准入修复

> **2026-09-19 授权移交：本页仅保留历史实施/验证事实。** 当前施工改由
> [新总计划 M0–M6](llm-moe-coverage-completion-plan.md)与[M0 卡](llm-moe-m0-provider-scope.md)管理；
> 下文旧 FIX、下一步与 AUDITED 叙述不再授权继续实施或性能重测。
> [所有自实现冻结](llm-moe-self-operator-freeze.md)，最后一次有限比较落败即删除。
> 若 registry 仍指向本页的 BLOCKED，原 blocker/resume_condition 继续有效，新 goal 不自动解除。

## Closure card

- owner：`/root`；`compute_provider_onednn` 定向重开，`compute_provider_native` 沿
  [C1 活动卡](llm-moe-c1-tensor-state.md)追加有限 FIX，不重开 CPU/Graph route 或 SIMD。
- baseline：`55830705d95150ecfd71a4d49798c07457dc379c`。
- trigger：共同 `MatMul` schema 接受数值累加类型；oneDNN Direct `select_route` 与 Native
  `build_plan` 的 MatMul 分支却没有匹配 `accumulation_type`，实际实现只使用 F32。
  同样的 F32 `[16,5] * [5,4]` 描述配 `MatMul{DataType::kFloat64}` 因而可能被错误接纳。
  oneDNN Graph、cuBLASLt、Apple 与 Vulkan 的 MatMul 已拒绝非 F32 累加，不能靠下游掩盖 CPU
  普通入口的缺陷。先用回归重现接纳错误，再修准入；尚不声称已运行数值反例。
- 前轮 terminal：[R6 交付](../build/vulkan-r6-product-delivery.md)；本轮满足其可复现语义缺陷
  reopen trigger，原交付/P1 事实保留，不把旧 CLOSED 当本轮验收。
- 主 consumer：普通 CPU Direct/Graph 显式指定矩阵累加精度，候选必须保留该语义或明确拒绝。
  正交 consumer：专家限定 oneDNN/Native 或冷提案与精确准备分离；不能在 fallback 中降精度。
- scope：两个 CPU Provider 的 MatMul 冷准入与相关查询/准备/默认入口回归。
- non-goals：新增 F64 GEMM、改矩阵算法、扩 C2、改 SIMD、重排性能规则、重审其他算子或治理工具。
- 依赖/消费者：现有 Compute schema/preparation、CPU backend、oneDNN 3.13.2；直接消费者为
  CPU route 和 Graph oneDNN。只修实际拥有数值能力的 Provider，不在共同 schema 禁止合法类型。
- 性能：P0。一次冷 enum 判断；成功的 prepared invocation、内存布局、计算/同步/资源成本不变。
  此修复不进行 P1，不把缺检查实现的旧性能当作非 F32 域的合格证据。
- 预算：源码核对与回归设计最多20分钟，最多两轮具名产品纠正；不扩大为所有 Provider 全库复审。
- 预声明验证：不可变候选的 selector owner/真实一跳消费者；W1 Release/WERROR static/shared
  对应单测及普通 CPU Direct/Graph，WSL x64 对应定向语义回归；DLL 邻置、PATH unset。
  Native 的其他平台完整 C1 资格仍归主卡。公开头、ABI、CMake、vendor pin 均不变时复用原三交付
  形状及安装证据，不重复全 package。若 selector 因实际修改选择更广门，按其结果执行。
- 停止：有限修复与适用门完成后 oneDNN 定向轮 CANDIDATE→CLOSED；Native 保持 C1 AUDITED。
  未运行风险不签 PASS；外部条件确实阻断则给出具体 blocker 与恢复条件。

## 有限待审项

| ID | 具体问题 | 责任边界 |
|---|---|---|
| Z01 | oneDNN Direct 冷提案/准备没有匹配 MatMul 累加类型 | 共享准入函数拒绝本实现不支持的精度；保留 F32、shape/resource 和失败语义 |
| Z02 | Native fallback 存在同一遗漏 | 在 Native 自己的 MatMul 计划资格处检查；不能由 oneDNN 拒绝后静默降级 |

SCOPED 登记提交为 `17674262`。下列审计与有限 FIX 提交后才实施产品修改。

## AUDITED：有限 FIX 与冻结决定

2026-09-18，owner `/root`。共同 `validate_accumulation` 允许 Unknown 或数值类型，
`OperationPreparationAccess` 只核对 schema/资源，不代表具体 Provider 实现支持该精度。
oneDNN 的提案与准备共用 `select_route`；Native 两条路径共用 `build_plan`。本轮在各自共享
入口修复，避免只挡公开 `prepare` 而让 Graph 冷提案或精确准备旁路。

| FIX | 允许改动与验收 | 当前裁决 |
|---|---|---|
| Z01 | oneDNN `select_route` 只接纳显式 F32 accumulation；F64/F16/BF16/I32/Unknown 返回 Unimplemented。测试提案、普通 prepare 和复用 F32 提案但改变精度的精确 prepare 均拒绝 | FIX；第三方 BRGEMM 原样 BORROW |
| Z02 | Native MatMul 的共享计划入口执行同一资格判断；普通 CPU 默认工厂不能在 oneDNN 拒绝后落到不合格 Native。回归覆盖解码/预填充及转置权重域，既有 F32 数值测试必须继续通过 | FIX；既有 SIMD/计算实现 KEEP |

未知累加类型没有被这两个 Provider 声明为可自由选精度的域；默认 `MatMul{}` 已明确 F32。
普通 C++ API、布局、资源所有权、运行状态和错误类型不变：合法但未实现的语义返回
`Unimplemented`，非法描述仍由共同 schema 拒绝。只增加冷 enum 判断，不增加 payload 检查、
热路径分支、分配、锁、拷贝、同步或常驻字段；F32 计算与既有路由形状阈值保持原实现。

七问裁决：① oneDNN 3.13.2 pin/许可证/升级边界不变；② 具名显式精度请求不再静默降级，
不承诺新增 F64 实现；③ 复用共同 schema/准备与既有 SIMD，不复制基础设施；④ P0 冷检查，
没有同合同热路径变化，不跑 P1；⑤ 不增加 C/FFI；⑥ 私有 C++ 两处判断，无头/CMake/package
变化，原三交付形状证据复用，static/shared 对新实现重跑；⑦ 默认 CPU/专家入口都有回归，
用户 guide 明确有限 F32 域。测试先在未修产品上确认失败，再以同一测试验证修复。

## Candidate

- 审计提交：`06499f6a`；回归提交：`4d6cf12fd3263baaa0697eefc43dad7b6a7af274`。
  未修产品上 Native、oneDNN、默认 Compute context、默认 Graph context 四个定向用例均因
  错误接纳 F64 请求失败，确认缺陷可复现；这不是浮点容差或性能 fixture 的推断。
- 产品候选：`e191351522340dd997cfc925abe3a69d47746d14`。两个共享冷准入函数共增加6行；
  成功 F32 的算法、Provider 阈值、执行/资源代码、布局和构建边界不变。回归同时保存有效
  F32 提案的 resource requirements，避免失败查询清空输出槽影响精确准备测试。
- W1 MSVC 19.50 Release/WERROR/shipping layout static/shared 各8个定向 CTest目标通过：
  154个单测及独立 CPU executor 消费样例；包括既有 F32 数值/重放、组合和 Graph region。
  DLL 邻置、PATH unset；CPU 编译固定 parallel2，动态构建复用未变 oneDNN 对象。
- 初次 CTest runner 的 regex 缺少 `hyle_` 前缀，选中0项；该次不计 PASS。修正为真实目标名、
  加 `--no-tests=error` 后才取得上述8项结果；没有修改测试语义或门限。
- WSL/GCC14 使用 Windows 候选 Git archive 的验证副本；进入 CANDIDATE 时尚未验收，
  此后完成下列终验。没有启动 P1 或重跑无变化的 package 门。
- 同一23路径的统一 LOC（含 GLSL、不含 tests/bench/vendor）为 **46,085行 / 191个实现文件**，
  self-check PASS。相比上一产品候选仅增加6行资格检查，不计为 C1 新算子覆盖。

## 终验与裁决（2026-09-18）

[结构化证据](../../../../evidence/correctness/compute-llm-moe-c1/cpu-matmul-precision.json)、
[原始日志/命令/身份](../../../../evidence/correctness/compute-llm-moe-c1/cpu-matmul-precision-raw.zip)和
[统一 LOC](../../../../evidence/correctness/compute-llm-moe-c1/cpu-matmul-precision-loc.json)绑定同一
产品候选 `e1913515`；其后的提交只改文档/状态/证据，没有因此重建或计时。

| 环境 | 实际结果 |
|---|---|
| W1 MSVC static | 8/8 CTest目标；154个单测 + CPU executor consumer；零 skip |
| W1 MSVC shared | 同上；DLL 邻置、PATH unset |
| WSL GCC14 static | 同上；Windows Git archive 只作验证副本，归档 SHA 已记录 |

每套154例为 Native108、oneDNN13、组合1、oneDNN普通Graph1、Graph region20、默认CPU4、
默认Graph7。既有数值/重复执行/资源回归仍通过；这不代签 C1 全矩阵、真实 checkpoint 片段或 P1。
selector 选择两个 owner 与 CPU/Graph oneDNN/Graph CPU 三个真实一跳消费者，上表覆盖全部
所选 factor correctness。headers/package/P1 的 skipped reasons 保留在原始计划与证据中。

WSL 首次完整编译621步已结束后，任务临时脚本因运行期间追加日志收集代码而在旧文件偏移
报 EOF；最终脚本语法检查通过，同一直接 build 返回0且 `no work to do`，随后测试全部通过。
这属于本轮 wrapper 纠正，没有修改产品或测试断言，也没有重跑重型构建。

| Gate | 证据与适用范围 | Verdict |
|---|---|---|
| scope | Z01–Z02 完成；明确不新增 F64 算法，不扩 C2/全库检查 | PASS |
| architecture | 资格检查归实际 Provider 的共享冷函数；共同 schema 仍可表达合法但未实现的精度 | PASS |
| correctness | 未修候选四个回归入口失败；修复后三环境全部目标通过，精确准备与默认 Graph 不旁路 | PASS |
| delivery | W1 static/shared 实际源码构建与普通 C++ Direct/Graph；公开头/ABI/CMake/package seam 未变，继承前轮三交付形状/安装证据，不冒充本轮跑过 relocation/offline | PASS |
| usability_docs | 普通调用保留默认 F32；其他类型明确 Unimplemented；用户 guide 已说明 fallback 不降精度 | PASS |
| red_team | 独立复核候选差分：共同 inference 只校验描述，两个 Provider 都在提案/精确准备共享入口拒绝；F32 计算、阈值、SIMD/资源/执行函数无改动；没有仅挡一个公开入口 | PASS |
| engineering_performance | P0：两次各自独立的冷 enum 判断，热路径与对象成本不变；P1 不适用本定向语义修复 | PASS |

oneDNN 本定向轮 **CLOSED**；Native 继续归 C1 的 AUDITED，不把这6行修复当成整个 Native/C1
关闭。后续 oneDNN 更广算子/region 接入依据新 consumer 范围另行 scope/audit，不从本卡取得授权。
剩余 C1 平台/真实权重/完整三交付和等责性能仍在主卡；没有外部 blocker 阻碍本次修复。
reopen trigger：可复现精度/资格/失败回归，新增已准入累加类型或依赖升级改变该域，或公开合同变化。

磁盘：Windows static/shared 复用一个构建目录；WSL 只保留固定验证副本/构建，已校验并删除
79,667,200字节传输归档；本轮 Temp `hyle-*` 检查为0项。不复制模型、不安装工具或改变系统性能模式。
