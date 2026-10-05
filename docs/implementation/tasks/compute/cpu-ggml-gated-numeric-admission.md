# CPU ggml：门控非有限值语义接缝

<!-- MODULE REVIEW: CLOSED -->

## Closure card

- owner：`compute_provider_ggml`；baseline `df9006e8`，Windows canonical。
- trigger：[相同输入的实际反例](../../../../evidence/correctness/cpu-native-domain-admission/README.md)：
  generic separate SiLU gate=-Inf/value=1，当前 ggml 输出 NaN，稳定零极限语义输出 0。
- 原范围：[D5 真实片段](llm-moe-d5-real-fragment-acceptance.md)保留为历史证据；本轮不是 GPU、
  recurrent 模型程序或原版 ggml kernel 重审。旧有限数资格不外推非有限值。
- consumer：普通 CPU Direct `GatedActivation{kSilu}` 的 packed/separate F32/F16/BF16；
  Graph 中 gate→MatMul 的中间结果必须先符合语义，再进入下游节点，不能只修最终导出。
- 正交 consumer：批处理门控数值，包括 NaN、±Inf、±0、有限极大/极小值，以及非有限 value。
- 冻结合同：activation 的 -Inf 零极限后再乘 value；NaN 传播，+Inf 按原公式；F16/BF16 在 F32
  activation 与乘法后一次输出舍入。正常 finite 域采用已接原版 SwiGLU 的原数值门。
- scope：已有 CPU generic SiLU 门控的语义适配；其它激活/Unary 的成熟准入仍由
  [逐域卡](cpu-native-provider-domain-decision.md)批量调查，不因为尚未接入就签 UNIQUE。
- 依赖/消费者：既有 ggml CPU 3rd-party graph、资源、整程序、pool/slot；CPU Direct 与 Graph。
  不改 ggml 源、ISA 程序 ABI、公开 API、Provider 顺序、SIMD 或 Native 数学；不新增万能执行器。
- 产品形状：原版 SwiGLU 后接一个 ggml 内部的原位语义边界节点，修正 -Inf 的结果；原片段继续
  由 ggml 拥有内存和执行。callback 在冷 lowering 按 packed/separate、F16/F32 固定，无 userdata。
- 成本：一次额外 O(n) gate 检查、一个内部 node；复用 SwiGLU 输出，不增加 payload 全尺寸缓冲。
  这不是原版无适配程序的性能；后续 Native P1 必须计入检查、必要 cast 和完整片段的全部成本。
- tier：P0 数值准入。本轮不调研/优化/计时自研算子，不签任何 WIN。
- 预算：一次具体源码审计、两项有限 FIX、一批资格化；最多一次具体失败后的纠正。
- 预声明验证：Windows/Linux ggml provider 数值/资源/生命周期测试；F32/F16/BF16、packed/separate、
  变化输入重放、不同 native_threads；普通 Graph 下游 MatMul 实际使用修正后的中间结果；
  复用三交付 producer 的同一独立 Direct+Graph consumer；selector、closure/docs/diff。
  已关闭 ISA/loader/SDK 机制无变化，不重复整包全量资格；本轮适用项未运行仍是 UNPROVEN。
- stop：若边界节点不能保持数值、alias/slot 生命周期或自然 Graph 下游语义，记录实际失败，
  在一次纠正预算内解决或 BLOCKED；不把原版 NaN 冒充正确，也不据有限输入签唯一实现。

## 能力审计与有限 FIX（已固定）

原版 `ggml_swiglu{,_split}` 在 `src/ggml-cpu/vec.{h,cpp}` 直接计算 SiLU；-Inf 导致不定式。
公开 `ggml_map_custom3_inplace` 生成内部 view node，保留原 buffer/依赖/任务池；现有 `matrix()`
将 gate/value 变为 contiguous 2D，现有 BF16 cast 使用 F32。adapter 因而可以只处理已核对的
F32/F16 storage 与 packed 行映射，不增加完整激活算法或 ISA 扩张。

| ID | 允许改动 | 验收 |
|---|---|---|
| GN1 | generic SiLU lowering 接原位 ggml 数值边界节点，只对 gate=-Inf 修正为零极限乘 value；finite 原结果保持 | 三 dtype、两输入形状、非有限值、signed zero 与尾值 |
| GN2 | 定向数值/重放/whole-region 下游测试和交付 consumer；记录真实检查成本 | 原始核不改；Graph 中间值正确；原位 view 随 slot 回收，未执行不写 PASS |

## 七问与状态

借用版本/许可证/升级仍由原 ggml runtime factor 管理；计算数学核保持原版，HYLE 只拥有必要数值边界。
普通 Direct 和 Graph 不增加 glue 或公开控制项。复用现有准备、资源检查、失败/完成与原生 pool；
冷绑定而不在请求中计时/选 Provider。无新 FFI/package 因子，三交付仍必须实际运行新接缝。
性能比较仍未开始；SIMD 基础与自研 payload 冻结。原 owner 已处 AUDITED，本次在同一阶段重新绑定
此卡及具名 GN1/GN2；未改公开架构合同，不借旧卡授权新修改。SOURCE/资源/私有 ABI 审计已经固定，
实现前 registry 已指向此 AUDITED 卡；实现后的新资格按以下固定候选记录。

## 固定候选与资格进度

产品候选为 `65a02982ac108d57b13769029f56a6a6903895be`；GN1/GN2 已实现并停止变化。
Linux ggml Provider 44/44 通过，包含三 dtype、packed/separate、非有限值与多线程变化输入重放；
普通 CPU/Graph 批量测试通过，新增 gate→MatMul 完整 region 在三 dtype 实际消费修正后的中间值。
旧模型 fixture 的 14 个跳过保持明确记录，不计为通过。Windows 对应数值批次与两平台独立三交付
已通过。该候选先绑定 CANDIDATE，再接受收据并进入以下 terminal；Native 裁决未开始。

验证期出现的 GCC 问题为旧 CPU 模型测试的 aggregate 默认初始化、GoogleTest 宏外缺少花括号；
仅在该批 fixture 中修正，断言保留，警告不关闭，所有原失败日志保留。不是新数学优化预算。

## CLOSED

固定候选全部适用有限验收通过；
[实际收据、失败纠正与限制](../../../../evidence/correctness/cpu-mature-admission-batch-v1/README.md)。

| 门 | 裁决 |
|---|---|
| scope | PASS：generic SiLU 的已有数值合同，GN1/GN2 固定；未扩张自研算子 |
| architecture | PASS：原版 SwiGLU 与整程序、原生 pool/slot；内部原位边界在冷 lowering 固定 |
| correctness | PASS：两平台 Provider 各 44/44；三 dtype、两布局、1/4 threads、非有限值与重放；普通完整 Graph 的下游实际消费修正中间值 |
| delivery | PASS：两平台 source/static/shared，同一独立普通 Direct＋完整 Graph 消费者实际重复执行 |
| usability_docs | PASS：普通 CPU/Graph 无新增 glue；明确原版核、稳定边界与线性检查成本 |
| red_team | PASS：NaN/±Inf/±0、非有限 value、奇数行与线程尾部；现有资源/slot/生命周期测试通过，未部署旧 fixture 不计 PASS |
| engineering_performance | PASS/P0：O(n) 检查、一个内部 view node、无完整 payload 缓冲或热选路；后续 Native P1 必须计入全部适配成本 |

Windows 4/4 CTest 程序通过，377 个注册 GTest 中 363 实际通过、14 旧 fixture 跳过；Linux 共同批次
8/8，428 中 414 实际通过、14 跳过。跳过不代签完整模型权重或 D5/P1；本卡不签 Native WIN。
停止本卡施工。重开条件为稳定门控数值、原位 view/slot 资源生命周期、下游完整 region 或交付的实际失效。
其它激活的成熟准入仍由逐域卡处理，不因本卡关闭而签 UNIQUE。
统一脚本：Compute + Graph **61,035 实现行 / 248 文件**，不含测试与 vendor。
