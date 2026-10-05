# D5 CPU：原版 ds4 状态量化批量接线

<!-- MODULE REVIEW: CLOSED -->

## 范围与重新打开依据

baseline `5fabf548565a4598b8b7bf490b2eaabfc942cca1`；owner `compute_provider_native`。
重新打开依据：D5官方DeepSeek两个状态量化合同是实际缺口，
[新原生资格](../../../../evidence/correctness/llm-moe-d5-real-fragments/mature-low-precision-batch/README.md)
证明ds4原CPU group64 FP8、group32 FP4可以承载。D4的自研退场不撤销，不恢复旧自研数学。
普通用户仍使用既有CPU Direct/Graph入口；不新增公开API、Provider排序或兼容层。

## 有限 FIX N1–N4

1. 将MIT许可的ds4 `9139e2ae58a41503968a500f36f75895c1ba63fc`原函数窄摘录为私有依赖，
   固定完整source/函数/归档/许可hash，沿既有CMake lock/package seam投影；不复制整模型。
2. 同批接入既有`BlockQuantizationRoundtrip`的FP8 group64和FP4 group32：冷期只按
   精确操作、BF16 dense whole-group形状绑定原函数；group128无成熟同合同原函数，继续拒绝。
   每组用固定不超过256字节栈暂存，经已有SIMD基础BF16转换适配原F32函数，再写回BF16。
   无热分配、权重缓存或新线程池；不修改NativePlan布局，不改变算术顺序/scale/rounding。
3. 复用既有CPU资源映射、invalidate/flush/close、Completion和失败通道。非法descriptor、
   alias、容量由原公共层检查；非有限payload安全失败，已接受前缀不承诺回滚。
   invocation不共享可变scratch，完成对象保活和OOM继续走现有设施。
4. 同一候选批量覆盖普通Direct/Graph、两种格式、变化输入、极小值/全部有限BF16/中点、
   group128冷拒绝、别名/资源失败与重复执行；验证受影响owner和真实CPU消费者。

## 取舍、验证和停止

本卡是两个必要成熟操作的接线，不能代签完整官方FP8/FP4 MoE或增加完整画像格。
C10转换的完整group128线性组合另有真实缺口，本卡不把自写循环包装为新成熟算子。
默认策略只覆盖当前其它成熟Provider拒绝的精确域；未引入热期搜索或性能调优。

P0审成本；没有自研算法比较或逐算子P1。D5完整片段P1义务保持，新增状态能力的完整
消费者成本尚未验时不能写总性能PASS。源码/静态/动态及Windows/WSL CPU正确性按受影响
批次完成，不重跑GPU/Apple或已合格其它D5全矩阵。
一次批量接线、一个不可变候选的低并行构建/串行验证；仅明确adapter/构建错误允许一次纠正。
适用选择器、docs/closure（明确previous-ref）、统一LOC；三交付/资源未验仍待验。
原函数出现不符合合同的新反例则停止该域，不优化数学或放宽精度。完成有限FIX后保留D5其它缺口。

## AUDITED 决定

来源归档通过本地代理取得并与Windows固定Git blob核对，六个数学函数原样借用；
许可为MIT。只有既有CPU provider取得私有依赖，不改变公开头或路线对象。
N1–N4固定；同一操作家族的group128仍拒绝，C10和Apple失败候选不进入本批产品。

### 首次配置的接缝纠正

候选 `1486d10c` 首次 configure 检出 `model_ops.h` 的 owner 是 `compute_ops`；
native 过去仅间接使用它，新增直接 include 必须如其他 Provider 一样声明直接私有依赖。
本卡的 adapter/build 纠正包含 native 的 `PRIVATE_DEPS compute_ops`；
这是现有 schema 的显式依赖，不增加运行路线、模块或自研数学。配置失败不计任何测试 PASS。
后续依赖审计发现生成摘要时 Windows Path 使用不区分大小写排序，与 CMake 的文件名排序
不一致；按现有 CMake 算法纠正 lock 摘要，三个 vendor 文件和原函数全部不变。
两次均停在配置期、尚未编译，保留各自日志；合并为本次构建接缝修正，不开启数值重试预算。

## 不可变候选与补验

产品候选 `dd3009f0ff2014de048e6a765d07746d5eba1bd4`；2026-10-05 补验时的
`223589e4` 仅更新设备可用性文档。N1–N4 产品接线固定，后续无算法/API 改动。
原 `custom_as` optional 适配、测试 include 和终态 invocation 退役用法均已修正。
源码交付补验修复的是包装脚本的环境变量大小写，核对源文件和消费者后复用已编译产物。

[本批证据](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-ds4-roundtrip/README.md)
保存普通 Windows 两入口 6 项、Windows 三交付 18 项、WSL 两入口 6 项通过。
WSL 使用 oneDNN OFF 的实际 CPU/Graph 产品和两个独立消费者，产品 `-Werror` 保留；
旧完整测试目标的无关 oneDNN 测试和两类夹具告警没有扩成本卡产品改动。
中止的无关构建和原夹具失败保留，未记为通过。Windows 默认组合/三交付仍 oneDNN ON。

有限能力范围仍只含两种成熟 roundtrip，group128、完整官方 MoE 与新增完整片段 P1 保持缺口。
统一实现 LOC 为 60,481 行 / 242 文件；本轮补验没有产品代码变化。

## 有限能力收口：2026-10-05

本轮只关闭 N1–N4 的两个 CPU 状态量化原子，不关闭 D5 模型画像或 Graph/Vulkan 阻塞。
能力和入口回映如下；semantic owner 为 `compute_ops`，算法来源为 ds4，
执行/资源适配 owner 为 `compute_provider_native`，交付复用现有 lock/CMake/package 合同。

| 能力族 | 当前旅程与边界 | 覆盖 / 动作 |
|---|---|---|
| FP8 group64 状态 roundtrip | BF16 dense whole-group 输入，经原函数量化再恢复 BF16；普通 CPU Direct/Graph | COMPLETE / BORROW |
| FP4 group32 状态 roundtrip | 同一旅程，使用原 FP4 scale/nearest-even 合同 | COMPLETE / BORROW |
| 已准备程序的资源与失败生命周期 | 既有映射/flush/close/Completion；失败实例退役，从原程序重建；metadata OOM 后恢复 | ROUTED / KEEP |

全部入口仍为既有 `model_ops::block_quantization_roundtrip` 和 CPU prepare/Graph compile；
没有新公开头、FFI、Provider 排序或运行时择优接口。普通用户说明在
[Compute 指南](../../../user/compute.md)。这是既有通用原子的窄接入，没有模型加载或应用策略下沉，
不需要另造正交 API。group128 与完整官方低精度 MoE 不在本卡关闭范围，D5 原 MUST 保留。

| Gate | 实际证据与适用边界 | Verdict |
|---|---|---|
| scope | N1–N4 固定；仅两个已资格化成熟原函数域，无新增 HYLE 数学 | PASS |
| architecture | 既有 Operation/Provider/资源路径；原函数、MIT、source/hash/lock 固定；不支持域明确拒绝 | PASS |
| correctness | Windows 普通两入口及 WSL 两入口；数值、变化输入、别名、metadata OOM、accepted failure/退役/重建；逐项结果见本批 receipt | PASS |
| delivery | Windows source/static/shared 三交付各 6 项；独立搬迁、依赖/许可与邻置 DLL 来源检查 | PASS |
| usability_docs | 两种格式的普通入口、形状、非有限 payload 失败、实例重建和 group128 限制已写入用户指南 | PASS |
| red_team | 候选后独立源码复核原函数/精度、资源关闭、可变状态隔离、私有头与许可交付；见证据的复核段 | PASS |
| engineering_performance | 本卡预声明 P0：线性分组、有界栈转换、无新增热分配/共享 scratch/线程池/搜索；完整消费者 P1 未签 | PASS（P0） |

命令、工具链、测试 XML、修正前失败/中止日志和 SHA 均在
[本批收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-ds4-roundtrip/receipt.json)。
WSL receipt 只签两个独立公共消费者，不签完整 Linux 套件或 Linux 三交付。
本卡没有运行新的 sanitizer、GPU/Apple 或完整模型 P1，也没有把这些写成通过。
原产品影响计划不代签全路线；按已存在的 Provider partial-region 合同验证本卡独立能力。
本次 terminal 只改文档/状态/证据，selector 仅选 docs/closure，无产品重建或 P1。

最终裁决为 **CLOSED（本轮有限能力）**；产品 candidate 仍为 `dd3009f0`。
重开仅由该域可复现数值/UB/资源生命周期缺陷、相关 ds4 源码/许可/ABI 升级、
真实新消费者所需的语义或合格同合同完整成本回归触发。完整官方 MoE、group128 和 D5
Vulkan 数值/测量资格另属当前聚合尾项；不以这里的 CLOSED 自动解除。

58 份永久证据逐项解压/核对 SHA 后，删除本批 5 份临时源码传输归档，共释放
190,668,800 字节（约 182 MiB）；[清理收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-ds4-roundtrip/cleanup.json)
保留每项路径、大小和摘要。可复用构建、依赖缓存、模型及 canonical 源码均保留。
