# R4 CUDA 性能比较：预审与有限合同

本记录从属 [R4 主卡](cuda-v1-four-route-foundation.md) 的 C1–C5。当前为 **AUDITED / 采集资格已恢复**，
不是新的产品 owner 或一次已通过的 P1。[停止事实](../../../../evidence/correctness/cuda-r4-qualification-progress-stop/README.md)
保留两次启动和有限诊断；以下是计时前冻结的合同。API/算法、正确性及交付固定后，才把完整 commit/tree、
编译器、依赖和机器身份写入统一仪器合同并执行正式采样。原 5% 门、n=15→最多31、一次源码归因
机制纠正预算保持；没有修改公共仪器。

后续已取得[独立 CUDA 的线程调度证据](cuda-r4-context-causality.md)。新的
[有限采集接线](cuda-r4-native-helper-qualification.md) 使用 `bench/contracts/{compute,graph}-cuda-ffn-p1-v2.json`
与已单独验证的 `native-runtime-pair` 工具支持；v1 文件、失败数据和原门全部保留。v2 未通过进度/A/A
前不启动 P1；工具修改不代表本模块已经恢复或通过性能。

`ea8a7101` 接线候选现已通过[全部预声明进度与 A/A 回归](../../../../evidence/correctness/cuda-r4-native-helper-qualification/README.md)。
主卡正在按 BLOCKED → SCOPED → AUDITED 恢复；尚无新 P1 verdict。正式运行使用 v2 合同和
fresh build，准确 artifact 不同则取得自己的 A/A receipt；不沿用旧单核样本或新 build 之外的 PASS。

v2 现已由 `a6775659` 实际完成 Direct/Graph 两包，均 n=15、qualification 与5% NI PASS；
详见主卡链接的完整 manifest/raw replay。全部分层没有成立的 HYLE 明确优势，局部小输/赢不删。
按原一次纠正预算进入[成熟默认规则接入](cuda-r4-mature-activation-rule.md)，最终选定实现再冻结
适用 P1 与同数学原生成本；不把本次基线比较误记为最终默认已交付。

## 1. 必须分开的比较

裸 cuBLAS、Marlin launcher、CUDA Graph 与 HYLE 公共 API 不提供完全相同的 admission、close、
错误传播、失败槽退役和 Completion/owner 保活。数值和最终设备事件相同，不足以把它们标成完整
`SAME_CONTRACT`。因此固定下列命题，不在计时后更换：

| 项目 | 两侧实际执行 | 类型 / 用途 |
|---|---|---|
| 默认规则绑定 | 普通 CUDA factory 与显式组成相同成熟 Providers；同一公共 Compute 路径 | P0/正确性证明 cold 绑定和 hot 固定调用；同实现 A/A 用于仪器资格，不计作算法胜利 |
| Direct 完整 FFN | 同样三个 Compute operation/invocation/Completion；只替换 SwiGLU 为下述精确成熟参考 kernel | 同层 Provider 组合的 `SAME_CONTRACT` P1，原 5% 门，独立记录实际方向 |
| Graph 完整 FFN | 同样的 Graph compile/bind/prepare_replay、资源规划和实际 captured 程序；只替换同一激活片段 | 完整 captured FFN 的 `SAME_CONTRACT` P1，原 5% 门，独立记录实际方向 |
| 原生包装成本 | 同实现 native 数学链与对应完整完成边界；包括实际调用、转换、同步和回收责任 | `NATURAL_ROUTE` 成本诊断，公开额外语义与成本；不能签同合同 NI 或 SOTA |

Direct 的三个真实 Completion 不能与一个最终事件的 peer 混成 NI；两侧按同一 queue 提交三段，
在读结果/重用槽前等待实际终态。Graph 必须对比 captured 完整链路，不能拿逐 kernel host launch
作更弱对照。原生诊断的负向事实仍记录，不以其不适合排名为由隐藏消费成本。

## 2. 成熟参考与数值预审

具名源码身份沿主卡：vLLM `51918e252af4dfff381001f42b7ebe6a0454501b`、SGLang
`06992c92ad4494fe9f9636e75be3e35040512447`。

- HYLE `src/compute_provider_cuda/native_kernel.cu`：FP32 SiLU 和乘法，只在最后转 FP16，
  不启用 fast-math。cuBLAS 保留相同 accumulation/precision 配置；Marlin 保留相同 group128
  Q4、FP16 scales/dequant/output 与 FP32 reduction，不能换权重格式以换取胜利。
- vLLM `csrc/libtorch_stable/activation_kernels.cu` 的 FP16 默认路线先把 activation 转回
  scalar/packed FP16，再乘另一半，数值合同不同；不能直接作为精确 NI peer。
- SGLang `python/sglang/kernels/jit/csrc/elementwise/activation.cuh` 的
  `kRoundActivation=false` 是 FP32 activation/multiply、最终一次输出转换。
  `ops/activation/activation.py` 在 SM89 的默认模块却启用 fast-math；参考明确采用其
  `activation_module(dtype, fast_math=False)` 对应的精确配置。JIT `utils/arch.py` 的基础 flags
  为目标架构、C++20、O3、relaxed-constexpr，不另外隐含 fast-math。
- 使用 packed、out-of-place、PDL=false、expert filter=false、reuse input=false 分支；16-byte
  tensor/vector alignment、FP16 hidden%8、FP32 hidden%4 与 32-bit 索引限制显式成立。

有限接入只有一个 **benchmark reference adapter**：保留该 pin 的设备 kernel 算术与索引，
隔离不需要的 Torch/TVM host glue，保留 Apache 许可、原文件 hash 和差异；使用与产品相同的
HYLE shared-queue Provider 生命周期，只替换 kernel。双方都实际支付正常生命周期服务，不能
给裸 kernel 加空原子、空循环或等待来对齐数字。参考本身须先编译、通过独立数值 oracle 和
完整 FFN 消费，才能取得比较资格。[编译与完整数值证据](../../../../evidence/correctness/cuda-r4-bench-preflight/README.md)
现已通过；原生同实现链也通过[独立 oracle](../../../../evidence/correctness/cuda-r4-native-cost-preflight/README.md)。
这些检查不替代尚未运行的 A/A 或 P1。

本阶段不再追加 peer 搜索。若上述窄接入在一次有明确原因的适配纠正后仍不能满足合同，记录
具体 blocker；不换更有利 peer、不改精度、不增加公共 Provider API 或第二套产品执行器。

## 3. 有来源的有限形状

主要原型来自 vLLM pin 的 `benchmarks/kernels/benchmark_shapes.py` 中 Llama-2-7b TP1：
gate/up K4096→N22016、hidden11008、down11008→4096；`benchmark_marlin.py` 正常 M 集含1与16。

| Workload | 冻结形状 / 链路 | 角色 |
|---|---|---|
| decode FFN | M1，Q4 gate/up → precise SwiGLU → FP16 dense down | primary，Direct 与 Graph 分别比较 |
| short-prefill FFN | M16，同 K/N/hidden、格式与完整链路 | holdout，Direct 与 Graph 分别比较 |

数据、权重 packing、两个独立输入/输出/运行槽在两侧同分布；轮换两组预先驻留的不同 activation，
不把固定地址误写成固定输入。资源准备和上传在两侧相同阶段，普通 GPU resident 计算不偷偷混入
单边 host 上传。权重不是每次展开为 FP32 的假压缩驻留。

这是 `PROVISIONAL_ARCHETYPE`，不是生产 trace、模型吞吐或已经证明的调用频率。material 分母为
一次完整 FFN 的受理至真实终态，预算是主卡已有的完整调用 5%；不按 kernel 数稀释或放大影响。
正交 caller-owned dense transform 的 FP32/layout/OOM/并发正确性沿 C1/C4 检查；没有测量的 dtype、
shape、SM 或模型性能不排名。本轮不扩算子或性能 case 目录。

## 4. 生命周期、时序与成本矩阵

| 项 | NI 两侧的合同 | 原生诊断的区别 |
|---|---|---|
| 数值 | 同 dtype、Q4 format/scales、FP32 reduction、精确激活和最终输出 rounding | 不同精度不得掩盖为包装差异 |
| prepare | 相同 cold pack、描述符/算法绑定、每槽 workspace/handle；冷成本另报 | 同样先准备，不把某一侧 preparation 留在 hot |
| GPU 资源 | 两槽、相同大小/对齐与驻留类型，常量共享，变化输入 | 同样实际资源；owner 服务不假称等价 |
| admission / failure | 同 HYLE 拒绝、accepted prefix、失败槽退役和错误映射 | 裸 API 由 caller 承担更多管理责任 |
| ownership / close | 同 Context/artifact/invocation/Completion 保活；borrow 活到终态 | 不假称裸 API 自动保活这些对象 |
| completion | Direct 三段的真实完成；Graph 一个实际 captured program 完成；返回后才能复用输出 | 单列对应 stream/event 边界及责任 |
| allocation | 预留 Completion，无新增 hot HYLE metadata；SDK opaque storage 仍 unknown | 不以未观察到 HYLE 分配证明 driver 零分配 |
| copy / fusion | 无单边隐藏转换/上传下载；相同完整链和捕获粒度 | 任何不同融合/转换必须披露，不能签精确 NI |
| teardown | 真实终态后销毁；冷与回收账单分列，不能单边延后析构 | 真实回收，但 API 责任不同 |

P0 源码账单：default factory 只冷组装；prepare 固定 descriptor/algo/kernel/config；hot 不重看
Provider 排序、device 或 timing profile。cuBLAS 每槽 handle/workspace；Marlin 每槽 reduce/locks；
SwiGLU 单 launch。Graph 冷捕获三段，hot 原生 replay。成熟参考只替换实际 kernel。

## 5. 环境条件与采样停止

用户已确认规则可按 GPU 型号/架构/SM、驱动、Toolkit/runtime 与 Provider 版本匹配。
2026-09-16 的只读 W1 inventory 为 RTX 4070、SM89、NVIDIA 驱动 **610.62**、Toolkit/nvcc
**13.3.73**。这不是已经执行的 P1；最终采样仍须记录实际 driver API/runtime 版本、编译目标、
Provider pin、编译器/flags/linkage、GPU 状态及机器身份。driver API 兼容版本不是驱动发行版本。

只有证据覆盖的条件可以产生默认 winner；环境不匹配时用预定合法成熟基线或明确拒绝。规则在
研发固定，cold prepare 自动匹配/绑定；不让普通用户提供 profile，不在 hot 查询/计时。
HYLE 只有取得完整消费链的明确优势才保留该域的默认优选；若无此证据，即使原 5% NI 门通过，
也采用合格成熟实现。需要替换时按原一次机制纠正预算更新适用默认规则并重新冻结；无法表达的
separate/非向量尾部继续依法保留所需 HYLE 能力，不以单 kernel 点值决定完整路线。

冻结机器合同为 `bench/contracts/{compute,graph}-cuda-ffn-p1-v1.json`。模块自己的 A/A validator
先邻置真实 DLL，核对 peer 原文片段、GPU/驱动、CUDA 编译器身份，并运行 16 个独立进程：两种
同实现路线 × 两个代码位置 × 两种执行顺序 × 两种分配顺序。每个进程含 M1/M16；5% 因子中位
偏差门与 10% 单进程严重失稳停止门沿已用方法。M16 是预声明正交形状；统一 runner 的统计
train/holdout 分割保持关闭，不能将其冒充独立未见数据。该 bundle 后置检查仅复用完全相同
artifact/编译器/DLL/validator 身份及原始日志 hash 的 A/A receipt，并重新检查 GPU/驱动状态。
不重写公共仪器、统计或 power profile。

首次 `cc38c398` 资格启动保留在 `p1-compute-v1`：A/A 16 进程通过，但 A/A 当时只固定 caller，
统一 runner 则固定整个进程到 CPU2。首个正式子进程在冷准备阶段超过 120 秒，人工停止；没有
完成的产品样本，也没有 P1 verdict。两核完整夹具、随后单核相同 binary、独立 CUDA 内存/事件/
cuBLAS heuristic、小型 HYLE context/packing 与受控 caller 优先级探针均正常完成，未稳定复现
根因。此事实不证明驱动、Sched 或回收线程缺陷，不据此改产品或公共仪器。

有限资格纠正仅将 A/A 的整个进程也固定为同一 CPU2，核对 mask 后再启动子进程，保留真实 DLL
邻置、完整数值和原门。此前 A/A 只作诊断，不能签新资格。修正冻结后允许一次相同算法的 fresh
资格启动；若再有同类冷启动超时，停止本次性能采样并保留具体现场，不用扩 n 或改变亲和性掩盖。

正式 n=15 之前必须通过：源码时序/成本审查、独立正确性 oracle、同 workload A/A、双执行与
分配顺序、两个 placement、分辨率/功效和设备环境资格。使用现有统一仪器、P 核 caller、实际 DLL
邻置；W1 是 GPU 工程验证，不借用 CPU Formal 身份。任何 preflight 失败先停止；不得靠 n=31
补救不合格比较、地址效应或仪器噪声。API/算法未固定之前不启动正式 P1。
