# D5：优先补齐 CPU / Apple 官方低精度三类缺口

<!-- MODULE REVIEW: AUDITED -->

- baseline：`5bb3d153bfe137feb6b7e3d22ced40e513b1fd5a`；Windows canonical，初始工作树干净。
- 当前授权：2026-10-05 用户要求优先处理上一答的三类残余能力，再回到 Vulkan。
- 当前实际 owner：`compute_provider_native`；由缺失的真实官方消费能力重新打开。
  新增成熟来源调查已结束；`53be4356` 完成 SCOPED、`feba1184` 绑定有限 FIX 为 AUDITED，
  CPU-O1–O4 已接线并进入集中验证，CPU-O5 尚未完成。
  此次不重开已关闭的 SIMD、CPU/Graph 运行时或 oneDNN，不新增数学内核。
- 固定目标：CPU GLM 官方 grouped FP8、CPU DeepSeek 官方 grouped FP4/FP8、
  Apple DeepSeek 官方混合低精度完整片段，共原账本 12 个拒绝入口格。
  CUDA 的 44 个通过格、其有效三交付/P1 和 Vulkan 原阻塞不变。

## 已有事实与禁止重复项

需求继续绑定原固定模型/profile/端口/分组和量化数学，不改成其它量化制品或 A16W8/A16W4。
[CPU 边界](llm-moe-d5-cpu-official-boundary.md)、
[MLX FP4 边界](llm-moe-d5-apple-fp4-native-qualification.md)及
[C10/ds4 资格](llm-moe-d5-mature-low-precision-batch.md)已经完成的探针不重跑。
CPU 原 ds4 group64 FP8/group32 FP4 状态 roundtrip 已按
[有限接入卡](llm-moe-d5-cpu-ds4-roundtrip-binding.md)收口，但完整专家链仍未成立。
单个 scalar cast、源码符号、Provider target 或原子 PASS 均不增加完整画像格。
原 Apple 失败有 BF16 中间边界和极小值差异，不能只扩大 tolerance 或删除不利输出。

## 有限资格动作 Q1–Q4

| ID | 本次具体调查 / 资格 | 停止边界 |
|---|---|---|
| Q1 | 核对已登记 ds4/ggml 的实际专家程序是否保留官方 A8 量化；只查具名调用链，不重翻模型历史 | 已转换 GGUF 的支持不能代签官方程序 |
| Q2 | 对 CPU 新成熟承载，优先查 FBGEMM/TorchAO 的完整分组量化和原版矩阵/参考程序；固定来源 revision、license、实际 CPU 能力与 C++ 交付接缝 | 只有标量转换、AMX 限定或数值合同不符即排除，不包装 HYLE 数学为成熟实现 |
| Q3 | Apple 只查一个可区别于旧失败的成熟 FP4 程序/版本或已存在的原生组合；核对 BF16 中间存储、floor、scale 和非零极小值 | 无新机制不重跑 MLX/ds4 原失败；不加 CPU 数学回退或修改上游数学过测 |
| Q4 | 存在同合同完整候选时，先批量原生资格，再固定实际 owner 与接线 FIX | 每路线最多一批资格及一次明确编译/API 接缝纠正；数值失败即停该候选 |

来源调查仅选支撑三缺口的具名强来源；新候选没有完整程序时，给出具体缺失与可实施取舍，
不把未知写成全生态缺失，不无限搜索。来源 archive/blob 为验证输入，不是自动产品依赖。
MacBook Air 是唯一 Apple 验证机，先核对实际硬件/系统/工具链；远端均为验证副本。

## 接线与验收顺序

完整成熟程序原生资格通过后，沿既有公开 operation/profile 路由接入；语义 owner、数学来源、
执行/资源 owner 与 delivery owner 分开记录。只在准备期匹配和绑定，执行不搜索或计时。
三类能力的公共数学保持；先固定有限产品 FIX，再集中接线，一个不可变候选统一验证。

适用验证为真实完整配方的普通 Direct/Graph、变化输入和实际状态反馈、OOM/非法 ID/
Completion/关闭、受影响 source/static/shared、静态成本和新增完整消费者的适用 P1。
已有未变化证据按身份复用，不把 CPU 两原子的 P0 或旧 GGUF MoE P1 代签官方新域。
构建并行 2、重型验证串行、Windows DLL 邻置和隐藏子进程；不下载巨型模型。
完整片段未通过前保留原 154/20/2；D6 不首次接线。

本阶段 source-only 变更只执行 selector docs/closure、账本结构检查、统一 LOC 和 diff check；
产品接线前重新冻结适用验证和纠正预算。新自研 CPU/CUDA/Metal/Vulkan 数学继续冻结，
SIMD 基础保留。实质来源/平台限制超出当前授权时，给出具体方案和理由再请求决策。

## 新来源审计结果与产品范围

[本批原始证据](../../../../evidence/research/llm-moe-d5-cpu-apple-official-completion/README.md)
固定 TorchAO `a701b6a6058720c21f95908b7ae4a24bf0cae1b6` 和 LibTorch `2.8.0+cpu`。
原量化各 107,264 输入 codes/scales 精确通过；32 个真实 N/K 的 encoded FP8/FP4 投影
共 417,792 输出一致；独立 C++ CPU-only DLL/转换/矩阵接缝通过。
这些是资格输入，不增加完整画像格。FBGEMM 具名 CPU FP8 入口拒绝，排除。
TorchAO 原 block GEMM 是 Triton；实际候选为其原量化/codec 与成熟 ATen matrix 的组合，
不声称上游存在未发现的完整 CPU grouped kernel。

Q1、Q2 的来源阶段结束。Q3 没有解决 Apple 极小 scale 和 BF16 中间边界的新候选，
禁止用同程序重复测量、默认数值例外或 host fallback 解封；Apple 产品 FIX 为零。
Q4 转入下列 CPU 有限接入，原通用数值合同保持。

### 正常旅程、owner 与 API 冻结

普通用户继续 `cpu::create_context()` / `graph::cpu::create_context()`，直接准备既有
operation 或编译既有图；权重是 consumer 已提供的 encoded bytes，不增加模型加载器。
Compute native factor 只拥有成熟 CPU 运算组合与适配；Graph 继续拥有 region 边界、liveness
与运行槽；ATen 拥有数学调用、内部 allocator 与 CPU 库线程。没有 HYLE 手写 payload 数学循环。

主纵切为 GLM 5.3/Flash 与 DeepSeek V4 Pro/Flash 官方精度的完整 MoE、attention、状态反馈
配方。正交纵切为既有 block128 编码矩阵/分组专家 API 的 F32/BF16 输入、尾块、重复与非法 ID；
不因模型名称额外发明公开 API。直接依赖仍是当前 Native 的 Compute/Compute Ops、Mem、Error
等闭包，新增外部 CPU SDK 和 BSD-3-Clause 原语 provenance；Compute 不取得 Graph 依赖。

执行绑定固定函数和 shape，运行只处理变化 payload/ID。encoded expert 在索引后才展开；
共享和逐 slot 激活均保留真正的 A8 量化；K128/K32 的 F32 dot、activation scale、weight scale
及顺序累加保持。原始权重编码不改成 GGUF，也不提前将其改为 BF16 权重。

CPU 库有私有 allocation、选中权重展开、结果复制和进程线程 runtime，诚实声明
`managed`、private workspace/threads、同步 host 完成；不伪造 prepared-only、零热 allocation
或 caller executor 借入。不会调用全局 `set_num_threads` 去实现 Context 局部参数。
严格 replay 或无法证明的内部内存预算请求明确拒绝；默认合法 managed 路径的 OOM 转成
Completion 失败并回收库临时对象。输出失败时不可消费，无回滚承诺。
并发槽只共享不可变计划，各次 native Tensor/临时 allocation 独立；本批不设计第二套 Queue。

### 有限产品 FIX（已冻结，AUDITED 后实施）

| ID | 允许改动 | 验收 |
|---|---|---|
| CPU-O1 | 在现有 Native factor 组合固定原 TorchAO 分组 FP8 / rceil / FP4 codec 与 ATen matmul；激活量化为私有序列，公开接口继续使用 block linear、weight decode、encoded MatMul、grouped FP8/FP4、缺失的 group128 roundtrip | code/scale/存储和完整四画像的普通 Direct/Graph；不新增公开 quantize API，不恢复自研算子 |
| CPU-O2 | 将新组合的 managed replay、private memory/threads、host completion、OOM/非有限输入/越界 ID、资源 mapping/flush/close 绑定到现有运行协议；保持 NativePlan 布局，原路线成本声明不被改成 managed | 变化/重复/非法 ID、多个 replay、不同槽、保活、失败后重建、预算拒绝；原 ds4/普通 Native 回归 |
| CPU-O3 | Native owning CMake seam 接入版本固定的 CPU-only LibTorch SDK，Provider OFF 不探测；private includes/flags/link，拒绝 CUDA SDK；来源和 license、source/static/shared 闭包完整 | Windows/WSL C++ 调用、CPU-only 隔离、三交付；HYLE static 仍依赖 mature shared runtime，明确文档 |
| CPU-O4 | 原默认 Native fallback 仅承担当前未覆盖的原精度请求；普通入口、版本/实现 identity、资源/规则同构建能力投影 | 不让用户排序 Provider，不替换原有通过域的赢家；完整画像未 PASS 前账本不变 |
| CPU-O5 | 批量 ordinary 完整四画像消费及资源测试；冻结后一次适用新完整消费者 P1，更新用户能力表、账本及收口记录 | 完整 MoE/MLA或KDA/压缩/反馈链原门通过，全部适用正确性/三交付/P0/P1 后才裁决完成 |

SDK 与 borrowed 资产必须先完成独立依赖身份和交付审计，不能把研究用的部分 ZIP entry 资格
manifest 直接伪装成完整产品依赖锁。现已取得官方完整 Windows CPU SDK archive 的 hash；
其 CMake/version metadata 与 shared closure 在有限 FIX 内接入，库 runtime 为外部 CPU SDK，
静态 HYLE 消费方仍需要它，不能宣传完全静态进程。若需要新的长期 component identity，再依法 QUEUED→SCOPED，
不在本卡臆造空 target。既有 Native 接入不要求新的普通产品 façade。

### 预声明门、预算与停止

产品测试一次集中接线后对不可变 candidate 统一运行；重型串行、parallel 2。
新域：4 官方画像 × Direct/Graph 共 8 格、所有实际原子/完整片段、变化输入与实际反馈；
原域：Native/ds4 直接消费者、CPU Graph 组合/资源回归。受影响公开交付 source/static/shared，
Windows CPU-only 与 WSL CPU correctness，依赖/许可证/独立 install consumer/关闭。
Apple 只保留已知边界，不假跑新门。

P0 核对选中专家展开的 O(T*S*N*K)、量化 O(T*A*K)、复制/临时内存、完成/线程机制。
不可变候选的适用 P1 只比较完整同 Provider 程序与 HYLE 普通 Direct/Graph：相同数学、线程、
输出 ownership、完成边界和必要复制/转换。沿用统一仪器、A/A 与双顺序预检、5% material
margin、n=15；预检不合格不正式采样，不把库自身成本或机器停跑误判成 HYLE 算法 FAIL。
一次原生资格、一次集中产品验证及一次具名编译/API/正确性纠正；P1 失败先查成本结构，
只有本批可修复机制才准一次纠正，不不断换姿势加样本。

首次冻结时产品 gates 全部 `UNPROVEN`；新来源资格不代签产品 gates，当前结果见后面的实际批次。
数值、资源、构建隔离或许可不能保持时止于具体 BLOCKED/INCONCLUSIVE；不扩大算法范围。
本卡源码阶段统一 LOC 为 60,481 / 242 实现文件（无产品修改，不含测试）；接线后重新统计。

### 七问的当前裁决

1. 第三方：来源、revision、原始 BSD-3-Clause 与 CPU-only ABI 已固定；产品 SDK/license/lock 待 CPU-O3。
2. 完整性：原来拒绝的四个 CPU 官方画像是具体 consumer 缺口；原生投影可行，完整配方待 CPU-O5。
3. 复用：沿现有 Native/CPU backend、资源/Completion 和 Graph，不动已关闭 SIMD 或库线程治理。
4. 性能：成本与 managed 档位明确；本批尚无产品 P0/P1 PASS，也未宣称与其它成熟实现胜负。
5. 边界：仅 C++，不引入 Python runtime、模型 loader 或 Native FFI。
6. 交付：HYLE 三交付均是 C++，CPU SDK 作为私有成熟 runtime，待真实独立消费者证明。
7. 易用：普通入口与既有 op 保持；高级构建可裁剪，缺 SDK/严格 replay/不明内存上界明确拒绝。

## 第一批产品验证与一次定向纠正（2026-10-05）

不可变 candidate `0c4f636fb9c3b513f339a8c606c645e03781aab8` 已成功编译。
Windows Native owner 测试及三组公开头检查通过，包括真实 ATen OOM、恢复、重复/非法 ID、
并发槽、冷 metadata 配额与 managed 声明。CPU Direct 全套 157 项完成，153 PASS / 4 FAIL；
Graph 测试启动后为避免重复资源峰值而中止，不能计 PASS。此批不是完整画像验收。
原始输出保存在本卡对应研究证据的 `first-product-batch/`，完整账本仍为 154/20/2。

资源观察：Direct FP4 完整专家片段的进程峰值 working set 为 48,651,546,624 bytes，
此时本机可用物理内存约 1.6 GiB。源码确认：选中专家后，整块 FP4 解码生成 F32/int64/mask
Tensor 临时对象，随后所有 K-block 批量 matmul 再次物化布局，放大 O(T*S*N*K) 中间资源。
CPU-O1/O2 内的一次纠正：保留选中原 encoded bytes，按固定 K32/K128 顺序逐块调用同一
TorchAO codec 与 ATen matmul；激活量化、F32 scale 乘法/累加、BF16 输出边界不改。
不新增数学 kernel，不以减少实际 N/K/slots 验收，不把库私有内存改报已知硬预算。

三个 DeepSeek 失败首先来自 `expect_mature` 未纳入已 CLOSED 的原 ds4 group64/group32
roundtrip；修正测试只允许精确 operation/format 与现有 implementation 身份，不白名单全部
Native，自研 fallback 仍禁止。GLM Flash 的 MLA 最后一次反馈在一个 BF16 输出超原门：
actual -0.006866455078125 / expected -0.007080078125，门为 0.00019700196571648121。
新组合是独立 ATen multiply/add，原 fixture 默认绑定 CUDA FMA oracle；核对既有独立
unfused oracle 与具体中间边界，按源码固定 CPU policy，原完整片段数值门保持，不能事后
挑较有利答案。若该原始机制纠正后仍失败，保留数值 FAIL，停止此候选而不扩预算。

本轮重用未改变的 CMake 配置和已编译对象，只批量重编受影响 TU，串行复核受影响完整片段。
P1 尚未启动，三交付/WSL 尚未验收；不以来源 probe 或旧 GGUF P1 代签。

## 已完成的 Windows 批次与覆盖补验

产品固定候选 `3c22b7b9f230abc86e749225174d86b59c79881e`；
[普通入口与交付原始记录](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-official-torch-binding/README.md)。
分块 codec/matmul 后 Windows Native 106、Direct 157、Graph 173 项全部通过，另有三组公开头和
CPU executor 检查通过；0.5 秒采样观测的最大单进程 RSS 为 4,404,211,712 bytes（约4.1 GiB），
不将其宣传为严格 allocator 上界。GLM 使用源码预先确定的独立 F32 multiply/add oracle，
未改变原数值门。源码、静态、动态交付各跑两入口13项官方测试，共26项，均通过。
已验证安装搬迁、独立 include/link 闭包、borrowed licenses 与 DLL 邻置/来源；HYLE static
仍依赖外部 shared LibTorch。源码首次消费者漏链接已有词表 oracle companion TU，日志保留；
仅补齐消费者后通过，产品代码未改。

SDK 配置检查确认：OFF 加不存在 SDK root 不加载 Torch；ON 对不存在 root 在自身版本检查处
明确拒绝。初次组合选择器参数和换行消息断言的验证包装错误均保留，不伪装产品 FAIL，
未改公共治理/仪器。旧研究 SDK 与 probe 中逐项 SHA 相同的12份 DLL/lib 副本已删除，
释放559,220,140 bytes；正式 SDK、输入、原归档和证据保留。

红队范围对照发现继承的官方剩余测试只为 Flash 执行完整 MLA/NextN；非Flash GLM 5.3 还须
证明冷期原 FP8 KV 解码/拆分、实际缓存反馈、完整 NextN/MoE 与全词表 head。本次增加一条
对应消费者测试，仍在 CPU-O5 内，不修改产品算法/API。只补跑这个旅程的普通/三交付/WSL，
其余未变化13项及原基础模块结果按同产品/SDK/工具链身份复用，不重复全部批次。

WSL 正在同步同一 Windows 产品候选并使用官方 CPU-only Linux LibTorch 2.8.0 archive；
版本/CPU shared closure 已核对，但尚无 Linux 产品 PASS。正式新增完整片段 P1 尚未启动。
原154/20/2账本保持；没有把单个来源资格、未跑的补验、旧GGUF P1或Apple限制计成完成。
统一 Compute+Graph 实现为60,866行 /244文件，不含测试和第三方源码。

Apple 有界检查将 ds4 upstream HEAD `0aaea5a238fb41a35106a551e73c8409dfb751ac` 的
`metal/dsv4_kv.metal` 与已登记 `9139e2ae58a4` 比较，SHA完全相同；相关 FP4/Hadamard 原函数
没有提供区别于旧失败的新机制。`dsv4_misc.metal` 有其它变化，不能据此宣称FP4已修复。
没有重跑旧资格或打开Apple产品FIX；不外推全生态缺失。
