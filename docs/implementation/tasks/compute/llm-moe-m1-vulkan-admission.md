# M1-V：原版 ggml Vulkan 与条件 IREE 接入

## Closure card

- owner：`/root`；阶段：AUDITED；baseline：`0e9cf67dbb233132c081f711402ec9c2529d1c07`，
  Windows main，工作树干净。M1-G/R/C/A 已关闭各自有限范围。
- 当前 owner：`compute_provider_ggml_vulkan`、`compute_route_vulkan`、`graph_route_vulkan`。
  M1-G 只验 ggml 资源，M1-R 只验冷身份；本轮新的完整成熟 region / 外来原生程序消费者构成
  reopen trigger。共同 `compute_ggml_runtime` 保持 CLOSED，只有具体新缺陷才另行绑定。
  原 ggml CPU / oneDNN 的 BLOCKED 不变。
- 需求来自 [M0 接入清单](llm-moe-provider-integration-ledger.md) M1-V / Q-VK，以及
  [64 行映射](llm-moe-provider-map.json) 的 F02.4、F06、F10、F18、F19.4。
  不重新搜集模型，不扩大已有模型/数值合同。
- 主纵切：原版 ggml whole recurrent region 接普通 Direct/Graph，显式初态与输出/新状态，
  同 GPU 成熟片段组合、重复运行与两请求；Q-VK 另验证参数化整数 scan 和低精度标准 MatMul。
- 正交纵切：无模型标签的整数前缀/压缩权重线性片段，沿同一普通资源和完成路径；
  不以模型名称作为通用能力开关。
- scope：原版 ggml managed region 的必要 schema/多输出接缝、原生资源/完成/失败；
  IREE 条件资格先行，合格后才接其独立原生 Vulkan 程序及普通入口；对应三交付与文档。
- non-goals：全部冻结 CPU/CUDA/Metal/Vulkan 自研数学的扩张、研究、优化、比较或删除；
  SIMD 基础重审；手写 scan/排序/解码/shader，或用 MLIR/模板给自写数学改名；
  完整模型 consumer、M2/M3 全域批量映射、模型 loader、CPU 模型 fallback、公共治理重写。
- 依赖/方向：Vulkan backend 与锁定 ggml，现有 Mem/Error/Platform 资源和冷环境；Graph 单向
  消费 Compute 原生程序。IREE 若通过，原生 C runtime 是私有第三方边界，不新增 HYLE C 产品。
- tier：P0。核对编码/分配/转换/同步/常驻成本，不签 P1 或 SOTA；成熟路径完整消费性能归 M2/M3。
- 预算：源码/设计、Q-VK 准备及首次资格合计最多 4 小时，不与子资格叠加；开始于
  `2026-09-20 08:02:33 UTC`，截止 `2026-09-20 12:02:33 UTC`。本轮每根因最多两次纠正，
  每台设备一个串行资格 bundle；合格后的产品实现/预声明验收另记。预算到期必须裁决，不无限研究。
- 机器：Windows RTX 4070，Air/MoltenVK；WSL/Mini 仅适用 OFF/公共语义或已声明 portability。
  Windows 唯一产品真相，验证机只取可追溯前向副本；parallel2、重型串行、DLL 邻置、无 UAC。
- stop：有限接入和适用门齐全后 CLOSED。Q-VK 失败保留精确格式/feature/资源/交付反例，按
  原清单先核对具名 ncnn/MNN/上游替代，另定有限后续动作；不能删需求或默认转自写。
  **M5 开始前暂停汇报并等待用户明确继续**，M1–M4 继续。

## 条件资格 Q-VK

IREE pin `6ef606b6fa3e12bd2c12d36a07463069eb71185f`，Apache-2.0 WITH LLVM-exception；
ggml pin `456172ec733a135778adcd32d00e576a58232e45`，MIT。复用 M0 保存的源码入口与身份，
只查本卡问题。首先核对同 pin 编译工具与 C runtime 的实际交付形状，再做数值/原生资源资格。

1. 标准 I32 scan；标准 E4M3/E2M1/UE8 scale 转换和 MatMul，由原版 lowering 生成。
   使用独立逻辑值 oracle；压缩输入驻留、转换 workspace 有界，不展开常驻整模型。
2. Windows RTX 4070 与 Air/MoltenVK 上真实执行；显式 Vulkan 1.3 feature/limits，不能用
   device enumeration 或 copy 替代计算，也不能以 Windows 成功代签 MoltenVK。
3. 核对外来 VkDevice/queue/buffer 的 owner、可见性、完成和销毁；无真实共享桥时只允许
   明确计费的有界 GPU copy。GPU payload 不回落 CPU 计算。
4. 编译器只作构建工具，参数化制品与 C runtime 进入 source/static/shared SDK；普通 C++
   consumer 不安装 Python、不提供 MLIR、不运行编译器。版本、资源、许可须可追溯。

首次 Q-VK 已运行，IREE 单独承担 packed FP4 的候选失格；详见下方实测记录。资格通过才决定
新增 IREE factor，先登记 QUEUED，再依协议 SCOPED/AUDITED；当前没有为该候选建立空 target 或
Provider。后续同设备组合资格未通过，不能把 Windows 部分数值通过写成整项 PASS。

## 原 SCOPED 审计边界

下表是 SCOPED 的审计边界，尚不是产品修改授权；实际 API、owner/lifetime、冷/热成本与有限
FIX 在 AUDITED 前固定。原版 whole region 不为逐 kernel 调度而拆开。

| 项 | 正常消费与审计问题 | 裁决出口 |
|---|---|---|
| V1 | ggml 原版 recurrent/multi-output 的类型、head/state 和资源限制；现有 managed schema 到普通入口 | BORROW 原版 fused 或已有组合；FIX 必要 descriptor/region glue，禁止补写数学 |
| V2 | Q-VK 的原版 scan/低精度、离线制品与 VkDevice/resource seam | 合格后新 factor；失败给精确成熟替代动作，不减少需求 |
| V3 | Direct/Graph 跨 native region 的资源/可见性/失败/旧结果、两个 Invocation、close/drain | 复用本路线原生 owner；明确命令复用与 managed 编码差异，不新增通用 Queue |
| V4 | 冷期规则和支持诊断；模型/配置/phase、GPU/driver/工具/Provider identity | 仅匹配绑定，执行不搜索/计时；普通用户不排 Provider |
| V5 | 模块拥有的 CMake、资源/许可、OFF 隔离、三交付、P0 与用户指南 | 有限接入验收；全 64×4 覆盖继续留在 M2/M3 |

## 预声明验证与证据边界

- SCOPED/AUDITED 文档阶段：selector、closure（明确 previous ref）、docs、diff；不构建、不跑 P1。
- Q-VK：保存准备/编译/运行命令、精确 pin/hash、机器事实、独立 oracle、数值及资源/制品结论；
  每设备一个串行 bundle，失败原样保留。研究探针不计入产品覆盖。
- 产品固定后：selector 所选受影响 factor/头与真实消费者；普通 Direct/Graph 的独立 oracle、
  非零状态、repeat/两请求/旧结果、shape/type/feature 拒绝。
- 资源：外来 owner 保活、容量/对齐/布局/设备检查、受控 OOM/部分构造回滚、提交/完成失败、
  accepted-prefix drain、cancel/close。host ASan/UBSan 只签适用 host，不代签 Vulkan driver。
- Windows/Air source/static/shared 独立消费者、prefix 搬移、producer 隐藏，原生资源/许可证/
  私有符号完整；Vulkan OFF 的适用 Windows/WSL/Mac 语义/配置门按改动选择。
- 一次有界 P0/候选复核，统一 Compute+Graph LOC；日志归档后清理本卡具名 build/SDK，保留
  共享缓存和本地模型。所有未运行适用项保持 UNPROVEN，不以 metadata 提交计产品完成。

启动时 Compute+Graph **51,892 行 / 222 文件**（统一 self-check，排除测试/vendor，含冻结存量），
模型新增完整覆盖 **0/256**。M1-A 的验收与限制保留在其 terminal record，不因本卡重开而失效。

## 首次资格与有限重新定界（2026-09-20）

[首次资格收据](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/qvk-first-candidate.json)
及同目录 raw archive 保留全部编译失败、数值失败、原输入与独立 oracle。Windows 首次 I32
inclusive/exclusive scan、E4M3+F32-scale MatMul 逐值通过，UE8 出现 230 个 NaN。
原版 LLVM `arith-expand` 的 UE8 专用转换使同一组 UE8 数据逐值通过；只使用上游工具配置，
没有 HYLE decoder。IREE 自身没有注册该 pass，且解析错误仍返回 0；该产物未当成有效纠正运行。

随后 packed FP4 数值失败（最大绝对误差 66.6875）。原版 LLVM F4 转换、再加原版 IREE
narrow-type 预处理两次纠正均在 `vector<2xi8> -> vector<4xi4>` 合法化失败；这条路径停止。
原有精确相等门、逻辑值与打包输入均未放宽。其余样例、Mac、外来资源和三交付未运行项
不签 PASS，首次聚合资格为 INCONCLUSIVE，packed-FP4 候选 REJECT。

按预声明出口检查 ncnn/MNN 的锁定源码：已检查的 Vulkan cast/MatMul 接缝分别是
FP32/FP16 与整数 INT8，尚未证明能接本合同 E4M3/E2M1+UE8；不据此声称所有后端都不支持。
ggml 同 pin 已有原版 Vulkan MXFP4，因此下一项限定为 **IREE scan/FP8 + ggml MXFP4**
的同设备组合资格。各自承担其成熟实现；F4/UE8 原需求保留，不能靠删除失败样例关闭 M1。
仍受原 **12:02:33 UTC** 截止约束，不另开四小时预算。新组合目前 UNPROVEN。

## AUDITED：当前可实施的有限 FIX

先实施与 Q-VK 独立的原版 recurrent 接入。下面只授权既有三 owner；IREE factor、其普通入口
与量化组合产品改动仍须先完成资格并补齐自身 owner/有限 FIX。M1-V 尚未完成，不能因这一纵切
单独通过就关闭整个包。共同语义和 ggml runtime owner 保持 CLOSED。

| FIX | 裁决与正常使用路径 | 资源、失败和成本合同 |
|---|---|---|
| V1-G | BORROW `ggml_gated_delta_net`，`vulkan::prepare(context, GatedDeltaNet{}, q, k, v, g, beta, state)` 和普通 `graph::vulkan::compile` 均直接可达；无模型标签同样可用 | 沿既有六输入/两输出公共语义；dense F32、等头、D16/32/64/128；固定初态与独立新状态，不原位修改输入 |
| V3-G | FIX managed region 的六输入边界、原版 fused 输出/状态的 view + cont，以及 metadata 容量；KEEP NativeDevice/NativeGraph/NativeWork 和路线 Completion | 准备期校验原版索引、合并输出大小、descriptor/dispatch/device 限制；沿既有 binding owner、两 invocation、accepted-prefix drain、失败后退休与结果 lease |
| V4-G | FIX 冷期原版 recurrent 支持规则和 Graph 报告；KEEP M1-R workload/environment identity | prepare 只按 dtype/shape/硬件限制绑定；submit 不搜索、不计时、不重建数学图；不扩大公共对象或万能 Queue |
| V5-G | FIX 本纵切的独立 double oracle、Direct/Graph、repeat/旧结果/非法域/资源失败、Windows/Air 三交付和使用文档 | 适用 selector、host 插桩/OFF 门按实际改动选择；P0 必须计入已有输入对齐 GPU copy、原版编码、fused 输出发布复制，不冒称零边界成本或 P1 |

普通输出与新状态沿现有 owning Tensor 返回；原版 fused region 内部仍由 ggml 调度。现有输入
offset 适配器可导致六个边界 GPU copy，输出拆成两个公开 owning port 也有原版 cont 成本；
本次不为隐瞒这笔成本而改变公共 binding/layout。资源在 invocation 冷期规划，旧结果持有时
拒绝覆盖该 slot。异常路径复用已有 drain/close，不增加第二套执行循环。

资格探针只使用第三方原版运算；MXFP4 探针同样只证明候选，不能代签普通产品或 256 格。
当前公开 GatedDeltaNet 的 shape/错误推导已在 M1-A 收口，因此不重开 `compute_ops`。

## 进行中取证：普通 recurrent 与参数化原生程序

[进度收据](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/native-progress.json)
和同目录 raw archive 保留首次失败、逐次纠正、机器/输入/制品身份与原始结果。
产品候选 `db770b6fc1e46aa8d8aa86c1102d50acf26eb49f` 已实现 V1-G/V3-G/V4-G 的
普通 Direct/Graph 原版 GatedDeltaNet；Windows selected 命令和 Air 三 factor/头测试通过。
MXFP4 两组原始逻辑样例在两台机器各 repeat3 通过；测试输入改用独立持久 buffer，避免被
原版 gallocr 的临时区复用。产品数学未改变，也未放宽 oracle。

IREE 的 scan/FP8 固定八例和增加形状变化后的十六例均在 RTX4070、Air 逐值相等通过。
参数化 FP8 的半仿射分块缩放不能直接融合进本 pin 的动态 MatMul；两次有限纠正采用上游
`util.optimization_barrier` 与 `flow.tensor.tie_shape`，没有修改数学或编译器源码。
代价是该 region 的 F32 activation/weight workspace 与明确分派边界，不能冒称原生 FP8 GEMM
或整模型压缩格式已验收；原压缩输入保持驻留，不展开全部模型。

同一组离线制品也由真实 C++ static/shared 消费者在两台机器通过：外来 VkDevice/queue、
有界 GPU copy、异步 ready/done、两请求同时在途 repeat3、已接受请求的依赖失败传播、
输入不变、VM context 释放后旧结果保留、runtime 不销毁外来设备。运行消费者无需
Python/编译器。一个原版 HAL device group/module 配两个 VM context；不能为同一设备各建
独立 runtime session，否则本 pin 拒绝重复 queue frontier。

这些是**基础数值/资源可行性证据**，当前聚合接入仍为 INCONCLUSIVE，M1-V 不关闭：

- Air 只有一个 compute queue。探针的外来队列访问发生在全部 IREE completion 之后；普通
  HYLE/ggml/IREE 混合提交必须补齐共享 host 同步，不能把两个互不相关的 mutex 当成保护。
- I32 正常样例尚未证明公开 `Scan` 的 checked-overflow 合同，exclusive 的最终总和同样
  必须检查；不能用 wrapping I32 代替。普通低精度入口与 Graph 接线也尚未完成。
- IREE factor 及必要原生桥修改先完成自身登记/审计与有限 FIX，再写产品代码；既有冻结
  shader/算子不得借接入扩张。产品三交付搬移、适用 host 插桩/OFF 与最终 P0 尚未签字。

调查截止仍为 **12:02:33 UTC**，不因新证据重置预算。统一脚本当前统计 Compute+Graph
**51,953 行 / 222 文件**，排除测试/vendor；相对本包启动增加 61 行接线实现。
新增完整模型覆盖仍为 **0/256**；M5 开始前的用户停止点不变。

## 条件资格收口：checked Scan 与单队列异步续执行

[限定资格收据](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/checked-queue-qualification.json)
及同目录 raw archive 保存 Windows / Air 的输入、原版 recipe、独立 oracle、编译/执行日志和
资源补丁首次失败。下面补齐上一节的资格缺口，不改变首次 packed-FP4 失格，也不关闭产品包。

- checked I32 inclusive / exclusive 共 12 例在两台机器逐项通过，包含正/负中间前缀溢出后抵消、
  exclusive 仅最终总和溢出、单元素极值和变化形状。使用原版 I64 scan、标准转换/比较/归约，
  没有手写 scan 算法或 shader。首次发现原版 exclusive 的 accumulator 不含最后输入，增加
  原版 reduce 检查总和，一次纠正后通过。shaderInt64 必须实际启用；私有截断结果不能在错误时
  发布到普通输出，后者继续列为产品验收。
- 原版 IREE 计算 → 原生 GPU 边界 copy → 原版 IREE 计算，两条同时在途链路各 repeat3，
  Windows / Air 的 C++ static/shared 消费者全部通过。调用方只等最终状态，自动推进由原版
  proactor semaphore-wait 驱动；前段依赖失败会跳过后段。Air 的 compute queue 实数为 1。
- 四文件私有资源 overlay 给原版 QueueSubmit2/BindSparse 增加同步 owner callback，保留其
  原内锁与默认路径。产品将通过既有 NativeDevice::with_queue 复用 ggml 的锁与 NVIDIA 锁顺序，
  不能把两个独立 mutex 当作共用保护；不在持有 native queue 锁时调用 IREE。数学和编译器未改。
  opaque VkQueue 声明遗漏和 offset-zero HAL resource header 布局失误均保留原始失败及纠正。
- 原版每队列 completion thread 与 proactor runner 都是实际资源成本；跨片段通知、GPU copy、
  原版 VM/临时资源分配与编码同样计费，不承诺零开销。不增加 HYLE 通用执行器或轮询线程。

**Q-VK 分工的数值和外来队列续执行资格通过；普通产品接入仍为 INCONCLUSIVE。**
当前实际产品仍只完成原版 recurrent 纵切，未把资格程序计作实现或 256 格覆盖。
`compute_provider_iree_vulkan` 现在只登记 QUEUED 身份，没有空 target；先完成自身 SCOPED / AUDITED，
再接普通 Scan / 低精度 MatMul、实际 ggml owner 的 feature/queue/lifetime 桥及 Direct/Graph。
具名压缩格式工厂由共同语义 owner 单独按新消费者重开，不扩张任何自研数学。

这部分资格调查停止，后续按已验证接法实施；必要有限 API/资源细化仍受原 **12:02:33 UTC**
总调查截止约束，不重开资格预算。产品 Windows/Air 三交付搬移、host 插桩/OFF、失败不写输出、
实际混合片段和最终 P0 仍待完成。Compute+Graph 仍为 **51,953 行 / 222 文件**（排除测试/vendor）。

本次 QUEUED 登记只增加身份，不改变已实现 target/依赖/头。selector 的完整计划保留；其因中央
manifest 变化选出的产品门仍是 SELECTED，留给固定后的产品候选，未写 PASS。本阶段执行
closure（previous ref `2e4d8ecb`）、docs 和 diff；没有为状态/证据登记重建产品或运行 P1。

IREE 普通接入与共同格式的新 owner 已转入[独立产品卡](llm-moe-m1-vulkan-iree-product.md)
AUDITED（经独立SCOPED检查点）；本卡已有三 owner 保持 AUDITED。IREE 相关产品改动严格限于
该卡 I1–I7 已冻结的共同格式、实际native feature/queue/quota/异步片段适配和普通入口，禁止
扩张自研数学。本卡原版 recurrent 已完成的范围与父包未决验收不变。

新增产品检查点 `d5cd5568` 已通过普通MXFP4 Direct/Graph的Windows shared / Air static四factor
及头验收，详见[独立产品卡](llm-moe-m1-vulkan-iree-product.md)和
[产品收据](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/mxfp4-product.json)。
共同具名格式已实现；FP8/checked Scan的IREE普通接线和整包未决门继续保留，不能把格式或原生资格
代签产品。Compute+Graph **52,110 行 / 224 文件**，新增完整覆盖仍 **0/256**；M1-V未关闭。

IREE 原生产品候选 `dee6ad64` 已完成独立 runtime/制品、共享queue/feature、实际显存配额和
原版异步推进，见[原生检查点](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/iree-native-checkpoint.json)。
Windows shared、Air static Release及Air host ASan各通过7项native与6项source验证；ASan发现的
原版POSIX callback释放后写经有限资源overlay纠正。普通Scan/FP8、混合链和整包交付/剩余门仍待
验收，不能以native factor代签普通产品。当前统一LOC为 **53,003行 / 230文件**，完整覆盖仍0/256。

普通产品候选 `c6386628` 已完成 IREE Scan/FP8 Direct/Graph、自动混合续执行和取消/失败寿命；
Windows Release、Air Release/host ASan各10项CTest通过。Windows原版C runtime实际插桩后，
取消测试的配额纠正已单独通过，其余用例复用同产品证据；原失败/skip保留，见
[普通产品检查点](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/iree-ordinary-checkpoint.json)
及[子卡](llm-moe-m1-vulkan-iree-product.md)。IOCP与POSIX均修正final callback后访问操作内存的
资源缺陷，没有扩张原版或冻结数学。三交付和剩余selected/host/OFF仍待验收，M1-V保持AUDITED。
统一Compute+Graph **53,484行 / 232文件**，新增完整覆盖仍0/256。
