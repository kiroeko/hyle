# M1-V：原版 ggml Vulkan 与条件 IREE 接入

## Closure card

- owner：`/root`；阶段：SCOPED；baseline：`0e9cf67dbb233132c081f711402ec9c2529d1c07`，
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

当前 Q-VK 尚未运行，不能签 PASS。资格通过才决定新增 IREE factor，先登记 QUEUED，再依协议
SCOPED/AUDITED；当前没有为该候选建立空 target 或 Provider。

## 待审计的有限事项

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
