# M2 组合路由资格表达

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- component / owner：`compute`、`compute_provider_cuda`；`/root`。
- baseline commit：`eb4489b94533bf820d6c801471fa0332eb9a7ff0`。
- trigger 与证据：用户要求路由能够把平台、四条原生路线、硬件/架构、驱动与
  Toolkit/SDK、Provider 版本、模型族/模型名/版本配置和量化级别作任意组合；部分事实由构建或
  Context 固定，剩余事实只允许在冷准备阶段低成本匹配。现有
  [`compute-routes.md`](../../../architecture/compute-routes.md#13-组合路由键与两级绑定) 已定义该
  合同，但产品只有模型谓词和各 Provider 分散的手写条件，没有可复用的稀疏规则值、版本范围、
  未知值语义或拒绝诊断。前一张[模型身份补充卡](m2-b2-route-identity.md)保存已经完成的
  family/name/revision/configuration 迁移事实，本卡不回写该历史记录。
- 正常 consumer 任务：Provider 作者声明一组随实现版本化的静态资格规则；Context 创建时缓存原生
  环境事实，`propose/compile` 将真实 workload、operation、shape、格式和资源与规则合取，绑定一个
  artifact；普通用户仍只选择路线并提供已知模型事实。
- 主产品纵切：`Windows + CUDA + SM89 + driver/runtime/toolkit 区间 + ggml/CCCL Provider 版本 +
  qwen3_5/qwen3.5-2b + decode + Q4-S128` 命中一条静态规则，任何具名字段未知或不符均拒绝，再由
  同路线成熟基线接管或明确失败。
- 正交反偏置纵切：`macOS + Apple + M4 + SDK/OS/MLX 版本 + 通用模型 + BF16` 以及
  `Linux/Windows + CPU + ISA 属性 + oneDNN/ggml 版本 + Q8-B32` 使用同一共同组合语义，但保留各
  路线自己的原生属性和执行 owner。
- 跨模块 public journey 与每段 owner：构建/CMake 删除未交付路线；路线 Context 捕获设备与软件栈；
  `compute` 持有稀疏规则、模型/量化合取和冷匹配；Provider 持有规则次序与实现；Graph 继续转发同一
  `WorkloadContext` 和真实 DAG，不新增第二套路由器；prepared artifact/Program 持有最终绑定结果。
- scope：新增 allocation-free 的共同 route-facts/rule/match 合同；固定 Windows/Linux/macOS 与
  CPU/CUDA/Apple/Vulkan 身份；支持 namespaced 原生数值/字符串属性、数值闭区间、Provider 身份/
  版本、分层模型、phase/region、Q4/Q8 mode/group/物理格式的稀疏合取；给出 Direct/Graph 冷绑定
  consumer，并迁移至少一条已有真实版本/硬件资格判断。
- non-goals：不建立进程全局 Provider registry、用户可配优先级、运行时竞速/benchmark、通用硬件
  万能对象、模型 loader、跨进程 plan cache、热 submit 查表，亦不改变任何数学实现或扩张算子。
- 直接依赖 / 直接消费者：沿用 `compute` 对 `core/error/mem/compute_ops` 的现有依赖；直接消费者为
  自定义 Provider、内置 Provider 的 `propose`、Graph region proposal。原生属性由各路线已存在的
  Environment/Platform/SIMD 权威来源填入，不反向复制其探测。
- 性能 tier 与原因：`P0`；匹配只发生在 Context/prepare/compile 冷路径，规则和事实使用调用方静态或
  Context 持有的 span，不分配；prepared submit 对象布局和回调不增加字段与分支。
- 适用机器 / 仪器 / material margin：Windows canonical；共同 matcher 和 Direct/Graph consumer
  单测、公开头闭包、selector 选择的 source/static/shared 门。无热算法变化，P1 不适用。
- 调查与纠正预算：40 分钟 API/consumer 审计，最多两次 API 形状纠正；实现和预声明验证另计。
- 预声明验证：新增共同 matcher 边界测试；Direct/Graph 冷绑定一次、重复 replay 不再匹配；迁移的
  真实 Provider 定向测试；`validation-select.py`；`check-module-closure.py`、`check-docs.py`、
  `git diff --check`。
- stop / blocker 条件：若表达需要在 submit 中读取模型字符串/驱动、要求统一四路线 native handle，
  或需要用户排序 Provider/提供 timing profile，则停止并重新定界。

## SCOPED 审计清单

1. 逐项核对已有 `BackendId`、`ProviderIdentity`、四路线 Environment、`WorkloadContext`、
   `Operation` 与 `StorageFormatId`，不复制权威事实。
2. 对比固定大结构与 namespaced 稀疏属性；验证后者能保留路线私有事实并容纳未来硬件条件。
3. 冻结 wildcard、unknown、重复属性、非法范围、模型层级、版本编码和量化物理格式语义。
4. 明确规则匹配与候选优先级的边界：共同层只回答资格与拒绝轴，Provider 的版本化顺序决定窄规则和
   成熟基线的选择。
5. 核对 Direct 与 Graph 都在 cold proposal 使用同一请求事实，prepared replay 不保留规则解释器。

## 能力与形状审计

| 能力族 | 层级 | 权威事实与生命周期 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|
| build/platform/route | 原子机制 | 平台是编译常量；CPU/CUDA/Apple/Vulkan 是所选 Context 的冷事实 | 文档有合同，代码无共同值 | PARTIAL | FIX |
| 原生硬件/软件属性 | 原子机制 | Context/Provider 已缓存 CUDA/Apple/Vulkan Environment；CPU ISA 仍由 Platform/SIMD 拥有 | 四路线字段异构 | ROUTED | KEEP + FIX 稀疏视图 |
| Provider 身份/版本 | 原子机制 | `ProviderIdentity` 随实现静态存在 | Direct/Graph 都可见 | COMPLETE | KEEP |
| 模型/阶段/region | 原子机制 | `WorkloadContext` 冷持有并随 artifact 保活 | 四路线已贯通 | COMPLETE | KEEP + FIX 部分规则 |
| operation/量化/物理格式 | 原子机制 | `Operation`、`TensorDesc`、`StorageFormatId` 是唯一事实 | Provider 已逐项手写 | PARTIAL | FIX 共同谓词 |
| 规则次序/实现选择 | 应用策略 | Provider 版本化代码拥有；共同层不排序、不计时 | 各 Provider | ROUTED | KEEP |
| prepared replay | 复合功能 | 已绑定 artifact/Program 执行，不携带规则解释 | 四路线 | COMPLETE | KEEP |

审计了 CUDA `Environment` 的 SM/driver/runtime/build Toolkit、Apple Environment 的 device/architecture/
OS/MLX/ggml/Metal compiler/SDK/deployment、Vulkan Environment 的 vendor/device/driver/API/subgroup/features，
以及 CPU 的 `BackendId` 和 Platform/SIMD ISA owner。不存在一种不丢信息的固定硬件总结构；因此共同层只
定义稳定平台/路线字段和 namespaced 稀疏属性。属性 ID 决定语义，数值编码只在同时限定路线的规则内
比较；Provider 私有新事实可以增加自己的 ID，无需改共同 ABI。

规则值由只读 span 组成，调用方拥有其存储。静态规则可为 `constexpr std::array`；Context 原生事实可
在创建时放入固定数组。匹配器不分配，返回首个拒绝轴和可选属性 ID。规则没有写出的字段是通配；
写出的属性缺失、类型不符或值不符均拒绝；重复 fact/predicate、非法区间和断层模型规则是非法规则，
不得降级成 wildcard。数值范围是闭区间；字符串只支持字节精确匹配，避免错误解释供应商版本字符串。

模型规则允许 family、family+name、再选择性限定 revision/configuration；每个具名字段都要求实际身份
存在且相等。量化规则直接检查真实 `QuantizedMatMul` mode/group、activation/weight/output dtype 和
weight/scale `StorageFormatId`；`ANY/NONE/QUANTIZED` 明确区分，不从模型名推断格式。普通 MatMul 的
FP8/BF16/FP16 可由 dtype/format 条件表达，不伪装成 Q4/Q8 operation。

### 有限改动集（AUDITED）

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| RQ1 | 平台/四路线没有共同稳定值 | 新增 `RoutePlatform`、`NativeRoute` 和当前构建平台常量 | 四路线值独立于 CMake 开关；未知不命中具名规则 |
| RQ2 | 原生条件手写且版本范围语义不一 | 新增 namespaced unsigned/text fact 与 exact/range predicate | 任意稀疏合取、缺失/错型/重复/非法范围确定拒绝 |
| RQ3 | 模型和量化只能由每个 Provider 重写 | 新增 `RouteRule` 对 Provider、模型、phase/region、dtype、Q4/Q8/group/format 的共同匹配 | 主纵切全命中；逐轴反例各自给出拒绝轴 |
| RQ4 | Direct/Graph 是否只冷匹配缺少共同证明 | 自定义 Provider 与 region fixture 使用同一 matcher | prepare/compile 计数一次；三次 replay 的 match/propose 计数不变 |
| RQ5 | CCCL 的 SM89/CUDA 13.3/上游版本资格是散布的布尔式 | 只迁移这条既有真实资格到静态 rule + Context 固定 facts | 支持域不扩大；不合版本/SM 仍拒绝；submit 无 matcher |
| RQ6 | 架构文档仍称“不要求通用规则引擎”易被误解为继续手写 | 说明共同 matcher 只统一合取语义，不拥有发现、优先级或执行 | 普通用户 API 和四路线 native owner 不变 |

## API / usability freeze

- 普通 C++ 调用草图：Provider 在静态存储声明 `RouteRule` 与 predicates；Context 创建时构造原生
  facts；`propose` 用真实 request 构造借用的 `RouteMatchInput` 并调用一次 `match_route_rule`；命中后
  仍按原 Provider API 返回 proposal/prepare token。
- ownership / lifetime：rule/fact spans 只借用到 matcher 返回；matcher 与 prepared artifact 都不保留；
  `WorkloadContext` 的模型 owner 仍由现有 proposal/artifact 保活。
- failure：非法 rule/facts 返回 `kInvalidRule`/`kInvalidFacts`，资格不符返回具体 mismatch；Provider 对外
  仍以 `Unimplemented` 安全 decline，不把规则诊断变成执行错误。
- concurrency：matcher 是纯只读函数；同一静态 rule 和 Context facts 可并发匹配。
- 常态成本：cold O(predicate count + 固定字段)，无 allocation/lock/atomic/syscall；submit 零字段、零
  调用。规则数量和优先级仍由 Provider 的有限版本化代码控制。
- 默认与误用：`kUnknown`/空 ID/空字符串在 rule 中是 wildcard，在 fact 中是 unknown；specified rule
  永不被 unknown 满足。重复属性和跨类型同 ID 拒绝。
- 交付：新头和实现归 `compute`，不增加第三方依赖、feature target 或 native SDK 暴露；CCCL 迁移仍
  在既有 CUDA 可选 factor 内。

## AUDITED 决定

冻结 RQ1--RQ6，状态进入 `AUDITED`。共同 matcher 是资格谓词，不是中央 Provider registry 或优先级
引擎；它统一“一个规则是任意字段子集的合取”这件事，同时让四路线继续拥有原生 Environment、资源、
程序和完成语义。除 CCCL 既有资格表达迁移外不改任何 Provider 支持域、数学、shape 或默认赢家。
本专题期间继续暂停 M2-B2 的 Provider 算子接线和候选验证。
