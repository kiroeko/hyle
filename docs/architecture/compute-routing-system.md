# Compute / Graph 冷路由系统

本文定义 CPU、CUDA、Apple MLX/Metal、Vulkan 四条路线共同遵守的冷路由合同。它描述长期产品形状，
不证明某条规则或某个模型已经交付；当前实现状态仍以
[module-status](../implementation/module-status.json)及其活动施工卡为准。

## 1. 要解决的问题

应用先选择一条计算路线，再提交一个 Direct operation 或 Graph DAG。HYLE 必须在不运行候选、不计时、
不让普通用户排列 Provider 的前提下，依据已验证的条件绑定具体实现。条件可以同时包含：

- 构建中实际交付的路线和 Provider；
- OS、CPU ISA、GPU vendor/device/架构、SM、feature 与 limits；
- driver、runtime、toolkit/SDK 和 Provider 版本；
- 模型 family/name/revision/configuration 与固定模型制品；
- prefill/decode、versioned region 语义；
- 实际 operation、参数、全部 tensor 端口、dtype、物理格式和 shape；
- Graph 的真实 DAG、状态/effect、canonical boundary、资源与完成边界。

[目标模型审计](../implementation/tasks/compute/llm-moe-model-weight-operator-audit.md)证明，同一个部署文件
会按 tensor role 混合 BF16/F32、FP8、MXFP4、Q8/Q6/Q5/Q4/Q3/IQ 等格式。文件名里的 Q4/Q8 不是
整个模型的数值合同。因此路由不能把模型压成一个全局量化枚举，也不能只检查固定的前三个端口。

本系统不是模型 loader、推理 engine、运行时 autotuner、跨进程 plan cache 或跨路线万能 executor。
四路线继续拥有各自的 Context、资源、编译、Program、run state、同步和完成方式。

## 2. 三类事实与一个结果

路由只组合三类只读事实，输出一个已绑定制品：

| 事实 | owner | 建立时机 | 生命周期 |
|---|---|---|---|
| `RouteEnvironment` | 路线 Context | Context 创建 | Context 及其制品存活期内不可变 |
| `ModelArtifactProfile` / workload | consumer，Compute 保活 | 模型制品准备 | 被所有相关 prepared artifact 共同持有 |
| `RouteSubject` | Direct request 或 Graph compile | prepare/compile | 只借用到冷匹配和 Provider 资格化结束 |
| 已绑定 artifact | 入选路线/Provider | prepare/compile | submit 只使用它，不再访问前三项的规则表示 |

### 2.1 `RouteEnvironment`

构建系统先裁剪未交付的路线和 Provider。每个路线 Context 再一次性捕获本路线的环境：platform、
native route、device、架构/SM/ISA、features/limits、driver/runtime/toolkit/SDK，以及 Provider 的稳定
身份和版本。路线原生代码负责取得事实；共同层不重新探测设备，也不把 CUDA 版本解释成 Vulkan 版本。

事实采用带 namespace/version 的 typed ID。值至少支持 unsigned exact/range、text exact 和 version
range；未知表示事实不存在。显式数值零仍是已知值。规则要求的事实未知时必须拒绝，不能当 wildcard。

### 2.2 `ModelArtifactProfile`

模型画像是一个 immutable、owned、可复制的冷身份。它至少包含：

```text
model publisher / repository / family / name
fixed model revision
canonical configuration digest
artifact repository / fixed artifact revision / variant
artifact manifest digest / container-format family
```

官方 BF16 safetensors 与 Unsloth `UD-Q4_K_XL` 即使指向同一逻辑模型，也具有不同制品画像。画像不保存
模型权重，不解析 GGUF/safetensors，也不声明所有 tensor 都是某个 dtype。manifest digest 用于绑定
已经由 loader/engine 核对过的不可变制品；真实 tensor dtype/format/shape 仍从 subject 和 Graph 取得。

空画像表示普通、模型无关调用。它能走通用成熟基线，但不能命中任何具名模型或制品 predicate。
字符串和 digest 在创建时一次性复制到 allocator-backed owner；复制 handle 共享 owner，并发只读安全。

workload 另外携带 `prefill/decode/unspecified` 和 versioned region semantic ID。region ID 是 consumer 与
Provider 共同约定的优化意图，不代替实际 operation/DAG 校验。

### 2.3 `RouteSubject`

Direct subject 由真实 `Operation`、全部 input/output `TensorDesc` 和 shape domain 组成。Graph subject
由完整 Graph、候选 node set、canonical inputs/outputs、state/effects、常量和资源要求组成。

共同 selector 只表达可稳定比较的事实：

- operation kind 及由该 operation schema 规范导出的 typed semantic facts；
- 任意 input/output 端口的方向和索引；
- 端口 dtype、`StorageFormatId`、rank；
- 指定 axis 的 exact/range/multiple-of shape 条件；
- 模型制品画像、phase、region 和环境 predicates。

端口索引越界、descriptor 无效或规则要求的 semantic fact 不存在，都属于不匹配。duplicate port/fact
predicate、非法范围、零整除数和同一 axis 的矛盾条件属于非法规则。

量化不再有全局特殊标签。Q4/Q8/MXFP4/FP8 的资格由实际 operation 语义（mode、group、axis、
accumulation 等）和实际 codes/scales/zero-point tensor 端口共同证明；物理 layout 由
`StorageFormatId` 证明。Provider 仍须校验完整数值、资源和 shape 合同。

复杂 Graph region 不被拍平成第二套路由 IR。共同层只检查 workload 与 canonical boundary；Provider
直接查看真实 DAG、内部 tensor、状态和 effects，返回稳定 implementation key。模型专用融合必须同时
满足 exact profile/region 条件和 Provider 的真实 DAG 检查。

## 3. 规则与确定性选择

一条 `RoutePolicyEntry` 是一个 selector、一项目标 implementation 和一个 HYLE 维护的 preference。
selector 是稀疏合取；未写的轴是 wildcard，写出的轴必须有真实相等或范围证据。目标由稳定
`ProviderId + ImplementationKey` 标识，不使用地址、注册顺序或临时 proposal token。

普通用户不构造 policy，也不提供 Provider 顺序。每条路线随实现交付 versioned decision table；规则
来源是 HYLE 开发阶段完成的正确性、资源和完整成本验证。表的版本进入受影响的 plan identity。

冷选择顺序固定为：

1. build envelope 移除未编译、未交付的路线和 Provider；
2. Context 创建时按 immutable environment 裁剪不可能命中的行；
3. prepare/compile 验证 workload、operation/Graph、ports 和资源事实；
4. 对匹配行调用目标 Provider 的 hard qualification；Provider 必须核对真实 operation/DAG；
5. 在 hard-eligible candidates 中应用 HYLE 固定 preference，Graph 还以完整 plan 的边界 copy、sync、
   dispatch、workspace 和融合成本作既有有界规划；
6. 只对最终选择创建重型 artifact，并把 rule/implementation identity 写入冷诊断；
7. submit 调用固定 artifact，不解释 policy、不枚举 Provider、不查设备、不计时。

preference 是 HYLE 产品数据，不是“规则越具体自动越快”。两个同时匹配且同 preference、但目标不同的
候选构成歧义；表的静态检查能证明时在构建/测试期拒绝，否则在 prepare 时返回结构化错误。注册顺序
不能破除歧义。

Provider 在接纳工作前返回 `unsupported` 或明确的 resource-budget decline，才允许尝试 decision table
中下一条已声明的成熟 fallback。无 fallback 时明确拒绝。创建 artifact、提交或接纳工作后的任何错误
都是运行错误，不能静默换实现重算。

## 4. Direct 与 Graph 的自然使用路径

画像由模型 loader/engine 根据它已经验证的文件清单创建；HYLE 不读取模型文件：

```cpp
auto profile = hyle::compute::ModelArtifactProfile::create({
    .model_repository = "Qwen/Qwen3.5-2B",
    .model_family = "qwen3_5",
    .model_name = "qwen3.5-2b",
    .model_revision = "15852e8c...",
    .configuration_digest = "...64 hex...",
    .artifact_repository = "unsloth/Qwen3.5-2B-GGUF",
    .artifact_revision = "f6d5376b...",
    .variant = "UD-Q4_K_XL",
    .manifest_digest = "...64 hex...",
    .format_family = "gguf",
});
```

Direct 用户把相同 workload 作为普通 prepare option 传给路线 Context；inputs/outputs 仍是这次真实
operation 的 descriptors。Graph 用户把 workload 传给 compile options；Graph 把它原样交给 operation
和 region Provider，并继续拥有 region 边界与 whole-plan 选择。两条路径均允许空 profile 的通用调用。

公开类型表达事实和生命周期，不暴露内部 decision table 给普通用户。专家可通过构建 factor、路线和
现有明确实现要求裁剪选择；本轮不增加“用户自定义 Provider 排名”或外部规则脚本。

## 5. 画像、计划和缓存身份

prepared artifact 自己持有 model profile owner，不能借用调用栈字符串。以下任一相关变化都要求重新
prepare/compile：

- build/provider/policy version；
- route environment 中被入选规则读取的 device、driver、runtime/toolkit/SDK 或 feature；
- model/artifact/profile 任一被读取字段；
- phase、region semantic version；
- operation semantic、port dtype/format/shape domain；
- Graph canonical boundary、内部 DAG/state/effects 或资源合同；
- 入选 Provider implementation key。

动态 tensor payload、token 值和已经被 shape domain 覆盖的单次实际尺寸不进入身份。Graph 的 constant
packing cache 继续由 Graph/Provider owner 管理，但 key 必须间接绑定相关 profile、真实 constant identity、
packing configuration 和 provider/environment；不能用源地址或仅用模型名共享。

本合同只要求进程内 owner 和显式 consumer cache。没有通用进程全局 cache，也不声称 identity 可跨
进程、编译器或 Provider build 持久化。

## 6. 诊断与失败边界

冷诊断使用稳定的结构化轴：invalid policy/facts、environment、profile、phase/region、operation
semantic、port index、dtype、format、rank、dimension、provider qualification、resource、ambiguity。
端口诊断携带 direction/index，fact 诊断携带 typed ID；可选文字由错误 owner 保活。matcher 不保留
借用 span，prepared artifact 也不保留 policy predicate。

unknown 与 mismatch 分开记录，便于区分“事实未提供”和“事实已知但不满足”。二者都不能命中 exact
predicate。普通基线路径只在自身未要求这些事实时继续。

## 7. 四路线如何使用同一合同

| 路线 | Context 事实示例 | 原生资格和 artifact owner |
|---|---|---|
| CPU | OS/ABI、ISA/features、NUMA、线程资源、oneDNN/ggml 版本 | CPU Provider 的 primitive/region、packing 与持久 executor |
| CUDA | GPU vendor/device、SM、driver、CUDA runtime/toolkit、库版本 | cuBLASLt/Marlin/FlashInfer/ggml CUDA 等各自 stream、workspace、capture/program |
| Apple | OS/device family、Metal feature、MLX/Metal 版本 | MLX compiled region 或 Metal pipeline/encoder 的原生资源 |
| Vulkan | vendor/device、driver、API/features/limits、shader/provider 版本 | ggml Vulkan/IREE/自有合格 pipeline 的 descriptor、command 与同步资源 |

共同 matcher 只复用相同的合取语义。它不统一四条路线的设备探测、Queue、allocator、Program 或完成
对象，也不要求所有 Provider 把内部每个 kernel 暴露给 HYLE。

## 8. 实现和验收要求

重构必须一次性迁移所有产品调用者；删除旧固定 input0/input1/output0 字段、全局量化 rule 和旧模型
身份双轨。测试至少覆盖：

- profile ownership、OOM、copy/move、并发只读、exact/unknown/mismatch 和 digest 校验；
- 任意 arity、input/output 高索引、rank、axis range/multiple、dtype/format 和 operation semantic；
- 同一 Qwen 模型的官方 BF16 与 Unsloth mixed-Q4 分流；
- Qwen3.8 mixed IQ/Q 与 Kimi/DeepSeek MXFP4 + BF16/F32 region；
- environment/profile/shape/version 条件的 preference、歧义、decline fallback 和 fatal-stop；
- Direct 与 Graph 重复执行不重新进入 matcher/qualification；
- CPU、CUDA、Apple、Vulkan 普通入口和 source/static/shared 交付；
- P0 账单证明新增分配、字符串比较、规则遍历、设备查询和 Provider 搜索都停留在冷路径。

具体交付状态和有限 FIX 见活动
[MR 施工卡](../implementation/tasks/compute/llm-moe-model-profile-routing-refactor.md)。
