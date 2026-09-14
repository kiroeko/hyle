# Graph v10：整体复核后的 foundation 合同

## Closure card

- component / owner：`graph` / Graph；round=`graph-v10-reviewed-foundation-contracts`；
- baseline：`610a219f57e613ac6f1aaf79b337742122e2569f`；承接未 terminal 的
  [v9](graph-v9-foundation-routing.md)，保留已实现的 node-set/三轴资源/Completion 骨架。
- trigger：[整体复核 A1--A6](compute-graph-route-unification-audit.md#整体架构复核与合同修正)证明独立
  bounds、全局 effect chain、一律预算失败和不区分 replay 成本不足以承载既定旅程。
- 主纵切：动态 token 的 normalization→matmul→gated activation、精确 capture 变体与 general 路线；
  反偏置：单节点 exact CPU、自定义异步资源任务；四引擎的来源与取舍见总审计。
- scope：G100--G107；这是复合 planning/replay 能力，不是模型/KV/batching policy。
- non-goals：算子目录扩充、第三个 public Plan、通用符号求解器、全局自动调优/缓存、Provider 内核重写。
- 依赖：Compute/ComputeOps/Container/Mem；复用 Error、Queue/Completion；没有新增反向依赖。
- 性能：迁移 P0；全部有限 FIX 与真实 CPU/CUDA/Apple 探针稳定后，统一 one-op/MLP P1，沿用基础卡
  5% 工程门与完整边界/反向顺序，不以 managed kernel-only 成本对比 prepared 的完整调用。
- 预算：shape、selector、资源/variant ownership 各一轮表示与一轮纠正；定向红队一次。不在这些任务里
  改公共仪器；正常开发中的编译/fixture 错误不冒充最终性能裁决。
- 验证：G100--G107 的反例、原 exact/node-set 全回放、OOM/并发/完成与销毁顺序、逐头、三交付；
  Windows/Linux/两台 Apple 与适用 sanitizer 在完整候选验证，不因 docs/状态提交重跑 P1。
- stop：热态 HYLE search/JIT/分配、接纳后换路、用模型策略填基础合同缺口，或 required 约束被静默忽略。

## 已审计的有限改动集

| ID | 能力现状与动作 | 固定验收边界 |
|---|---|---|
| G100 | FIX replay 成本传播与冷筛选；复用 Compute 的属性校验/比较，不再复制字段列表 | unknown 不等于 prepared；prepared-only 排除 managed/unknown；report 原样保留，热提交不筛选 |
| G101 | FIX independent/derived shape 与 guards，复用 Compute provider-neutral program | 独立维度只绑定一次；flatten/concat/head split/broadcast、越界/overflow/非整除；完整域证明或 exact-only |
| G102 | FIX finite exact variants + 完整 general fallback | compile 后不新增 variant；concrete binding 固定路由，capture 精确绑定 shape/storage/queue |
| G103 | FIX leaf/region/placement 同池有界选择与完整成本 | 硬约束优先；copy/repack/sync/workspace/persistent/关键路径与 qualified identity；无证据稳定回退，不承诺全局最优 |
| G104 | FIX 有限事务化重选与预算终止 | 搜索预算停止扩搜而非禁止完成已有候选；有合格可行解返回并报告 optimization-limited，strict caller 可拒绝；无解失败，不吞真实错误 |
| G105 | FIX resource/effect domains，替换全局 last-effect 假串行 | schema access 来自 ComputeOps，资源 ID 属于 Graph；同域冲突有序、独立域可并行、未知保守；bind alias 与 region 边界反例 |
| G106 | FIX constant borrow/materialize 与 variant 聚合 | 成功 materialize 后 source 可释放；兼容 immutable artifact 显式共享，不能共享的副本披露；unique resident/冷峰值/并发/capture 全计预算 |
| G107 | KEEP exact replay/lifetime，并完成 report 与交付 | shortlist/拒绝/成本来源有界且可追溯；安装包普通 C++ 与 custom execution 消费同一 Compute Provider，不依赖 src SPI |

这些 FIX 来自 architecture 已接受的具体矛盾，不因本卡延长而重新审全库。Compute 的公开 owning custom
factory、ComputeOps access schema 必须先在各自有限卡完成；本卡不越过 owner 写其产品代码。

## API / owner / 成本

保留 `GraphBuilder → Graph → Executable → Invocation`，Plan 仍私有。Builder 拥有 shape 表达式与资源
identity，compile 拥有有界候选和不可变制品，Invocation 只拥有本次 scratch/bind/capture/完成状态。
普通 exact 用户无需提供表达式、资源域或 profile；动态用户只绑定独立维度，派生维度由预构建有限程序
求值。冷编译默认允许已声明且满足硬约束的 prepared/managed 路线；显式 prepared-only 不放宽。

G100 的首步仅迁移已有 runtime facts：`ProviderReplayClass` 为 unknown/prepared/managed，字段
`replay_class` 默认 unknown，公共 value type 提供 `valid()` 与完整 equality；unknown 可被诊断/保留，
不能据此证明 prepared-only。实际政策筛选与 managed 成本预算不是这一个字段自动完成的能力。

派生 shape 不读 payload；资源访问不携带模型 KV/communicator 策略；常量共享不靠裸地址猜内容；无解/
真实 error 仍返回 Status。异步对象一直保活到真实 Completion，不能用 host 提交返回替代设备终态。
新私有存储布局和有界搜索实现可在上述合同内实验；若要改变 owner/默认政策/完成边界，先回到审计。

## 七问与当前状态

无新外部依赖/FFI；Provider 接入只复用已有因子，版本事实不按名称猜测。能力与易用性由四引擎的真实
旅程和 exact 反偏置约束；正确性/性能/三交付需按上述具体反例实证，不能以代码量、已编译或历史通过代替。
已有 AUDITED 轮按接受的架构基线绑定本后继有限卡；当前仍 `AUDITED`，尚未形成新的 Candidate/PASS。

## 2026-09-14 G100 首步

Graph 已复用 Compute `ProviderRuntimeProperties::valid()`/equality，region proposal 与 artifact 的 replay
class 变化会报错并销毁制品；CompileStepReport 原样传播 unknown/prepared/managed。region extension
版本为 6；实际 oneDNN MLP 声明 prepared，不从 shared ownership 推断。反例与 Windows shared/static
及 oneDNN 回归见 [Compute C91 检查点](compute-v9-shape-replay-substrate.md#2026-09-14-c91-成本分类检查点)。

G100 的 prepared-only 自动筛选和 managed 预算尚未实施，其余 G101--G107 未因这次元数据迁移自动
完成。仍为 `AUDITED`；未运行的 CUDA/跨平台/sanitizer/installed/P1 格不得记为 PASS。

### G101 表示审计（实现前固定）

复用 Compute C92 程序作为 Graph 私有 dimension record，不复制整数计算/guard checker。公开保留
`dimension(bounds)` 创建独立维度，新增 `dimension_sum/product/quotient/broadcast(Dimension, Dimension)`
创建派生 `DimensionId`；`require_dimension_equal/less_equal` 添加 guard。操作数必须为常量或同 Builder
已存在 ID；失败不追加半条记录。`dimension_count()` 枚举全部维度，新增 `independent_dimension_count()`
与 `dimension_is_independent(id)` 帮消费者识别需要绑定者；bounds 允许零，与 TensorDesc 一致。

compile 仅要求每个独立维度一个 ShapeBinding，拒绝用户绑定派生 ID；一次程序求值后解析 TensorSpec，
signature 保存全部解析值，因此不丢失派生维度 identity。exact-only 路线继续可用；这个步骤先以真实
Native concat 和派生 row spec 完成消费，不据此声称完整 bounded artifact、dynamic View attributes、
variant 或 runtime shape selection 已完成。后几项仍属于 G101/G102 的待办，不用固定 View 常量冒充动态
reshape。普通 concrete 图和 direct Compute 不需要构造 shape 程序。

## 2026-09-14 G101：有限派生维度接入 exact Graph

上述公开 dimension/guard API 已实现；Builder 只追加已验证的算式，Graph 私有记录直接使用 Compute
的有限程序。compile 只接受独立维度 binding，冷求值后解析 TensorSpec；signature 保留全部已解析值。
zero bounds 与 TensorDesc 对齐，未改写用户提供的固定 View attributes。两条实际执行是 concat 派生输出
轴与 batch×tokens normalization，细节和定向验证见
[Compute C92 检查点](compute-v9-shape-replay-substrate.md#2026-09-14-c92--g101-首个真实消费检查点)。

Windows shared/static 与 oneDNN 回放通过不等于 G101/G102 完成：目前 Executable 仍是一个 exact point，
缺少 bounded artifact/全域 schema proof、动态属性和 runtime variant 选择。G100 政策筛选、G103--G107
的统一成本/资源/常量/custom execution 仍待落实。保持 `AUDITED`，没有完成声明或新性能结论。

### G100 冷筛选表示审计（实现前固定）

新增 `ProviderReplayPolicy::{kAllowAny,kRequirePrepared}`，由 `CompileOptions::provider_replay` 指定，默认
allow-any。严格政策在 leaf proposal 后、资源选择与 artifact prepare 前筛除 managed/unknown；region
discovery 必须先声明 replay class，先筛选再影响 target/region 选择，不能等制品编译后才发现不合格。
region extension 升至 7；初始 known class 与后续 resource proposal 必须一致，unknown 可在默认政策下
细化，但不能通过 prepared-only。最终 artifact 继续用既有完整 runtime equality 校验。

沿用既有有限 proposal/node 预算，被政策排除的 proposal 仍计 discovery 预算；不在本步改变预算终止
策略。非法枚举/前后矛盾是合同错误，不静默 fallback。政策不把 prepared 等同于 shared/capture，也不
替调用者 executor/回调证明零分配。View/Copy 等非 Provider 步骤不受 Provider 成本类别筛选。

验证限定为：strict leaf/region 拒绝在 artifact 创建前发生；prepared delegated 可以入选；默认政策仍可
选择 managed；未知初始值不能伪装成 prepared，known 值不得改变；实际 Native/oneDNN 回放、非法政策、
fallback 与 proposal 预算反例。只改冷 compile，热提交不存储/查询这个政策，不启动单独 P1。

## 2026-09-14 G100：prepared-only 冷约束检查点

上述政策已实现，公开入口与限制见 [Graph 指南](../../../user/graph.md#何时要求-prepared-only)。leaf 在
proposal 后、resource/prepare 前筛选；region 在影响 target/partition 前筛选。discovery 的 known 类别
不得在 resource proposal 改变，artifact 继续完整比较 runtime facts；extension v7 无旧布局适配。oneDNN
discovery/resource/artifact 复用同一个 prepared 常量，未修改 MLP kernel 或调度。

定向源码/反例复核覆盖 unknown 后续细化仍不合格、managed 默认可用、prepared delegated 可执行、无
合格路线失败、跨 target 不被拒绝 region 误导、非法枚举、known 类别改变和被筛除 proposal 仍占预算。
View + 跨 storage Copy 的实际执行在严格政策下通过。新增 probe 的实际 artifact/resource 回调计数证明
拒绝发生在编译制品之前；两次 delegated submit 不重进 proposal/prepare。首次测试编译发现 fixture
未包含 CPU convenience 头，改用其已有显式 Context/Native 构造后通过，没有改产品便利入口。

- Windows Release source-tree shared/static：Graph builder/executable/逐头各 3/3，分别含 114 单元用例。
- oneDNN-enabled shared：Graph executable、真实 oneDNN MLP 与双方逐头 4/4（112 单元用例）；完整
  MLP 数值用例现明确采用 prepared-only，原普通默认路线仍由其他 region 用例覆盖。
- P0：每个 proposal 只增加 `O(1)` 冷比较，候选数/预算与复杂度不变；政策不存入 Invocation，也不由
  submit/Queue/kernel 查询。拒绝的错误 Status 属于冷路径，不承诺其零分配。
- impact selector 为 READY，只选择 Graph/Graph-oneDNN/逐头/交付及公共检查，跳过 89 个不受影响项。
  closure/docs/diff 通过；并行编译最多 2，未运行 P1 或全库测试。

这些不是完整安装 SDK、Linux/macOS、sanitizer 或真实 Apple managed 路线的资格化。G100 的分类传播与
严格冷筛选已落实，managed 完整成本、C93 identity、G101/G102 bounded/variants、G103--G107 其余范围
仍待完成。保持 `AUDITED`，不宣称基础架构实现/终局性能已经收口。
