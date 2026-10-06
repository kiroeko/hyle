# CPU Native 数值不合格域退役：ND-R1

产品 candidate `f32d261a5a68d5af99233286fcc81603c4bdb87a`；fixture
`f83aa9f14348c50457ffd4d28cb77bec6ba63e5e`。Windows 是产品真相，Linux 是前向验证副本。
本批只验收 [当前卡](../../../docs/implementation/tasks/compute/cpu-native-final-domain-verdict.md)
固定的 ND-R1，不关闭整个 CPU goal，不签性能 WIN/LOSS。

## 实现与边界

依据 [同合同反例](../cpu-native-final-semantic-audit/README.md)退役 16 个 Native 通用域：

| 操作 | 退役域 |
|---|---|
| Unary GELU / GELU-tanh | F32/F16/BF16，6 域 |
| Unary SiLU | F32，1 域 |
| GELU / GELU-tanh gate | F32/F16/BF16，6 域，packed/separate 都退役 |
| SiLU gate | F32/BF16，2 域，packed/separate 都退役 |
| RoPE | F32 数据/频率与 I64 positions，1 域 |

删除对应的 Native 算子公式、准入和专用 callback；成熟 ATen 相同公开端口实际接替。
SIMD 基础及其函数、原库数学不改。Native F16 SiLU gate、I32 RoPE 与其它未裁决数学仍保留冻结。
不能从本批小 I32 witness 外推任意位置精度，也不能把仍保留称为性能领先。
专家 Native-only 及不允许成熟 managed 程序的资源政策明确拒绝已退役域，不悄悄放宽政策。

首次退役后发现 Unary 的旧选择失败返回 InvalidArgument，打断普通成熟候选匹配；定向改为
Unimplemented。保留首次失败日志；这项修复只改变冷拒绝状态，没有数学优化。

## 实际验证

| 门 | Windows | Linux |
|---|---|---|
| Native factor，含独立成熟 ATen | 142/142，无失败/skip | 142/142，无失败/skip |
| 普通 CPU 受影响 suite | 12 通过、1 旧 fixture skip | 12 通过、1 旧 fixture skip |
| 受影响 Graph suite，排除未变 D5 | 17/17，无失败/skip | 17/17，无失败/skip |
| 公开头自包含 | 1/1 | 1/1 |
| 普通 Direct/Graph consumer | source shared/static、installed shared/static 全通过 | 同四种旅程全通过 |

新增冷回归同时检查 propose/prepare 对全部 16 个域与两种 gate 输入形式明确拒绝。
旧 Native-only Unary/gate 预期按需求迁移；其余原生能力继续实际执行。普通 consumer 核对 F32
稳定 Unary、三种放大 gate 的 packed/separate、长 I64 RoPE，验证实际成熟身份、下游 Copy、
两次变化输入、真实 wait、close，以及 Native-only Graph 的拒绝。此 consumer 不冒充三 storage
的完整数学误差资格；数值反例和原成熟资格仍分别引用自己的证据。

固定 candidate 的同一无计时数值 witness 实际记录 322 条：Native 拒绝 31 条，普通成熟 161 条
全部执行，已执行行 0 误差失败；拒绝不是执行 PASS。数值门和数据分布没有放宽。
归档时发现旧 witness driver 继承了 cde6fb3a 的过时 receipt stamp；原记录保留并注明限制，
只重新编译并执行这一短 witness，最终两条 receipt 明确绑定 f32d261a。没有重建产品或重跑 P1。

1337 个产品/测试/构建输入前向 SHA256 匹配；110 个 SIMD、成熟数学与默认组装器输入和 CP
证据相同。只复用未变 ATen 原程序、生命周期、SDK-off 与 D5 合同，不把它们称为本批重跑。
本批自身的 Native 冷迁移、旧测试迁移及新普通 consumer 都完成实际验证。

## 终审、成本和限制

ND-R1 的 scope、architecture、correctness、delivery、usability/docs、red-team 与 P0 成本门
通过。终审检查了有效域拒绝状态、默认成熟选择、两种 gate 输入、专家拒绝、未放宽资源合同、
SIMD/原库边界与三交付一致性。原错误数学不进入性能竞速；删除减少算子分支和代码体积，
不增加热搜索、计时或同步。没有新 P1、GPU 或整模型推理完成声明。

统一脚本统计 Compute+Graph 实现 **61,387 行 / 249 文件**，不含测试/vendor；统计 commit
cde6fb3a，后续 f32d261a 仅单行 Status 替换，代码行数不变。四个本批 consumer 目录已删除
**625,435,572 logical bytes**；SDK、producer 和依赖 cache 保留。这不是物理磁盘释放量保证。

全部命令、XML、forward identity、consumer、driver、数值 witness、源码与 cleanup 在
[qualification.zip](qualification.zip)（82 entries，205,819 bytes），SHA256 与摘要见
[receipt.json](receipt.json)。整个 CPU goal 仍需对剩余合格自研域作一次有限同合同性能/路由裁决。
新的有效数值反例、依赖/公开合同变化或交付缺陷才触发本批重新审计。
