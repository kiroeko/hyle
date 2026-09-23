# M2-B2 组合路由模型身份补充

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- component / owner：`compute`；`/root`。
- baseline commit：`9efdda05e`。Windows canonical 工作树保留正在施工的 B2 Provider 批次；本卡只授权下列公共身份补充。
- trigger 与证据：用户要求固定规则可同时限定平台、硬件/SM、驱动/Toolkit、模型族、模型名、四条路线和量化级别。现有 `ModelDescription` 只有 family/revision/configuration，不能独立表达“一个家族中的具名模型”。
- 正常 consumer 任务：消费者在模型加载完成后构造一次不可变身份，随 `WorkloadContext` 进入 Direct/Graph 的冷准备；执行期不再读取或匹配这些字符串。
- 主产品纵切：`qwen3_5 + qwen3.5-2b + checkpoint + config + decode + region` 与实际 operation/format、路线原生环境共同限定一个 Provider 候选。
- 正交反偏置纵切：family-only 和 family+name 规则可以匹配更宽范围，但不得命中要求完整 revision/configuration 的规则；未知或断层身份明确拒绝。
- scope：只扩展 `ModelDescription`/`ModelIdentity`、迁移现有直接消费者并验证冷匹配；同步组合路由合同。
- non-goals：不增加通用规则解释器、硬件描述总对象、模型加载器、热路径查表、Provider 数学实现或量化标签副本。
- 直接依赖 / 直接消费者：依赖 `mem`/`error`；直接消费者为 Compute prepare、自定义 Provider 与 Graph workload 传递。
- 性能 tier 与原因：P0；新增一段冷字符串和一次既有单块分配内的拷贝，普通 submit 成本不变。
- 调查与纠正预算：20 分钟 API/consumer 审计；实现、编译和预声明验证另计。
- 预声明验证：`compute`/`graph` 模型身份与 workload 定向测试、公共头闭包、三交付中 selector 选中的受影响项、`check-module-closure.py`、`check-docs.py`、`git diff --check`。
- stop / blocker 条件：若模型名必须由热数据推断、需要模型加载器或改变 Provider runtime ownership，则停止并重新定界。

## SCOPED 审计清单

1. 核对现有平台/路线/设备/驱动事实仍由构建和原生 Context 环境拥有，不在 `WorkloadContext` 复制。
2. 核对量化 mode/group/物理 packing 仍由 operation 与 `StorageFormatId` 拥有，不写入模型标签。
3. 核对 family、name、revision、configuration 的层级和空字段不产生歧义。
4. 核对所有 `ModelDescription` 直接消费者并一次迁移，避免兼容双轨。

### 有限改动集（待 AUDITED 冻结）

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| RQ1 | 缺少 family 内的独立模型名 | 增加 `name`，继续由 `ModelIdentity` 单块冷分配持有 | 输入字符串失效后四字段仍稳定；空字段层级受检 |
| RQ2 | 只有四字段 exact 不足以表达宽规则 | 增加 family、family+name、完整身份三档冷谓词 | 宽规则不冒充窄规则，完全相同才命中 exact |
| RQ3 | 现有 Direct/Graph 测试使用旧三字段聚合初始化 | 一次迁移全部直接消费者与测试 | 无旧 API 双轨，Graph/Compute 均保留 workload |

## 当前裁决

状态为 `SCOPED`。尚未授权修改 `compute` 产品代码；完成上述审计并冻结 API 后再进入 `AUDITED`。
