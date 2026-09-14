# Compute v8：可组合的 provider route proposal 合同

## Closure card

- component / owner：`compute` / Compute；round=`compute-v8-route-proposal-contract`；
- baseline commit：`49d32deacf80fbc046e53be64de47042144edc5a`；前轮记录为
  [compute-v7](compute-v7-typed-operation-facade.md)；
- trigger 与证据：当前 proposal 只有 route/runtime/workspace/persistent facts，Graph 无法把 leaf 与 region
  放入同一成本空间，也不能区分无副作用 decline、合同违约和真实 terminal error；proposal 只接收 exact
  `TensorDesc`，不能为 bounded Graph route 给出完整 subdomain proof；
- 正常 consumer 任务：direct Compute 仍按显式 provider 顺序准备 exact shared-queue operation；Graph 冷编译
  可取得同一 provider 的 leaf skeleton、资源/结构成本与 shape proof，再只 prepare 入选 route；
- 主产品纵切：Native/oneDNN leaf 与 oneDNN region 竞争的 gated MLP；反偏置为单个 direct CPU op；
- scope：扩展私有 factor SPI，表达 exact/bounded proposal domain、完整结构成本、证据 identity、稳定 candidate
  identity与事务化 prepare outcome；保留 Device/Context/Queue/Completion、public prepare façade 和热执行 ABI；
- non-goals：在 Compute 内实现 Graph selector、模型/batch/KV 策略、运行用户 graph 计时、跨进程 plugin/FFI、
  算子扩充；
- 直接依赖 / 消费者：依赖 Core/Error/Mem/ComputeOps；Graph、Native、oneDNN 等 provider 消费；
- 性能 tier：`P0` 先证明 cold-only、有限 metadata 和 warm 零变化；candidate 后由 Graph one-op/MLP 做 P1；
- 调查与纠正预算：一轮 SPI 表示、一轮两 provider 迁移、一轮 Graph consumer 红队；
- 预声明验证：exact direct parity、bounded/guarded proof/拒绝、成本字段完整性、prepare rollback/decline、
  partial acceptance 禁止 fallback、逐头、三交付、sanitizer、P0；
- stop：如果 Graph 类型反向泄漏到 Compute、steady submit 增加搜索/guard/JIT，或 decline 无法证明完整回滚，
  退回 `SCOPED`。

## AUDITED 结论与有限改动集

Compute 拥有 provider-neutral leaf proposal；Graph 拥有 node-set、boundary 与全局选择。shape domain 使用
Compute 私有、provider-neutral 的 rank-bounded view，不能依赖 Graph ID；Graph 负责把自己的符号域投影为该
view。provider 必须对提案覆盖的整个 subdomain 负责；无法证明时只提 exact route。

| ID | 允许改动 | 验收 |
|---|---|---|
| C80 | 给 leaf proposal 增加 stable candidate key、domain proof/guard、resource/layout/precision/alias/failure/completion 与完整结构成本 | Graph 不再从 provider 名称、顺序或事后 artifact 猜成本；unknown 与 exact zero 分离 |
| C81 | 增加带 provider/build/device/shape-domain/completion-boundary 身份的可选 qualified profile 引用 | 未资格化 provider-local ns 不可跨 provider 排序；direct route 不读取 profile |
| C82 | 把 exact prepare 结果分成 success、无副作用 unsupported/预算 decline、contract violation/terminal error | 只在未接纳 work、已完整 rollback 时允许有限重选 |
| C83 | 迁移 direct prepare adapter，保持显式 provider 顺序与 shared-queue-only 语义 | public API、对象生命周期、warm submit 和错误边界不变 |
| C84 | 版本化内部 SPI 并验证 source/static/shared factor | 旧私有表不保兼容双路线，错误版本明确拒绝 |

### API / usability freeze

公开 Compute API 不新增“自动调优”开关。direct 用户仍通过 Context provider 顺序获得确定路线；高级 Graph
消费者使用内部 proposal seam。proposal/prepare 仅在冷路径运行，所有 span 均借到调用结束，artifact 与
per-run Invocation 继续分离。steady submit 不搜索、不 JIT、不打包、不分配 metadata。

## 七问与当前裁决

无新增第三方；真实 consumer 是 Graph 与现有 Native/oneDNN；复用既有 Error/Mem/Queue/Completion；性能
先签 P0、P1 随完整 Graph；Native FFI N/A；三交付必须跨 factor；public C++ journey 不变。
当前为 `AUDITED`：只授权 C80--C84。

