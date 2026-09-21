# M2-B1 oneDNN：独立新能力接入

## Closure card

- component / owner：`compute_provider_onednn`、`graph_provider_onednn`；`/root`。
- baseline：`625ae14c`，Windows canonical，工作树干净。阶段SCOPED，尚未AUDITED，不改产品。
- trigger：2026-09-21用户明确批准将新能力接入与旧Q2资格失败分开，要求后续查清同实现差距。
  原版Select、比较/掩码、reorder、min/max与稳定函数候选来自[M2-D1](llm-moe-m2-b1-cpu.md)。
- 旧Q2当前工程裁决：用户接受定向诊断已证实的环境污染机制与历史唯一归因限制，并授权
  翻案。旧17×257同实现差距按环境干扰最可能的无效测量处理，撤销产品FAIL含义与接入
  阻塞；原始日志、原统计门保留，不改成性能PASS、正式产品样本仍0，不重跑Q2。见
  [当前原因裁决](llm-moe-onednn-aa-cause-followup.md#当前工程裁决旧样本无效恢复-m2)。
  本卡新能力独立完成验收；不重新比较或淘汰冻结自研。
- 主纵切：模型的动态mask/Select→稳定门控/归一化组合；正交纵切：无模型注解的广播、布局
  转换与普通张量片段。两请求独立，旧结果保活，普通CPU入口不要求用户排序Provider。
- scope：已锁oneDNN3.13.2 `71094df4`（Apache-2.0）的原版primitive/合法组合资格和批量
  Direct/Graph绑定；复用已有engine/stream、prepared primitive、请求资源和错误/完成。
- non-goals：旧norm/MatMul数学重写；冻结CPU/GPU算子研究/优化/比较/删除；新线程池或
  公共build/selector/性能仪器；以同名primitive代签不相容的极限/舍入语义。
- owner边界：本卡只修改上述两个factor。CPU普通route已由父卡AUDITED，必要消费者接线
  在父卡有限FIX内审计；新增公开API、依赖或资源形状须先重定界，不自动授权下层改动。
- 预算：继承CPU B1已用118/120分钟、余2分钟的源码/设计额度；复用已登记API/数值定位。
  新名字不重置额度；原版资格执行、正常实现/构建和预声明测试单列。具名根因最多两次修正，
  需要超额新调查时记录具体缺口，不能无限扩展来源。
- 性能：P0全调用账单；新能力正确性/API固定后，仅对不可变候选按父卡5%门及原仪器做必要
  同原版成熟调用验收。旧Q2不重跑、不继承PASS；新qualification失败即保留事实按预算裁决。
- stop：本卡有限合格域及适用门完成后关闭；CPU父卡21行与四路线256格分母保持。
  用户要求的[后续A/A原因调查](llm-moe-onednn-aa-cause-followup.md)单列，M5前暂停不变。

## 待审有限 FIX

| ID | 有限接入 | 验收 |
|---|---|---|
| D1 | 原版Select/比较、min/max/reorder与softplus的类型/布局/特殊值资格 | C0独立oracle；F32/F16/BF16、广播/stride/empty、NaN/Inf/尾值/舍入；不合格域明确拒绝 |
| D2 | 共享既有engine/stream/primitive与每请求scratch，Direct/Graph成批lowering | 自然入口、固定冷绑定、错误与实际完成、两个slot、旧结果、OOM/部分构造/关闭 |
| D3 | 所属factor构建、真实组合consumer、指南及成本记录 | 四环境CPU与适用插桩、source/static/shared、relocation/producer隐藏、CPU-OFF；同合同成本与必要P1 |

先完成D1有限审计并提交AUDITED，再实现D2/D3。不是逐原子新建Provider，不复制oneDNN数学。
所有未运行验收均UNPROVEN；本卡尚无新增产品/覆盖PASS。旧Q2与新接入在记录中分别裁决。

执行顺序更新：用户先要求原因调查优先，现已接受机制证据与历史限制，并授权工程翻案及
恢复M2。本卡继续有限D1审计、原版资格和D2/D3批量接入，不再等待旧Q2或追加原因调查。
