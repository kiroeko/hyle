# M2-B1 oneDNN：独立新能力接入

## Closure card

- component / owner：`compute_provider_onednn`、`graph_provider_onednn`；`/root`。
- baseline：`625ae14c`，Windows canonical，工作树干净。SCOPED基线；当前已进入下述AUDITED有限实施。
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

## 有限 FIX

| ID | 有限接入 | 验收 |
|---|---|---|
| D1 | 原版Select/比较、min/max/reorder与softplus的类型/布局/特殊值资格 | C0独立oracle；F32/F16/BF16、广播/stride/empty、NaN/Inf/尾值/舍入；不合格域明确拒绝 |
| D2 | 共享既有engine/stream/primitive与每请求scratch，Direct/Graph成批lowering | 自然入口、固定冷绑定、错误与实际完成、两个slot、旧结果、OOM/部分构造/关闭 |
| D3 | 所属factor构建、真实组合consumer、指南及成本记录 | 四环境CPU与适用插桩、source/static/shared、relocation/producer隐藏、CPU-OFF；同合同成本与必要P1 |

D1有限审计已完成，按以下AUDITED边界实施D1–D3。不是逐原子新建Provider，不复制oneDNN数学。
所有未运行验收均UNPROVEN；本卡尚无新增产品/覆盖PASS。旧Q2与新接入在记录中分别裁决。

执行顺序更新：用户先要求原因调查优先，现已接受机制证据与历史限制，并授权工程翻案及
恢复M2。本卡继续有限D1审计、原版资格和D2/D3批量接入，不再等待旧Q2或追加原因调查。

## AUDITED：复用原版 primitive 与现有资源接缝

baseline `02ca7627`；两个owner SCOPED→AUDITED。复用父卡的oneDNN源码/API证据，有限接法
核对扣余下2分钟，CPU B1源码/架构审计累计120/120分钟，停止扩张来源；以下正常实现、原版
资格、构建和已预声明验证独立推进。未发现需要改变公共API或runtime owner的新事实。

- KEEP：已有NormPlan/Invocation的engine、线程executor、stream、argument memory、scratch、
  异常翻译及销毁顺序；原norm/MatMul算法不变，vendor allocation未知不记成零。
- FIX D1：在现有oneDNN私有CMake声明增开BINARY/ELTWISE/REORDER，版本/Graph-OFF/
  THREADPOOL/CPU隔离及三交付owner不变；先运行原版无计时的类型/广播/stride/特殊值资格。
  比较只作原版组合的内部步骤，公共schema没有比较操作时不擅自增加公共API。
- FIX D2：合格primitive复用上述执行资源，用同一私有描述映射Select、min/max、Copy/Cast、
  softplus及必要稳定组合。Graph优先复用已有叶程序桥接及完整成熟片段；数值不合格域不绑定，
  不由HYLE手写补数学；不能把输入finite假设藏在普通路由里。
- FIX D3：普通CPU路由的固定成熟基线、Direct/Graph真实组合与资源差分测试、指南、P0及
  原有四环境/三交付门。oneDNN仍为x64 Provider；Air/Mini验证CPU整体成熟后备及oneDNN
  AUTO关闭/显式ON拒绝，不把arm64 oneDNN写成已支持。适用P1等API/算法固定后才开始。

准备绑定及运行职责不变：非连续布局按oneDNN stride描述，不能未经资格暗中复制或变精度；
每个请求拥有独立memory handles与scratch，共享制品只读。转换不得擅自支持不等价alias；
空张量保持公共语义。具名不兼容只能记录并审计已有成熟组合，不扩大本轮预算或转自研。
