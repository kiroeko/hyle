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

## D1/D2 Windows 开发检查点

原版资格确认后，普通CPU Direct/Graph已接Copy/Select的一组共同描述映射：把plain存储的
字节轴交给oneDNN原版u8 reorder/binary_select，Bool条件按原版要求描述为s8。没有自己写
选择/复制kernel，不把F16/BF16原始位经过F32；支持所有已登记1/2/4/8字节类型、rank0..8、
广播/非负及奇数字节stride、空输出，输出不alias。共享既有engine/stream/argument handles/
executor/scratch和异常/完成销毁路径，常态无新增HYLE allocation；vendor成本仍未知且managed。

三输入广播边界同时复现了shared schema遗漏condition的缺陷，已先按[M2-S S4](llm-moe-m2-scatter-numerical-semantics.md#s4-audited-增量select-的条件参与输出形状)
独立定界再定向修复；没有改冻结数学。首次artifact检查漏了必需的拒绝alias回调，已补明确
no-alias回调；失败日志保留，没有放宽运行时检查。

Windows Release/shared/WERROR，11个定向CTest门通过，包括43项schema、26项oneDNN主
测试、普通CPU入口/Graph、既有融合消费者和4个公开头检查。新用例覆盖15种storage、F16/
BF16全部65,536位模式、跨轴广播、rank8张量条件选标量branch、empty、alias拒绝、两个slot、
变输入重放、保留旧结果与HYLE allocator平衡。对应日志/原版探针与artifact身份由本轮开发收据保存。

D1尚未全完：浮点reorder会改变部分signed-zero/NaN/subnormal，min/max直接primitive在
单侧NaN的语义不符合公共fmin/fmax；本机直接half/bf16 elementwise域不被原版接受。原版
F32 softplus的诊断有小尾值差，有限样本最大绝对误差约1.91e-6且特殊值通过；诊断相对门不是
模型验收门，不据此宣判其通用失败。上述数值Cast/稳定函数/内部比较的完整合格组合与普通
绑定仍待完成。本次只绑定已验真的raw-storage Copy/Select，没有声称同名primitive等价。

四环境/插桩/三交付与最终selected闭包、必要P1仍待整批稳定后完成，不签CANDIDATE/CLOSED，
不增加256格完整覆盖。中央vendor选项触发的广泛selected门如实保留未执行状态；没有为了
缩小选择改治理工具。旧Q2继续按用户决定视为无效测量，源码审计额度仍120/120分钟。

本检查点产品提交`02fe54151d00b34c09de6d7a452c62097c14182f`，源码tree
`e2db09aa625912d45f6a7004db7d2c7d8b6d381f`；[开发收据及原始日志](../../../../evidence/correctness/compute-llm-moe-m2-b1-onednn/README.md)
包含失败与纠正、11/11最终结果及未运行门。统一脚本全Compute+Graph为**56,908实现行/
249文件**，不含测试/vendor，self-check PASS。本轮无性能采样、无新增自研数学。

复用后续工作所需的单一活动oneDNN build（约1.62GiB）；已归档的临时探针EXE/OBJ/CRT共
18,756,425字节，删除命令被自动审批以“blocked by policy”拒绝且未执行，暂时保留。
这不是产品/验收阻塞，不追加权限请求或改变用户机器设置。

## D1/D2 稳定 Softplus 的固定绑定

实施基线 `90d52243`。本增量在既有 D1–D3 内复用原版资格执行和正常实施额度，
CPU B1 源码/设计仍为120/120分钟；不新增库调研、自研比较、公共 API 或执行资源。

原版 oneDNN F32 `soft_relu(alpha=1, beta=0)` 已完成115,382个独立样本的无计时资格：
极限、NaN/Inf、正负零、[-120,120]步长1/64及100,003个确定性原始位模式。
FP64稳定公式 oracle，沿用现有稳定函数测试的 `5e-6 * abs(expected) + 1e-43` 门，
特殊值单独核对。`jit:avx2` 有6,999个不符，`jit:avx`/`jit:sse41` 各6,995个；
`ref:any` 零不符，最大绝对差约5.10e-7。JIT尾部归零不满足本合同，不能凭默认优先级接入。
这只资格化稳定 F32 函数，不代签完整模型误差、其他类型或性能。

有限实施固定为：

- 只接 F32 Softplus，复用现有 tensor prepared/invocation、线程 executor、scratch、
  memory handles、完成/错误与 no-alias 合同；不新增数学 kernel 或隐藏类型转换。
- 准备时用原版 primitive descriptor 的实现迭代接口匹配已资格化的 `ref:any`，
  不计时、不重选赢家；缺少该实现则 Unsupported。执行只调用已经冻结的 primitive。
- 标量及rank1..8、empty与元素对齐 stride 在原版描述可接受的域内验证；不合格布局在
  冷路径明确拒绝。普通 Direct/Graph 的默认 provider 已包含 oneDNN，验证自然绑定、
  两请求/重放/旧结果、资源平衡和现有错误路径；不修改冻结的后备数学。
- 复用Windows单一活动构建，执行本卡预声明的定向测试；三交付/其余平台/必要 P1
  仍按稳定整批候选执行，未运行不得算通过。源码、原版探针和原始失败均随开发收据保存。
