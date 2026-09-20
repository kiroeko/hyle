# M2-B1 CUDA-I：成熟 CCCL 索引 collectives 接入

## Closure card

- component / owner：`compute_provider_cuda`、`compute_route_cuda`；`/root`。
- 2026-09-20后续定界：`fa65e0dc`独立提交I5 Gather待审范围后，以下原版资格及有限接法已冻结
  为AUDITED；registry未作非法AUDITED→SCOPED倒退。未决验收和父卡需求不缩小。
- baseline：`93b5095f`，Windows canonical，工作树干净。当前阶段 AUDITED；`c1923e1a`独立绑定SCOPED。
- trigger：[M0 CUDA-I](llm-moe-provider-map.json)已将成熟 CUB scan/sort 分给现有 CUDA Provider。
  前者仍绑定 M0 空 FIX，须换成本卡；普通 CUDA 工厂需要新成熟选择，满足 M1-C 新能力 reopen。
- 需求：B1 的 F02.4 前缀计数和 F04.6 TopK，以及同卡 F19 的资源/失败/重放；不缩小父卡21行。
  [父卡](llm-moe-m2-b1-cuda.md)其余域仍由原 owner 推进，完整覆盖未完成。
- 主纵切：MoE router 的 stable TopK→专家计数 Scan，两个请求独立，值/索引双输出和错误保留。
  正交纵切：无模型注解的任意轴/stride 扫描与排序，NaN/相等值、整数溢出、旧结果/重放。
- 来源：CUDA Toolkit13.3.73 随附 CCCL/CUB3.3.4（`CUB_VERSION=300304`），原版
  DeviceScan/DeviceRadixSort/DeviceReduce 与标准迭代器/tuple 操作。CUB为BSD-3，Thrust为
  Apache-2.0/Boost例外，libcu++为Apache-2.0 WITH LLVM-exception；按原版汇总LICENSE随三交付
  保存，不引入第二个toolkit下载器。
- 范围：原版成熟算法的语义/布局/类型接线、独立 Provider identity、已有 CUDA 队列与
  workspace/Completion、普通 Direct/Graph 默认接线、对应 CMake源文件/指南/验收。
- 非目标：修改、删除、优化、性能比较或研究任何冻结自研 kernel；新 `__global__` 算法、
  自写 scan/sort/reduction；公共调度/验证/性能仪器重构；oneDNN Q2 恢复；独立新增空 factor。
- 资源/API 形状：复用现有 factor 和公开 `compute/providers/cuda.h`，新增具名 CCCL factory
  与 identity，普通 CUDA 工厂自动组装；Graph 复用已有 leaf 与真实原生完成边界。
  冷期匹配与workspace sizing；热执行只调用固定原版算法。CUB运行时派发按 managed 如实声明，
  prepared-only消费者继续由既有合格路径承载，普通用户不排序Provider。
- 预算：与父卡共用120分钟源码/设计总额；此前72分钟，本次已用8分钟，累计80/120、余40。
  首次原版资格 probe 和实现细节调查合并计费，不因换 owner/卡续杯。编译/预声明正确性单列；
  每具名根因最多2次源码驱动修正，不比较旧自研。
- 性能：父卡同合同完整调用/P0→稳定不可变候选一次P1；5%门与已声明资格预算不改。
- stop：两行完整能力与全部适用门完成才可关闭；主方案不合格先保留具体反例和成熟后备责任。
  到预算明确裁决，不转成自写授权。M5前按用户要求暂停。

## 待冻结有限 FIX

| ID | 有限边界 | 准入前需证明 |
|---|---|---|
| I1 | 原版 ScanByKey，F32/I32，任意轴、inclusive/exclusive、公开合法stride | I32提升到I64原版scan，全部prefix（含exclusive最后总和）越界则整个输出不改；F32按合同允许重关联 |
| I2 | 原版稳定 radix sort，TopK值与索引，k=1/configured/E、相等值/NaN和任意轴 | 使用库的稳定排序；标记NaN、行ID和方向只作语义键投影，原值bit pattern通过索引提取；不写排序/归约循环或kernel |
| I3 | cold shape/workspace、请求私有资源、失败与原生stream完成 | GPU上检查所需标量可回传host控制，不能把张量模型计算放CPU；检查/同步/转换成本全计；校验失败不改用户输出，原生故障保留已接收prefix的实际错误，不承诺硬件事务 |
| I4 | 现有factor内的真实CCCL Provider及普通CUDA工厂、Graph leaf、文档/三交付 | 新identity可审计；不修改旧自研kernel和选择内核；无新插件框架或万能executor |

## 预声明验证

- 先一组原版资格 probe：Scan I32边界/overflow、exclusive最终溢出；stable TopK的NaN/Inf、
  signed-zero、同值顺序、两行；独立host oracle，不读旧自研数学。
- 产品阶段扩展任意轴、非连续输入、空/尾/容量、两输出、无原位/部分重叠、两slot与旧结果，
  以及workspace/OOM、提前close和失败排空。普通Direct/Graph必须自动选择具名成熟Provider。
- Windows/WSL真GPU、适用host插桩；Air/Mini CUDA-OFF；稳定程序的一份source/static/shared
  relocated消费旅程，producer隐藏、依赖/许可证/DLL邻置。重型串行，parallel2，不弹UAC。
- selector选中门、docs/closure（状态更改用明确previous-ref）、最终候选复核及适用P1。
- 定界时仅SCOPED；后续AUDITED及开发证据见下节，完整能力/验收未签，不用probe代签产品覆盖。

## 已核对的原版事实

- ggml `getrows.cu`直接用运行期索引读地址，没有本合同的checked失败通道；不能仅绑定函数名。
- ggml `cumsum.cu`支持原版scan，但I32路径不提供整个输出不变的溢出裁决。
- 原版CUB提供InclusiveSumByKey、独立Reduce和稳定RadixSort；NaN默认按浮点bits排序，不自动
  满足HYLE的NaN尾部/稳定索引要求。该语义必须由合法的库key/decomposer接法资格化。
- 本卡不授权完整Gather/Scatter接入，也不消除其余父卡缺项；新的库/owner变化必须具名审计。

## AUDITED：有限实现冻结

[原版资格记录](../../../../evidence/correctness/compute-llm-moe-m2-b1-cuda/cccl-qualification.json)
实际执行8组I32扫描、12组F32/I32排序/取值检查并通过，没有读取或对比旧自研数学。
原始源、编译命令、日志、SDK文件hash及失败边界均已保存；不签产品覆盖。

- I1：`DeviceScan::InclusiveSumByKey`按逻辑行分段；I32用库transform iterator提升到I64，
  原版Min/Max检查全部prefix，再只回传两个标量。无溢出后才由原版DeviceTransform投影到
  dense输出；exclusive从同一inclusive结果取前一个prefix/行首0，包含最终总和检查。F32
  使用同一库scan和输出布局投影。非空count≤INT32_MAX，I32总和因此不会溢出I64；空输入走no-op。
- I2：DeviceTransform生成库的复合排序key `{row,NaN标记,方向化score}`，DeviceRadixSort的
  公开decomposer解释字段并稳定排序；I32 score提升I64后取方向，不丢失超过2^24的整数。
  两次原版Transform按排序索引提取原始值与I32下标，返回sorted结果也满足sorted=false。
  F32/I32直接资格通过；F16/BF16沿成熟CUDA转换生成F32 key，原值按storage位提取，须在产品
  测试中实际补齐。只写接口需要的标量投影和地址换算，不写scan/sort/reduce循环或kernel。
- I3：原生CUDA stream直接记录CUB；不创建第二个runtime/queue/event。Prepared只持描述符/
  offsets，caller-owned workspace覆盖所有影子输出、keys/indices和原版临时区；每slot只保存
  已验证workspace地址。热期不重新计算workspace或选实现；CUB内部dispatch如实记managed。
  I32检查有host等待，其他路径无适配层host等待；不声称整个原版完全无内部查询/分配。
- I4：`cuda_native::cccl_provider()`与独立ID，普通CUDA工厂在旧native前自动组装；Graph普通
  leaf自动消费同一Provider，prepared-only会拒绝managed候选。现有factor CMake只增加两个
  source、同一个CUDA SDK和原版LICENSE资源，无新组件或依赖下载机制。CUDA13.3/CCCL3.3.4及
  SM89为当前资格条件，未知条件冷拒绝，不影响其余Provider。所有旧kernel文件保持不变。

程序描述接受公开rank≤8和合法plain byte stride，通过标量地址投影交给库；输出dense、不别名。
工作区字节/对齐与类型容量先检查。TopK的空axis/k越界按共同schema拒绝，其它零元素外维可no-op。
同一header新增factory触发selector相应API/交付门；其它CLOSED owner不自动重开。

补充原版接缝、资格与打包核对计8分钟，父卡共同源码/设计累计 **88/120分钟、余32分钟**。
I1–I4由本节冻结后方可实施；未运行的产品/跨平台/三交付/P1均未签。

## Windows 开发检查点

I1–I4产品接线已实现；[产品receipt](../../../../evidence/correctness/compute-llm-moe-m2-b1-cuda/collectives-windows.json)
保存精确实现hash、失败及修正和原始日志。Windows SM89 Release/shared实跑 **10/10 CTest、
100次GoogleTest执行**，包括7个CCCL Provider测试、普通Direct与两个新增Graph测试。

- Scan：F32/I32，正负axis、rank6/8、奇数字节stride/零stride、空域和8193列；特殊浮点前缀；
  I32最终/中间正负溢出覆盖inclusive/exclusive，所有输出保留，fresh invocation可恢复。
- TopK：F32/F16/BF16/I32，三个axis、rank8、奇数字节stride/零stride、k=1/整轴/8193列取7；
  ±0、NaN payload、Inf、重复值、超过2^24的整数，独立stable host oracle验证下标与原始bits。
- 资源：caller workspace不足、精确/部分别名和外部capture提前拒绝；Context/Prepared/Invocation
  每个HYLE metadata分配点失败清理及重试；Completion保活Provider句柄，提前close后正确排空。
  Direct的payload、workspace、Queue仍由caller保活；Graph管理每slot workspace并保留另一slot旧结果。
- 普通工厂自动选择独立CCCL ID；Graph TopK→Scan使用默认prepare_replay选项、两个slot和变化输入。
  旧capture验收明确要求prepared，保留原有capture/失败门。F32不同实现按既有FP64合同和原容差
  验证，不要求逐位相等；整数及TopK原值/索引仍精确。
- 第一次CMake误用需要源码lock的runtime-asset注册；CCCL由已安装CUDA SDK提供，因此在所属factor
  用标准`install(FILES ... COMPONENT Hyle_Runtime)`交付原版合并LICENSE，不新增下载器或改治理。
  CUDA-OFF不安装该文件；源码包原样携带。实际relocated三交付仍待本批稳定后验证。
- 首轮测试失败为新fixture非法1字节tensor起始偏移、旧Provider数量断言；已按公共格式对齐与新增
  identity修正，保留奇数stride。另一新测试提前释放borrowed workspace，导致cuMemFree等待尚未
  释放的测试Gate；按既有合同保活caller工作区，仍提前释放Prepared/Invocation并close后验证排空。
  构建还修正一处GoogleTest宏的逗号括号。均保留原失败日志；无产品数学纠正或冻结实现比较。

P0：Scan工作区保存N个F32/I64前缀，I32另含两个I64标量和原版reduce临时区；完整成本包括
scan、两次reduce、16字节回传/host等待及输出投影。TopK使用两份复合keys/indices和SDK临时区，
固定radix位数下按输入规模增长；layout投影、原版runtime dispatch及所有输出提取均计入成本。
不将原版内部查询/分配记零，不保证原生GPU故障有整事务回滚；校验错误的输出保留已实测。

SDK版本条件精确为CUDA toolkit/runtime13.3、CCCL3.3.4、SM89；未知环境准备时拒绝候选。
`HostWaitPolicy`只控制Graph跨队列依赖，I32 Scan内部等待仍由`blocks_host=true`声明，不会被该
选项消除。资源/打包/失败接缝补充核对共8分钟，父卡共享源码/设计累计 **96/120、余24分钟**。
实现规模按统一脚本为Compute+Graph **56,172行/244文件**，不含测试/vendor；新增代码是描述符、
原版调用、标量key/layout投影和资源adapter，没有自写scan/sort/reduce kernel。

状态仍AUDITED，整卡门仍INCONCLUSIVE：WSL GPU、两Mac CUDA-OFF、host插桩、三交付、最终候选
复核及不可变候选P1尚未运行。本检查点不签父卡21行、两行四平台完整验收或新增覆盖格。

## I5 Gather 增量：待审有限定界

- baseline：`9d65d637`，工作树干净。具名触发是F02.1 embedding/索引读取；原版ggml get_rows
  没有checked动态越界返回，M0 CUDA-I/CCCL可承载相邻indexed primitive。I1–I4不回退，也不
  因增量已接就继承尚未运行的WSL/交付/插桩/P1门。
- 只扩`compute_provider_cuda`同一CCCL identity，新增原版`thrust::gather`；原版DeviceReduce
  Min/Max验证I32/I64索引，回传两个I64控制标量，成功后才gather。caller声明unchecked时不重复
  验证；空输出按共同合同no-op。库transform/permutation iterator的函数对象仅做dtype位宽和
  source/index逻辑坐标到byte stride的地址投影，没有自写gather loop或kernel。
- 接受plain 1/2/4/8-byte source、I32/I64 indices，公共合法rank/axis/byte stride与dense输出；
  output/indices各≤INT32_MAX，存储跨度≤PTRDIFF_MAX。checked负数或≥axis width均OutOfRange，
  用户output完全不改。输入/output不得重叠；源张量不缩窄为F32，保留原始storage位。
- caller workspace仅保存两个控制标量及原版reduce临时区；每invocation独立，实际host等待以
  `blocks_host`声明。沿同一原生stream/Completion/冷版本条件，不创建第二套runtime。
- 预声明：先原版I32/I64、全部storage位宽、坏索引/输出不变资格；再产品的所有axis、rank6/8、
  非连续source/index、重复/空/边界/巨大I64索引、资源/失败/恢复；普通Direct和TopK→Gather的
  Graph组合、两slot/旧结果，继承本卡其它适用平台/三交付/不可变候选P1。
- 不接Scatter、不改冻结kernel/公共schema/Graph产品，也不开展自研比较。本次前置定界与
  原版gather/合法状态流核对计9分钟，父卡源码/设计共 **105/120、余15分钟**；不新增研究预算。
- 当前只允许原版资格探针；I5未AUDITED之前不得改产品。失败最多两次源码驱动纠正，预算耗尽
  明确裁决；整个B1及64×4需求分母保持不变。

### I5 AUDITED：原版接法冻结

[原版资格](../../../../evidence/correctness/compute-llm-moe-m2-b1-cuda/gather-qualification.json)实跑
40组（4种storage位宽×I32/I64×5种索引payload），奇数字节stride、重复/变化索引、负数、axis
width与最大有符号索引，以及非法时整个输出不变均通过。未调用HYLE/frozen实现或计时。

产品仅扩既有私有Program描述符、同一CCCL proposal/prepare/submit和factory说明：原版
`DeviceReduce::{Min,Max}`与`thrust::gather(thrust::cuda::par_nosync.on(stream), ...)`，标量函数对象
只负责逻辑source/index地址换算和I32→I64类型提升；storage按同宽无符号类型传递，不做数值转换。
checked原版归约先裁决`min>=0 && max<axis_width`，只回传16字节；unchecked按公共caller前提
省略验证，空输出不读payload。原版Thrust异常在adapter边界保留CUDA错误码，可恢复分配错误
转现有Status；Provider晚期错误仍保留已接受prefix。无外部capture、无别名，Graph叶子自动消费。

这仍是I1–I4同一owner/SDK/资源与执行方式；Gather新增索引layout是冷Prepared描述符，不新增
常态线程/队列或张量常驻副本。两条输入的resource/alignment都要验证，caller工作区活到Completion。
原版资格及接缝核对计3分钟，累计 **108/120、余12分钟**。本节与registry摘要提交后才能实施I5；
全部产品/跨平台/三交付/P1仍待验证，不用原版probe填完整覆盖矩阵。

### I5 Windows 开发检查点

原版Gather已进入同一CCCL Provider及普通Direct/Graph；Windows Release/shared实跑
**10/10 CTest、105次GoogleTest执行**通过，其中Provider53、普通Direct8、普通Graph26、
ggml Compute13、ggml Graph5。原始构建、测试和精确实现身份见
[Gather receipt](../../../../evidence/correctness/compute-llm-moe-m2-b1-cuda/gather-windows.json)。

- 15种plain存储类型、I32/I64、三个axis与source/index各两种layout共360组；另有rank8、
  rank6/scalar index、零stride。独立host索引oracle逐位检查，无浮点转换或旧数学比较。
- checked的负数、轴宽、I64最大/最小值均使整个输出保持原值，槽退役，新槽可恢复；
  workspace缺失/不足、source/output与index/output精确/部分重叠拒绝。unchecked无工作区，
  空外维/空索引不读取非法payload；host resource冷拒绝。共享metadata OOM/句柄寿命回归通过，
  不冒称注入了GPU原生OOM。
- 普通Direct自动绑定CCCL；普通Graph使用默认prepare_replay，TopK索引直接喂给Gather，
  两queue/slot重放变化输入并保留另一slot旧结果；独立失败Graph验证输出不变及fresh恢复。
  旧capture测试显式要求prepared，保持既有capture和异步失败门，没有修改冻结kernel。
- I5产品构建与首轮完整定向测试直接通过，没有新增产品根因修正或性能样本。此次容量、
  别名与资源接缝复核计2分钟，父卡源码/设计累计 **110/120分钟、余10分钟**。

P0：checked每次包含两次原版GPU归约、16字节控制摘要回传/host等待和一次原版Gather；
workspace只含摘要及SDK临时区。unchecked省略检查，原版Gather仍按输出元素进行地址投影。
每slot持独立caller工作区，继续使用原生stream和Completion，不新增runtime/线程或张量副本。
原版dispatch与必要同步全部计入成本；这里不签吞吐/延迟或SOTA。

统一Compute+Graph实现 **56,374行/244文件**，排除测试/vendor。资格源和日志已归档，清理
具名Gather探针目录，复用现有CPU/CUDA开发树。状态保持AUDITED；WSL GPU、两Mac CUDA-OFF、
host插桩、relocated三交付、最终候选复核和不可变候选P1尚未运行，聚合INCONCLUSIVE。
父卡21行及完整覆盖矩阵仍未签；M5开始前停下。
