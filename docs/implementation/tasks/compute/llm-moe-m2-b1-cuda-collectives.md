# M2-B1 CUDA-I：成熟 CCCL 索引 collectives 接入

## Closure card

- component / owner：`compute_provider_cuda`、`compute_route_cuda`；`/root`。
- baseline：`93b5095f`，Windows canonical，工作树干净。当前阶段 SCOPED，审计冻结前不写产品。
- trigger：[M0 CUDA-I](llm-moe-provider-map.json)已将成熟 CUB scan/sort 分给现有 CUDA Provider。
  前者仍绑定 M0 空 FIX，须换成本卡；普通 CUDA 工厂需要新成熟选择，满足 M1-C 新能力 reopen。
- 需求：B1 的 F02.4 前缀计数和 F04.6 TopK，以及同卡 F19 的资源/失败/重放；不缩小父卡21行。
  [父卡](llm-moe-m2-b1-cuda.md)其余域仍由原 owner 推进，完整覆盖未完成。
- 主纵切：MoE router 的 stable TopK→专家计数 Scan，两个请求独立，值/索引双输出和错误保留。
  正交纵切：无模型注解的任意轴/stride 扫描与排序，NaN/相等值、整数溢出、旧结果/重放。
- 来源：CUDA Toolkit13.3.73 随附 CCCL/CUB3.3.4（`CUB_VERSION=300304`），原版
  DeviceScan/DeviceRadixSort/DeviceReduce 与标准迭代器/tuple 操作。版本和 BSD-3 许可随原 SDK
  及三交付记录，不引入第二个 toolkit 下载器。
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
| I3 | cold shape/workspace、请求私有资源、失败与原生stream完成 | GPU上检查所需标量可回传host控制，不能把张量模型计算放CPU；检查/同步/转换成本全计；错误不能伪造完成或写部分用户输出 |
| I4 | 现有factor内的真实CCCL Provider及普通CUDA工厂、Graph leaf、文档/三交付 | 新identity可审计；不修改旧自研kernel和选择内核；无新插件框架或万能executor |

## 预声明验证

- 先一组原版资格 probe：Scan I32边界/overflow、exclusive最终溢出；stable TopK的NaN/Inf、
  signed-zero、同值顺序、两行；独立host oracle，不读旧自研数学。
- 产品阶段扩展任意轴、非连续输入、空/尾/容量、两输出、无原位/部分重叠、两slot与旧结果，
  以及workspace/OOM、提前close和失败排空。普通Direct/Graph必须自动选择具名成熟Provider。
- Windows/WSL真GPU、适用host插桩；Air/Mini CUDA-OFF；稳定程序的一份source/static/shared
  relocated消费旅程，producer隐藏、依赖/许可证/DLL邻置。重型串行，parallel2，不弹UAC。
- selector选中门、docs/closure（状态更改用明确previous-ref）、最终候选复核及适用P1。
- 当前仅SCOPED；所有能力/验收未签，没有完整新增需求PASS，不用probe代签产品覆盖。

## 已核对的原版事实

- ggml `getrows.cu`直接用运行期索引读地址，没有本合同的checked失败通道；不能仅绑定函数名。
- ggml `cumsum.cu`支持原版scan，但I32路径不提供整个输出不变的溢出裁决。
- 原版CUB提供InclusiveSumByKey、独立Reduce和稳定RadixSort；NaN默认按浮点bits排序，不自动
  满足HYLE的NaN尾部/稳定索引要求。该语义必须由合法的库key/decomposer接法资格化。
- 本卡不授权完整Gather/Scatter接入，也不消除其余父卡缺项；新的库/owner变化必须具名审计。
