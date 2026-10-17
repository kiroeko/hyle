# SIMD：静态实现选择与批量边界

SIMD提供连续buffer上的计算及target-bound源码扩展，不承担tensor stride/broadcast、设备调度或模型政策。
普通接口不泄漏Highway/xsimd寄存器类型，具体用法见[用户文档](../user/README.md)。

## 为什么不维护第三套可移植寄存器库

Highway/xsimd承担可移植寄存器、编译器和ISA演进。Hyle保留自己的语义、数值/alias合同与批量算法，
只在真实语义、复杂度、指令或完整消费者收益上保留有限自有实现。
成熟实现可自然承载时不复制；自有算法获益也不证明它的历史register DSL值得保留。

选择细化到operation/family、scalar type、target/compiler，不能把一个获胜点扩成全平台赢家。
同合同维护默认优先Highway，xsimd和Hyle优化按实际适用条件保留。
比较先核对accuracy、NaN、稳定性、layout、遍历和临时存储，再评估完整调用成本。

## 两次不同的选择

Provider选择在编译期由route manifest、模板和target namespace消除。
未知部署CPU的target选择发生在进程内的冷初始化；随后每个whole-buffer调用只经过一层已缓存的kernel入口。
不能先由Hyle选ISA，再进入会重复探测的vendor dynamic dispatch，元素循环里也不能逐primitive跨vendor桥接。

已知ISA的源码/exact路径可以直接绑定并内联；fat源码为同一implementation生成每ISA TU，
在pointer/POD批量边界选择。宏能消除已知target的分派，不能预知可分发二进制最终运行的CPU。
很小的融合循环用源码扩展摊销调用成本，而不是要求普通预编译库假装编译期零开销。

Provider-coherent完整body优先，混合register类型只有确认无额外copy、spill或二次分派才有意义。
否则选择另一完整kernel，或把获益算法lower为当前target的原生实现。

## 事实与实际target分开

Platform拥有OS/CPU事实，SIMD拥有实际编译body、family policy与canonical feature requirements。
最终选择是CPU事实、可编译实现和产品规则的交集，不存在全库通用的“最高ISA”。
同一机器不同family可采用不同target，报告实际选择不能替代原始硬件探测。

ISA flags必须与runtime gate一致，不能为满足vendor target名字偷偷提高部署最低要求。
F16C、VNNI、BF16、AES等是有独立前置的扩展；公开inline所需ISA由owner传播，
更宽compiled TU不应抬高普通consumer的baseline。

经典cl.exe与clang-cl是不同frontend。锁定Highway对某些AVX-512实现有编译器限制，
expert固定target不得静默降档；普通fat路径则可如实采用较低的合法body。
Hyle自己的某个AVX-512算法可用，也不能推出全部vendor/expert surface可用。

## 数值与内存合同

RNE、饱和、NaN/total-order、稳定排序、prefix/alias、空输入与tail属于具体能力，不只是函数名。
标量fallback与宽ISA须保持相同语义，近似数学不能偷换完整精度路径。
FP8/f16/bf16 storage转换不包含模型block scale、tensor layout或量化政策，这些归Compute。

pointer/count边界保留caller所有权，算法临时存储和额外遍历要显式。
argsort、partial/select、stable/unstable与in-place/out-of-place不能按同一“sort”成绩混比。
随机数需区分可复现非加密批量流与OS安全熵，不因用于采样而把生成策略放进SIMD。

## 依赖与长期port

直接产品依赖、checked-in port和benchmark peer分别记账。普通头不泄漏vendor ABI；
专家源码target可以携带锁定开发头，但寄存器类型不跨共享库边界。
依赖来源和许可证由[中央锁与交付](build-system.md)统一处理，改名不抹掉第三方版权。

保留的VQSort port承担必要的compiler/scalable/public-POD适配：完整上游实例化存在实际构建/正确性限制。
它不是第二套公共register库；direct接入能保留合同并消除维护债时应删除port。
配置生成与维护命令见[SIMD维护指南](../developer/maintenance/simd-dispatch-guide.md)，架构不复制机器清单或旧获胜统计。
