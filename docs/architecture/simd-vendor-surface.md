# SIMD：第三方实现、源码扩展与路由边界

SIMD 的公共语义由 HYLE 拥有，可移植寄存器与部分算法借用 Highway/xsimd。普通 pointer/span
接口不泄漏 vendor 类型；专家源码路径可以在 target-bound TU 内使用锁定 provider，但不把其
寄存器布局变成共享库 ABI。总体设计见 [SIMD 架构](simd-design.md)，修改步骤见
[SIMD 路由维护](../developer/maintenance/simd-dispatch-guide.md)。

## 三种来源分别记账

直接产品依赖、checked-in 移植和性能对照 checkout 不是同一种关系：

| 来源 | 所在路径或作用 | 谁负责什么 |
|---|---|---|
| Highway direct | target primitive、whole-buffer、fixed/fat kernel | 上游负责 register/math 实现，HYLE 负责语义、选择与交付 |
| xsimd direct | 适用类型/target 上的完整 candidate | 上游负责实现，HYLE 负责 adapter、编译期选择与用户合同 |
| Highway integrated port | `src/simd/sort/vqsort_hyle/` 与相关 TU | 保留上游算法身份，HYLE 维护必要的 compiler/scalable/POD/static-entry 差异 |
| HYLE body | `include/hyle/simd/provider/hyle_winners*-inl.h` 等有限实现 | HYLE 负责真实语义、ISA、复杂度或完整消费者收益 |
| benchmark peer | 对照工程或测试依赖 | 不自动成为产品依赖，不进入安装消费者闭包 |

simdutf 主要承载 Container 的 UTF/Base64 机制，不属于 `hyle::simd` 的寄存器或算法面。
同一个第三方可服务多个模块，其公共语义仍归实际模块 owner。

[HyleDependencyLock.json](../../cmake/HyleDependencyLock.json) 记录 direct/port 的 revision、archive、
内容 hash 与许可来源。两种 Highway 来源可以有不同 revision，不能用一次依赖升级覆盖掉 port
的真实身份。改路径或 namespace 不抹去来源；许可证与交付细节由 [构建架构](build-system.md)
和 [许可说明](licensing.md) 统一处理。

## 公开接口与专家源码接口

普通用户选择整批 API，不接触 Highway/xsimd vector、mask 或 tag。寄存器在实现 TU 中创建和
消费，跨 TU/shared boundary 的 entry 使用 pointer/count/POD。由此 vendor 升级不会直接要求
消费者迁移另一套寄存器 ABI，但 HYLE 自己仍须检查数值、alias 与 target 合同。

专家 kernel 通过 `hyle::simd_expert`、`hyle_target_simd()` 或 `hyle_add_fat_kernel()` 编译。
它们带入中央锁定的开发头，故源码级使用仍受该 SDK/工具链版本约束；“无 vendor ABI”不等于
任意 vendor 版本都能源码兼容。

adapter 只补 HYLE 必需的 dtype/target/语义 glue，不重新命名完整 Highway/xsimd 操作集。
某个自有算法获益，不足以证明其历史 register DSL、无关 operation 或整套平行 backend 应保留。
也不能为了迁就两个 provider 的局部胜出，在元素循环中逐 primitive 来回转换寄存器。

优先使用 provider-coherent 的完整 kernel。只有最终 codegen 确认没有额外 copy、spill 或第二层
dispatch 的 bridge 才有意义；否则选择另一个完整实现，或把获益算法 lower 成当前 provider。

## Provider 选择与 CPU 选择是两件事

机器链路是：

```text
scripts/simd-op-routing.json + simd-family-routing.json
                           |
                           v
       生成的 operation_manifest / routes_generated
                           |
                           v
          RouteFor -> Candidate<provider>::run
```

provider 根据 operation/family、lane/conversion 合同、target 与 compiler/version 在编译期确定。
RouteFor 的默认模板表示没有有效实现；实际可调用路线由生成特化给出，缺项或被删除的别名不能
偷偷落入未审查的 fallback。这是可执行配置完整性，不是开发审批状态。生成清单规模与曾经获胜
的操作数量不属于长期公共合同，不在本页复制。

普通 unknown-CPU artifact 在冷初始化解析 CPU/OS facts 与实际编译 target 的交集，缓存 family
入口；随后一次 whole-buffer 调用只走一层 target 间接调用。provider 已在该 target body 内确定，
不能再接一层 vendor 动态分派，更不能逐元素探测 CPU。

fixed/exact source kernel 已知 ISA，CMake 生成准确 flags、宏与静态 provider body，运行时没有
target selector。fat source kernel 为同一 implementation 生成 per-ISA TU，在 pointer/POD 入口
选择一次。宏能消除已知 target 的选择，不能预知可分发二进制最终运行的 CPU。

若直接使用 Highway dynamic pointer，必须先完成 ChosenTarget 初始化，或由 HYLE 合法绑定具体
target symbol。把未初始化的 detection trampoline 当成稳定函数地址缓存，会让每次调用重新探测。

## 编译事实必须与 runtime gate 一致

Platform 提供 CPU/OS facts；provider 提供可编译实现与 compiler 限制；SIMD 提供 canonical feature
要求和 family policy。三者取交集，不是全库找一个“最高 ISA”后让全部模块照用。

selected provider 的头、namespace、definition 与 ISA options 在同一个 target 声明中产生。
未选 candidate 不应因同一文件里顺便 include 而进入该 body 的实例化闭包。普通 consumer baseline
也不能被私有宽 ISA TU 抬高。

HYLE canonical x86 tier 不默认承诺 AES/PCLMUL；Highway TU 使用其禁用开关对齐实际 gate。
专用 crypto kernel 需要自己的扩展事实，不能为满足上游 target 名字偷偷提高部署要求。

经典 cl.exe 与 clang-cl 不同。锁定 Highway 对经典 MSVC 的 AVX-512 源码路径有限制，专家
`hyle_target_simd(ISA AVX512)` 明确拒绝，source-fat 路径采用可用的较低 target，不能伪称静态
AVX-512。HYLE 的某个自有 AVX-512 body 可编译，也不意味着整个 vendor/expert surface 都可用。

## 为何仍保留 VQSort port

完整上游 ordering 类型与操作集的直接实例化，曾在经典 MSVC 下产生实际构建成本与适配问题。
当前保留 port 只承载需要的 compiler/scalable glue、HYLE 公共 POD 与 static entry，以及已有
产品合同，不扩成第二套公共寄存器库。

port 是维护债，不是永久选择。direct 接入能保持相同稳定性、NaN/ordering、layout、临时存储和
交付边界时，应删除不再必要的差异。升级时分别看上游算法修复和本地适配，不能只用新的 direct
Highway revision 声称 checked-in 算法也已更新。

## 新实现与升级如何收敛

先比较语义、accuracy、alias、异常、ISA 与空间复杂度，再比较完整消费者成本。Highway 是同合同
标准 primitive 的维护默认；xsimd 或 HYLE 实现按真实适用的类型、ISA、编译器保留有限例外。
稳定的小差距可以调查，但某个 buffer 地址、benchmark 标题或执行顺序不能成为产品路由维度。

新优化先作为完整 candidate body 存在，再更新单一机器配置并生成头；不手改生成物。升级依赖
只扩大到实际受影响的 implementation groups、target、数值边界和直接消费者，不把旧实验清单
当成每次全平台重跑的义务。port 内容 hash 用现有构建工具重算，不手写猜测值。

tensor stride/broadcast、模型量化、GPU backend、per-ISA DLL、稳定跨工具链 C++ ABI 都不因 vendor
升级进入 SIMD。它们由拥有相应资源和执行语义的模块处理。
