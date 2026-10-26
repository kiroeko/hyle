# Math：整数尺寸的精确计算

Math 是极小的 header-only 标量整数模块，只补 C++23 尚无自然返回形状、且会在尺寸与资源计算中
重复出错的能力。普通标量数学继续使用标准库；Math 不是向量计算的 fallback 目录，也不承担
线性代数或模型算子。具体接口和示例见 [Math 指南](../user/math.md)。

## 为何与 SIMD 分开

两类计算的失败与成本合同不同：

| 层级 | 处理的值 | 职责 |
|---|---|---|
| Math | 一个整数 count、shape、stride、offset、byte-size | 可表示性、精确 checked arithmetic 或明确 saturation |
| SIMD | lane、row 或连续 buffer | dtype 转换、packing、activation、向量数学与整批算法 |
| Compute / 消费层 | tensor、设备布局、模型算子 | 领域语义、资源范围、融合与执行 |

不能把“SIMD 还没有的函数”当成 Math 的准入规则。只属于 wire format、kernel 或某种容器布局的
公式仍由其 owner 持有；多个 owner 确实需要同一个通用整数合同，则应复用 Math 而不是各写一份。

Math 的唯一 HYLE 依赖是 Core，供可信入口使用始终生效的不变量检查。编译器 overflow primitive
是实现细节，不形成可选 provider。`hyle::math` 是 INTERFACE 产品；static/shared 配置不制造空的
Math archive，也不把 SIMD、Platform 或第三方数值库拉入其依赖闭包。

## 三种数值合同不能混用

**精确或失败。** `try_add/subtract/multiply/divide/remainder`、multiply-add、sum/product 与
`try_integer_cast<To>` 成功时返回精确值，否则返回 `std::nullopt`。失败不返回 wrapped 或 saturated
值，也不分配诊断对象。输入类型必须一致且为非 bool integral；需要混合尺寸类型时先显式 checked
cast，避免 signed/unsigned promotion 在检查之前已改变数学含义。

multiply-add 先检查乘积，再检查加法，不允许中间溢出后又绕回可表示结果。sum/product 单遍检查
每一步，空 identity 分别是 0 和 1。signed minimum 除以 -1 的 quotient 不可表示而失败；余数的数学
结果是零，必须特判，不能先执行会触发 UB 的语言 `%`。

**可信尺寸的不变量。** `ceil_div` 与 `round_up_to_multiple` 要求非负值、正除数/倍数以及可表示
结果，违反时 fail-fast。它们用于程序结构已经保证合法的尺寸；模型、文件或网络输入使用对应
`try_` 入口。实现不能使用 `(n+d-1)/d` 一类会在合法最终结果前溢出的捷径。负数的 rounding
direction 是另一种合同，不能从函数名自动猜测。

**明确选择 clamp。** `saturating_add/subtract/multiply/divide/cast` 返回最近的可表示值，适用于
业务本来就允许截到边界的场景。它们不是 checked failure 的无错误替代品。除零没有自然的 clamp
结果，因此 trusted saturating divide 仍要求 divisor 非零，外部输入先检查。

精确 sizing 常用于分配前证明大小安全，saturation 常用于数值更新；把前者的失败静默改成后者
可能把“非法形状”变成一个可分配但语义错误的 tensor。

## 离散 sizing 的边界

`try_lcm` 先用无符号 magnitude 求 GCD，再约分、乘法与判界，避免 signed minimum 取绝对值和
乘法 intermediate 的 UB。成功结果总是非负，任一输入为零时结果为零。欧几里得算法使其复杂度
与整数位数相关，不应把它描述成批量或向量化操作。

`try_bit_ceil` 先证明下一幂可表示，再调用 bit primitive；0/1 返回 1，有符号负值与不可表示的
下一幂失败。已经证明输入安全时直接使用标准 `std::lcm/std::bit_ceil` 即可。

这些能力只计算值，不建立内存权限。pointer/address alignment 归 Mem，tensor stride/alias 合法性
归 Compute，用户可见的资源拒绝错误归执行该操作的 owner。Math 的 optional 结果应在那个边界
转成有领域含义的错误，而非在标量 helper 里携带模型名或设备状态。

## 演进时保留的约束

新增能力应具有稳定的跨消费者整数语义，并确实弥补标准库空缺；普通 `<cmath>/<numbers>/<bit>`
功能不需要重新包装。浮点 approximate equality、NaN/ULP tolerance 与死区 sign 要由具体算法选择，
不能在通用 helper 中指定一个适用于所有模型的容差。

维护重点是 constexpr、signed/unsigned 边界、可表示性和中间步骤，不能仅核对普通正数算例。
共同错误与不变量边界见 [基础机制](foundation-design.md)，整批计算边界见 [SIMD 设计](simd-design.md)。
