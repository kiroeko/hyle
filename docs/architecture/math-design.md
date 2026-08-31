# Scalar math 边界

## 历史背景

`math` 不是从一组明确用户任务正向设计的模块。它来自 clay math 迁移：多数向量计算后来由
HYLE simd 承载，少量没有进入 SIMD 的标量/bit/numeric helper 留在一个很小的 math 目录。这造成
用户难以判断“某个计算应该找 `hyle::simd` 还是 `hyle::math`”，也容易让两个模块逐步重复。

目录存在、代码量小或已有调用者都不能证明它应继续作为公共模块。

## 决策

`math` 是极小的 header-only scalar numeric 组件，只承载标准库缺失、语义稳定且
能防止溢出/输入边界重复出错的值计算。它不是通用数学伞头，也不是 SIMD 的 scalar
fallback 目录。

当前准入两类能力：一类是非负整数的 ceiling division 与可表示 multiple round-up；另一类是
不可信 integral shape、stride、offset 与 byte-count 的 checked add/subtract/multiply。它们在
tensor/block/launch shape、allocation sizing 和 layout validation 中反复出现，C++23 又没有统一的
自然 C++ 返回形状。ceil/round 的正常入口把
非负值、正除数和可表示结果当作程序不变量并 fail-fast；`try_` 入口用 `std::optional`
处理不可信 shape/count。同一调用只允许一个非 bool integral 类型，避免隐式 signed/unsigned
promotion 改变结果。

checked arithmetic 只提供 `try_add`、`try_subtract`、`try_multiply`。成功值是精确数学结果，overflow/
underflow 返回 `nullopt`，不返回 wrapped 或 saturated value。可信且已经验证的算术直接用语言
`+/-/*`；Math 不增加 fail-fast operator wrapper。binary primitive 可以显式组合出 tensor element count、
byte size 和 end offset，而不把 mixed-type promotion、variadic fold 或某个模型 layout 固化进 API。

## 与其他 owner 的边界

| 能力性质 | 默认 owner |
|---|---|
| C++23 已清楚、正确的标量 primitive | 直接使用标准库；HYLE 不包一层改名 |
| lane/vector、dtype conversion、向量 transcendental、bitpack | simd |
| 只服务某个数据结构或 kernel 的 helper | 对应 container/compute/consumer internal |
| power-of-two 地址对齐 | mem |
| 浮点近似等值、死区符号和阈值 | 由 consumer 或数值领域库选择 absolute/relative/ULP、NaN 与 zero 政策 |
| 线性代数、稀疏求解、FFT、几何和模型运算 | 成熟依赖、compute 或上层领域库 |

`sgn(eps)` 和 multiplicative `approx_eq` 不再是公开能力：它们把消费域的容差策略伪装成了
通用数学事实，对 NaN、无穷与近零的结果也不能从旧函数名自然推导。

## 成本与交付

所有入口均为 `constexpr` inline value function，O(1)、零分配、零锁、零常态 metadata。
ceil/round 使用 quotient/remainder 形状，避免 `(a + b - 1)` 的先行溢出；round-up 在加法前
检查结果是否可表示。checked add/sub 在 unsigned bit domain 判定 carry/sign overflow；runtime checked
multiply 使用工具链 overflow primitive 或高半积，constant evaluation 使用先判界再乘，均不执行
signed overflow UB。`math` 公开头使用 core fail-fast primitive，因此中央组件图中是
`math -> PUBLIC core`。组件本身无 binary artifact；source/static/shared SDK 都交付同一公开头和
`hyle::math` INTERFACE target，不制造虚构的 math archive。Native FFI 不适用。
