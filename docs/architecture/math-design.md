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

当前准入三类能力：非负整数的 ceiling division 与可表示 multiple round-up；不可信 integral
shape、stride、offset 与 byte-count 的 exact checked conversion/arithmetic/aggregation；以及明确选择
clamp policy 的 saturating integer operation/conversion。它们在
tensor/block/launch shape、allocation sizing 和 layout validation 中反复出现，C++23 又没有统一的
自然 C++ 返回形状。ceil/round 的正常入口把
非负值、正除数和可表示结果当作程序不变量并 fail-fast；`try_` 入口用 `std::optional`
处理不可信 shape/count。同一调用只允许一个非 bool integral 类型，避免隐式 signed/unsigned
promotion 改变结果。

Exact checked 路线提供 add/subtract/multiply/divide/remainder、multiply-add、span sum/product 与显式
destination 的 `try_integer_cast`。成功值是逐步精确结果；overflow/underflow、不可表示 conversion 或
非法 divisor 返回 `nullopt`，不返回 wrapped/saturated value。`try_remainder(min,-1)` 特判返回数学上的
精确零而不执行 UB；对应 quotient 不可表示，`try_divide` 失败。可信且已经验证的普通算术仍直接用语言
operator。

Saturating 路线使用 `saturating_add/subtract/multiply/divide/cast`，返回最近的可表示值；它不是 checked
失败的替代品。除零没有可定义的 clamp，因此 trusted saturating divide 以 Core invariant fail-fast，
不可信 divisor 必须先走 `try_divide`。名称遵循 HYLE 可读 verb，不冒充尚未由 C++23 提供的标准 ABI。

## 依赖与组合

`math` 是一个不可再拆的 header-only factor。当前唯一 HYLE 依赖是 `core`：普通 `try_`/saturating 入口只依赖
C++ 标准库和编译器 overflow primitive，fail-fast 的 `ceil_div`/`round_up_to_multiple` 复用 Core 的
release-surviving assert。编译器 intrinsic 是实现机制，不是产品组件或可选第三方依赖。因此中央图的
完整产品边就是 `math -> PUBLIC core`；Math 不依赖 SIMD、Platform、Mem 或外部数值库。

Math 与 SIMD 在产品能力图上互补，在构建图上相互独立：

| 计算层级 | 语义 | owner |
|---|---|---|
| control plane | 单个整数 shape/count/stride/offset/byte-size 的可表示性 | Math |
| data plane | lane、row、buffer 的 dtype、rounding、packing、activation 与向量数学 | SIMD |
| model plane | tensor layout、模型算子、图级策略与融合 | Compute 或消费层 |

这不是“SIMD 没有的函数都放进 Math”的差集规则。普通标量数学仍直接使用标准库；只属于某个 wire
layout、容器或 kernel 的计算仍由其 owner 持有。反过来，较高层若真实需要 Math 已公开的通用 checked
scalar 合同，应声明直接依赖并复用，而不是因为模块边界再写一份。只有公式本身定义领域 layout、失败
状态或成本合同（例如 bitpack 的 packed-size 分解）时，专用实现才继续留在 owner 模块。

新增候选进入 Math 前必须同时回答：它是否是单值 integral control-plane 计算；C++23 是否缺少自然
等价物；失败语义是否跨 consumer 稳定；它是否至少有真实消费或多个成熟 peer 的重复证据。缺少任一项，
不能仅凭 SIMD/peer 有同名函数或 benchmark 方便性扩大 Math。

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

所有 scalar 入口均为 `constexpr` inline value function，O(1)、零分配、零锁、零常态 metadata；span
sum/product 是 O(n) 单遍、短路、零分配，empty identities 分别为 0/1。
ceil/round 使用 quotient/remainder 形状，避免 `(a + b - 1)` 的先行溢出；round-up 在加法前
检查结果是否可表示。checked add/sub 在 unsigned bit domain 判定 carry/sign overflow；runtime checked
multiply 使用工具链 overflow primitive 或高半积，constant evaluation 使用先判界再乘，均不执行
signed overflow UB。`math` 公开头使用 core fail-fast primitive，因此中央组件图中是
`math -> PUBLIC core`。组件本身无 binary artifact；source/static/shared SDK 都交付同一公开头和
`hyle::math` INTERFACE target，不制造虚构的 math archive。Native FFI 不适用。
