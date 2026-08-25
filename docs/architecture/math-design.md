# Math 模块边界：待重新裁决

> **V2 状态：`OPEN / NOT_REVIEWED`。** 本页记录历史来源和审查问题，不批准当前 math 形状。

## 历史背景

`math` 不是从一组明确用户任务正向设计的模块。它来自 clay math 迁移：多数向量计算后来由
HYLE simd 承载，少量没有进入 SIMD 的标量/bit/numeric helper 留在一个很小的 math 目录。这造成
用户难以判断“某个计算应该找 `hyle::simd` 还是 `hyle::math`”，也容易让两个模块逐步重复。

目录存在、代码量小或已有调用者都不能证明它应继续作为公共模块。

## 待裁决路线

math 轮次必须逐能力建立 ledger，并只能选择以下去向：

| 能力性质 | 默认 owner |
|---|---|
| C++23 已清楚、正确且无性能缺口的标量 primitive | 直接使用标准库；HYLE 不包一层改名 |
| lane/vector、dtype conversion、向量 transcendental、bitpack | simd |
| 只服务某个数据结构或 kernel 的 helper | 对应 container/compute/consumer internal |
| 通用标量 numeric primitive，标准库缺失且多个消费者需要 | 才考虑保留 math public surface |
| 线性代数、稀疏求解、FFT、几何和模型运算 | 成熟依赖、compute 或上层领域库 |

若所有现存能力都能自然归入其他路线，删除 public `math` component 比维持一个含义模糊的小模块
更可维护。若保留，名称、类型、精度、异常值、`constexpr`、向量对应物和复杂度必须让用户无需
查实现就能选对。

## 审查要求

- 从公开头、实现和全库语义消费者生成逐 symbol ledger；
- 对照 C++23 `<bit>/<cmath>/<numbers>`、HYLE simd、Eigen 及适用成熟库；
- 删除仅为迁移兼容、benchmark 或旧编译器补丁存在的 wrapper；
- 对保留 primitive 做精度/exceptional-domain oracle、三工具链 codegen 和真实 consumer 证据；
- 防止标量 helper 反向使 simd 依赖 math，或 math 吸收 compute/领域语义；
- 三种 C++ 交付都验证；Native FFI 为 `N/A`。

在这轮完成前，math 只能作为既有内部/预发布实现使用，不应向用户承诺功能完整度、稳定 API 或
性能证书。
