# SIMD-04：数学 vendor 合并与 port 消减

- ticket state：`ACTIVE`
- kind：conditional product
- depends on：SIMD-03
- budget：1 工程日；性能预算 60 分钟

- input commit：`9d006769b6ccf3ff67291bab1ad6d4a43b446bfb`

以 SIMD-03 的 Highway 直接承载结果为默认，先资格化 function/domain/ULP/special-value 合同，再确认
xsimd 或 SLEEF 是否仍填补真实缺口。能由 Highway 等价承载的 xsimd/Highway checked-in math port 删除；
只有明确 accuracy/capability gap 才保留最小 fallback。输出 upstream/local-delta/license manifest、受影响
accuracy property、代表性 exp/tanh/erf sentinel 和最终 `DIRECT/DROP/MINIMAL_PORT` 裁决；不以一个输入域
的点值决定全函数 backend。

## 当前候选裁决

锁定产品机制为 Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c`。HYLE 不执行上游 CMake project，
由固定目标 TU 直接包含 header-only 实现；普通接口仍是 HYLE pointer/count，Highway 类型不跨 TU、static/
shared 或公开头边界。

| family | verdict | HYLE 保留内容 |
|---|---|---|
| Highway 已提供的 f32/f64 实数一元/二元、sincos、gamma、hypot | `DIRECT` | ordinary 完整域恢复、alias/尾部合同、runtime/exact 路由 |
| `erfc`、`exp10`、`fdim` | `HYLE GAP` | Highway primitive 上的最小组合与必要特殊值恢复 |
| `fmod`、`remainder`、`nextafter` | `SCALAR FLOOR` | 当前 vendor 没有同语义向量原语；保留 API，不保留未经证明的复制算法 |
| complex SoA | `HYLE GAP` | 直接 Highway 实数原语上的组合、branch-cut/异常 lane 与精确原位合同 |
| 旧 `math_hyle-inl.h`/table 与 xsimd port/TU | `DROP` | 产品、integrated-source lock、许可证投影和旧 scalable compile probe 一并删除 |
| xsimd `3b3151ff...` | `BENCH PEER` | 不进入产品；具名真实 consumer gap 出现时重新准入 |

这不是把实现责任改名：数学 port、平行 target body 与旧 dispatcher 合计从本轮 diff 中减少约 6.5k 行，
新的直接 Highway façade 约 0.4k 行。剩余 HYLE 代码只表达上游没有替 HYLE 决定的产品合同。

## API 与语义结果

- ordinary f32/f64 入口保持对应 `std::` domain/special-value 语义；上游明示有效域之外按 block/lane 使用
  scalar oracle，不把 vendor 前置条件暗中转嫁给用户；
- 补齐对称的 `exp_in_range_f32_batch` / `log_in_range_f32_batch`，与既有 f64 fast contract 一致。
  f32 exp 要求 finite `[-80,80]`，log 要求 finite normal positive；fast 入口不检查；
- unary 精确原位、任意尾部、signed zero、NaN/Inf、gamma、f64 full-domain ULP、complex 特殊值和 alias
  均进入当前 property；最新 Highway 的 tanh/erf/atan/cbrt 已能直接承担特殊值，不再支付重复 horizontal
  screening。`asinh(inf)` 的资格化反例证明其仍需 HYLE recovery，未因一台机器“看起来能跑”删掉合同。

## 有界性能结果（W1-WIN AVX2）

`math_bench --sentinel` 只测 exp/tanh/erf，输入 4096、同 TU 同 AVX2 peer，执行五遍并同时输出 65,536
样本 accuracy property。raw vendor 行被明确标成 valid-domain implementation ceiling，不冒充 ordinary API：

- f32/f64 `exp_in_range` 与 raw Highway 在约 `0.29--0.31 ns/elem` / `0.82--0.95 ns/elem` 同档；
  ordinary exp 的约 10--15% 差额来自其额外完整域恢复，属于不同合同，不归因为 wrapper/dispatch；
- HYLE tanh/erf 与直接 Highway 在观察 spread 内同档；f32 tanh 的 raw xsimd 在该单域约快 4--9%，且
  accuracy 为 1 ULP（Highway/HYLE 2 ULP）。这不足以按 benchmark 行给通用 math 增加第二产品 vendor，
  但作为真实 activation consumer 的待验证机制路由 SIMD-08；
- accuracy probe 观察 f32 max ULP：HYLE/xsimd/Highway 分别为 exp `1/1/1`、tanh `2/1/2`、erf
  `6/4/6`。HYLE 三项 special mismatch 为 0；raw Highway exp 在上游有效域外出现 1 项 mismatch，正好证明
  ordinary recovery 不是虚构开销。

## 已完成验证与剩余停止线

- W1-WIN MSVC 19.50 shared selected-component build：`PASS`；`Math.*` 30/30；
- W1-WSL GCC 14.2 shared selected-component build：`PASS`；`Math.*` 30/30；AVX2/AVX-512 target TU 均编译；
- W1-WIN MSVC 19.50 独立 SIMD consumer：源码嵌入、relocated shared SDK、relocated static SDK
  三种 journey 均 `PASS`；consumer 通过 public math/runtime API 证明实现与运行时依赖均已交付；
- dependency lock、直接 Highway clean identity、offline/source acquisition 与许可证路径已进入 CMake；
- 尚需 M1-MAC NEON、文档/closure checker 和最终 immutable output commit。完成前 ticket 保持
  `ACTIVE`，不把当前结果写成 SIMD 整模块性能证书。
