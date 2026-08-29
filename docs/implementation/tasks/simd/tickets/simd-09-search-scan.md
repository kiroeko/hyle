# SIMD-09：Search 与 scan

- ticket state：`ACTIVE / AUDITED`
- kind：product capability/correctness
- depends on：SIMD-08
- budget：1 工程日；性能预算 60 分钟
- input commit：`8c0aadf00d6a3d54e12ad39a0d587be5cbcdcad6`

归属当前 extrema-index 草稿。确认 count/find/value extrema/index extrema/minmax/is-sorted 与 inclusive/
exclusive scan 的类型矩阵、first-tie、empty、NaN、overflow、tail、alias 和 Runtime 版本合同。scan 必须
同时提供 in-place 与 `const input -> output` 路线，exclusive 支持显式 init；custom operation 继续使用标准库
或 expert source，不增加阻断内联的 runtime callback。extrema index 若不能在所有支持 backend 上正确保持
first tie/first NaN，则重写或删除草稿，不能只保留 scalar 假完成。

P1 仅使用 f32 count/argmax 与 u32 scan 的 cache/streaming consumer 行；其余由 property/differential 覆盖。

## 资格化审计

- 锁定 Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c` 已提供 target-bound
  `Count`、`Find`、`IsSorted`、`MinValue/MaxValue` 以及 first-tie、长区间分段安全的
  `IndexOfMin/IndexOfMax`。现有 `native/search.h` 重复实现前五项；未提交 extrema-index 草稿又近似复制
  后两项并添加 HYLE NaN 语义。草稿继续裁决为 `REWORK`：整数/普通机制直接借 Highway，只有“任意
  floating NaN 时返回第一 NaN”及单遍 propagating minmax 属于 HYLE semantic gap，不能维护完整平行
  vendor 算法。
- llama.cpp `7584430716ee229751771ed0d6bbcb780d105eeb` / ggml
  `36da57138425487184aa1da2eee2cde155909c6f` 的 CPU row 实际包含 f32 argmax 与 out-of-place f32
  cumsum；采样、greedy graph 和多种模型路径均消费它们。ggml argmax 当前选择最后 tie，而 Highway 与
  C++ `min/max_element` 选择第一 tie；HYLE 明确采用第一 tie，不能把不同语义的 ggml 循环直接当 oracle。
  标准 `inclusive_scan/exclusive_scan` 与 oneDPL 同样证明 input/output 与 exclusive init 是普通 API 形状，
  不是 benchmark 扩张。
- 当前 `scan.h` 只有 in-place、zero-init 的六类型入口，无法直接承担 ggml 的 `const src -> dst` row 或
  非零 offset prefix。`scan_dispatch.cc` 的 signed scalar tail/pure 路线以 signed `+` 后再 cast，溢出已经
  进入 C++ UB，不能证明文档声称的 modular arithmetic；现有测试没有覆盖 overflow、out-of-place、init
  或 overlap 合同。
- scan 还在 public call 内维护一组 `static ScanFn` resolver，与 SIMD-06 的私有 Runtime target authority
  平行；`Runtime::scan.isa` 只作诊断，实际函数指针由另一套平台分支决定。Linux-x64 全平台强制 scalar
  也只来自旧单机 benchmark 叙事。候选必须让 metadata 与实际 pointer 同源；任何 lower-tier policy 只可
  由本票同合同证据保留。
- `native/search.h`、`native/scan.h` 只有 target wrapper 与 `native_test` 自消费，仓库外没有语法 consumer；
  两者依赖 migration-only `vec.h`，前者复制 Highway contrib，后者又让 fixed 与 ordinary 可能分叉。
  普通 pointer/count API 是产品面；custom predicate/op 由 `std` scan/search 或 direct source expert template
  承担，不为它们增加阻断内联的 runtime callback。
- 现有 `search_bench` 读取已私有化 Runtime，混合 std/Highway/public 不同 dispatch 合同；`scan_bench` 把
  refill 放进计时区、无 A/A/反序/布局控制。二者均为 HISTORICAL/DIAGNOSTIC，不能签本票。

## 冻结合同与有限 FIX 集

1. 保留十种普通 scalar 类型的 count/find/value min/max/minmax/is-sorted，新增
   `index_of_min/index_of_max`。empty index 返回 `n`，tie 取第一项；float equality 使用 IEEE `==`，
   value extrema 遇任意 NaN 返回 quiet NaN，index extrema 返回第一 NaN。`n == 0` 可传空 data；
   `minmax_value` 的两个输出指针始终必须有效且互不相同。全部 O(n)、O(1)，无 heap/lock。
2. scan 核心改为 `const T* src, T* dst, n`：inclusive 无 init，exclusive 显式 `init`；原地 zero-init/
   explicit-init spelling 只作 inline convenience。支持现有 u/i32、u/i64、f32/f64；exact alias 或完全
   disjoint，部分 overlap 未定义；`n == 0` 可传空指针。整数逐 lane 按 modulo `2^bits` 定义且任何 scalar
   路径不得依赖 signed-overflow UB；float 允许 target-dependent association。8/16-bit、widening、custom
   op、segmented/parallel scan 路由标准库、source expert 或 Compute，不进入本票。
3. search target TU 直接采用 Highway contrib；只保留 NaN propagation/combined minmax 的最小单遍 HYLE
   gap。scan 在同一 direct-Highway target TU 组合 load/add/slide/store，不复制 register DSL。删除无真实
   consumer 的 `native/search.h`、`native/scan.h` 迁移面；ordinary API 不暴露 vendor type。
4. scan function pointers 与真实 `isa/features` 进入唯一私有 Runtime 表，search 也取得独立 module metadata；
   ordinary 每个完整 buffer 最多一次 cached indirect call，exact 直接绑定 suffix。Runtime layout 只在本票
   增加一次版本；不得新增公开 selector、第二 vendor runtime、常态 metadata 或逐元素 callback。
5. property/differential 覆盖十类型 search 的 zero/tail/first tie/first NaN/Inf/signed-zero/长 counter segment，
   scan 的 in/out/exact alias/init/modular overflow/float tolerance/tail；同一 SVE/RVV binary 跨多 VL/VLEN，
   `search.h`/`scan.h` self-contained 及三种交付实际消费。P1 只测 finite-normal f32 count/argmax 与 u32
   scan 的 4096/1M row；candidate 与 direct Highway/同合同 composition 必须同 ISA、同 TU、同
   semantics/dispatch/resource，正反序、A/A、双 layout，5% material margin。不同 tie/NaN/init 或 refill
   计费一律 `NOT_COMPARABLE`。

本票只允许上述五项。predicate/custom-op callback、contains 命名糖、parallel/segmented scan、tensor axis/
stride/threading、top-k/order statistic、GPU prefix 与模型融合分别由标准库、source expert、Ordering、Compute
或 accelerator owner 承担；不得在本票扩张。
