# SIMD-09：Search 与 scan

- ticket state：`DONE`
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

## 实现与验证结果

- `search.h` 保留十种普通 scalar type 的 count/find/value min/max/minmax/is-sorted，并完成 first
  `index_of_min/index_of_max`。empty identity、first tie、first floating NaN、Inf/signed-zero、长 counter
  segment 与 Runtime variant 均有 differential/property 覆盖。
- `scan.h` 以 `const src, dst, n` 为核心，exclusive 接受显式 init，原地/zero-init 只作 inline convenience；
  u/i32、u/i64、f32/f64 覆盖 out-of-place、exact alias、tail、init 与 signed modular overflow。整数路径把
  signed storage bitcast 到 unsigned accumulator，不依赖 signed UB。
- search generated TU 直接使用锁定 Highway contrib，scan 直接组合 Highway load/add/slide/store；HYLE 只
  保留 floating NaN propagation、single-pass minmax 与产品 glue。重复 vendor DSL 的
  `native/search.h`、`native/scan.h` 及其自消费测试已删除。
- Runtime table 只由本票从 version 8 升到 9，scan pointers/metadata 和 search module metadata 与真实
  target 同源；旧 Linux-x64 scalar benchmark policy 和第二 resolver 已删除。ordinary 每个完整 buffer
  只有一次缓存函数指针调用，exact route 直接绑定 suffix。
- immutable `f29a34bc` 的 W1-WIN MSVC 19.50 focused gate 为 14/14；`search.h`/`scan.h` self-contained
  2/2。相同 source-fat binary 在 QEMU SVE VL 128/256/384/512/2048 与 RVV VLEN
  128/256/512/1024 各通过 15 项；exact binary 在相同矩阵各通过 14 项。模拟器只签 correctness。
- Windows selected-component source embedding、relocated shared SDK、relocated static SDK 均重新构建、
  发现并实际执行 count/argmax 与 in/out scan 示例；三种 journey 全部通过。

## 有界性能 verdict

本票签 `PASS / P1_ENGINEERING / 5% / NO MATERIAL LOSE`，不签“整个 SIMD/search/scan SOTA”。W1-WIN、
MSVC 19.50、固定 AVX2 下，shipping target body 与同 TU direct Highway/同合同 composition 对齐输入域、
ISA、alias、tail、dispatch 和资源计费。延长 count epoch 后四份日志无 instability marker；A/A、正反序、
双 layout 与 certification 全部通过。含 anchor 的 7 行为 5 `TIE`、2 `INCONCLUSIVE`、0 `LOSE`；产品行
只有 1M argmax 保持 `INCONCLUSIVE`。4K count 的 public/body informational center 为 0.955，诚实记录
tiny body 的一次 batch boundary；其他五行是 0.986--0.999。完整结果、hash、重放和未签范围见
[SIMD-09 performance evidence](../../../evidence/performance/simd-09-search-scan.md)。

## 终止 receipt

- product output commit：`f29a34bc6ad2e32106ab9851d486d19ce2ea1cd4`，tree
  `30fca958a8b6e70d49584de3532d0cad4e854bcd`；产品实现 commit 为 `606bb629`，后一提交只提高 count
  instrument epoch。相对 input commit 共 27 个文件，1,230 additions、1,037 deletions。
- 四项 LOC 由 immutable output 以 `cloc 2.08`、
  `python scripts/loc-report.py --module simd --commit f29a34bc --self-check --json` 复算且
  `self-check=pass`：全库代码 **312,867**；自实现包含测试 **121,415**；自实现不包含测试
  **79,587**；当前 SIMD 实现 **33,904** 行。
- known limits：SVE/RVV 未签真硅性能；tiny compiled façade 仍有一次不可消除的 batch call；predicate/
  custom-op、parallel/segmented/tensor/GPU scan 与 ordering 保持既定 owner；2 行 `INCONCLUSIVE` 不能改写
  为胜负。下一张有限票是 SIMD-10 bitpack/random。
- reopen：公开 search/scan 语义或复杂度出现可复现缺陷，Runtime metadata/pointer 再次分叉，热循环出现
  第二次 dispatch/heap/lock/整段 copy，三种交付失效，或同合同真实 consumer 在资格化真机上复现超过
  5% 的 material regression。
