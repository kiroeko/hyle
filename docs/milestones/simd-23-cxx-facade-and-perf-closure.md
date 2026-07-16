# simd-23：C++ 易用性外观与性能证据收口

> 状态：📋 计划（2026-07-16）。依赖 simd-22；不改变 close-01 的稳定 C ABI 承诺。

## 为什么有这一项

simd-22 已解决“同一 kernel 如何同时服务未知 CPU、整库固定 ISA 和用户 TU 固定 ISA”，但普通 C++
用户仍要认识各模块的 pointer/count 命名，且容易把 `vec.h` 的 381 个寄存器 op 误认为日常入口。
本里程碑补的是**发现性、类型安全和组合便利**，不是再造一层 kernel，也不是给 381 个 op 逐个加动态派发。

另一条独立问题是性能证据。`container/hash.h` 与 SwissTable 按依赖图有意不依赖 `simd`；这不妨碍
它们追求 SOTA，但必须用独立 benchmark 证明，不能为了复用 SIMD API 反转 `container -> simd` 依赖。

## 产品边界

```text
                       C++ source user
                              |
                 hyle/simd/algorithms.h
                 span overloads + clean names
                 header-only, no exported ABI
                              |
                 pointer / scalar / count API
                 stable semantic batch boundary
                    /                    \
       unknown CPU / exact library       fixed user TU
          Runtime / direct symbol          native:: kernel
                    \                    /
                     shared D-tag body
                            |
                         vec.h DSL

  close-01 stable ABI: include/hyle/c/simd.h -> pointer/POD/opaque handle only

  container/hash + SwissTable
          |
          +-- self-contained scalar/ISA implementation and platform probe
          +-- separate SOTA benchmark line; NEVER depends on simd
```

四层含义：

1. `algorithms.h` 是普通 C++ 用户入口，只有 inline overload/view；它不产生新 ABI 符号。
2. 现有 pointer/count batch API 是唯一 kernel 语义源，也是未来 C shim 的包装对象。
3. `native::`/`vec.h` 继续给明确目标 ISA 的专家用户；不强迫普通用户理解 D-tag。
4. hash/hashmap 属于 container。允许其内部使用自包含 ISA 路径，但不经 simd Runtime，也不从 simd
   facade 再导出。

## C++ 外观设计

新增统一入口 `include/hyle/simd/algorithms.h`，在 `hyle::simd` 中为已有语义 API 添加 `std::span`
overload 和无类型后缀的重载。模块头仍可单独 include；伞头只是可发现性入口。

目标调用形状：

```cpp
#include "hyle/simd/algorithms.h"

std::vector<float> x(n), y(n), out(n);
hyle::simd::add(x, y, out);              // contiguous_range convenience
const float d = hyle::simd::dot(x, y);
hyle::simd::exp(x, out);                 // overload -> exp_f32_batch
hyle::simd::inclusive_scan(out);
hyle::simd::sort(out);

hyle::simd::RandomState state{12345};
hyle::simd::random_fill(state, out);
```

第一版只接受 `std::span`；是否再接受任意 `contiguous_range` 由编译时间、诊断质量和隐式 const 转换
实测决定。不能让一个漂亮调用掩盖临时对象悬垂或非连续 range。

### 首批家族

| 家族 | 普通外观 | 转发目标 |
|---|---|---|
| 基础数组 | `add/dot/saxpy/scale/reduce_*` | ops pointer/count API |
| RNG | `random_fill(state, span)` | random pointer/count API |
| scan | `inclusive_scan/exclusive_scan` | scan pointer/count API |
| dtype | `convert(src, dst)` | 已支持的 dtype pair |
| math | `exp/log/tanh/... (src, dst)` | `*_f32_batch` / `*_f64_batch` |
| sort/select | `sort/stable_sort/partial_sort/select` | sort pointer/count API |
| bitpack | typed wrapper，显式所需输出大小 | bitpack pointer/POD API |

不纳入首批：

- `vec.h` 单寄存器 op：它们是 expert DSL，不是 batch facade。
- hash/hashmap：归 container，不能制造 `simd::hash` 的错误所有权。
- image/matvec/thread-pool：分别等待真实 domain/compute/sched 消费者，不照抄 Highway contrib 目录。
- 纯粹换名但没有更清楚语义的 overload。

### 合同与开销

- 两输入/输出长度不一致是调用者 bug，使用 release-surviving `HYLE_FATAL_ASSERT`；不返回 `Result`。
- 外部资源、分配或平台失败才走 `Status/Result`；本 facade 不引入这些失败点。
- wrapper 必须 inline、零分配，只做 `data()/size()` 提取和一次底层调用。
- 不缓存第二份 Runtime，不新增函数表 slot，不在 wrapper 中自行探测 CPUID。
- 代表性 wrapper 用优化后 IR/assembly 或 link map 证明被完全消除；合成 wrapper 不建 nanobench。

## 381 个 op 与 benchmark：先修口径，再补真缺口

`scripts/simd-op-coverage.py` 当前按名字/标题匹配，适合发现候选，不足以定义 benchmark 债务。例如
`TableLookupBytesInRange` 已是 `TableLookupBytes` 对拍行的 HYLE 被测侧，
`DemoteToNearestIntInRange` 已是 `run_third` 信息行，却仍被报为“未覆盖”。

先建立显式 manifest，每个公开名只能属于一类：

| 分类 | 是否进入 op-parity 分母 | 处理 |
|---|---:|---|
| `native` | 是 | 有独立硬件/算法或性能合同，补 Highway parity |
| `alias` | 否 | 精确转发；由目标 op 覆盖 |
| `synthetic` | 否 | 只组合现有公开 op；正确性测试 + wrapper 零开销检查 |
| `paired/info` | 已覆盖 | 成对行或 `run_third`，修报告而非重复 bench |
| `algorithm` | 否（L1） | 在真实工作集/吞吐 benchmark 中测 |
| `type/tag` | 否 | 编译/正确性测试 |

对现有“41 个待补 bench”候选的首轮审计结论：

- **应补独立 op-parity 的约 14 项**：`BitsFromMask`、`MaskFromVec`、`VecFromMask`、
  `LoadMaskBits`、`StoreMaskBits`、`ConvertInRangeTo`、`DemoteInRangeTo`、
  `TableLookupBytesOr0`、`InterleaveEvenBlocks`、`InterleaveOddBlocks`、`SlideUpBlocks`、
  `SlideDownBlocks`、`SumsOfShuffledQuadAbsDiff`、`Shuffle01`。
- **已有测量但报告误判的 2 项**：`TableLookupBytesInRange`、
  `DemoteToNearestIntInRange`。
- **`Stream` 单列算法 benchmark**：工作集必须大于 LLC、地址满足非临时写合同，并报告带宽；不能用
  L1 单寄存器 nanobench 得出结论。
- 其余候选初判为 alias/synthetic，例如 `SignBit`、`ShiftRightUniform`、mask half、
  `MaskedSet*`、`PromoteInRange*`、`OrderedTruncate2To`、`TruncateStore`、`CompressBits*`、
  `SafeCopyN/SafeFillN`。它们不因“能写一行 bench”而进入分母。

最终数字以 manifest 逐项代码审计为准；验收不能继续宣称“补 41 个 benchmark”。

## hash/hashmap SOTA 证据线（不接入 simd）

### 已有证据

- `Hash<T>` 的 16B inline、`mix64` 和 `hash_bytes` 已分别对照 raw XXH3/wyhash；当前结果说明核心
  hasher 没有明显空洞。
- SwissTable 已对照 absl flat_hash_map，在当前 u64/100k 工作负载的 hit/insert/erase/iterate 上有优势。
- roadmap 承诺的 F14 对照尚不存在，所以“hashmap SOTA 已闭合”目前不能成立。

### 必须分开的两组 benchmark

1. **哈希函数本体**：同 linkage、同输入池，对照 raw XXH3 与 wyhash；长度至少覆盖
   `0/1/4/8/16/17/32/64/128/129/256/1024`，命中 HYLE 的策略切换边界。报告 ns/hash 和 GB/s，
   避免只反复哈希同一常量让前端或 cache 美化结果。
2. **哈希表本体**：HYLE / absl / F14 尽量使用同一个 hasher 和 allocator，再单独跑各自默认配置。
   覆盖 u64、16B POD、短/长 string；小表/L1 与 100k 级；hit/miss、insert、erase、iterate、
   reserve/rehash、混合 churn、不同 load factor，并记录 bytes/entry。

这样能回答两个不同问题：“HYLE 的 hash 快不快”与“控制字节探测/表布局快不快”。若默认 hasher 场景
赢、统一 hasher 场景输，就不能把结果归功于 hashmap；反之亦然。

## 实施顺序

### A. 覆盖账本先行

- 将 381-op 提取结果与显式 manifest 合并，脚本校验新增 op 必须分类。
- 修正 orderswap 报告，把 paired/info 与真正 missing 分开。
- 冻结“synthetic 不进 benchmark 分母”的 CI 规则。

### B. 最小外观骨架

- 新增 `algorithms.h`、编译测试与文档索引。
- 先落 ops/RNG/scan 三家族，确定 span、const、aliasing 和长度合同。
- 用汇编/IR 抽查 debug 合同存在、release 成功路径无额外循环/分配/派发。

### C. 扩到现有语义面

- dtype/math，再到 sort/select/bitpack；每家族只转发已存在的 batch kernel。
- 同步普通用户、整库 exact、`native::` 三条示例，避免把 fixed target 描述成“免动态税”。
- 生成或维护普通 API 索引，清楚地区分 107 个语义名与 381 个 expert op。

### D. SIMD 性能缺口

- 按 manifest 补约 14 个原生 op parity；遵守现有 HYLE-first/Highway-first 双序和校准闸门。
- 独立补 `Stream` 大工作集带宽 benchmark。
- Win x86、WSL x86、macOS NEON 跑性能；AVX512/SVE/RVV 没有可信真机时只作正确性，不用模拟器
  数字宣称性能。

### E. container 性能收口

- 扩 hash 边界长度 benchmark。
- 接入 F14 gated benchmark，补统一/default hasher 两套 hashmap 矩阵。
- 结果记入 performance ledger；只在跨平台、多轮统计达到闸门后写 SOTA 结论。

### F. ABI 与发布检查

- C++ facade 不出现在 ABI symbol allowlist，公共模板/STL 不穿过 DLL seam。
- close-01 实现时只包装 pointer/POD batch core，不包装 span overload、`Runtime` 或 D-tag。
- C shim 与 C++ facade 共用同一批 scalar oracle 和 per-ISA kernel，不形成第三套实现。

## 验收标准

- 普通用户只 include `hyle/simd/algorithms.h`，无需宏、CPUID 或目标 ISA 知识即可调用所有首批家族。
- exact-library 和 `native::` 用法保持可用；facade 不增加第二次派发或每元素分支。
- 所有 facade 长度/可写性/类型合同有编译或运行测试，代表性 release wrapper 证实零额外开销。
- coverage 报告不再把 alias/synthetic/paired 项计为 missing benchmark。
- 真正 native 缺口完成三平台可用项的校准 parity；`Stream` 以带宽口径测量。
- hash 与 hashmap 独立出报告；hashmap 至少有 absl + F14、统一 hasher + 默认 hasher 两套证据。
- `container` 依赖图中仍无 `simd`，`simd` facade 中也无 hash/hashmap 再导出。

