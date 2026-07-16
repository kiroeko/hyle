# simd-23：SIMD 性能证据收口与发布入口裁决

> 状态：✅ SIMD 41-name 性能闭环 + 实用 op 审计完成；❌ C++ 容器 facade 于 2026-07-17
> 按 close-01 产品铁律撤回。依赖 simd-22；发布给用户的唯一稳定入口是纯 C ABI。

## 为什么有这一项

simd-22 已解决“同一 kernel 如何同时服务未知 CPU、整库固定 ISA 和用户 TU 固定 ISA”。本里程碑先
完成性能证据闭环；期间曾增加 `algorithms.h` 作为 C++ 容器便利面，但复核 close-01 后确认这会制造
第二套“普通用户入口”：即使 STL 没有穿过 DLL seam，核心库仍替一种消费语言拥有了容器适配层，
与“发布只提供 C ABI，消费方看着办”的约定冲突，因此撤回。

另一条独立问题是性能证据。`container/hash.h` 与 SwissTable 按依赖图有意不依赖 `simd`；这不妨碍
它们追求 SOTA，但必须用独立 benchmark 证明，不能为了复用 SIMD API 反转 `container -> simd` 依赖。

## 产品边界

```text
                 close-01 public product boundary
                    include/hyle/c/simd.h
                 pointer / POD / opaque handle
                              |
                 internal C++ pointer/count core
                    /                    \
       unknown CPU / exact library       fixed user TU
          Runtime / direct symbol          native:: kernel
                    \                    /
                     shared D-tag body
                            |
                         vec.h DSL

  container/hash + SwissTable
          |
          +-- self-contained scalar/ISA implementation and platform probe
          +-- separate SOTA benchmark line; NEVER depends on simd
```

四层含义：

1. `include/hyle/c/simd.h` 是未来唯一发布入口；纯 C、语言中立、版本化。
2. 现有 C++ pointer/count batch API 是 kernel 语义源和 C shim 的内部转发目标，不是稳定发布 ABI。
3. `native::`/`vec.h` 是固定目标源码级 expert surface，不承诺跨工具链 ABI。
4. hash/hashmap 属于 container；不能为了复用 SIMD 反转依赖。

## 发布入口裁决（2026-07-17）

- 删除 `include/hyle/simd/algorithms.h` 及其 `std::span/std::vector` 测试。
- HYLE 核心不维护 C++、C#、Java、Rust 等任一语言的容器 facade；消费方用本语言容器取得
  pointer/count 后调用同一 C ABI。
- close-01 必须把 SIMD whole-buffer 语义面纳入 `include/hyle/c/simd.h`；无状态操作用 pointer/POD，
  RNG 等状态对象用 create/destroy 配对的 opaque handle。
- 现有 `include/hyle/simd/*.h` 在 close-01 前只服务仓库内部和同工具链开发，不得在发布文档中称为
  普通用户入口。

## 387 个公开/selector 名与 benchmark：显式分类已经闭环

`scripts/simd-op-coverage.py` 当前按名字/标题匹配，适合发现候选，不足以定义 benchmark 债务。例如
`TableLookupBytesInRange` 已是 `TableLookupBytes` 对拍行的 HYLE 被测侧，
`DemoteToNearestIntInRange` 已是 `run_third` 信息行，却仍被报为“未覆盖”。

当前机械树是：387 个公开名 − 26 个非性能项 − 13 个 `*Native/*Emulated` selector − 4 个 alias
= 344 个 benchable；其中 314 个有独立 source benchmark，30 个由 manifest 明确归为
alias/synthetic/paired-info/algorithm。旧 41-name 候选已经全量核账；后来新增的 4 个 128-bit upper
合成接口也立即进入 manifest，因此 truly unclassified 始终为 0。

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

- **应补独立 op-parity 的 15 项**：`BitsFromMask`、`MaskFromVec`、`VecFromMask`、
  `LoadMaskBits`、`StoreMaskBits`、`ConvertInRangeTo`、`DemoteInRangeTo`、
  `TableLookupBytesOr0`、`InterleaveEvenBlocks`、`InterleaveOddBlocks`、`SlideUpBlocks`、
  `SlideDownBlocks`、`ShiftRightUniform`、`SumsOfShuffledQuadAbsDiff`、`Shuffle01`。
- **已有测量但报告误判的 2 项**：`TableLookupBytesInRange`、
  `DemoteToNearestIntInRange`。
- **`Stream` 单列算法 benchmark**：工作集必须大于 LLC、地址满足非临时写合同，并报告带宽；不能用
  L1 单寄存器 nanobench 得出结论。
- 其余候选初判为 alias/synthetic，例如 `SignBit`、`ShiftRightUniform`、mask half、
  `MaskedSet*`、`PromoteInRange*`、`OrderedTruncate2To`、`TruncateStore`、`CompressBits*`、
  `SafeCopyN/SafeFillN`。它们不因“能写一行 bench”而进入分母。

最终数字以 manifest 逐项代码审计为准。验收必须**闭环全部 41 个名字**，但不能宣称“补 41 个
nanobench”：只有 native 项补独立 parity；alias/synthetic/paired/algorithm 各交自己的正确证据。

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

## HYLE 相对 Highway 的量化结论

不能用一个总倍数回答“HYLE 强多少”。simd-23 维护分层 scorecard，所有格子标明平台、工作集、复杂度
与统计置信度：

| 层 | 比什么 | 合格结论 |
|---|---|---|
| 寄存器 op | orderswap 全量同 ISA 同 TU | WIN/TIE/LOSE 占比 + 稳定异常；目标是零明确 LOSE，不把 TIE 宣传成优势 |
| whole-buffer kernel | dot/add/scan/RNG/math/dtype/sort/bitpack | 同语义、同复杂度的吞吐/延迟曲线，报告小 N 调用成本与大 N 带宽/计算墙 |
| 算法复杂度 | sort/select/top-k/bitpack 等 | 输入规模阶梯验证 O(n)、O(n log n)、O(n log k) 等，不允许只在一个 N 上报倍数 |
| 未知 CPU 产品面 | HYLE ordinary batch vs Highway dynamic 用法 | 量化用户代码量、首次选择和每 batch 调用成本；功能优势与速度优势分栏 |
| 固定 ISA 产品面 | HYLE exact/native vs Highway static target | 代表 kernel 汇编/吞吐，证明两者都能零运行时判断，不把它叫“省动态税” |

最终回答应诚实区分：底层原语若大多 TIE，说明 HYLE 抽象没有性能损失；HYLE 是否值得选，主要取决于
预编译 batch/双轨架构、自有算法核、跨模块依赖与 C ABI，而不是宣称每条 intrinsic 都比 Highway 快。
任何“快 X%”只来自 scorecard 中同合同的实测格子，不把 API 易用性换算成虚假的性能倍数。

## 2026-07-16 闭环证据

- `scripts/simd-op-manifest.json` 对旧 41 项给出唯一分类：15 native、6 alias、15 synthetic、
  2 paired/info、3 algorithm；新增 4 个 128-bit upper 接口同样归 synthetic。coverage CI 校验 native
  必须有独立 source row、alias/synthetic 禁止误入独立分母、algorithm 证据文件必须存在。
- 15 个 native 新行在 Windows MSVC AVX2、WSL GCC AVX2、macOS AppleClang NEON 各自可用
  target 的 focused 双序多轮报告均为 **0 LOSE**。最大的真回归 `SumsOfShuffledQuadAbsDiff` 来自 HYLE 的
  store/scalar/reload oracle：Windows 约 0.08x、Mac 约 0.11x；改成寄存器 recipe 后，WSL 的恒等/
  非恒等排列认证为约 1.49x/1.43x WIN，Mac 两者中心值约 0.99、认证为 TIE。SSE4 同步移除 scalar
  oracle，AVX-512 保持原生 DB-SAD。
- `Stream` 用 256 MiB 工作集、每进程 6 次交错中位数测量。三次外层复跑的 HYLE/Highway 吞吐比：
  Windows `1.001/1.021/1.005`，WSL `0.988/1.047/1.058`；Apple Silicon
  `3.787/3.316/3.468`。ARM 大幅差异可由源码解释：Highway AppleClang 路径在每个 16B store 前
  发 write-prefetch，而 HYLE/普通 Store 直接写；这是 whole-buffer 合同证据，不进入 L1 op 分母。
- 实用缺口复核纠正了旧文档：基础 `Masked*` 家族早已完整；Highway 的
  `WidenLow/HighMulAdd` 是内部 helper，HYLE 已公开更完整的 accumulate 消费者语义。本轮只新增
  `Eq128Upper/Ne128Upper/Min128Upper/Max128Upper` 四个有用户价值的 128-bit key 对称接口。
- MSVC AVX-512 编译闸同时发现上一轮 AVX2 quarter-width 测试误套到 512-bit target，以及分层 include
  后 capability 宏缺 `Vec512` ScatterIndexN overload；测试现只覆盖适用边界，AVX-512 增加原生 masked
  scatter，`hyle_simd_avx512_test` 已恢复编译通过。
- 对 ops、RNG、scan、dtype、real math、sort/select/radix/argsort、bitpack 完成普通用户语义目录；
  目录转入 `docs/guide/simd-usage.md`。短暂实现过的 `algorithms.h` 容器原型已于 2026-07-17 撤回，
  避免在 close-01 C ABI 之外形成第二套发布入口。

## 实施顺序（性能四步先行，facade 最后）

> **2026-07-17 状态**：性能步骤 1–4 已完成，步骤 5–6 已完成产品裁决和语义目录、C ABI 实现等待
> close-01。九份 v4 report 的 union 为 303/303；320-case 表中
> Windows/WSL/Mac 明确 LOSE 均为 0，旧 ScatterIndexN 与本轮发现的 SatWiden/CompressNot/Mod/
> SetOnlyFirst/IsFinite 回归均已修复，所有 `<0.98` 行已分类并写入
> `simd-orderswap-matrix.md`。旧 41-name 候选已显式分类，15 个 native 新行在各平台可用 target 的第二轮狩猎为
> 0 个明确 LOSE；普通用户语义已写入 `docs/guide/simd-usage.md`，不再发布 STL 容器 facade。
> container hash/hashmap 的独立 SOTA 证据线仍是后续任务，不属于本次 SIMD 收口。

### 1. 重测 303 个已有 benchmark 覆盖项并修仪器

simd-22 改动了头文件组织、target 选择和共享 native body。旧的 2026-07-15 orderswap 数据仍是历史
证据，但不能自动证明重构后的二进制仍保持原结论。新增 op benchmark 和 facade 零开销结论之前，先：

- 按 `simd-op-benchmark-methodology.md` 从当前 HEAD 新鲜构建，不复用重构前 exe；检查目标时间戳与
  AVX2/NEON target define/flags。
- Windows 与 WSL 错开运行；三个 x86 TU（parity/full/widen）均双序至少 2 轮。macOS 三个 NEON TU
  串行、双序各 2–3 轮，使用 8x epoch 与 `--gate 0.06`。
- 每个报告必须先过 anchors；任何 calibration fail 使该份整表作废，不拿后续单元格解释性能。
- “303”来自 coverage tree，不是 GTest case 数；正确性测试仍按各平台实际 discover 数报告。
- 先审计 benchmark 对称性、pointer-hoist、每 TU 自有 anchor、双 layout-bank 和报告 CI，再看 op 数字。
- Windows 使用固定 `VSLANG` 的全新性能 build tree，验证头依赖、obj/exe 时间戳和真实重编译，封死
  Ninja `msvc_deps_prefix` 语言错配造成的 stale-exe 假数据。
- 形成“重构后仪器证书”：HEAD、编译器/flags、机器、输入原始文件、anchor 区间、各判决计数和完整
  report 命令。没有这份证书，后续性能判决不得进入权威矩阵。

### 2. 用新旧全表与 HYLE 独有 benchmark 狩猎重构回归

- 对比重构前后 303×Windows/WSL/macOS orderswap 全表，另比 HYLE 独有测试；按 TU、op、平台列出
  新增 LOSE、WIN 消失和置信区间显著下移。
- 优先查 interface shape、Highway/xsimd 内化、头/target 机械搬运和宽度表示变化；这类回归恢复前，
  不把下降归类成“HYLE 算法本来就输”。
- 修共享原语后重跑受影响平台全表和复合消费者金丝雀，不以单 op 探针收口。

### 3. 排除回归后狩猎稳定真输 Highway 的 op

- 先复验历史 NEON `ScatterIndexN i32`，并把所有 0.90–0.96 项按 `LOSE / INCONCLUSIVE / TIE`
  分桶；0.93 裸数字不是结论。
- 稳定 LOSE 执行规范化逐指令 diff、端口/依赖链和多形态实验；完整步骤以主方法论 §4 为准。

### 4. 闭环 41 个名字并做第二轮 Highway 落后项狩猎（✅）

- 将 387-name 提取结果与显式 manifest 合并，脚本校验新增 op 必须分类。
- 修正 orderswap 报告，把 paired/info 与真正 missing 分开。
- 冻结“synthetic 不进 benchmark 分母”的 CI 规则。
- 对全部 41 个名字逐项交付证据：native parity、paired/info 账本、alias/synthetic 正确性与零开销、
  algorithm 真实工作集 benchmark；然后对新产生的稳定 LOSE 再走一轮 Highway 落后项狩猎。

### 5. 发布入口裁决（✅）

- 撤回 `algorithms.h` 原型及测试，不在核心仓库拥有某一种消费语言的容器包装。
- 发布入口统一归 close-01 的纯 C ABI；无状态能力用 pointer/count/POD，状态对象用 opaque handle。
- C++、C#、Rust 等调用者从自己的容器取得地址和长度，自行维护薄绑定。

### 6. 普通用户语义目录（✅；C ABI 实现等待 close-01）

- 目录覆盖 ops/RNG/scan、dtype/math、sort/select/bitpack，每家族只对应已存在的 batch kernel。
- 同步普通用户、整库 exact、`native::` 三条示例，避免把 fixed target 描述成“免动态税”。
- `docs/guide/simd-usage.md` 清楚地区分普通语义面与 387 个 expert/selector 名，并预留 close-01
  名称和生命周期迁移清单。

### 7. container 性能收口

- 扩 hash 边界长度 benchmark。
- 接入 F14 gated benchmark，补统一/default hasher 两套 hashmap 矩阵。
- 结果记入 performance ledger；只在跨平台、多轮统计达到闸门后写 SOTA 结论。

### 8. ABI 与发布检查

- 核心不发布 STL/container facade，公共模板/STL 不穿过 DLL seam。
- close-01 实现时只包装 pointer/POD batch core，不包装 `Runtime`、D-tag 或 target-dependent vector。
- C shim 与内部 C++ core 共用同一批 scalar oracle 和 per-ISA kernel，不形成第三套实现。

## 验收标准

- `docs/guide/simd-usage.md` 覆盖普通用户常用家族、调用合同和三种目标选择；不要求理解寄存器 op。
- close-01 验收时，普通用户只 include `hyle/c/simd.h`，无需宏、CPUID 或目标 ISA 知识即可调用
  首批家族；当前里程碑不冒充已经交付该 ABI。
- exact-library 和 `native::` 用法保持可用；C shim 不增加第二次派发或每元素分支。
- coverage 报告不再把 alias/synthetic/paired 项计为 missing benchmark。
- 当前 HEAD 的 Windows/WSL/macOS 全量 orderswap 已按方法论重新出数，所有报告 calibration pass；
  旧 `ScatterIndexN` 与全部 0.90–0.96 项均有稳定性分类，稳定异常有逐指令证据。
- 真正 native 缺口完成三平台可用项的校准 parity；`Stream` 以带宽口径测量。
- hash 与 hashmap 独立出报告；hashmap 至少有 absl + F14、统一 hasher + 默认 hasher 两套证据。
- `container` 依赖图中仍无 `simd`，`simd` facade 中也无 hash/hashmap 再导出。
