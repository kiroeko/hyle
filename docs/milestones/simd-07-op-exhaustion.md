# simd-07：op 面穷尽（11 批补齐）+ 全 lane 类型 + 首个真实消费者

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/（+ container/ 消费者） |
| **前置** | simd-02（SSE4/AVX2 后端）、simd-03（NEON/AVX512 后端）、simd-05（int lane 广度） |
| **状态** | ✅ 完成（2026-07-06，commit a118ce8~9d3b6d3）。三平台真机对拍：Win MSVC / WSL GCC14.2 vec_test 168、每 ISA 变体 138；mac AppleClang neon 变体 138 + 全量 ctest 24/24；AVX512 无机仅编译验 |

> **范围**：把 vec.h 的 op 集从 ~35 补到 ~90+（对齐 Highway 的能力面），全 lane 类型；并让 SIMD 首次有真实内部消费者（container count/rfind）。**用户纠偏驱动**：变宽 Promote/Demote 与 swizzle 曾被以"需 partial 向量 / 无消费者"后置，用户指出"高性能 infra 库消费面最广，能力要提供到位，不能因暂无消费者后置"——推翻误判后补齐。
> **cloc**：五后端头 5,840 行（scalar 867 / sse4 1194 / avx2 1279 / avx512 1185 / neon 1315，均在 ~2k 设计上限内）+ str_search 消费者 109 行 + vec_test 增至 168 测试。

## 为什么是它（能力面而非"够用就停"）

simd-01~06 交付的是 **f32/f64/i32/u32 + ~35 个 op** 的核——够跑 dot/saxpy/transcendental，但远不是一个 SIMD 库该有的能力面。Highway 有 ~200 op、12 lane 类型；HYLE 停在子集上，量化（int8/int16 GEMM）、稀疏（gather/compress）、通用向量化循环（掩码尾部、变宽转换）都表达不了。

**关键认知（用户 2026-07-06 纠偏）**：infra 库的消费者是"整个世界"，无法预料谁会用哪个 op。所以能力要**提供到位**，不能拿"暂无消费者"当后置理由——那是应用层逻辑，不是基础库逻辑。此前我把变宽 Promote/Demote、swizzle、gather 以"需 partial 向量 / 等消费者驱动"后置，是把库当应用做了。纠偏后一次补全。

## 核心决策：抄 Highway 契约、per-op intrinsic、分批 + 三平台真机对拍

- **抄契约不 vendor 库**：每个 op 照 Highway 的 per-op intrinsic 序列写进四后端（sse4/avx2/avx512/neon）+ scalar，而非引入 Highway 依赖。vec.h 的价值正是"一份源码，多后端"。
- **分 11 批**（每批一提交，独立三平台验）：移位 → 饱和/平均 → 变宽转换 → 浮点舍入 → 掩码尾部 → reduce 全家族 → 变宽 Promote/Demote → 变量移位/加宽乘 → swizzle 全套 → gather/scatter/compress/expand → 位计数/符号/Clamp/Rotate。
- **scalar 后端是 oracle**：每个 op 的 scalar 实现是三个 SIMD 后端的对拍基准（vec_test 逐 lane 比对）。
- **真机纪律**：x86 三档在 WSL GCC14 跑（sse4/avx2 真跑、avx512 编译验），NEON 在 Apple Silicon（`ssh kiro@172.22.10.198`）真跑——tests/CMakeLists 按 `CMAKE_SYSTEM_PROCESSOR` 建对应 ISA 变体。

## 补齐的 11 批（op 分类）

| 批 | op | 要点 |
|---|---|---|
| 移位 | ShiftLeft/Right（编译期常量） | 见 simd-05 int lane |
| 饱和/平均 | SaturatedAdd/Sub、AverageRound | 量化配套 |
| 变宽转换 | ConvertTo/RoundToInt（f32↔i32） | (de)quant 核 |
| 浮点舍入 | Round/Floor/Ceil/Trunc | |
| 掩码尾部 | FirstN/MaskedLoad/BlendedStore/LoadN/StoreN | 向量化循环余数（std::simd 表达不了的 90%）|
| reduce 全家族 | ReduceSum/Min/Max/**Mul/And/Or** | 补齐横向归约 |
| **变宽 Promote/Demote** | PromoteLowerTo/UpperTo、Demote2To | **量化核**；half-at-a-time 形式 |
| 加宽乘 | MulHigh、**WidenMulPairwiseAdd**（i16→i32 madd） | int8/16 GEMM 点积核 |
| **swizzle 全套** | OddEven/Interleave/Concat/Reverse/DupOdd-Even/TableLookupLanes | |
| **gather/scatter + compress/expand** | GatherIndex/ScatterIndex、Compress/Expand | 稀疏/embedding 核 |
| 位计数/符号 | PopCount/LZCnt/TZCnt、CopySign/BroadcastSignBit/Clamp/Rotate/Not | |

## 纠"后置"误判（本里程碑的核心教训）

- **变宽 Promote/Demote**：曾判"定宽后端做不了"——错。Highway 的 **half-at-a-time 形式**（`PromoteLowerTo`/`PromoteUpperTo` 各处理输入的半段、`ReorderDemote2To` 两输入拼一输出）完美契合定宽后端，根本不需要 partial 向量。误判源于只看了 Highway 的可变长接口没看定宽落地形。
- **swizzle / gather**：曾判"等消费者驱动"——错。这是把库当应用。补齐后 count/rfind 立刻就用上了 SumsOf8/VecFromMask/FindLastTrue（见下）。

## 首个真实消费者：container count/rfind 统一到 vec.h

SIMD 一直没有内部消费者（只喂未开工的 compute）。本轮把 **container `string_ref` 的 count/rfind 从手写 x86-only intrinsic（旧 `string_ref_avx2.cc` + inline SSE2）统一到 vec.h 变体-TU 派发**（`str_search_{scalar,sse4,avx2,avx512,neon}.cc`，与 simd/ops 同模型）。

- **补三原语保住 SAD 计数技巧**：count 用 psadbw 式"比较→掩码→SAD 求和"避免每字节分支，需要 `SumsOf8`（u8→u64 SAD）+ `VecFromMask`（掩码转 0xFF）+ `FindLastTrue`（rfind 反扫），本轮一并补进四后端。
- **x86 打平（证 vec.h 零成本）**：apples-to-apples（两边都过 .so）新 vec.h 核 vs 旧手写 AVX2 = count 96-101% / rfind 99-100%，统计平局。
- **arm 白得（真收益）**：NEON count **3.7x** / rfind **8.0x** vs 重构前 arm 落的 scalar 字节循环（重构前 count/rfind 是 `#if HYLE_ARCH_X64` 门控，arm 无向量加速）。

## 对抗式自审（六类）

- **① 正确性**：vec_test 每 op 逐 lane 对拍 scalar oracle，全 lane 类型；三平台 168/138/138 全过。
- **② 性能**：count/rfind 回归 apples-to-apples 打平手写核；**bench 陷阱**——早先测出"新版慢 3-11%"是假象，`__attribute__((noinline))` 对同一份核复现同 gap 坐实是"手写核可内联进 bench TU、.so 调用不可内联"的测量不对称（教训独立成 memory）。
- **③ 规范**：抄契约不 vendor；各后端 ~2k 行上限内；MSVC C4702（Shl/Shr else-return 不可达）修为 else 赋值。
- **④ 三平台真机**：NEON 首次经正式 gtest 变体真跑（此前只 objdump），全量 mac ctest 24/24；顺带揪出 dns_test 的 `::ntohs`/`::ntohl` macOS 宏 bug（全量 mac 构建才暴露）。
- **⑤ 加速审计（用户追问"还有哪些能 SIMD"）**：全树扫无"该用 simd 却手写"的不当例——simd 外唯一手写真向量 op 是 probe.h（SwissTable 分组匹配），手写是**内联刚需**（match 内联在最内层 probe 循环，必须编译期 baseline SSE2/NEON 零派发；vec.h 宽 ISA 靠运行期 best_isa 间接调用，每 probe 付一次致命），非 DAG 约束（已订正其注释）。现有代码 SIMD 空间挖尽：count/rfind 是唯一真收获，iequal（HTTP 头名 4-20B）实测 SSE2 反慢 28%。
- **⑥ 残留**：**仅缺 native-intrinsic 性能优化档**（gather/compress/popcount 现走 store+reload 或 emulation，非 vpgather/vpcompress/vpopcnt 硬件指令）——那是 profiling 驱动的加速档、非能力缺口，留 compute 消费时按热路径证据加。AVX512 全程无真机仅编译验。

## 交付物

- `include/hyle/simd/backend/{scalar,sse4,avx2,avx512,neon}.h`（~90+ op，全 lane 类型）
- `include/hyle/simd/vec.h`（新 op 的 D-tag 接口声明）
- `tests/simd/vec_test.cc`（168 测试，每 op 对拍 scalar）；`tests/CMakeLists.txt`（按 arch 建 sse4/avx2 或 neon 变体）
- `src/container/str_search_impl.h` + `str_search_{scalar,sse4,avx2,avx512,neon}.cc` + `str_search.h`（count/rfind 消费者，变体-TU 派发）
- `bench/container/string_ref_bench.cc`（count/rfind 4/64/1024KB 回归门 + 防内联/防提升）

## 参考

- simd-design.md（op 覆盖节 + D-tag 契约）。Google Highway（op 契约来源，per-op intrinsic 序列）。simd-05（int lane 广度，本轮的直接前身）。performance.md container·str 节（count/rfind 三平台实测）。memory：project_simd_int_lanes（op 全景 + 陷阱登记）、feedback_benchmark_inlining_asymmetry（bench 内联不对称教训）、reference_cuda_env（无关，compute）。
