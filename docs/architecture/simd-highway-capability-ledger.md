# SIMD × Highway 能力账本

> 本账本比较用户能解决的问题和实现形状，不以函数名数量或 LOC 判输赢。参考快照是本地
> `C:\public\project-hyle\highway` 的 `83eff39371e5e6494b305b253f8a3c06f831faf4`；HYLE
> 不链接 Highway，后者只进入测试和 benchmark。

## 为什么 Highway 更大

2026-07-26 的 raw physical-line 审计（含空行/注释，**不是**项目四项 `cloc` 口径）得到：

| 范围 | 行数 | 如何理解 |
|---|---:|---|
| HYLE `include/hyle/simd` + `src/simd`（本轮前） | 46,837 | 公共算法、七后端、dispatch 和实现 |
| 其中 HYLE backend 文件 | 22,396 | scalar/SSE4/AVX2/AVX-512/NEON/SVE/RVV + crypto |
| Highway 非测试产品头/源 | 147,504 | 还包含更多目标、编译器兼容层和 contrib 产品 |
| Highway 与 HYLE 支持平台近似相交的 ops | 72,515 | 排除 PPC/LoongArch/WASM/emu128 后的后端层 |
| Highway runtime/support | 12,730 | target 宏、动态派发、编译器/平台矩阵 |
| Highway `contrib` 产品 | 29,288 | algo/bitpack/hash/image/math/random/sort/thread pool 等 |
| Highway 本账本明确不追的平台后端 | 29,655 | PPC/LoongArch/WASM/emu128；不含 SVE/RVV |

这些分组用于解释量级，存在公共文件归类差异，不能简单相加。主要差距来自 Highway 更宽的
CPU/编译器平台矩阵、SSE2/SSSE3/SSE4 等细粒度 target、宏展开的类型×宽度实例，以及把
`float16_t`/`bfloat16_t` 当一等寄存器 lane。它既不能证明 HYLE 少，也不能证明 HYLE 已完整；
下面按能力逐项判断。

## 用户能力对照

| 能力轴 | Highway | HYLE 当前 | 判断 |
|---|---|---|---|
| i/u8/16/32/64、f32/f64 lane | 一等 D-tag/vector | 一等 D-tag/vector | 主力齐平 |
| f16/bf16 lane | 一等 special-float vector，按 target gate | 仅 storage carrier、batch convert、mixed dot；通用 D-tag 后端未成立 | **P0 真缺口** |
| fp8 | 无统一同形产品面 | e4m3/e5m2 carrier + batch convert | HYLE 产品面更宽，但当前转换多为标量级收益 |
| fixed/scalable vector | x86/NEON/SVE/RVV，另有更多平台 | x86/NEON/SVE/RVV；SVE/RVV 为真实 scalable tag | 主部署面齐；变宽真机性能欠证 |
| SVE2/RVV 扩展 | target/feature 分层广 | SVE2、SVE AES/i8mm/bf16 与 RVV Zvbb/Zvbc/Zvkned/Zvfh 已探测，部分有快路 | 不应把扩展探测冒充完整产品 tier；随真实 kernel 补 |
| load/store/tail/masked memory | 完整 | 完整，含 `LoadN/StoreN` 精确尾部 | 齐平 |
| gather/scatter/interleave | 完整 | 完整；部分 ISA 有硬件墙或更强精确写合同 | 齐平，按合同解释差异 |
| lane/mask/shuffle/compress/expand | 完整 | 七后端主力面完整，机械 coverage + 对拍 | 主力齐平；不追名字数量 |
| conversion/widen/narrow/saturating | 含 special float | base lane 完整，dtype 另有 batch API | base 齐平；f16/bf16 回到 P0 |
| register reduction | 完整 | 完整 | 齐平 |
| whole-range 通用算法 | Generate/Transform/Replace/FindIf/CountIf/CopyIf 等 | **本轮新增** native target-bound 同形层 | 已补最常见 kernel 拼装缺口 |
| scan/search | contrib 基础算法 | 预编译 Runtime + native kernel，类型面较宽 | HYLE 产品面完整 |
| sort | VQSort/VQSelect，含部分 f16 | sort/partial/select/radix/argsort，key 产品更宽 | HYLE 广度强；f16 total-order 是缺口 |
| math | vector math + contrib foundation | 批量实数/复数/激活/softmax，择优自有/xsimd/Highway 算法思想 | HYLE 产品面更宽；逐函数证据见性能文档 |
| random | vector RNG | 标量状态 + batch fill、跨 ISA 可复现 | 同类能力已覆盖 |
| bitpack | contrib bit pack | u8/u16/u32 batch，固定 logical block 合同 | 同类能力已覆盖 |
| crypto primitive | AES/CLMul 等 register ops | AES/CLMul register ops，按 feature gate | 原语级覆盖；不承诺完整密码产品 |
| runtime dispatch / multitarget | foreach-target + dynamic/static target | Runtime / exact / native + fat-kernel | 两者都覆盖；HYLE 的交付编译器矩阵较窄 |
| C ABI | Highway 不是其主要产品形状 | 247 个 pointer/count/POD 入口（244 动态符号） | HYLE 优势，但错误 seam 尚未统一 |

### RVV 是既有后端，不是缺口

- 寄存器实现位于 `include/hyle/simd/backend/rvv.h`；`ScalableTag` 通过运行期 `vsetvlmax`
  使用 LMUL=1 物理宽度，不把 RVV 伪装成固定 128-bit。
- `src/simd/CMakeLists.txt` 为 ops/random/scan/search 生成 RVV variant TU，并接入
  `cvt_scalable.cc`、`transcendental_highway_scalable.cc`、`bitpack_hyle_scalable.cc` 和
  `vqsort_hyle_rvv.cc` 产品核。
- `platform::cpu_features()` 经 Linux `riscv_hwprobe` 探测 V、Zvbb、Zvbc、Zvkned、Zvfh/Zvfhmin；
  `src/simd/runtime.cc` 的 `Isa::kRvv` 表选择并发布 RVV dtype/math/sort/bitpack tier。
- `tests/simd/rvv_test.cc` 与 product tests 在同一 binary 下通过 VLEN 128/256/512/1024-bit
  四档各 99 项；新增 range helper 的 full vector、partial tail、stable in-place compaction 也逐档执行。

因此剩余 RVV 项是**真实硬件性能证书和可选扩展的产品化收益证明**，不是“还没支持 RVV”。

## 本轮补上的真实使用形状

`hyle/simd/native/algorithm.h` 为写自定义图像、信号、量化、几何和 compute kernel 的用户提供：

- `generate`；
- unary/binary/ternary `transform`；
- `replace` / `replace_if`；
- `find_if` / `count_if`；
- stable `copy_if`，含 exact in-place compaction。

这些 helper 自己拥有 full-vector 循环、非整倍数尾部和 mask；lambda 接收目标特定 vector，编译器在
固定 ISA TU 内内联。未知 CPU 的发布包把它放进 `hyle_add_fat_kernel` 的 impl header，由现有机制
为完整 kernel 多版本化。它不是逐元素 callback，因此没有把 benchmark 形状做进公共 ABI。

## 剩余优先级

### P0：严格收口前必须解决

1. **f16/bf16 一等 lane 设计。** 一次设计覆盖 scalar、四个 fixed backend 与 SVE/RVV，明确
   arithmetic、compare、load/store、promote/demote 的可用子集；不能只上一个后端或只补名字。
2. **窄浮点算法闭环。** 至少补真实消费者需要的 sort/search/reduction；f16 排序必须先写清
   NaN、`-0/+0` 和 total-order。bf16 是否同批进入由消费者证据决定。
3. **C 错误边界统一。** `hyle_simd_runtime_info_get` 和 RNG create 当前仍使用模块私有 TLS
   `hyle_simd_last_error`；应按已冻结 error 模块的 `hyle_status`/owned diagnostic 规则重审。
4. **scalable 真机证书。** SVE 与 RVV 已有正式实现和多 VL/VLEN QEMU correctness，仍缺真实
   SVE/RVV 硅上的代表性 workload 性能；SVE2/RVV 可选扩展只在真实收益成立时升级产品 routing。
5. **完整 closure 验证。** 新 range 层已过 MSVC/GCC fixed-width 及 SVE/RVV 多宽度 QEMU；仍需
   Clang、sanitizer、NEON 真机，C ABI 符号机械闸和安装包消费也须在最终轮重跑。

### P1：有消费证据后补

- whole-range argmin/argmax、通用 reduce 或 foreach；先看是否已有 `search`/`transform` 更清楚的表达。
- 为语言绑定增加**具名语义**的 batch kernel；不增加逐 lane C function-pointer transform。
- 扩展 compiler/package 矩阵，并把 exact/native/fat-kernel 示例纳入安装后 consumer 测试。
- 对 SVE2/RVV Zv* 增加独立产品 tier，仅限能在真实 workload 中稳定获益的操作族。

## 明确的 negative space

- PPC、S390、LoongArch、WASM 等当前非目标平台；这不包括 ARM SVE/SVE2 与 RISC-V RVV。
- `copy`/`fill`：标准库或 `mem` 已能表达，SIMD 包装没有独立语义收益。
- Highway contrib 的 hash、image、matvec、thread pool、topology/allocator：owner 分别是
  `container`、图像/compute 消费层、`compute`、`sched`、`platform`/`mem`，不能为能力表好看塞进 SIMD。
- 通用逐元素 C callback：调用边界妨碍内联/向量化；C ABI 保留具名 pointer/count 产品。
- 完整加密协议或密码库：寄存器 AES/CLMul 只是积木，安全产品需独立 threat model 和审查。

## 结论

HYLE SIMD 不是 Highway 的小型拷贝：它把大量能力收成预编译 batch API 和稳定 C ABI，也额外覆盖
AI dtype/激活/复数数学；Highway 则在平台、编译器、special-float lane 和通用 source-algorithm
层更成熟。第一轮补 range algorithm 是真实架构缺口，不是 LOC 驱动。SIMD **尚未严格收口**：
f16/bf16 lane、共享 error seam、scalable 真机性能和最终多平台证书仍是明确的 P0。
