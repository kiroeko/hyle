# simd-25：SVE/RVV 语义与产品层追平

> 状态：🚧 进行中（分类账、寄存器语义闭包、SVE/RVV 四类产品核、SVE 全 VL partition 与精细
> CPU feature 探测已落地；真机性能待完成）。依赖 simd-22 双轨与
> simd-24 已有产品核。
>
> 本里程碑重新限定 simd-15~18 的“追平”：它们完成的是 scalable P0~P4 正确性与
> generic 实例化，不代表后端本地名字、Runtime 产品 kernel 或真机性能证据已经和定宽同档。

## 目标

SVE/RVV 的完成标准不是复制 SSE4/AVX2/AVX512/NEON 的全部名字，而是同时闭合四层：

1. 每个定宽/scalable 差异都有语义分类，不再把本地定义数当用户能力数；
2. 可移植且有意义的寄存器语义能够在 SVE/RVV 实例化；
3. `dtype/math/sort/bitpack` 有真实 scalable kernel，不再无条件钳制到 NEON/scalar；
4. QEMU 多 VL/VLEN 正确性和真机 Highway 性能证据达到可签字水平。

```text
fixed-only 名字差集
        |
        v
语义 manifest + compile probes
        |
        +-- generic / selector ----------> 验证，不复制
        +-- fixed block/half model ------> 产品驱动实现或 N/A
        +-- portable / extension-gated --> SVE/RVV 原语
        |
        v
bitpack -> dtype -> math -> RVV sort -> SVE sort
        |
        v
Runtime / exact / native + QEMU + real-hardware parity
```

## 阶段 0：可信缺口底账

`scripts/simd-scalable-coverage.py` 机械复现四个定宽后端并集，并把 SVE、RVV **分别**与之比较，由
`scripts/simd-scalable-op-manifest.json` 逐项分类。v1 曾使用 SVE∪RVV：某个 op 只要任一后端实现，
另一后端的洞就会消失。RVV sort 补原语时该缺陷被 `--check` 暴露，v2 改为分类“至少一个 scalable
后端仍缺”的名字，并同时输出两份 backend-local missing 集。当前 23 个名字分为：

| 分类 | 数量 | 含义 |
|---|---:|---|
| `shared_generic` | 10 | 已在 `vec.h` 一次实现，并由正式 SVE/RVV 多 VL/VLEN 门禁直接实例化验证 |
| `selector` | 13 | Native/Emulated 内部选择器，不是独立用户能力 |
| `fixed_model` | 0 | 原 20 项 half/128-bit block/固定寄存器拼装均已按 scalable 合同补齐 |
| `extension_gated` | 0 | 当前固定宽 public union 中没有仍缺的扩展语义；硬件快路仍按 feature tier 独立演进 |
| `portable_missing` | 0 | 22 项可移植 backlog 已全部补齐；same-shift/rotate/rounding/shuffle/average、whole/pairwise/demote/scale/lookup 与 SAD 家族均有 SVE/RVV 实现 |

脚本同时输出 backend-local、`backend | vec.h`、scalable union/common 和 SVE/RVV 各自 missing 数；
这些仍只是语法面，不冒充编译可实例化或产品能力。当前 SVE、RVV 都只缺定宽并集 23 个名字，
两者共同实现 227 个、至少一端缺 23 个。它们恰好是 10 个已实例化的 `vec.h` shared generic 与
13 个非独立能力 selector；backend-local 的 portable/fixed/extension semantic gap 已归零。`--check`
要求这 23 项恰好分类一次，任何新增、消失或重复都会失败。

### 阶段 1 首轮进展（2026-07-17）

真实 SVE math 全函数实例化探针补出了并已修复两组隐藏洞：

- 浮点 `And/Or/Xor/AndNot/Not` 必须先 bitcast 到同宽 unsigned；原实现把浮点向量直接喂给
  整数 intrinsic；
- scalable mask 不是带 `PrivateT/PrivateN` 的固定宽向量，现有 mask logical 与
  `RebindMask` 必须有 predicate-native 路径；
- 补齐 `Combine/LowerHalf/UpperHalf/ZeroExtendVector` 的 sizeless 逻辑半向量语义，以及
  `ConvertInRangeTo/DemoteInRangeTo/GetBiasedExponent`。

修复后 SVE 的 f32/f64 全套 math（所有 unary、`atan2/pow/hypot`）可在 AArch64+SVE 目标完整
实例化。旧 v1 并集口径曾从 80 降至 73；该数字不能再作为双后端完成度，因为会掩盖 asymmetric
gap。v2 以逐后端口径重建时为 80 项；补齐两端 sort 传递原语与 SVE `ReverseBits` 后曾降至
52 项，后续只按这一口径收口。

随后补齐两端 9 项有稳定跨平台合同的 fixed-model 操作：byte/lane 左右移、combine-shift、
`PairwiseAdd128/PairwiseSub128` 与 `Per4LaneBlockShuffle`。实现均以 128-bit logical block 为边界，
SVE 使用 predicate + `tbl`，RVV 使用 mask + `vrgather`，没有 store/reload 标量化。SVE 的 384-bit
三块非二次幂 VL 也纳入门禁，证明 block 边界不串扰；缺口因此由 52 曾降至 43 项。

第二批继续补齐 `SlideUp/DownBlocks`、四种 block interleave、block extract/insert/broadcast、
`InsertIntoUpper` 与 `ResizeBitCast`，fixed-model 20 项至此归零。奇数 block 数沿用 Highway 合同：
384-bit 下 lower 为 `[a0,b0,a1]`，upper 为 `[a1,b1,a2]`；提取出的 sizeless native 值由
128-bit capped tag 消费低 block。两端均使用 table/gather、slide 与 predicate/mask，未引入标量数组。
sizeless 值本身不编码 capped source 宽度，因此跨逻辑宽度 resize 使用精确形状
`ResizeBitCast(to, from, v)`；它在 widening 时按 source tag 补零。两参数形状仅表示 full-register view，
不能用来猜测一个已经擦除的 capped source 宽度。
机械差集由 43 降至 32；随后 scalable extension/semantic 收口将其降至 23，剩余仅为已验证的
shared generic 与非独立能力 selector。

第三批裁决了原 `extension_gated` 的 9 项。`BitShuffle`/`MultiRotateRight` 参照 Highway generic
算法改为 SVE/RVV 纯向量 byte-table + variable-shift 实现，不依赖 SVE2/Zvbb；反汇编分别确认
SVE `tbl/lsr/lsl` 与 RVV `vrgather/vsrl/vsll`。`Stream` 在 SVE 使用架构级 `stnt1`，RVV 1.0
没有可移植 non-temporal store，按 Highway 合同退化为普通 `Store`，API 可用但不谎报 cache bypass。
SVE 六个 AES op 原已具备 SVE2-AES 快路和软件保底；RVV 现在补齐同一份 per-128-bit-block 软件能力
底座，base `rv64gcv` 不假定 Zvkned。它关闭的是能力洞，不是 RVV AES 性能证书；未来硬件快路必须
进独立 feature TU 并由 probe/dispatcher 选择。

### 首个产品层纵切：SVE/RVV math（2026-07-17）

`transcendental_highway_scalable.cc` 把同一份 HYLE D-tag math body 分别编成独立 `+sve` / `+v`
产品 TU，保持 pointer/count 接缝与 per-call `Lanes(d)`。baseline `math_dispatch.cc` 只在 Runtime
选中 SVE/RVV 时进入对应 TU；exact SVE/RVV 在编译期恒选对应符号，不查询 Runtime/CPU probe。已接入：

- f32/f64 的 HYLE/Highway unary 集合；
- `atan2/pow/hypot/sincos`；
- f64 table `exp/log` 及 in-range 入口；完整入口遇到异常 lane 时按整个 vector chunk 回落 libm，
  保持现有特殊值合同。

随后把 HYLE-owned xsimd-derived D-tag body 也编入同一 scalable TU：`erfc/exp10/fmod/fdim` 的
f32/f64、`tgamma` 的 f32/f64、`hypot_in_range` f32，以及完整 SoA complex 集合
`cexp/clog/csqrt/csin/ccos/ctan/csinh/ccosh/ctanh/cpow` 的 f32/f64。复数 typed D-tag body
在 AVX2、NEON、SVE、RVV 共用；regular chunk 保持向量计算，特殊值、分支切线、极端幅度或 pole
风险 chunk 在写输出前回落 `std::complex`，因此 exact in-place 仍成立。现在 math 只有 6 个有明确
正确性裁决的 scalar-only API：`lgamma/remainder/nextafter` 的
f32/f64；这不是模块降级。`lgamma` 是已复现的 vendor lane-mixing bug，后两者要求精确 libm 语义。

当前证据：两套脚本同时链接 CPU probe、真实 `Runtime`、baseline dispatcher、scalar 与 scalable
variant 表，不再只直调内部 kernel。加入 platform dispatch 与 public product 探针后，SVE 当前脚本为
**98 tests × VL 128/256/384/512/2048**（其中 1 项直调 NEON typed complex 产品核），RVV base 为 **93 tests × VLEN
128/256/512/1024**；公开 ops/random/search/scan/math/dtype/bitpack/sort batch API、tail、对应 module
tier 以及 extension/semantic 能力均在门内验证。
exact SVE/RVV dispatcher 的未定义符号检查也确认不引用 `runtime()`、`best_isa()` 或
`cpu_features()`。QEMU 证明正确性与派发，不构成真机性能结论。扩展构建也独立全绿：
SVE2-AES 的历史门为 **94×5**，当前 RVV Zvbb/Zvbc 各为 **94×4**；因此 base fallback 与已有硬件扩展路径没有
同名重定义或越级 codegen 冲突。

### 当前产品层矩阵

| 模块 | SVE | RVV | 下一缺口 |
|---|---|---|---|
| ops/random/scan/search | 真实 scalable variant | 真实 scalable variant | 真机性能证据 |
| math | 主 HYLE/Highway + HYLE-owned xsimd-derived scalable TU；6 个正确性裁决 scalar op | 同左 | 真机性能与复杂特殊值扩大验收 |
| dtype f16/bf16↔f32 | 真实 SVE 产品 TU | 真实 RVV 产品 TU | 真机性能；fp8 保持 scalar/可选硬件 |
| sort/select | full-VL partition + 128/256-bit capped network | 真实 RVV LMUL=1 产品 TU | 两端真机性能 |
| bitpack | 真实 SVE 产品 TU（固定 128-bit logical block） | 真实 RVV 产品 TU（固定 128-bit logical block） | 真机性能与多 logical-block 并行化 |

## 阶段 1：target/feature mechanization 与编译探针

- 扩充 platform feature，而不是只保留 `sve/rvv` 两个粗粒度 bool：SVE2/BF16/I8MM/AES，
  RVV Zvbb/Zvbc/向量 fp16 等分别探测；
- optional feature kernel 独立 TU，baseline detector 不得被高级 `-march` 污染；
- 用一份 target manifest 生成 source list、声明、Runtime 表和 exact binding，偿还 simd-24 已记录的
  “exact 绑定正确但产物仍含 fat TU”欠账；
- 为 `portable_missing`、`shared_generic` 和每个产品 body 建立 SVE/RVV compile-only probe。

SVE 的 VL 是线程属性；Runtime 只缓存 ISA/feature，具体 `Lanes(d)` 继续在 kernel 中求值，不能把某一
线程的 VL 固化成进程全局结论。

精细探测已落地：AArch64 baseline `cpu.cc` 用 `AT_HWCAP2` 分别报告
SVE2/AES/PMULL/I8MM/F32MM/F64MM/BF16，并保留非 SVE 的 Arm I8MM/BF16；RISC-V baseline 用 Linux
6.4+ 的 `riscv_hwprobe` 报告 Zvbb/Zvbc/Zvkned/Zvfh/Zvfhmin，旧内核或不支持该 syscall 的模拟器安全
保持 false。探测 TU 仍以基线 ISA 编译且 LTO 关闭，高级 `-march` 没有污染 detector。

QEMU 边界也已实测而非隐藏：qemu-aarch64 8.2 能通过 HWCAP2 正向报告 SVE2-AES/PMULL；同版本
qemu-riscv64-user 能执行 Zvbb/Zvbc 指令，但不模拟 `riscv_hwprobe` 的 optional-extension 结果。
因此 RVV 的 V 基础位在 QEMU 正向断言，optional 位用交叉编译期 synthetic bit-map `static_assert`
覆盖；脚本保留 `HYLE_QEMU_EXPECT_OPTIONAL_FEATURES=1`，供新版 QEMU、full-system guest 或真机开启
正向断言。不能用 `/proc/cpuinfo` 替代 hwprobe，因为异构 CPU 上它没有“所有可运行 CPU 的能力交集”
契约，可能误选后 SIGILL。

这些 bool 是 C++ 平台诊断 POD 的开发期扩展，不穿过 HYLE C ABI。现有 SVE2/Zvbb/Zvbc 能力仍属于
静态 target-bound 寄存器 API；在出现真实 pointer/count 产品 kernel 之前，`Runtime::Module::features`
不虚报“已选择该扩展核”。将来新增 optional 产品核时必须独立 TU，并由这些精细 feature 选择。

## 阶段 2：bitpack 与 dtype

### bitpack

`bitpack_body.inc` 现由 scalable 产品 TU 以 `CappedTag<T, 16/sizeof(T)>` 实例化。物理寄存器仍是
SVE/RVV sizeless native type，但 active lanes 固定为 16 bytes，packed block 因而不依赖线程 SVE VL 或
机器 RVV VLEN，并与 NEON layout 一致。Runtime/exact/public API 已接入 SVE/RVV。

验收覆盖 u8/u16/u32 的全部 56 个合法 bit width、两个完整 logical block、direct kernel、真实 Runtime
和 public API；每次输出逐字节对拍测试专用 scalar-128 D-tag 参考核，并在 SVE 五档/RVV 四档下复验。
11/13-bit u16 继续由独立手写布局 oracle 与全域测试钉住。当前实现每次只激活 128 bits；未来可在不改
格式的前提下并行处理多个 logical block，但必须先有真机收益证据。

### dtype

bf16↔f32、f16↔f32 已由共享 D-tag body 分别编成 SVE/RVV 产品 TU，并接入 Runtime/exact/public batch
API。bf16 四条路径 bit-exact；f16 解码全向量化，f32→f16 的 normal finite chunk 向量化，特殊值与
subnormal chunk 回到现有标量 oracle。验收穷举全部 65,536 个 f16/bf16 存储编码，并对 16,387 个
全位域 f32 样本验证编码结果。接口保持 pointer/count/POD，close-01 可直接包 C ABI。

fp8 继续保持 scalar 或 x86 AVX10.2 optional hardware path：目前没有可假定存在且已证收益的 base
SVE/RVV fp8 convert，不因 dtype module tier 已升级就把它伪装成 scalable。

## 阶段 3：math（产品 tier 已完成，性能验收继续）

SVE/RVV 的 HYLE/Highway 主 body 与 HYLE-owned xsimd-derived body 已生成真实产品 TU 并接 Runtime，
QEMU 多 VL/VLEN 的 direct、Runtime 和公开 API 门均完成。`tgamma`、f32/f64 complex、direct hypot 与 extras
已经 scalable；`lgamma/remainder/nextafter` 的 scalar 裁决有正确性证据。下一步是真机性能验收。
验收沿用现有 ULP 合同，继续扩大 NaN/Inf/subnormal/domain edge、tail 和跨 lane 混合输入。

Runtime 改为逐 op/feature 的 scalable 或已认证 fallback；不能仅修改 `runtime().math.isa` 元数据。

## 阶段 4：sort 分开处理

### RVV

HYLE VQSort 已完成 RVV 产品纵切。后端新增原生 gather/index 形式的 `LoadDup128`、
`OddEvenBlocks`、`SwapAdjacentBlocks`、`ReverseBlocks`、`Shuffle1032`，并用专门 backend test 钉住
VLEN=128 单 block 边界。产品核使用全宽 LMUL=1：负 `kPow2` 虽可缩短 D-aware op 的 `vl`，但
LMUL=1 native type 无法让 V-only swizzle 的 `DFromV` 恢复逻辑半宽，因此不能把它冒充 Highway 的
fractional LMUL。未来只有在真机证明寄存器压力/吞吐收益后，才扩展真正的 `mf2` native type。

验收覆盖六种 scalar key 与 KV32/KV64 的 sort/partial/select、升降序、随机/少值/有序/逆序、NaN
始终末尾、payload multiset、左右 canary、Runtime 和 public API。当前完整门禁 44 tests 在 VLEN
128/256/512/1024 全绿；exact RVV dispatcher 对 `runtime/best_isa/cpu_features` 的未定义符号检查为空。

### SVE

上游 Highway 会禁用真正 length-agnostic SVE VQSort，因此这不是机械移植任务。实测完整移植在
VL=128/256/512 全绿，但 VL=384 的 12-lane u32 基础网络即段错误，坐实了“合法 VL 不保证 2 的幂
lane 数”的模型墙。最终产品采用 hybrid：TU-local `HYLE_SIMD_SVE_DFROMV_MAX_BYTES=32` 与
`CappedTag` 只限制 sampling/base sorting network；递归的 O(n) partition 在子问题大于两个物理向量
时用 `ScalableTag` 恢复完整 VL。partition 的块余数从 power-of-two bitmask 改为通用 modulo，scratch
容量也与网络 cap 解耦、按 2048-bit 物理上界分配。VL=128 执行 128 bit；VL=256 时两层同为
256 bit；VL=384/512/2048 的 partition 分别使用 384/512/2048 bit，基础网络保持 256 bit。

它全程生成 SVE 指令、没有 NEON/scalar fallback，并在五档 VL 覆盖全部 sort/partial/select、六种
key、KV、NaN、canary、Runtime/public API，以及按物理 lane 数构造的 full-partition 专项；当前完整
门禁共 48 tests 全绿。SVE object 反汇编同时确认 `cntw/cntd` 与 `compact` 留在产品核中。基础网络
继续 capped 是算法合同，不再意味着大数组整核闲置 VL>256 lanes。

两者都覆盖 sort/partial/select、升降序、重复值、NaN total-order、KV、canary/ASan 和规模阶梯。

## 阶段 5：剩余原语（backend semantic gap 已收口）

产品传递闭包之外继续补 `AverageRound`、same-shift/rotate、rounding shift、pairwise、abs-diff、
two-table lookup 等 portable 项。fixed-model 项遵守三选一：

1. 产品依赖且合同明确：实现逐 128-bit block 语义；
2. scalable 有更自然的等价语义：提供并记录映射；
3. 只代表定宽寄存器拼装：manifest 记录 N/A，不制造 store/reload 假快路。

上述 portable 与 fixed-model 项已全部完成；最后 9 项的裁决和实现见阶段 1 第三批。现在 manifest
中的 23 个差集名不要求 backend 复制：shared generic 已由两端多 VL/VLEN 直接实例化，selector
只是实现选择钩子。后续新增固定宽 op 时，审计脚本会重新打开未分类缺口，不能靠更新数字掩盖。

## 阶段 6：验收与派发裁决

- SVE：VL 128/256/384/512/2048；
- RVV：VLEN 128/256/512/1024，工具链允许时加 2048；
- base 与 SVE2/Zvbb/Zvbc 等扩展分别构建；
- fat Runtime、exact、native 三条入口和实际 tier 探针；
- 反汇编确认无越级 ISA、无意外 scalar 主循环；
- 真机同编译器/同 ISA/同合同对拍 Highway，同时对比 SVE→NEON、RVV→scalar 的升级收益；
- 继续使用 `simd-op-benchmark-methodology.md` 的双序、多轮和置信区间，稳定低于 0.98 的项逐个狩猎。

若某 scalable kernel 经真机证据稳定不如 fallback，Runtime 可以选择后者，但 exact/native 实现仍应
可用，且诊断必须报告真实选择；这叫 per-module best-of，不叫缺实现。

## 完成闸门

- manifest 零未分类、零陈旧项；
- portable semantic gap 全部实现或有可审计的 ISA-gated 裁决；
- `dtype/math/sort/bitpack` 不再存在无条件 hard clamp；
- 所有 scalable 产品 kernel 通过多 VL/VLEN、tail、sanitizer/边界和 dispatch 探针；
- 真机 scorecard 完成，稳定 LOSE 均已修复或归因到明确 ISA/算法墙；
- 普通入口保持 pointer/count/POD，寄存器类型不穿过 Runtime/C ABI。
