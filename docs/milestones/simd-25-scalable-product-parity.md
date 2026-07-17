# simd-25：SVE/RVV 语义与产品层追平

> 状态：🚧 进行中（分类账、基础闭包与首个 SVE math 产品核已落地）。依赖 simd-22 双轨与
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

`scripts/simd-scalable-coverage.py` 机械复现四个定宽后端并集与 SVE/RVV 并集，并由
`scripts/simd-scalable-op-manifest.json` 逐项分类。初始 80 个 fixed-only 名字分为：

| 分类 | 数量 | 含义 |
|---|---:|---|
| `shared_generic` | 10 | 已在 `vec.h` 一次实现，只欠 scalable 实例化证明 |
| `selector` | 13 | Native/Emulated 内部选择器，不是独立用户能力 |
| `fixed_model` | 28 | half/128-bit block/固定寄存器拼装，按产品合同裁决 |
| `extension_gated` | 3 | 依赖可选 ISA 或流式存储合同 |
| `portable_missing` | 26 | 当前首要寄存器实现 backlog |

脚本同时输出 backend-local 与 `backend | vec.h` 两种机械数；后者仍只是语法面，不冒充编译可实例化
或产品能力。`--check` 要求全部 80 项恰好分类一次，任何新增、消失或重复都会失败。

### 阶段 1 首轮进展（2026-07-17）

真实 SVE math 全函数实例化探针补出了并已修复两组隐藏洞：

- 浮点 `And/Or/Xor/AndNot/Not` 必须先 bitcast 到同宽 unsigned；原实现把浮点向量直接喂给
  整数 intrinsic；
- scalable mask 不是带 `PrivateT/PrivateN` 的固定宽向量，现有 mask logical 与
  `RebindMask` 必须有 predicate-native 路径；
- 补齐 `Combine/LowerHalf/UpperHalf/ZeroExtendVector` 的 sizeless 逻辑半向量语义，以及
  `ConvertInRangeTo/DemoteInRangeTo/GetBiasedExponent`。

修复后 SVE 的 f32/f64 全套 math（所有 unary、`atan2/pow/hypot`）可在 AArch64+SVE 目标完整
实例化。机械差集从初始 80 降至 73：`fixed_model 28→24`、`portable_missing 26→23`；manifest
只保存当前未完成项，初始数字保留在上表作为基线。

### 首个产品层纵切：SVE math（2026-07-17）

`transcendental_highway_sve.cc` 已把 HYLE D-tag math body 编成独立 `+sve` 产品 TU，保持
pointer/count 接缝与 per-call `Lanes(d)`。baseline `math_dispatch.cc` 只在 Runtime 选中 SVE 时进入该
TU；exact SVE 在编译期恒选该符号，不查询 Runtime/CPUID。已接入：

- f32/f64 的 HYLE/Highway unary 集合；
- `atan2/pow/hypot/sincos`；
- f64 table `exp/log` 及 in-range 入口；完整入口遇到异常 lane 时按整个 vector chunk 回落 libm，
  保持现有特殊值合同。

这不是“整个 math 模块已纯 SVE”：gamma、complex、direct-form hypot 与 xsimd-only extras 仍使用原有
NEON/scalar per-op fallback。`Runtime::math.isa == kSve` 表示主 HYLE/Highway math tier 已启用 SVE，
不是声称每个冷门函数都有 SVE body。

当前证据：Windows Clang AArch64+SVE 对产品 TU、dispatcher、Runtime 与测试完成交叉编译；Windows
普通 x86 构建及 SSE4/AVX2/AES 回归通过。`sve-qemu-test.sh` 已纳入产品 TU、全 math compile probe、
多 VL core/tail 测试；实际 QEMU 128/256/384/512 执行仍等待可用的 Linux AArch64 cross toolchain，
因此此纵切尚未签为完成，亦无真机性能结论。

## 阶段 1：target/feature mechanization 与编译探针

- 扩充 platform feature，而不是只保留 `sve/rvv` 两个粗粒度 bool：SVE2/BF16/I8MM/AES，
  RVV Zvbb/Zvbc/向量 fp16 等分别探测；
- optional feature kernel 独立 TU，baseline detector 不得被高级 `-march` 污染；
- 用一份 target manifest 生成 source list、声明、Runtime 表和 exact binding，偿还 simd-24 已记录的
  “exact 绑定正确但产物仍含 fat TU”欠账；
- 为 `portable_missing`、`shared_generic` 和每个产品 body 建立 SVE/RVV compile-only probe。

SVE 的 VL 是线程属性；Runtime 只缓存 ISA/feature，具体 `Lanes(d)` 继续在 kernel 中求值，不能把某一
线程的 VL 固化成进程全局结论。

## 阶段 2：bitpack 与 dtype

### bitpack

现有 `bitpack_body.inc` 已是一份 D-tag body；SVE 全 body 已能编译实例化，但**尚不能直接接 Runtime**：
当前 packed layout 以 `Lanes(d)` 分块，SVE VL 又是线程属性。同进程中若 pack/unpack 发生在线程 VL
不同的两个线程，格式会变化。现有 SVE→NEON 钳制因此同时承担了“固定 128-bit 格式”的职责。

先把 bitstream 合同改成与线程 VL 无关的稳定格式（固定 logical block，或显式 format/VL metadata，
由最终跨线程与持久化合同裁决），再新增 SVE/RVV TU 与 Runtime/exact。覆盖所有类型、bits、非整 VL
tail、跨页 canary、跨线程不同 VL 的 pack/unpack round-trip。不能把“body 能编译”等同于格式可发布。

### dtype

依次实现 bf16↔f32、f16↔f32、经收益裁决的 fp8。基础位运算路径和 optional hardware path 分开；
接口保持 pointer/count/POD，close-01 可直接包 C ABI。RVV 当前 scalar fallback 优先消除。

## 阶段 3：math

SVE 的 HYLE/Highway 主 body 已生成真实产品 TU 并接 Runtime；下一步是完成 QEMU/真机验收，并裁决
gamma、complex、direct hypot 等 per-op fallback 是否值得新增 SVE 实现。RVV 仍需补传递闭包并生成
对应 TU。验收沿用现有 ULP 合同，并覆盖 NaN/Inf/subnormal/domain edge、tail 和跨 lane 混合输入。

Runtime 改为逐 op/feature 的 scalable 或已认证 fallback；不能仅修改 `runtime().math.isa` 元数据。

## 阶段 4：sort 分开处理

### RVV

Highway VQSort 有 RVV capped/fractional-LMUL 路径；先验证 HYLE LMUL=1 + 逻辑 `vl` 是否能维持网络
合同，再决定是否引入 fractional LMUL。传递缺口包括 block shuffle、`LoadDup128`、
`OddEvenBlocks`、`ReverseBlocks` 等。

### SVE

上游 Highway 会禁用真正 length-agnostic SVE VQSort，因此这不是机械移植任务。先以 SVE
predicate/compact 完成大数组 partition，小 sorting network 可暂用 NEON 或 capped SVE。只有产品热核
真实执行 SVE，模块才报告 `Isa::kSve`。

两者都覆盖 sort/partial/select、升降序、重复值、NaN total-order、KV、canary/ASan 和规模阶梯。

## 阶段 5：剩余原语

产品传递闭包之外继续补 `AverageRound`、same-shift/rotate、rounding shift、pairwise、abs-diff、
two-table lookup 等 portable 项。fixed-model 项遵守三选一：

1. 产品依赖且合同明确：实现逐 128-bit block 语义；
2. scalable 有更自然的等价语义：提供并记录映射；
3. 只代表定宽寄存器拼装：manifest 记录 N/A，不制造 store/reload 假快路。

## 阶段 6：验收与派发裁决

- SVE：VL 128/256/384/512，另加 2048 边界；
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
