# SIMD native/runtime 双轨指南

> **当前状态**：本页只维护 C++ SIMD 派发与 ISA 形状。Native FFI 已裁决为 `N/A`；旧
> C schema、bridge、入口计数和 managed fixture 已由 BUILD-03 删除，不参与当前架构、用户面或证书。

HYLE 的派发边界按**类型契约**划分，不按“头文件还是 `.cc`”划分。一个函数只要让
target-dependent vector/mask 类型穿过边界，就只能在编译期确定 target；一个函数若只有指针、标量、
长度和不透明状态，就可以编成多个 ISA 变体并在稳定边界选择。

## 架构总图

```text
                         semantic whole-buffer kernel
                                      |
             +------------------------+-------------------------+
             |                        |                         |
     unknown customer CPU     fixed HYLE library       fixed consumer TU
       ordinary batch API      ordinary batch API          native:: API
             |                        |                  hyle_target_simd
      baseline dispatcher      generated target manifest      |
             |                 one Runtime table/source set   |
             |                 direct target symbol           |
      cpu_features/best_isa            |                      |
             |                         |                      |
      immutable Runtime                |                      |
      fn table / module metadata       |                      |
             |                         |                      |
             +---------- per-ISA stable entry ----------------+
                                      |
                          one shared native D-tag body
                                      |
                                    vec.h
                                      |
                         compile-time selected backend
             scalar / SSE4 / AVX2 / AVX512 / NEON / SVE / RVV

     custom kernel: hyle_add_fat_kernel -> per-ISA entries -> HYLE_SIMD_SELECT_FAT_KERNEL
```

fat 与 exact 的 generated TU、手写产品 TU、Runtime table 和 exact suffix 都由
`hyle_simd_target_manifest` 的同一 target list 驱动。动态和固定目标只在进入完整 kernel 前分叉；进入 `native` body 后都使用同一个 backend 实现。寄存器
类型绝不穿过 compiled per-ISA entry；target-dependent register/mask 类型只留在对应 C++ TU。

## 用户面分层与易用性判断

“387 个公开/selector 名”不等于普通应用要直接面对 387 个函数，也不等于它们全是 internal：

| 层 | 当前规模（2026-07-16） | 面向谁 | 稳定边界 |
|---|---:|---|---|
| C++ batch core（开发期） | 公开 whole-buffer overload | HYLE 内部/同工具链测试 | 进入同一 dispatcher；不是发布 ABI |
| `native::` whole-buffer kernel | ops/random/scan/search | 固定 ISA、需要跨调用内联的源码用户 | target-bound C++，不承诺跨 ISA 类型稳定 |
| `vec.h` 寄存器 DSL | 387 个去重公开/selector 名 | 编写自定义 SIMD kernel 的高级用户 | target-bound vector/mask；只在 TU 内编译期实例化 |
| `detail::*_{isa}` | 每 ISA 稳定符号薄包装 | HYLE dispatcher/测试 | internal，不是用户 API |

规模可用 `rg "^HYLE_SIMD_API" include/hyle/simd` 与 `scripts/simd-op-coverage.py` 重算。普通语义面已经覆盖
math、dtype、sort/select、scan、bitpack、RNG 与基础 ops，并非只有少数函数。

与 Highway 的边界不同：Highway 核心同样是几百个 target-bound op，普通动态调用往往需要用户自己写
target function 再 `HWY_EXPORT/HWY_DYNAMIC_DISPATCH`；HYLE 的预编译 batch API 更直接。Highway 的优势是
`contrib/{algo,dot,hash,image,intdiv,matvec,random,sort,...}` 分类和组合入口更成熟。

当前 C++ 用户面与 internal seam 必须分开：

1. C++ 应用使用 `include/hyle/simd/*.h` 的 whole-buffer source API；它不承诺跨编译器/STL ABI。
2. 是否补 `algo`、hash、image、matvec 等高层能力按 HYLE 模块边界和真实消费者决定，不为了镜像 Highway
   contrib 目录而堆 API。
3. `vec.h`/`native::` 是源码级 expert surface，不承诺稳定 ABI，也不伪装成跨 ISA 类型边界。

`container/hash.h` 和 SwissTable 不属于这个 facade。container 在依赖图上有意不依赖 simd，可在模块
内部使用自包含 ISA 路径和 platform CPU probe；其性能用独立 hash/hashmap SOTA benchmark 收口，不能
为了复用 Runtime 反转依赖，也不能从 `hyle::simd` 再导出 hash。

## 380+ 个 op 是否都要双腿

当前 `scripts/simd-op-coverage.py` 从 `vec.h` 与各 backend 机械提取出 387 个公开/selector 寄存器名。
它们**都能用于两种部署方式的 kernel body**，但不应各自拥有一个动态函数：

| 能力层 | 例子 | native | runtime | 原因 |
|---|---|---:|---:|---|
| tag/类型/寄存器构造 | `ScalableTag`、`Lanes`、`Set` | 是 | 否 | 类型和 lane 数就是 target 的一部分 |
| 单寄存器计算 | `Add`、`MulAdd`、`Eq`、`Shuffle*` | 是 | 否 | 每 op 间接调用会阻断内联和寄存器生命周期 |
| 单寄存器内存原语 | `Load`、`Store`、`Gather`、`Compress` | 是 | 否 | 动态包装仍无法提供统一返回向量类型 |
| 硬件积木 | AES round、CLMul、mask/bit 操作 | 是 | 否 | 应派发完整 AES/hash/parser kernel，而非一轮指令 |
| whole-buffer kernel | `dot`、`add`、`random_fill`、scan、dtype convert | 是 | 是 | 指针/POD ABI 稳定；一次选择可被整个循环摊销 |
| 算法 kernel | sort、bitpack、transcendental、softmax | 按需 | 是 | 预编译库必须适配未知 CPU；静态腿只在确有源码用户需求时提供 |
| 只有 scalar 胜出/实现的操作 | 部分 libm 或策略路径 | 可选 | 直接 scalar | 没有第二个有价值的实现就没有派发必要 |

所以双轨的单位是**完整 kernel**，不是 387 个寄存器名。判断一个新能力是否进入 runtime 表，需要同时满足：

1. 能用 pointer/POD/opaque-state 表达，不泄漏 vector/mask 类型；
2. 工作量足以摊销每 batch 一次函数指针调用；
3. 至少有两个值得保留的实现，或必须提供 scalar 安全兜底；
4. 它是用户能理解的语义操作，而非某个 ISA 的内部拼装零件。

## 不知道目标 CPU：C++ runtime batch

可分发 C++ binary 通过 whole-buffer API 进入 baseline dispatcher，由 `Runtime` 根据 platform probe
选择已编译且受支持的实现。用户不自行调用 CPUID，也不让 D-tag/register 类型跨 compiled seam。
现有 C++ surface 仍待 SIMD v2 复审，不能由旧寄存器 benchmark 自动获得产品证书。

HYLE 内部诊断或同工具链开发需要把选择提升到更外层时，可以保存 C++ 表引用：

```cpp
const hyle::simd::Runtime& rt = hyle::simd::runtime();
rt.ops.add_f32(x, y, out, count);
auto math_tier = rt.math.isa;
bool has_f16c = (rt.dtype.features & hyle::simd::Runtime::kFeatureF16c) != 0;
```

fat 产物中的 `runtime_for(Isa)` 用于测试、限档或显式运行策略。它仍走稳定函数表边界，**不是固定目标
直连接口**；请求当前 CPU/OS 不支持或本架构未编译的 tier 时返回 `nullptr`。exact 产物只有一张
canonical table：请求该 tier 返回它，其余一律 `nullptr`，并且不会为了诊断查询 CPU。

`Module::isa` 表示该模块的**主产品 tier**，不是“每个 op 都使用该 ISA”。`features` 公开以下诊断事实：

| feature | 含义 |
|---|---|
| `kFeatureMathExactLibmScalar` | 精确语义裁决的 libm 操作仍逐项 scalar |
| `kFeatureDtypeFp8Scalar` | fp8 基础路径仍是 scalar/可选硬件档 |
| `kFeaturePerOpLowerTier` | 至少一个已认证 winner 有意使用较低 canonical tier |
| `kFeatureSortCappedNetwork` | SVE partition 吃满 VL，但基础 network 有 128/256-bit cap |
| `kFeatureBitpackLogical128` | SVE/RVV 指令可 scalable，但 logical block 固定 128 bit，字节不随 VL/VLEN 改变；不承诺跨 tier wire format |

该表是版本化的 C++ 诊断便利面（当前 `kTableVersion=6`），不是 close-01 的 Native FFI 候选 schema。

## 已知目标用户：两种精确目标模式

若整个 HYLE 库只部署到一种 ISA，配置：

```cmake
cmake -S hyle -B build-avx2 -DHYLE_SIMD_EXACT_TARGET=AVX2
```

此时所有普通 batch API 都保留原签名，但 dispatcher 在编译期直连该 ISA 的实现；ops、RNG、scan、
dtype、math、sort、bitpack 的 dispatch object 均不引用 `runtime()`。库不会执行 CPUID，并要求运行 CPU
满足所选 canonical ISA。构建只生成一张 canonical Runtime table 和一套对应的 generated kernel suffix；
scalar 精确语义核、AVX512 中实测胜出的 AVX2 per-op 核仍可作为目标产品的显式依赖，这不等于重新带回
fat dispatch。AVX-512-BF16/AVX10.2 这类非 canonical 可选扩展不因 `AVX512` 名称而被臆定。

每个 exact 配置都会提供 `hyle_simd_exact_artifact_check`：它核对 manifest、Ninja source graph、CPU-probe
导入和 x86 对象的越级指令。当前 Windows 已覆盖 scalar/SSE4/AVX2/AVX512；SVE/RVV 用
`scripts/simd-scalable-exact-qemu-test.sh` 检查单 suffix、无 probe 引用并执行多 VL/VLEN。

若只固定自己的一个 target，并希望 kernel 跨调用内联，必须使用源码嵌入构建：

源码用户在 CMake 中把自己的 target 固定到一个 ISA：

```cmake
hyle_target_simd(TARGET my_app ISA AVX2)
```

helper 会把中央锁定 Highway 的 include path、单目标策略和 ISA flag 绑定到该 target。kernel 直接使用
Highway：

```cpp
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;
const hn::ScalableTag<float> d;
const auto sum = hn::Add(hn::LoadU(d, x), hn::LoadU(d, y));
hn::StoreU(sum, d, out);
```

`hyle_target_simd` 不调用 `runtime()`、不探测 CPUID、也不经过函数指针；生成的二进制要求部署 CPU
真正支持所选 ISA。这个 target seam 不随 installed static/shared SDK 导出，因为 vendor register 类型
不是 HYLE ABI。经典 cl.exe 的 Highway AVX3 被上游 blocklist 禁止，helper 对 AVX512 fail-fast；应使用
clang-cl 或 AVX2。

旧 `vec.h/native/*` 只服务尚未迁移的 HYLE family 与 container build-private seam；不要新增消费者。
若同一个自定义算法要发布给未知 CPU，应把**整个 pointer/POD kernel**编成多个 variant，再在外层派发，
不能动态调用每个 `Add`/`Load`。当前 fat-kernel 的 Highway 迁移归 SIMD-06，在其完成前不扩张用户承诺。

## 自定义 fat kernel

源码用户用 `hyle_add_fat_kernel` 将同一 impl header 机械编成各 ISA 的稳定 pointer/POD entry，baseline
TU 用 `HYLE_SIMD_SELECT_FAT_KERNEL(...)` 选择并缓存函数指针。该宏在 fat 构建中调用
`select_fat_kernel`，在 exact 构建中会在预处理阶段只保留 manifest 生成的那个 symbol，因而不会
odr-use 未生成的其余变体。未知 CPU 中只在缓存前探测一次。该机制禁止 vector/mask 类型跨 entry，
variant header 必须只维护一份算法 body。

fat 构建的 `TARGETS` 必须包含当前平台的完整 canonical tier（x86 四档、Linux ARM64
scalar/NEON/SVE、macOS ARM64 scalar/NEON、RISC-V scalar/RVV）；exact 构建至少必须包含所选 tier。
CMake 在配置期拒绝缺档清单。调用代码仍可写跨平台的完整参数列表：macOS 会在预处理阶段丢弃不存在的
SVE symbol。

以 x86 四档为例，调用拼写在 fat/exact 构建中不变：

```cpp
using Fn = void (*)(const float*, float*, std::size_t) noexcept;
static const Fn fn = HYLE_SIMD_SELECT_FAT_KERNEL(
    &transform_scalar, &transform_sse4, &transform_avx2, &transform_avx512);
fn(src, dst, count);
```

`select_fat_kernel` 仍是所有变体都真实存在时可直接使用的低层函数。公开 CMake helper 与 exact
manifest 配套时应使用大写选择宏；`tests/simd/fat_kernel_fixture.h` 是一份真实 pointer/count
纵切回归，不再只测试四个手写空函数。

## 一份实现，两个入口

HYLE 自带 kernel 遵循：

```text
native header 中的一份 D-tag body
        ├─ 用户 TU 直接 inline 调用             (native)
        └─ CMake 生成 scalar/SSE4/AVX2/... TU
                 └─ Runtime 函数表选择           (runtime)
```

`ops`、`random` 与 `scan` 已采用这套结构。不得维护一份 fixed-target 算法和另一份 dynamic 算法；variant TU 只能是
给同一 native body 加稳定符号名的薄包装。

## 状态对象与随机数

返回 native vector 的 `native::VectorXoshiro` 只服务固定宽 target-bound 代码；SVE/RVV 的向量类型
不能作为类成员，使用同为零派发的 `native::random_fill`。普通用户使用不透明
`RandomState + random_fill`；它维护固定的八个逻辑流，各 ISA 只改变一次并行更新几个流，因此相同 seed
与调用序列跨 scalar/SSE4/AVX2/AVX-512/NEON 位一致。

这仍是非加密 xoshiro256++。若未来 dropout/分布式采样要求在 worker 数、任务切分和 GPU 之间也保持
按元素严格复现，应增加 counter-based Philox 合同，而不是继续扩大 native vector state 的 ABI。

## SVE 与 RVV

Linux AArch64 以 `AT_HWCAP.SVE` 探测基础档，并从 `AT_HWCAP2` 分别探测
SVE2/AES/PMULL/I8MM/F32MM/F64MM/BF16；Linux RISC-V 以 `AT_HWCAP.V` 探测基础档，并用
`riscv_hwprobe` 探测 Zvbb/Zvbc/Zvkned/Zvfh/Zvfhmin。探测器是独立的 baseline、LTO-off TU，老内核
不支持 `riscv_hwprobe` 时可选位保持 false，不会因编译机 feature 或 `/proc/cpuinfo` 误报而 SIGILL。

预编译 runtime 为 ops/random/scan/search 生成 SVE/RVV 变体；math 的主 HYLE/Highway op 与 HYLE-owned
xsimd-derived op 也已有独立 scalable 产品 TU。只有 `lgamma/remainder` 的 f32/f64 因明确
正确性裁决保留 scalar；`nextafter` 的普通 finite-normal chunk 已是精确 1-ULP scalable body，
边界逐 lane 回 libm。常用 dtype f16/bf16↔f32 已有真实 SVE/RVV 产品 TU；fp8 因 base scalable ISA
没有已认证的通用转换收益而保留 scalar/可选硬件路径。

bitpack 的物理指令使用 scalable backend，但 logical D-tag 固定为 128-bit，因此格式不随线程 SVE VL
或机器 RVV VLEN 改变，并与 NEON block layout 相同。sort/partial/select 覆盖六种 scalar key 与
KV32/KV64：RVV 使用诚实的 LMUL=1 全宽 tag；SVE 的递归 partition 使用完整物理 VL，只有需要 2 的幂
lane 数的 sorting network 保持 128/256-bit cap。

当前 fat 脚本门为 SVE/SVE2-AES **102 tests × VL 128/256/384/512/2048**，RVV
base/Zvkned **97 tests × VLEN 128/256/512/1024**；exact 门另为 SVE **44×5**、RVV **44×4**。门内包含 CPU probe、
Runtime、公开 ops/random/search/scan/math/dtype/bitpack/sort batch API、tail 和 feature 蕴含关系。
qemu-aarch64 能正向验证 HWCAP2 扩展位；qemu-riscv64-user 8.2 能执行 Zvbb/Zvbc，但不模拟
`riscv_hwprobe` 可选位，故 RVV
可选位解码另有交叉编译期 synthetic bit-map 断言，真机仍须正向复验。dtype 还穷举全部 f16/bf16
存储编码并覆盖全位域 f32 样本。sizeless SVE/RVV vector 不能成为 `VectorXoshiro` 成员，所以固定目标
代码使用 `native::random_fill`。这些 QEMU 结果是正确性与派发证据，真机性能证书仍待补齐。

## 与 Highway 的关系

Highway 通过 `foreach_target.h` 重包含 TU、target namespace 与 `HWY_EXPORT/HWY_DYNAMIC_DISPATCH`
实现相同边界。HYLE 借其原则，不复制自包含宏系统：CMake 生成独立 variant TU，`Runtime` 保存选中的
函数表，native header 仍是正常 C++ 头。

## Native FFI

当前为 `N/A`。旧 SIMD shim 已由 BUILD-03 删除。未来只有通过全局
[准入策略](../../architecture/native-ffi-policy.md)的真实任务才另建窄的 batch 边界；本页的 C++
`Runtime`、D-tag、register/mask、模板和类布局都不会由此自动导出。

## 变体安全合同

1. dispatcher/platform probe 保持 baseline 编译，SIMD 模块关闭 LTO；宽指令只存在于 per-ISA TU。
2. scalar fallback 永远存在；显式 native target 除外——它按合同要求对应硬件。
3. 每个 runtime kernel 对拍 scalar oracle；每个 native header 至少在 SSE4/AVX2/AVX-512/NEON 编译；
   scalable body 还必须在至少三种 SVE VL / RVV VLEN 下执行。
4. 只在 whole-buffer 边界付一次间接调用；禁止在 lane/寄存器循环内查 runtime 表。
5. x86 partial 以 XMM/YMM/ZMM 分层表达；sub-XMM 逻辑向量物理上仍是 XMM。升级编译器时必须跑
   exact-byte/guard-page、half/combine oracle 与 `op_parity_widen_bench`，不能从“类型有 N”推断 codegen 必然健康。
