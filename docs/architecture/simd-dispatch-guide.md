# SIMD native/runtime 双轨指南

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
      baseline dispatcher      exact compile-time             |
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

     custom kernel: hyle_add_fat_kernel -> per-ISA entries -> select_fat_kernel
     future C ABI: opaque handle + POD + version/size vtable -> ordinary batch API only
```

动态和固定目标只在进入完整 kernel 前分叉；进入 `native` body 后都使用同一个 backend 实现。寄存器
类型绝不穿过 per-ISA entry，C ABI 也绝不暴露 `Runtime`、D-tag、C++ 类布局或 STL。

## 用户面分层与易用性判断

“387 个公开/selector 名”不等于普通应用要直接面对 387 个函数，也不等于它们全是 internal：

| 层 | 当前规模（2026-07-16） | 面向谁 | 稳定边界 |
|---|---:|---|---|
| C++ batch core（开发期） | 107 个不同函数名、153 个导出 overload | HYLE 内部/同工具链测试 | pointer/POD/opaque state；不是发布 ABI |
| `native::` whole-buffer kernel | ops/random/scan | 固定 ISA、需要跨调用内联的源码用户 | target-bound C++，不承诺跨 ISA 类型稳定 |
| `vec.h` 寄存器 DSL | 387 个去重公开/selector 名 | 编写自定义 SIMD kernel 的高级用户 | target-bound vector/mask；只在 TU 内编译期实例化 |
| `detail::*_{isa}` | 每 ISA 稳定符号薄包装 | HYLE dispatcher/测试 | internal，不是用户 API |

规模可用 `rg "^HYLE_SIMD_API" include/hyle/simd` 与 `scripts/simd-op-coverage.py` 重算。普通语义面已经覆盖
math、dtype、sort/select、scan、bitpack、RNG 与基础 ops，并非只有少数函数。

与 Highway 的边界不同：Highway 核心同样是几百个 target-bound op，普通动态调用往往需要用户自己写
target function 再 `HWY_EXPORT/HWY_DYNAMIC_DISPATCH`；HYLE 的预编译 batch API 更直接。Highway 的优势是
`contrib/{algo,dot,hash,image,intdiv,matvec,random,sort,...}` 分类和组合入口更成熟。

普通应用的发布外观不归 SIMD C++ 头所有：

1. `include/hyle/c/simd.h` 将由 close-01 提供唯一稳定入口，只含 pointer/count、POD、opaque handle
   和版本化函数表；C、C++、C#、Java、Rust 等消费方在各自工程里适配本语言容器。
2. HYLE 不提供 `std::span`/`std::vector` 伞头，也不在核心仓库维护某一种语言的容器便利层；这避免
   “C ABI 是发布契约，但文档却把 C++ facade 当普通用户入口”的双重产品面。
3. 是否补 `algo`、hash、image、matvec 等高层能力按 HYLE 模块边界和真实消费者决定，不为了镜像 Highway
   contrib 目录而堆 API。
4. `vec.h`/`native::` 是源码级 expert surface，不承诺稳定 ABI，也不伪装成发布入口。

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

## 普通用户：不知道目标 CPU

发布完成后，普通用户只调用 close-01 的 C ABI，例如：

```c
#include "hyle/c/simd.h"

hyle_simd_add_f32(x, y, out, count);
hyle_simd_exp_f32(x, out, count);
hyle_simd_inclusive_scan_f32(out, count);
```

上面的确切命名和版本化函数表在 close-01 开工时冻结；这里钉的是边界形状，不提前伪造已实现 API。
预编译库内含各 ISA kernel，C shim 转发到内部 batch core，由 Runtime 根据 platform 探测选择最佳实现。
用户不自行调用 CPUID，也不需要知道 HYLE 内部的 C++ `Runtime`、D-tag 或 STL。

在 close-01 完成前，现有 `include/hyle/simd/*.h` 仅供仓库开发、测试和同工具链源码集成；它们不是
可发布承诺。消费方需要容器便利时，在自己的边界上用 `.data()/size()`、slice、Span 或 MemorySegment
包装 C ABI，HYLE 核心不替任一语言选择容器模型。

HYLE 内部诊断或同工具链开发需要把选择提升到更外层时，可以保存 C++ 表引用：

```cpp
const hyle::simd::Runtime& rt = hyle::simd::runtime();
rt.ops.add_f32(x, y, out, count);
auto math_tier = rt.math.isa;
bool has_f16c = (rt.dtype.features & hyle::simd::Runtime::kFeatureF16c) != 0;
```

`runtime_for(Isa)` 用于测试、限档或显式运行策略。它仍走稳定函数表边界，**不是固定目标直连接口**；请求当前
CPU/OS 不支持或本架构未编译的精确 tier 时返回 `nullptr`。

## 已知目标用户：两种精确目标模式

若整个 HYLE 库只部署到一种 ISA，配置：

```cmake
cmake -S hyle -B build-avx2 -DHYLE_SIMD_EXACT_TARGET=AVX2
```

此时所有普通 batch API 都保留原签名，但 dispatcher 在编译期直连该 ISA 的实现；ops、RNG、scan、
dtype、math、sort、bitpack 的 dispatch object 均不引用 `runtime()`。库不会执行 CPUID，并要求运行 CPU
满足所选 canonical ISA。AVX-512-BF16/AVX10.2 这类非 canonical 可选扩展不因 `AVX512` 名称而被臆定。

若只固定自己的一个 target，并希望 kernel 跨调用内联：

源码用户在 CMake 中把自己的 target 固定到一个 ISA：

```cmake
hyle_target_simd(TARGET my_app ISA AVX2)
```

然后调用 `native`：

```cpp
#include "hyle/simd/native/ops.h"
#include "hyle/simd/native/random.h"

hyle::simd::native::add(x, y, out, count);
hyle::simd::native::random_fill(state, random, count);
```

`hyle_target_simd` 同时设置 target 宏和编译器 flag。当前 `native` 完整 kernel 面为 ops、RNG 和 scan；
函数具有 TU-local linkage 并直接使用当前
backend，允许完全内联；它不调用 `runtime()`、不探测 CPUID、也不经过函数指针。生成的二进制要求部署
CPU 真正支持所选 ISA。

需要组合新算法的高级用户也在固定 target 的 TU 中使用 `vec.h` 的 387 个公开/selector 积木。若同一个自定义算法要
发布给未知 CPU，应把**整个 pointer/POD kernel**编成多个 variant，再在外层派发，不能动态调用每个
`Add`/`Load`。

## 自定义 fat kernel

源码用户用 `hyle_add_fat_kernel` 将同一 impl header 机械编成各 ISA 的稳定 pointer/POD entry，baseline
TU 用 `select_fat_kernel(...)` 选择并缓存函数指针。精确目标 TU 中选择在编译期完成；未知 CPU 中只在
缓存前探测一次。该机制禁止 vector/mask 类型跨 entry，variant header 必须只维护一份算法 body。

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

Linux AArch64 通过 `AT_HWCAP.SVE`、Linux RISC-V 通过 `AT_HWCAP.V` 探测。预编译 runtime 为
ops/random/scan/search 生成 SVE/RVV 变体；HYLE/Highway math 主 op 也由同一个 scalable body 分别编成
独立 `+sve` / `+v` 产品 TU。gamma/complex/xsimd-only 等少量 op 仍保留已验证 NEON/scalar fallback。
SVE 的 dtype/sort/bitpack 仍保持 NEON，RVV 的 dtype/sort/bitpack 仍保持 scalar；这些产品层缺口由
simd-25 逐模块消除，不用“后端头能编译”代替 Runtime 证据。

SVE math 已通过 24 tests × VL 128/256/384/512，RVV math 已通过 21 tests × VLEN
128/256/512/1024；门内包含真实 CPU probe、Runtime、公开 batch API 与 tail。sizeless SVE/RVV vector
不能成为 `VectorXoshiro` 成员，所以固定目标代码使用 `native::random_fill`。这些 QEMU 结果是正确性与
派发证据，真机性能证书仍待补齐。

## 与 Highway 的关系

Highway 通过 `foreach_target.h` 重包含 TU、target namespace 与 `HWY_EXPORT/HWY_DYNAMIC_DISPATCH`
实现相同边界。HYLE 借其原则，不复制自包含宏系统：CMake 生成独立 variant TU，`Runtime` 保存选中的
函数表，native header 仍是正常 C++ 头。

## C ABI 边界

本页描述的是 C++ 便利面，不改变 `close-01`：发布时唯一承诺稳定 ABI 的接缝仍是
`include/hyle/c/*` 的纯 `extern "C"` shim。C shim 将使用 opaque handle、POD、显式版本/大小和函数
指针 vtable；不会把 `Runtime` C++ 类型、`RandomState` 类布局、模板或 STL 类型直接冻结为 C ABI。

## 变体安全合同

1. dispatcher/platform probe 保持 baseline 编译，SIMD 模块关闭 LTO；宽指令只存在于 per-ISA TU。
2. scalar fallback 永远存在；显式 native target 除外——它按合同要求对应硬件。
3. 每个 runtime kernel 对拍 scalar oracle；每个 native header 至少在 SSE4/AVX2/AVX-512/NEON 编译；
   scalable body 还必须在至少三种 SVE VL / RVV VLEN 下执行。
4. 只在 whole-buffer 边界付一次间接调用；禁止在 lane/寄存器循环内查 runtime 表。
