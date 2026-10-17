# SIMD 路由与维护指南

本文面向维护者，说明当前 SIMD 实现怎样选择 ISA/provider、怎样增加或升级实现，以及怎样验证没有把
benchmark 噪声写进产品。长期边界见 [`simd-design.md`](../../architecture/simd-design.md)，用户用法见
[`simd-usage.md`](../../user/simd-usage.md)。

## 当前结构

SIMD 有两种消费路径，但只有一个公开语义面：

| 路径 | 用户看到的形状 | 选择成本 | provider 类型边界 |
|---|---|---|---|
| ordinary source/static/shared | `hyle/simd/*.h` 的 pointer/count、span、POD API | 未知 CPU 时每次整批调用一次已缓存 target 间接调用；exact build 为零 | Highway/xsimd 类型不出 implementation TU |
| expert source/static/shared SDK | `hyle::simd_expert`、`hyle_target_simd()` 或 `hyle_add_fat_kernel()` | fixed TU 为零；source-fat 在完整 pointer/POD kernel 边界选一次 | vendor vector/mask 只能留在 target-bound TU；installed SDK 自带锁定 provider 头 |

旧 `vec.h`、`native.h`、`native/*` 和 `backend/*` 已在 SIMD-16 物理删除。不要恢复兼容头、转发层或第三套
portable register DSL。普通寄存器操作直接使用中央锁定 Highway；xsimd 只进入具名 whole-buffer route；
HYLE 只保留有限且有证据的 target-native body。

主要 owner：

- `src/simd/**/kernel_impl.h`：ordinary family 的 Highway target body；
- `src/simd/math/highway_math_kernel-inl.h`：Highway math 与 xsimd family route；
- `src/simd/sort/vqsort_hyle/**`：保留的 Highway VQSort algorithm adaptation，内部直接使用锁定 Highway
  register 类型；
- `include/hyle/simd/provider/hyle_winners_{avx2,neon}-inl.h`：有限 HYLE winner body；不提供 fallback，
  不构成公共寄存器 API；
- `scripts/simd-op-routing.json`：操作清单与 Provider 选择；
- `scripts/simd-family-routing.json`：whole-buffer family 默认路线与编译器/ISA 例外；
- `include/hyle/simd/provider/{operation_manifest,routes_generated}.h`：由配置生成的 source-only 头；
- `include/hyle/simd/provider/route.h`：失败关闭的编译期选择类型，不包含运行时路由函数。

Platform 是 CPU/OS facts 的唯一 owner。SIMD 只把 Platform facts、当前编译器可生成的 target 和 family policy
求交集；不得再造 CPUID/sysctl/hwcap 探测器。

## 成本模型

ordinary fat library 在进程内初始化一次私有 target 表。之后每个 whole-buffer 调用最多执行一次已缓存的
函数指针跳转；这不是“每个 buffer 探测一次”。provider 已在 target TU 编译时确定，所以循环内不得出现：

- provider switch 或 CPU feature probe；
- 第二层 `HWY_DYNAMIC_DISPATCH`；
- 逐 lane callback；
- register bridge copy、spill 或临时 buffer；
- hidden allocation、lock 或 syscall。

`HYLE_SIMD_EXACT_TARGET` 和 `hyle_target_simd()` 必须把选择完全折叠为直接调用。source-fat 只允许在完整
pointer/POD kernel 边界选择一次。不同 provider 必须各自拥有完整 body；只有 release codegen 证明零额外
指令时才允许薄 adapter。

## Route policy

Highway 是同合同标准 primitive 的默认实现。以下情况优先保留 Highway，避免引入缺乏依据的专用实现：

- 最终汇编相同；
- 相同工作量下没有可复现的实际收益；
- 执行顺序、地址位置、编译器或相邻批次反转；
- 当前分辨率无法区分双方。

xsimd/HYLE 只有在语义、复杂度和适用的 type × target × compiler scope 都明确，且 current evidence 显示
material 优势时才能成为例外。例外不能以 buffer size、benchmark title、地址或采样顺序为 route 维度。

操作与 family 配置是代码生成输入。`RouteFor` primary 保持 `PENDING`，生成器为配置中的路线建立
特化；每项需要完整默认路线，编译器/ISA 例外不得重叠。`REMOVE_REDUNDANT` 的 route 不可调用；
`QualifiedProviderFor` 会失败关闭。

`Stream` 是一个重要示例：AVX2 最终 codegen 相同且测量反转，因此默认 Highway；AppleClang 21/NEON
上 Highway 的逐向量写预取在 Apple M4 大工作集慢 2.9–6.4 倍，因此只有该精确 scope 保留 HYLE plain-store
body。`FlushStream()` 只完成当前 core 的 non-temporal-store lifecycle；跨 core 发布仍须 release atomic、
mutex 或其他同步，不得把 store fence 当成线程同步协议。

## 怎样增加真实能力

1. 先写 consumer contract：输入/输出、alias、tail、NaN/rounding、复杂度、临时空间、target 和公开 owner。
2. 普通连续数据需求优先增加 pointer/count family API；只服务一个 fused source kernel 的 register spelling
   不进入 ordinary API。
3. 先在锁定 Highway 上实现完整 provider-coherent body；Highway 缺合同或真实热点存在时，再加入 xsimd/HYLE
   candidate。
4. 先比较语义、复杂度、遍历数、allocation/dispatch，再测相同工作量下的实际规模。
5. 配置中的 route 使用 `ROUTE_HIGHWAY`、`ROUTE_XSIMD`、`KEEP_HYLE` 或
   `REMOVE_REDUNDANT`；不得留下隐含 fallback。
6. target-bound correctness、最终 codegen 和普通 consumer 验证通过后再接入 family；不要先改公共 layout。

没有真实 consumer、可复现缺陷、标准或成熟 peer gap 时，不新增能力。未来推理引擎提出 tensor layout、
block quant、GEMM、attention、RoPE 或 fusion 需求时，先由 Compute/engine 拥有其 tensor/operator contract，
SIMD 只接收可复用的连续数据或 target-kernel primitive。

## 怎样升级 Highway/xsimd

1. 在中央 vendor lock 更新官方 commit；benchmark checkout 仍只是 peer，不能代替产品依赖。
2. 重算 integrated source hash 和 notices/license 投影。
3. 找出受上游源码/target 影响的操作和 family，验证其适用 ISA 与正常消费者。
4. 对 HYLE 例外重新确认其 material 优势。优势消失就路由回 Highway并删除 body；出现新的显著优势则新增
   一个有限 candidate，不复制整个 vendor surface。
5. 对 VQSort adaptation 移植相关 upstream 修复，核对 origin、local delta 和内容 hash。
6. 按受影响平台、构建与交付方式运行回归，记录真实结果。

## 配置与生成物

生成器读取操作配置、family 配置及中央依赖锁，检查名称、完整默认路线、selector 重叠和 Provider
revision，并核对两个生成头是否与配置一致：

```text
python scripts/simd-op-coverage.py --check
```

修改配置后重新生成并检查：

```text
python scripts/simd-op-coverage.py --write-provider-manifest --write-provider-routes
python scripts/simd-op-coverage.py --check
```

不要手工修改生成头。配置检查不执行 benchmark，也不证明数值或性能；这些用实际测试与消费者验证。

## 构建 seam

固定 TU：

```cmake
hyle_target_simd(TARGET my_kernel ISA AVX2)
```

未知 CPU 的 source-fat kernel：

```cmake
hyle_add_fat_kernel(
  TARGET my_app
  NAME my_kernel
  IMPL "src/my_kernel-inl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
```

entry 必须是相同的 pointer/count/POD 函数类型。impl header 内可以直接使用 `hwy::HWY_NAMESPACE` 或
xsimd；vendor 类型不能出现在 entry、公共类 layout、installed header 或 shared ABI 中。

经典 cl.exe 被锁定 Highway blocklist 禁止 AVX3：fixed/exact AVX-512 必须 configure fail-fast，fat build
明确封顶 AVX2。clang-cl/GCC/Clang 按支持的 target 构建。canonical x86 tier 不隐含 AES/PCLMUL；
专用 crypto kernel 必须有独立 feature 合同。

## 最小验证

每次 route/body/consumer 变更至少运行：

```text
python scripts/simd-op-coverage.py --check
cmake --build <build> --target hyle_simd_tests
ctest --test-dir <build> -L simd --output-on-failure
```

其中 SIMD suite 必须覆盖 ordinary API、route static assertions、有限 winner correctness、provider codegen
probe 与当前 ordering target。涉及 target/build seam 时还要验证：

- Windows MSVC、WSL GCC、macOS AppleClang 的适用 real target；
- source embedding、relocated static SDK、relocated shared SDK；
- exact/fat target；
- 受影响的 direct consumer；
- WSL ASan/UBSan（只有共享可变状态时才需要 TSan）。

性能使用与改动相关的 ordinary/inference whole-buffer workload。小幅落后也检查；先排除工作量、
仪器分辨率、顺序和位置偏差，再根据实现原因与真实收益调整 route。

## 交付与负空间

installed static/shared SDK 同时交付 ordinary SIMD 头、HYLE library，以及显式 opt-in 的
`hyle::simd_expert` seam 所需 `target.h`、`fat_kernel.h` 和锁定的 Highway/xsimd provider 头；普通
`hyle::simd` target 不传播这些 provider include。`provider/*` 仍是 HYLE 私有路由，不会安装。

SIMD 不提供 C API、Native FFI、稳定跨工具链 C++ ABI、per-ISA DLL、公开 register façade、GPU backend 或
模型 operator。不要为不存在的跨语言消费者导出伪 C ABI；未来有明确 runtime/engine FFI 时在其 owner
边界做类型擦除，而不是暴露模板或 target-dependent vector layout。
