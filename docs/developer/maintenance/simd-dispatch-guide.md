# SIMD 路由与维护指南

本文面向维护者，说明当前 SIMD 实现怎样选择 ISA/provider、怎样增加或升级实现，以及怎样验证没有把
benchmark 噪声写进产品。长期边界见 [`simd-design.md`](../../architecture/simd-design.md)，用户用法见
[`simd-usage.md`](../../user/simd-usage.md)，当前状态只看
[`module-status.json`](../../implementation/module-status.json)。

## 当前结构

SIMD 有两种消费路径，但只有一个公开语义面：

| 路径 | 用户看到的形状 | 选择成本 | provider 类型边界 |
|---|---|---|---|
| ordinary source/static/shared | `hyle/simd/*.h` 的 pointer/count、span、POD API | 未知 CPU 时每次整批调用一次已缓存 target 间接调用；exact build 为零 | Highway/xsimd 类型不出 implementation TU |
| 完整源码 expert | `hyle_target_simd()` 或 `hyle_add_fat_kernel()` | fixed TU 为零；source-fat 在完整 pointer/POD kernel 边界选一次 | vendor vector/mask 只能留在 target-bound TU |

旧 `vec.h`、`native.h`、`native/*` 和 `backend/*` 已在 SIMD-16 物理删除。不要恢复兼容头、转发层或第三套
portable register DSL。普通寄存器操作直接使用中央锁定 Highway；xsimd 只进入具名 whole-buffer route；
HYLE 只保留有限且有证据的 target-native body。

主要 owner：

- `src/simd/**/kernel_impl.h`：ordinary family 的 Highway target body；
- `src/simd/math/highway_math_kernel-inl.h`：Highway math 与已签 xsimd family route；
- `src/simd/sort/vqsort_hyle/**`：保留的 Highway VQSort algorithm adaptation，内部直接使用锁定 Highway
  register 类型；
- `include/hyle/simd/provider/hyle_winners_{avx2,neon}-inl.h`：有限 HYLE winner body；不提供 fallback，
  不构成公共寄存器 API；
- `scripts/simd-op-routing.json`：冻结的 387-operation baseline 与终态 provider 账；
- `include/hyle/simd/provider/{operation_manifest,routes_generated}.h`：由账本机械生成的 source-only 合同；
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

Highway 是同合同标准 primitive 的默认实现。以下情况直接选 Highway并停止测量：

- 最终汇编相同；
- 合格批次持平或差异小于 5%；
- 执行顺序、地址位置、编译器或相邻批次反转；
- 冻结的有限资格化仍不能分出 material winner。

xsimd/HYLE 只有在语义、复杂度和适用的 type × target × compiler scope 都明确，且 current evidence 显示
material 优势时才能成为例外。例外不能以 buffer size、benchmark title、地址或采样顺序为 route 维度。

当前 operation 账本已终态；它是旧实现迁移与未来重签索引，不是普通用户 API。`RouteFor` primary 保持
`PENDING`，生成器只为已签 operation/family 建特化。`REMOVE_REDUNDANT` 的 route 不可调用；
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
4. P0 先比较语义、复杂度、遍历数、allocation/dispatch；P1 才跑同合同真实 size workload。
5. route 只能在机器账本中落为 `ROUTE_HIGHWAY`、`ROUTE_XSIMD`、`KEEP_HYLE` 或
   `REMOVE_REDUNDANT`；不得留下隐含 fallback。
6. target-bound correctness、最终 codegen 和普通 consumer 验证通过后再接入 family；不要先改公共 layout。

没有真实 consumer、可复现缺陷、标准或成熟 peer gap 时，不新增能力。未来推理引擎提出 tensor layout、
block quant、GEMM、attention、RoPE 或 fusion 需求时，先由 Compute/engine 拥有其 tensor/operator contract，
SIMD 只接收可复用的连续数据或 target-kernel primitive。

## 怎样升级 Highway/xsimd

1. 在中央 vendor lock 更新官方 commit；benchmark checkout 仍只是 peer，不能代替产品依赖。
2. 重算 integrated source hash 和 notices/license 投影。
3. 用 route ledger 找出受上游源码/target 影响的 implementation groups，只重签这些组与正常 consumer
   sentinels；不要重跑 387 × type × size × ISA 的开放笛卡尔积。
4. 对 HYLE 例外重新确认其 material 优势。优势消失就路由回 Highway并删除 body；出现新的显著优势则新增
   一个有限 candidate，不复制整个 vendor surface。
5. 对 VQSort adaptation 移植相关 upstream 修复，核对 origin、local delta 和内容 hash。
6. 完成三平台与 source/static/shared 回归后才更新 candidate/terminal record。

## 账本与生成物

迁移完成后，旧源码不存在是预期状态。只用冻结账本验证，不再从已删除文件重建 inventory：

```text
python scripts/simd-op-coverage.py --check-ledger \
  --require-route-decisions --check-provider-routes
```

修改终态账本后机械生成 route：

```text
python scripts/simd-op-coverage.py --write-provider-routes
python scripts/simd-op-coverage.py --check-ledger \
  --require-route-decisions --check-provider-routes
```

`--check` 与 `--reports` 是旧 register-source coverage 模式；物理删除后会明确失败，不能用于当前产品
verdict。基线 definition site 仍以 ledger 中的 commit/digest 定位，不得把缺少当前文件误报成 inventory
丢失。

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
明确封顶 AVX2。clang-cl/GCC/Clang 可按各自资格化 target 构建。canonical x86 tier 不隐含 AES/PCLMUL；
专用 crypto kernel 必须有独立 feature 合同。

## 最小验证

每次 route/body/consumer 变更至少运行：

```text
python scripts/simd-op-coverage.py --check-ledger \
  --require-route-decisions --check-provider-routes
cmake --build <build> --target hyle_simd_tests
ctest --test-dir <build> -L simd --output-on-failure
python scripts/check-module-closure.py --previous <round-snapshot>
python scripts/check-docs.py
```

其中 SIMD suite 必须覆盖 ordinary API、route static assertions、有限 winner correctness、provider codegen
probe 与当前 ordering target。涉及 target/build seam 时还要验证：

- Windows MSVC、WSL GCC、macOS AppleClang 的适用 real target；
- source embedding、relocated static SDK、relocated shared SDK；
- exact/fat target；
- 受影响的 direct consumer；
- WSL ASan/UBSan（只有共享可变状态时才需要 TSan）。

性能只运行 closure card 冻结的 ordinary/inference whole-buffer groups。微观 route 已有终态证据时不因一次
小波动重开；新结果只有在合同合格、超过 material margin、正反顺序和机制检查一致时才可改变 route。

## 交付与负空间

installed static/shared SDK 只安装 ordinary SIMD 头和 HYLE library，不安装 vendor register headers、
`target.h`、`fat_kernel.h` 或 `provider/*`。完整源码交付才有 expert seam。

SIMD 不提供 C API、Native FFI、稳定跨工具链 C++ ABI、per-ISA DLL、公开 register façade、GPU backend 或
模型 operator。不要为不存在的跨语言消费者导出伪 C ABI；未来有明确 runtime/engine FFI 时在其 owner
边界做类型擦除，而不是暴露模板或 target-dependent vector layout。
