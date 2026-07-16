# simd-24：基础 SIMD 能力补全

> 状态：🚧 进行中（阶段 1 vqsort 正确性/ASan 已收口；NEON KV64 残余落后和 exact 单腿产物缺口已显式记录）。依赖 simd-22/23。
> 本阶段不再以“等待消费者”为缺口延期理由：HYLE 是基础库，已确认有通用价值的能力按顺序补齐。

## 目标与边界

本里程碑收口 `simd-gap-inventory.md` 剩余的六组能力：

1. vqsort 向量 partition；
2. 连续数组 search/reduction；
3. mixed-precision dot；
4. register `mul_hi/mul_hilo`；
5. sort key 类型扩展；
6. complex 家族与 f64 complex。

每个 whole-buffer 能力都遵守 simd-22 双轨：未知 CPU 的普通入口经一次 Runtime 函数表选择；整库
exact build 编译期直连；源码级专家用 `native::`/D-tag 内联。close-01 再把普通入口包成 pointer/count/POD
的纯 C ABI，不发布 STL 容器外观。寄存器 op 只属于 fixed/native 轨，不机械制造数百个动态函数指针。

```text
                 close-01 C ABI (pointer / count / POD)
                              |
                  whole-buffer semantic core
                 /             |              \
          Runtime table    exact direct     native source TU
         (unknown CPU)     (known deploy)   (expert inline)
                 \             |              /
                    shared per-ISA kernel body
                              |
                vec.h register DSL + backends
                              |
       search/dot/sort/complex     mul_hi/mul_hilo primitives
```

## 阶段 1：vqsort partition（已实现）

### 根因

simd-20 端口把 `HYLE_VQSORT_USE_VECTOR_PARTITION` 默认置 0，所有大数组递归退回 HeapSort。该保险丝
与同一提交一起进入仓库，没有独立复现证据。重新启用后先暴露出四个私有 128-bit upper helper 与
后来公开 op 重名；删除重复实现后可编译。

真正的性能缺口在 AVX2 64-bit `Compress`：HYLE 仍用 store + scalar pack + reload，Highway 用
`vpermd` 把每个 u64 当作相邻 u32 对搬运，并在一条结果向量中形成左右 partition。KV64 因此从
Highway 的单 permute/双 store 退化成两次 scalar compress。

### 修复

- 删除 `HYLE_VQSORT_USE_VECTOR_PARTITION` 及 HeapSort 保险分支，递归 partition 成为唯一产品路径；
- 删除 vqsort 私有重复的 `Eq/Ne/Min/Max128Upper`；
- AVX2 增加覆盖全部 16 个 u64 mask 的 constexpr pair-index table，`Compress/CompressNot` 使用
  `vpermd`，返回稳定真 partition；快路只对满宽 u64x4 生效，capped u64 tag 保留窄后端/generic
  语义并增加独立回归测试；
- vqsort 的 AVX2 64-bit/KV 路径恢复单次 `CompressKeys` + 两次整向量 store；其他宽度和 ISA 继续
  使用 exact-count store，不能依赖未定义 tail；
- NEON 16/32-bit table compact 补齐 inactive tail，和原有 64-bit 路径一起形成真 partition；vqsort
  因而同样使用单次 table lookup + 两次整向量 store；
- 新增相邻活对象 canary，覆盖 sort/partial/select、升降序、随机/重复值和边界长度；分配边界由
  WSL ASan 再验证。

### Windows 证据（2026-07-17）

- 全类型 `Sort.*` 20 轮：17 × 20 全绿；额外 U32/KV/canary 100 轮全绿；
- AVX2 `CompressI64` 对全部 16 个 mask 的现有 exhaustive oracle 全绿；
- MSVC AVX2/AVX512 vqsort TU 均从干净 object 强制重编译通过；
- Mac 首次启用编译抓到 `HWY_COMPILER_CLANG` 被错误布尔化（上游按版本比较）的隐藏 port bug，已改为
  `major*100+minor`；这是保险丝过去遮住的跨平台编译路径；
- `n=65536,k=32768,random,asc` 的 nanobench smoke（ns/key，越低越好）：

| 类型/操作 | 修复前 HYLE | 修复后 HYLE | Highway | 结论 |
|---|---:|---:|---:|---|
| KV64 partial | 90.83 | 20.66 | 72.83 | 根因修复，HYLE 约 3.5x |
| U64 partial | — | 6.03 | 72.20 | HYLE 约 12.0x |
| U64 select | — | 0.68 | 35.20 | 需在认证 benchmark 中复核绝对值，正确性已过 |
| U32 partial | 39.45（旧 HeapSort smoke） | 10.08 | 62.73 | partition 恢复 |

这些是定点 smoke，不代替 simd-op 方法论的多轮认证；但足以证明 KV64 回退的代码形状和修复收益。

### WSL ASan 证据（2026-07-17）

- GCC 14、AVX2、`-O0 -g -fsanitize=address -fno-omit-frame-pointer`；
- 直接把 ASan 编译的 `vqsort_hyle_avx2.cc.o` 与审计 harness 链接，避免 exact AVX2 构建错误携带且
  实际不会执行的 AVX512 TU；
- 20 轮覆盖 17 个边界/大尺寸、升降序、随机/重复值、sort/partial/select 和左右活对象 canary；
- 另以 KV64 覆盖 sort/partial/select 及 16-byte 对象 canary；`halt_on_error=1` 全绿。

最初 fat/exact-library ASan 编译并非测试失败，而是 AVX2 与 AVX512 两个巨型模板 TU 并行时各占约
2 GiB，被 WSL 总内存上限终止；改串行后 AVX2 object 正常产出。该现象也是下文 exact 产物缺口的
可观测成本证据。

### Apple Silicon 证据（2026-07-17）

- AppleClang 21 从当前 Windows 权威工作树重编译 `vqsort_hyle_neon.cc`；
- NEON vec/native/partial-boundary/half-oracle：279/279；
- Runtime `Sort.*:Math.*`：41/41，其中 sort 17/17，新增 canary 真走 NEON partition；
- 首轮编译发现并修复 `HWY_COMPILER_CLANG` 版本宏布尔化问题，证明不是 stale binary 假绿。

同一 Apple Silicon、同一输入的 direct/core/public 拆分证明动态派发不是 KV64 落后的来源：

| `n=65536,k=32768,random,asc` | HYLE core | HYLE public | Highway | 结论 |
|---|---:|---:|---:|---|
| KV32 partial（ns/key） | 约 11.22 | 约 11.22 | 10.92 | 基本持平 |
| KV64 partial（ns/key） | 26.92 | 26.75 | 21.02 | **HYLE 真 LOSE，约慢 27%** |

KV64 全排序也约为 45.26 vs 36.81 ns/key；小规模和重复值输入仍有约 5–26% 的稳定差距。新增的
NEON u64x2 `LoadN/StoreN` lane 路径消除了栈临时数组和 stack-canary，但没有消除差距。单 op 对拍中的
`Lt128/Lt128Upper/CountTrue/IfThenElse` 基本持平，因此当前证据只把问题缩小到 128-bit key sorting
network/组合 codegen，不能武断归因给某一个 primitive。该项继续作为性能狩猎任务，不以阶段正确性
完成或统计口径掩盖。

## 临时/测试宏审计

只记录会改变算法、能力或产物的宏；ISA target/include guard 不列为异常。

| 项 | 性质 | 处理 |
|---|---|---|
| `HYLE_VQSORT_USE_VECTOR_PARTITION` | 事故保险丝，生产可退回 HeapSort | **删除** |
| math vendor port 的 7 个 `#if 1` | 移植残留，不改变当前语义但伪装成能力门 | **删除** |
| `HYLE_TESTING` | 由 `HYLE_BUILD_TESTS` 派生的正规 test-only ABI gate | 保留；SIMD 生产算法未受它切换 |
| `HYLE_SIMD_*_HAS_NATIVE_*` | 后端 capability selector | 保留；表达 codegen 能力，不是测试开关 |
| `HYLE_SIMD_PREFER_VECTOR_DIVISOR_SIGN` | NEON divisor codegen 策略 | 保留，后续可改 constexpr trait |
| `VQSORT_PRINT` | 私有编译期诊断级别 | 暂留在 port 内，不进入公开 build contract |
| `VQSORT_ENABLED` / `HWY_IDE` / 产品内 `VQSORT_ONLY_STATIC` | Highway port 兼容残留；曾可改变产品代码形状 | **删除**；benchmark 中同名宏只配置外部 Highway 静态入口 |
| `HWY_MEM_OPS_MIGHT_FAULT` | 决定尾部加载是否必须 fault-safe | 保留为内部移植 shim；值固定为安全侧 |
| `HWY_HAVE_FLOAT64/16` | 当前 sort 类型能力门 | 随阶段 5 类型扩展改为 HYLE 类型 trait |

扫描未发现 `HYLE_TESTING`、`DEBUG`、`TEMP` 一类宏切换 SIMD 产品算法。`HWY_HEADER_ONLY` 只存在于
对照 benchmark，用来配置参考库；`HYLE_SIMD_*_HAS_NATIVE_*` 是 backend 向 `vec.h` 报告 codegen
能力的内部选择器，不是用户开关。

## exact 单腿产物的已知 mechanization 缺口

本轮为缩短 WSL sanitizer 构建曾尝试让 `HYLE_SIMD_EXACT_TARGET=AVX2` 只生成/编译 AVX2 TU。链接随即
暴露：当前 exact 模式已把运行时选择编译期绑定到指定 ISA，但多个 dispatch/runtime 表仍统一引用
scalar/SSE4/AVX2/AVX512 的符号集合，手写 sort/dtype/math TU 列表也仍按 fat library 组织。因此它是
“固定调用腿”，还不是真正只含一套 kernel 的最小单腿二进制。

该实验性 CMake 过滤没有提交，因为只删 TU 会制造未解析符号。完整修复必须由同一份 target manifest
同时生成 source list、declaration、dispatch table 和 exact binding，再验证每个 exact ISA 的链接符号及
无越级指令。它不影响未知 CPU 的 fat runtime 正确性，也不在热循环引入额外判断，但属于 simd-22
双轨产物合同尚未彻底机械化的明确欠账。

## 后续实施顺序

### 阶段 2：search/reduction

- 首批语义：typed `count/find/find_if-equal/min/max/minmax/is_sorted`；空输入、not-found index、NaN 与
  升降序合同先冻结；
- scalar oracle + SSE4/AVX2/AVX512/NEON/SVE/RVV kernel，共用 pointer/count body；
- Runtime/exact/native 三条路径和 std/Highway benchmark 同时落地；
- close-01 只需薄 C shim，不再重写算法。

### 阶段 3：mixed-precision dot

- 顺序：bf16×bf16→f32 accumulate、f32×bf16、i16×i16→i64/明确溢出合同、f16×f16；
- 禁止先转换完整数组再 dot 的额外内存 pass；tail 与 alias 走同一 oracle；
- x86 F16C/BF16/VNNI、Arm FP16/BF16/DotProd 只有硬件存在时走原生，否则 widening kernel。

### 阶段 4：`mul_hi/mul_hilo`

- 统一现有 `MulHigh`/`MulHigh32` 命名和 lane 合同，补 i/u8、i/u32、i/u64；
- `mul_hilo` 共享乘法 partial，不实现成两个独立 multiply；
- register op 只走 fixed/native 轨；用全位宽 oracle、边界值、汇编和 Highway/xsimd parity 验收。

### 阶段 5：sort key 扩展

- i16/u16、128-bit POD key 先落；f16/bf16 在统一 dtype 与 NaN total-order 后落；
- 每种类型都覆盖 sort/partial/select、KV 携带、canary/ASan 与规模阶梯；
- 与 radix crossover 分开测，不用 radix 的胜利遮住 vqsort 落后。

### 阶段 6：complex/f64 complex

- 补 f32 `ctan/csinh/ccosh`，再把现有 `cexp/clog/csqrt/csin/ccos/ctanh/cpow` 扩为 f64；
- 继续使用 split real/imag pointer C ABI 形状，避免固定某语言的 complex ABI；
- 特殊值、极点附近、alias/tail 与 libm/`std::complex` 对拍；性能对照 xsimd。

## 每阶段验收

- Windows MSVC、WSL GCC/Clang、Apple Silicon NEON 的正确性和适用 sanitizer；
- AVX512 至少编译闸，能用 SDE 时补执行；SVE/RVV 用现有 runner/QEMU 矩阵；
- 同合同、同 ISA、同复杂度与 Highway/xsimd/std 对照，稳定 LOSE 不能用统计口径隐藏；
- 普通入口不泄漏 vector/mask/STL 类型，fixed/native 不承担不必要的 Runtime 判断。
