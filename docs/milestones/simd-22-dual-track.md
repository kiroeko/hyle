# simd-22：native/runtime 双轨重构

> 状态：C++ 双轨重构完成（2026-07-16）。纯 C ABI shim 仍按 close-01/02 独立收口。

## 目标

同一份 D-tag kernel body 同时服务：

- 已知部署 ISA 的源码用户：`hyle::simd::native`，零 CPUID、零函数指针、可内联；
- 未知部署 CPU 的预编译库用户：target-neutral batch API + `runtime()` 函数表；
- close-01：在二者之外另封纯 C shim，不把 C++ 类型误当稳定 ABI。

不把 381 个寄存器 op 逐个动态化。范围与用户决策树见
[`../architecture/simd-dispatch-guide.md`](../architecture/simd-dispatch-guide.md)。

## 总体架构图

```text
                         semantic whole-buffer kernel
                  add / random_fill / scan / dtype / math / ...
                                      |
             +------------------------+---------------------------------+
             |                        |                                 |
     A. unknown customer CPU   B. fixed HYLE library           C. fixed consumer TU
        prebuilt library          exact deployment             source + exact ISA
             |                        |                                 |
      ordinary batch API       ordinary batch API              native:: API
      pointer/POD/opaque       same public signature           hyle_target_simd(...)
             |                        |                                 |
      baseline dispatcher      compile-time #if direct                  |
      (no wide instructions)   HYLE_SIMD_EXACT_TARGET_*                 |
             |                        |                                 |
      +------+------------------------+                                 |
      |                                                                 |
      | first use only                                                  |
      +-> cpu_features()/best_isa()                                     |
          CPUID+XGETBV / HWCAP                                          |
                 |                                                      |
          immutable Runtime                                             |
          +-------------------------------+                             |
          | ops/random: function table    |                             |
          | scan: cached resolver         |                             |
          | other modules: ISA/features   |                             |
          +---------------+---------------+                             |
                          |                                             |
                          v                                             |
       per-ISA stable entry / thin wrapper                              |
       *_scalar / *_sse4 / *_avx2 / *_avx512 / *_neon / *_sve / *_rvv   |
                          |                                             |
                          +----------------------+----------------------+
                                                 |
                                  one shared native D-tag body
                                  native/ops.h · random.h · scan.h
                                                 |
                                               vec.h
                                      target-neutral register ops
                                                 |
                                  compile-time selected backend
                       scalar / SSE4 / AVX2 / AVX512 / NEON / SVE / RVV

     custom whole-buffer kernel
              |
     hyle_add_fat_kernel  --generate--> per-ISA stable entries
              |
     select_fat_kernel    --dynamic: select/cache once
                           --exact: compile-time select

     close-01/02 C ABI
     opaque handle + POD + version/size vtable  ---> wraps only ordinary batch API
```

图里只有三种合法边界：

1. `vec.h` 和 backend 之间传递 target-dependent vector/mask；它必须在编译期固定 ISA。
2. variant 与 dispatcher 之间只传 pointer、标量、长度、POD/opaque state；它才允许运行期派发。
3. C ABI 只包第二种边界；`Runtime`、D-tag、C++ 类布局和 STL 都不能穿过 C ABI。

## 动态调用到底怎么走

以 `add(x, y, out, n)` 为例：

```text
用户 -> add() baseline dispatcher
     -> runtime()                         // function-local static，只初始化一次
        -> cpu_features()/best_isa()      // CPUID+XGETBV 或 HWCAP，只在初始化路径
        -> 选中整张不可变 Runtime table
     -> rt.ops.add_f32(...)               // 每个 whole-buffer 调用一次间接调用
     -> detail::add_f32_avx2(...)         // 例：当前机器选中 AVX2
     -> native::add(...) 的共享 D-tag body
     -> vec.h -> backend/avx2.h
```

这里没有 lane 级或 381-op 级动态判断。CPU 探测和建表只发生一次；ops/random 在每个完整 batch
边界有一次函数指针调用。scan 缓存解析后的函数指针；dtype/math/sort/bitpack 读取同一张 Runtime 的
module ISA/features，保留其 per-op winner、F16C/BF16 等非 canonical feature 和算法阈值。

`runtime_for(Isa)` 是显式取某张当前 CPU/OS 确实支持的表，主要用于测试、限档和外层策略；它不是
强行执行不支持指令的接口，也不是 fixed-target 接口。

## 固定式调用到底怎么走

固定式调用有两级，二者都不做 CPUID，但目的不同：

### 整库 exact：保留普通 API

```text
-DHYLE_SIMD_EXACT_TARGET=AVX2
  -> hyle_exact_simd_library 给 hyle_simd dispatcher 定义 HYLE_SIMD_EXACT_TARGET_AVX2
  -> add()/random_fill()/scan()/dtype/math/sort/bitpack 的 #if 分支直接引用 *_avx2
  -> dispatch object 不引用 runtime()
  -> variant 薄包装 -> 同一 native body -> AVX2 backend
```

它适合容器镜像、游戏主机、设备固件或已规定最低 ISA 的服务部署。普通 API 名字不变，整个 HYLE
库要求运行机器满足该 ISA。AVX512 只代表 canonical AVX-512 group，不擅自假定可选 BF16/AVX10.2。

### 用户 TU exact：走 `native::` 并允许内联

```text
hyle_target_simd(TARGET my_kernel ISA AVX2)
  -> 给 my_kernel 同时加 /arch:AVX2 或 -mavx2 与 HYLE_SIMD_TARGET_AVX2
  -> vec.h 在预处理/模板实例化时只看到 AVX2 backend
  -> native::add/native::random_fill/native::scan 在用户 TU 内实例化并内联
  -> 不生成 Runtime 引用，也不经过稳定符号薄包装
```

常被口语称为“偏特化固定腿”，但要准确理解：**公共 batch API 并没有靠 C++ 模板偏特化派发**。
固定选择由 CMake target 属性和 `HYLE_SIMD_TARGET_*` 宏完成；D-tag 模板只负责把共享 kernel body
实例化成当前 backend 的具体 vector/mask 类型。这样不会让不同 lane 数或 sizeless SVE/RVV 类型泄漏
到稳定函数签名。

## 一份 kernel 如何同时服务两条腿

以 ops 为模板，所有适合双轨的新 whole-buffer kernel 应保持这个依赖方向：

| 层 | ops 示例 | 维护职责 |
|---|---|---|
| 公共动态 API | `include/hyle/simd/ops.h` | 只声明稳定 pointer/POD 签名 |
| 公共固定 API/body | `include/hyle/simd/native/ops.h` | 唯一算法 body；用 D-tag/`vec.h` 写循环与 tail |
| variant wrapper | `src/simd/ops/kernel_impl.h` | 只把 native body 导出成 `detail::*_scalar/avx2/...` 稳定符号 |
| variant declarations | `src/simd/ops/kernels.h` | 声明 dispatcher 能引用的各 ISA 符号 |
| 动态/exact dispatcher | `src/simd/ops/dispatch_ops.cc` | dynamic 取 Runtime 表；exact `#if` 直连目标符号 |
| 机械构建 | `src/simd/CMakeLists.txt` + `HyleDispatch.cmake` | 从同一 impl header 生成并施加每 ISA codegen flag |
| Runtime 元数据 | `runtime.h` + `runtime.cc` | 整表版本、函数子表、模块真实 ISA/features 与降级关系 |
| 验证 | `runtime_test.cc` + `native_test.cc` + 模块测试 | dynamic、exact、跨 ISA oracle、符号无 Runtime 引用、多 VL |

维护者新增 kernel 时按以下顺序判断：

1. 返回值或参数若含 vector/mask，只加到 `vec.h`/backend/native，不进 Runtime。
2. 是 pointer/POD whole-buffer 语义且有多条有价值实现，先写一份 native body，再加薄 variant 和动态入口。
3. 只有一个获胜实现时，Runtime module metadata 可以诚实指向 scalar/NEON 等现有核，不为“覆盖率”伪造宽腿。
4. exact dispatcher 必须直接引用目标符号；用对象符号检查确保没有意外 `runtime()`/`cpu_features()` 引用。
5. SVE/RVV 要用至少三种 VL/VLEN 运行；不得把 sizeless vector 存进对象或跨稳定边界。
6. 新增/调整 `Runtime` 字段必须递增 `kTableVersion`，并同步 `table_size` 测试；未来 C vtable 独立版本化。

## 阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| D0 | C++ `Runtime` table-version/size/ISA + ops/random 子表；自动和精确 tier 查询 | 已完成 |
| D1 | `native/ops.h` 作为唯一 body；原 variant TU 降为薄符号包装 | 已完成 |
| D2 | `RandomState` 不透明固定逻辑流；runtime/native fill；VectorXoshiro 归 native | 已完成 |
| D3 | `hyle_target_simd(TARGET ... ISA ...)` 固定 target helper | 已完成 |
| D4 | scan 迁移共享 native body；移除全部独立 active-tier API，统一 module metadata | 已完成 |
| D5 | dtype/math 的特殊 feature 与 per-op winner 纳入共享 Runtime；exact build 编译期直连 | 已完成 |
| D6 | sort/bitpack 保留算法分流，runtime 诊断统一；exact build 编译期直连 | 已完成 |
| D7 | 公共 `hyle_add_fat_kernel` + `select_fat_kernel`，稳定 entry contract | 已完成 |
| D8 | Linux SVE/RVV HWCAP、Isa、构建和 ops/random/scan runtime 变体 | 已完成 |
| D9 | close-01 纯 C `hyle_simd_api_t`/opaque state 薄封装 | 等 close-01 |

## 首个纵切验收

1. 普通 `dot/add/saxpy/random_fill` 经 `runtime()` 自动选择；函数表只解析一次。
2. `native::dot/add/saxpy/random_fill` 不引用 runtime，分别编成 scalar/SSE4/AVX2/AVX-512。
3. RNG 相同 seed 在 active runtime、scalar runtime 与 native target 输出逐位相同；任意 fill 分块不改变序列。
4. `RandomState` 对 C++ 用户不暴露布局；未来 C shim 用 opaque object，不冻结 C++ ABI。
