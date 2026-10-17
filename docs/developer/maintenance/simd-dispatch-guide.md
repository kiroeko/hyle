# SIMD 路由维护

用户入口见[SIMD 使用](../../user/simd-usage.md)，这里解释修改实现和路由需要维护的少数事实。
普通 API 的向量循环由一个 provider 完整实现；CPU/OS facts 由 Platform 提供，不另造探测器。

## 先区分两个边界

| 路径 | 入口与选择 | 类型边界 |
|---|---|---|
| 普通整批调用 | `hyle/simd/*.h`；未知 CPU 使用缓存的 target 表，exact build 直接调用 | vendor vector/mask 留在实现 TU |
| 显式专家 kernel | `hyle::simd_expert`、`hyle_target_simd()` 或 `hyle_add_fat_kernel()` | target-bound TU 内可用锁定 provider，entry 是 pointer/count/POD |

不要把 provider switch、CPU probe、第二层动态 dispatch 或逐 lane callback 放进普通数据循环。
确需 adapter 时检查最终指令与实际成本，不从“薄封装”这个名字推断零开销。
`FlushStream()` 只完成当前 core 的 non-temporal-store lifecycle，不替代跨线程发布同步。

## 修改哪一份事实

| 内容 | 所属位置 |
|---|---|
| 普通 family target body | `src/simd/**/kernel_impl.h` |
| 数学路线 | `src/simd/math/highway_math_kernel-inl.h` |
| VQSort 移植与局部适配 | `src/simd/sort/vqsort_hyle/` |
| 有限的 HYLE 专用 body | `include/hyle/simd/provider/hyle_winners*-inl.h` |
| 操作与 provider 选择 | `scripts/simd-op-routing.json` |
| whole-buffer family 与编译器/ISA 例外 | `scripts/simd-family-routing.json` |
| 生成头 | `include/hyle/simd/provider/{operation_manifest,routes_generated}.h` |
| 实现的依赖版本与来源 | `cmake/HyleDependencyLock.json` |

配置修改后生成并检查，不手改生成头：

```text
python scripts/simd-op-coverage.py --write-provider-manifest --write-provider-routes
python scripts/simd-op-coverage.py --check
```

`ROUTE_HIGHWAY`、`ROUTE_XSIMD`、`KEEP_HYLE` 和 `REMOVE_REDUNDANT` 是实际路由语义，
不是开发状态。每项有完整默认路线，编译器/ISA selector 不重叠；删除的 route 不留隐式 fallback。
配置检查只验证输入与生成物一致，不替代数值或性能测量。

## 新能力与依赖升级

先明确真实任务的输入/输出、alias、tail、数值边界、复杂度和临时空间。普通连续数据需求优先进入
整批 API；只服务一个 fused kernel 的寄存器 spelling 留在专家 TU。复用成熟 provider，
真正缺少合同或存在实际热点时再增加有限专用实现，不复制整套 vendor surface。

provider 例外需要可解释的语义和适用编译器/ISA；相同工作量下没有可复现收益，就保留更简单实现。
不按 buffer 地址、benchmark 名称或采样顺序选择产品路线。涉及 tensor layout、量化或模型算子的
请求由 Compute/引擎拥有合同，SIMD 只承接其中可复用的连续数据或 target primitive。

升级依赖时更新中央 lock 与相应许可/来源身份，检查受影响操作、target 和普通消费者；
VQSort 的原算法修复与 HYLE 局部差异分别核对。旧专用实现的收益消失后可删除，不为保留历史决策继续绕路。

## 编译与验证

固定 TU 与 source-fat kernel 的接口分别为：

```cmake
hyle_target_simd(TARGET my_kernel ISA AVX2)
hyle_add_fat_kernel(
  TARGET my_app NAME my_kernel IMPL "src/my_kernel-inl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
```

entry 的函数类型在所有 target 中一致；target-dependent vector layout 不跨共享 ABI。
当前锁定 Highway 对经典 MSVC 的 AVX-512 有限制：exact/fixed 请求在配置时拒绝，fat build 封顶 AVX2；
不能仅凭 CPU 支持就推断工具链可生成该路线。

使用已启用测试的相关初始化配置，例如：

```text
python scripts/build.py --target hyle_simd_tests
python scripts/test.py --filter simd
```

按改动选择普通 API、route、数值边界、target、codegen 与直接消费者检查。
变更编译或交付边界时再检查对应的 exact/fat、源码嵌入或安装 SDK；需要时在实际 ISA/系统上运行。
不因修改一条路线自动扩大全平台验证。性能分析见[测量方法](../performance/README.md)，
已知数值与仪器陷阱见[排障](../debugging/README.md)。
