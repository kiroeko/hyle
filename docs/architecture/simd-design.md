# HYLE SIMD 设计

本文只描述 SIMD 的长期产品边界、ownership 和调用成本。当前施工状态以
[`module-status.json`](../implementation/module-status.json) 为准，当前轮及历史验证事实见
[`SIMD/Math epic`](../implementation/tasks/simd/simd-epic-v1.md)。用户可依赖的接口和陷阱见
[`SIMD 使用指南`](../user/simd-usage.md)。旧实现、实验点值和阶段结论只保留在 implementation/evidence，
不得从历史任务反推当前承诺。

SIMD 当前是 C++23 source/static/shared 组件。它不提供 Native FFI，也不承诺跨任意 C++ 工具链 ABI。

## 产品目标

HYLE SIMD 同时服务推理引擎和普通连续数据消费者：

- ordinary 用户获得按语义组织的 pointer/count whole-buffer API，不需要理解 ISA 或寄存器类型；
- 源码 consumer 可以为固定 ISA 或多版本 kernel 直接使用项目锁定的 Highway，保持模板单态化和内联；
- HYLE 拥有 dtype、rounding、alias、layout、错误、选择和交付合同；成熟 vendor 拥有其已经正确解决的
  register/target/math 机制；
- 性能优化以真实 consumer、复杂度或同合同 peer 为输入，不按 benchmark 名字复制 API；
- tensor shape、stride、broadcast、block quant、operator fusion、模型格式和 GPU policy 属于 Compute 或
  更高层，不下沉进 SIMD。

## 分层

```text
ordinary C++ pointer/count API
              |
      private family dispatcher  <--- Platform CPU/OS facts
              |
     generated per-target entry
              |
  direct Highway body / minimal HYLE semantic gap

source expert TU
      |
hyle_target_simd() or hyle_add_fat_kernel()
      |
direct locked Highway template
```

第三方 vector/mask/tag 类型只存在于 target-bound implementation TU。跨 TU、静态/动态库和用户公开边界
只传 pointer、count、标量 POD 或 HYLE 自有稳定 storage type。这样可以独立升级 Highway，也不会把
target-dependent lane 数或 vendor namespace 固化为 HYLE ABI。

### 普通 whole-buffer 路径

未知 CPU 的预编译库在初始化时由 Platform facts、已编译 target 和 family policy 共同构造一张私有不可变
函数表。每个 whole-buffer 调用最多经过一次可预测间接调用；元素循环内禁止再次探测、callback、第二层
`HWY_DYNAMIC_DISPATCH`、allocation 或 buffer copy。`selected_target(family)` 只报告实际产品选择，不用于
用户自行派发。

### 固定和源码路径

- `HYLE_SIMD_EXACT_TARGET` 让整个 HYLE SIMD artifact 只包含一个目标，保留相同 ordinary API。
- `hyle_target_simd()` 为一个源码 consumer TU 提供中央锁定 Highway 头、静态 target 宏和准确 ISA flags；
  它不探测 CPU，也不经过 HYLE 函数表。
- `hyle_add_fat_kernel()` 从同一个 direct-Highway implementation header 生成本平台的 per-ISA TU，只在
  完整 pointer/POD kernel 入口选择并缓存一次。

fixed source 的热循环应与同 TU 原样 Highway 等价。source-fat 与 ordinary 路径只承担一次 batch-boundary
选择。tiny/fused kernel 若无法摊销一次库调用，应由源码 consumer 融合，而不是重新暴露一套 HYLE register
DSL。

完整构建合同见 [`SIMD dispatch 维护指南`](../developer/maintenance/simd-dispatch-guide.md)。

## Target 与探测 ownership

Platform 是 CPU/OS facts 的唯一 owner；SIMD 不复制或转发其 detector。Highway 拥有实现 target、编译器
blocklist 和寄存器 lowering。SIMD family 选择二者与产品 policy 的交集，所以主机最高 ISA 与某个 family
实际 target 可以不同。

支持矩阵以 scalar 为兜底，包括 x86 SSE4/AVX2/AVX-512、AArch64 NEON/SVE 和 RVV；VNNI、BF16、F16C、
AES 等正交 feature 只由需要它的 specialty kernel 单独探测。canonical x86 tier 不偷偷提高 AES/PCLMUL
最低要求。

锁定 Highway 在经典 `cl.exe` 下禁用 AVX3 family：fixed/exact AVX-512 必须 fail-fast；unknown-CPU
source-fat 明确封顶 AVX2。需要 direct Highway AVX-512 时使用 clang-cl。静默把 AVX2 产物称为 exact
AVX-512 不受允许。

## 能力 ownership

| family | HYLE 拥有的公开合同 | 不属于本层 |
|---|---|---|
| row ops | 连续 f32/f64 算术、归约、SAXPY；f16/bf16/i8/i16 mixed/storage dot | stride、broadcast、tensor fusion |
| dtype | f32/f16/bf16/E4M3/E5M2 storage、转换、RNE/饱和/NaN/alias | block quant scale、tensor layout |
| narrow | f16/bf16 数值/bit 搜索、f32 sum、IEEE total-order | 平行 register DSL、tensor narrow policy |
| math/activation | whole-buffer 数学 accuracy/domain/alias 与激活语义 | 模型 operator graph |
| search/scan | 连续数组的查找、归约、有序性和前缀计算 | container index ownership |
| bitpack/random | 明确 layout 的内存压缩；可复现非加密批量 RNG | 文件/网络 wire format、密码随机数 |
| ordering | sort/select/partial/radix/argsort 的 key、NaN 和稳定性合同 | distributed/external sort |

当前函数清单只以公开头和用户指南为准；本表划分 owner，不代替 capability closure 证据。

### Row 合同

普通 f32/f64 row operation 是 O(n)、O(1) extra storage，不分配、不加锁。输出允许与一个完整输入精确同址；
任意部分重叠不受支持。浮点 reduction 的结合顺序可随 target 变化，FMA operation 允许最后舍入不同。
i8/i16 dot 的乘积精确，累加按模 `2^64` 定义后以二补码 i64 返回，不依赖 signed-overflow UB。

### Dtype 与 narrow 合同

`convert(src,dst,n)` 覆盖五种公开 storage 间全部 20 条异类型有向边；同类型复制由标准库承担。
相同宽度允许 exact alias，不同宽度要求 disjoint，任意部分重叠不受支持。转换是 O(n)、O(1)，不能通过
建立整段 f32 临时数组实现 storage-to-storage。

Highway 直接承担可表达的 f16/bf16 promotion/demotion 和 target lowering。HYLE 只修补已经确认的上游
语义差异，并保留 OCP fp8、整数 modulo dot、精确 feature specialization 等具名最小实现。ordinary
`narrow_ops.h` 保留用户需要的整段语义；源码寄存器 consumer 直接使用 Highway，并自行声明 numeric
contract。

## Vendor 与升级边界

锁定 Highway 是 header-only product dependency，不形成第二个 runtime ABI。xsimd 当前只作独立 accuracy/
performance peer；只有 Highway 无法满足的具名 capability、accuracy 或真实 consumer 性能缺口才能进入
产品，不能为“同时支持两个 vendor”在热循环增加 runtime abstraction。

checked-in Highway-derived port 是待裁决债务，不因改 namespace 就变成自实现。direct wrapper 可行即删除；
必须保留时只保留真实 contract delta，并登记 origin、local delta、许可证和内容 hash。完整规则见
[`SIMD vendor ownership`](simd-vendor-surface.md)。

## 依赖与交付

SIMD 的 HYLE 依赖只有 Core 和 Platform。Container 可通过 build-private seam 复用尚未迁移的 header
implementation，但这不是公开反向依赖。Compute 可以消费 ordinary SIMD API；SIMD 不依赖 Compute。

source、installed static 和 installed shared 三种交付必须分别验证。installed SDK 不交付第三方 register
headers；direct vendor expert seam 只属于完整源码交付。符号检查只允许 HYLE ordinary API 与小型诊断面，
不得导出私有 Runtime 表或 vendor 类型。

## 性能与验证原则

- P0 先检查 O(n) 下界、遍历次数、临时 storage、dispatch 次数和可选 feature gate。
- P1 candidate 与 direct Highway/成熟 peer 必须同 ISA、同 TU 或等价 codegen 边界，并对齐 alias、rounding、
  accuracy、dispatch 和资源合同；运行 A/A、正反顺序和真实 size distribution。
- 5% 是 SIMD 当前 family 的 material margin。亚纳秒点值、地址或顺序翻转、单次结果不能修改公共 API、
  layout 或常态 metadata。
- fixed/fat codegen 必须确认最终 loop 没有 HYLE wrapper、逐元素 callback 或第二次 vendor dispatch。
- SVE/RVV 模拟器只签 correctness 和 codegen；性能必须来自对应真机。

性能仪器与停止规则见
[`SIMD benchmark methodology`](../developer/performance/simd-op-benchmark-methodology.md)。

## 明确负空间

Native FFI、稳定跨工具链 C++ ABI、per-ISA DLL、公开 register façade、逐 lane runtime callback、tensor/
model operator 和 GPU backend 均不因 SIMD vendor 集成进入本组件。真实 consumer 改变边界时，按模块收口
方法登记新事实并 reopen；benchmark 新增一行不构成 reopen trigger。

## 参考

- [Highway 官方文档](https://google.github.io/highway/)
- [xsimd](https://github.com/xtensor-stack/xsimd)
- [`SIMD/Math epic`](../implementation/tasks/simd/simd-epic-v1.md)
- [`SIMD dispatch 维护指南`](../developer/maintenance/simd-dispatch-guide.md)
