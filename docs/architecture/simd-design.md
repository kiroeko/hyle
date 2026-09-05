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
- 三种 C++ 交付的 consumer 都可以为固定 ISA 或多版本 kernel 使用 HYLE 的 target-bound seam；其实现可以静态路由到
  项目锁定的 Highway、xsimd 或经证明的 HYLE body，保持模板单态化和内联；
- HYLE 拥有 dtype、rounding、alias、layout、错误、选择和交付合同；成熟 vendor 拥有其已经正确解决的
  register/target/math 机制；
- Highway、xsimd 与 HYLE 实现可以同时作为候选，但同合同标准 primitive 以 Highway 为维护性默认；路由以
  `operation/family × scalar type × target/compiler` 为键，xsimd/HYLE 只保留已证 material 例外。升级
  vendor 或出现更好的 HYLE 算法时只重签受影响单元，逐步替换而不重塑公开语义；
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
 provider-coherent Highway / xsimd / HYLE body

expert consumer TU
      |
hyle_target_simd() or hyle_add_fat_kernel()
      |
compile-time route manifest -> provider adapter/body
```

第三方 vector/mask/tag 类型只存在于 target-bound implementation TU。跨 TU、静态/动态库和用户公开边界
只传 pointer、count、标量 POD 或 HYLE 自有稳定 storage type。这样可以独立升级 Highway，也不会把
target-dependent lane 数或 vendor namespace 固化为 HYLE ABI。

### 多实现与静态路由

HYLE 不把“引入 vendor”解释为删除全部自研，也不把“保留优势”解释为继续独立实现整个 portable SIMD
栈。每个准入能力可以同时拥有三类 implementation candidate：

1. `HIGHWAY`：覆盖广、target/ISA 演进快，通常作为完整性和可移植性的基线；
2. `XSIMD`：在其支持的 type/ISA/compiler 上作为同合同候选，尤其用于数学和不同 codegen 路线；
3. `HYLE`：只保存语义、复杂度、指令选择或真实 consumer 证据证明有价值的实现。

这套边界有四个长期产品原则：

1. Highway/xsimd 已经承担 portable SIMD 的广度、编译器适配和新 ISA 演进；HYLE 不再维护第三套大全；
2. 历史 HYLE 实现中已经明确击败两者的有限算法或 type × ISA/platform scope 必须保全，不能因采用 vendor
   基线而浪费；
3. 汇编相同、完整批次无法稳定区分、顺序/编译器反转或差异不 material 时，都没有足够的 HYLE 产品优势，
   统一采用 Highway；真正显著的优化会在有界证据中明显胜出；
4. route 必须由模板、宏、target namespace 或生成特化在编译期消失，build system 负责给每个 target body
   正确的 provider、依赖和 ISA flags。未来真实 consumer 暴露新热点时，可以加入一个 HYLE candidate 并只
   重签受影响 scope，逐步替换 vendor，而不重建平行 register 库。

路由清单是内部机器合同，不是用户 ABI。fixed/exact 构建把选择落实为宏、模板特化或直接 target symbol，
因此操作调用本身没有函数指针、分支或 CPU 探测。unknown-CPU ordinary/source-fat 仍只在 whole-buffer kernel
入口选择一次 target；这是未知 CPU 的 ISA 选择成本，不是 provider 路由成本。provider 已在每个 target body
编译时确定。一个 kernel 内不得因逐个 primitive 选择不同 vendor 而反复桥接寄存器类型：可以由
同一 provider 完整承载时优先保持 provider-coherent；必须混合时，只有编译后确认没有 spill、copy、二次
dispatch 或额外指令的 adapter 才可准入，否则把优势算法改写为当前 kernel 的 native lowering，或在更高的
whole-buffer 边界选择另一套完整 body。

历史 register 面机械提取出的每一个 operation 都必须在迁移账本中出现。逐项审计不等于运行
`operation × overload × size × ISA` 的无限笛卡尔积：相同实现和相同 codegen 可以共享一个资格化 evidence
group，但每项都必须显式引用该组；alias、selector、constant 和 composition 也必须有结构证明。任何漏项或
`PENDING` 都禁止删除旧实现和冻结候选。

账本把“候选发现”和“合同资格”分开。vendor 源码中的同名 symbol、operator 或 token 只说明值得审查，
不能证明 lane type、mask、NaN、fault suppression、rounding、alias 或资源语义相同。387 个 operation 的每个
route scope 都分别保存 HYLE、Highway、xsimd 三方的合同结论；三方未全部成为具证据的
`SAME_CONTRACT`、`ADAPTED_SAME_CONTRACT`、`COMPOSED_SAME_CONTRACT`、`NOT_AVAILABLE` 或
`NOT_COMPARABLE` 前，该 scope 仍是 `PENDING`。因此“不支持”也是逐项审计后的结论，不是没测到就省略。

同合同后的选择顺序固定为：先比较复杂度、遍历/临时存储和语义强弱，再看 material 的 consumer 或 P1
性能差异，最后才处理等价实现。Highway 是标准 primitive 的维护性默认值：合格的 `TIE`、最终 codegen
等价、低于 5% material margin、顺序/编译器反转，或在有界资格化后仍无法分出赢家时，均选择 Highway，
而不是继续保留一套平行 HYLE backend 等待更多微基准。只有同合同证据在可表达的 type × ISA/platform
scope 上证明 HYLE 或 xsimd 有可复现的 material 优势时，才为该有限 scope 建立例外；Highway 没有能力或
合同确实不同则保留 HYLE。未知项采用 Highway 不等于声称 Highway 全局最快，后续真实 consumer、依赖升级
或新硬件上的 material 证据可以触发具名 route 重签。这个规则允许实现逐项替换，也防止为微基准噪声维护
无意义的第四套形状。

### 普通 whole-buffer 路径

未知 CPU 的预编译库在进程内只解析一次 Platform facts、已编译 target 和 family policy，并构造一张私有
不可变函数表。这里必须区分“解析”和“调用”：CPU/target **解析只发生一次**；之后每个 whole-buffer API
调用只经过一次已缓存的 kernel-target 间接函数调用。provider 不在运行时选择，也不增加第二个间接层。元素循环内禁止再次探测、callback、第二层
`HWY_DYNAMIC_DISPATCH`、allocation 或 buffer copy。若采用 Highway dynamic dispatch，它就是唯一
decision authority，不能先由 HYLE 选一次再进入会二次选择的 vendor 入口。`selected_target(family)` 只报告
实际产品选择，不用于用户自行派发。

### 固定和源码路径

- `HYLE_SIMD_EXACT_TARGET` 让整个 HYLE SIMD artifact 只包含一个目标，普通入口由 generated manifest
  在编译期直接绑定该 target symbol；没有 runtime selector 或间接函数调用。
- `hyle_target_simd()` 为一个源码 consumer TU 提供中央锁定的 provider 头、静态 target 宏、编译期
  route manifest 和准确 ISA flags；kernel 由宏、target namespace 和模板直接单态化并可内联，不探测 CPU，
  也不经过 HYLE 函数表。需要 vendor 原生 API 的 consumer 仍可显式固定 Highway 或 xsimd；此时不承诺
  HYLE 会替它自动切换 provider。
- `hyle_add_fat_kernel()` 从同一个 provider-coherent implementation header 生成本平台的 per-ISA TU，只在
  完整 pointer/POD kernel 入口选择并缓存一次。

fixed source 的热循环应与该 route 在同 TU 原样调用获胜 provider 等价。source-fat 与 ordinary 路径只承担一次
batch-boundary 间接调用；target 解析本身只在进程内发生一次。宏可以消除已知 target 的 runtime 路由，
但不能在编译期获知一份可分发二进制最终运行的 CPU。tiny/fused kernel 若无法摊销一次库调用，应由源码
consumer 通过 fixed/exact seam 融合；HYLE 的 source seam 只暴露经准入的语义能力和 provider 选择，不为
兼容旧拼写复制 vendor API 全集。

完整构建合同见 [`SIMD dispatch 维护指南`](../developer/maintenance/simd-dispatch-guide.md)。

## Target 与探测 ownership

Platform 是 CPU/OS facts 的唯一 owner；SIMD 不复制或转发其 detector，也不要求 Platform 给出一个
“最高 ISA”。SIMD 独占公开的无顺序 `KernelTarget`，它只表示 HYLE 实际编译或选择的 body；每个
canonical target 的完整 feature requirements 也由 SIMD 声明。Highway/xsimd 拥有各自的寄存器 lowering
和编译器适用边界，SIMD family 选择 host facts、已编译 body、provider blocklist 与产品 policy 的交集，
所以同一主机上的不同 family 可以诚实报告不同 target。

公开面不提供“当前 consumer TU 编译 target”常量：源码嵌入、static SDK 与 shared SDK 对这种常量没有
统一语义。fixed/fat helper 的 per-TU target 只在内部/provider 编译期存在；普通用户通过
`selected_target(family)` 读取运行库实际选择，通过 `cpu_features()` 读取原始主机事实，不能把二者互换。

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

### Activation 与 row reduction 合同

模型无关的 f32 elementwise activation 与普通连续 row softmax/log-softmax 属于 SIMD。核心入口采用
`src, dst, n`，允许 exact alias 或完全分离的区间，不支持任意部分 overlap；`n == 0` 允许空指针。
它们是 O(n)、O(1) extra storage，不分配、不加锁、不建立整段临时数组。softmax 使用稳定 max-shift；
至少存在一个 finite lane 时，`-inf` 可用作 mask。含 NaN、`+inf` 或全为 `-inf` 的 row 没有受支持的
normalizer，整行确定写入 quiet NaN，避免静默制造概率。

activation body 与数学原语在同一个 direct-Highway target TU 中组合；ordinary/fat 批次只允许一次已缓存
目标选择，element loop 内不得再次进入 public dispatcher。attention mask、scale、bias、stride、线程、
online/blocked 算法、backward、半精度和模型 operator fusion 由 Compute 拥有，不能为消除一个 tiny-call
边界而把 tensor policy 倒灌到 SIMD。

### Search 与 scan 合同

普通 search 覆盖 i/u8、i/u16、i/u32、i/u64、f32/f64 的 count/find、value min/max/minmax、first
argmin/argmax 与升降序检查。空 index 返回 `n`，相同极值保留第一项；浮点 value extrema 传播任意
NaN，index extrema 返回第一个 NaN。它们均为 O(n) time、O(1) space、无 heap/lock。Highway contrib
直接承担 count/find/sortedness 与普通 extrema/index 机制；HYLE 只保留 NaN propagation 和单遍 combined
minmax 等产品语义差异，不维护第二套 register/search DSL。

scan 的核心是 `const T* src, T* dst, n`，inclusive 无 init，exclusive 显式接受 init；原地和 zero-init
spelling 只是 inline convenience。支持 u/i32、u/i64、f32/f64，exact alias 或完全 disjoint，部分 overlap
不受支持。整数按模 `2^bits` 定义且不能借 signed UB 实现；浮点结合顺序随 target。target TU 直接组合
Highway load/add/slide/store，Runtime 的函数指针与 `selected_target` metadata 同源；每次 ordinary
whole-buffer API 调用最多经过一次已缓存间接函数调用。custom predicate/op、segmented/parallel/tensor/GPU scan 分别归标准算法、source expert、
Graph/Compute 或 accelerator owner。

### Ordering 合同

普通 `sort/partial_sort/select` 是原地 comparison ordering，不建立与 `n` 成比例的 heap workspace；
`partial_sort` 只保证前 `k` 项有序，`select` 只保证第 `k` 项及两侧 partition。普通路径不稳定，不能按
长度暗中切换到稳定 radix，也不能为 benchmark 的小 `k` 特例保留 `O(n*k)` insertion route。numeric、
`UInt128` 与整数 `KeyValue32/64` 的 full/partial/select entry 和 `selected_target` 由同一 Runtime 表生成，
每次 ordinary whole-buffer API 调用只有一次已缓存间接调用。经典 cl.exe 因 Highway 官方 AVX3 blocklist 将 ordering 降到
AVX2；这是可报告的 per-family feature，不得伪称 AVX-512 kernel。

`partial_sort` 按算法结构选择机制：请求 prefix 能完整落入 VQSort 已有 sorting-network base-case 容量时，
使用成熟的 in-place heap selection（`O(n log k)`、`O(1)` extra storage）；更大 prefix 使用向量化
average-`O(n)` selection 后排序。边界来自 target/base-case 容量，不是输入总长度或某条 benchmark 的
经验阈值；两条路径都保持相同 NaN、payload、allocation 与公开 API 合同。

稳定 `radix_sort/argsort` 使用 base-256 LSD，时间为 `O(n * sizeof(key))`，空间为 `O(n)`。核心重载接收
caller-owned typed span，不分配；便利重载只负责分配同样大小的 workspace，失败传播 `std::bad_alloc`，
不会静默改用不稳定算法。`argsort` 保留 keys，输出 32 位稳定 permutation。浮点有限值按 numeric order，
NaN 在升降序都排在有限值之后；NaN payload 和 `-0/+0` 的输入顺序保持。keys、indices 与 scratch 的有效
字节区间必须两两不重叠，不能依赖只比较首地址的弱 alias 检查。

ordering 使用来自 Highway `83eff393...` 的具名最小 VQSort adaptation，而不是第二套 HYLE 排序设计。
直接以当前锁定 Highway 为 11 种公开类型、三种 operation 生成全部 target TU，在 classic MSVC 的有界
资格化尝试中无法满足构建时间预算；保留 adaptation 的 delta 限于 compiler/scalable target glue、HYLE
公开 POD、静态入口及上述产品合同。来源、路径、许可证和内容 hash 由中央 dependency lock 机械验证。

## Vendor 与升级边界

锁定 Highway 与 xsimd 都是 header-only product candidates，不形成第二个 runtime ABI。每个 route 只有在
语义、资源、type/target 和 current peer 资格化后才进入产品；“两套都能编译”不是胜负证据。多实现通过
编译期 manifest 或完整 whole-buffer entry 并存，不能为此在元素循环增加 runtime abstraction。

checked-in Highway-derived port 不因改 namespace 就变成自实现。direct wrapper 可行即删除；必须保留时
只保留真实 contract delta，并登记 origin、local delta、许可证和内容 hash。ordering 的有界裁决保留上述
最小 adaptation；后续 Highway 升级必须重新移植相关 upstream 修复，而不是把它视为冻结自实现。完整规则见
[`SIMD vendor ownership`](simd-vendor-surface.md)。

## 依赖与交付

SIMD 的 HYLE 依赖只有 Core 和 Platform。Container 的选定 kernel 与 SIMD 可共享中央锁定 vendor 的
build-only 外部实现 target，但 Container 不调用或依赖 `hyle::simd`；Compute 可以消费 ordinary SIMD API，
SIMD 不依赖 Compute。

source、installed static 和 installed shared 三种交付必须分别验证。三者都交付 ordinary pointer/count
API，也都能由用户显式链接 `hyle::simd_expert`，取得精确锁定 Highway/xsimd 开发头以及同一套 fixed/fat
CMake helper。旧 `vec.h`、`native/*`、`backend/*` 已物理删除；`target.h` 与 `fat_kernel.h` 属 expert SDK，
`provider/*` 仍是 HYLE 内部 route 实现，不安装。expert target 只编译进 consumer TU，不产生新的 HYLE 或
vendor binary ABI；符号检查只允许 HYLE ordinary API 与小型诊断面，不得导出私有 Runtime 表或 vendor 类型。

Highway/xsimd 的原生 namespace 是专家层本身：HYLE 不把其完整函数集重新包装成数百个同义 API。锁定版本
提供但 387-operation 历史迁移账本未覆盖的 header-only 能力，可由用户在 expert TU 直接调用；需要上游
`.cc` 的 thread pool、完整 VQSort/image/hash 等 compiled contrib 不因安装头文件而自动成为 HYLE 产品，必须
由真实 capability owner 单独准入、构建和验证。

## 性能与验证原则

- P0 先检查 O(n) 下界、遍历次数、临时 storage、dispatch 次数和可选 feature gate。
- P1 candidate 与对应 Highway/xsimd peer 必须同 ISA、同 TU 或等价 codegen 边界，并对齐 alias、rounding、
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
