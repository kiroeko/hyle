# SIMD-11：Ordering、radix 与 top-k

- ticket state：`ACTIVE`
- kind：product capability/correctness/performance
- depends on：SIMD-10
- budget：1 工程日；性能预算 60 分钟
- input commit：`3185aa48c4d63ee2bd6b8275abbf78c3218823e7`
- qualified peers：Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c`、ggml
  `36da57138425487184aa1da2eee2cde155909c6f`、llama.cpp
  `7584430716ee229751771ed0d6bbcb780d105eeb`

## 资格化审计

- 公开面已经覆盖 u/i16/32/64、f32/f64、`UInt128`、`KeyValue32/64` 的 full sort、partial sort 与
  select；stable radix/argsort 草稿覆盖八种 scalar key。当前缺口不是扩大容器、tensor 或并行排序面，
  而是把已准入形状的资源、稳定性、浮点顺序和交付合同做实。
- 普通整数 `sort(n >= 4096)` 会按 benchmark 阈值偷偷分配 O(n) radix scratch；small-k partial route
  在随机输入很快，但逐个接受逆序/对抗输入时最坏为 O(n*k)。两者都违反普通 no-allocation API 与成熟
  comparison ordering 的成本合同，不能因单一 benchmark 获胜保留。
- ordering 仍在 `sort_dispatch.cc` 自行读取 Platform ISA 并维护 per-type symbol ladder；唯一 Runtime 表
  只有 sort metadata，没有对应 function pointer。这会让真实函数、metadata、exact 与 ordinary route
  分叉，是控制面缺陷而非性能微差。
- checked-in VQSort port 源自旧 Highway，并包含约 3.5K 行平行实现。当前锁定 Highway 已能直接提供 fixed
  target numeric/KV/U128 sort/partial/select，但 release classic MSVC 和 scalable SVE 会禁用 VQSort，直接
  调用会退为 heap algorithm。候选只允许一次有界兼容验证：能以小型 target seam 证明正确性、codegen 与
  无 material regression 时转为 direct Highway；否则保留并具名最小 compiler/ISA delta，不得为了减少行数
  静默引入数量级回退。
- 当前公开 `UInt128`/KV 类型与 vendor internal struct 只证明 size/alignment 相同，却通过跨类型
  `reinterpret_cast` 排序；这不足以证明 C++ object lifetime/aliasing 合法。候选必须使用可证明合法且无
  whole-buffer copy 的 bridge，或保留由 HYLE public type 直接实例化的最小实现。
- stable radix 草稿采用 caller-owned typed scratch、base-256 stable passes，复杂度为 O(sizeof(T)*n)、
  O(n) scratch；allocating convenience 不静默降级。但 argsort 尚未拒绝 keys/indices/scratch 的全部交叠，
  异常说明也与实际 `bad_alloc` 路径不一致，32-bit index 边界、NaN payload、signed-zero/first-equal stability
  仍需系统证明。
- 真实 north-star 中，ggml/llama 的 `argsort` 与 logits top-k 都先构造 index permutation；HYLE 的 stable
  f32 argsort 后取 prefix 已是保留 keys 的自然 O(n) scratch route。破坏 keys 时使用 `partial_sort`，整数
  payload 使用 KV 类型。本票不因 benchmark 另造 `partial_argsort`/tensor-row/threaded API；有具名 consumer
  证明 full argsort 成本不合适时再按 capability admission 重开。
- 现有 `sort_bench` 把输入 copy、不同资源/稳定性合同和 broad Cartesian case 混入计时，缺少 A/A、反序与
  shared parity qualification，且主矩阵没有完整覆盖 f32/f64；只能作为历史诊断，不能签本票。

## 冻结合同与有限 FIX 集

1. 普通 full/partial/select 保留现有 type family 与 pointer/count C++ shape，`count==0` 允许空指针；非空
   buffer 有效且 in-place。ordinary route 无隐藏 heap、whole-buffer copy、lock 或逐元素 dispatch。删除
   `n>=4096` auto-radix 和 benchmark-derived streaming small-k route；复杂度回到所选成熟 comparison
   mechanism 的有界合同。stable ordering 只通过显式 radix/argsort workspace 或 allocating convenience。
2. 用既有 generated-kernel seam 直接绑定锁定 Highway VQSort，并只做一次 classic-MSVC/SVE compatibility
   spike。通过则删除对应 checked-in vendor body；未通过则记录精确 compiler/ISA 原因，只保留 public-type、
   dispatch 或 scalable-width 所需的具名最小 delta。scalar 使用标准算法；不得强开 classic-cl AVX-512，
   不得把 upstream-disabled heapsort 冒充等价高速替代。
3. ordering 的 numeric、U128、KV full/partial/select pointers 与真实 `isa/features` 进入唯一 Runtime 表，本票
   只允许 table version 从 10 升到 11 一次。ordinary 每个操作一次 cached batch dispatch，exact 直接绑定
   suffix；删除第二套 Platform resolver。跨 public/vendor 类型的调用必须满足对象生命周期和 aliasing，
   且不得为规避 UB 添加 O(n) conversion copy。
4. stable radix/radix argsort 提供 u/i16/32/64 与 f32/f64；caller scratch 是 hot route，allocating overload
   是明确 convenience，分配失败传播 `std::bad_alloc` 而不降为不稳定算法。浮点升降序均为 NaNs stable-last，
   相等 key（含 signed zero）保持输入顺序。拒绝 undersized scratch 与所有 live range byte overlap；公开
   32-bit argsort index 上界和长度失败合同。算法保持 O(sizeof(T)*n)、O(n) scratch、无额外最终 whole-buffer
   copy。
5. user guide 给出“破坏 key 的 top-k / KV payload / 保留 float key 的 stable index prefix”选择表，说明
   ordinary 与 stable route、workspace 复用、OOM、NaN/signed-zero、稳定性和 index 上界。architecture 记录
   vendor ownership 与负空间；dependency/port lock 按最终实际保留内容机械重算。不开 C ABI，不新增 tensor、
   strided、parallel、external/distributed sort 或排序容器。
6. correctness 覆盖所有公开 type、zero/one/tail、升降序、duplicate、NaN payload、signed zero、first-equal、
   scratch/OOM/overlap/index boundary、ordinary/exact/runtime、header self-containment 与 source/static/shared
   consumer；W1-WIN shipping tiers 和 exact artifact 必须过，SVE/RVV 在现有 QEMU 多 VL/VLEN 只签
   correctness。P1 仅测 f32 full sort 4096/1M、f32 partial_sort(4096,10)、scratch u32 radix 1M、f32
   argsort 1M；仅同 contract/resource/ISA peer 使用统一 parity engine，执行 A/A、正反序、双 layout 和 5%
   material margin。不同稳定性、NaN、payload、workspace 合同标为 `NOT_COMPARABLE`，不得扩张仪器或测试姿势。

本票只允许以上六项和它们直接暴露的构建/测试/文档修复。若 direct Highway compatibility spike 不能在一次
有界尝试内证明，必须选择具名最小 delta 或把该适用 gate 裁决为 `BLOCKED`；不得无限研究编译器、地址布局或
亚纳秒噪声。到达预算后以 `DONE` 或明确 blocker 终止，SIMD-12 才负责聚合 closure。
