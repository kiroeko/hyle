# SIMD-10：Bitpack 与 random

- ticket state：`ACTIVE`
- kind：product capability/correctness
- depends on：SIMD-09
- budget：1 工程日；性能预算 60 分钟
- input commit：`c2ff702218dbe692f3178a4e8eb9ed00b982dd64`

把当前“同 tier 才可解码且没有 format tag”的普通 bitpack 修成 canonical little-endian wire layout：
u8/u16/u32、1..lane bits、跨 x86/NEON/SVE/RVV/scalar 字节一致、VL 独立、明确 overlap/size。native quant
layout 与协议 framing/压缩格式仍由 Compute/格式 owner 持有。

同时完成 caller-owned non-crypto RNG state 的 seed/copy/advance/split/fill 生命周期，明确复制会复制流、
同一 state 非并发安全、worker 应持有独立 split state。entropy、密码 RNG 和分布采样不属于本票。P1 只测
u16 11/13-bit 与代表性 f32 fill；优先检查 bytes moved、额外 pass 和隐藏状态。

## 资格化审计

- 锁定 Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c` 的 contrib bit-pack 提供完整
  u8/u16/u32 register mechanism，但 layout 以 target lane count 为参数；它不能直接成为 persisted ordinary
  format。当前 HYLE 又把旧 Highway port 逐行迁入约 2,800 行 `bitpack_hyle-inl.h`，只有 u16 11/13 的
  correctness delta 与固定 logical-block 编排是具名 HYLE gap。其余平行源码应由 direct Highway 替代。
- 当前 ordinary `pack/unpack` 的 full block 随 AVX2/AVX-512/NEON lane count 改变，scalar tail 又是另一种
  contiguous layout；公开 buffer 没有 format/version tag。SVE/RVV 只冻结到 NEON layout，仍不能由 x86 或
  scalar 解码。`packed_size` 虽然 tier-independent，不能补救可观察 bytes 不同。这是 persisted/model-input
  journey 的 correctness/API 缺陷，不是 benchmark 缺口。
- canonical v1 采用固定 128-value logical block：每个 full block 使用 128-bit logical lanes 的 Highway
  Pack8/16/32 布局，按 block 顺序输出；不足一个 block的尾部按 value-order LSB-first 连续打包；所有 T word
  以 little-endian bytes 表示，最后 padding bits 为零。这样可保留成熟 bit-pack mechanism，又让 x86、
  NEON、SVE、RVV 与 scalar 共享唯一 bytes。它是无 framing 的 payload codec；type/count/bits/version 仍由
   文件、网络或 quant-format owner 记录。资格实现同时证明 Highway 的高速 Pack 以输入已落在
   `bits` 范围为前置条件；其官方测试也先 mask 输入。审计草案中的“自动忽略高位”会要求热路径逐值
   mask，违背 direct-mechanism 与单遍成本目标，故在首个产品提交前校准为调用者前置条件。
- 当前 bitpack dispatcher 在私有 Runtime 只读取 module `isa`，随后自己维护 `static bool` 与平台分支来选
  symbol；metadata 与 function pointer 不是同一 authority。候选必须让六个 type/op pointer 和 module
  metadata 同表产生，ordinary 每完整 buffer 只有一次 cached indirect call，exact route 直接绑定 suffix。
- Highway random contrib 证明 xoshiro256++ 的 `Jump`/`LongJump` worker-stream 机制；标准 engine lifecycle
  证明 reseed/copy/discard 是普通形状。当前 `RandomState` 的八个 jump-separated stream、固定 round-robin
  输出、u64/f32/f64 fill、跨 ISA/chunk 一致性已成立且无 heap/TLS/lock，但只能构造 seed，缺 reseed、skip
  和并行 worker split；复制语义与非并发安全也只靠默认语言行为，用户指南没有完整说明。
- `native/random.h` 只有测试自消费，继续依赖迁移期 `vec.h`；ordinary generated random TU 也未使用已经锁定
  的 Highway source seam。候选直接使用 Highway register ops，HYLE 只保留固定八流序列、float 映射、
  state lifecycle 与 runtime glue。Highway 自带的 lane-count-dependent `VectorXoshiro`/allocating vector return
  不是同合同替代品，不能直接泄漏为 ordinary API。
- llama.cpp/ggml 没有 generic bitpack consumer；其 quant/GEMM packing 是格式与算子融合，继续归 Compute。
  llama sampling 主要消费单值 engine，不能用来扩张 bulk SIMD distributions。当前 HYLE 产品树同样没有
  非测试 direct consumer，因此本票只完成已经准入的 generic codec 与 bulk deterministic state，不新增
  entropy、distribution、token sampler 或 quant block API。
- 旧 `bitpack_bench` 混合 tier-dependent/错误 vendor/layout 不同的行，且无统一反序与 A/A；不能签本票。
  RNG 没有当前资格化 P1 instrument。二者只作为历史诊断输入，候选使用同格式/同序列 direct Highway
  composition，并按 `closure-perf-v1` 生成本票证据。

## 冻结合同与有限 FIX 集

1. 保留 u8/u16/u32、`bits=1..lane width` 与 `packed_size<T>` 的 ordinary pointer/count core；增加自然
   `std::span` convenience 和 `kBitpackFormatVersion=1`。公开 payload 使用上述 canonical fixed-128 +
   little-endian v1 layout；`count==0` 可传空指针，非空时指针有效，输入/输出不得重叠，超过 `bits` 的高位
   必须已经落在 `bits` 范围内，codec 不执行隐式截断 pass。`packed_size<T>` 以 T words 计数且 O(1)，
   codec 为 O(n)/O(1)、无 heap/lock/whole-buffer copy。
2. fixed/scalable target TU 直接实例化锁定 Highway bit-pack；宽寄存器可以在有界 stack tile 内并行多个
   fixed logical block，但不得改变 bytes、建立整段临时区或增加第二遍 whole-buffer traversal。仅保留
   fixed-format 编排与已经由独立 oracle 证明的 u16 11/13 correction；删除其余 checked-in vendor port。
3. bitpack 六个 type/op pointer 与真实 `isa/features` 进入唯一私有 Runtime 表，本票只允许把 table version
   从 9 升到 10 一次。ordinary 一次 batch call，exact 直接 suffix；删除第二 platform resolver 和旧
   `kBitpackLogical128` 的“只有 scalable 才固定”含义，canonical v1 对所有 tier 恒成立。
4. `RandomState::seed(seed)` 复位确定性流；默认 copy/assignment 明确定义为复制当前位置。`advance(n)` 跳过
   与 fill 相同的 n 个 logical outputs，O(n) 但不写临时输出；`split()` 返回父状态当前 segment 的 child，
   同时把父状态的八条 stream 各自 LongJump 到下一个 2^192 segment，固定成本、无 allocation。fill 保留
   u64/f32/f64 `[0,1)` 与跨 ISA/chunk 位一致；同一 state 非并发安全，worker 只使用预先取得的独立 child。
5. random generated target TU 改为 direct Highway ops，删除无真实 consumer 的 `native/random.h` 与对应自测；
   stream semantic、state memory 与 float conversion 仍由 HYLE 拥有。Runtime 增加真实 random module metadata
   和 advance pointer，不使用 TLS、全局 mutable state、entropy 或逐元素 dispatch。
6. correctness 覆盖 canonical golden bytes、所有 type/width、zero/tail/multi-block/input-fit/padding、
   exact/public/cross-tier bytes、SVE/RVV 多 VL/VLEN、header self-containment 与三交付 consumer；RNG 覆盖
   seed/copy/advance/split/chunk/float bounds、跨 tier sequence 和明确并发负空间。P1 只测 u16 11/13
   pack/unpack 与 f32 fill 的 4096/1M batch；candidate 与同 TU direct Highway/同合同 composition 对齐格式、
   stream、ISA、dispatch、资源，正反序、A/A、双 layout，5% material margin。旧 target-dependent format、
   broken 11/13 vendor 或不同随机序列一律 `NOT_COMPARABLE`。

本票只允许上述六项。generic u64 bitpack、variable-bit-per-value、frame-of-reference、delta/varint、streaming
incremental codec、quant block/GEMM fusion、协议 framing、entropy/crypto RNG、分布采样和 thread-safe shared
state 分别由未来 consumer、Compute、format owner、标准/专用随机库承担；不得在本票扩张。
