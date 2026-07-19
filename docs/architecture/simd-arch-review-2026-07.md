# HYLE SIMD 架构评审(2026-07)

> **结论先行**:架构没有系统性问题;评审识别出**两处真实的架构级债**——①定宽后端表示层丢 N
> （NEON 2026-07-15 根治，x86 2026-07-17 以分层 XMM/YMM/ZMM + N-aware 低半消费闭环）、
> ②手写 op 面的横切成本(simd-19 是它第一次真正收费)。债①已还清，债②有不推倒重来的改善路径。**不需要推倒重来,
> 也不该走向 vendor Highway。**
> 背景:一个月 HYLE-vs-Highway 性能对齐(见 `docs/milestones/simd-orderswap-matrix.md`)后的架构侧复盘。

> **2026-07-16 补充（simd-20/22）**：本文评审的两项债针对寄存器表示与 backend op 横切面，结论仍成立；
> 当时尚未收口的第三类问题——分散 `active_*` API、只有静态 `VectorXoshiro`、普通 API 与固定目标入口
> 缺少统一模型——已由 [`simd-dispatch-guide.md`](./simd-dispatch-guide.md) 的 Runtime/native/exact 双轨
> 解决。产品也已零 Highway/xsimd，外部库只保留为 benchmark 对照。这是增量重构，不推翻本文对 D-tag/
> backend 地基的判断。

> **2026-07-17 补充（x86 partial 收口）**：把原先仅挂 NEON 的 exact-byte boundary 与 exhaustive
> half/combine oracle 架构中立化后，首次在 x86 暴露 13 个 sub-XMM N-blind 形状；根因是 SSE4
> `UpperHalf/ZeroExtendVector/InsertIntoUpper` 固定按 64-bit half 操作。现已改为按逻辑
> `N*sizeof(T)/2` 生成 shift/mask/splice。Windows 真跑 SSE4 294/294、AVX2 325/325；WSL GCC14
> partial 专项 SSE4 69/69、AVX2 99/99，AVX512 129/129 由 Intel SDE `-future` 真跑全绿。该门永久
> 接入基础 ISA targets。

> **2026-07-19 模块收口复审（非性能面）**：性能矩阵不重跑，本轮专查 Runtime/exact/native 三腿、
> 构建机械化和稳定 C ABI。发现并修复 5 个接缝问题：① detection-only `kAvx` 在公开 fat selector
> 错降 scalar（现与 Runtime 一样降 SSE4）；② `hyle_add_fat_kernel` 从未有真实消费者纵切，exact
> 只生成一个 symbol 时调用点仍会引用其余 symbol（现由同一选择宏在预处理阶段消除，fat/exact 都有
> pointer/count kernel 回归）；③ macOS 不生成 SVE TU，ARM64 调用点却仍 odr-use SVE symbol（现由
> 平台可用性宏预处理消除，并由 CMake 拒绝缺少本平台必要 tier 的 fat 清单）；④ C runtime snapshot
> 把“当前完整 sizeof”误当最低版本，未来追加字段会拒绝旧客户端（现钉 `V1_SIZE` 可读前缀）；
> ⑤ C module metadata 暴露裸 feature 数字和隐式 padding（现有具名位、显式零 reserved 与布局断言）。
> 结论仍是不推倒寄存器 DSL，但“公开双轨已完成”现在有端到端证据，不再只靠内部 ops 推断。

## 一、判断依据(架构质量怎么测,不靠感觉)

在校准仪器(锚点校准 + 双序判定 + 三平台,见 `docs/performance/simd-op-benchmark-methodology.md`)下:

- 299 个已 bench op 中 **291 个 SOTA**(与 Highway 平或胜);MSVC / GCC 真输**清零**;NEON 剩 2——
  一个是 PromoteMaskTo 0.575(即债①的量化样本),一个是 ScatterIndexN 0.87(编译器寄存器类微差,非算法)。
- 推断逻辑:**如果地基是歪的,逐 op 修不可能把表打平到这个程度**。~20 轮修复全部是 op 级(指令选择、编译器
  fork、契约收窄),没有一轮需要动 tag 系统、分派结构或测试基建——能打平,说明主体结构是对的。

## 二、两处架构级债

### 债① 表示层丢 N(simd-19;**NEON 与 x86 均已收口**)

类型系统携带 N(`Rebind`/`Repartition` 产生 partial tag),scalable 后端(SVE/RVV)用 `kPow2`+`vl` 正确
执行了它,但**定宽后端的物理表示曾不携带 N**——修复前 `VecNeon<int16_t,4>` 物理上仍是 `int16x8_t`,
op 被迫满宽做再掰回。这是设计时漏掉的一个维度,是缺陷不是权衡。

- **NEON:已根治**(simd-19,当日立项/施工/收口)。`RawNeonNT<T,N>` D 档表示 + 常绿垫片迁移 + ~17
  热点 native-D;PromoteMaskTo 0.575→TIE(C12 单 op 双序 + C18 全量双证,反汇编与 Highway 同形),
  全量复跑零回归,顺带修复两个既有 bug 家族(partial LoadU/StoreU OOB、子满宽 half/pad N-盲语义)。
  全记录:`docs/milestones/simd-19-partial-vector-repr.md` §八、`simd-19-ledger.md`。
- **x86:已根治**(2026-07-17)。x86 没有比 XMM 更窄且值得使用的 SIMD 寄存器，正确形态不是复制 NEON
  D 表示，而是：`Vec128/256/512<T,N>` 与 `DFromV` 保留逻辑 N；AVX512 half 真降 YMM、AVX2 half
  真降 XMM；sub-XMM 保留物理 XMM，但 exact-byte Load/Store 与 half/combine 消费者按逻辑 byte count
  操作。架构中立回归门覆盖所有 power-of-two partial LoadU/StoreU 边界和所有 half-able 形状；它实际抓出
  并修复了 13 个 sub-XMM N-blind case，而不是只给既有实现补绿灯。

### 债② 手写 op 面的横切成本

每 op 手写 `if constexpr` + 硬编满宽 intrinsic,是当年拿扩展性换可读性/可断点/贴硬件的**有意权衡**(章程
"不 vendor Highway 宏体系")。simd-19 是这个权衡第一次真正收费:横切维度一变(加 D-form),270 op 全要碰。
今后每加一个新维度(f16/bf16 lane 类型、SVE2、AVX10)都会再收一次费。

这个债**不用还清**(还清 = 走宏生成 = 丧失可读性,违背章程),但需要一层收敛结构:**宽度分发 helper**
(`neon_bin<QFn,DFn>` 模式)——op 语义仍手写,"类型→intrinsic 选择"收敛到 helper,未来新维度只改 helper 层。
simd-19 阶段 1 正好是它的落地场。

### 债② 追问:Highway 宏体系是不是"正确答案"?HYLE 不走它会不会持续出问题?(2026-07-15)

数据(当时用于调研的 Highway checkout `hwy/ops/arm_neon-inl.h`,10982 行):`HWY_NEON_DEF_FUNCTION*` 宏调用
286 处,`template <` 966 处——**Highway 的 NEON 后端也只有规整那部分由宏生成,其余大量仍是手写特化**。
宏解决的是"规整 op × (type×size) 矩阵的机械展开";难的那部分(本月 ~20 轮修复对应的指令选择、编译器
怪癖、契约边界)两家都得手写。

宏体系对 Highway 是对的,因为它的**结构**需要:一份 op 源码经 `foreach_target.h` 对每个 target 反复
re-include 展开(x86 128/256/512、NEON、SVE、WASM…),没有宏就是复制地狱。HYLE 结构不同:每 ISA 一个
后端头、每 target 一个 TU,不存在同源多 target 展开需求;C++20 `if constexpr` + 模板 helper 能以可读、
可断点的形式覆盖同样的分派。(行数对照:HYLE neon.h 3297 行 vs Highway arm_neon-inl.h 10982 行——后者
覆盖面更宽(armv7、f16/bf16、全 partial 档),不是纯赘肉,但说明宏并没有让后端变小。)

不走宏的真实风险 = **横切新轴时的一次性迁移成本**,且轴可枚举:
1. partial size——本里程碑(simd-19),一次垫片迁移 + ~20 热点升级;
2. 新 lane 类型(f16/bf16)——将来若做,helper 层落地后成本可控(只改"类型→intrinsic 选择"一层);
3. 新 ISA 代际(SVE2/AVX10)——两个世界成本相同,Highway 也要写新 -inl.h。

且有**检测器兜底**:校准 bench 会把轴缺失暴露为量化输局(PromoteMaskTo 0.575 正是这样被抓住的),
不会静默腐烂。结论:宏体系不是普适正确答案,是 Highway 结构约束下的正确答案;HYLE 的对应答案是
helper 层 + 校准闸门。持续风险有界、可检测、有着力点。

## 三、什么不是架构问题(防误诊)

- **MSVC 复合上下文劣化 / 寄存器类微差**:编译器层的事,fork 纪律(`#if _MSC_VER && !__clang__`,GCC-反赢
  陷阱教训)已是正确应对;不存在架构级修法(除非生成 asm)。
- **覆盖面缺口已闭环**：旧 41 个候选已由 manifest 分类；masked arith P1 是旧差集误报，代码早已
  完整。15 个 native 新行在三平台各自可用 target 上的第二轮狩猎为 0 LOSE，见 simd-23 / orderswap matrix。
- **API 镜像 Highway**:是使能逐-op 对拍的**选择**,不是被动跟随的证据——InRange 契约族(TableLookupBytes
  等)证明能在镜像之上有节制地做得更好。代价是继承了 Highway 的 op 面设计决策(如 mask 宽度桥接族的存在),
  可接受。

## 四、改善路线(按 ROI 排;状态更新 2026-07-17 收口)

1. **simd-19 常绿落地** ✅ **完成**。表示层补 N 维(`RawNeonNT<T,N>` D 档),常绿垫片迁移全程兑现
   (W0a/b/c 每步绿,~17 热点 native-D);PromoteMaskTo 0.575→TIE(C12+C18 双证),全量复跑无回归,
   neon.h +21%(Highway 的 36%,警戒线未触发)。NEON 物理 D/Q 形态与 x86 分层宽度形态均已闭环；
   x86 见路线 3。
2. **宽度分发 helper 层——被实践修正,未建且暂不建**。simd-19 实际用"垫片(to_q/from_q)兜底 + 热点
   op 内联 `if constexpr (kIsDBand)`"替代了设想中的 `neon_bin<QFn,DFn>` 层:宽度轴上两形指令仅差 q
   后缀,内联分支更贴文件风格、无额外抽象。**修正后的判断**:垫片只解宽度轴;未来的**类型轴**
   (f16/bf16——全新指令,垫片无从兜底)才是债②的真考验,收敛层在那时按量化需要再建,现在建属
   过度抽象。simd-19 留下的真资产是**已实证的横切迁移方法**(脚本包裹 + 常绿三步 + 工单制:270 op
   规模一天完成)——债②本金未还,利率大降。
3. **✅ x86 partial 健康化完成(语义审计 + correctness canary，不造 MMX 表示)**。分层
   `Vec128/256/512<T,N>` 保留 N；SSE4 exact-byte I/O 与按逻辑宽度的 half/splice 消除脆弱路径；
   boundary + scalar oracle 永久挂入 SSE4/AVX2/AVX512。Windows 全套与 AVX512 SDE 门均通过。
4. **✅ gap-inventory 复核完成**：补 4 个 upper-only 128-bit key 接口；拒绝复制 Highway internal
   widen helper；新增 op 同提交进 manifest/oracle。
5. **把测量体系当架构质量的常设准绳**。锚点校准、双序判定、三平台 gtest、复合上下文 canary——这套仪器是
   这轮对齐战最大的资产;"架构好不好"从此靠它读数,不靠感觉。

## 五、底线判断

simd-19 已落完(helper 层按路线 2 的修正暂不建,类型轴到来时再议),HYLE simd 的形态(**手写可读 op 面
+ 类型级 tag + 校准闸门**)长期站得住:
新增 op 有方法论文档护航,新增维度有 helper 层着力,性能回归有仪器兜底。推倒重来没有依据;vendor Highway
只在触发 §5.2 退出条件时才重新讨论。

## 六、模块收口复审：便利 kernel 与 C ABI 是否够用（2026-07-19）

### 双腿与非算子架构

- 内建 ops/random/scan/search 的 variant 仍只是一层稳定 symbol，唯一算法 body 在 `native/`；
  exact dispatcher 直接引用 manifest suffix，fat Runtime 只在 whole-buffer 边界间接一次。
- math/dtype/sort/bitpack 保留手写策略是必要差异：它们有 per-op winner、可选 feature 和阈值，
  不应为了“生成统一”抹掉真实策略；Runtime v6 的 module isa/features 已把降级事实显式化。
- 自定义 fat kernel 现在由 `tests/simd/fat_kernel_fixture.h` 真正编成各 ISA 并执行，不再把
  `select_fat_kernel` 对几个返回常数的测试冒充构建机械化验收。
- exact 构建只链接 manifest 的唯一 symbol；macOS fat 构建只链接 scalar/NEON，不再虚假引用 SVE。
  CMake 同时拒绝缺失本平台必要 tier 的 fat 清单，配置与调用点不能静默漂移。
- `vec.h` 仍是 target-bound DSL；387 个寄存器名不跨 Runtime/C ABI。这个边界正确，无须扩大。

### 便利 kernel 面

对照 Highway `contrib`，HYLE 已覆盖适合 SIMD 模块且能形成稳定 pointer/count 产品的主体：

| Highway/HYLE 语义 | HYLE 产品入口 | 结论 |
|---|---|---|
| count/find/minmax/is_sorted | search | 已覆盖，且有 runtime/native 双腿 |
| dot | 同/混合精度 dot、i8/i16 dot | 已覆盖 |
| random | opaque RNG + u64/f32/f64 fill | 已覆盖 |
| bit_pack | u8/u16/u32 pack/unpack/size | 已覆盖 |
| sort | sort/partial/select/radix/argsort/KV/u128 | HYLE 面更宽 |
| math | 实数、复数、激活、softmax | HYLE 产品面明显更宽 |
| scan/dtype | prefix scan、f16/bf16/fp8 转换 | HYLE 自有产品面 |

不纳入 SIMD 的 `hash/matvec/thread_pool/image` 分别属于 container/compute/sched/上层领域；`copy` 没有
超越 libc/编译器的独立语义，回调式 `transform` 又不能形成跨 C ABI 可内联的通用 kernel。基础库无需
为目录齐平复制它们。普通数组工作流已经能完成“算术/查找/归约/随机/转换/数学/排序/压缩”，因此
便利 kernel 数量足够模块收口；后续新增应由新的用户语义或稳定 Highway LOSE 驱动，不再按 387 名扩面。

### C ABI 易用性

稳定 C 面覆盖全部上述 whole-buffer 能力：244 个导出符号 + 3 个 header-inline packed-size helper，
数组由调用方拥有，窄类型/KV/u128 用固定 POD，RNG 用 opaque handle，诊断用 `struct_size` snapshot。
本轮补上 feature 常量和 snapshot 前向演进后，没有要求 C#/Java/Rust 理解 C++ Runtime、STL 或
target vector。语言绑定只需把本语言数组 pin/borrow 成 pointer/count；容器 facade 应留在绑定层，
不在核心库再造一套 `std::vector` 产品 ABI。

### 本轮验收证据

- Windows MSVC fat 与 exact-AVX2：各 369 项，368 通过、1 个宿主无 AVX512-BF16 的预期跳过；
  两种构建都执行真实自定义 pointer/count kernel，exact artifact 门确认只含 AVX2。
- WSL `/root/hyle-simd-review` GCC 14：369 项同结果；纯 C 用例通过；机械覆盖
  247/247，ELF 导出 244/244。
- Apple Silicon AppleClang fat 与 exact-NEON：各 366 项，365 通过、1 个非 x86 预期跳过；
  exact artifact 门确认只含 NEON。fat 首轮真实抓出并修复“未生成 SVE symbol 仍被引用”的漏洞。

## 七、参考

- 对齐战全记录与真输清单:`docs/milestones/simd-orderswap-matrix.md`
- 债①修复里程碑:`docs/milestones/simd-19-partial-vector-repr.md`
- 覆盖面清单:`docs/milestones/simd-gap-inventory.md`;复杂度机会:`docs/milestones/simd-complexity-opportunities.md`
- 测量方法论:`docs/performance/simd-op-benchmark-methodology.md`
- 仪器伪影狩猎记录:`docs/debug/op-parity-harness-artifacts-hunt.md`
- 章程(不 vendor / 行数警戒线):`docs/architecture/engineering-charter.md`,`simd-design.md` §229
- 双轨派发终态与维护规则：[`simd-dispatch-guide.md`](./simd-dispatch-guide.md)
