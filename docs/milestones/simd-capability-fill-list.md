# SIMD 能力补缺清单（只补能力/性能）

> **状态**：当前真相账本（2026-07-19 回填，对齐 simd-25 + `simd-scalable-coverage.py`）。
> **范围**：寄存器/算法/产品核/性能洞与相对 Highway 的能力差。
> **不在本文**：close-01 C ABI、推广合同文案、用法包装（另线）。
> **姊妹文档**：
> - 施工：[`simd-24-foundation-completion.md`](./simd-24-foundation-completion.md)、[`simd-25-scalable-product-parity.md`](./simd-25-scalable-product-parity.md)
> - 差集审计：[`simd-gap-inventory.md`](./simd-gap-inventory.md)
> - 复杂度轴：[`simd-complexity-opportunities.md`](./simd-complexity-opportunities.md)
> - 变宽底账：`scripts/simd-scalable-coverage.py` + `scripts/simd-scalable-op-manifest.json`
> - 覆盖率：`scripts/simd-op-coverage.py`

**一句话**：定宽面与主 whole-buffer 核已立；**SVE/RVV 主产品核（math/dtype/bitpack/sort）已有真实 TU + QEMU 门，SVE partition 已吃满物理 VL，精细 feature probe 已落地**，不再是「整模块 hard clamp」。simd-24 的 C1–C3 能力面已补完，D1/D2/D3/D5/D6/D7 机制收口也已完成；**NEON KV64 已在 Apple 真机追平 Highway**。还缺的是 **变宽真机签字**；GFNI/类型 helper 仍须有产品消费或类型轴才成立。不要再为齐 387 名字抠常数。

---

## 0. 已落地（不要再当缺口）

| 项 | 状态 |
|---|---|
| 七后端 + D-tag 寄存器 DSL（~387 公开/selector 名） | ✅ |
| 产品零 Highway/xsimd；bench 外部 pin 对拍 | ✅ |
| Runtime / exact / `native::` 双轨（ops/random/scan/search 等） | ✅ |
| vqsort 向量 partition（删 HeapSort 保险丝；AVX2 u64 Compress 真 partition） | ✅ 正确性与定宽性能；NEON KV64 见 A1 |
| search/reduction whole-buffer | ✅ |
| mixed-precision dots | ✅ |
| NEON mask 查询 primitive（nibble 路径） | ✅ |
| SVE/RVV 寄存器正确性 + 多 VL/VLEN QEMU | ✅ |
| **portable_missing 可移植原语 backlog** | ✅ **0 项**（原 ~22 已全部补齐；见 simd-25 §阶段 0） |
| **SVE/RVV math 主 HYLE/Highway 产品 TU + Runtime** | ✅ QEMU 多 VL；复数 f32/f64 走真实 scalable body，仅 6 个精确实数 op 保留 scalar |
| **SVE/RVV dtype f16/bf16↔f32 产品 TU + Runtime** | ✅；fp8 故意保持 scalar/可选硬件 |
| **SVE/RVV bitpack 产品 TU + Runtime** | ✅（固定 128-bit logical block，与 NEON layout 一致） |
| **RVV sort 产品 TU（LMUL=1）+ Runtime** | ✅ |
| **SVE sort 产品 TU + Runtime** | ✅ full-VL partition；只有要求 2 的幂 lane 的基础 network 保持 128/256-bit cap |
| **SVE/RVV 精细 feature probe** | ✅ HWCAP2 / riscv_hwprobe；baseline、LTO-off detector |
| **公开 scalable 产品派发门** | ✅ SVE 99×5；RVV base 94×4，Zvbb/Zvbc 各 95×4 |
| 定宽 op 常数因子对齐战（历史矩阵主体） | ✅ 无新 LOSE 证据时不再全表重抠 |

---

## A. 性能真洞（有数字，优先）

| # | 项 | 现状 | 做什么 | 验收 |
|---|---|---|---|---|
| **A1** | **NEON KV64 sort/partial** | ✅ **已收口（2026-07-19）**。根因是 upper-128 compare 用 `InterleaveUpper`，NEON 的通用数组实现被 scalarize，在每 key 热循环形成 `xtn→GPR→dup`；改用已有单指令 `DupOdd` | Apple 同 ISA 5 轮 random/asc partial：HYLE core 22.46–23.28、Highway 22.46–23.57 ns/key；desc/few 与 full sort 交叉验证均持平 | ✅ 达到 ≥~0.98；Windows AVX2 `Lt128Upper` 四副本持平，SSE4/AVX2 正确性全绿 |
| **A2** | （观察）**AVX2 f32×bf16 dot** | 2026-07-19 重构后 WSL 1M 元素四次独立 scorecard：native/Highway **0.974–0.993×**，单行误差约 1–3%，public 行跨 **0.936–1.039×**；仍是误差带/频率敏感的边界项，不构成真 LOSE | 若要重开，只按主 methodology 做 AB/BA、多 replica 符号检验；稳定 <0.98 再动 | 用单遍 nanobench 排名下结论 |
| **A3** | **变宽真机 scorecard** | 仅有 QEMU 正确性 | 有硅时 vs Highway + vs 旧 NEON/scalar fallback | 方法论双序/多轮；稳定 &lt;0.98 逐项狩猎或 best-of |

---

## B. 变宽 SVE/RVV（剩余）

权威施工：[`simd-25-scalable-product-parity.md`](./simd-25-scalable-product-parity.md)。

### B 已完成（主战役）

| # | 项 | 状态 |
|---|---|---|
| **B0** | 语义分类账 v2（分后端 missing，防并集假绿） | ✅ 当前 missing-any **22**；仅 shared generic 9 + selector 13；`portable_missing` **0** |
| **B1** | portable_missing 实现 | ✅ |
| **B2** | math / dtype / bitpack / sort 真实 scalable 产品 TU + Runtime | ✅ 主路径；见下边界 |
| **B3** | RVV sort LMUL=1；SVE full-VL partition + capped base network | ✅ |
| **B4** | SVE2/BF16/I8MM/AES、Zvbb/Zvbc/Zvkned/Zvfh 等精细探测 | ✅ |

### B 未完成

| # | 项 | 现状 | 做什么 | 别做什么 |
|---|---|---|---|---|
| **B5** | 真机性能（= A3） | 未签字 | 同 A3 | 无硅宣称 SVE SOTA |
| **B6** | math 精确标量 per-op | `lgamma/remainder/nextafter` 因正确性合同保留 scalar | 只有 scalable 实现能保持精确合同且有收益时再换 | 为齐 tier 元数据牺牲语义 |
| **B7** | bitpack 多 logical-block 并行 | 每次 128b active | 真机收益证据后再做 | 无证据改格式 |
| **B8** | shared_generic / fixed_model 长尾 | **9 + 0**；shared 已由产品/QEMU 实例化，fixed_model 已清零 | 维持机械闸；出现新消费者再分类 | 为齐名抄 128-block 全家桶 |
| **B9** | extension_gated（含 RVV AES 等） | 语义缺口 **0**；可选扩展只剩产品机会项 | 有扩展、产品合同与收益证据再做独立 feature TU | 无硬件装假快路 |

### 机械参考（语法面，≠ 用户能力）

`simd-scalable-coverage.py` / `simd-op-coverage.py`（回填时）：

| 后端 | local public ops |
|---|---:|
| avx2 | 236 |
| neon | 223 |
| avx512 | 222 |
| sse4 | 219 |
| **sve** | **227** |
| scalar | 202 |
| **rvv** | **227** |

- fixed missing：SVE **22** / RVV **22**；missing-any **22**（shared_generic 9 + selector 13 + fixed_model 0 + extension_gated 0 + portable_missing **0**）
- 产品种子：math/sort/dtype 传递闭包 **0**；bitpack 余 **Xor3** 等个位

---

## C. simd-24 未完成算法面

| # | 项 | 说明 |
|---|---|---|
| **C1** | **`MulHigh` / `MulHighLow`** | ✅ 单一 `MulHigh` 覆盖 i/u8/16/32/64；删公开 `MulHigh32` 特例；`MulHighLow` 返回 low、写 high，widen/64-bit 路径共享 partial；AVX2 首轮 i/u8 0.1–0.15×、i64 0.8× 回退已修至约同速；register/native-only，不进 C ABI/Runtime 表 |
| **C2** | **sort key 扩展** | ✅ `i16/u16` + unsigned `UInt128 {lo,hi}` POD 已覆盖 sort/partial/select 与 scalar/AVX2/AVX512/NEON/SVE/RVV；顺带修复 SVE1 16-bit register-only Compress 和 NEON 16-bit true-partition 丢 key；f16/bf16 仍须先统一 dtype/NaN total-order |
| **C3** | **complex** | ✅ f32 补 `ctan/csinh/ccosh`；`cexp/clog/csqrt/csin/ccos/ctan/csinh/ccosh/ctanh/cpow` 全部覆盖 **f64**；公开边界保持 SoA pointer/count，AVX2/NEON/SVE/RVV 共用 typed D-tag body；Windows 与 SVE/RVV 多宽 QEMU 已过，真机性能待 A3 |

与 A/B 正交。无真机时默认序：**C3 → D1/D6 → D5/D7 → D2**；A1/A3/B5/B7 进入“有硅再签字”队列。

---

## D. 机制残留：双轨 · 垫片 · 动态分发 · 宏

> 概念对照（完善度判断依据，非再开产品线）：
> - **双轨（两条腿）** = Runtime / exact / `native::` 如何进同一 whole-buffer kernel（见 `simd-dispatch-guide.md`）。
> - **垫片** = 定宽 partial 宽度语义迁移（simd-19：`to_q`/`from_q` 等），**不是**派发双轨。
> - **动态分发** = batch 边界选 ISA；单位是 kernel，不是 387 个寄存器 op。
> - **宏** = HYLE **有意不走** Highway `foreach_target` 宏生成后端；自有少量生成/接线宏保留。

### D.0 已够用（勿当半成品重做）

| 面 | 状态 |
|---|---|
| 双轨**语义模型**（pointer/POD 边界、共享 native body、vector 不过界） | ✅ simd-22 |
| fat Runtime 主路径（未知 CPU 选表） | ✅ |
| exact 单 canonical target 产物（普通 API 固定直连） | ✅；允许该 target 的逐-op scalar/lower-tier 依赖 |
| ops/random/scan/search 等生成/表驱动 + 模块手写 dispatcher 同构 | ✅ 主行为 |
| NEON partial 表示 + 垫片迁移 + 热点 native-D | ✅ 债① NEON 侧 |
| 拒绝 Highway 宏体系；手写后端 + `if constexpr` | ✅ 章程收口 |
| 变体-DLL「进库零间接」 | ✅ **有意不做**（batch 粒度无结构性收益） |

### D.1–D.8 机制项现状

| # | 项 | 现状（不够在哪） | 做什么 | 不做/边界 |
|---|---|---|---|---|
| **D1** | **exact 单腿产物机械化** | ✅ `hyle_simd_target_manifest` 同时控制 generated TU、手写产品 TU、Runtime table 与 exact symbol；exact 只含一个 canonical table，不做 CPUID | `hyle_simd_exact_artifact_check` 检查 manifest、build graph、CPU-probe import 和 x86 对象越级指令 | scalar/per-op lower-tier 核可作为所选产品 target 的诚实依赖，不伪称 optional BF16/AVX10.2 |
| **D2** | **x86 partial 健康化 + canary** | ✅ AVX2 widen 显式 XMM half，SSE4 truncate/promote 为寄存器路径；exact-byte/guard-page、half/combine oracle 与 `op_parity_widen_bench` canary 在位 | 编译器升级继续跑 SSE4/AVX2 专项和 widen benchmark；出现稳定 <0.98 再狩猎 | x86 sub-XMM 仍物理占 XMM，这是 ISA 模型，不造假窄类型 |
| **D3** | **AVX-VNNI runtime 档** | ✅ 新增有用户语义的 `i8×i8→i64` batch dot；独立 `AVX2_VNNI`（VEX）与 `AVX512_VNNI`（EVEX）TU，CPUID 后只覆盖该 op，默认/exact AVX2/AVX512 仍不要求 VNNI | `Runtime` v6 的 `ops_module.features` 暴露 `kFeatureAvxVnni` / `kFeatureAvx512Vnni`；fallback、两档 VNNI、NEON dotprod、SVE dot 共用模 2^64 合同；MSVC 明确使用 `_avx_` intrinsic，避免在 Raptor Lake 上误发 EVEX；13900K 真跑，Sapphire Rapids SDE 跑 EVEX 档；WSL 1M 元素单次 VEX/public 对 AVX2 baseline 约 **1.38×**（性能只作 smoke） | 不把 VNNI 升成伪 broad ISA，不污染 baseline |
| **D4** | （可选）**GFNI** | 机会型 | 有 Ice Lake+/SDE 再上；可反哺 ReverseBits 等 | 非硬缺 |
| **D5** | **模块 isa 元数据合同** | ✅ `Module::isa` 明确表示主产品 tier；feature bits 标出 AVX-VNNI/AVX512-VNNI、exact-libm scalar、fp8 scalar、per-op lower tier、SVE capped network、bitpack logical-128 | `Runtime` C++ 诊断表 v6；C ABI 提供只读 snapshot | 不为元数据齐平强行实现全部冷门 scalable |
| **D6** | **dispatcher 生成统一** | ✅ generated 模块由 manifest 过滤；math/dtype/sort/bitpack 的 exact 分支统一用 manifest suffix，fat 分支保留各自必要的 feature/per-op 策略 | 新的机械同构模块继续用 `hyle_add_dispatched_kernel` | 不把不同算法策略强塞成同一阶梯 |
| **D7** | **双轨测试矩阵** | ✅ Windows/WSL fat 367（366 pass + 1 无 BF16 硬件 skip）；exact AVX2 同套 367；x86 scalar/SSE4/AVX2/AVX512 artifact 闸；SVE exact 44×5、RVV exact 44×4；fat scalable 为 SVE 99×5、RVV 94×4 / 扩展 95×4 | 真机到位后补执行/性能，不改正确性结论 | 不为每个 op 造动态函数指针 |
| **D8** | **垫片/helper 层后续** | simd-19 **未建** `neon_bin<QFn,DFn>` 式宽度 helper（实践用垫片+热点分支）；**类型轴**（f16/bf16 lane 指令）到来时垫片兜不住 | 仅当 f16/bf16 后端横切收费时再建类型→intrinsic 收敛层；沿用已实证的工单迁移法 | 现在为空建 helper = 过度抽象；**不要**为「消灭垫片痕迹」重写 270 op |

### D.9 宏：不是「重构做到一半」

| 判断 | 说明 |
|---|---|
| **战略已收口** | 不 vendor Highway `foreach_target` / 宏炸全 target；后端手写可读 |
| **保留的合理宏** | `HYLE_FORCE_INLINE`、API 导出、`vec.h` Masked* 生成、dispatch tier 宏、`HYLE_SIMD_*_HAS_NATIVE_*`、exact/target 开关 |
| ** mechanization 欠账** | 在 **D1/D6**（manifest 生成表与 TU），**不是**「再引入一套宏后端」 |
| **禁止** | 为「宏重构彻底」把后端改成 Highway 风格 re-include |

---

## E. 明确不做

- 为齐名补满 **fixed_model ~20**（无产品依赖 → N/A）
- selector（`*Native`/`*Emulated`）当用户缺口
- 无新 LOSE 时重开全 op 常数战
- Highway：`thread_pool` / `matvec`（compute）/ `hash`（container）/ `image`
- callback 万能 `Transform`/`Foreach*If`
- 无优势的第二套 Copy/Fill
- 产品 vendor Highway；**变体-DLL**；C++ span/vector 官方 facade
- 把「math.isa == kSve」说成每个冷门 math 函数都有 scalable body
- 把 387 寄存器 op 各挂动态函数指针
- 为 partial 在 x86 上造「窄于 xmm 的物理寄存器类型」
- 现在空建宽度 helper 层 / 为消灭垫片而全量重写后端
- 用 Highway 宏体系「补完」所谓宏重构

---

## F. 建议施工顺序（更新后）

```text
1.  A1    NEON KV64 性能狩猎  ✅
2.  A3/B5 变宽真机 scorecard（有硅时）
3.  C1    mul_hi / mul_hilo  ✅
4.  C2    sort key i16/u16/u128（✅）
5.  C3    complex + f64  ✅
6.  B6/B7/B8  按消费与证据穿插
7.  D1+D6 exact 单腿 + dispatcher 生成收敛  ✅
8.  D5+D7 模块 isa 合同与 exact/native 测试闸  ✅
9.  D2    x86 partial 健康化 + canary  ✅（真机性能随常规 scorecard）
10. D3    AVX-VNNI + i8 dot  ✅
11. D4    有产品消费再上
// D8 仅 f16/bf16 类型轴启动时评估
```

算法补缺线 C1–C3、机制线 D1/D2/D3/D5/D6/D7 与 A1 均已收口。A3/B5 与 B7
性能决策挂起到取得对应硅；D4/D8 只在出现产品消费或类型轴时启动。

---

## G. 相对 Highway：还应补什么

### G.1 已齐平或不必再追

| 面 | 判断 |
|---|---|
| 定宽寄存器广度 | 主力齐平；扩名 ROI 低（除 C1/D4） |
| 同 ISA op 常数 | 主体已对齐 |
| whole-buffer 主核（dot/search/bitpack/random/vqsort 主路径） | 已有；A1 的 NEON KV64 性能洞已收口 |
| **SVE/RVV 主产品部署路径** | **主路径已有真实 TU**（正确性 QEMU），SVE partition 已 full-VL；差在真机性能 |
| 派发模型 | 与 Highway dynamic 同档 |
| AI 激活 / dtype 产品 | HYLE 优势面，保持 |

### G.2 仍建议补

| 优先级 | 能力 | 建议 |
|---|---|---|
| ✅ | NEON KV64 性能 | **A1 已收口** |
| P0 | 变宽真机证据 | **A3** |
| P1 | sort key 面 / mul_hilo / complex | **C2 / C1 / C3** |
| P2 | GFNI / 可选扩展产品 TU | **D4 / B9**（VNNI D3 已完成） |

### G.3 刻意不在 simd 补

thread_pool → sched；matvec → compute；hash → container；image / 万能 Transform / 无优势 Copy·Fill。

### G.4 应保持的优势

预编译 batch API、dtype/AI 核、scan 产品面、测量与契约纪律、手写可读后端。

### G.5 相对 Highway 优先级（更新）

```text
1. 变宽真机（A3/B5）          — 部署差从「无核」变为「有核欠证」
2. D4 / B9                     — 指令类与扩展产品核，有消费再上
3. 不再镜像 contrib 其余目录

已收口：NEON KV64（A1）、类型/积木/数学完整度（C1/C2/C3）、AVX-VNNI i8 dot（D3）。
```

### G.6 判定口诀

1. pointer/count/POD 或强化现有 kernel → 可补。
2. detail / 纯 128-block 拼装 / 回调模板 → 默认不补。
3. Highway 更快且复杂度更低 → 借手法，不护短。
4. 缺硬件 → 模拟器正确性；性能要真机。
5. 名字齐平不是 KPI。

---

## H. 完成判据（能力线阶段性收口）

可宣告「算法核告一段落」（再交 close-01）当且仅当：

1. **A1** 已平或书面墙；
2. **B2 主路径**保持无 hard clamp（✅ 今日）；math 冷门 fallback 合同写清（**D5**）；
3. **B3** full-VL partition + capped base network 保持通过正式多 VL 门（✅ 今日）；
4. **A3** 至少一档真机 scorecard，或明示「仅 QEMU 正确性、无性能主张」；
5. **C1–C3** 落地或有证据拒绝；
6. 稳定 LOSE 不进 SOTA 宣传；
7. 387-name 扩面冻结（C1/D4 类例外）。

**机制线可后置、但不应假装已完**（与算法收口可分两阶段签字）：

8. **D1** exact 单腿可链、无越级指令闸（✅）；
9. **D2** x86 partial canary 在位（✅）；
10. **D7** exact/native 关键路径有自动化闸（✅；真机性能另算）。

---

## I. 维护

- 落地一项：改 ✅，回指 simd-25/commit；删「将做」口吻。
- 跑 `simd-scalable-coverage.py` / `simd-op-coverage.py` 更新机械表，勿手改过期数字。
- 与 gap-inventory 冲突时：**以代码 + 更新后的 simd-25 为准**，再改本文。
- 禁止把「主路径有 SVE TU」写成「全 VL / 全 math / 真机 SOTA」。
