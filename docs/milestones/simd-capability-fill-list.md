# SIMD 能力补缺清单（只补能力/性能）

> **状态**：当前真相账本（2026-07-17 晚间回填，对齐 simd-25 + `simd-scalable-coverage.py`）。
> **范围**：寄存器/算法/产品核/性能洞与相对 Highway 的能力差。
> **不在本文**：close-01 C ABI、推广合同文案、用法包装（另线）。
> **姊妹文档**：
> - 施工：[`simd-24-foundation-completion.md`](./simd-24-foundation-completion.md)、[`simd-25-scalable-product-parity.md`](./simd-25-scalable-product-parity.md)
> - 差集审计：[`simd-gap-inventory.md`](./simd-gap-inventory.md)
> - 复杂度轴：[`simd-complexity-opportunities.md`](./simd-complexity-opportunities.md)
> - 变宽底账：`scripts/simd-scalable-coverage.py` + `scripts/simd-scalable-op-manifest.json`
> - 覆盖率：`scripts/simd-op-coverage.py`

**一句话**：定宽面与主 whole-buffer 核已立；**SVE/RVV 主产品核（math/dtype/bitpack/sort）已有真实 TU + QEMU 门，SVE partition 已吃满物理 VL，精细 feature probe 已落地**，不再是「整模块 hard clamp」。还缺的是 **NEON KV64 性能**、**变宽真机签字**、**simd-24 尾部算法面（C）**，以及 **双轨产物/垫片收尾/分发机械化（D）**。不要再为齐 387 名字抠常数。

---

## 0. 已落地（不要再当缺口）

| 项 | 状态 |
|---|---|
| 七后端 + D-tag 寄存器 DSL（~387 公开/selector 名） | ✅ |
| 产品零 Highway/xsimd；bench 外部 pin 对拍 | ✅ |
| Runtime / exact / `native::` 双轨（ops/random/scan/search 等） | ✅ |
| vqsort 向量 partition（删 HeapSort 保险丝；AVX2 u64 Compress 真 partition） | ✅ 正确性；性能见 A1 |
| search/reduction whole-buffer | ✅ |
| mixed-precision dots | ✅ |
| NEON mask 查询 primitive（nibble 路径） | ✅ |
| SVE/RVV 寄存器正确性 + 多 VL/VLEN QEMU | ✅ |
| **portable_missing 可移植原语 backlog** | ✅ **0 项**（原 ~22 已全部补齐；见 simd-25 §阶段 0） |
| **SVE/RVV math 主 HYLE/Highway 产品 TU + Runtime** | ✅ QEMU 多 VL；gamma/complex 等仍可 per-op fallback |
| **SVE/RVV dtype f16/bf16↔f32 产品 TU + Runtime** | ✅；fp8 故意保持 scalar/可选硬件 |
| **SVE/RVV bitpack 产品 TU + Runtime** | ✅（固定 128-bit logical block，与 NEON layout 一致） |
| **RVV sort 产品 TU（LMUL=1）+ Runtime** | ✅ |
| **SVE sort 产品 TU + Runtime** | ✅ full-VL partition；只有要求 2 的幂 lane 的基础 network 保持 128/256-bit cap |
| **SVE/RVV 精细 feature probe** | ✅ HWCAP2 / riscv_hwprobe；baseline、LTO-off detector |
| **公开 scalable 产品派发门** | ✅ SVE 94×5；RVV base 90×4，Zvbb/Zvbc 各 91×4 |
| 定宽 op 常数因子对齐战（历史矩阵主体） | ✅ 无新 LOSE 证据时不再全表重抠 |

---

## A. 性能真洞（有数字，优先）

| # | 项 | 现状 | 做什么 | 验收 |
|---|---|---|---|---|
| **A1** | **NEON KV64 sort/partial** | 正确；Apple 上约 **慢 Highway 27%**；非派发问题 | 对拍 Highway 同路径 network/组合 codegen；禁止改合同凑数 | Mac 同 ISA 认证 bench ≥~0.98，或书面 ISA/算法墙 |
| **A2** | （观察）**AVX2 f32×bf16 dot** | 诊断 ~0.98–0.99×，误差带重叠 | 多轮符号检验；稳定 LOSE 再动 | 非稳定输可不动 |
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
| **C3** | **complex** | f32 `ctan/csinh/ccosh`；现有 c* 扩 **f64**；SoA real/imag |

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
| exact **调用**绑到固定 ISA（固定调用腿） | ✅ |
| ops/random/scan/search 等生成/表驱动 + 模块手写 dispatcher 同构 | ✅ 主行为 |
| NEON partial 表示 + 垫片迁移 + 热点 native-D | ✅ 债① NEON 侧 |
| 拒绝 Highway 宏体系；手写后端 + `if constexpr` | ✅ 章程收口 |
| 变体-DLL「进库零间接」 | ✅ **有意不做**（batch 粒度无结构性收益） |

### D.1–D.8 不够完善、应补的机制项

| # | 项 | 现状（不够在哪） | 做什么 | 不做/边界 |
|---|---|---|---|---|
| **D1** | **exact 单腿产物机械化** | exact 只固定**调用腿**；dispatch/runtime 表与 sort/dtype/math TU 列表仍按 **fat 符号集**组织；删非目标 TU → link 未解析符号 | **同一 target manifest** 生成：source list、声明、Runtime/dispatch 表、exact binding；每 exact ISA 验证链接符号 + 无越级指令 | 不改热路径多一次分支；不阻塞 fat Runtime 正确性 |
| **D2** | **x86 partial 健康化 + canary** | NEON 垫片债已收；x86 partial 多靠编译器「碰巧」低半路径 → **脆弱 TIE**，升级可能回归 | 审计 Rebind/partial 热点 → 显式低半指令；bench **canary** 防静默回退；文档写清 partial 合同 | 不做 x86 上「比 xmm 更窄寄存器」的假表示改造 |
| **D3** | **AVX-VNNI runtime 档** | 有 `cpu_features().avx_vnni` + 编译期 `#if`；默认 avx2 TU **不**带 `-mavxvnni` | 独立 `AVX2_VNNI`（或等价）TU + Runtime/表选择；禁止编进无 VNNI 的 baseline | 无消费可继续挂起 |
| **D4** | （可选）**GFNI** | 机会型 | 有 Ice Lake+/SDE 再上；可反哺 ReverseBits 等 | 非硬缺 |
| **D5** | **模块 isa 元数据合同** | 如 `Runtime::math.isa == kSve` 只表示**主** HYLE/Highway tier，**≠** 每个冷门 op（gamma/complex…）都有 scalable body；dtype/sort/bitpack 亦有边界（capped、logical block、fp8 scalar） | 文档/诊断 API 写清「模块 tier vs per-op fallback」；禁止对外一句「已上 SVE」掩盖 fallback | 不为元数据齐平强行实现全部冷门 scalable |
| **D6** | **dispatcher 生成不统一** | ops/search 等可 `hyle_add_dispatched_kernel`；math/dtype/sort/bitpack **手写** tier 宏/阶梯，与 exact 单腿问题耦合 | 与 **D1** 同一 manifest/生成器收敛；新模块默认走生成路径 | 不强制一夜重写所有手写 dispatcher 若行为已正确 |
| **D7** | **双轨测试矩阵缺口** | scalable exact 已在 QEMU 门检查不引用 Runtime/CPU probe；定宽 exact 各 ISA「仅含目标符号、无越级」自动化仍不足；`native::` 与 public 同合同对拍未处处强制 | CI：exact-AVX2/NEON/… 符号/指令闸；关键 kernel public vs fixed/native 诊断 bench | 不为每个 op 造动态函数指针 |
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
1.  A1    NEON KV64 性能狩猎
2.  A3/B5 变宽真机 scorecard（有硅时）
3.  C1    mul_hi / mul_hilo  ✅
4.  C2    sort key i16/u16/u128（✅）
5.  C3    complex + f64
6.  B6/B7/B8  按消费与证据穿插
7.  D1+D6 exact 单腿 + dispatcher 生成收敛（可与算法并行）
8.  D5+D7 模块 isa 合同与 exact/native 测试闸
9.  D2    x86 partial canary（可全程并行）
10. D3/D4 有消费再上
// D8 仅 f16/bf16 类型轴启动时评估
```

主工单线：**simd-24 的 C3**；机制线 **D1/D2/D5/D6/D7** 可在无真机环境继续。A1、A3/B5 与 B7 性能决策挂起到取得对应硅。

---

## G. 相对 Highway：还应补什么

### G.1 已齐平或不必再追

| 面 | 判断 |
|---|---|
| 定宽寄存器广度 | 主力齐平；扩名 ROI 低（除 C1/D4） |
| 同 ISA op 常数 | 主体已对齐 |
| whole-buffer 主核（dot/search/bitpack/random/vqsort 主路径） | 已有；A1 为性能洞 |
| **SVE/RVV 主产品部署路径** | **主路径已有真实 TU**（正确性 QEMU），SVE partition 已 full-VL；差在真机性能 |
| 派发模型 | 与 Highway dynamic 同档 |
| AI 激活 / dtype 产品 | HYLE 优势面，保持 |

### G.2 仍建议补

| 优先级 | 能力 | 建议 |
|---|---|---|
| P0 | NEON KV64 性能 | **A1** |
| P0 | 变宽真机证据 | **A3** |
| P1 | sort key 面 / mul_hilo / complex | **C2 / C1 / C3** |
| P2 | VNNI runtime / GFNI / 可选扩展产品 TU | **D3 / D4 / B9** |

### G.3 刻意不在 simd 补

thread_pool → sched；matvec → compute；hash → container；image / 万能 Transform / 无优势 Copy·Fill。

### G.4 应保持的优势

预编译 batch API、dtype/AI 核、scan 产品面、测量与契约纪律、手写可读后端。

### G.5 相对 Highway 优先级（更新）

```text
1. NEON KV64（A1）              — 同 ISA 真 LOSE
2. 变宽真机（A3/B5）          — 部署差从「无核」变为「有核欠证」
3. C1/C2/C3                    — 类型/积木/数学完整度
4. D3/D4 / B9                  — 指令类与扩展产品核，有消费再上
5. 不再镜像 contrib 其余目录
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

8. **D1** exact 单腿可链、无越级指令闸；或书面「仅固定调用腿、产物仍 fat」；
9. **D2** x86 partial canary 在位；
10. **D7** exact/native 关键路径有自动化闸（不必全矩阵一次做完）。

---

## I. 维护

- 落地一项：改 ✅，回指 simd-25/commit；删「将做」口吻。
- 跑 `simd-scalable-coverage.py` / `simd-op-coverage.py` 更新机械表，勿手改过期数字。
- 与 gap-inventory 冲突时：**以代码 + 更新后的 simd-25 为准**，再改本文。
- 禁止把「主路径有 SVE TU」写成「全 VL / 全 math / 真机 SOTA」。
