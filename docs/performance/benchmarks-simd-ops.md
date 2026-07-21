# simd op 对拍证书（vs Highway）

> 本页是 [performance 总账](./README.md) 的 **simd 逐 op** 详节。  
> **全量矩阵（最新、最全）**：[../milestones/simd-orderswap-matrix.md](../milestones/simd-orderswap-matrix.md)（2026-07-16 校准仪器 v4；**勿手抄旧快照**）。  
> **产品 kernel / 数学核 / dtype**（另一口径）：[benchmarks-simd.md](./benchmarks-simd.md)。  
> **怎么跑 / 怎么判真输**：[simd-op-benchmark-methodology.md](./simd-op-benchmark-methodology.md)。  
> **覆盖计数（唯一机械口径）**：`python scripts/simd-op-coverage.py`（现跑；禁止对话里旧 16/30/41/303 手加）。

## 0. 读法（防散）

| 问题 | 看哪里 |
|---|---|
| 还有没有 **明确 LOSE** Highway？ | 本页 §1 + 矩阵「真输清单」+ 全量表 LOSE 列 |
| 某个 op 的比值 / CI / 修史？ | **只**打开矩阵全量表 / ratio&lt;0.98 全扫 |
| 覆盖了多少 op？ | `simd-op-coverage.py`（§2） |
| dot / exp / dtype 产品路径？ | [benchmarks-simd.md](./benchmarks-simd.md) |
| 新增 op 怎么测？ | [simd-op-benchmark-methodology.md](./simd-op-benchmark-methodology.md) |

**口径铁律**：

1. 比值 = HYLE/Highway（**&gt;1 = HYLE 快**）。  
2. 只有整个 CI 落进 0.97–1.03 才叫 **TIE**。  
3. **INCONCLUSIVE (`?`) ≠ TIE，也不算输**——证据不足，禁止偷换。  
4. 锚点校准未 PASS 的整表 **作废**。  
5. 矩阵正文与历史段落可能并存；**判决以全量表 + 真输清单为准**。

## 1. 三平台总账（矩阵 2026-07-16）

九报告 union：`303` source-benched op 名 / `320` benchmark case（Win/WSL）；NEON `308` case（缺 12 = 单 block 不适用的 x86 multi-block 形，非漏跑）。

| 平台 | 编译器 / ISA | case | **LOSE** | WIN | TIE | INCONCLUSIVE |
|---|---|---:|---:|---:|---:|---:|
| Windows | MSVC 19.50 / AVX2 | 320 | **0** | 39 | 103 | 178 |
| WSL | GCC / AVX2 | 320 | **0** | 41 | 38 | 241 |
| Mac | AppleClang / NEON (M4) | 308 | **0** | 7 | 54 | 247 |

### 明确真输

**MSVC 0 / GCC 0 / NEON 0。**

历史 LOSE（Mod、BroadcastLane、FindLastTrue、NEON store+reload 族、AESKeyGenAssist、PromoteEven/Odd、TableLookupLanes、…）均在矩阵「真输清单」写明 **→ 已清零** 与修法。当前 ratio&lt;0.98 行在矩阵全扫中一律标为 **INCONCLUSIVE + 检查结论**（同指令 / placement / CI 跨等价带），**没有**未处理的算法级回归。

### 不进本分母

| 类 | 说明 |
|---|---|
| 算法 composite | softmax / log_softmax 等 → 矩阵「算法层备忘」或 complexity 文档 |
| AES/CLMul 部分 | 多在 `crypto_bench`，不在校准 op_parity 分母 |
| SVE/RVV 真机吞吐 | 仅有 QEMU 正确性时 **不** 声称对 Highway 性能平 |
| 无 Highway 对手的 HYLE 独有 op | 矩阵另节；不能偷换成「已证 SOTA」 |

## 2. 覆盖树（2026-07-21 现跑 `simd-op-coverage.py`）

```text
387  公开 op 名（8 后端头去重）
 └─ 344  可测全集
      ├─ 314  有独立 source benchmark（91.3%）
      └─  30  不独立测，manifest 已分类
           (alias / synthetic / paired_info / algorithm)
           truly unclassified = 0
```

- **314/344** = 可测且有 bench；**30** = 有意不独立测，**不是**「还欠 30 个 nanobench」。  
- 矩阵文首计数树与脚本应对齐；冲突时 **以现跑脚本为准**，再回填矩阵文首。  
- `simd-op-manifest.json` + CI：拒绝新增未分类项、缺 row 的 native、误进分母的 alias/synthetic。

## 3. 方法论摘要（细节见 methodology 页）

- 钉核 / 高优先级（Mac：USER_INTERACTIVE + 更长 epoch）；双序（HYLE 先 / Highway 先）。  
- 每侧多 replica + stagger，抗 ICF / 代码放置 / 4K 别名。  
- **锚点闸门**：两侧同指令的 op 必须读 ~1.00，否则整表拒绝。  
- 定罪 LOSE：须过 methodology 证据阶梯（双序、objdump 形状、控制变量），禁止单次 ratio&lt;1 定死。

## 4. 复现

```bash
# 覆盖
python scripts/simd-op-coverage.py
python scripts/simd-op-coverage.py --check

# 每平台构建 hyle_op_parity{,_full,_widen}_bench_<isa>
# 双序各跑（Mac 建议 4 遍），再：
python scripts/op-parity-report.py --certify h1.txt h2.txt g1.txt g2.txt
# Mac 常用更宽 gate：
python scripts/op-parity-report.py --gate 0.06 --certify mac_h*.txt mac_g*.txt
```

完整 case 表、ratio&lt;0.98 全扫、修史、算法层备忘 → **只维护**  
[../milestones/simd-orderswap-matrix.md](../milestones/simd-orderswap-matrix.md)。
