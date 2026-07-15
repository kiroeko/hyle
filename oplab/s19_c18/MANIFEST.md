# simd-19 C18 全量 NEON 采集清单

- 主机: Mac Apple Silicon (`kiro@172.22.10.198`)
- 时间: 2026-07-15 19:17–20:15 CST
- 二进制 mtime: 2026-07-15 19:16 (`hyle_op_parity_{bench,full,widen}_bench_neon`)
- gtest: `hyle_simd_neon_test` **276/276 PASSED**（13 DISABLED 名仍在，also_run 此前 23/23 绿）
- 编排: `run_c18.sh` 串行 18 跑（3 TU × 双序 × 3），防 jetsam 并发
- 换序 env: **`HYLE_BENCH_HWYFIRST=1`**（无第二下划线）；自检=首个 `#` 标记行序

## 完整性（行数基线）

| TU | 二进制 | 期望行数 | 6 文件结果 |
|---|---|---:|---|
| parity | `hyle_op_parity_bench_neon` | **769** | 6/6 = 769 |
| full | `hyle_op_parity_full_bench_neon` | **2293** | 6/6 = 2293 |
| widen | `hyle_op_parity_widen_bench_neon` | **308** | 6/6 = 308 |

注: 手册 §3-4 旧写 parity `wc -l=2227` 已过时；本轮实测完整基线见上表。

## 文件

命名: `s19_<tu>_<order>_<n>.txt`，order ∈ {`normal`,`hwyfirst`}，n ∈ {1,2,3}。

全部 18 份 + `c18_runner.log` + `run_c18.sh` 同目录留存。

## 换序自检

18/18 `ORDER_OK`（normal 首标 HYLE#；hwyfirst 首标 HWY#）。

## PromoteMaskTo i16→i32（只记录不判定）

六遍 parity 副本对 relative 均在 **~99.5%–100.1%**，两侧 ~0.08 ns/op / ~12.65–12.72 G op/s。
判定归高能力方 / C19。

## 判定权限

G4 数值只记录；全表判读与矩阵回写 = **C19 / 高能力方**。
