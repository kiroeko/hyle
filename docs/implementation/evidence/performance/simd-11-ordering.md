# SIMD-11 ordering 有界性能证据

> verdict：`W1-WIN / P1_ENGINEERING / 5% / NO MATERIAL LOSE`。本页只签署 f32 ordinary
> full sort 4,096/1,048,576、ordinary partial sort 4,096/k=10，以及 scratch u32 radix 与
> stable f32 argsort 1,048,576 的冻结合同；它不是全部 ordering 类型、输入分布、ISA、平台或 Formal
> SOTA 证书。

## 被签署的命题

最终产品候选是 commit `6233fde58a21173971c41da144b66f4fdd4719b9`（完整 ordering 产品面起于
`9d6e2767`），最终仪器是 commit `14d6bcb4`。锁定 Highway 是
`f503c485cec6b4f2f596fa60d83484797c9eeb5c`。候选使用普通公开 HYLE API，经唯一 Runtime 表一次缓存
dispatch 到 shipping AVX2 body；peer 在独立 AVX2 TU 中直接实例化当前锁定 Highway 的同类型、同顺序、
同 NaN、同 in-place/no-allocation 合同。

资格后的第一次 P1 暴露了真实算法缺口：HYLE 对 4,096/k=10 先做向量 quickselect 再排序 prefix，约慢
Highway 成熟 heap partial-sort 4 倍。最终产品没有恢复历史上最坏 `O(n*k)` 的 streaming 插入路线，也没有
加入 4,096/10 benchmark 阈值；当请求 prefix 能完整落入 VQSort 已有 sorting-network base-case 容量时，
复用同源 in-place heap selection，复杂度为 `O(n log k)`、额外空间 `O(1)`，更大 prefix 继续使用向量化
average-`O(n)` select 后排序。dependency port hash 随最终内容机械更新。

radix/argsort 的稳定性、NaN/signed-zero 与 caller workspace 合同没有同合同 Highway/ggml peer，因此只签
独立 A/B repeatability 和绝对吞吐，不产生相对 SOTA verdict。

## 机器、构建与执行合同

- machine：登记的 `W1-WIN`，Intel Core i9-13900K，性能逻辑核 2，core class 1；pin 与 high priority
  四次均成功；
- source：干净 detached worktree `14d6bcb4`；build 为 Windows 11、Ninja Release、MSVC
  19.50.35728、x64/AVX2，target `hyle_ordering_parity_bench`；
- input accounting：每次 mutating lambda 消费一份独立、预分配、预填充 buffer；allocation、输入复制、
  初始化和 page fault 均在计时外；
- timer floor：4K full/partial 每个 replica 使用 31 epochs，显式 epoch iterations 为 64/16；1M
  radix/argsort 使用 5 epochs、4 iterations；1M full sort 使用 5 epochs、1 iteration。框架因此知道真实
  迭代数，而不是把 lambda 内隐藏 batch 当成一次测量；
- shared parity：每侧四个 code-placement replica、两个 layout bank、两个执行顺序各两次；A/A anchor、
  正反序、输入唯一性和 `scripts/op-parity-report.py --gate 0.05 --certify` 全部适用；ratio 大于 1 表示
  HYLE 更快。

## 结果

四份最终日志均无 nanobench instability marker。A/A `Add f32 (4096)` 的归一化中心为 `1.000`、95% CI
`[0.995, 1.004]`，calibration、metadata、layout、输入唯一性和执行 audit 全部 `PASS`。四条可比较行得到
3 `WIN`、1 `TIE`、0 `LOSE/INCONCLUSIVE`：

| workload | paired ratio | 95% CI | verdict |
|---|---:|---:|---|
| f32 partial sort 4,096/k=10 | 1.147 | [1.108, 1.185] | WIN |
| f32 full sort 4,096 | 5.950 | [5.883, 6.052] | WIN |
| f32 full sort 1,048,576 | 7.463 | [7.374, 7.556] | WIN |
| A/A Add f32 4,096 | 1.000 | [0.995, 1.004] | TIE |

不同合同的 informational rows 只比较同一 HYLE body 的 A/B repeatability：scratch radix u32 的中心为
`0.996`、CI `[0.980, 1.015]`；stable argsort f32 为 `1.003`、CI `[0.990, 1.016]`。它们支持仪器重复性
与正常绝对吞吐记录，不支持相对 Highway/ggml 排名。

## 原始证据与重放

- [HWY-first run 1](../../../../evidence/performance/simd-11-ordering-hwy-first-1.txt)，SHA-256
  `b94be3eeac19ebb9f32e84a86dc8d63adc417d4dfe53e0c6a6da42298bca36b5`；
- [HWY-first run 2](../../../../evidence/performance/simd-11-ordering-hwy-first-2.txt)，SHA-256
  `f9d87d29a6e0764e274d75360be7a14e5a1c99a8303f8ed1e998cad8e0f7116d`；
- [HYLE-first run 1](../../../../evidence/performance/simd-11-ordering-hyle-first-1.txt)，SHA-256
  `c8b519442b81557f084e3abe58de497225c29a26fde8129f0d115255109f8e3f`；
- [HYLE-first run 2](../../../../evidence/performance/simd-11-ordering-hyle-first-2.txt)，SHA-256
  `62afe57ef6415cd953268531d6e4f002a53734fe1bfaccbdde35c4e7cb49e129`。

```powershell
python scripts/op-parity-report.py --gate 0.05 --certify `
  evidence/performance/simd-11-ordering-hyle-first-1.txt `
  evidence/performance/simd-11-ordering-hyle-first-2.txt `
  evidence/performance/simd-11-ordering-hwy-first-1.txt `
  evidence/performance/simd-11-ordering-hwy-first-2.txt
```

## 未签署的内容

- 其他 scalar/KV/U128 类型、升降序、duplicate、tail、NaN payload、signed zero、overlap/OOM/index boundary
  由 correctness 与复杂度合同签署，不从四条代表性计时外推排名；
- SVE/RVV 只有多 VL/VLEN correctness，AVX-512 只有 exact artifact；QEMU 不用于 timing，macOS 也不由
  本页签固定性能；
- radix/argsort 没有同合同 peer，因此 verdict 保持 `NOT_COMPARABLE`，不能把 A/B repeatability 写成 SOTA；
- parallel/distributed/external/tensor/strided ordering 与 GPU 排序不属于 SIMD-11；真实 consumer 若在相同
  合同、ISA 和资格机上复现超过 5% 的 material regression，才触发本结论重开。
