# SIMD-09 search/scan 有界性能证据

> verdict：`W1-WIN / P1_ENGINEERING / 5% / NO MATERIAL LOSE`。本页只签署 finite-normal f32
> count/argmax 与 out-of-place u32 inclusive scan 的 4,096/1,048,576 元素固定 AVX2 机制；它不是
> 整个 SIMD、所有 search/scan 类型、跨 ISA、跨平台或 Formal SOTA 证书。

## 被签署的命题

候选是 commit `f29a34bc6ad2e32106ab9851d486d19ce2ea1cd4`（产品实现进入
`606bb6295827985ab4dd13a98c1760a7afb36cd8`），锁定 Highway 是
`f503c485cec6b4f2f596fa60d83484797c9eeb5c`。instrument 在一个固定 AVX2 TU 中实例化 exact shipping
target body；count/argmax peer 直接调用 Highway contrib，scan peer 独立拼写相同的 Highway
load/add/slide/store composition。双方使用相同 pointer/count、输入域、tail、输出流量、alias、ISA 与资源
计费，均不经过 public runtime façade。

count/argmax 输入只含 finite normal，argmax 有唯一最大值；NaN propagation 与 first-tie 由 correctness
gate 验证，不拿不同语义的 ggml last-tie loop冒充 peer。scan 双方均从不变的 caller-owned input 写向同一
独立 output，不把 refill、allocation 或初始化计入计时。4,096 是 resident row，1,048,576 是 streaming
row；count 的 epoch 分别为 262,144/8,192，其余为 8,192/512。初次资格运行按 nanobench 提示延长 count
epoch 后才生成本页四份证据，最终日志没有 instability/error marker。

## 机器、构建与执行合同

- machine：登记的 `W1-WIN`，Intel Core i9-13900K，性能逻辑核 2，core class 1；pin 与 high priority
  均成功；
- build：Windows 11、Ninja Release、MSVC 19.50.35728、x64、AVX2，target
  `hyle_search_scan_parity_bench`；
- instrument：每侧四个 placement replica、两个 layout bank、两个执行顺序各两次；
  `scripts/op-parity-report.py --gate 0.05 --certify`；
- accounting：candidate/direct peer 在同 TU；输入、输出和初始化在计时区外；每次完整 row body 在计时区
  内；ratio 大于 1 表示 candidate 更快。

## 结果

A/A calibration、四份输入唯一性、正反序、双 layout、pin/priority audit 与 certification 均为 `PASS`。
含 calibration 的 7 行得到 5 `TIE`、2 `INCONCLUSIVE`、0 `LOSE`；其中 A/A 自身因 placement uncertainty
为 `INCONCLUSIVE`，产品行只有 1M argmax 保守为 `INCONCLUSIVE`。六条产品中心比为：count
`0.990/0.990`、argmax `0.985/0.977`、scan `0.988/0.991`（4K/1M）。没有超过预声明 5% margin 的
material regression，且不能把这些小于 3% 的表观差额写成确定输赢。

public/body informational center 为：count `0.955/0.986`、argmax `0.989/0.999`、scan
`0.991/0.989`（4K/1M）。正常大批次未检测到 material integration loss；4K count 的约 4.5% lower
throughput 是极短 body 上一次 whole-buffer function-pointer/ABI boundary 的可见成本。它没有第二次
dispatch、copy 或 allocation；tiny/fused kernel 应用 source fixed/fat 路线融合，不为消除该边界复制一套
register API。

## 原始证据与重放

- [HWY-first run 1](../../../../evidence/performance/simd-09-search-scan-hwy-first-1.txt)，SHA-256
  `dff7e8a777a797bfe5063bddbd71d54b0b349dd58275c75128eb2fbd46550c76`；
- [HWY-first run 2](../../../../evidence/performance/simd-09-search-scan-hwy-first-2.txt)，SHA-256
  `8bc759ca058c65d218720399d1b2c52f6b37a544d2e4c36b8a96bc6175f92bb5`；
- [HYLE-first run 1](../../../../evidence/performance/simd-09-search-scan-hyle-first-1.txt)，SHA-256
  `6c01da97f202e7a92ca937451d0bc720e3dbfd14ebf2eabded2f6c898a17cf61`；
- [HYLE-first run 2](../../../../evidence/performance/simd-09-search-scan-hyle-first-2.txt)，SHA-256
  `d3aab6abf612a3263ed13acf8331c2251f857a072f32eec14464c652b2a50093`。

```powershell
python scripts/op-parity-report.py --gate 0.05 --certify `
  evidence/performance/simd-09-search-scan-hwy-first-1.txt `
  evidence/performance/simd-09-search-scan-hwy-first-2.txt `
  evidence/performance/simd-09-search-scan-hyle-first-1.txt `
  evidence/performance/simd-09-search-scan-hyle-first-2.txt
```

## 未签署的内容

- 其他八种整数/浮点类型、find/value extrema/minmax/sortedness、exclusive/init 与 signed modular scan 由
  complexity、property/differential 和多 ISA correctness 签署，不从六行计时外推性能排名；
- NaN、Inf、signed zero、first tie/first NaN、empty、exact in-place 和 tail 不属于 finite-normal P1 行，
  但属于产品 correctness 合同；
- SVE/RVV 本轮只有多 VL/VLEN correctness，没有真硅性能；macOS/WSL 也不由本页签固定 timing；
- custom predicate/op、parallel/segmented/tensor/GPU scan、top-k/order statistics 分属标准库、source expert、
  Graph/Compute、accelerator 或 Ordering；
- 若真实 consumer 在同合同、同 ISA、资格化真机上复现超过 5% 的 material regression，或循环内出现
  第二次 dispatch、heap、lock、refill/整段 copy，才触发本结论重开。
