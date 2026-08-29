# SIMD-08 activation 有界性能证据

> verdict：`W1-WIN / P1_ENGINEERING / 5% / NO MATERIAL LOSE`。本页只签署 f32 exp、
> tanh-GELU、SiLU 与普通连续 row softmax 的固定 AVX2 实现机制；它不是整个 SIMD、所有
> activation、跨 ISA、跨平台或 Formal SOTA 证书。

## 被签署的命题

产品候选是 commit `29b248fc13b9a5061813ff682a42f1d911da5c2c`，锁定 Highway 是
`f503c485cec6b4f2f596fa60d83484797c9eeb5c`。shipping target helper 与独立重写的 direct Highway
composition 在同一个 TU、同一个 AVX2 target、相同 pointer/count、accuracy、alias、tail 和资源合同下
运行；双方都不经过 public runtime facade。这样比较的是 HYLE 保留的 semantic glue，而不是 DLL
边界、第二套 dispatcher 或不同近似公式。

代表行是 exp、tanh-GELU、SiLU 和稳定 softmax，各使用 4,096 元素 resident row 与 1,048,576 元素
streaming row。resident epoch 为 8,192，streaming epoch 为 32。工具还加入同一 direct Highway add
body 对自身的 A/A 校准，并把 public facade 对 selected target body 的结果单列为 informational，不混进
算法 parity verdict。

## 机器、构建与执行合同

- machine：登记的 `W1-WIN`，Intel Core i9-13900K，性能逻辑核 2，core class 1；pin 与
  above-normal priority 均成功；
- build：Windows 11、Ninja Release、MSVC 19.50.35717、x64、AVX2，benchmark target
  `hyle_activation_bench`；
- instrument：四个 placement replica、两个 layout bank、两个执行顺序各两次；
  `scripts/op-parity-report.py --gate 0.05 --certify`；
- accounting：输入、输出分配和初始化在计时区外；整个 row body 在计时区内；无 heap、lock、整段
  临时转换或循环内 dispatch；ratio 大于 1 表示 candidate 更快。

## 结果

校准与 certification 均为 `PASS`。verifier 对含 A/A anchor 的 9 行给出 6 `TIE`、3
`INCONCLUSIVE`、0 `LOSE`；全部 paired center 位于约 0--2.5%，没有超过预声明 5% material margin 的
落后。三条 `INCONCLUSIVE` 都是 candidate 的小幅表观领先，但区间或 placement 不允许签署正式胜利；
它们不能改写为 WIN，也不授权根据地址布局改产品。

public/body informational center 在八个 size/family 行上为 0.999--1.005：SiLU 为 1.002/1.004，exp 为
0.999/1.002，softmax 为 1.005/1.001，tanh-GELU 为 1.001/1.005（1M/4096）。因此在本票的普通
whole-buffer 尺寸上，没有检测到可归因于 HYLE public 中间层的 material integration tax。结合
SIMD-07 已记录的 fixed/source 相同指令序列，当前可签的结构结论是：target TU 内接近原样 Highway，
ordinary/fat 路径只承担一次 buffer-boundary cached indirect call；tiny 或模型融合 kernel 仍应由源码
consumer/Compute 融合。

## 原始证据与重放

- [HWY-first run 1](../../../../evidence/performance/simd-08-activation-hwy-first-1.txt)，SHA-256
  `fe2af824b3e7bee83bea05e5db6837179b055bef1c792b602818317eb22dcefd`；
- [HWY-first run 2](../../../../evidence/performance/simd-08-activation-hwy-first-2.txt)，SHA-256
  `cbc9e5f8f1e287aaa1d889577b509c9bfbbaba1b6e7587032cb57e664c39d86b`；
- [HYLE-first run 1](../../../../evidence/performance/simd-08-activation-hyle-first-1.txt)，SHA-256
  `5f8abfc19a890f336be51682ff10491389a1ed37392082578666b62dc61384a3`；
- [HYLE-first run 2](../../../../evidence/performance/simd-08-activation-hyle-first-2.txt)，SHA-256
  `27a409569f0b89f217f6e5e1f209e3ff98c28853a3c903363e0e6a4bec0e4ca9`。

```powershell
python scripts/op-parity-report.py --gate 0.05 --certify `
  evidence/performance/simd-08-activation-hwy-first-1.txt `
  evidence/performance/simd-08-activation-hwy-first-2.txt `
  evidence/performance/simd-08-activation-hyle-first-1.txt `
  evidence/performance/simd-08-activation-hyle-first-2.txt
```

## 未签署的内容

- ReLU/leaky-ReLU/ELU/hard-sigmoid/hard-swish 由复杂度、直接 Highway lowering 与 correctness gate
  签署，不需要为每个 spelling 制造性能行；exact GELU、sigmoid 与 log-softmax 也不由代表行外推 WIN；
- SVE/RVV 本轮只有多 VL/VLEN correctness，没有真硅性能；macOS 和 WSL 只签 correctness/delivery；
- attention mask/scale/bias fusion、online/blocked softmax、backward、半精度 activation 和 tensor operator
  属于 Compute 或后续真实 consumer，不由本票承诺；
- 若真实 consumer 在同合同、同 ISA、资格化真机上复现超过 5% 的 material regression，或热循环出现
  第二次 dispatch、heap、lock 或整段临时 buffer，才触发本结论重开。
