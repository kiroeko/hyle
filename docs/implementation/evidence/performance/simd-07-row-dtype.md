# SIMD-07 row/dtype 有界性能证据

> verdict：`W1-WIN / P1_ENGINEERING / 5% / NO MATERIAL LOSE`。本页只签署普通 row、两条
> dtype conversion 与 mixed-bf16 dot 的固定 AVX2 实现机制；它不是全操作、跨 ISA、跨平台或
> Formal SOTA 证书。

## 被签署的命题

产品候选是 commit `63a1e614c9a137c140216d1f744b9e0a7074edad`，锁定 Highway 是
`f503c485cec6b4f2f596fa60d83484797c9eeb5c`。候选 shipping template 与 direct Highway peer 在
同一个 TU、同一个 AVX2 target、相同 pointer/count、alias、rounding、tail 和资源合同下强制内联；
双方都不经过公共 runtime façade。这样比较的是 HYLE 保留的 semantic glue，而不是两个函数地址、
第二套 dispatch 或 DLL 边界。

代表 family 为 f32 add/multiply/affine/sum、f16→f32、f32→bf16 和 bf16 mixed dot；每项使用
4096 元素的 resident row 与 1,048,576 元素的 streaming row。resident epoch 为 65,536，streaming
epoch 为 3,072。工具还加入一条同一 direct Highway body 对自身的 A/A 校准，并把 public façade 对
selected body 的结果单列为 informational，不混进算法 parity verdict。

## 机器、构建与执行合同

- machine：登记的 `W1-WIN`，Intel Core i9-13900K，性能逻辑核 2，core class 1；pin 与
  above-normal priority 均成功；
- build：Windows 11、Ninja Release、MSVC 19.50.35717、x64、AVX2，benchmark target
  `hyle_row_dtype_parity_bench`；
- instrument：四个 placement replica、两个 layout bank、两个执行顺序各两次；
  `scripts/op-parity-report.py --gate 0.05 --certify`；
- accounting：输入、输出分配和初始化在计时区外；整个 row body 在计时区内；无 heap、lock 或
  整段临时转换；ratio 大于 1 表示 HYLE 更快。

## 结果

校准与 certification 均为 `PASS`。verifier 对含 A/A anchor 的 15 行给出 7 `TIE`、8
`INCONCLUSIVE`、0 `LOSE`；所有 paired center 位于约 0–2% 范围，没有超过预声明 5% material
margin 的落后。`INCONCLUSIVE` 表示在当前 placement/order 不确定性下不能签 exact tie/win，不允许
改写为胜利，也不构成追逐地址布局的产品授权。

代码生成检查补充了统计结果：candidate/direct add 与 multiply 的各 placement 测量主体分别形成
126 条完全相同的指令序列，且没有 helper call/jump；固定 AVX2 公共 `hyle::simd::add` 只尾调用
`add_f32_avx2`。ordinary/fat 路线在整块入口读取一次已缓存 Runtime table 并间接调用一次，元素循环
内部没有 HYLE wrapper、callback 或第二次 Highway dispatch。因此可签的结构结论是：fixed/source
路径接近原样 Highway，ordinary/fat 路径只承担一次 buffer-boundary 间接调用。

public/body 行仅用于估计该边界调用，不能作为 Highway 胜负：1M row add、f16 conversion 与 bf16 dot
的中心比约为 1.002、1.002、1.006；4096 row add 的 0.938 受固定 callee 地址/小 batch 放大，按预先
规则保持 `INFO`，不驱动 API、layout 或 metadata。

## 原始证据与重放

- [HWY-first run 1](../../../../evidence/performance/simd-07-row-dtype-hwy-first-1.txt)，SHA-256
  `f41b61102e5910e1fff75a3f9650dd4286f081a421ce86b2f63765ec04d5879f`；
- [HWY-first run 2](../../../../evidence/performance/simd-07-row-dtype-hwy-first-2.txt)，SHA-256
  `8813279e1e6de11205ff81d6ca6dff2c331a03073b7d627a17873bee7717898c`；
- [HYLE-first run 1](../../../../evidence/performance/simd-07-row-dtype-hyle-first-1.txt)，SHA-256
  `69a5fd56b4fb87227c4ed77ca1f27542cf3a255989ad29f6801eae17326415a6`；
- [HYLE-first run 2](../../../../evidence/performance/simd-07-row-dtype-hyle-first-2.txt)，SHA-256
  `21c3e4b397fd4243e70a923fb17e8ac2d51aa01db57abee4a5caf4a127c6db58`。

```powershell
python scripts/op-parity-report.py --gate 0.05 --certify `
  evidence/performance/simd-07-row-dtype-hwy-first-1.txt `
  evidence/performance/simd-07-row-dtype-hwy-first-2.txt `
  evidence/performance/simd-07-row-dtype-hyle-first-1.txt `
  evidence/performance/simd-07-row-dtype-hyle-first-2.txt
```

## 未签署的内容

- conversion/reduction 受 HYLE 的 rounding、NaN、reduction-order 与 tail 合同约束；只能在明确同合同
  行上比较，不能推导为所有 Highway spelling 的 bit/codegen identity；
- SVE/RVV 本轮只有多 VL/VLEN correctness，没有真硅性能；macOS 只签 correctness/delivery；
- activation、search/scan、bitpack/random、ordering 和 legacy backend 删除由 SIMD-08～12 分别裁决；
- 若真实 consumer 在同合同、同 ISA、资格化复现中出现超过 5% 的 material regression，或发现热循环
  内二次 dispatch，才触发本结论重开。
