# SIMD-10 bitpack/random 有界性能证据

> verdict：`W1-WIN / P1_ENGINEERING / 5% / NO MATERIAL LOSE`。本页只签署 canonical-v1
> u16 11/13-bit pack/unpack 与 deterministic f32 bulk random fill 的 4,096/1,048,576 元素固定
> AVX2 机制；它不是整个 SIMD、所有 bit width/type、所有 ISA/平台或 Formal SOTA 证书。

## 被签署的命题

最终产品候选是 commit `ee0c77e07f9be962e0a50afa4ea7a2cd1995a3ab`（canonical API/实现起于
`62458bd3`，仪器进入 `09f662f3`，最终 epoch 资格提交是
`07bf8843b0cb25be37abc6d1ca5dd1f4bf33c8ab`），锁定 Highway 是
`f503c485cec6b4f2f596fa60d83484797c9eeb5c`。instrument 在一个固定
AVX2 TU 中实例化 exact shipping target body；bitpack peer 独立使用 Highway
`CappedTag<uint16_t, 8>` 对相同 fixed-128 canonical block 调用 `Pack16<11/13>`，random peer 独立拼写
相同八流 xoshiro256++ 状态迁移与 f32 映射。双方对齐格式/序列、ISA、pointer/count、tail、状态所有权、
读写流量和资源计费，均不经过 public runtime façade。

资格后的第一次 P1 暴露了真实产品问题：候选为“吃满”AVX2/AVX512 寄存器，把多个 canonical block
先转置到栈 tile，调用一次宽 Highway pack，再转置回 wire order。该额外 O(n) 搬运令八条 bitpack 行的
中心比只有 `0.199..0.297`，不是测量噪声。有限修复删除两组 stack tile 与双向重排，直接逐 fixed-128
block 调用成熟 Highway mechanism；API、wire bytes、复杂度和常态 metadata 均未改变。Windows 四档、
SVE 五种 VL、RVV 四种 VLEN 与 exact AVX512 artifact gate 在修复后重跑通过。

## 机器、构建与执行合同

- machine：登记的 `W1-WIN`，Intel Core i9-13900K，性能逻辑核 2，core class 1；pin 与 high priority
  均成功；
- source：干净 detached worktree `07bf8843b0cb25be37abc6d1ca5dd1f4bf33c8ab`；产品实现仍绑定上述
  `ee0c77e0`。该隔离同时发现并修正了被另一张 ordering 票未提交改动污染的 `highway_port` content
  hash；
- build：Windows 11、Ninja Release、MSVC 19.50.35728、x64、AVX2，target
  `hyle_bitpack_bench`；
- instrument：每侧四个 placement replica、两个 layout bank、两个执行顺序各两次；4K/1M 的
  `minEpochIterations` 分别为 65,536/4,096；`scripts/op-parity-report.py --gate 0.05 --certify`；
- accounting：初始化、资格对拍和 allocation 在计时区外；一次完整 buffer body 在计时区内；ratio 大于
  1 表示候选更快。

## 结果

A/A calibration、四份输入唯一性、正反序、双 layout、pin/priority audit 与 certification 全部 `PASS`；
四份日志均无 nanobench instability marker。11 行得到 3 `WIN`、2 `TIE`、6 `INCONCLUSIVE`、0
`LOSE`。产品中心比为：

- bitpack 4K：pack 11/13-bit `0.966/0.997`，unpack `1.021/0.983`；
- bitpack 1M：pack 11/13-bit `1.154/1.047`，unpack `1.110/1.130`；
- random f32 4K/1M：`0.993/1.003`。

1M 的 b11 pack 与两条 unpack 通过 `WIN` 门；1M b13 pack 的中心虽为 `1.047`，其置信区间和统一
5% margin 仍使它保持 `INCONCLUSIVE`。4K b11 pack 的中心 `0.966` 也因置信区间跨越 5% loss 边界而
保持 `INCONCLUSIVE`，不能写成确定回退。最终 product/peer placement spread 最大约 `1.092`。P0 成本
结构与源码检查已证明最终候选和 peer 都是 single-pass O(n)/O(1) fixed-block direct Highway
composition，没有 transpose、whole-buffer copy、heap、lock 或逐元素 dispatch；P1 没有裁决出超过预声明
5% margin 的 material lose。这支持“同机制正常水平”，不支持夸张的全局排名。

public/body informational 中心为：bitpack pack 4K/1M `1.003/0.985`、unpack `0.975/0.997`，random
`0.992/1.005`。正常调用路径没有第二次算法、copy 或 allocation；其中点值仍受同一 placement uncertainty
约束，不单独触发产品改动。

早期短 epoch 证据曾出现 nanobench `Unstable` 标记，因此不能用于签署。资格轮先把 4K 提到 65,536，
随后按仪器对一条 1M unpack replica 给出的至少约 2,778 次建议，把 1M 固定为 4,096；这里列出的四份
原始日志完整替换早期日志，没有改变 workload、操作、peer 或计费边界。

## 原始证据与重放

- [HWY-first run 1](../../../../evidence/performance/simd-10-bitpack-random-hwy-first-1.txt)，SHA-256
  `32548c35c8a674a49a7b634cc2f8a15d91cc01f7128f83aed41adbed8c51f006`；
- [HWY-first run 2](../../../../evidence/performance/simd-10-bitpack-random-hwy-first-2.txt)，SHA-256
  `21596ab55391f8a34f255bca6a773b438a023ec000444215a88434e20ed16a88`；
- [HYLE-first run 1](../../../../evidence/performance/simd-10-bitpack-random-hyle-first-1.txt)，SHA-256
  `f9df3305e28469840dc10ea0934989465051ea6df6b16d72da5314b0a2a0dafd`；
- [HYLE-first run 2](../../../../evidence/performance/simd-10-bitpack-random-hyle-first-2.txt)，SHA-256
  `7dba37df40d457c916320e54abd2940d9e96ee0e383c2fc93b48875c31b1a37d`。

```powershell
python scripts/op-parity-report.py --gate 0.05 --certify `
  evidence/performance/simd-10-bitpack-random-hwy-first-1.txt `
  evidence/performance/simd-10-bitpack-random-hwy-first-2.txt `
  evidence/performance/simd-10-bitpack-random-hyle-first-1.txt `
  evidence/performance/simd-10-bitpack-random-hyle-first-2.txt
```

## 未签署的内容

- u8/u32、其他 bit width、zero/tail/golden/padding/cross-tier bytes 由 exhaustive correctness gate 与
  O(n)/O(1) 成本结构签署，不从四条代表性 u16 行外推排名；
- SVE/RVV 本轮只有多 VL/VLEN correctness 与 single-target artifact，没有 QEMU timing；AVX512 只做
  clang-cl exact artifact，macOS 也不由本页签固定 timing；
- entropy、crypto、distribution sampler、variable-bit、FOR/delta/varint、quant block/GEMM fusion 与协议
  framing 不属于本票；
- 若真实 consumer 在同格式、同 ISA、资格化真机上复现超过 5% 的 material regression，或产品再次出现
  transpose/第二遍 whole-buffer traversal/heap/lock，才触发本结论重开。
