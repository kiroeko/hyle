# SIMD 模块严格收口

> 状态：🟡 进行中（2026-07-26）。“历史算法里程碑完成”不等于当前实际使用导向的严格证书。

## 收口合同

SIMD 只有同时满足以下条件才可进入预发布冻结：

1. 以实际 kernel/上层消费者和成熟库为参照，常见能力实现或书面划出 negative space；
2. 热路径复杂度和实现形状不落后，代表性同合同基准覆盖 Windows/Linux/macOS；
3. C++ runtime、exact、native/fat-kernel 三条使用路径清楚且可安装消费；
4. C ABI 覆盖适合 FFI 的具名 pointer/count/POD 产品，且统一使用 `error` 边界；
5. C/C++ guide 分别回答“什么场景选什么”，写清尾部、重叠、数值与 ISA 合同；
6. architecture 只留慢变设计，performance 只留可复现数字，本页记录施工；
7. `error/platform/time` 依赖与全库 SIMD 调用方向经过反向审计；
8. scalar、SSE4、AVX2、AVX-512、NEON、SVE、RVV 正确性和相应真实性能证据闭环。

## Round 1：Highway 能力账本与 native range 算法

本轮先解决一个横跨图像、信号、量化、几何与 compute 自定义 kernel 的缺口：以前 `native`
只给寄存器积木和少数具名 kernel，用户必须重复编写 vector loop 与 tail。新增
`native/algorithm.h`，提供 generate、1/2/3 输入 transform、replace、find/count/copy-if，
并让现有 fat-kernel fixture 用真实 helper 编译全部变体。

边界决策：

- 这是 expert C++ source surface；lambda 在 target TU 内联，向量类型不跨 ABI。
- 不提供 C callback 镜像；FFI 继续使用具名 whole-buffer API。
- `copy_if` 稳定且支持 exact in-place，其他 offset overlap 不受支持。
- partial vector 的 inactive lane 会零填但不计数/不写出，predicate/transform 不应依赖 inactive side effect。

完整对照、P0/P1 与不做项见
[`../architecture/simd-highway-capability-ledger.md`](simd-highway-capability-ledger.md)。

## 本轮验证

Windows MSVC 19.50 Release：scalar/SSE4/AVX2 三个 `native_test`/fat-kernel 目标 3/3 通过。
AVX2 同 TU、同 ISA、1M u32、15 个独立进程的 Highway 对拍：

| workload | HYLE ns/element | Highway ns/element | HYLE/Highway throughput | 判定 |
|---|---:|---:|---:|---|
| unary transform | 0.075 | 0.075 | 1.004 | TIE |
| count_if 1/8 | 0.065 | 0.064 | 0.954 | TIE（sign 4/15，p=0.12） |
| find_if absent | 0.064 | 0.064 | 1.002 | TIE |
| copy_if 1/8 | 0.085 | 0.099 | 1.165 | WIN（15/15，p=6.1e-5） |

这里只证明新算法层在 Windows AVX2 的常用形状没有结构性包装税；它不替代 NEON 真机或
SVE/RVV 真硅性能证书，也不把亚纳秒噪声写成胜负。

WSL GCC 14 Release 的 scalar/SSE4/AVX2 同三目标 3/3 通过。scalable correctness 使用同一
目标二进制跨宽度执行：SVE 在 VL 128/256/384/512/2048-bit 各 104/104；RVV base 在 VLEN
128/256/512/1024-bit 各 99/99。两套均含新增 range helper 的 full-vector + partial tail 和
exact in-place `copy_if`。QEMU 只计正确性，不计真实性能。

该门同时发现并修复了交叉脚本的两处基建漂移：隔离 worktree 现在可显式传 gtest source；
error 收口后新增的 `status.cc` 依赖和 public fat-kernel 变体对象重新进入手工链接配方。
GCC ASan AVX2 全目标因大型 VQSort TU 在低并行度下两段构建仍未完成而停止，未产生诊断但也不计通过。

## 尚未完成

- [x] MSVC/GCC fixed-width 与 SVE/RVV 多宽度 QEMU 对新模板层的正确性。
- [ ] Clang、ASan/UBSan 与 NEON 真机验证（本轮 ASan 低并行构建超时，不能记通过）。
- [x] f16/bf16 显式 narrow tag、七后端 storage/promote/demote/基础算术闭环。
- [x] 窄浮点 register reduction、target-bound search 与 IEEE/bitwise total-order 合同。
- [ ] 窄浮点 prebuilt C ABI、scalable 重验与 sort 产品决策/实现。
- [x] SIMD 私有 TLS error seam 已删除；runtime-info/RNG-create 返回共享 owning `hyle_status`。
- [ ] SVE/RVV 真机 workload 性能证书；SVE2/RVV feature routing 按证据补。
- [ ] C ABI 声明/纯 C 引用/导出表和三种交付形状最终机械复查。
- [ ] 反向审计其他模块的 SIMD 使用与 math/simd owner 边界。

因此本轮是可独立审查的能力补齐，不是 SIMD closure 签字。

## Round 2A：共享 error seam

`hyle_simd_runtime_info_get` 改为直接返回 owning `hyle_status`；RNG create 改为 status + out-handle。
空 output、过小 snapshot 与分配失败分别保留可拥有、可传播的 code/message，删除
`hyle_simd_last_error` 及其 thread-local 状态。纯 C 端每个成功/失败 status 都显式 free；机械 API
账由 247/244（含 inline/导出）收成 246/243，移除的唯一符号就是私有 last-error。

验证：Windows MSVC 19.50 shared 的纯 C test 通过；WSL GCC 14.2 shared 的纯 C test 通过，
预处理引用 246/246、ELF 动态导出 243/243。成功、过小 runtime snapshot、NULL runtime output、
RNG 成功创建和 NULL RNG output 均执行 owning status 的查询与释放。

## Round 2B：f16/bf16 register lane 基础

新增 `narrow_float.h`。`F16Tag<DF>` / `BF16Tag<DF>` 从 f32 tag 取得相同逻辑 lane 数，复用
u16 native carrier，但 load/store、promote/demote 与 Add/Sub/Mul/Div/MulAdd 都要求显式格式
入口，避免 SVE/RVV 的 carrier 类型擦除后误做整数算术。AVX2/AVX-512 f16 full-width 使用
F16C；其余 baseline 使用统一 vector 位运算，不要求 ARM/RISC-V 可选 half 扩展。C ABI 不承载
target-specific vector 值，已有 dtype pointer/count API 保持正确 owner。

验证：Windows MSVC 19.50 Release 的 scalar/SSE4/AVX2 全 SIMD 目标 3/3 通过，AVX-512 目标
编译通过；窄浮点专项在 scalar/SSE4/AVX2 实际执行 3/3，遍历全部 65,536 个
f16/bf16 编码并对 f32 全编码空间做 65,536 点确定性抽样。AArch64 QEMU 的 NEON 3/3，SVE
同一 binary 在 VL 128/256/384/512/2048 各 3/3；RVV 同一 binary 在 VLEN
128/256/512/1024 各 3/3。QEMU 只计正确性，SVE/RVV 真硅性能仍未签字。

## Round 2C：narrow reduction/search 与 total-order

`narrow_float.h` 增加 IEEE 数值比较、exact-bit 比较、可逆 IEEE-754 `totalOrder`
key，以及 f32 sum、NaN-propagating/Number/total-order min/max。`-0/+0`、NaN
sign/kind/payload 都有机械合同。`narrow_algorithm.h` 在 target-bound C++ 层提供
O(n) 四路 sum、IEEE 数值 find/count、exact-bit find/count 和 total-order sorted check。
它们是用户 kernel 的 pointer/count 形状，没有为 benchmark 增加元数据或 callback ABI。

sort 本轮只审计/设计。Highway VQSort 的 f16 仅在 native-half target 开启，NaN
统一等价放末尾，bf16 仍有明确 TODO。HYLE 候选是“就地可逆 total-order key
transform → 现有 u16 sort/partial/select → inverse”：O(n) 转换、无分配、无常驻
元数据、无 optional-half ISA 依赖。但在 prebuilt/C ABI 产品合同审定前不落地。

验证：Windows MSVC 19.50 Release scalar/SSE4/AVX2 的窄浮点专项各 6/6，
包括全 65,536 编码的 total-order inverse/landmark、NaN 与 signed-zero 语义、
full/tail range；三个全 SIMD 目标 3/3，AVX-512 编译通过。本轮遵守资源协调
未占用 WSL，所以 SVE/RVV 新算法重验仍是未完成项。

## 本轮四项 LOC

| 固定口径 | `cloc` code |
|---|---:|
| 全库代码（含 docs/third-party，计入本轮新文件） | 264,682 |
| 自实现含测试（`include/hyle + src + tests`） | 128,439 |
| 自实现不含测试（`include/hyle + src`） | 88,596 |
| 当前审查模块实现（`include/hyle/simd + include/hyle/c/simd.h + src/simd`） | 38,408 |
