# SIMD-17：交付与收口

- ticket state：`DONE`
- kind：candidate / verification / closure
- budget：1 工程日；串行机器 4 小时
- depends on：SIMD-16

冻结不可变 candidate，验证 source embedding、installed static/shared、offline Highway+xsimd、fixed/exact/fat、
三平台适用 correctness、受影响 consumer 与冻结 P1 groups。检查 387/387 ledger 零漏项/零 `PENDING`、公开
C++ 易用性、Native FFI 清洁负空间、architecture/user/developer/implementation 分流、依赖/license、红队、
六问、四项 LOC 与 terminal receipt。未运行的适用 gate 必须是 `BLOCKED/INCONCLUSIVE`，不得写成 `PASS`。

## Candidate 前验证输入

- ledger/provider manifest/generated routes：387 operation、422 scope、零 `PENDING`，机械检查 `PASS`；
- W1-WIN MSVC 19.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21 release：五个 SIMD test artifact
  均为 5/5 `PASS`；
- W1-WSL GCC 14.2 ASan 与 UBSan：均为 5/5 `PASS`。ASan 暴露的 VQSort debug-network threshold
  mismatch 已按 upstream `HWY_IS_DEBUG_BUILD` authority 修复并完成三平台 release 回归，见
  `evidence/simd-17-vqsort-sanitizer-fix.json`；TSan 对本轮纯 compile-time route、无新增共享可变状态的
  改动为 `NOT_APPLICABLE`；
- locked/offline Highway+xsimd 依赖的 positive/negative contract `PASS`，integrated VQSort source hash 为
  `24d39fbef0b8b53c2d1bdbf7cfba0f9ccbb42e4a91ec4302be51015a97503442`；
- W1-WIN 当前 P1：普通 row/dtype 六个产品 row 的 95% lower bound 全部高于 0.95；Highway 对 xsimd 的
  tanh/GELU 四个 row 全部 material `LOSE`，因此产品按已冻结例外选择 xsimd。收据为
  `evidence/performance/simd-17-w1-win-p1.json`。

上述是冻结 candidate 之前的 finding/fix 输入，不能代替不可变提交回放。下一步只允许：创建 candidate、
回放 source/static/shared 与适用 correctness、完成文档/红队/六问/LOC，并形成 terminal receipt。

首个候选 `4a1c216c/5d7269b9` 在 SVE 交叉回放中发现 VQSort scratch authority 仍引用已删除旧常量，已按
协议撤回。有限修复使用锁定 Highway 的 `HWY_MAX_BYTES`，并校正可扩展产品测试入口；SVE 128/256/384/
512/2048-bit 与 RVV 128/256/512/1024-bit QEMU、W1-WIN、W1-WSL ASan/UBSan、M1-MAC 均通过后，第二个
candidate 冻结为 `e49ba0ec8e17956a23a7ba83c7abc2da91879b1e` /
`69702d9e9a2ec5399b9d1a4c5665bc0105f1303b`。随后 source-delivery 的 integrated-source audit 正确
拒绝了未随 VQSort 修复更新的 provenance hash，因此该候选也已撤回；当前有限修复只更新
`highway_port` 内容锁与相应证据，后续新 candidate 才是终态绑定身份。

锁修复后的三平台 source/shared/static relocated consumer 与 locked/offline 正负合同全部通过；最终
candidate 为 `27633329949a88321e5e0d71bad28e6dbb997f9c` /
`457630a17039041ecc2bd077b0cdc149c55bebdc`。SIMD-17 的 terminal receipt 只绑定该身份。

## Candidate 后验证结果

不可变 candidate 后没有 SIMD 产品文件变化；完整机器可读收据为
[`evidence/simd-17-candidate-verification.json`](../../../../../evidence/simd-17-candidate-verification.json)。

| 风险面 | 结果 | 当前轮事实边界 |
|---|---|---|
| inventory / route | `PASS` | 387 个 operation、1,797 个 baseline definition site、422 个 terminal scope；366 `ROUTE_HIGHWAY`、34 `KEEP_HYLE`、22 `REMOVE_REDUNDANT`、零 `PENDING`；具名 xsimd family 例外另由 route manifest 约束 |
| W1-WIN correctness | `PASS` | MSVC 19.50 Release，ordinary/provider/codegen/xsimd-winner/ordering 共 5/5 |
| W1-WSL correctness | `PASS` | GCC 14.2 ASan 5/5、UBSan 5/5；本轮 compile-time route 不引入共享可变状态，因此 TSan 语义上 `NOT_APPLICABLE` |
| scalable correctness | `PASS` | SVE QEMU 128/256/384/512/2048-bit 与 RVV QEMU 128/256/512/1024-bit 全部通过；只签 correctness/codegen，不签 QEMU 性能 |
| M1-MAC correctness | `PASS` | AppleClang 21 arm64 Release 5/5 |
| source/static/shared | `PASS` | 三平台 source embedding、relocated static、relocated shared consumer 全部通过；每种交付均在最终 candidate 后回放 |
| offline / provenance | `PASS` | Highway、xsimd lock 正负合同与 integrated VQSort 内容锁全部 fail-closed；三镜像关键文件 SHA-256 一致 |
| engineering performance | `PASS` | W1-WIN `closure-perf-v1` 5% 门：普通 row/dtype 六组全部 non-inferior；MSVC AVX2 f32 tanh/GELU 的四组 material winner 为 xsimd，产品 route 与证据一致 |
| C++ usability / docs | `PASS` | ordinary pointer/count、span 与 HYLE POD 面不泄漏 vendor register；source-only fixed/fat seam 和 pin provider 逃生口有明确 guide |
| red team | `PASS` | 旧 register/backend 公开引用为零；产品 `extern \"C\"`、Native FFI/C ABI、运行时逐 operation provider dispatch 与 installed vendor header 泄漏均为零 |

fixed/exact 的 provider 选择在编译期消失；ordinary/source-fat 只在 whole-buffer 入口做一次已有的 cached target
选择，不在元素循环内桥接、复制或二次探测。P1 只证明冻结的普通/source-primary contract；不扩写为全部
operation、size、平台或 Formal SOTA。

## Owner 六问直答

1. **第三方依赖是什么？** SIMD 直接依赖 HYLE `core`、`platform`；解析后的内部闭包还包含
   `container_intrusive/error/time`。产品 provider 是锁定的 header-only Highway
   `f503c485cec6b4f2f596fa60d83484797c9eeb5c` 与 xsimd
   `3b3151ff665514a00a437cc985f47b7de99b772b`；VQSort 是带 origin revision 和内容锁的最小集成源码，
   不冒充 HYLE 原创。三者升级都触发受影响 route 重审，而不是整库重写。
2. **对推理引擎与 ggml/llama.cpp north-star 是否完整？** 当前准入的可复用 CPU SIMD 层已覆盖连续 row
   代数、混合/storage dot、五种 storage 转换、narrow、批量数学/activation/softmax、search/scan、
   bitpack/RNG 与 ordering；没有未处置的 SIMD `MUST`。tensor layout、block quant、GEMM、attention、
   RoPE、fusion、threading 和 GPU 是 Compute/engine 能力，不应为追逐 north-star 符号数塞进 SIMD。
3. **同合同性能是否合理？** `PASS`。先比较语义、复杂度和 codegen，再用真实 whole-buffer 工作负载执行
   5% P1 门；明显赢家才保留 HYLE/xsimd，等价、反转、测不准或小于 margin 的 scope 归 Highway。
   当前证据不声称全部平台全面 SOTA，SVE/RVV 也没有真机性能结论。
4. **伪 C ABI 是否清理？** `PASS`。SIMD 不导出 C API、Native FFI 或产品 `extern \"C\"`；测试中的
   `extern \"C\"` 只为稳定反汇编标签。未来只有真实 foreign-runtime consumer 通过准入，才在运行时
   owner 处建立类型擦除边界，不为模板/register 面制造伪 ABI。
5. **三种交付与构建形状是否成立？** `PASS`。source embedding、installed static、installed shared 在
   Windows/WSL/macOS 均由 relocated consumer 验证；ordinary 安装面只有 13 个 HYLE SIMD 头，provider、
   `target.h` 和 `fat_kernel.h` 保持 source-only。fixed/exact/fat 路由和 offline lock 均通过。
6. **C++ API 是否自然且文档权威？** `PASS`。普通用户只见 pointer/count、span 和 HYLE-owned POD/storage
   type；专家 seam 明确承担 target-bound 翻译单元与 vendor pin。架构、用户选择表、维护升级/路由流程和
   历史事实分别位于 architecture/user/developer/implementation，不要求用户阅读源码猜用法。

## Terminal decision

SIMD-17 的七项 aggregate gate 全部为 `PASS`，本轮裁决为 **`CLOSED`**；Formal 性能仍是独立
`PENDING`，不阻塞工程收口。产品身份固定为
`27633329949a88321e5e0d71bad28e6dbb997f9c` / tree
`457630a17039041ecc2bd077b0cdc149c55bebdc`。

可接受限制：

- Formal performance certification 尚未完成；当前只有冻结 W1-WIN contract 的 P1 工程结论；
- SVE/RVV 只有 QEMU correctness/codegen 资格，没有真实 silicon winner；
- classic `cl.exe` 的 direct Highway target body 受 upstream AVX3 blocklist 限制为 AVX2，direct AVX-512
  使用 clang-cl；
- 不承诺 Native FFI、C ABI 或稳定跨工具链 C++ ABI；
- integrated VQSort 随 upstream 变化时必须重新 rebase、审计并刷新内容锁。

重开条件：锁定 Highway/xsimd 或 VQSort 升级改变受影响合同；真实推理消费者发现未归属的可复用 SIMD
primitive 或 correctness 缺陷；同合同资格化证据在支持 compiler/ISA 上出现 material route reversal；
新增目标改变有限 target matrix，尤其是 SVE/RVV 真机或 classic-MSVC AVX3；具体 foreign-runtime consumer
通过 Native FFI 准入。

四项 LOC 从 immutable candidate 运行
`python scripts/loc-report.py --module simd --commit 27633329949a88321e5e0d71bad28e6dbb997f9c --self-check --json`，
`self-check=PASS`：

1. 全库代码：401,264；
2. 自实现包含测试：96,041；
3. 自实现不包含测试：59,898；
4. SIMD 实现：14,211（不含 tests/bench/docs/CMake）。
