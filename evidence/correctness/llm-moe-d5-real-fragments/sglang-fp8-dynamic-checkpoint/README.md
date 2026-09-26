# F38：动态 FP8 局部通过，严格量化边界未通过

产品候选 `deac9fd01348fca47998ce8cb581b6b6fb1847fa`，Windows RTX 4070 / SM89，
CUDA Driver API 13040。完整画像 ledger 保持 **140/176**；本页不签完整官方画像或性能。

## 已实现与已验证

- 显式 `model_ops::Float8BlockLinear`：浮点激活 → 原版 group128 量化 → 原版 block128 GEMM。
  区分普通 F32 scale/floor 1e-10 和官方 DeepSeek 幂次 scale/floor 1e-4，保留 BF16/F32 输出。
  调用者 workspace 持有 codes/scales；两次原生发射，无 host 同步或执行期搜索。
- 原版 BF16 AOT 探针的72个具名案例通过：[收据](native-receipt.json)。包括零组、floor、
  幂次边界、FP8 舍入和 N 尾块。这些输入通过不等于任意激活满足严格编码门。
- [普通入口收据](initial-numeric-receipt.json)：构建通过；7项公共语义、3项 Provider 测试通过。
  Direct/Graph 各9签名 × M1/M3 × 三种激活/scale模式 × 两轮输入，共216次动态投影通过；
  已有72次预编码投影回归通过。资源测试含 OOM、独占 workspace、并发、重复执行和 Completion 保活。
- Direct/Graph 的完整 GLM 共享 FFN 测试均失败。失败是最终 down 输出的严格1 BF16 ULP检查；
  **不将该完整片段标 PASS**。全部原日志压缩保留，没有改门、改输入或改原版 kernel 凑通过。

16个 GoogleTest 中14通过、2失败，无跳过。四个 CTest 中2通过、2失败。
首次 C++ 编译的 optional 类型错误和同作用域 trace 宏错误已修复，原失败日志/收据保留。

## 已定位的原因

[诊断结果](ffn-diagnosis.json)保存 M1/M3、两次输入的实际三个中间输出与独立 oracle。
为定位临时加入只写观测数据的测试回调，未改变产品、测试输入或数值断言；回调已移除。

1. gate/up 投影和 SiLU×up 激活在全部四组中与参考**逐值完全一致**。不是上一段误差累积。
2. 对实际激活单独调用同一原版量化 cubin，四组分别有14、2、29、5个 FP8 code 与直接
   F32 归一化后 nearest-even 参考不同。普通 scale 最大相对差约1.12e-7。
3. [固定生成 PTX](quantize_bf16.ptx)显示 SM89 的转换先 `cvt.rz.f16.f32`，再
   `cvt.rn.satfinite.e4m3x2.f16x2`。例如0.033233173先截断为0.033203125，恰好落到 FP8
   中点，得到 code16；直接转换为 code17。该中间截断解释主要差异，另有倒数乘法与近似除法
   在量化中点附近的差异。不能把所有不同编码都归因于同一条转换指令。
4. 用实际 codes/scales 和独立 double 求和核验最后 GEMM，四组最大误差均为1 BF16 ULP。
   下游矩阵乘符合局部门，差异来源在量化程序，不是 HYLE 的权重布局或参数接线。
5. 完整 FFN 同时通过已有 `2e-5 + 0.025*abs(reference)` 整链门，最大绝对差约6.11e-5。
   但 F38 声明了严格编码门，**整链通过不能覆盖该局部失败**，尚不能授予通用动态量化资格。

二进制观测文件按 `ffn-M-replay-port.bin` 命名，每文件前半为实际 F32 可见值，后半为独立参考。
[重现脚本](diagnose.py)从当前 vendor 头恢复原版 cubin，在 Windows SM89 上重现逐 code 诊断，
结果写入 `.scratch/validation/d5-f38-diagnosis-reproduce/`；需要本机 NumPy，不构建产品、不跑 P1。
文件身份见 [manifest](manifest.json)。

## 有界后续与未完成

核验同 pin 的原版 CUDA `per_token_group_quant_8bit_kernel`，其普通 F32 scale 分支直接
使用 CUDA FP8 类型转换，并支持 runtime eps。先用本次反例及原72案例资格化；通过才替换接缝。
不新增 HYLE 量化数学，不无限切换编译参数；此处只是具名成熟候选，尚未运行或证明通过。

完整官方 MoE、其它路线 FP8、独立三交付、A6 性能与 D6 仍未验。
共享 vendor 锁触发的扩展 selector 门保留为待验，不由本页的局部测试代签。
统一 Compute+Graph 实现为 **59,300行/235文件**，不含测试、第三方源及其生成程序。
