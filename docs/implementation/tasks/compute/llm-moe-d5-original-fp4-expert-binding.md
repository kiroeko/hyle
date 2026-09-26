# D5-F44：官方 FP8×FP4 专家接线

<!-- MODULE REVIEW: SCOPED -->

- baseline：`815550677`；沿用当前 `compute_ops` / `compute_provider_cuda`，实施前重新绑定AUDITED。
- trigger：F43完整MoE组合的最终数值误差超过原门；现有ggml MXFP4点积增加Q8_1激活转换，
  不能用已验的权重repack或单投影代签原版E4M3激活语义。
- 具名成熟来源：DeepSeek V4 Pro HF revision `b5968e9190ef611bbf34a7229255be88a0e937c1` 的
  `inference/kernel.py::act_quant_kernel` 与 `fp4_gemm_kernel`，原MIT/TileLang/CUTLASS许可沿用。
  F43原生N4096/K2048的6案例精确通过；其余原生形状和普通入口尚未资格化。

## 有限 FIX（实现前冻结）

1. 明确原始FP4专家权重的独立格式：全部连续E2M1 packed字节[E,N,K/2]，随后F32
   scales[E,N,K/32]；每字节低/高nibble依次表示相邻元素，源UE8M0只无损扩宽。
   不复用ggml交错格式身份，不在热期展开BF16/F32权重。
2. `GroupedFloat4Linear`：上述权重、BF16[T,A,K]、I32[T,S]→BF16[T,S,N]，A=1或S。
   激活为原版group128 E4M3/power2/floor1e-4，F32累加；重复ID合法，越界走Completion失败。
3. 原版TileLang离线生成四形状(3072,7168)/(7168,3072)/(2048,4096)/(4096,2048)。
   CUDA接入适配只把原生成GEMM入口改为内联device函数，并增加checked索引入口：grid.z选择
   一个token/slot，验证ID后偏移A/B/C/scales指针，再调用原运算主体，M=1；无数学主体修改。
   每块仍由官方程序完成FP4解码、FP8 Tensor Core GEMM和scale累加。该逐slot方式的tile利用率
   与成本明确计入验收，不宣称大prefill最优；不为过测计算全部未选专家。
4. 扩展现有DeepSeek Provider，不新增工厂或热期搜索。每次两次launch：共享token或独立slot
   激活量化、索引GEMM；caller workspace容纳FP8 codes+F32 scales。IDs不D2H，无host同步，
   不新增私有allocator。沿用后端device status、Queue和Completion；失败结果不可消费。
5. F43两模型score/hash完整MoE直接消费新语义，去掉该腿重复roundtrip及ggml重排；旧失败记录
   保留。批量覆盖四形状、共享/独立激活、变化/重复/越界ID、完整MoE、OOM/并发/保活和公开头。

## 审计与预算

新格式必须checked sizing并拒绝非canonical payload；原native形状限定SM89/Driver13040，
四个N/K均为完整128块。寻址使用64位字节偏移；slot grid与量化原生行数分别核对容量。
原代码每tile M32、每slot有效M1的成本属于这个具名成熟候选，不隐藏padding或额外量化。
三交付仍使用已有CUDA包内嵌程序及许可证，不引入运行时Python/TileLang。

预算：四形状一次AOT批次、一次有界indexed原生资格，再进行共享两入口/资源一批构建验证。
只因具体失败定向修正；不改变既有数值门、不替换输入、不开发数学核。若该原版候选也未通过，
按实际失败定界，不循环扩大候选。完整attention/compressor/MTP仍属F43，A5/A6/D6未决。

当前SCOPED，无本卡产品修改。原生六案例不能签动态专家普通入口或完整画像。
