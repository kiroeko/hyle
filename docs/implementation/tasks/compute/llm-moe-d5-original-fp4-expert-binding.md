# D5-F44：官方 FP8×FP4 专家接线

<!-- MODULE REVIEW: AUDITED -->

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

`afd024e8`固定SCOPED，当前两个AUDITED owner重新绑定本卡，上述FIX已冻结。
原GEMM以指针为参数且M为运行时参数；索引wrapper只偏移参数并调用同一运算主体，
无需读取ID至host、修改原缩放/累加、扩展图执行器或引入模型级CPU计算。
默认资格以明确运算/编码、四个N/K和SM89/Driver13040冷绑定；执行时不选赢家。
原生六案例只证明具名程序可用，动态入口、资源、数值和成本仍按本卡批量验证。

### 重建的已定位差异

`bc72243d`普通入口21项与独立头4项已通过。离线重建中四个量化程序raw hash一致；
四个专家程序的CUDA源码完全相同，ELF只有`.strtab`内NVCC的内部符号路径hash不同，
全部指令、常量、重定位和其他元数据均逐字节一致。有限工具纠正仅允许该字符串表中
`_INTERNAL_`/`_GLOBAL__N__`后的8位hash差异；仍比较完整其余二进制，保留raw hash不一致事实。
不重生成产品程序、不更换数学或重复普通入口验证。

### 本批已验结果

`bc72243d`的21项普通测试全部通过：两入口64次真实形状专家投影逐BF16精确，Pro/Flash
两种路由模式完整MoE共24次执行通过原4%门；资源/失败、4项独立头、13项输入工具通过。
40个原生案例和八程序重建身份通过，raw hash差异按上节单列。统一实现60,150行/240文件。
见[完整收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/deepseek-original-fp4-entry/README.md)。
F43的ggml组合失败保留，本批不增加完整画像格。后续只需继续F43 attention/compressor/MTP
消费者批次；A5/A6/D6仍未完成，owner保持AUDITED，不将本地正确性写成整体CLOSED。

F43后续消费者现已在`9e846d87`完成，两种官方DeepSeek的CUDA Direct/Graph全部必需region
通过，见[完整画像收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/deepseek-official-cuda-complete/README.md)。
ledger新增4格到148/176；本卡产品与F44资源证据未变。其它路线、A5/A6/D6继续未完成。
