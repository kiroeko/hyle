# D5-F37：官方 FP8 原版程序的原生接入资格

<!-- MODULE REVIEW: AUDITED -->

## SCOPED → AUDITED：先做有限资格，不修改产品

- baseline：`24b674bb9`；Windows canonical。承接[四路线源码边界](llm-moe-d5-official-fp8-source-boundary.md)，
  不把现有HYLE入口拒绝或cuBLASLt拒绝当作全部上游缺失。
- 当前D5 `compute_ops` / `compute_provider_cuda` owner继续AUDITED。本轮产品FIX为零；
  若下述资格成立，再固定原生适配的有限FIX、来源锁与三交付影响，不先扩建Provider目录。
- 成熟来源：SGLang `06992c92ad4494fe9f9636e75be3e35040512447` 的
  `fp8_kernel.py::_per_token_group_quant_8bit` 和 `_w8a8_block_fp8_matmul`。
  原函数体及decorator逐字节提取，只隔离Python模型引擎import；不修改数学/索引/分组。
- 编译候选：原版Triton的公开AOT能力，固定wheel、版本、hash和`cuda:89:32` target。
  WSL仅生成可丢弃验证artifact，源码/脚本与最终候选仍在Windows；运行验证在Windows NVIDIA GPU，
  使用CUDA Driver加载cubin。不能把Python引擎执行说成HYLE C++普通入口。

## 有限动作与停止条件

1. 不装完整PyTorch/模型引擎。仅一个有大小记录的Triton wheel和隔离工具目录，关闭下载缓存，
   复用现有CUDA Driver；不新建大型产品build树。将编译器、原函数及cubin身份保留。
2. 原版group128量化和block128 GEMM各编译一次；先在本机SM89验证动态M/N/K的原生launch及
   八个已登记投影签名，包含GLM N576尾block；输入非零且更换两轮。与独立FP8 codes/scale
   oracle比较；量化/点积各自诊断，不改变官方FP8语义或以BF16 MatMul替代。
3. 原程序仅接受其具名布局/精度。明确scale类型、epsilon、饱和、尾block、资源上限、需要的
   workspace和launch ABI。算法不合格或AOT在SM89拒绝时保留精确证据，停止该候选。
4. 最多一次源码/工具接缝纠正；不搜索编译参数来碰通过，不跑P1。

本卡不签完整画像、不签普通入口或三交付。Apple/CPU/Vulkan、grouped专家和官方MTP仍保留；
后续只沿成熟程序继续接缝，禁止借此新增HYLE FP8 kernel。若需要新的产品runtime组件，先按
component/owner协议单独重新定界。A5/A6不可由这个probe代签。

### 首次运行前冻结的检查

Triton3.8.0的cp312 Linux wheel为247,972,921字节，SHA256
`e91ffa46d095b252248297292dd22bcbacd53a125a0c2eefbbbf74925a320bc3`；原版AOT在SM89已编译
两函数，未执行。矩阵使用SGLang默认64×128×128、4warps、3stages；量化group128、4warps。
不调参。运行八个已登记dense签名及N576/K6144尾块，各M1/M3、两次变化输入。
权重为已登记两非零项编码夹具；激活含非均匀group scale、正负极值、零group和RNE边界。
独立E4M3FN查表oracle要求codes精确相等；scale相对误差≤2e-6（零group的epsilon也保留）；
每个block的点积及scale应用在独立double oracle求和后转F32，输出绝对误差≤2e-6加
相对误差2e-6。只签这些有限数值域，不能覆盖未测NaN/Inf或官方UE8M0动态scale策略。

## 原生资格结果与 F37.1–F37.4 接线审计

2026-09-27：原版两函数已在 Windows RTX4070 上通过36项有限检查，
见[原生证据](../../../../evidence/correctness/llm-moe-d5-real-fragments/sglang-fp8-native/README.md)。
首次 launch 缺少两个 Triton scratch ABI 指针，按原版工具补齐后通过；没有改变数学或重新调参。
该证据允许以下有限产品接线，取代上文资格开始时的产品 FIX 为零；owner 仍为本轮 AUDITED 的
`compute_ops`、`compute_provider_cuda`，Direct/Graph 改动仅为真实消费者测试。

- **F37.1**：公共 block128 权重存储支持 N 方向不足128的最后一组，scale 行数使用 ceil(N/128)；
  K 仍须完整分组。现有完整块字节布局不变。审计其它使用者：cuBLAS不接受opaque格式，
  IREE原生程序已显式拒绝N尾块，保留其精确拒绝，不扩大未验证执行域。
- **F37.2**：现有 SGLang CUDA Provider 接入原版 block128 GEMM 的固定 SM89 cubin；
  完整 provenance/编译重现脚本/许可证随源码，CMake将产物嵌入本模块，静态和动态交付无需外置
  文件或 Python/Triton。SM89与已验证driver域在冷准备时匹配；其它设备明确拒绝该候选。
- **F37.3**：既有 MatMul 只接受两个已编码 E4M3FN+F32 block128 operands，结果F32。
  不把普通F32 MatMul输入悄悄量化。原CUDA context/module由prepared生命周期持有，调用仅
  按已固定参数发射原生程序；0 workspace、0复制、0同步、1 dispatch，不在执行时选核。
  处理完整描述符校验、I32索引界、分配/module加载失败回滚和无别名；Completion使用既有CUDA队列。
- **F37.4**：共用真实dense签名及GLM尾块在普通Direct/Graph各两次输入运行；
  另验不支持的格式/转置/设备域、元数据OOM、并发invocation与资源释放。
  原冷拒绝测试迁移为真实运行断言，不能保留错误的永久拒绝预期。

批量接完后仅一次所选CUDA构建及上述测试，公共描述符运行compute_ops所选测试；
三交付与P1仍归A5/A6，不将探针或局部接线代签。若出现数值错误先保留输出和源码归因，
最多一次明确接缝修复；没有证据不调整误差门。

本批不签动态激活量化、官方完整MoE或32格官方画像；它们仍沿已登记模型合同继续组合。
DeepSeek的UE8M0动态scale不能由F32 scale探针代签。源码/static/shared仍须实际消费验收。
