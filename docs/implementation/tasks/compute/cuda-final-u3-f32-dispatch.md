# CUDA U3：原 F32 分发精度接缝

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- baseline：`0964d17ade1fb5342aba47b19bbecf0a416612e1`；Windows canonical。
- owners：compute_ggml_runtime 的原 CUDA manifest；当前 Compute/Graph ggml CUDA owner 保留 R1。
- trigger：当前 Kimi MLA head projection 请求 F32 乘数；原 ggml `mmf.cuh` 的 float tile 最终
  在 `mma.cuh:1089` 执行 `mma...f32.tf32.tf32.f32`。`ggml_cuda_mul_mat_id`／普通 MUL_MAT
  选择 MMF 前不检查 `dst->op_params[0]` 的 F32 要求，cuBLAS 分支却有相应原 F32 路径。
- consumer：普通 Kimi MLA／plain GroupedMatMul；正交为大尺寸非 BF16 可表示乘数、重复 ID、
  独立 slot／变地址／capture 失效与正确性。不要通过补零改 shape 强迫 dispatch 或放宽误差。
- scope：三处原 MMF 分发／capture suitability 检查在 F32 权重且要求 F32 时避开 TF32 分支。
  原 cuBLAS／sorting 程序承载完整 F32；quantized／BF16/F16 原程序与原数学 kernel 不改。
- non-goals：不写新 GEMM、kernel、模型 engine；不改公共 schema、CPU-only 数学或性能仪器。
- resources：仍原 stream/pool/scratch／Completion，原排序的同步在原 capture suitability 中如实体现。
- P0：原排序／转换／同步计入；正确性接线不是性能优选。U4 尚未运行，不发赢家证书。
- budget：一次已锁源码 dispatch 账单、一批精度选择修复、一批不可变候选的两平台集中验证。
  需要 source-id 变更而重编原源，低并行2、重型串行；不重复盲测／追加自研。
- predeclare：F32 MMF 可达的大 shape 独立方程／非 TF32 可表示输入、Mla完整程序、已有559合同
  ／current state回归、原捕获失败／寿命；最终U3联合source/static/shared、CUDA-only/mixed、
  offline／DLL邻置。docs/closure/selector与统一LOC；未运行不PASS。
- stop：范围审计后固定有限FIX，AUDITED绑定实际 runtime owner 后改 manifest。若原 F32 carrier
  仍不满足合同，具名说明具体领域，不改误差门或新写数学。

来源事实：实际 query/cache 上的独立 attention 方程误差仅约1e-7～4e-7；head grouped误差
relative-L2约2.3e-4符合 TF32 乘数截断量级。后者只是线索；原指令和未消费精度 flag 的 dispatch
才是可复现源码反例。旧误判为“仅 cache reference”的推断不足，当前模型FAIL不撤。

## AUDITED 有限 FIX

SCOPED `23f86716`；已锁定原文件／F32指令／精度参数，固定下面单一批次：

1. 在原普通MUL_MAT／MMID／MMID-needs-sync三个MMF条件添加同一完整F32条件：
   F32权重且`op_params[0]==GGML_PREC_F32`时不进入MMF。数学、默认native精度、量化分发不变。
2. 原context manifest更新精确after SHA、NOTICE、在树dependency hash及版本；SDK/source-id
   如实变更，旧版本性能证书不可外推。CPU-only source选择与数学不变。
3. 以K=128,N=512的独立F32乘数方程覆盖MMF可达形状；测试值不可TF32精确表示，不能只用
   0.25或小N绕开实际分支。保留重复ID、actual state、完整Kimi链和既有捕获/资源门。

当前 runtime owner绑定本卡AUDITED，保留G1/G2原IQ1/runtime与三交付尾项；R1 Compute/Graph
owner不变。P1等待U4，本批只修正原生选择合同和资格验证，无新self数学。

明确build接缝：原owner仅watch生成的build-local manifest，未watchcanonical CUDA resource/
IQ1 manifests，单独`cmake --build`没有重新组装新精度patch。首次大shape回归实际命中旧TF32
source并FAIL，不能计作新候选产品结果。有限FIX同时将这两个原manifest与权威dependency lock
加入本owner的configure dependency，使后续正常增量构建取得真实source-id；不改公共selector。

## Windows 已运行结果与待验

产品`7b15b62f`实际configure／生成`source-f99368c68335266b`后重编译通过，大尺寸F32／重复ID
独立方程两项通过。临时actual-cache包装已在test-only`7f6c9107`移除；恢复原独立完整四replay
轨迹与全部九输出、原误差门后，Kimi三profile在普通Direct/Graph均通过。没有换shape、舍弃
状态反馈、以actual输入验证替掉最终轨迹，或放宽数值门。

原源审计将三个dispatch条件逆变换后，与上一版composed source逐字相同；数学kernel、原IQ1
和其它分发保持。早先“仅reference/cache”的解释被原TF32指令和新精度回归证伪，已删除临时
诊断代码与reference factory，保留失败原日志。当前Windows更大联合回归正在运行；Linux与
最终U3联合三交付仍未验证，不签本卡CLOSED／新版本性能WIN。
