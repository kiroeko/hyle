# SGLang FP8：原版 AOT 与 Windows 原生执行资格

[收据](receipt.json)记录固定 SGLang `06992c92` 的原版 group128 动态量化和 block128
FP8 GEMM，通过 Triton 3.8.0 离线编译，在 Windows RTX 4070 / SM89 / CUDA Driver 13040
实际执行。九组投影签名（包括 GLM N576/K6144 尾块）、M1/M3、两次变化输入，共36项通过。
没有 PyTorch、模型引擎或新写的 HYLE 数学核。

量化结果与独立 E4M3FN 编码查表逐字节相等；scale 和 GEMM 沿首次执行前固定的误差门检查。
权重为非零稀疏夹具；激活包括非均匀分组 scale、正负极值、零组和 RNE 边界。
不外推到任意 payload、NaN/Inf、BF16 激活或 DeepSeek 官方 UE8M0 动态 scale 策略。
它证明成熟程序可在本机原生运行；尚未签 HYLE C++ Direct/Graph、完整模型画像、三交付或 P1。
完整画像 ledger 仍为140/176。

首次 launch 因探针遗漏 Triton 3.8 的两个 scratch 指针返回 CUDA_ERROR_INVALID_VALUE。
按原版 `tools/compile.py` 补齐两个空指针后通过；编译产物、数学和编译参数未变。
保留首版探针以及修正后的完整重放脚本、源码 hash、编译元数据、PTX、cubin 和 IR。
压缩文件的原始大小与 SHA256 均在收据中；`.gz` 仅压缩归档。

WSL只执行固定 AOT 编译，产品源码与脚本来自 Windows，GPU验证也在 Windows。
编译工具 wheel 为247,972,921字节，SHA256见归档；普通用户消费不依赖 Python/Triton。
[统一实现统计](loc.json)为58,971行/233文件，本探针不增加产品实现行数。
