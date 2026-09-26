# D5-F42：DeepSeek 官方激活量化还原边界

<!-- MODULE REVIEW: SCOPED -->

- baseline：`25a7fec577f2137629470d8f48d6b22376d47a6f`；公开语义 owner 为当前 D5 的
  `compute_ops`，原生接线 owner 为 `compute_provider_cuda`。审计完成后绑定本卡，产品修改前固定 FIX。
- trigger：固定 DeepSeek V4 Pro/Flash 官方程序在 raw/compressed KV 的前448维执行
  E4M3FN/group64 量化再还原；indexer query/compressed key 在 Hadamard 后执行
  E2M1/group32 量化再还原。只保留 BF16 或套用通用 MXFP4 默认缩放会省略真实数值边界。
- 来源：[固定官方 kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py)
  的 `act_quant_kernel` / `fp4_quant_kernel`，以及同 revision `model.py` 的
  Compressor/Indexer/Attention。已缓存全文 SHA256 分别为
  `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`、
  `ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71`。
- 主纵切：普通 BF16 输入→原版组量化/还原→普通 BF16 输出，随后由 consumer 接回 KV 或 indexer。
  正交纵切：任意完整组的行宽、跨32组尾块、变化输入、全部有限 BF16 编码与精确舍入中点。
- 边界：本卡补必要原子机制；模型状态/位置/尾部64维切分仍属 consumer。不增加模型加载器、
  Python 运行依赖、自研数学或自动性能搜索。不以该原子通过代签 DeepSeek 完整画像。

## 有限 FIX（待审计后冻结）

1. 显式 `BlockQuantizationRoundtrip` 两种语义：E4M3FN/group64/floor1e-4 和
   E2M1/group32/floor6×2^-126。均为 F32 组 absmax、缩放向上取二次幂、饱和最近偶数编码、
   F32 还原后存 BF16；输入/输出是相同形状的 plain dense BF16 矩阵，整组、无别名、有限输入。
   不将 E2M1 的该缩放规则与其它成熟实现的 round-even scale 视作同合同。
2. BORROW 上述未改数学程序；固定 TileLang0.1.14 离线生成 CUDA、CUDA13.3 编译 SM89。
   将每个独立量化组当原版的一行，N等于组宽，M等于总组数；两份程序支持任意整组矩阵，
   不按模型名称生成多份程序。原版仍写出 F32 scales，作为 caller workspace 计费。
3. 扩展已有 DeepSeek Provider；Context/Prepared/Invocation/Queue/Completion 复用已有机制。
   冷阶段匹配格式和本机资格并加载程序；热阶段一次 launch，无重新匹配/计时，无私有 allocator。
   16字节输入/输出对齐，workspace 为4×组数字节，动态 shared 为32×组宽×2字节。
4. 原版来源/许可证/重建与现有依赖声明同批接齐；普通 Direct/Graph、schema、资源/OOM/并发/
   Completion 保活一批验证。后续 DeepSeek 完整混合片段消费本机制；三交付和性能仍入 D5-A5/A6。

## 生命周期、成本与验证预算

调用者持有输入、输出和 invocation workspace 至 Completion 终态；准备程序可共享，workspace
由各 invocation 独占，异步完成保活程序。准备不读 tensor payload；shape/index/alias/resource
在提交前拒绝，native 失败走既有 Completion，失败结果不可消费。有限 BF16 输入包含可能因
量化还原溢出产生的 infinity；不承诺 NaN/Inf 输入的数值语义，也不增加热路径扫描。

P0：O(elements)，一次读/写与一份 F32 组 scale 临时空间，无新增 host 同步。现有原版 FP8 权重
解码、默认 Provider 注册和后端不重写。普通 C++ 只指定运算语义和 CUDA 路线；不排序 Provider。
source/static/shared 均编译同一内嵌程序，无运行时 Python/TileLang；离线重建工具及其许可单独固定。

预算：两个具名原版候选一次编译和原生数值批次；之后一批普通入口/资源资格、一次适用独立头检查；
仅具体失败允许定向修正，不扩充自研核、不扩大性能实验。本卡不运行 P1，待接口/完整消费固定后
统一进入 D5-A6。只跑 selector 所选门、来源/重建核对、docs/closure；未运行的交付/性能保留未决。

## 当前原生审计结果

原版数学已在 Windows RTX4070 / SM89 / Driver13040 上通过24个 native 案例，包含全部有限
BF16 编码、两次变化输入、midpoint ties、signed zero/tiny、跨32组尾块、输出及 scale guard。
组 scale 与输出逐位匹配独立参考；两份原版均可用，没有需要自写量化核的事实。
这不是 HYLE 普通入口 PASS，完整 ledger 仍144/176。
