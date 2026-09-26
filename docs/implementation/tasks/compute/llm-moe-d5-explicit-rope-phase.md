# D5 F35：显式 RoPE 频率的原生适配精度

<!-- MODULE REVIEW: AUDITED -->

## 范围与证据

- baseline：`4f18811a45d6cb378388e975986f3b6d111c1565`；Windows 为产品真相。
- owner：`compute_ggml_runtime` 的私有 ggml graph 构造胶水，以及已处于 AUDITED 的
  `compute_provider_ggml`、`compute_provider_ggml_cuda`、`compute_provider_ggml_vulkan`。
- 触发：F34 完整 DeepSeek attention 与长位置压缩缓存的 CPU Direct/Graph 固定误差门失败。
  原始结果保存在 `.scratch/validation/d5-deepseek-focused-r2/cpu`；失败没有计入画像通过数。
- 已定位：三个 adapter 将显式 F32 frequency 翻译成 F32 reciprocal，再交给 `ggml_rope_ext`
  相除。公开语义及 MLX adapter 使用 position × frequency。以 frequency=0.4728708267211914
  为例，position=2380 时两种 F32 phase 差 0.0001220703125，65536 时差 0.001953125；
  与失败维度吻合。固定 ggml CPU `ops.cpp::ggml_rope_cache_init` 明确执行 theta/factor。
- 这不是固定模型原生 base/YaRN 参数接口的错误；那个接口有自己的相位计算顺序。
  本轮只修已有显式频率 Operation，不能用原生参数语义替换该公开输入。

## SCOPED → AUDITED 有限 FIX

| ID | 改动 | 验收 |
|---|---|---|
| F35.1 | 一份私有 ggml 构图 helper，复用锁定 ggml 的 cast/multiply/sin/cos/view/concat；直接构造 F32 phase，随后由成熟原语旋转；不写 CPU/GPU 数学 kernel | 长位置、正负/零频率、两种配对方式、partial rotary、batch/head/sequence 及未旋转尾部 |
| F35.2 | CPU/CUDA/Vulkan adapter 共用 helper；不改变公共 API、准入域、路由或上游版本 | F34 完整消费者和既有 RoPE 类型边界；三路线 owner 批次 |
| F35.3 | 保留首次失败和修正收据；Apple 既有乘法语义继续执行共享消费者 | 不放宽原门，不移除65536位置，不改独立 oracle 匹配错误倒数 |

`compute_ggml_runtime` 从 CANDIDATE 退回 AUDITED，仅增加上述私有构图函数。
[Metal 分配失败任务](llm-moe-d5-metal-buffer-failure.md)的 MB1–MB3 候选、故障证据和待完成的
D5-A5 三交付全部承接，不重新打开 Metal 修复或更改其 patch。

成本：phase/cos/sin 只按 sequence×rotary-pairs 计算并向 head/batch 广播；旋转及必要布局拷贝
仍由该路线的原生 ggml region 执行。由一个 fused RoPE 变为固定个数的成熟节点，workspace 和
dispatch 成本增加，不能沿用旧 RoPE 的性能结论。P0 检查节点数/冷 arena 容量，D5-A6 对最终
稳定路线做适用同语义 P1；保留融合成本，不把不等精度的旧 reciprocal 路线当强制基线。
没有新 executor、热路由、数学 kernel、依赖或公开字段。

预算：一次源级定界和一个三路线批量修复，CPU 定向失败用例先验，再串行完成 CUDA/Apple/Vulkan
当前批次；只允许一轮具体失败纠正。新能力或成本等级变化须重新定界；不无限追加数值姿势。
三交付与 P1 保持 D5-A5/A6 待验，单片段通过不代签完整画像。运行既有 selector、closure、docs；
registry 迁移使用本轮 baseline 作为 previous-ref。

## 首次修正结果

候选 `b111e3b7` 的 CPU 定向7项全部通过：既有成熟线性/RoPE/Hadamard与F16/BF16边界、
新增长位置/负频率/零频率/partial tail，以及Direct/Graph完整DeepSeek attention和压缩状态。
相同失败用例未修改oracle、误差门或位置；前后原始证据见
[显式相位证据](../../../../evidence/correctness/llm-moe-d5-real-fragments/explicit-rope-phase/README.md)。
同一候选随后通过CPU完整owner的9个CTest目标，包括404个GTest及executor/源码/原生分配故障
检查，无SKIP。GPU路线、三交付和P1尚未验收；不增加完整画像格，不写CLOSED。
