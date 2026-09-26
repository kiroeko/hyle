# D5-F36：DeepSeek GPU 批次的两处接线修正

<!-- MODULE REVIEW: AUDITED -->

## SCOPED → AUDITED

- baseline：`92372a9404fcb6326385deb640cf156b67ed215b`，Windows canonical，工作树干净。
- owner：当前 D5 已 AUDITED 的 `compute_provider_ggml_metal` 与 `graph_route_vulkan`。
  本卡只增加下述两个有限 FIX；F33/F35 的旧失败、其它 owner、三交付/P1 待办全部保留。
- 触发：`92372a94` Apple 定向复验越过 Bool 上传后，原版 Metal 请求不存在的
  `kernel_mul_mm_bf16_bf16`；`6c27e592` Vulkan Graph 在执行前拒绝 SinkAttention。
- 源码审计：锁定 ggml `456172ec733a135778adcd32d00e576a58232e45` 的
  `ggml-metal/kernels/mul_mm.metal` 只实例化 `bf16_f32`，未实例化 `bf16_bf16`。
  现有 BFloatLinearF32 直接把 BF16 激活送给原版 MatMul，遗漏适配。BF16→F32
  是精确扩宽，原版 `ggml_cast` 已有 Metal 实现；权重仍保持 BF16。
  Vulkan Provider 已支持 SinkAttention，Direct 原生 sink 测试通过；Graph 的
  `mature_custom` 列表漏项，导致无法到达同一 Compute plan。

| FIX | 有限改动 | 适用验证 |
|---|---|---|
| F36.1 | Metal BFloatLinearF32 使用原版 cast 扩宽激活，再进入现成 BF16×F32 MatMul；中间 buffer 由 NativeProgram 持有，沿已有分配/失败/析构合同；复用 recurrent 中间分配代码 | 两入口 DeepSeek attention/压缩、既有 Metal Provider、BF16 投影及 recurrent/OOM 直接消费者 |
| F36.2 | Vulkan Graph 将 SinkAttention 送入已接好的成熟 plan，不要求用户提供 custom shader | 同一失败的 sink/attention Graph 用例，保留原 schema/误差门及其它 custom 拒绝 |

P0：Apple 每个相应 prepared program 增加一个 F32 activation workspace 和一个原生 cast，
不展开 BF16 权重，不转 host，不搜索或计时；workspace 在准备时分配，重放复用。
现有 MetadataArena 的12 tensor/32 node容量足够两输入、cast及输出；原生分配失败返回
ResourceExhausted，已分配部分由 State 析构释放。Vulkan 仅修冷分类，不增加 dispatch。
新增成本由 D5-A6 最终候选承担，三交付仍归 A5；不借正确性结果宣称性能完成。

预算：两处改完后各路线一个定向批次，具体失败最多一轮定向纠正；CPU/CUDA 未变证据复用。
不再执行耗时大型 MoE 整批。Vulkan 长位置三角函数误差和 attention 数值差异是独立未决项，
本卡不授权修改其数学、误差门或扩大上游版本。未通过项不计入完整画像。
执行 change-impact selector、closure、docs。无需重写公共验证驱动。

## 验证结果

不可变候选`91af7569`：Apple Direct19、Graph13、Provider7项定向全部通过；静态Provider12项
全部通过，包括新增BF16 activation workspace的逐分配故障注入及既有recurrent/完成失败。
Vulkan Graph原先拒绝的sink用例与两个custom拒绝回归通过；两入口attention/压缩仍有数值失败，
不由本卡分类修正代签。三交付/P1待办保留。原失败、增量复验及12个完整画像新增格见
[三路线收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/deepseek-gguf-three-route/README.md)。
