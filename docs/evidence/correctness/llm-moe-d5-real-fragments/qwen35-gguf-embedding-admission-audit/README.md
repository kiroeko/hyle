# Qwen3.5 GGUF embedding 原生行读取准入审计

- baseline：`0c7293ff38ac078c095e623062398087a5a89ba1`，Windows canonical 工作树；
- consumer：普通 CPU Direct `prepare`，精确
  `Qwen/Qwen3.5-2B@15852e8` / `unsloth/Qwen3.5-2B-GGUF@f6d5376` /
  `UD-Q4_K_XL` workload；
- 形状与格式：Q6_K、`[7,2048]` embedding table，I32 两 token 行号，F32
  `[2,2048]` 输出；隐藏宽度没有缩小；
- 结果：当前 adapter 在 prepare 阶段拒绝该合法 GGUF 输入，最终错误为
  `hyle-native: native indexing requires plain scalar tensor storage`。定向测试把此拒绝锁定为
  修复前收据，见 [cpu-admission-test.txt](cpu-admission-test.txt)；
- adapter 根因：CPU/CUDA/Metal 只允许 GGUF storage 出现在 MatMul/GroupedMatMul 权重端口；
  Vulkan 的 Gather 资格域只接受 plain I32。各 adapter 尚未把 embedding Gather 接到已有的
  `ggml_get_rows`；
- 固定上游能力：vendored ggml `b9d43bc809551dde` 的 CPU `ops.cpp`、CUDA
  `getrows.cu`、Metal `quantize.metal` 与 Vulkan pipeline inventory 均具名支持
  Q4_K、Q6_K、Q8_0 的 GET_ROWS。这里不存在需要 HYLE 新写的解量化或 Gather 数学。

## 裁决

这是四个 ggml adapter 的有限接线缺陷。D5-F8 只放通 rank-2 embedding table、axis 0、I32
indices、dense F32 output 的 Q4_K/Q6_K/Q8_0 域，复用固定上游 `ggml_get_rows`，并保留 checked
越界失败不发布输出。其它 GGUF 类型、轴、输出类型、路由优先级与自研数学均不扩张。
