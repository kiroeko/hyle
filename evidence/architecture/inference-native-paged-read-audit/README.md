# 原生分页读取的有限源审计

固定MLX1f8e74e3与ggml456172ec，来自实际Air原锁缓存；HYLE既有patch manifest另行核验。
[原片段与文件](fixed-source-reads.zip)、[哈希/结论](receipt.json)；没有升级依赖或执行数学。

MLX实际NAX路径的kU=16，但Ktile<T,2,1>一次读32 rows，任意page16不能仅换tile base。
vector/2pass、旧steel loader和NAX（包括dsplit）读取与tail必须分别审计，保持原数学。
Vulkan原push constants为128bytes已到原保证上限，不能为pages给普通layout加字段。
原HYLE FullF32 tag实际强制scalar/f32 shader并拒绝dequant/repack；此已资格路径先保，
不把pristine coopmat源码当成HYLE已有同数学资格。BF16 carrier存在也不等于当前SDPA接受。

共用MaterializedNativeState归compute_batch，原native bindings、metadata bank、精确frame
校验和retirement须复用。公共typed调用和新的页metadata/controller形状仍须在AUDITED
前冻结，本证据不是性能、数学、交付或整个G01/G02完成证书。
[施工卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-native-paged-read.md)。
