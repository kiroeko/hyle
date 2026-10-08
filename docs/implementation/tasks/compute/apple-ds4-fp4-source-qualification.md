# Apple 官方 FP4：原 DS4 Metal pass 资格

<!-- MODULE REVIEW: AUDITED -->

父轮[Apple parity](apple-cpu-cuda-parity-closure.md)，baseline `d7c06ce74`；owner仍Apple三
AUDITED记录，不更改旧[F46](llm-moe-d5-apple-fp4-native-qualification.md)失败或预算。
这次不是重跑原MLX MXFP4 probe，而是新增具名成熟source候选：已经登记CPU来源的
antirez/ds4 `9139e2ae58a41503968a500f36f75895c1ba63fc`，原
`metal/dsv4_kv.metal::dsv4_e2m1fn_dequant`及indexer内完整group32 FP4 pass。
其原代码明确包含floor `7.052966104933725e-38f`、ceil power2 scale、饱和/RNE与乘scale。
另有原host `DS4_METAL_MATH_SAFE` 对应Metal Safe math模式的具名责任分支，测试只给本
候选library设置，不改变全局driver、系统或其它native库编译。

## 固定有限资格

仅原helper、原group32 reduction/量化pass与声明式IO wrapper；Hadamard输入是消费者的
另一步，本probe不借其旋转替换普通FP4 pass。原数学表达式保持，不能改任意codebook/
rounding/floor/scale算法。先用只读临时probe在当前Air验证全部finite BF16的原F32展开、
原CPU ds4公式参考、完整32组、normal/tiny边界；分别记录原默认及原Safe-mode source。
只做一批，不调整容差，不重复原F46命令，不跑P1；shader/ABI具名问题最多一次定向纠正。

结果通过后才重新固定有限产品FIX、公开Direct/Graph、完整expert/indexer片段、资源与三
交付；probe或floor存在本身不是路线PASS。若tiny仍不符，记录精确source/runtime边界并
裁决成熟缺口，停止同候选微调；不能悄悄用CPU模型计算或换BF16模型数学。此轮产品FIX零。
