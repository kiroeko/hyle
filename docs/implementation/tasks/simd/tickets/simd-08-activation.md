# SIMD-08：Transcendental 与 activation

- ticket state：`QUEUED`
- kind：product capability/performance
- depends on：SIMD-07
- budget：1 工程日；性能预算 60 分钟

归属当前两个 activation 草稿头。审查 real/complex transcendental accuracy tier，以及 GELU、tanh-GELU、
sigmoid、SiLU、ReLU/leaky-ReLU、ELU、hard-sigmoid/hard-swish、softmax/log-softmax 的 allocation、alias、
out-of-place、NaN/Inf 和 reduction 合同。该有限前向集合由 ggml 与 oneDNN 的共同任务证明；Mish、Softplus、
backward 和 fused norm/GLU 不在本票。接受实现必须 O(n)、O(1) extra storage、无隐藏 heap/lock，并完整接入
target/runtime/build/test；否则删除草稿。

性能只测资格化的 exp、tanh-GELU、SiLU、softmax normal row；模型特化 fusion 仍由 compute 拥有。
