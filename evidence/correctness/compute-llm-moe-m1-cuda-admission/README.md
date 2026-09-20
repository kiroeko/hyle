# M1-C / Q-CUDA-API

本目录保存[M1-C资源卡](../../../docs/implementation/tasks/compute/llm-moe-m1-cuda-admission.md)
的有界上游入口资格，不是Provider完成证书。产品实现、两请求并发、失败寿命、三交付和完整
模型需求覆盖仍待该卡后续验证；新增64域×四路线覆盖格为0。

## FlashInfer原版入口

- 上游`69ff11fc4954396d98326656dc85debd2223f637`，archive SHA见[q-cuda-api.json](q-cuda-api.json)。
- WSL Ubuntu24.04、GCC14.2、CUDA13.3.73、RTX4070 SM89；原版CUDA头、不含Torch。
- head256，8个Q head、2个KV head；FP16/BF16各执行prefill(q=kv=17)与decode(q=1,kv=257)。
  每项3次调用，独立stream/device buffer/workspace，double oracle以实际dtype输入计算。
- 原始运行四项通过；中断工具调用未返回进程receipt，因此仅对同一已编译可执行文件再运行
  一次保留exit0收据，没有重新编译或扩展测试。最大绝对误差分别为0.000141220、0.000005271、
  0.001307193、0.000048135，门限FP16=0.002、BF16=0.01。
- 这里只验证当前形状的数值入口；不同项各有独立资源，不冒充同时在途请求或故障恢复验证。

## Windows准入裁决

MSVC14.50.35717/CUDA13.3.73的原版编译失败。保留标准预处理器修正，以及GNU属性、
LLP64 `std::max`、`ushort`的窄等价诊断。后续NVCC把一个host模板的限定符生成为裸`restrict`；
两次修正仍失败，第二次host宏污染SDK `__declspec(restrict)`。预算到此停止，Windows本pin
不准入；全部诊断修改不进入产品。Windows以ggml原版attention继续相同计算需求，尚未签PASS。

## 文件

- [q-cuda-api.json](q-cuda-api.json)：上游/探针/二进制身份、限定结论与原始包SHA。
- `q-cuda-api-raw.zip`：实际探针、编译/运行脚本、每次原始失败、窄改动身份、生成host片段与
  WSL终态收据。未把编译产物和完整第三方源码加入Git。

所有性能比较与HYLE自研算子调查仍未开启。入口资格不能推出SOTA、Provider全量支持或模型完成。
