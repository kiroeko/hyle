# M1-C / 原生接入检查点

本目录保存[M1-C资源卡](../../../docs/implementation/tasks/compute/llm-moe-m1-cuda-admission.md)
的有界上游入口资格及后续Linux FlashInfer接入检查点，不是M1-C整包完成证书。
ggml CUDA普通接入、剩余平台/风险门和完整模型需求覆盖仍待验证；新增64域×四路线覆盖格为0。

## Linux FlashInfer产品接入（2026-09-20）

实现`67f3651f`；`6ef8884d`只补nvcc主机代码的私有符号可见性，数学与C++ adapter不变。
[接入收据](flashinfer-admission.json)及`flashinfer-admission-raw.zip`记录实际命令、原失败、
工具链和文件哈希。默认CUDA Direct/Graph接入原版head256、8/2 causal GQA，FP16/BF16，
batch2的prefill(q17/kv31)和decode(q1/kv513)。

- 6项WSL native/header门通过；FlashInfer五项测试包含两请求同时提交、三次复用、旧输出、
  放弃Completion、逐冷分配OOM回滚、域拒绝及部分原生失败。失败槽退役后，新槽可恢复。
- 源码嵌入、搬移后的static/shared安装消费通过；安装消费期间隐藏producer源码/构建目录。
  普通Direct/Graph调用方不排序Provider。首次共享消费虽通过，导出复核发现原版模板可见；
  修正后重新通过源码/shared消费，FlashInfer动态库只导出公开C++工厂。
- Windows GQA公共语义/旧CPU能力隔离4门通过；WSL同两owner的ASan通过。
  FlashInfer ASan及公共语义/FlashInfer UBSan通过，主机插桩不代签GPU设备内存安全。
- 原ASan在CUDA初始化前置失败；无HYLE的`cuInit/cudaGetDeviceCount`控制也返回OOM。
  按[NVIDIA说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-analysis-tools.html)
  设置进程级`protect_shadow_gap=0`后恢复。控制和产品均残留WSL `libcuda.so.1`的260+20字节，
  LSan只抑制该具名驱动库，HYLE/FlashInfer host分配仍检查；未关闭泄漏或越界检测。

managed派发含原版capability/occupancy/kernel属性查询，不承诺external capture或性能赢家。
Windows FlashInfer仍未准入；Mac GQA与CUDA-OFF门尚待运行。此检查点不解除自研冻结。

## ggml F32 Direct/Graph 接入检查点

[具名收据](ggml-f32-admission.json)与`ggml-f32-admission-raw.zip`保留Windows/WSL配置、编译、
原始失败、最终8项CTest门、二进制/源码身份及统一LOC。原版RMSNorm+残差资源测试在两环境通过，
WSL另有Begin/encode/End capture故障。普通Direct、完整Graph片段、ggml→cuBLAS串接及
无关cuBLAS节点插入后的数据流片段均通过；每invocation原生资源独立，变化输入/旧输出及
HYLE metadata逐分配OOM回滚通过。WSL Provider测试还验证失败Completion、退役槽与新槽恢复。

这些是in-tree构建的接入收据。ggml head256 attention、host sanitizer、源码嵌入/搬移static/
shared SDK与Mac CUDA-OFF尚未运行，不从FlashInfer的既有收据继承PASS。M1-C仍AUDITED，
新增签署覆盖格0；当前Compute+Graph全部实现为51,012行/217文件，不含测试。

已归档的FlashInfer ASan/UBSan/旧static构建与Windows GQA构建共393,920,250字节临时文件
已按明确路径清理；当前ggml构建、共享交付复用目录、依赖缓存和本地模型保留。

## ggml head256 attention 接入检查点

[具名收据](ggml-attention-admission.json)与`ggml-attention-admission-raw.zip`记录C4：
Windows/WSL各8项native/header门通过，FP16/BF16、batch2、8/2 GQA、head256，
prefill 17/31与decode 1/513，独立double oracle与原容差。双请求/变化输入、地址重绑、
旧输出、放弃Completion、零负scale及BF16大指数反例均通过。普通Direct/Graph在Windows
选择ggml、Linux保持FlashInfer；原F32 region和cuBLAS组合回归通过。

原版ggml的SM89融合attention可能将BF16 K/V缩成FP16。本接法对FP16保留原版融合，
BF16使用原版F32运算图，含score矩阵与转换成本；未修改kernel或降低精度要求，未签性能赢家。
原始编译/样例失败与最终运行、源码/二进制身份、统一LOC均在包内。ggml host sanitizer、
三交付、Mac CUDA-OFF尚待完成；本段不关闭M1-C、不签模型覆盖，当前51,158行/217文件。

## FlashInfer原版入口资格

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
不准入；全部诊断修改不进入产品。Windows相同计算需求已由上方ggml C4检查点签正确性，
其余M1-C收口门仍待完成。

## 文件

- [q-cuda-api.json](q-cuda-api.json)：上游/探针/二进制身份、限定结论与原始包SHA。
- `q-cuda-api-raw.zip`：实际探针、编译/运行脚本、每次原始失败、窄改动身份、生成host片段与
  WSL终态收据。未把编译产物和完整第三方源码加入Git。

所有性能比较与HYLE自研算子调查仍未开启。入口资格不能推出SOTA、Provider全量支持或模型完成。
