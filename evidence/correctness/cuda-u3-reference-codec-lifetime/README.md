# CUDA Q3_K 整批失败：参考 capture 的跨程序寿命

## 根因

原独立源码650项消费批次的唯一失败是`g3560f36d25c826bd`：replay1的slot1
4096个元素均失败，actual／expected分别按8个prototype精确重复。
fresh原CUDA Q3_K的八输出与旧actual一致；把同一Q3_K bytes按IQ3_S执行时，
八输出与旧expected一致。原expected没有代表声明的Q3_K计算。

两种格式的块大小均为110 bytes／256元素；K=2048时元数据shape／nb相同。
原ggml CUDA capture缓存以首node地址为key，properties复制output node和source的
data／ne／nb，没有source type。test-only reference的thread_local backend比每次
临时graph／buffer寿命长，因此重新分配到旧地址的逻辑程序可能命中其它codec的capture。

原source反例位于`source-f99368c68335266b`：

- `src/ggml-cuda/common.cuh`：`ggml_cuda_graph::node_properties`；
- `src/ggml-cuda/ggml-cuda.cu`：`ggml_cuda_graph_get_key`／`ggml_cuda_graph_update_required`。

[最小反例源码](reproducer.cc)固定同址、相同shape／nb，将已暖机的IQ3_S source type
换为Q3_K；第三次仍精确返回IQ3_S结果。它确定地模拟回收地址后的cache属性碰撞，
没有声称保留了历史运行的allocator地址trace。错误codec的八输出指纹与原日志逐值一致，
加上缺失type的原源码与确定复现，足以作本轮工程根因。

## 有限修复与边界

只把`tests/compute_models/cuda_sidecar_quant_reference.h`中的projection／grouped
reference backend改为各自逻辑program的局部owner。原codec、输入、GPU数学、误差门不改。
HYLE产品每Invocation独占backend且recipe不可变，没有修改任何产品cache或kernel。

回归同时检查不同codec的CUDA数值与capture寿命：旧thread_local版本会观测到跨已销毁
program的capture暖机，旧owner回归FAIL；局部owner修正后不得出现该capture。
最初只比较数值的短序列恰未撞地址而PASS，原记录也保留；不能把它冒充旧版稳定数值复现。

本批只编译受影响test TUs，资格化格式切换与fixed物理编码组；未重建产品或启动P1。
原整批FAIL留档，分类为无效reference的误报；不会把原650项日志改为PASS，也不代签
尚未完成的source/static/shared、mixed或U4/U5。

## 验证

原程序指纹、固定同址反例、旧owner FAIL及修正验收的结果见`receipt.json`和`evidence.zip`。
修正后旧owner回归／两codec consumer共3项通过；独立static SDK的Direct／Graph固定物理
编码组各560项通过、无skip。模型输入检查13项与docs／closure通过。没有重跑650项整批。
