# D5-F41：官方 FP8 KV 权重的一次性准备

<!-- MODULE REVIEW: AUDITED -->

## 合同与 owner

- baseline：`f0ee1e9d`；Windows canonical。`compute_ops`持有公开语义及模型consumer，
  `compute_provider_cuda`持有原生程序和依赖声明，`compute_route_cuda`仅重开默认Provider登记。
- reopen trigger：GLM-5.3官方`kv_b_proj`为E4M3FN/F32 block128 `[28672,512]`，成熟模型在加载时
  解码BF16再拆分/缓存K/V。现有HYLE只接FP8矩阵乘，没有这个原生准备入口，普通默认CUDA无法绑定。
  不重开其它CLOSED组件，不恢复旧HYLE数学。
- 主纵切：原FP8字节上传→成熟解码→既有View/Copy分离K/V→作为常量绑定MLA/NextN，重复执行不解码。
  正交纵切：N129尾块、F32/BF16输出与变化源权重，证明接口不绑定某个模型名称或只支持一个形状。
- 来源：固定SGLang `06992c92` 的 `deepseek_weight_loader.py::post_load_weights` 调用
  `fp8_utils.py::block_quant_dequant`，再split/transpose/contiguous。原生候选为
  [DeepSeek官方程序](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/kernel.py)
  `weight_dequant_kernel`；MIT，仅导入未改数学及已生成程序，不引入Python运行依赖或模型加载器。

## 有限 FIX 与冻结接口

1. `model_ops::Float8WeightDecode`：输入为既有逻辑`[K,N]` block128权重描述，输出明确为
   dense `[N,K]` 的BF16或F32解码矩阵；有限类型、正维度、原布局、溢出与别名规则先检查。
   这是权重准备的原子操作，模型如何拆分/缓存属于consumer；不让描述符prepare读取tensor payload。
2. 同一CUDA模块注册独立DeepSeek Provider身份，绑定固定SM89/Driver13040原版AOT解码程序。
   一次launch，无workspace/独立运行状态、无运行时编译/搜索；复用现有Queue、Completion与资源检查。
   新vendor只用现有CMake依赖声明和license安装，不改公共构建或验证系统。
3. 普通Direct/Graph入口资格：原始FP8实际解码，尾块、编码边界、形状/资源拒绝、OOM/Completion保活、
   并发及重复执行。之后共享GLM consumer在冷阶段执行一次解码与布局准备，热片段引用缓存；
   实际MLA/NextN状态反馈和全词表head按原数值门，不用host数学制造缓存来冒充运行资格。
4. 源码/静态/动态的声明和安装随实现接齐；适用最终外部journey与P1纳入已有D5-A5/A6批次，
   本卡局部通过不代签它们。普通用户只选CUDA路线，不选/排序Provider。

## 生命周期、成本与验证

caller拥有源/目标storage至Completion终态；成功后可释放FP8源，保留解码及重排后的K/V常量。
准备对象持有native module，Completion保活；并发调用只读程序、独立输出，没有可变共享workspace。
资源/descriptor在提交前检查；native launch失败走已有异步终态，不发布失败结果。scope不承诺取消GPU。

P0：解码O(NK)只在消费者准备阶段发生，持久BF16缓存2NK bytes，临时完整矩阵与拆分Copy峰值明确计费；
热MLA仍由既有成熟BF16 grouped调用执行。没有每token解码、额外CPU数学或CPU+GPU联合模型执行。

预算：一次原生资格批次，一次schema/Provider/普通入口批次，接齐共享consumer后一次组合批次；
只针对具体失败修正。16项原生资格已运行通过（全256编码、N1/129/576/28672、BF16/F32、变化输入、
输出guard），不作为HYLE入口通过。低并行-j2、重型串行、复用现有CUDA树，DLL邻置。
预声明：owner/schema与直接消费入口、来源事实、AOT重建、selector所选组件/依赖/头检查、docs/closure。
如果固定程序或实际缓存边界不成立，保留精确缺口并停止该路径；不写自有解码数学或无限扩大调查。

## 审计结论

原版程序已按原BLOCK128、默认4warps/3stages离线编译为BF16/F32两个cubin，native共享内存均为0；
原算术/索引/掩码未改。Windows实际16案例逐值与独立解码一致，NaN位置和输出guard一致。
存储变换BORROW原DeepSeek程序，native ownership ROUTED到现有CUDA backend；模型K/V拆分与缓存
生命周期为APPLICATION，View/Copy复用现有能力。默认Provider登记是唯一重开路线的变动。
上述FIX及成本已冻结；当前普通C++、资源和完整模型链资格尚未运行，不能据native探针写PASS。
