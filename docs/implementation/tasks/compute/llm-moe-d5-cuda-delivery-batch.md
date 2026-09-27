# D5-A5：CUDA 真实消费者三交付批次

<!-- MODULE REVIEW: AUDITED -->

- baseline：`894f22e8b`（Git解析完整commit）；承接当前D5，产品FIX为零。
- CUDA的22画像Direct/Graph正确性44/44已在ledger有不可变收据；本批只补A5。
  不等待其它路线，也不代签A6性能、Linux专有FlashInfer或其它路线的交付。
- 使用Windows RTX4070、MSVC及现有CUDA13.3工具链，低并行2、测试串行。

## 有限范围

1. 从不可变候选复制四个现有CUDA普通入口消费者及其测试头闭包到独立外部目录。
   编译只通过`hyle::compute`/`hyle::graph`公开产品消费HYLE；既有host oracle使用
   `hyle::simd`数值类型，因此外部fixture显式请求SIMD产品，CPU计算路线仍关闭。
   GTest及原版ggml-base仅用于fixture断言/量化，不作HYLE运行替代，不引用HYLE私有头。
2. 固定测试：两个入口各运行Qwen3.5官方真实BF16权重FFN、卷积/GDN状态重放，
   DeepSeek官方完整FP4专家/FP8共享MoE链，以及FP4/FP8 quantization roundtrip。
   共8个测试；实际case数、变化输入、实际状态反馈和原误差门继承原消费者，不重写oracle。
3. 先完成同一外部工程与当前shared SDK；再完成源码父工程和static SDK。
   三交付都使用同一候选/consumer/filter，installed形状从搬迁后的prefix配置运行。
   只链接公开HYLE产品；GTest和fixture量化器来自锁定独立源码。
4. 复用当前CUDA shared构建树；SDK构建使用tests/instrumentation OFF、install ON。
   source/static共用一个父工程静态producer；不为Compute/Graph再各建三棵树。
   DLL在可执行文件旁，由公开runtime staging接缝布置；构建和运行均隐藏窗口。
5. 外部编译记录include/link来源，运行时隐藏producer的运行产物目录和原staging位置，
   保留relocated SDK与consumer自身目录。临时改名在finally恢复，不改Windows产品源码。

预算：一批三交付及最多一次明确的consumer/API或安装接缝纠正；不改数学和容差。
若发现产品缺陷，先给当前owner定界有限FIX，不在验证驱动内补缺失的产品文件。
公共selector、docs/closure按本批变更选择；本批证据/consumer不触发P1。

## 结果

源码、static、shared均待运行。尚不签A5 PASS；没有新增完整画像正确性格。

### 首次外部编译的单次consumer纠正

`5f5b657c`的shared SDK已构建、安装、搬迁；外部消费者编译发现四个测试文件无条件
引用Linux专用FlashInfer头。Windows SDK正确省略未启用Provider，故不是安装漏文件。
有限纠正仅把该Provider的测试身份断言按公开`HYLE_COMPUTE_FLASHINFER`条件编译，
其它Provider、模型夹具和数值门不变。新consumer复用产品输入完全相同的shared SDK，
保留第一次失败，随后source/static消费同一纠正候选；不因此重编CUDA产品或放宽原断言。
外部构建另保留原Direct/Graph两个binary的边界，并带上各自已有的词表helper实现，
避免把两个不同执行入口的同名测试辅助符号合进一个binary；运行清单仍为总计8项。

后续两个binary均已链接，但在隔离PATH/producer时遇到实际CUDA SDK DLL缺失。
按[独立构建接缝修复](../build/cuda-sdk-runtime-staging-v1.md)绑定backend owner；
不在消费者脚本手工复制DLL代替产品修复，也不重开数学实现。
