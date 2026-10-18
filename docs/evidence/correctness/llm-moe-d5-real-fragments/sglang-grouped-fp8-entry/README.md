# SGLang grouped FP8：CUDA 普通入口局部验收

不可变实现候选 `9a7d5e5c790d08c8d67bf079b214569fe9cbcc9c`，Windows canonical、RTX4070
SM89、CUDA Driver API13040。**本批通过局部能力；完整画像仍140/176，不签整个goal完成。**

F40 适用域补注：本批 Flash 输入未达到 `swiglu_limit=10`，不能外推截断分支。F40 已补接原模型
截断，并通过跨阈值激活与完整 MoE 回归；新增混合注意力仍未通过，见
[当前检查点](../glm-official-mixed-checkpoint/README.md)。本页原始测试事实保留。

## 已实现和验证

- 原SGLang group128量化、原CUDA small/generic专家排序、原grouped FP8 GEMM接入同一Provider。
  GEMM默认tile64×128×128、4warps、3stages；无计时搜索、逐专家host调度或权重解码。
- 新公开`grouped_float8_linear()`：编码权重[E,N,K]，BF16/F32激活[T,A,K]，I32 IDs[T,8]；
  A1共享一次量化，A8保留每分支激活。独立权重描述保留所有codes后接所有F32 scales的布局。
- **224次普通Direct/Graph专家投影**：两种GLM真实H6144/4096、I2048、gate/up与down，
  BF16/F32、两种激活布局、两轮变化输入/重复ID；另有N129尾块、T17/129、E9/256/288。
- **8次完整GLM MoE执行**：两模型×两次实际router成员变化×两入口，真实router宽度256/288，
  9个resident experts保留原hidden/intermediate。独立double参考检查router→专家→激活→
  down→加权聚合及共享分支。投影保留F32 abs2e-6+rel2e-6、BF16 1ULP；整链沿用既定BF16门。
- 分配失败逐点注入、两invocation并发、workspace短缺/错位、输入错位、冷范围拒绝、
  销毁Prepared/Invocation/Context后Completion保活，负数/越界/I32两端ID安全失败均通过。
  所有失败ID在专家寻址前检查；有效ID的原排序/scan/矩阵数学不改。
- 四个CTest factor共**23个GTest**通过、零SKIP；两个独立头检查通过。
  F38动态投影/共享FFN同批回归。构建11.67秒、四factor验证11.01秒。
- 原生资格80例、bounds排序24例保留；5个AOT程序按固定Triton3.8/CUDA13.3重建，header字节一致。

## 成本与剩余门

每invocation持有16字节对齐的激活codes/scales、sorted IDs、expert IDs、post-pad计数，
generic路径另有cumsum。原CUDA小批条件为slots<1024且E+1<=64；其它采用原generic两步。
一次提交为量化+排序+GEMM共3或4次launch，另有既有Queue的status初始化/完成处理；
没有host读回专家ID或host同步。容量和原生I32索引在准备时检查。

这是Windows CUDA局部正确性/资源收据；selector所列跨路线、交付和其余适用验收仍由D5-A5/A6
批次承担，未运行不记PASS。四个官方FP8画像的32格及DeepSeek GGUF Vulkan四格仍未签。
统一脚本统计：**59,645行／237个实现文件**，不含测试和第三方生成程序。

## 初次辅助断言失败

[初次记录](initial-tests.xml)中资源测试硬编码了绝对1e-4，在输出1152/3200等处误拒约一个F32
舍入单位。更正为施工卡预声明的abs2e-6+rel2e-6后定向通过，并在上述不可变候选重验通过。
没有修改产品程序或精度门；最终原始输出见[native-tests.log](native-tests.log)，身份和计数见
[receipt.json](receipt.json)。
