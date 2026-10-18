# F44：原版 FP8×FP4 专家与完整 MoE

产品候选 `bc72243d49746ce1b1b938b31e2f37cffd5094c6`，Windows RTX4070 / SM89 /
CUDA Driver API13040。**两种官方 DeepSeek 的 CUDA Direct/Graph 完整 MoE 通过原门。**
这解决 F43 的具体消费者失败；完整 attention/compressor/MTP 尚未齐备，ledger 仍144/176。

## 结果

- 四个实际N/K、T1/3、共享或逐slot激活、两轮变化输入，在两个普通入口共64次专家投影，
  BF16输出逐值精确。源packed E2M1权重与UE8M0 scale保持原含义，没有热期解包权重。
- Pro/Flash各score/hash两种完整MoE、三轮变化输入、两个入口共24次执行通过原4%门；
  gate/up/down、pre-down路由权重、共享FP8专家和最终BF16边界均进入实际链路。
  未更改原输入或容差，F43失败保留。
- 本批21项GTest全部通过，零失败/零跳过；workspace、对齐/别名拒绝、元数据OOM回滚、
  双队列、异步保活与非法ID Completion失败均通过。独立头4项及模型输入工具13项通过。
- 原生40案例包含四形状、变化/重复/越界ID与输出/workspace guards，均精确通过。
  原生结果独立记录，不替代普通入口资格。

增量构建71.99秒，普通测试4.64秒。统一Compute+Graph实现60,150行/240文件，不含测试和vendor。

## 成熟来源、成本和限制

固定DeepSeek官方revision `b5968e9190ef611bbf34a7229255be88a0e937c1` 的原量化和FP4 GEMM。
HYLE仅增加checked设备索引和资源接合；未新增数学核。每次两次launch，caller workspace，
不下载专家ID、不热期搜索Provider。原M32 tile每slot仅有效M1的成本明确保留，未宣称大prefill最优。

八份程序完成离线重建。四份量化程序raw hash相同；四份专家程序原CUDA源码相同，整个ELF
仅`.strtab`中11个NVCC内部符号hash随源路径变化。原raw-hash失败日志保留。限定忽略这些字段后
全部二进制字节一致；指令、常量和ELF头的故障注入均被拒绝。没有更换产品内嵌程序或重跑产品测试。

完整官方画像、其它路线、DeepSeek GGUF Vulkan四格、A5剩余selected gates/三交付、A6 P1和D6
仍未完成。当前资格仅限上述硬件/驱动与形状，不能外推其它GPU或运行版本。

[收据](receipt.json)记录案例数量、范围、原始日志hash和待办。
`native-probe.py --artifacts <原始AOT目录>`复核原生40案例；普通消费者在仓库测试目录内。
