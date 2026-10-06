# C4 冷热边界与成本审计

产品身份为 `216242d2fabdfae8e69d00872786aac0148442c1`；本包新增的是公开 C++
组合消费者与资格，没有新增产品数学核、路由搜索或执行机制。

## 绑定与状态

- Direct 在 replay 前 prepare 每个实际 operation 并创建 invocation；Graph 在
  `compile/create_invocation/prepare_replay` 完成计划与 workspace 绑定。
  两入口的 replay 循环只更新 activation、cache 和索引，然后提交并等待。
  见 `tests/compute_route_cpu/d5_qwen38_official_test.cc` 和对应 Graph 消费者的
  `run_q24_semantic_fragment`。这些消费者的 Direct 展开不限制 Provider 的产品 region 能力。
- `src/compute_provider_ggml/ggml_cpu_provider.cc` 的 `create_invocation` 创建 ggml
  图、arena、私有 backing 与 CPU execution；`run` 只映射当前输入、校验索引、绑定
  backing 并执行原程序。没有在 submit 编译或选择 Provider。
- `src/graph_route_cpu/ggml_region.cc` 把 region invocation/submit 委托给同一原程序，
  不额外创建逐核动态搜索。不同 invocation 拥有独立可变 backing；共享参数是只读绑定。

## 编码与必需转换

- GGUF Qwen、Flash 按原 Q3/Q4/Q6/Q8/MXFP4 编码执行；动态 activation codec 是原
  ggml 计算合同的一部分。参考包含相同 codec，未以未量化 F32 点积代签。
- 官方 FP8 权重的 scale widening/必要 cold decode 在显式 preparation 执行一次，
  输出绑定为常量；完整 draft 共享 embedding/head 参数，没有每轮展开整个专家库。
- `src/compute_provider_native/torch_cpu_program.cc` 保留官方 FP8/FP4 原 ATen 程序。
  selected expert 的 encoded payload 可能按索引复制，重复专家也保留原语义；浮点
  decode/dot 临时按一个 K block 创建。动态 A8 编码、selected payload copy、
  ATen 分配、结果 copy 都是真实热成本，不能宣称零分配或严格私有内存上界。
- C3 的 NF4/AWQ/CT 是 caller 显式 cold decode 后持有普通矩阵；MLX 压缩路径按
  当前 K group 与选中专家执行。C4 没有将这些转换移入隐式重复全权重解码。
- 全词表 head 资格使用常量驻留分片降低验证 RAM；这不是产品模型分片引擎，
  不据此宣称整模型驻留、完整实权重推理或性能吞吐。

## 性能适用性

本批没有新增成熟 Provider 的赢家规则：新编码/官方程序使用唯一已资格的成熟 carrier，
普通浮点重叠域沿用未变的成熟默认规则与软件身份边界。不能把本包数值通过称为
oneDNN/ggml 新版胜负或 SOTA 性能结论。旧 Native WIN 对新 peer 不越界。

本审计是 P0，不是 P1 PASS。是否存在总矩阵尚未覆盖的同合同成熟重叠域或接入成本风险，
须在 C5 准入矩阵对账时逐项核定；若有，按总计划最多两个完整消费族的一份预算执行。
未出现这种风险时不为单 carrier 生成逐格式计时竞赛。
