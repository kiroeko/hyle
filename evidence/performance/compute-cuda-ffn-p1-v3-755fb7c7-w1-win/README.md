# R4 CUDA Direct 最终默认 FFN 验收

产品候选 `755fb7c76132908e15cc8e2c298c69c227af775a`；资格化 checkout
`f01e92379048b920b2545fa228921ef95038d5b3` 仅追加状态/文档/正确性证据，产品、benchmark、工具输入不变。
W1-WIN RTX4070/SM89、driver610.62、Toolkit13.3.73、MSVC19.50，engineering 身份。
合同 `compute-cuda-ffn-p1-v3`，caller CPU2、process mask0x14、AboveNormal；两颗独立 P 核，
SDK helpers 原生管理，运行前物理邻置 CUDA/CRT DLL。

两侧都使用完整 HYLE 资源/完成/失败合同、Marlin group128 W4A16、cuBLASLt，以及同一 pin 的
SGLang precise SwiGLU 数学实现。一侧使用自然默认 factory，另一侧使用独立精确 reference adapter。
该比较检查默认接入成本，**不证明 HYLE 算法胜过 SGLang**；采用成熟算法的依据是此前 v2 完整链比较。
M1/M16、双 placement、执行和分配双顺序保持原合同；原始样本数和资格详情保留在 manifest/summary。

qualification: **QUALIFIED**。各 workload 的非劣结论与严格方向如下。
ratio>1 有利于发布路线；保留小幅输赢，不把5% practical PARITY解释为严格等速。

| Workload | 分层 median ratio 范围 | 各 CI 端点外界 | 各分层严格方向 | 5% NI |
|---|---|---|---|---|
| decode-q4-ffn-m1 | 0.996930–1.004249 | 0.986473–1.012000 | 0 WIN / 0 LOSS / 8 INCONCLUSIVE | PASS |
| short-prefill-q4-ffn-m16 | 0.998512–1.003080 | 0.988165–1.014748 | 0 WIN / 0 LOSS / 8 INCONCLUSIVE | PASS |

CI 外界不是新的聚合区间。性能只签测过的两个 resident FFN 原型与当前硬件/软件；不是模型吞吐，
不外推其它 GPU、驱动、Toolkit、dtype 或 shape。裸 native 成本在另一份 NATURAL_ROUTE 证据中报告。

raw-bundle.zip 保留所有原始输出、A/A、controls、manifest、验证结果及 recipe；sha256.json 记录归档。
原始 output `C:/hyle-scratch/r4-cuda-20260916/mature-rule/p1-compute-final`、build `C:/hv4dp3` 继续保留，
可读 JSON 只规范换行。不伪称脱离实际 artifact 路径取得永久 Formal 资格。
