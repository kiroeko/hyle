# R4 CUDA Direct 完整 FFN 基线比较

不可变候选 `a677565959ea5534c5aca4ce90b41d1740a4d1a5`，W1-WIN RTX 4070 / SM89、
driver 610.62、CUDA Toolkit 13.3.73、MSVC 19.50，engineering 身份。
合同 `compute-cuda-ffn-p1-v2`，native-runtime-pair：caller CPU2，process mask0x14，
AboveNormal；两个不同物理 P 核，SDK helpers 原生管理。运行前真实 DLL 邻置。

两侧都是 HYLE 的完整资源/完成/失败合同，固定 Marlin group128 W4A16 与 cuBLASLt，
仅激活分别为 HYLE scalar 与 SGLang 06992c92 precise vector。没有快数学或额外 FP16 中间舍入。
两个 shape、两个 placement、正反执行/分配顺序；各 stratum n=15 即裁决，没有启动 n=31。
同 workload A/A、五项控制、build/artifact 验证、完整 raw replay 都 PASS；5% 非劣门 PASS。

## 方向与限制

ratio>1 有利于 HYLE 激活。以下方向按每个 stratum 的 CI 与1比较，保留小幅 LOSS；
仪器原始 5% practical PARITY 不能解释为严格等速或 HYLE 胜出。

| Workload | paired median ratio 范围 | 各分层 CI 外界 | 分层严格方向 | 5% NI |
|---|---|---|---|---|
| decode-q4-ffn-m1 | 0.997986–1.001508 | 0.989634–1.010078 | 0 WIN / 0 LOSS / 8 INCONCLUSIVE | PASS |
| short-prefill-q4-ffn-m16 | 0.989008–0.999529 | 0.983660–1.014258 | 0 WIN / 1 LOSS / 7 INCONCLUSIVE | PASS |

CI 外界是八个区间的最小/最大端点，不是另一个聚合置信区间。完整结果见 summary.json。
没有证据签 HYLE 完整链优势；默认实现仍须按主卡完成选择/接入。此处不是最终路线 CLOSED，
不是模型吞吐，也不外推其它 GPU/driver/toolkit/dtype/shape。

## 保留

raw-bundle.zip 保留全部原始输出、A/A receipt、control、manifest 和采集 recipe。
可读 JSON 副本仅规范 LF；原引用路径及结果不变。原 output 为 `C:/hyle-scratch/r4-cuda-20260916/qualification/p1-compute-helper`，
build 为 `C:/hv4dp2`，实际 artifact 继续保留；不伪称脱离这些
文件也已取得永久 Formal 资格。sha256.json 记录归档身份。
