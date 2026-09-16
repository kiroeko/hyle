# R3 CPU Direct 默认入口工程 P1：PASS

不可变候选 `d9712e68d2b812449a170f474d71a347a27b0ba1`，合同 compute-cpu-route-p1-v3，
W1-WIN / MSVC 19.50 / caller-thread 串行；两种 placement × 执行/分配正反，n=15 即裁决。
默认 CPU 入口与明确 Native 使用同一输入/数值/存储/线程/完成合同；M=32 的默认规则选择 oneDNN，
M=1/M=4 以及转置输入域保留 Native。oneDNN 固定 71094df451909e95ceeefc8f64c275cdcbbcdff5。

qualification_gate=PASS，performance_gate=PASS；实际 workload A/A、五项公共 controls、
build/artifact validators 及 deep verify 全部 PASS。本轮没有启动 n=31，原误用 Formal 的失败包保留。
普通 Windows/WSL/两 Mac 三交付与声明 sanitizer 见
[CPU 正确性证据](../../correctness/compute-cpu-r3-foundation/README.md)。

## 全部性能方向

ratio>1 有利于默认入口。严格方向检查每个 CI 与 1 的关系；非劣 PASS 不自动改写成 parity。

| Workload | median ratio 范围 | 全部分层 CI 外界 | 严格方向 | 非劣门 |
|---|---|---|---|---|
| decode-row-major-f32-matmul-m1-k1024-n1024 | 0.996999–1.001596 | 0.970106–1.021232 | INCONCLUSIVE | PASS |
| decode-row-major-f32-matmul-m4-k1024-n1024 | 0.997917–1.003004 | 0.993853–1.006296 | INCONCLUSIVE | PASS |
| short-prefill-row-major-f32-matmul-m32-k512-n1024 | 1.047825–1.063461 | 1.027823–1.076037 | WIN | PASS |
| decode-transposed-weight-f32-matmul-m1-k1024-n1024 | 0.997157–1.003399 | 0.981472–1.011034 | INCONCLUSIVE | PASS |

## 保留与限制

raw-bundle.zip 保留完整 output，包括所有原始产品/对照/A/A 输出、manifest、observations、
实际 driver、非交互 Windows recipe 和 verify。JSON 可读副本仅规范 LF 行尾，结果/引用路径不变；
SHA-256 在 sha256.json。原工程文件与 build 保留在 Windows
`C:/hyle-scratch/r3-cpu-20260916/p1-direct-engineering` 及 `p1-direct-engineering-build`。
工程身份/文件可复核，不声称脱离这些 artifact 的永久 Formal 资格。

该证据只签四个冻结的 Direct 默认入口 workload，不外推整个算子目录、Graph 或任何 GPU 路线。
正在进行的 Graph 私有 replay 修正没有修改 Direct/Compute/Native/oneDNN 实现；不因文档或该独立修正
重新采样 Direct。完整 CPU R3 仍须完成 Graph 的实际门。
