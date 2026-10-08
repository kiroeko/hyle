# CUDA U2/U3：联合资格完成

2026-10-08，U2构建交付与U3成熟接线的冻结范围已完成。数学/source-id
`f99368c68335266b`，联合source快照`0c394c59443f1239fb3894446ca95300d5ec8ddb`；
之后仅工具/测试控制/证据变化。本页是包级资格，不把owner尚待U4的默认/性能事项或
U5/U6写作CLOSED，不解除CPU/其它路线历史状态。

## 已实现与验证

普通compute-cuda preset、CPU=ON高级mixed、driver release/API/runtime/toolkit独立冷事实，
以及stock GGUF、原narrow IQ1、N1原NF4/nested/AWQ/CT、N2原MLX公式、必要完整region/state
已实现；大embedding、重复expert IDs、F32精度与原pool接缝已修正，没有扩张自研数学。
Windows/Linux普通Direct651项、Graph649项固定联合数学请求通过，无skip；资源/失败/close/
独立槽/replay/精度/公开头证据见[native资格](../../../../evidence/correctness/cuda-final-u3-f32-native/README.md)、
[N1](../../../../evidence/correctness/cuda-final-u3-artifact-carrier/README.md)、
[N2](../../../../evidence/correctness/cuda-final-u3-mlx/README.md)、
[软件事实](../../../../evidence/correctness/cuda-final-u23/README.md)。

| OS/选择 | source | static SDK | shared SDK |
|---|---|---|---|
| Windows CUDA-only | PASS | [PASS](../../../../evidence/correctness/cuda-final-u3-windows-static/README.md) | [PASS](../../../../evidence/correctness/cuda-final-u3-windows-shared/README.md) |
| Windows CPU+CUDA | PASS | [PASS](../../../../evidence/correctness/cuda-final-u3-windows-mixed-static/README.md) | [PASS](../../../../evidence/correctness/cuda-final-u3-windows-mixed-shared/README.md) |
| Linux CUDA-only | PASS | [PASS](../../../../evidence/correctness/cuda-final-u3-linux-static/README.md) | [PASS](../../../../evidence/correctness/cuda-final-u3-linux-shared/README.md) |
| Linux CPU+CUDA | PASS | [PASS](../../../../evidence/correctness/cuda-final-u3-linux-mixed-static/README.md) | [PASS](../../../../evidence/correctness/cuda-final-u3-linux-mixed-shared/README.md) |

source逐格命令/结果在对应SDK归档。SDK实际install、prefix搬移、offline独立配置/链接/运行；
Windows DLL邻exe/精简PATH，Linux实际RPATH/cache加载/LD_LIBRARY_PATH未设。
最终shipping覆盖60个GGUF角色/type、34 non-GGUF及必要完整程序/state/生命周期；完整shape
数学由同原源native/Windows static全集复用，不把317个shipping检查写成559 shape全集。
mixed完成CPU F32/Q8/Graph replay，不做模型级CPU+GPU联合推理、不重开CPU性能轮。

## 七门与限制

scope/architecture：共同固定真实tensor合同、原八族/104GGUF/559数值合同/45plain/34packed/
三MTP范围保持；成熟融合/资源不拆为逐kernel万能执行器。
correctness/delivery：上表及各子批实际通过，首次FAIL/timeout保留。
usability_docs：用户只选路线，必要Provider自动组装，DLL/SDK/cache沿公开recipe。
red_team：TF32分发、重复ID、大embedding、reference retired-capture/公共writer的具体缺陷
已修并复核；没有放宽精度门/删stage/用copy或mock签路线。
engineering_performance：本包P0接缝成本/资源资格；P1/best-of在U4，未运行不PASS、不签WIN。

资格限实际SM89、Windows/WSL工具链与锁定版本，未测硬件不PASS。第三方private cache/
分配仍为managed未知上界，不承诺strict prepared无分配。Linux合并packed进程近12GiB且
超时的原FAIL保留；同34完整case按进程隔离、原组预算/逐名union通过，没改shape/精度/数学。
全权重整模型驻留、loader/tokenizer/采样/引擎、HIP/分布式不在本轮。

## 有界收尾与下一包

用户恢复09:59:04 +08:00，四个剩余组合运行结束11:15:13，**76分09秒**，低于120分钟预算。
[机器汇总](../../../../evidence/correctness/cuda-final-u3-joint-delivery/final-delivery-summary.json)
保存原步骤/尝试/时间；241步最新结果全PASS，原失败没有改写。全Compute+Graph由统一脚本
复算**60,658行/253文件**，不含测试/bench/第三方源码。

仍待U4合法默认/有限best-of、U5自研CUDA清退、U6终审。当前新CUDA goal工程估算约55%，
不称整体完成；下一包产品实现前按实际owner重新定界、固定有限FIX。
