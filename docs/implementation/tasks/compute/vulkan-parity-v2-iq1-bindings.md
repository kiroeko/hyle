# Vulkan V2：三种原 IQ1 专家程序

<!-- MODULE REVIEW: AUDITED -->

baseline `db77f733d25486656aa86addab51667ba3696ef1`；owners 为
`compute_provider_iree_vulkan`、`compute_route_vulkan`、`graph_route_vulkan`。
属于[父轮 V2](vulkan-cpu-cuda-apple-parity-closure.md)，不重开 CPU/CUDA/Apple。

## 需求与承载

[共同端口表](cpu-final-demand-carriers.json)仍有 8 个 Expert 合同使用
IQ1_XS/XXS/XXXS：XS 两项、XXS 两项、XXXS 四项；K=2048/3072/3584/8192，
N=2048/3072/3584/8192，原专家银行 E=512/896。锁定 ggml Vulkan 无相应原程序。
已登记 Unsloth `a809594fd2035979ad2b7ef23398127c1d07e891` 有三原 GPU 网格及
完整 F32 激活程序，见 `third_party/ggml_metal_iq1/origins.json` 与 `patches.json`。
按原 packed byte、网格索引、delta/scale 和 32 元子组贡献表达为标准 tensor/arith/
linalg/scf，沿既有 IREE compiler/runtime 原版本离线生成；不写新数学、GLSL 或量化器。

主纵切为普通 GroupedMatMul：原 E/N/K bank、共享或分立 F32 activation、checked IDs、
F32 输出；changed weight/activation/IDs 与重复执行。正交纵切为 packed/index/storage
边界及两个独立 invocation 的 lease/cancel/close/OOM。不新增模型加载或 host 数学 fallback。

## 有限审计与 FIX

| ID | 缺口 | 允许改动与验收 |
|---|---|---|
| I1 | 原三 GPU 格式未承载 | 原网格、block layout、子组贡献及原序列 provenance；三个离线 entry |
| I2 | 现有 GroupedMatMul 没有绑定 | 私有 native spec/三个输入映射，普通 Direct/Graph hook；原 checked 发布 |
| I3 | 全银行浮点展开不可取 | 仅当前 selected 32 元子组；冷常量复制复用，mutable 真实复制与 native 完成计费 |
| I4 | 微型测试不能签实际 K/N | 三格式双 activation 档、多组重放、编码边界；8 合同 K/N 与原 bank 索引范围资格另批集中验收 |
| I5 | 三交付必须可离线复现 | 原 source/hash/许可证、内嵌 VMFB 与同版本 recipe；父轮 V4 集中三交付 |

复用公共 GroupedMatMul、TensorDesc、Mem quota、Vulkan resource/queue/fence、既有
safe-index/checked-copy、IREE native 完成；不修改 CLOSED shared ggml core。
普通 API 与 source/static/shared 产品名保持。只有上述三格式选择这个 carrier，已有格式不变。

P0：一子组 workspace O(T*S*N*32)，没有 O(E*N*K) F32 bank；native loop、内部
分配/复制/同步全部计成本。单 mature carrier 不造性能竞赛，父轮 V3 正式 P1 尚未签。
调查/接线预算 60 分钟，资格纠正 30 分钟；每个具名源码/call 错误一次定向纠正。
若原程序/设备/容量不合格，报告具体未资格域，不改误差门或擅写新 kernel。

预声明：Windows 同一 producer build2/test1，一候选批末构建，三格式 Direct/Graph
独立差分、changed 输入与 duplicates/坏 IDs、empty/overflow/features/OOM/lease/
cancel/close；已有共享资源域仅必要 delta。Air/WSL/三交付/完整模型片段由父轮集中验，
本包不能代签。文档、registry 只跑 closure/docs/whitespace，不构建或 P1。

七问：原程序身份和许可具名；两纵切与自然入口具名；资源/checked/类型复用；P0 成本
明确；无公开 C/FFI；offline 交付沿现有机制；用户只选 Vulkan，缺能力冷期具名拒绝。
当前只定界，runtime 尚未运行，父轮完整目录与所有平台/交付门仍未完成。

## AUDITED：原程序与接线冻结

已核对三原 CPU codec、三原 GPU 网格/Metal 程序与现有 IREE selected-group 接口。
每 block 256 元，XS/XXS/XXXS 为 46/42/38 bytes；原 GPU grid nibble 为 0/1/2，
32 元组贡献为 d*(dot(grid,A)+sum(A)*(-1±0.125))*(2*scale+1)，不得误用 CPU
-1/0/1 网格偏移。共享/分立 activation 与 IDs 按现有 GroupedMatMul schema，原
expert bank 原样 U8 承载，native 只展开 selected 子组，输出 F32；坏 IDs 先 safe
索引后 checked 发布。原循环/舍入误差按已有 Vulkan 矩阵门，不制造逐位同 CPU 承诺。
原下层 CLOSED core 不修改；本包最多三个私有 entry，不扩公共 API。I1–I5 固定后施工。
