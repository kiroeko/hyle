# M1-A：原版 ggml Metal / MLX 接入验收

- 候选：`140054b13ce2515a05bc6eedac7e03e56bbc3662`。
- 范围与七门：[terminal card](../../../docs/implementation/tasks/compute/llm-moe-m1-apple-admission.md)。
- 结论：A1–A6 PASS / P0；M2/M3 完整模型覆盖尚未验收，新增 0/256。未运行 P1，无自研比较。

## 原始证据

| 收据 | 实际内容 |
|---|---|
| [schema-windows.json](schema-windows.json) | 公共 GatedDeltaNet schema、合法/非法数值与 shape/arity、公共头 |
| [native-air-checkpoint.json](native-air-checkpoint.json) | Air 普通 Direct/Graph、原生资源入口、数值/状态与 MLX 组合；Windows OFF、selector 59 项 |
| [resource-host-air-delivery.json](resource-host-air-delivery.json) | 原生故障回滚、checked failure、Air host ASan/UBSan；Air 三交付；Windows runtime/OFF；完整产品 hashes |
| [cross-native-delivery.json](cross-native-delivery.json) | Mini 原生与三交付；WSL runtime/OFF/UBSan；可追溯前向完整快照 |
| [selected-final.json](selected-final.json) | Windows 119 个原 selected gates 精确集合、最终复核、跨候选产品相同依据、失败及逐项日志导航 |

每份 JSON 指向对应 `*-raw.zip` 和 SHA-256；ZIP 保留实际执行脚本、完整命令、环境、原始失败和
修正后输出。checkpoint 中的 pending 是其当时状态，最终状态由 terminal card 与最终收据裁决。
后续三个 CUDA fixture 文件修复不改变 286 个已记录产品输入；1,719 个非文档输入与验证快照相同。
原候选的 PASS 保留原身份，未重写 selector cache。四次环境/fixture 失败未被删除。

## 机器、边界与成本

Air/Mini 分别用 Mac16,12/Mac16,10 M4 原生 Metal；Air macOS27.0/26A428、clang21.0，Mini
macOS26.6.2/25G83、clang21.0。Windows W1 用 MSVC19.50、SM89 做选中的公共回归；WSL
Ubuntu24.04/GCC14.2 做 Apple-OFF、公共 runtime 和 UBSan。精确工具路径/版本/命令见原日志。
Mini 只签正确性/portability，不冒充 Air 性能机；没有跨设备 SOTA 结论。

原版 Metal 域为 dense F32、equal heads/square state、D32/64/128、1≤T≤32767 与 INT32
索引上限。strided MLX 输入的必要 GPU contiguous copy、每 invocation 的 backend/metadata/import、
每次 packed 输出分配及跨库 checked host wait 都计入 P0；没有 CPU 模型数学、影子 payload copy、
热 Provider 搜索或计时。MLX 原生 region 的内部融合保留，不声称零分配/recorded replay。

ASan/UBSan 只签 host owner；Metal driver/device 不在插桩范围，macOS leak detection 未启用。
native rollback 另有 live-owner 断言，失败实例先 drain 后退役。源码/设计120分钟预算用尽并停止。

## 临时文件

Air/Mini/WSL 完成的 source/build/SDK 副本已删除，清理收据在最终原始日志包。
Windows 仅在最终日志归档且无进程使用时，按任务 marker 和 canonical CMakeCache 校验后清理
本卡 `C:\ha\e8e0c0e99c7b4f69d7a8` 与 `C:\hyle-scratch\m1a-schema`；共享依赖缓存和模型保留。

Windows 本卡构建/SDK 目录现已清理，共 19275326511 bytes；见[清理收据](cleanup-windows.json)。
