# C3 独立工具修正：组合路线参数与 SDK 资格

<!-- MODULE REVIEW: CANDIDATE -->

- tool owner：C3 selected-component smoke；baseline `aaf1c8be5`。
- 不改产品 API/算法/路由/锁、closure/selector/perf 核心；不重开产品 owner。
- 反例：请求 `cpu;cuda` 时，source parent 正确，但 SDK child 实际收到
  `-DHYLE_COMPUTE_ROUTES=cpu` 和独立 `cuda` 参数。CPU SDK 被生成，CUDA feature guard 正确拒绝。
- 同处缺口：SDK 旧 guard 仅检查 cpu/cuda/onednn，未检查 Apple/Vulkan 与完整路线集合。
  Air CPU+Apple 两 SDK 实际 metadata 为 CPU，四项旧组合资格撤回，原数学日志和错误判断保留。

## 固定有限 FIX

1. `_hyle_selected_run` 用 CMake `PARSE_ARGV` 保留原参数边界，组合路线作为一个子进程参数。
2. 配置后的 producer 和安装 SDK 必须精确匹配请求的 canonical 路线集合；原 feature guard 覆盖全部四路线。
3. 在已有 compute-routes 工具测试中加入真实 CMake 子进程参数转发、路线一致/缺失/多余/无声明回归。

P0：只影响验证启动和资格判断，无运行期成本。本轮工具修改必须先由小 CMake fixture 证明，
再恢复 Windows CPU+CUDA shared/static、Air CPU+Apple shared/static 原目录；保留 source PASS，
不删除 source、不重编单路线、不跑 P1。新路线只新增实际缺失目标，已有 CPU objects 能复用即复用。
预算：一次具名工具审计、一个上述修正、两组原 SDK 阶段；新增失败单独记录，不改门、不重置预算。
适用检查为该工具测试、docs/closure/diff 和实际两组 SDK；未运行仍未资格。

Windows 实际反例 fixture 三个组合均失败，原日志保留；`PARSE_ARGV` 修正后完整11项工具测试 PASS，
包括真实子进程转发和缺/多/无声明 SDK 拒绝。只改本 driver 与已有工具测试，CPP/数学为零 diff。
selector 给 shared-CMake superset READY；非本工具的全库数学、headers、P1 无产品变化不适用，
不启动这些全库重编。两组合旧 SDK 资格仍未签，下一步在原目录恢复，其配置前置 guard 必须先通过。

工具 candidate：`ed2dd3c3869be140fa6e63c439c3f8479a3deb6c`。完整 Windows 来源 archive 的
SHA256 `98ef74990ec0e679a9293444c1f01800d8c3d3ac18e8a02bd8eb2e0c81ddd09a` 在 Air 核对一致；
Air 同两项转发/guard 工具测试实际 PASS，见
[工具收据](../../../../evidence/correctness/compute-unified-c3/mixed-sdk-route-failure/tool-repair-receipt.json)。
Windows 原 shared/static SDK 循环已恢复，source 不重跑，原 CPU objects 保留；只有错误 package
staging payload 被有边界删除，避免旧 ABI/未选资产遗留。Air 两 SDK 循环仍待串行恢复。

已逐份核对全部已归档的12个 SDK route metadata：十份单路线均与请求一致，只有两份 Air
组合与请求不符，见[元数据审计](../../../../evidence/correctness/compute-unified-c3/mixed-sdk-route-failure/archived-sdk-route-audit.json)。
这项只签 metadata，不重跑或升级数学资格。Air 原错误包 payload 已清理，source、build objects
和 machine SDK/cache 保留；修正后的原 shared/static 循环已经准备，待 Windows 重型批次结束后串行执行。

WSL 的完整验证源码也以同一 ed2dd3c3 archive/hash 更新，原目录保留；实际 CMake3.28.3 的
转发/完整路线 guard 两项 PASS。此项只签工具行为，不签 WSL CPU/CUDA/Vulkan 数学或 GPU。
后续 Unix launcher 已绑定新 source identity，不把新版工具误标旧 snapshot。

修正后的 Windows 原 shared SDK 已实际通过：完整 `cpu;cuda` 配置/安装 metadata 与四路线 guard，
搬迁后 Direct1.74s/Graph1.67s 真实数学 PASS，DLL 邻 EXE；
[当前收据](../../../../evidence/correctness/compute-unified-c3/windows-cpu-cuda-partial.json)保留工具、
产品及日志身份。C3 现36 PASS / 54 UNRUN；static 同一原循环继续，source/shared 不重跑。

Windows 原 static 也已实际 PASS：Graph1.84s/Direct1.85s，metadata 精确 `cpu;cuda`、DLL 邻 EXE，
SDK 循环 exit0；该组合六消费者全部通过，C3 当前38 PASS / 52 UNRUN。
下一串行 Air 原 SDK 循环已恢复，真实子进程为一个 `cpu;apple` 参数，配置 metadata 也精确一致；
shared/static 数学尚未签，旧错误包/判断与 source PASS 独立保留。没有重启任何 source root。
