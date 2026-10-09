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
