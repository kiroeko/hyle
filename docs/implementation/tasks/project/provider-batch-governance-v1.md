# Provider 批次治理 v1

<!-- PROJECT UNIT REVIEW: SCOPED -->

## Closure card

- baseline：`4d2805feb9982a9db06320f4900eed5f55f2fccc`；Windows canonical 工作树干净。
- trigger：M2-B1 的 Provider 接线已完成，但人工为每个能力重复配置、跨机同步、解释证据复用和
  更新多份状态，导致 oneDNN 540、Air 646、CUDA 255 项冷编译及重复文档成为主要耗时。用户于
  2026-09-23 明确授权在 B2 前优化项目治理。
- owner：独立 build/tooling 任务；不借 Compute owner 修改产品数学、API、CMake 组件图、closure
  状态机或性能统计核心。
- 主旅程：一份 Provider 批次 manifest 固定 B2--B6 能力超集、路线/环境/交付 profile、factor、
  test label、适用性和复用证据；工具生成确定性执行计划，并由逐 gate 结果生成终局收据。
- 正交旅程：从 Windows canonical 的 immutable candidate 生成相对 baseline 的 Git 增量 bundle，
  远端已有 baseline 时只更新实际 commit，不用 tar 覆盖整棵源码或重置无关文件时间戳。
- 非目标：不修改 `validation-select.py` 的选择、状态授权、receipt key、lease 或 P1；不修改
  `check-module-closure.py`、`hyle-perf.py`、CMake target 构造、依赖锁、测试语义或验收门限。
- 性能：P0；工具必须线性处理 manifest/gate 数，默认重型并行度2、路线重型验证串行；不运行 P1。
- stop：有限 FIX 与工具测试、docs/closure/diff 全部通过后关闭；若现有 selector 已能完整表达某项，
  只生成/消费其输入输出，不复制逻辑。B2 产品代码只在本卡结束且 B2 owner 另行 AUDITED 后开始。

## 待审有限 FIX

| ID | 有限动作 | 验收 |
|---|---|---|
| G1 | 定义 versioned Provider batch manifest：需求批次、Provider feature 超集、稳定 route/profile、适用系统、factor、delivery、test label 与 gate | schema/语义负例；禁止逐算子 profile、重复 ID、未知 profile 和没有理由的 skip/reuse |
| G2 | 新增薄编排 CLI 的 `validate`/`plan`，输出确定性配置、低并行 build、串行 test 计划 | 相同 manifest/candidate 得到相同 hash/计划；不执行产品构建；不解释 CMake 图或重写 selector |
| G3 | 新增 immutable Git `bundle`，要求 baseline 为 candidate 祖先并写 sidecar digest/commit/tree | 临时 Git 仓库验证远端持有 baseline 时 fetch 后得到精确 candidate；拒绝 dirty/WORKTREE 和非祖先 |
| G4 | 新增 `finalize`，只接受所选 gate 的实际 PASS、严格匹配的 REUSED receipt 和具名 SKIPPED，生成一份批次 terminal receipt | 缺失/FAIL/身份漂移/日志 digest 不符全部拒绝；不把 skipped 写 PASS |
| G5 | 登记 tooling test owner，更新脚本入口、change-impact 与 M2 批次合同；提供首份 B2--B6 feature/profile manifest | 工具定向测试、selector plan、docs、closure、JSON 与 diff 检查；不触发 native/P1 |

## 审计问题

1. selector 当前 receipt key 与 reuse 校验是否已经足够，批次工具只需保存哪些引用？
2. 哪些 profile 字段实际改变构建产物，哪些只是执行位置，如何避免每批 reconfigure？
3. Git bundle 如何在不包含整仓历史的情况下证明 baseline→candidate，并让远端 fail closed？
4. terminal receipt 如何区分 PASS、REUSED、SKIPPED 与 UNPROVEN residual？
5. M2 的 Provider feature 超集如何一次声明，而不把上游不存在的能力写成已构建或已覆盖？

当前只完成 SCOPED；尚未修改脚本、策略或 M2 manifest，也没有新的验证 PASS。
