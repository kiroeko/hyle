# Provider 批次治理 v1

<!-- PROJECT UNIT REVIEW: CLOSED -->

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

## AUDITED 有限 FIX

| ID | 有限动作 | 验收 |
|---|---|---|
| G1 | 定义 versioned Provider batch manifest：需求批次、Provider feature 超集、稳定 route/profile、适用系统、factor、delivery、test label 与 gate | schema/语义负例；禁止逐算子 profile、重复 ID、未知 profile 和没有理由的 skip/reuse |
| G2 | 新增薄编排 CLI 的 `validate`/`plan`，输出确定性配置、低并行 build、串行 test 计划 | 相同 manifest/candidate 得到相同 hash/计划；不执行产品构建；不解释 CMake 图或重写 selector |
| G3 | 新增 immutable Git `bundle`，要求 baseline 为 candidate 祖先并写 sidecar digest/commit/tree | 临时 Git 仓库验证远端持有 baseline 时 fetch 后得到精确 candidate；拒绝 dirty/WORKTREE 和非祖先 |
| G4 | 新增 `finalize`，只接受所选 gate 的实际 PASS、严格匹配的 REUSED receipt 和具名 SKIPPED，生成一份批次 terminal receipt | 缺失/FAIL/身份漂移/日志 digest 不符全部拒绝；不把 skipped 写 PASS |
| G5 | 登记 tooling test owner，更新脚本入口、change-impact 与 M2 批次合同；提供首份 B2--B6 feature/profile manifest | 工具定向测试、selector plan、docs、closure、JSON 与 diff 检查；不触发 native/P1 |

## 审计裁决

1. `validation-select.py` 已将 candidate tree、合同、toolchain、machine、观测 host、selector hash、
   command logs 与 P1 manifest 纳入 receipt key/复用校验。批次层只保存 selector plan/result 的路径、
   SHA256 和逐 gate verdict；不得自行宣告 `REUSED`。
2. build profile 的 artifact identity 固定为 factors、CMake cache、source/static/shared、sanitizer 与
   toolchain；system、machine、SDK/driver 和观测环境进入 execution identity。两类 identity 都由
   canonical JSON hash 命名，B2--B6 不按批次改 build root。
3. bundle 要求 baseline/candidate 均为 immutable commit 且 baseline 是祖先。创建时用临时具名 ref
   让 Git 正确登记 candidate head，并以 baseline 为 prerequisite；`git bundle verify`、sidecar digest、
   candidate tree 和 prerequisite 一并保存。远端必须先有 baseline，再 fetch bundle。
4. terminal 仅在全部 `SELECTED` gate 为 PASS、全部 `REUSED` 仍由 selector result 签署、全部
   `SKIPPED` 有适用性理由时生成。manifest 的 residual 保持 `UNPROVEN`，单独输出，不计 PASS。
5. 当前 oneDNN 的 `MATMUL;LAYER_NORMALIZATION;BINARY;ELTWISE;REORDER` 已覆盖 M2 所需 primitive
   编译超集；ggml CPU/CUDA/Metal/Vulkan、MLX 与 CUDA 成熟 Provider 均按路线整体编译。M2 manifest
   声明这些事实和锁定版本，不逐批改 vendor 选项；“编入”不等于“接通”或“覆盖”。

## 冻结实现形状

- `scripts/provider-batch.py` 只实现 `validate`、`plan`、`bundle`、`finalize` 四个子命令；不 import
  selector 私有实现，也不执行 CMake/CTest/P1。
- `plan` 输出 profile identity、串行路线顺序、每 profile 的 configure/build/test argv 模板，以及
  manifest gate 的 `SELECTED/REUSED/SKIPPED` 预期；实际 verdict 仍来自 selector result。
- `finalize` 读取原 selector JSON，校验文件 digest、candidate/tree、每个 gate ID/状态和 residual，
  生成单一 JSON 收据；不修改覆盖矩阵、registry 或文档。
- 首份 M2 manifest 固定 CPU→CUDA→Apple→Vulkan 重型顺序、并行度2、route/applicability、三交付
  profile 与 B2--B6 feature 超集。具体 B2 gates 在 B2 AUDITED 时追加，不重新定义 profile。
- 工具只用 Python 标准库和 Git CLI。测试在 disposable Git 仓库运行，不构建产品或访问网络。

## 实现与终局

实现候选为 `b92d84cfe1a1df4e8a11faeaed32abbcc4a91419`，tree
`522546b486d42aa55238eefec6e5dbe95df249a0`：

- G1：`provider-batch-schema-v1.json` 与 M2 manifest 固定5个批次、43行需求、5组 Provider feature
  超集、9个稳定交付 profile、并行度2和 CPU→CUDA→Apple→Vulkan 顺序；当前 gate/residual 为空，
  留给 B2 AUDITED 后按实际风险追加。
- G2：`validate` 与 `plan` 已绑定 manifest 语义 hash、immutable candidate/tree、artifact identity 和
  确定性 argv；test 固定串行，不执行产品命令。
- G3：真实仓库从 B1 baseline 生成21,468字节增量 bundle，SHA256
  `6ee27d251968e5e3835bf5d89f0cb4129ba941ec72440d83d8bedf825c5fa03f`；空接收仓先 fetch baseline，
  再 fetch bundle 的 `refs/hyle-batch/candidate`，`FETCH_HEAD` 精确等于实现候选。
- G4：`finalize` 只接受 digest、candidate/tree、selector 总 verdict 和逐 gate 状态全部匹配的结果；
  测试证明缺失/漂移 digest 被拒绝，具名 skip 与 `UNPROVEN` residual 不会变成 PASS。
- G5：工具已登记 `validation-policy.json`，入口及批次层合同已写入维护脚本、change-impact 和 M2
  合同；B2--B6 复用同一 feature/profile manifest，不再为每个算子重建治理单元。

## 验证与裁决

| 门 | 结果 |
|---|---|
| `provider_batch_test.py` | PASS，4组 disposable Git/manifest/selector 试验，约2.5秒 |
| `validation_selector_test.py` | PASS，59项，约97.2秒 |
| change-impact selector | PASS；仅 `docs`、`closure`、本工具测试与 selector 回归，130项 skipped/reused；没有 native/P1 |
| docs | PASS，503个 Markdown、2409个相对链接 |
| closure | PASS，57个 component、schema 4 |
| JSON / `py_compile` / `git diff --check` | PASS |
| immutable plan/bundle/fetch | PASS；9 profiles、精确 candidate、bundle prerequisite 与接收端 head 均匹配 |

P0：manifest 与 gate 数线性处理；工具没有启动 CMake/CTest/P1。B2 尚未审计的产品 gate 仍未运行，
也没有在本卡宣告覆盖；这正是下一张产品卡的输入，不是本治理任务残项。G1--G5 全部完成，本卡
裁决为 `CLOSED`，M2 可进入 B2 的 SCOPED→AUDITED 批量接线。
