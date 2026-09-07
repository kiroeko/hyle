# 按变更影响选择验证

`scripts/validation-select.py` 是普通本地迭代和 PR/main push 的验证入口。给定两份 Git 快照，
它输出 `selected_gates`、`skipped_gates` 和逐项理由，再按需执行；不改变产品状态，也不授权
重开 `CLOSED/FROZEN` 模块。适用风险及最终裁决仍以当前 closure card 和
[模块收口方法](module-closure-methodology.md)为准。

## 使用

先在 task 冻结 baseline、有限 FIX、适用门及停止条件。实现未提交时可以查看工作树计划，
或执行其中的文档/工具门：

```text
python scripts/validation-select.py plan --base <baseline-commit> --candidate WORKTREE --output .scratch/validation/plan.json
python scripts/validation-select.py run --base <baseline-commit> --candidate WORKTREE --output .scratch/validation/result.json
```

`WORKTREE` 包含 staged/unstaged/非忽略 untracked 文件，通过私有临时 index 取 tree，不改变
用户暂存状态；它不能执行产品构建、P1 或复用 immutable receipts。产品验证先提交候选，保持
该 checkout 干净，绑定执行环境：

```text
python scripts/validation-select.py run --base <baseline-commit> --candidate <candidate-commit> --context .scratch/validation/context.json --output .scratch/validation/result.json
```

context 含 `toolchain`、`machine_profile` 两个带 `id` 的对象；记录精确 compiler/version/hash、
stdlib/SDK、flags/configuration、实际机器 profile。整个对象进入证据键，`id` 不能替代精确事实。
工具另外采集 OS、架构、主机名与 Python identity。Linux CI 自动记录 compiler/stdlib 文件 hash、
CMake 版本和配置。产品命令使用已经准备好的工具链环境：Unix 可用 `hyle-unix.sh python3 ...`，
Windows 使用已导入 VS 环境的 PowerShell。默认 Ninja/Release/WERROR/shipping layout；其他
sanitizer、ISA、backend 风险由 closure card 另行声明，不被普通 selector PASS 替代。

## 影响规则

| 变更 | 自动选择 | 不自动选择 |
|---|---|---|
| 文档、状态、证据、格式配置、已证明格式变化 | docs + closure；registry 改变时保存 baseline registry 并传 `--previous` | C++ build、交付、P1 |
| 模块实现 | owning factor + 中央真实直接边的一跳消费者 | 传递消费者测试、无关 factor、package、P1 |
| factor 测试 | owning factor correctness | 消费者、P1 |
| 公共头/API、factor CMake seam | owning factor、反向消费者闭包、已有逐头测试、受影响产品三交付 | 自动 P1 |
| 组件图、公共 build/package seam、第三方产品输入 | 组件合同、已实现 factor 与交付闭包 | 未实现 `QUEUED` target、自动 P1 |
| 已映射维护工具/CI 调度 | `validation-policy.json` 中的工具测试 | 产品 build/P1 |
| 未登记路径/ownership、metadata 边界漂移 | 明确 `BLOCKED` 与原因 | 静默跳过、猜测全量通过 |

编译所选测试可能构建依赖 artifact，但依赖不因此成为额外测试根。只构建
`hyle_<factor>_tests`，CTest 精确匹配 factor label；独立选择 `headers`，不调用全量 `hyle_tests`。
没有独立 header suite 的因子明确显示限制；consumer/delivery 编译不冒充逐头 self-containment。
若 closure card 要求这一风险，仍需单独给出实际门，未完成保持 `BLOCKED`。

中央图始终只有 `cmake/HyleComponents.cmake` 一份。adapter 从 Git blob 取具名
`_hyle_component_public_headers` 边界前的完整 metadata 前缀，由 CMake 自身执行声明、闭包与
图检查；不运行 `project()`、compiler probe 或配置期全库源码审计，不用 Python 解释 CMake。
边界缺失/重复、声明移到边界后均拒绝。旧、新两份图参与删改影响计算；`SPLIT_PUBLIC_HEADERS`
决定精确 owner，façade 只决定交付产品。

自动格式证明覆盖 JSON 结构一致，以及 native source 保持行结构的缩进/尾空白/CRLF。
字符串内容、raw string、宏续行、行号敏感代码和 Python 缩进不能用 `git diff -w` 忽略。
无法证明等价的变化保留语义风险分类；复杂 formatter 迁移需要独立冻结其等价性证明。

## P1 冻结卡

P1 从不自动选择。只有 `--p1-card` 显式绑定不可变候选、diff 有相关产品/交付/benchmark 输入，
且卡中的 factor 已选 correctness，才可启动一次 bundle。docs/state/format/tool/test-only diff
即使传卡也不采样。卡放在 `.scratch` 或外部路径，避免自身 commit hash 循环：

```json
{
  "candidate_commit": "<full 40-hex commit>",
  "candidate_tree": "<full 40-hex tree>",
  "api_state": "FROZEN",
  "algorithm_state": "FROZEN",
  "factors": ["<factor>"],
  "contract": "bench/contracts/<frozen-contract>.json",
  "contract_hash": "<SHA256 of the exact candidate Git blob>",
  "peers": {"<peer-id>": "<locked clean peer checkout>"}
}
```

P1 context 必须包含 `toolchain.compiler_sha256`、`toolchain.stdlib`、`toolchain.flags`；
`machine_profile.id/profile_hash` 必须匹配候选 `scripts/perf-machines.json` 对应 profile。
profile hash 是 UTF-8、sorted keys、无多余空白的 canonical JSON SHA256，与工具 `digest()` 一致。
命令固定由选择器生成 `hyle-perf.py qualify-platform`；所有已选前置门通过后才采样。原仪器仍
负责机器资格、peer、合同、公平性与统计，结果必须同时为 qualification/performance `PASS`。

开始 P1 前在 Git common directory 的 `hyle-validation-attempts` 用独占创建写 claim，所有
worktree 共用。失败、中断、并发、改变 toolchain/machine/contract 或 `--cache` 日志目录均不能
自动重跑。额外的 `product_input_hash` 绑定产品输入树，追加文档/状态/工具提交也不重置额度。
claim 不自动删除；新的 material 产品候选按 closure 协议重新 scope，不能换 clone 或清账本
跑到绿。这是工程调度合同，不是对有磁盘写权限者的防篡改系统；多 clone/远程副本由 task owner
统一持有 canonical 账本，不能分别领取 P1 配额。

## 证据和结果

每个 gate key 含 candidate tree、contract hash、toolchain、machine profile、观测 host、selector
hash。合同 hash 含命令、策略、冻结卡；closure 还绑定 previous registry。只有同 key 完整 PASS
receipt 与全部命令 log hash 匹配才 `REUSED`。P1 还需原始 manifest hash 与现有 `hyle-perf.py verify`
的引用文件/资格校验通过；这是只读验证，不采样。失败、RUNNING、缺失或损坏证据不得复用。
仍有 native gate 待执行时重新确保 configure，证据不冒充本地 build tree。

日志默认保存在 Git common directory 的 `hyle-validation`，`--cache` 只改变日志位置。
scratch 按候选/context 隔离，同一候选/context 的执行 lease 防止并发重复调度；中断遗留 lease
需先确认相关进程已停止，再由 owner 处理具名锁。默认只复用相同 Git tree，文档更新本身只选
轻量门，不为旧产品证据补跑 P1。每个 gate 执行前后还检查工作树与计划一致。

`SELECTED/REUSED/SKIPPED` 是调度结果，实际运行后才有 `PASS/FAIL`。skip 不等于风险项
`NOT_APPLICABLE`；适用但未执行也不能写 PASS。未知映射/前置失败返回退出码 2 和 `BLOCKED`。
机器 JSON 保留所有 selected/skipped gate 与具体理由，stderr 只给进度。

PR/main CI 先生成计划，再按需准备 compiler/dependency cache，最后上传计划、context、结果和
logs。明确 `workflow_dispatch` 仍可选择全量三平台/交付/cross/feature 门，手工 full 也不含自动 P1。
