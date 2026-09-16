# 按变更影响选择验证

`scripts/validation-select.py` 是普通本地迭代和 PR/main push 的验证入口。给定两份 Git 快照，
它输出 `selected_gates`、`skipped_gates` 和逐项理由，再按需执行；不改变产品状态，也不授权
重开 `CLOSED` 模块。适用风险及最终裁决仍以当前 closure card 和
[模块收口方法](module-closure-methodology.md)为准。

产品实现、公开头、factor 产品 seam、中央归属和已登记 vendored 产品输入还受状态授权：变更发生前
和产品 commit 完成后，owning factor 都必须已经是 `AUDITED`。不能把 `SCOPED -> AUDITED` 与实现塞进
同一 commit，也不能修改已冻结 `CANDIDATE/CLOSED`。selector 按 commit 检查一次 push 中的完整合法
状态序列，所以不要求每个阶段单独 push；测试、文档和证据维护不要求伪造 reopen。

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

context 含 `toolchain`、`machine_profile` 两个带 `id` 的对象。普通 native 和 P1 都要求：

- `toolchain.compiler_path/compiler_id/compiler_version`：实际使用的编译器绝对路径、实现和版本；
- `toolchain.stdlib.id`：足以区分实现与版本的标准库身份；
- `toolchain.flags`：恰好两个 string 键 `CMAKE_CXX_FLAGS` 与 `CMAKE_CXX_FLAGS_RELEASE`，后者非空。
  selector 将 compiler/flags 显式用于 configure，并通过受控 CMake toolchain file 传给嵌套 smoke；
- 实际 machine profile。P1 还要求下面所述中央 profile hash；普通 native 另外绑定观测主机和影响
  native 编译的环境变量，包括 INCLUDE/LIB、CXXFLAGS、SDKROOT 等。

完整 context 进入工程 receipt key；缺字段、编译器路径不存在或执行前描述漂移会拒绝 native。
selector 不再散列 compiler/backend/stdlib binary 或整棵标准库头目录：这无法防御有写盘权限的恶意
维护者，却使换机和工具链升级付出巨大无关成本。正式性能 runner 仍按其 manifest 合同记录实际
artifact/toolchain 身份，发布级真实性由 Formal 流程承担。Linux CI 直接记录编译器与标准库版本。

无 context 的 `plan` 仍只做影响选择，方便 CI 先决定是否准备编译器；此路线不复用 native 证据，
`run` 必须重新绑定 context。docs/state/format/tool-only 不要求 native 字段，也不扫描标准库树或
启动编译器。产品命令使用已经准备好的工具链环境：Unix 可用 `hyle-unix.sh python3 ...`，
Windows 使用已导入 VS 环境的 PowerShell。默认 Ninja/Release/WERROR/shipping layout；其他
sanitizer、ISA、backend 风险由 closure card 另行声明，不被普通 selector PASS 替代。

## 影响规则

| 变更 | 自动选择 | 不自动选择 |
|---|---|---|
| 文档、状态、证据、格式配置、已证明格式变化 | docs + closure；registry 改变时传包含旧 registry 的 `--previous-ref` | C++ build、交付、P1 |
| 模块实现 | owning factor + 中央真实直接边的一跳消费者 | 传递消费者测试、无关 factor、package、P1 |
| factor 测试，包括 `tests/<factor>/CMakeLists.txt` 与目录内 `.cmake` | owning factor correctness | 消费者、独立 header/delivery gate、P1 |
| factor benchmark 源码与 `bench/<factor>/` 内 CMake | docs + closure；构建/资格按当前卡预声明 | 产品 header/三交付、自动 P1；全局 `bench/CMakeLists.txt` 仍按公共 seam |
| 公共头/API、factor 产品 CMake seam | owning factor、反向消费者闭包、已有逐头测试、受影响产品三交付 | 自动 P1 |
| 组件图、公共 build/package seam、第三方产品输入 | 组件合同、已实现 factor 与交付闭包 | 未实现 `QUEUED` target、自动 P1 |
| 已映射维护工具/CI 调度 | `validation-policy.json` 中的工具测试 | 产品 build/P1 |
| 未登记路径/ownership、metadata 边界漂移 | 明确 `BLOCKED` 与原因 | 静默跳过、猜测全量通过 |

`product_authorization` 会列出本次产品 owner、最终状态和逐 commit 违规。合并提交只对与所有 parent
都不同的冲突解决结果增加一次检查；普通分支产品 commit 仍按其真实 parent 的 registry 状态判断。

编译所选测试可能构建依赖 artifact，但依赖不因此成为额外测试根。只构建
`hyle_<factor>_tests`，CTest 精确匹配 factor label；独立选择 `headers`，不调用全量 `hyle_tests`。
没有独立 header suite 的因子明确显示限制；consumer/delivery 编译不冒充逐头 self-containment。
若 closure card 要求这一风险，仍需单独给出实际门，未完成保持 `BLOCKED`。

原生路线通过 policy 的 `factor_supported_systems` 声明 OS 适用性。当前 OS 不支持的路线
correctness 明确 `SKIPPED`，配置 feature 为 OFF 并保留独立的 OFF 公开头检查；这些 skip 不能签路线
完成，也不能作为 P1 前置 PASS。支持的 OS 一律配置 ON，SDK 或设备缺失仍按实际门失败，不能
探测不到便自动跳过。Apple 真实执行需 Darwin，CUDA 真实执行需 Windows/Linux；跨平台结果由当前卡汇总。

中央图始终只有 `cmake/HyleComponents.cmake` 一份。adapter 从 Git blob 取具名
`_hyle_component_public_headers` 边界前的完整 metadata 前缀，由 CMake 自身执行声明、闭包与
图检查；不运行 `project()`、compiler probe 或配置期全库源码审计，不用 Python 解释 CMake。
边界缺失/重复、声明移到边界后均拒绝。旧、新两份图参与删改影响计算；`SPLIT_PUBLIC_HEADERS`
决定精确 owner，façade 只决定交付产品。

自动轻量格式识别只覆盖 JSON 结构一致，以及 native source 保持行结构的缩进、尾空白和 CRLF。
字符串内容、raw string、宏续行、行号敏感代码、Python 缩进和无法机械证明的空白重排都保守地按
普通语义变更走 owner/consumer 验证，而不是 `BLOCKED` 或要求维护一份 blob-pair 证明账本。这样最多
多跑受影响模块的正常门，不会为了“证明没改代码”引入另一套当前事实。大规模 formatter 迁移若需要
更窄预算，应单独冻结一次性任务并由普通 diff/reviewer 说明，不把例外永久写进公共 selector。

## P1 冻结卡

P1 从不自动选择。只有 `--p1-card` 显式绑定不可变候选、diff 有相关产品/交付/benchmark 输入，
且卡中的 factor 已选 correctness，才可启动一次 bundle。docs/state/format/tool/test-only diff
即使传卡也不采样。卡放在 `.scratch` 或外部路径，避免自身 commit hash 循环：

```json
{
  "candidate_commit": "<full 40-hex commit>",
  "candidate_tree": "<full 40-hex tree>",
  "candidate_fixed": true,
  "factors": ["<factor>"],
  "contract": "bench/contracts/<frozen-contract>.json",
  "contract_hash": "<SHA256 of the exact candidate Git blob>",
  "peers": {"<peer-id>": "<locked clean peer checkout>"}
}
```

P1 context 必须满足上述 native compiler/stdlib/flags 合同；
`machine_profile.id/profile_hash` 必须匹配候选 `scripts/perf-machines.json` 对应 profile。
profile hash 是 UTF-8、sorted keys、无多余空白的 canonical JSON SHA256，与工具 `digest()` 一致。
命令固定由选择器生成 `hyle-perf.py qualify-platform`；所有已选前置门通过后才采样。原仪器仍
负责机器资格、peer、合同、公平性与统计，结果必须同时为 qualification/performance `PASS`。

输出 `p1_scopes` 逐 factor 列出中央图的完整产品依赖闭包、计入的 input paths、
`product_input_hash` 和 `scope_hash`。只计 owner/dependency 的源码、split headers、各自 vendored
输入及共享 build/package seam；不会把无关 `src/container` 混入 Queue scope。
外部文件 owner 由 `validation-policy.json` 的有限映射说明，不复制中央 factor 依赖图；未登记
vendored 输入阻塞 P1，nanobench 等仪器不是产品输入。

P1 的有效、失格、中断和纠正历史由当前 closure task 记录，selector 不在 Git common dir 建永久
one-shot claim。相同 candidate/context 的 execution lease 仍防止并发重复运行，完整 PASS receipt 仍按
精确 identity 复用。有效 material FAIL 必须依方法进入一次机制诊断并改产品、重定范围或 `BLOCKED`；
不得换 clone、cache、顺序或追加样本跑到绿。fixture/tool 缺陷和环境失格不产生产品 verdict，修复后
只重跑受影响 gate；这两类纠正不消耗产品修改名额，但仍受 task 的 wall-clock/性能预算约束。

## 证据和结果

每个 gate key 含 candidate tree、contract hash、toolchain 描述、machine profile、观测 host、selector
hash。合同 hash 含命令、策略、冻结卡；closure 还绑定 previous registry。只有同 key 完整 PASS
receipt 与全部命令 log hash 匹配才 `REUSED`。P1 还需原始 manifest hash 与现有 `hyle-perf.py verify`
的引用文件/资格校验通过；这是只读验证，不采样。失败、RUNNING、缺失或损坏证据不得复用。
仍有 native gate 待执行时重新确保 configure，证据不冒充本地 build tree。

日志默认保存在 Git common directory 的 `hyle-validation`，`--cache` 只改变日志位置。
scratch 按候选及完整观测 context 隔离，避免环境变化仍使用旧 CMake cache。相同候选/context 的执行
lease 防止并发重复调度；中断遗留 lease
需先确认相关进程已停止，再由 owner 处理具名锁。默认只复用相同 Git tree，文档更新本身只选
轻量门，不为旧产品证据补跑 P1。每个 gate 执行前后还检查工作树与计划一致。

`SELECTED/REUSED/SKIPPED` 是调度结果，实际运行后才有 `PASS/FAIL`。skip 不等于风险项
`NOT_APPLICABLE`；适用但未执行也不能写 PASS。未知映射/前置失败返回退出码 2 和 `BLOCKED`。
机器 JSON 保留所有 selected/skipped gate 与具体理由，stderr 只给进度。

PR/main CI 先生成计划，再按需准备 compiler/dependency cache，最后上传计划、context、结果和
logs。明确 `workflow_dispatch` 仍可选择全量三平台/交付/cross/feature 门，手工 full 也不含自动 P1。
