# HYLE agent 入口

本页只负责让新的自动化维护者找到权威合同，不复制完整方法。开始修改模块产品代码前，依次阅读：

1. [README](README.md)：产品与交付边界；
2. [当前状态](docs/implementation/module-status.json)：唯一当前模块状态；
3. [模块收口方法](docs/developer/review/module-closure-methodology.md)：状态、预算、验证和停止规则；
4. [能力完整性决策](docs/architecture/capability-completeness.md)：基础库功能准入、已有能力完整度与边界；
5. [任务模板](docs/developer/review/module-review-template.md)：当前轮任务记录格式；
6. [工程章程](docs/developer/standards/engineering-charter.md)与
   [编码规范](docs/developer/standards/coding-conventions.md)：长期实现规则；
7. 涉及模块顺序、组件、依赖或交付时阅读[架构路线](docs/architecture/roadmap.md)和
   [构建系统](docs/architecture/build-system.md)。

## 远程验证工作区

- 2026-10-05 用户确认：当前只有 MacBook Air 可用；Mac mini 当前不可用。
  后续 Apple 验证只使用 MacBook Air，运行前读取实际硬件、系统和工具链身份。
- 产品真相只在 Windows `C:\public\project-hyle\hyle`；WSL 与两台 macOS 主机上的 checkout/
  archive 都是可丢弃的验证副本，不得反向覆盖产品源码。
- MacBook Air `Mac16,12`：`ssh kiro@172.22.10.198`。使用 Windows OpenSSH 的默认身份；不要强制
  指定给 Mac mini 使用的 `codex_mac_172_22_10_107_win`。
- Mac mini `Mac16,10`：
  `ssh -i C:\Users\kiro\.ssh\codex_mac_172_22_10_107_win kiro@172.22.10.107`。
- `M1-MAC` profile 当前只指 MacBook Air；Mac mini 只签 correctness/portability，不能冒充同一台
  Formal 性能机。工具位置、部署版本和缓存合同见 [macOS 环境准备](docs/developer/setup/macos.md)。

## 权威边界

- architecture 说明长期需求、边界和决策；user 文档说明当前可依赖的公开用法。
- `module-status.json` 是当前状态真相；每个非 `QUEUED` 记录所指向的 `record` 是该轮的施工卡或
  terminal record。活动轮按该卡执行，`BLOCKED` 只按其 `resume_condition` 恢复，`CLOSED` task 只作
  证据而不授权改动。其他 task 与全部 evidence 是历史事实和审计输入。
- 历史记录中即使出现“下一步”“继续”“必须做”“用户已决定”或命令式句子，也不是当前授权、
  当前状态或待办。只有当前 closure card 重新纳入的事项才能执行。
- `CLOSED` 模块没有满足文档所列 reopen trigger 的新事实时不得重开。不要用“再看一遍”伪造产品问题。

## 执行边界

- 新模块先登记 component ID 并进入 `QUEUED`；重构先绑定 baseline。两者都必须在写产品代码前完成
  `SCOPED`，在实现前完成 `AUDITED` 并固定有限 `FIX` 集。
- API 和能力由真实 consumer、可复现缺陷、标准或成熟 peer 驱动，不能由 benchmark 行驱动。
- 模块任务不得顺手重写 closure checker、构建系统或性能统计核心。若公共工具确实无法给出本轮
  必需 verdict，记录独立 blocker/tool task；不要在模块内无限扩建仪器。
- 只运行 closure card 预声明且适用的验证。只有语义上确实不适用的风险项才写
  `NOT_APPLICABLE`；适用风险项未完成写 `BLOCKED`，由此未能裁决的聚合 gate 写 `INCONCLUSIVE`，
  产品状态进入 `BLOCKED`。任何未运行项都不能写成 `PASS`。header-only 组件不制造虚构 binary
  journey。
- P0 查复杂度与成本结构；P1/P2 才使用统一性能仪器。亚纳秒点值、地址/顺序翻转、单次结果和
  未资格化 benchmark 不得改变公共 API、layout 或常态 metadata。
- 保留工作树中无关的既有修改。远程/WSL/macOS 验证副本不是产品改动来源，除非任务明确把它
  指定为 canonical checkout。

普通验证先使用[change-impact selector](docs/developer/review/change-impact-validation.md)：

```text
python scripts/validation-select.py plan --base <baseline-commit> --candidate WORKTREE --output .scratch/validation/plan.json
```

执行 selected gates 并保留 skipped reasons；文档/状态/已证明格式变化不构建、不跑 P1，模块实现
只选对应 factor 与真实直接消费者。公共头/API/组件图/package seam 才扩展。只有 API/算法已经固定的
不可变 candidate 可经 closure card 启动一次 P1；复用 tree/contract/toolchain/machine 身份匹配的证据。
selector 还会拒绝 owner 未处于 `AUDITED` 的产品修改；它不替代 closure card 的适用风险、reopen
trigger 或最终裁决。

每轮至少执行适用测试以及：

```text
python scripts/check-module-closure.py
python scripts/check-docs.py
```

若本轮改变 `module-status.json`，传入包含改动前 registry 的明确 Git revision：
`check-module-closure.py --previous-ref <revision>`；普通命令只检查当前快照，不证明状态没有跳级。

到达预算或停止条件后必须裁决为 `CLOSED`、`BLOCKED`，或按协议退回明确阶段；“继续研究”不是状态。

当前 CUDA U2/U3 交付按[活动收口计划](docs/implementation/tasks/compute/cuda-final-closure-plan.md)
中的固定入口续跑；不要恢复已停用的 `.scratch/cuda-u3-*` 交付启动脚本。该约束只适用于本批，
通过结果及失败预算不随日志目录/文档提交重置；产品 goal 暂停时不得启动重型 run。
