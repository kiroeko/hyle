# Windows wrapper 的 source anchor

<!-- BUILD TASK: QUEUED -->

## Finding

从 Platform v8 的 exact archive 做消费者回放时，在 canonical checkout 作为当前目录、用绝对路径调用
另一个 checkout 的 `scripts/hyle-windows.ps1 -Action Configure -BinaryDir ...`，脚本虽然正确计算了
`$sourceDir`，但实际执行的是：

```text
cmake --preset <preset> -B <binary-dir>
```

CMake 因而从调用者当前目录读取 preset 并把 canonical checkout 作为 source。`CMakeCache.txt` 的
`CMAKE_HOME_DIRECTORY` 和 Boost.Context assembly 路径都证明第一次所谓 archive replay 实际编译了
`C:\public\project-hyle\hyle`。切换当前目录到 archive root 后重跑才得到真正 exact 的 10/10 PASS。

这不影响文档所示“在 checkout root 运行 wrapper”的普通命令，却会让绝对路径工具调用、并列 checkout、
候选归档验证和自动化维护者静默选择错误源树；因此是 Build/package 的独立维护缺陷，不能在 Platform
产品任务里顺手修改。

## 冻结修复边界

- wrapper 的 preset-backed Configure/Build/Test/Workflow 必须以脚本自身的 `$sourceDir` 为 preset/source
  anchor，而不是调用者当前目录；
- 显式 `-BinaryDir`、`-Component`、`-Factor`、VS 环境选择、额外参数和退出码语义保持不变；
- 从 checkout root 的现有命令不得回归；从另一 checkout/CWD 绝对调用必须在 cache 与 build graph 中只
  出现目标脚本所属 source；
- Unix wrapper、组件图、产品源码和 Platform candidate 不属于本修复。

## 验收

1. 建立两个不同 source root/candidate marker 的 fixture，从 A 的 CWD 绝对调用 B wrapper，B 的
   `CMAKE_HOME_DIRECTORY`、preset、源路径和生成 target 全部正确；
2. root-relative Windows Ninja、显式 isolated `-BinaryDir`、Build/Test 和至少一个 component smoke 通过；
3. 文档说明 wrapper 可从任意 CWD 调用，并运行 `check-docs.py` 与 Build/package 适用合同。

该任务在 Platform v8 terminal 裁决后单独进入活动轮；修复若改变 source/static/shared 交付内容，再按
依赖关系定向回放受影响的候选，不把 Build 维护伪装成 Platform 产品变更。
