# CUDA 交付入口：Linux XML 输出路径

<!-- MODULE REVIEW: AUDITED -->

- baseline：`eec92792`；仅本批工具，不修改产品/registry，Windows当前批次继续。
- trigger：Linux配方把`{logs}`展开为`C:/...`后传给WSL gtest，不能写入预期canonical XML。
  原[工具任务](cuda-delivery-execution-control.md)的实际配方缺陷reopen条件成立。
- scope/FIX：只将Linux子进程参数中的日志目录转换为对应`/mnt/<drive>/...`；本地proof路径、
  Windows配方/身份/预算、产品/数学/性能门保持。不建立通用远程路径或构建框架。
- 预声明：现有轻量测试、路径映射反例、一次真实WSL Python写XML并由入口核查、docs/closure。
- 预算/停止：一次实现与上述轻量验收；通过即关闭，不重跑已通过Windows/GPU数学。

AUDITED：gtest接收Linux进程本地路径；canonical日志目录为绝对Windows盘符路径，已有
WSL挂载为`/mnt/c`。固定只改Linux子进程的`{logs}`展开，本地记录仍用原路径。
Windows job不增加字段，因此此前输入身份和通过记录不失效。
