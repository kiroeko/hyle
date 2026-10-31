# 实验与历史证据

本目录只供按问题查证，不是当前支持矩阵、开发任务清单或日常阅读入口。先看[用户指南](../user/README.md)、[当前架构](../architecture/README.md)、[推理进度](../developer/inference-progress.md)或[性能资料](../developer/performance/decode.md)，需要复核具体结论时再进入对应实验。

## 查证什么

实验目录中的源码、日志、CSV／JSON及README绑定当时的版本、输入、硬件、工具链和完成边界；通过、失败与未运行分别解释。旧记录里的“当前”“下一步”和已撤回方案不成为现行指令。

[历史正文](history/)保留从现行指南移出的调查细节，不继续追加开发流水。新增结果优先引用现成测试／基准日志，保留复现所需命令、版本、结果与必要边界；不要求每轮新建JSON、ZIP、manifest或全量哈希。

## 旧路径与归档

本目录曾从仓库根`evidence/`移到`docs/evidence/`；冻结日志／JSON里的旧路径与身份信息保留原样，查找以`evidence/`开头的仓库路径时使用当前`docs/evidence/`。冻结源码只对应当时版本，独立复现脚本已按当前仓库定位。

已删除过程文档的引用固定到清理前Git版本，可用`git show <commit>:<path>`读取；追溯入口见[实施材料](../implementation/README.md)。

部分大型原始归档只在本地，路径和身份见[本地归档清单](local-archives.json)，不随Git克隆提供。相邻`*.records.zip`仅保留小型源码、脚本、命令、结构化结果与日志；包内`records-manifest.json`说明收录／省略内容。向量、MAP、数据库、profile、二进制和超过4MiB文本通常不在轻量包中，需要时仍查清单所列原包。
