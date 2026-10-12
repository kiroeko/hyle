# F5固定产品代码量清单

product25e，沿原scripts/loc-report.py/cloc2.08政策与函数，未改公共统计工具。
完整immutable报告与精确owner/文件/hash列表保存，口径分开：

| 口径 | 有效代码行 |
|---|---:|
| 全仓库、cloc识别所有语言（含依赖/工具/文档等） | 2,785,772 |
| HYLE自有native include/src，排除tests/vendor/文档/build工具 | 174,338 |
| 本轮20个owner完整native源码/公开头union | 57,297 |
| 新compute_batch，src+中央split public headers | 1,437 |
| 新graph_batch，src+中央split public headers | 400 |
| F5对F4的49个相关native文件集净代码增量 | 2,610 |

首个--module compute-batch/graph-batch采用conventional boundary，只计src，分别894/319，
不能当完整公开factor大小；按中央source owner+split headers显式补完整边界。
product commit25e的registry当时只有19个CANDIDATE，官方状态集报告56,409是那一历史范围；
当前20-owner union另用当前registry，逐文件与固定25e bytes核对，结果57,297。没有把
历史19改成20，也没有把全owner既有代码称作F5新增代码。

净增口径：固定F4 29d3与F5 25e的相关native文件before6,439→after9,049，差2,610；
排除tests/third-party/CMake，包含既有文件修改净量。它不是新能力的token数、引擎未来LOC
或用户节省行数；engine自实现/测试与基础库缺口要按实际消费者职责另估。test/工具等的
增量统计仍待最后快照，不用这里的产品计数代签。

原报告、精确文件/hash与工具版本见[receipt](receipt.json)/raw.zip。当前不是最终F5 closure。
