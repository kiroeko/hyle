# D5-A5：CPU 真实片段三交付

**CPU A5 PASS：源码、静态 SDK、动态 SDK 各 8/8，共 24/24，零跳过。**
消费者候选 `b20fb87c935103293cf3602f3f4fee53c660b95f`；产品输入与
`9273bc71017a1febb69db240b6ea8970b88dd77c` 相同，产品 FIX 为零。
见 [汇总](summary.json)、[文件校验](files.json)和[原始文件归档说明](archive.json)。

| 交付 | 实测结果 | 收据 |
|---|---|---|
| 源码父工程 | Direct 4 + Graph 4，通过 | [source](raw/source/receipt.json) |
| 搬迁静态 SDK | Direct 4 + Graph 4，通过 | [static](raw/static-producer-complete/receipt.json) |
| 搬迁动态 SDK | Direct 4 + Graph 4，通过 | [shared](raw/shared/receipt.json) |

每入口执行 Qwen3.5-2B 的真实 BF16 embedding/norm/head 与非法索引失败、完整 FFN、
causal-conv/GDN 状态回送，以及 DeepSeek Flash 两种 GGUF 画像的完整 hash-routing/MoE。
沿用原模型内宽、逐角色编码、oracle、容差和变化输入；普通入口在准备时绑定成熟实现。
R01 是交付代表消费者，不另签完整词表或整模型推理。CPU 官方 FP8/FP4 的八个拒绝格仍保留。

Windows MSVC 19.50.35728、Release/WERROR，CPU/oneDNN ON，其它三路线、插桩、bench OFF。
源码 archive 逐文件核对 Git blob；每批记录测试源码闭包 SHA256。
源码/static 共用同一 producer，shared 复用原构建树，低并行 2、测试串行。
installed 消费者只链接公开产品 target；compile/link 日志检查未引用 canonical 产品或 producer 路径。
运行时隐藏 producer 的 bin，PATH 仅 Windows/System32 与 Windows。
动态 SDK 的 16 个 HYLE DLL 由公开 staging 接口邻置并逐个核对 SDK 来源 SHA256；
静态消费者不需要 HYLE DLL，见各批 `dll-origins.json`。未触发提权或缺 DLL 弹窗。

## 一次有因纠正

首次 static install 失败：源码阶段只构建了两个消费者的依赖，没有生成安装清单中未被它们使用的
`hyle_time` 等基础库。原始 [失败收据](raw/static/receipt.json)保留。
在同一 producer 补齐已配置的默认目标，用时 74.531 秒；不重新配置、不修改产品、消费者或第三方，
见 [补齐命令](raw/static-producer-completion/receipt.json)。随后静态 SDK 安装、搬迁、独立构建与
全部测试通过。`run.py` 将已实测的默认目标命令纳入源码构建阶段，避免后续漏建。
此前通过的源码与动态库测试没有重跑。

## 范围、检查与清理

本批只签 CPU A5，不签 A6 性能、其它路线或 D6，不增加画像通过格。
当前完整 ledger 仍为 152 PASS / 8 BACKEND_REJECTED / 16 VALIDATION_NOT_RUN。
docs、closure 及 selector 的跳过理由随 [检查记录](checks) 保存；本次无产品变更，不重跑 P1。
原日志可无损 gzip；JSON/XML/TXT 归档统一 LF，原字节长度和 SHA256 保存在 archive.json。

两个独立 consumer 构建目录在归档后清理，释放 153,897,480 字节（约 147 MiB）；
有效 SDK 与唯一源码/static producer 留给 A6，见 [占用](storage.json)和[清理记录](cleanup.json)。
统一脚本统计 Compute+Graph 实现 **60,373 行 / 241 文件**，不含测试和第三方，见 [LOC](loc.json)。
