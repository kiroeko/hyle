# Hugging Face 固定源码的 in-tree 依赖声明

## AUDITED / 有限构建工具修复

- baseline：`abdd2a6c`；owner：build dependency helper；直接消费者为
  [D5-F42](../compute/llm-moe-d5-deepseek-quantization-roundtrip.md)。不新增 runtime component。
- 实际反例：DeepSeek V4 官方 `inference/kernel.py` 只在固定 Hugging Face 制品登记；
  `HyleDependencyVersions.cmake` 却要求每一个 in-tree 依赖都提供 GitHub URL、Git tree 和
  codeload archive。为满足该限制填写无关 GitHub 来源会使来源声明不真实。
- 唯一 FIX：允许 `in_tree` 的 canonical Hugging Face 模型仓库 URL + 40位 revision +
  一份具名上游 `source_file`（相对路径、规范化 SHA256、vendored 相对文件）。现有 GitHub 分支
  原样保留；Hugging Face 不进入 FetchContent/cache downloader，不下载模型权重或放开任意 URL。
  选中该依赖时核对原始文件摘要，继续核对完整 vendor tree 与全部许可证；关闭 CUDA 不读取该源码。
- P0：只增加冷配置字段与选中依赖的一次源文件 hash，运行产品无成本；不运行 P1。
- 预算：一次源码审计、一个有限合同测试批次、F42实际CUDA配置；仅定向修正。
- 预声明验证：固定 HF in-tree 正例；非法 repository/revision/source path/hash、HF非in-tree拒绝；
  原始源码改动拒绝；未选中时缺失 vendor 仍可投影；现有 GitHub锁投影不回归。docs/closure必跑。
  实际安装许可随 F42，完整三交付仍由 D5-A5 签署。
- 停止：上述源码来源门通过即停止本工具修复；不借机重写依赖缓存、selector或治理系统。

审计确认：in-tree 路径已有独立选中审计，下载器只服务 FetchContent；源码文件原文已取得，
无需假造 Git tree/archive。上述唯一 FIX 已冻结。HF source_file 保留原文，生成程序及其它工具
身份继续放入 vendor provenance，完整目录摘要负责绑定它们。
