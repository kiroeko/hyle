# 原始证据

`evidence/` 保存机器产生、供追溯的原始输出，例如 CSV、JSON、日志、二进制尺寸和哈希。它不是
人读文档目录，也不直接声明产品能力、性能或收口状态。

## 使用规则

1. 证据必须能定位到模块、候选提交、平台和测试配置；文件名或同目录 manifest 应承载这些身份。
2. [`docs/implementation/`](../docs/implementation/README.md) 中的调查或任务记录链接到确切证据，解释
   测量合同、环境、结论和限制。不要链接一个含义会变化的 `latest` 文件。
3. 原始结果只能支持生成它的候选。实现、依赖、编译器、开关或仪器变化后，旧结果自动降级为历史材料。
4. 性能结论还必须满足
   [`benchmark-evidence-standard.md`](../docs/developer/performance/benchmark-evidence-standard.md)；孤立 CSV
   或一次日志不能证明 SOTA。
5. 不在这里手写面向用户的摘要。当前行为写入 [`docs/user/`](../docs/user/README.md)，设计依据写入
   [`docs/architecture/`](../docs/architecture/README.md)。

现有 `core/` 与 `performance/` 内容是迁移前留下的历史快照；除非新的 implementation 记录明确绑定
它们和当前候选，否则不得当作当前验收证据。
