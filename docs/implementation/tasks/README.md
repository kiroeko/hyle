# 任务记录

这里保存当前施工卡和当前 terminal record。不要按目录中“最新文件”猜状态：模块唯一入口是
[`module-status.json`](../module-status.json) 及其生成的[状态页](../status.md)，其中每个非 `QUEUED`
组件都直接链接当前 record。

当前治理 schema 的迁移事实：

- [治理执行性与文档压缩 v4](project/governance-documentation-v4.md)。

当前跨产品能力审计事实：

- [HYLE 能力族深度复审](project/hyle-capability-family-deep-audit-2026-09.md)：当前 27 个已实现 factor 的
  能力族、已交付头反向追溯、四纵切与缺口/路由结论；
- [HYLE 范围重定标与全组件能力扫描](project/hyle-scope-capability-audit-2026-09.md)：此前 factor 粒度的
  边界扫描；当前结论以能力族复审为准；
- [Public C++ 消费者纵切组合审计](project/public-api-consumer-portfolio-audit-2026-09.md)：此前 F01–F08
  纵切事实；已完成项和当前边界以新扫描为准。

当前公共工具修复事实：

- [Structured pair placement validator 修复](project/performance-structured-placement-validator-2026-09.md)。

任务结束后只保留当前状态链仍引用、或被长期 architecture/developer/user 文档引用的记录。被替代的
完整 Jira 式历史由 Git 保存，不用导航页和兼容跳转页维持第二套当前事实：

```text
git log --all --name-status -- docs/implementation/tasks/<area>
git show <revision>:docs/implementation/tasks/<area>/<task>.md
```

新记录从[模块任务模板](../../developer/review/module-review-template.md)开始。横向 build/tool/docs 工作
使用 `project/` 任务；它们不伪装成拥有稳定 candidate 的产品 component。
