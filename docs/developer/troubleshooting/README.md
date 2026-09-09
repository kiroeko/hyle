# 故障排查

当前产品 blocker 只看 [`module-status.json`](../../implementation/module-status.json) 中的 `BLOCKED`
记录；没有 `BLOCKED` 就不维护第二份“已知问题”清单。构建、依赖和平台问题先查看
[setup](../setup/README.md)，已经发生过且仍有复用价值的复杂诊断见
[debugging](../debugging/README.md)。

新问题进入对应 closure card；解决后把可复用的“症状 → 检查 → 定位 → 恢复”沉淀到这里或 debugging，
一次性时间线留在 task/Git 历史。
