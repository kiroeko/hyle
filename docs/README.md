# HYLE 文档

HYLE 的人读文档只分四类。分类依据是文档回答的问题，而不是文件扩展名或作者。

| 目录 | 回答的问题 | 内容 |
|---|---|---|
| [`architecture/`](architecture/README.md) | 为什么做、希望成为什么、为什么这样设计？ | 项目背景、需求与愿景、系统/模块架构、正式决策及决策背景 |
| [`implementation/`](implementation/README.md) | 某次工作实际发生了什么？ | Jira 式任务、调查过程、结果、提交、当前施工状态与历史事实 |
| [`developer/`](developer/README.md) | 维护者下一次应当怎样正确地做？ | 开发规则、模块协作、review 方法、构建测试、性能仪器、debug 与 troubleshooting |
| [`user/`](user/README.md) | 外部 C++ 用户怎样完成任务？ | 安装集成、模块选型、自然 API 写法、生命周期和常见陷阱 |

## 不属于人读文档的内容

- 机器可读合同跟实现放置，例如组件图在 `cmake/`，未来 review manifest 放在相应工具目录。
- 原始 CSV/JSON/log/hash 放在仓库根级 [`evidence/`](../evidence/README.md)；实施文档只解释其
  candidate、环境和结论。
- 法律文本使用根级 `LICENSE`/`NOTICE` 及第三方许可证；architecture 只能解释许可决策。
- 生成式 API reference 属构建产物，用户入口可以链接，但不手工复制公开头。

## 归档纪律

1. 同一事实只有一个权威来源；其他位置只链接。
2. architecture 不写完成百分比、某次测试数或 benchmark 点值。
3. implementation 不自动成为当前产品合同；任务结束后把仍有效的决策、维护方法和用户写法
   分别沉淀到另外三个目录。
4. developer 不保存一次性任务状态；它只保留可重复执行的规则、工具和排查方法。
5. user 不要求读源码、内部里程碑或历史性能调查才能完成正常任务。
6. 删除被替代文档，不维护兼容跳转页；Git 历史承担恢复和追溯。
7. 修改文档后运行 `python scripts/check-docs.py`。
