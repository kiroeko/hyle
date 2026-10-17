# 实现速查

1. 先确认实际调用、所属模块和接口语义，检查现有 HYLE、标准库和成熟依赖是否已能自然满足。
2. 在声明附近说明 ownership、寿命、错误、并发、容量与必要转换；按[编码约定](../standards/coding-conventions.md)
   实现，避免让用户了解内部 provider 或手配内部宏。
3. 运行相关测试；影响公开头、构建或安装时验证实际消费者。性能改动按
   [比较方法](../performance/README.md)确认工作量与计时边界，保留完整样本。
4. 更新受影响的用法或设计说明，运行 `git diff --check`，报告结果与未验证项。

完整环境见 [setup](../setup/README.md)。验证副本使用独占 build 目录和本地文件系统，不反向覆盖
Windows 产品工作树；Windows 与同机 WSL 的性能实验错开运行。

构建维护时，内部 factor 的依赖与产品集合以 `cmake/HyleComponents.cmake` 为准，
对外链接仍使用 `hyle::<product>`。第三方选项在私有作用域设置普通变量，不强制覆盖父工程 cache；
测试 fixture 只扩展本次构建，不扩展安装包。Compute 能力通过
`hyle_target_compute_capability()` 同时声明 presence 宏与包元数据，不分别维护两份名单。
公开头的 DLL 导出宏集中在 `hyle/visibility.h`；增改 API 名称后用
`python scripts/generate-visibility.py --write` 更新，`--check` 检查同步。名称按 factor 自动生成，历史 IO 别名保留。
这些都是 target 的构建要求，库用户无需手配 `*_BUILD`、`HYLE_STATIC` 或 presence 宏。
