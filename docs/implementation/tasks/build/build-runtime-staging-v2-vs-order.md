# Windows runtime staging：Visual Studio 构建顺序

<!-- PROJECT UNIT REVIEW: AUDITED -->

## Closure card

- owner：build/package；baseline `8158d74d7b2b432b775314a8ca9695d9047aaff3`。
- trigger：[R1](compute-route-isolation-v1.md) 的 `delivery:compute` 首次源码构建失败：
  `hyle_runtime_hyle_selected_source_consumer` 在 Native DLL 生成前复制文件。失败日志为
  `5d34848a85fea055dde6877025275f4202c416b387f4821ba4d6671e7691bc4c-0.log`
  （Git common directory 的 `hyle-validation`）。其他已运行的 factor/header 与 component-contract 通过。
- 本卡是独立公共 build 修复，不在模块里重写治理/性能工具；旧
  [staging v1](build-runtime-staging-incremental-v1.md) 只证明当时 Ninja 的结果。
- consumer：Windows C++ 源码和 installed SDK 用户链接带 DLL 的公共 façade，首次构建即运行；
  只改 DLL 实现时不必重链接 EXE。正交 consumer 是静态 HYLE 加用户自定义 DLL。
- scope：既有 `HyleRuntimeStaging.cmake` 的 Visual Studio 顺序与对应有限回归。
- non-goals：新增依赖 resolver、枚举/复制全库 DLL、递归调用 build、改变产品 link/输出目录、
  Compute/Graph runtime、公共 selector 或性能仪器。
- tier：P0，仅构建时 copy-if-different；没有常态 runtime 成本，P1 不适用。
- 预算：定向调查最多 30 分钟，实施/验证最多 60 分钟、一次纠正，同时计入 R1 原四小时预算。
  不能证明首次及增量顺序即 BLOCKED，不靠二次 build 或手动 copy 签通过。

## 审计与有限 FIX

`add_custom_target(DEPENDS "$<TARGET_RUNTIME_DLLS:...>")` 的文件依赖在 Ninja 可建立顺序，
Visual Studio 的跨目录项目没有对应目标边。增加间接 INTERFACE 依赖并不能修正；不为此复制
CMake 的 transitive link/genex resolver。

本机 CMake `--help-command add_custom_command` 的 PRE_BUILD 合同区分 Visual Studio 与其他
generator；[CMake 官方生成器表达式文档](https://github.com/Kitware/CMake/blob/master/Help/manual/cmake-generator-expressions.7.rst)
说明 runtime DLL 列表来自真实 transitive shared targets。实际小型反例使用跨目录 DLL、INTERFACE
façade 和配置条件链接，确认 Visual Studio PRE_BUILD 在项目依赖完成后复制；只改 DLL 时 EXE/import
library SHA256 不变，staged DLL 更新；无改动构建不改 staged timestamp。探针只裁决构建机制。

| ID | 允许改动 | 验收 |
|---|---|---|
| VS1 | Visual Studio 使用其原生 PRE_BUILD 事件执行同一 per-config copy script；其余 generator 保留已有增量 target | 首次、增量、无改动、空闭包与重复调用；不增加虚构 link/源码 |
| VS2 | 现有 staging contract 接受明确 generator，覆盖跨目录 DLL + façade + 配置条件链接 | VS 与 Ninja，source/static-install/shared-install；实际 R1 Compute/Graph 交付 |

公开 helper 参数、真实 DLL 闭包、copy-if-different、安装文件及非 Windows 无操作不变。
Visual Studio 事件必须附着 consumer 所在 CMake 目录；这是 CMake 的 TARGET 事件约束。

## 七问与预声明验证

依赖版本和许可证不变；首次/增量是既有公开交付的真实缺陷；复用同一脚本和 CMake 原生事件；P0 无
运行时成本；无 FFI；source/static/shared 安装同一 helper；普通调用保持
`hyle_stage_runtime_dependencies(consumer)`。

只运行上述具名 staging 回归与 R1 已声明的 affected delivery/selector 门，保留旧候选通过的原始结果，
不将其改名为新 tree 的 receipt。每次 build 最多 2 jobs，重型验证串行；docs/closure/diff 必须通过。
未执行门不签 PASS。candidate 与最终七门在实现固定后记录；本卡当前 AUDITED，不表示修复已交付。
