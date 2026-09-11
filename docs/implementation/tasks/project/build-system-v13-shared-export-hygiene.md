# Build system v13：共享库导出边界

## Project unit card

- baseline：`526dd5aee87b33936835fbb7cd9c447f7574d489`；
- trigger：当前 user/architecture 合同已经把早期 C ABI 判为删除、Native FFI 判为 `N/A`，但对现有
  Windows shared SDK 的二进制扫描发现 `hyle_sched.dll` 仍公开
  `jump_fcontext/make_fcontext/ontop_fcontext` 三个内部汇编入口；公共 `hyle_add_module()` 也没有给
  Unix shared factor 设置默认隐藏 visibility，因而“只导出显式 C++ API”尚未成为机械合同；
- objective：让 source/static/shared 三种 C++ 交付继续同形，同时把共享 artifact 的公开符号限制为
  显式 `HYLE_*_API` C++ 面；内部 C、汇编和 vendored seam 可链接但不可成为无意 Native FFI；
- scope：公共 module target visibility、Sched 的 Windows x64 fcontext 汇编可见性、安装后 shared
  artifact 检查器、selected/whole-package smoke、构建/FFI/分发文档；
- non-goals：增加 Native FFI、承诺稳定 C++ ABI、维护跨编译器符号白名单、改公开 API/对象 layout/
  runtime 算法、修改组件 DAG 或依赖锁、重写 option 系统、触碰 Compute/Graph、运行 P1；
- performance：`P0`。改动只影响 link/export 元数据，不进入运行时路径；
- stop：若默认隐藏暴露出未显式标注的公开 C++ API，则修正该公开导出标注并继续交付验证；若需要改变
  API/算法或出现与本任务无关的 package 缺陷，则停止并另行 scope；
- 预声明验证：检查器 focused test；Windows Sched source/static/shared relocated consumer 与实际 DLL
  export scan；Unix shared artifact scan；全 CLOSED 产品 shared/static package smoke；component contract；
  `check-module-closure.py`、`check-docs.py`、`git diff --check`。不因构建元数据运行模块 P1。

## 审计与有限改动集

当前入口只交付 C++23：源码嵌入、静态 SDK、共享 SDK 使用同一 factor DAG 与产品 façade。根级普通
开关只有 tests/bench/install/static/werror；昂贵 feature 由 owner gate 采用 `AUTO/ON/OFF` 或显式
`ON/OFF`，没有退役 C API/FFI option、target、header 或 package metadata。`HYLE_ABI_VERSION` 只用于
shared loader 的 `SOVERSION`，0.x 使用完整项目版本，不代表 C ABI 或跨工具链稳定 C++ ABI。

| ID | 已证实问题 | 有限修复 | 验收 |
|---|---|---|---|
| BV13-01 | Unix compiled factor 默认导出全部 default-visible 定义 | shared `hyle_add_module()` 统一启用 hidden C/C++ visibility 与 hidden inline | 三交付 consumer；shared artifact 仍可调用公开 C++ API |
| BV13-02 | Windows Sched DLL 意外导出三项 raw fcontext | MASM x64 procedure 改为 object 内 `PUBLIC`，不使用 DLL `EXPORT` | Sched 能链接/运行；DLL 不再出现三项符号 |
| BV13-03 | 没有 artifact 级回归门 | 增加跨 PE/ELF/Mach-O 的受限 plain-C export 检查，并接入 selected 与 whole-package shared smoke | 旧 `hyle_*`、vendored `mi_*`、raw fcontext 复活时 fail closed |
| BV13-04 | 交付文档未明确默认隐藏与 artifact 门 | 校准 architecture、Native FFI policy、user distribution | 用户不把 shared C++ SDK 误认成 FFI；维护者知道私有 seam 要求 |

检查器只拒绝项目明确不应公开的 plain C symbol，不冻结编译器修饰后的 C++ symbol 集；HYLE 仍未承诺
稳定 C++ ABI。Sched 的 module seam 改动由独立的 `sched-v11-private-fcontext-export` 合法重开卡承接，
横向任务本身不伪装成产品 component。

## 结果

施工中。
