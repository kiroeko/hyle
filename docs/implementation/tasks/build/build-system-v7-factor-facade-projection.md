# Build/package v7：内部 factor 与外部产品 façade 投影

<!-- PROJECT UNIT REVIEW: CLOSED -->

> Build/package 是横向开发单元，不进入 `module-status.json`。本页只授权双层 target/selector 基础设施；
> 30 个 factor 的产品源码迁移仍须各自登记状态并经过 `SCOPED/AUDITED`。长期合同见
> [`build-system.md`](../../../architecture/build-system.md)，能力与因子来源见
> [`capability-factorization-v1`](../project/capability-factorization-v1.md)。

## Closure card（`CLOSED`）

- unit / owner / round / kind：`build-package` / build-package maintainers /
  `build-v7-factor-facade-projection` / `REOPEN`；
- baseline：`7c3030d50d6321004baeee344d100536d08b4fdf`；前一轮：
  `build-v6-shared-source-cache`；
- trigger：Stage C 为解环和并行施工导出了 30 个细粒度 factor，却把每个 factor 又直接写成用户
  `find_package(... COMPONENTS ...)` 名与 `hyle::<factor>` target。用户在 2026-09-01 明确更正：factor
  首先服务 HYLE 内部依赖与施工，普通用户应按完整产品域使用，不能被迫理解 `platform_host`、
  `mem_reclaim`、`container_queue` 等内部切面；
- real consumers：以 `add_subdirectory(hyle)` 直接取得授权源码的推理引擎；安装 static/shared SDK 的
  C++ 工程；需要只构建一个 factor 的 HYLE 模块维护者和 CI；
- scope：把内部 semantic factor DAG 与外部产品 façade catalog 分成两张由同一中央 manifest
  生成的图；`HYLE_COMPONENTS`/`find_package COMPONENTS`/`hyle::<name>` 只表达产品域；增加只供
  顶层维护流程使用的精确 factor selector；source/static/shared 同名 façade；安装包可为静态传递闭包
  导出未承诺的 detail target，但不得把它们列入可用产品组件；更新选择、包、正负合同和文档；
- non-goals：移动或重写 30 个 factor 的头/源码/API；修改产品算法、性能实现、依赖 revision、Native
  FFI、Compute/Graph 产品状态；在本轮决定一个全库 monolithic artifact；兼容未发布的
  `hyle::container_intrusive`、`hyle::util_core` 或 Stage C `_all` 名称；
- current migration slice：先让现有 16 个实现 component 作为 factor 输入，外部 catalog 投影为
  `core,error,time,platform,mem,obs,simd,math,container,util,sched,io,compute,graph`。其中
  `container` façade 聚合当前 `container_intrusive + container`，`util` façade聚合当前
  `util_core + util`；其余暂为一对一。未来 factor task 只改内部映射，不另造用户命名体系；
- tier / performance：P0。目标是 factor 闭包与产品投影均为 `O(V+E)` manifest 工作；INTERFACE
  façade 不增加 runtime dispatch、allocation、metadata 或调用层。不得用 configure 毫秒点值决定
  target 形状；
- budget / stop：设计与实现 3 小时、本地验证 60 分钟；最多一次 schema 实现和一次合同修正。若
  source/static/shared 无法保持同名产品 façade，detail target 泄漏成用户组件，或精确 factor build
  必须重新引入语义环，则 `BLOCKED`，不在本卡转向重写 linker/package manager；
- required gates：双层 manifest 正负合同；产品选择和精确 factor 选择；Core、Time、Container、Util、
  Mem 的 source/static/shared relocated smoke；all-product feature-OFF configure；package 中 product
  可见/detail 不可枚举；现有 dependency/offline registry；`check-module-closure.py`、
  `check-docs.py`、`git diff --check`；
- current state：`CLOSED`。双层 manifest、产品 façade、精确 factor 施工入口、安装包投影和三种交付
  合同均已实现并通过本卡预声明验证；后续 factor 迁移只消费这套机制，不重开本卡。

## 本地 mature-peer 审计

审计使用 `C:\public\project-hyle` 下的干净 checkout，不把它们变成 HYLE 依赖：

| peer / commit | observed build shape | HYLE verdict |
|---|---|---|
| Folly `23ebefdf96c116700d3e134f04d95a2b60a509de` | `folly_add_library` 为 compiled 单元建立 OBJECT target，静态形态可产生 granular archive；共享形态让 granular INTERFACE target 路由到一次聚合的 `folly`，默认包入口仍是 `Folly::folly`。当前实现也安装 granular target，因此不能声称它们被技术隐藏 | `BORROW` 内部一次编译、INTERFACE 投影与内外命名分层；`REJECT` 直接复制单一全库 DSO 或把全部 granular target 当 HYLE 用户菜单 |
| oneTBB `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` | 用户只在 `tbb/tbbmalloc/tbbmalloc_proxy` 等独立产品语义间选择，包补齐真实组件依赖 | `BORROW` 小而语义化的外部 catalog；内部实现目录数不决定用户组件数 |
| ggml `d4716378882593333721eb33f153144b6885caf2` | `ggml` façade 聚合 `ggml-base` 与启用的 backend；backend 可静态链接或动态装载，用户不按每个内部 CPU source group 组装 | `BORROW` 产品入口与 backend/build unit 分离；不借其大量全局 feature option |
| Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c` / xsimd `3b3151ff665514a00a437cc985f47b7de99b772b` | 对用户分别提供少量 cohesive target；ISA/header 实现粒度不成为 package component | 支持 HYLE SIMD 保持一个产品 façade，provider/ISA target 留内部或 expert journey |
| Abseil `2c004366e983c5be8334ac1ea3d4420e8fbcbea7` / Boost `78505eab9b203e3e26e83893d0685f160133fa0f` | 广泛公开细粒度 CMake target，适合用户确实按小库名称组合的生态 | `REJECT AS DEFAULT`：HYLE 的多数新 factor 是为内部解环产生，不自动满足“用户愿意独立选择并理解”的产品门；可作为内部 DAG/ownership 参考 |

这组 peer 没有给出唯一物理 artifact 答案，却共同证明逻辑 target 粒度、物理 archive/DSO 粒度与用户
产品入口是三个独立决策。本轮只冻结前两张逻辑图；是否像 Folly 一样把共享实现进一步聚合，是以后有
启动/部署/符号边界证据时的独立 build task。

## 冻结术语与合同

1. **factor**：`module-status.json`、源码 ownership、测试、直接 HYLE 依赖和施工状态的最小单元；只有
   factor 进入内部 DAG。factor target 不能依赖 product façade。
2. **product façade**：用户选择和链接的完整问题域入口；纯 `INTERFACE`，没有源码、状态、decision
   authority 或反向产品依赖，只投影一组 factor roots。
3. **feature**：会改变某个产品能力/外部依赖的显式构建选择，如 IO TLS/H2；不是为了 DAG 解环而产生的
   factor 别名。安装包必须报告实际 feature，不能把未构建能力伪装成完整 façade。
4. **build-only mechanism**：vendor、系统库、生成器或横切 config target；既不是 factor，也不是产品。

`HYLE_COMPONENTS` 保留名称但重新明确为**产品 selector**，因为这是源码消费者实际要回答的问题。
维护者精确构建 factor 使用具名 developer-only selector；两者互斥，后者不能出现在安装包、用户指南或
父工程默认路径中。`all` 表示所有产品，不表示对外公开所有 factor。

## Capability / gap ledger 与冻结 FIX

| ID | current gap | frozen action | verification |
|---|---|---|---|
| `BUILD-V7-MODEL-01` | `component` 同时表示状态单元、DAG vertex、用户 selector 和 imported target | 中央 manifest 分别登记 factor 与 product façade；factor 声明仍拥有依赖，product 只列 roots | unknown/duplicate/root/order/schema 负向 fixture |
| `BUILD-V7-SELECT-01` | `HYLE_COMPONENTS` 会迫使用户写内部 factor；维护者又需要精确根 | `HYLE_COMPONENTS` 只接受 product；新增 top-level/developer-only exact-factor selector，二者互斥；两种闭包都复用同一 O(V+E) resolver | Container/Util 产品根与 intrusive/util_core factor 根正负配置 |
| `BUILD-V7-TARGET-01` | 每个 factor 自动获得 `hyle::<factor>` 公共 alias | factor 保留无 namespace 的 build target；product 创建 `hyle::product_<name>` 实体与 `hyle::<name>` alias，INTERFACE 链接 exact factor roots；内部边永不指向 façade | final graph audit + source consumer target inventory |
| `BUILD-V7-PACKAGE-01` | `Hyle_AVAILABLE_COMPONENTS` 等于已构建 factor；安装名泄漏内部图 | factor 以 `hyle::_factor_<id>` 仅为传递静态/共享闭包导出，product 以 `hyle::<name>` 导出；available/find components 只列完整 façade | relocated package config 与 target inventory；请求内部 ID 必须失败 |
| `BUILD-V7-HEADERS-01` | selected headers 目前按“用户 component”安装 | 头仍由 factor 唯一拥有；安装产品时复制其 factor 闭包的 owned headers，不能按 façade 目录整树复制 | Time/Container/Util/Mem selected SDK header 正负门 |
| `BUILD-V7-DELIVERY-01` | Stage C 另造 `_all`，三种交付会形成两套入口 | 删除 `_all` 设计；source/static/shared 都只承诺同名 product façade。物理 factor archive/DSO 是 package detail，不是 ABI/API 产品 | 三交付同一 consumer CMake；无 `_all`/旧 factor target |
| `BUILD-V7-FEATURE-01` | `io_all` 会把可选 TLS/H2 与内部分解混在一起 | product façade 与 feature 正交；当前 option/feature metadata先保持，未来 IO factor化必须让 package 明确报告已构建 feature | feature-OFF package不声称 TLS/H2；feature-ON合同仍显式 |
| `BUILD-V7-STATE-01` | façade 若进入 module status 会与 factor 重复状态/authority | registry 只记录 factor；façade由 build manifest机械生成，不继承或制造 CLOSED 结论 | closure checker仍只有真实 component记录；迁移新 factor时另存 previous registry |
| `BUILD-V7-DOC-01` | architecture、Stage C和用户指南会同时给出冲突入口 | architecture写长期双层合同；Stage C保留历史表但加 superseding note；用户指南只写已实现 product façade | docs checker + stale-name scan |
| `BUILD-V7-WIN-01` | VS 2026 `VsDevCmd` 可回传重复 `PATH`/`Path`；PowerShell 后导入的旧值覆盖含 `cl.exe` 的新值，isolated wrapper 配置失败 | Windows wrapper 对 PATH 大小写归一，只接受包含已选 VS toolchain 的初始化值；其他变量维持原导入 | fresh wrapper Core configure；负面不再表现为 unknown compiler |

冻结 `FIX` 仅限上述构建/文档投影。30-factor 源码迁移、模块 API 或性能不属于本卡；实现某个 factor 时
若发现 product root 集错误，必须退回本卡 `AUDITED` 更新映射，不能在模块 CMakeLists 私造 façade。

## 实施顺序

1. 先在当前 16-factor 树上实现双层 schema、选择器与 façade，证明机制本身不依赖未来源码布局；
2. 改 package export/header selection 和正负合同，移除旧公开 factor 名；
3. 更新 source/static/shared 样例和用户文档，签署 current-tree candidate；
4. v7 `CLOSED` 后，按 Stage C 拓扑登记新增 factor ID；每个 factor 迁移只调整 factor 声明与 product roots，
   不再修改用户产品入口机制。

## Verification manifest

| gate | result | current-tree evidence |
|---|---|---|
| 双层 manifest 与 source target 正负合同 | `PASS` | 14 个公开 product、16 个当前 factor；产品选择与 `HYLE_DEV_FACTORS=container_intrusive` 精确选择通过；把 `container_intrusive` 当产品及用 exact-factor 模式安装均按合同失败 |
| selected source/static/shared relocated journey | `PASS` | Core、Time、Container、Util、Mem 均以同名 `hyle::<product>` 完成源码、静态包、共享包和搬迁后消费；Container/Util 包只公开产品名，同时保留所需 `_factor_*` 传递细节 target |
| all-product journey | `PASS` | feature-OFF 全产品配置通过；shared/static 全产品 SDK 均完成构建、安装、搬迁和全部 package consumer 探针（含 Obs、Platform、Sched、SIMD、Time、Util） |
| dependency source cache / offline / package registry | `PASS` | 共享源码缓存、严格离线依赖和 package-registry 三项合同均通过 |
| Windows isolated wrapper | `PASS` | VS 2026 fresh Core 与 Container 配置/构建通过；修复重复 `PATH`/`Path` 覆盖选中 MSVC toolchain 的问题 |
| 项目一致性 | `PASS` | `check-module-closure.py`、`check-docs.py`、`git diff --check` 与改动脚本语法检查通过 |

本轮最终代码量口径：全库代码 `412,830` 行；自实现含测试 `99,270` 行；自实现不含测试
`61,670` 行；Build/package 实现 `9,156` 行。文档改动不计入代码口径。

## 终态裁决

`CLOSED`。当前树已把**内部施工 factor**、**公开产品 façade**和**物理包细节**分开：普通源码、静态库和
共享库消费者只面对 14 个稳定产品入口；维护者可精确构建 factor，但该入口不能安装；内部 factor 即使因
静态传递闭包出现在包 target 文件中，也只以 `_factor_*` detail 名存在，不进入
`Hyle_AVAILABLE_COMPONENTS`。façade 是纯 `INTERFACE` 投影，不引入运行时分派、调用层、常态元数据或
算法性能成本。

本卡没有改变产品算法/API，也没有声称 Compute/Graph 已收口；本地验证覆盖当前 canonical Windows
checkout，未在本卡重新运行 hosted Linux/macOS CI。只有以下新事实可重开：source/static/shared 无法再
维持同名产品入口；内部 factor 泄漏为可请求产品；新 factor 无法由当前 O(V+E) manifest 表达；或可复现的
跨平台 package 缺陷。新增 factor 本身、产品功能重构和物理 monolith 取舍分别登记独立任务。
