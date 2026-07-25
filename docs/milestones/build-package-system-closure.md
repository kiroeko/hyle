# build/package system 严格收口任务（2026-07-25）

> 本页是施工与验收记录。收口后，稳定的项目组织模型归
> `docs/architecture/build-system.md`，用户迁移与组装 recipe 归 guide；本页不长期兼任用户手册。

## 核心判据

构建系统不是“把当前 checkout 编过”的脚本集合，而是 HYLE **项目组织架构的可执行描述**：
模块有哪些源码、公开头、C ABI、直接依赖和可选后端，必须从 target 图读出来。换目录、换机器、
换生成器、嵌入另一个项目或只组装部分模块时，不应依赖施工者记忆，也不应修改 HYLE 源码。

## 完成定义

- [ ] 清点根/模块 CMake、helper、preset、toolchain、CI、构建脚本和文档，画出现状 target/
  option/feature/dependency 图；删除重复或相互矛盾的环境契约。
- [ ] 每个产品模块由一个规范 target 描述自身 source、public header、C ABI、直接 link 依赖和
  compile feature；目录顺序、偶然 include 与全局变量不能冒充依赖。
- [ ] 整库与模块子集都能按依赖闭包组装；CPU-only 与 CUDA/Vulkan/TLS/trace 等可选能力隔离，
  未选择的模块/benchmark 不探测、下载或链接其第三方。
- [ ] shared/static、源码嵌入（FetchContent/`add_subdirectory`）、安装后
  `find_package`/component targets 四条路径使用一致的 public contract 和传递依赖。
- [ ] Windows、Linux/WSL、macOS 使用一致的高层入口；Ninja/Make/MSBuild 不是隐藏前置，
  compiler/toolchain/SDK 要么被发现，要么给出可操作失败，不能写死本机绝对路径。
- [ ] 交叉编译不依赖不可执行的 `try_run`；生成文件、cache、artifact 全在 build tree，复制源码
  或改变平台不会误用旧 cache；常见并发/中断不会留下需要手工猜测的锁死状态。
- [ ] tests/bench 按模块归属，并有机械 aggregate target；构建某模块验证面会同时产出其 C++、
  pure-C、header/surface/ABI 门，而不是让 CTest 运行缺失或陈旧 binary。
- [ ] architecture 解释组织模型和边界，guide 给出“选哪些模块、怎样嵌入/安装/迁移”的可运行
  recipe，setup 只记录工具链准备，milestone 只保留发现与施工证据。
- [ ] 三平台验证 clean configure/build/test；至少覆盖整库、core-only/代表性子集、CPU-only、
  shared/static、`add_subdirectory`、install/`find_package` consumer 和一条交叉编译 configure。

## core 收口期间已经暴露的基线问题

这些是进入本轮的证据，不预设最终修法：

1. 配置一个 core benchmark 会遍历 compute/io 等无关可选后端，并可能 FetchContent nghttp2；
   Linux core bench 还被全局 Highway/xsimd vendor 前置阻断。
2. macOS `bench` preset 假定 Ninja；同一机器使用 Unix Makefiles 能编，说明 generator 是入口
   假设，不是产品需求。
3. 只构建若干模块 C++ test target 后，CTest 仍会登记并尝试运行未生成或陈旧的 pure-C test；
   “登记什么”和“aggregate target 产出什么”没有闭环。
4. Windows 构建被外部超时中断后 Ninja 仍可能持有对象文件；复制到 WSL/macOS 的旧 build
   cache 又不能跨平台复用。源码镜像、build tree 和进程生命周期边界需要明确。
5. 当前 guide 明说 install/`find_package` 尚未实现，静态消费者需手写 mimalloc 等传递依赖；
   这意味着现有 target 图还不能完整承载迁移与组装。

## 当前进度

- [x] 引入 `HYLE_COMPONENTS` 产品根与直接依赖 fixed-point closure；unknown component fail-fast，
  configure 输出 requested/resolved manifest。
- [x] 第三方发现/取得受 io/compute 组件约束；core、mem、container 子集不再探测 GPU/协议依赖，
  默认依赖区从源码 `.deps` 移到各自 build tree；离线源码用标准 `FETCHCONTENT_SOURCE_DIR_<NAME>`
  注入，不跨 toolchain 共享依赖生成物。
- [x] TLS/CUDA 从“构建机碰巧找到就改变产物”改为 `AUTO|ON|OFF`：ON 缺依赖 fail-fast，OFF
  零探测，正式 SDK/CI 可声明确定 feature contract；Windows preset 不再写死维护者 vcpkg 路径。
- [x] HYLE 模块显式选择 STATIC/SHARED，不再改 embedding parent 的 `BUILD_SHARED_LIBS`、输出目录
  和 configuration types；三个 source-embed sample 已按组件根组装。
- [x] tests 按存在的 module target 创建；新增 `hyle_<module>_tests` / `hyle_tests` aggregate，
  pure-C binary 不再漏构建；static C surface 改查 archive globals、shared 仍查 exports。
- [x] Windows source embed 用 `hyle_stage_runtime_dependencies()` 局部复制传递 DLL；不恢复全局
  输出目录劫持。container sample 已 build+run。
- [x] benchmark 按组件惰性创建；`hyle_<module>_benches` / `hyle_benches` 明确聚合归属，core、
  mem、container benchmark 不再要求全局 SIMD vendor 或 compute/io 配置。
- [x] relocatable install/export/component package 落地；build/install target 同名 `hyle::<module>`，
  shared/static simd+mem 独立 consumer 均真实运行，static mimalloc link-only 后端自动传播。
- [ ] compute/ggml/volk 的 fetched 依赖打包；交叉编译、
  preset/toolchain/CI 与三平台最终矩阵。
- [x] io/nghttp2 package 闭环：固定版本协议核始终为私有 PIC archive；shared io 吸收实现，static
  io 通过 export graph 传递 archive/生成头；MIT 全文和 `io_http2` feature manifest 随包迁移。
- [x] 当前发布策略确定为暂时闭源：根 `LICENSE` 使用 `LicenseRef-HYLE-Proprietary` 对应的
  rights-reserved notice；`THIRD_PARTY_NOTICES.md` 区分原创、vendored、ported、fetched，安装包
  按 target 图携带 HYLE notice 与实际第三方 license payload。
- [x] 授权与 artifact shape 解耦：binary package 覆盖 headers+shared/headers+static；新增
  `hyle_source_package` 从 clean Git commit 生成完整 tracked source zip、commit manifest 与 SHA-256，
  允许特定客户在 proprietary 授权下取得源码而不把它误称为开源。

### 第一轮验证证据（`86cb325`）

- Windows MSVC 19.50：core-only shared 4/4、static 4/4；同一 build tree 从 core 切换到 mem，
  resolved=`error,platform,mem`，统一 aggregate 构建后 12/12；未出现 io/compute 依赖。
- Windows source embed：container sample 只组装 `error,platform,mem,container`，35 步构建后
  runtime staging 得到 4 个传递 DLL，程序真实运行并输出 SwissTable 结果。
- WSL GCC14：core-only shared `hyle_core_tests` 4/4，未探测 GPU/协议依赖。
- Apple M4 AppleClang 21：Unix Makefiles（机器没有 Ninja）core-only shared 4/4；证明组件入口
  不依赖 preset 的 generator 假设。
- 默认 `all` 全组件 configure 已通过；完整 `hyle_tests` 会重编 ggml CUDA，超过本轮合理成本后
  精确终止该构建进程树。它留到 compute 隔离与最终矩阵，不用烧机器冒充新增证据。

### benchmark 组装验证（本轮）

- Windows MSVC 19.50：`HYLE_COMPONENTS=core` 不提供 vendor 路径即可 configure；
  `hyle_core_benches` 7 步构建并真实运行，Ninja 图只有 core/nanobench/core bench。
- WSL GCC 14.2：同一 core-only 配置不提供 vendor 路径，7 步构建；目标图同样没有
  mem/platform/io/compute/simd 泄漏。默认 GCC 13.3 按既有 14+ 契约 fail-fast。
- Windows `mem`：resolved=`error,platform,mem`，`hyle_mem_benches` 只产出 mem allocator/scenario
  两个 benchmark；Windows `container`：resolved=`error,platform,mem,container`，模块 aggregate
  完整产出 5 个 container benchmark。
- Windows `simd`：只有此组件配置才校验固定版 Highway/xsimd；`hyle_simd_benches` 机械聚合
  当前平台全部 20 个 SIMD benchmark executable。

### install/package 第一轮验证

- Windows MSVC 19.50：core-only shared 成功 install，core-only package 对 `COMPONENTS simd mem`
  按预期拒绝；不会静默链接不存在的库。
- Windows shared `simd;mem`：安装 resolved=`error,platform,mem,simd`，独立 sample 仅凭
  `CMAKE_PREFIX_PATH` configure/build/run；runtime helper 自动 staging 4 个传递 DLL。
- Windows static 同一组件集：首次构建暴露 `Status::WithContext` 静态 strong/inline 双定义；修正为
  static 全 inline、shared 保留 DLL entry 后，独立 consumer 自动链接 platform/error/mimalloc 并运行。
- Windows pure-C consumer（project 仅启用 C）分别消费 shared/static error package，均构建运行；
  static imported target 正确带入 C++ linker runtime，不改变 `.c` 的编译语言。
- shared `simd;mem` 安装前缀复制到全新目录后，独立 consumer 从复制前缀重新 configure/build/run，
  证明 package config、targets、include/lib 路径不固化原 build/install 位置。
- WSL GCC 14.2：shared/static error 均 install；纯 C consumer 只凭 `CMAKE_PREFIX_PATH` 构建运行。
- Windows 自定义相对 GNUInstallDirs（`runtime/libs/headers/data`）安装后，纯 C consumer 从新
  include/lib/config 布局构建运行；绝对子目录被拒绝，迁移根只由 prefix 表达。
- missing-component 诊断列出 requested 与 available；安装包同时公开
  `Hyle_AVAILABLE_COMPONENTS`/`Hyle_AVAILABLE_FEATURES` 作为可审计组装 manifest。
- install/export target 直接来自产品 target 图；测试构建禁止安装，避免 `HYLE_TESTING` 布局泄漏；
  尚未随包迁移的 fetched runtime/archive 组合 configure fail-fast。
- proprietary 决策不覆盖第三方：Highway port 选择 Apache-2.0，JPEG XL/xsimd 保留 BSD-3，
  原来 9 个不可机读的 `SPDX: same as HYLE` 改为 `LicenseRef-HYLE-Proprietary`。
- Windows MSVC 19.50 io+HTTP/2 shared/static package 均由全新 Release build 生成；独立纯 C
  `samples/06_installed_io_http2` 只通过 `find_package(Hyle COMPONENTS io)` 构建并真跑。shared 包的
  nghttp2 PIC archive 被 `hyle_io.dll` 吸收、无第二 DLL；static 包自动传播 `hyle::nghttp2_static`、
  生成头和 archive，consumer 没有源码 checkout 路径或手写 link item。
- shared io 的 `Hyle_Runtime` 单组件安装带 7 个 HYLE DLL 及完整 license closure，只有
  `nghttp2-LICENSE` 而无 standalone nghttp2 binary；安装后的 CMake 文件没有 source/build 绝对路径。
- `hyle_source_package` 在施工脏树上按预期 fail-fast；干净提交上生成 zip + SHA-256，归档只含统一
  `hyle-0.1.0-source/` 前缀，manifest 固定 commit，解压后的 core-only consumer build 可独立完成。

### 本轮四项代码量

| 固定口径 | code |
|---|---:|
| 全库（含 docs/in-tree third_party） | 247,200 |
| 自实现（含测试） | 116,284 |
| 自实现（不含测试） | 79,957 |
| build/package 实现 | 3,431 |

最后一项统计自有 root/module/test/bench/sample CMake、`cmake/`、preset，以及
build/fetch/setup/QEMU scripts；不含 vendored CMake、测试源码和文档。

## 审查顺序

1. **现状建模**：列 target、option→macro、public/private dependency、第三方获取与生成目录；
   先用 clean configure 证明问题，不凭 CMake 文件长度猜。
2. **组织模型**：确定模块 target/namespace/component 与依赖闭包，修正 source/header/C ABI 归属。
3. **选择与隔离**：让子集、CPU-only、tests、bench 和可选后端按需求惰性进入图。
4. **消费路径**：统一 build-tree、source-embed、install-tree 的 namespaced targets、导出宏和依赖。
5. **环境入口**：收束 preset/toolchain/scripts/CI，执行契约由机器检查，文档只教用户选择。
6. **迁移矩阵**：从全新 build tree 在三平台与外部 consumer 验证，再归档四项 LOC 与关闭触发器。

## 负空间

- 不自造 C++ 包管理器；包管理器只负责取得 HYLE，CMake package targets 负责表达 HYLE。
- 不为了“一键全开 benchmark”让普通产品配置接管所有参考库。
- 不把开发机 CUDA、Vulkan SDK、OpenSSL、Ninja 路径固化成项目架构。
- 不用一个巨型 `hyle` target 抹掉模块边界；可以提供方便的 aggregate，但组件 target 才是事实。
