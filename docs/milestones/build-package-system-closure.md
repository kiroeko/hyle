# build/package system 严格收口任务（2026-07-25）

> 本页是施工与验收记录。收口后，稳定的项目组织模型归
> `docs/architecture/build-system.md`，用户迁移与组装 recipe 归 guide；本页不长期兼任用户手册。

## 核心判据

构建系统不是“把当前 checkout 编过”的脚本集合，而是 HYLE **项目组织架构的可执行描述**：
模块有哪些源码、公开头、C ABI、直接依赖和可选后端，必须从 target 图读出来。换目录、换机器、
换生成器、嵌入另一个项目或只组装部分模块时，不应依赖施工者记忆，也不应修改 HYLE 源码。

## 完成定义

- [x] 清点根/模块 CMake、helper、preset、toolchain、CI、构建脚本和文档，画出现状 target/
  option/feature/dependency 图；删除重复或相互矛盾的环境契约。
- [x] 每个产品模块由一个规范 target 描述自身 source、public header、C ABI、直接 link 依赖和
  compile feature；目录顺序、偶然 include 与全局变量不能冒充依赖。
- [x] 整库与模块子集都能按依赖闭包组装；CPU-only 与 CUDA/Vulkan/TLS/trace 等可选能力隔离，
  未选择的模块/benchmark 不探测、下载或链接其第三方。
- [x] shared/static、源码嵌入（FetchContent/`add_subdirectory`）、安装后
  `find_package`/component targets 四条路径使用一致的 public contract 和传递依赖。
- [x] Windows、Linux/WSL、macOS 使用一致的高层入口；Ninja/Make/MSBuild 不是隐藏前置，
  compiler/toolchain/SDK 要么被发现，要么给出可操作失败，不能写死本机绝对路径。
- [x] 交叉编译不依赖不可执行的 `try_run`；生成文件、cache、artifact 全在 build tree，复制源码
  或改变平台不会误用旧 cache；常见并发/中断不会留下需要手工猜测的锁死状态。
- [x] tests/bench 按模块归属，并有机械 aggregate target；构建某模块验证面会同时产出其 C++、
  pure-C、header/surface/ABI 门，而不是让 CTest 运行缺失或陈旧 binary。
- [x] architecture 解释组织模型和边界，guide 给出“选哪些模块、怎样嵌入/安装/迁移”的可运行
  recipe，setup 只记录工具链准备，milestone 只保留发现与施工证据。
- [x] 三平台验证 clean configure/build/test；至少覆盖整库、core-only/代表性子集、CPU-only、
  shared/static、`add_subdirectory`、install/`find_package` consumer 和一条交叉编译 configure。

## 2026-07-27 独立模块复审

旧证书证明三种交付和 target 图曾完整通过，不代表构建系统可以永久免审。它与代码模块同样有
功能面、正常路径性能、用户接口和文档合同，因此本轮因底座 API 变化和日常构建异常复开：

- [x] 把工具链环境、依赖获取、CMake target 图、生成器数据库和源码失效分层诊断；Windows
  wrapper 禁止位置参数误绑定，多 target/CTest 的剩余参数原样透传；
- [x] 公开 `version_defs.h` 成为 CMake/package/loader 唯一版本源；修正未冻结 0.x ABI 的包版本
  语义：当前 full version exact、同 minor compatible，共享库使用 `major.minor` SONAME；
- [x] Windows clean `time` 子集测得产品 configure 2.308s / reconfigure 0.128s / no-op build
  0.048s；本地注入 GoogleTest 后测试图为 2.323s / 0.423s / 0.053s；
- [x] `build/.ninja_deps` 截断导致每次重编约 100 个对象已确认为构建树损坏；重建该数据库后
  第二次同目标 70ms 且 `no work to do`，不是 HYLE 源码依赖图持续失效；
- [x] 消除 `HYLE_TESTING` 令普通单测库改变 public pool layout / 热路的双形状问题；严格正确性门
  以 `HYLE_TEST_INSTRUMENTATION=OFF` 覆盖 production layout，instrumented diagnostics 是额外层；
- [x] 在 WSL 与 Apple 对同一 clean/reconfigure/no-op/component-isolation 合同复证，并把所有
  build 性能点值集中到 `docs/performance/benchmarks-build.md`；
- [x] 对照 oneTBB/Abseil/libevent/mimalloc 后裁决 pkg-config：0.1 不发布只能覆盖简单 shared
  link、却无法完整表达 static C++ link driver、imported archives、CUDA/OpenSSL feature closure 的
  半接口；三种交付形状以 relocatable CMake package 为唯一机器合同，非 CMake 构建系统导入该
  package 或使用薄 CMake bridge。真实消费者提出 `.pc` 时，必须以 static/shared 全闭包 smoke
  为实现门，而不是只生成 `-lhyle_<leaf>`。

复审最终证据：production-shape 的 error/platform/time/mem C++、pure-C、surface 与 header 门在
Windows/WSL/M4 均为 16/16；shared/static mem SDK 的 C/C++ 外部 consumers 在三端均为 2/2，且
package 将 0.1.0 判为 exact、接受 0.1、拒绝 0.2。ELF SONAME 为 `libhyle_mem.so.0.1`，Mach-O install-name 为
`@rpath/libhyle_mem.0.1.dylib`。默认 instrumented Windows 相关组仍为 13/13，两个测试形状互补。

Windows 的 `windows-ninja` 入口本来就要求 `scripts/hyle-windows.ps1` 或已初始化的 Developer
Shell。普通 PowerShell 直接调用缓存中的 `cl.exe` 缺少 `INCLUDE/LIB`，是明确入口合同被绕过，
不是 HYLE 丢失标准头；不过错误信息与 wrapper 易用性仍纳入本模块 guide 审核。fresh WSL/M4
首配曾观察到 44--112s，包含 FetchContent 获取/配置，不能与上述已注入依赖的 CMake 图成本混写。

## 2026-07-29 模块归属重构后的严格回审

`30d3b40` 把 root/tests/bench 巨型清单下沉到各模块所有权目录后重新按产品合同审计，而不是只看
“能编过”。本轮在 `75a8d3f` 完成以下闭环：

- 显式 `-DHYLE_COMPONENTS=` 不再静默展开为 `all`；未设置仍默认全库。source-embed 父工程用普通
  变量选择组件时，cache/GUI 现在镜像实际值，不再出现实际 `container`、cache 却写 `all`；
- 声明时的 direct-dependency 检查保留，并新增所有模块装载后的 target property 终态审计；configure
  contract 会故意追加一条晚期内部 link 边，证明绕过会失败；
- GitHub Actions 连续失败的 run `30241911516` / `30381984449` 均落在 shipping-layout mem 同三条
  测试。两条 cache 测试错误假定 runner CPU 数对应的 shard refill 恒为 1；region 测试在线程先后
  结束时把合法 block 复用误判为并发生命期重叠。修正为容量合同和显式重叠生命周期后，同一 clean
  source archive 在 Windows/WSL/M4 均为 16/16，WSL 单 CPU 限制下原三条也为 3/3；Windows
  instrumented mem 仍为 4/4；
- CI 的 shipping-layout 只构建闭包聚合 `hyle_tests`。VS 2026 稳态 no-op 从四个 module roots 的
  约 4.44s 降为约 2.21s；Ninja 仍是一张 DAG，不虚构相同倍率；
- Linux CI 新增 complete-source 硬门：从 clean commit 生成 zip，核对 SHA-256、commit manifest、
  LICENSE/THIRD_PARTY_NOTICES，解包后脱离 Git checkout 构建 `hyle_core`。Windows 本地同脚本已对
  `75a8d3f` 全链通过；
- 当前三平台 clean/reconfigure/initial/no-op 点值已替换旧证书；no-op 为 Windows 47--48ms、WSL
  不超过 10ms、M4 60--120ms，没有模块拆分后的常态重编或数量级回归。

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
- [x] compute/ggml/volk fetched 依赖打包：只编私有 PIC CPU/可选 CUDA/volk 闭包；shared 吸收、
  static 导出 link-only archives/头/依赖，许可证和 feature manifest 随包迁移。
- [x] 平台 preset/toolchain/CI 入口：Windows/Linux/macOS 有同构 configure/build/test/workflow；
  Windows 明分 VS generator 与 vswhere+Ninja；AArch64 走标准 toolchain file；CI 含 core、
  production-shape 和 package smoke。
- [x] 三平台最终发布矩阵（all-component CPU SDK、代表性 source embed、远端 CI 结果）。
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
  fetched runtime/archive 只有闭包、头与 license 全部进入 target 图后才允许安装。
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

### compute package 闭包验证

- Windows MSVC 19.50 CPU shared compute：Release 从空 build tree 构建并安装；ggml-base/cpu/ggml 与
  volk 被吸收到唯一 `hyle_compute.dll`，安装前缀没有供应商 DLL/archive。独立
  `samples/07_installed_compute` 仅凭 package target 构建运行，枚举 2 个 device，Vulkan RTX 4070
  真可用；安装树没有 source/build 绝对路径。
- CPU static compute：134 步实际构建；包导出 ggml-base/cpu/ggml、volk、mimalloc 和 Threads 闭包，
  安装所需 ggml/volk/Vulkan 头。独立 consumer 未手写供应商库即可链接运行。
- CUDA static compute：显式 `CMAKE_CUDA_ARCHITECTURES=89` 后 275 步实际构建，产出 ggml-cuda 并
  导出 CUDA runtime/cuBLAS/driver/NVRTC 依赖；独立 consumer 真链接运行，枚举 CUDA RTX 4070。
  未指定 architecture 的 install 构建按预期拒绝 `native`，避免把构建机 GPU 冒充发布契约。
- shared/static CPU feature manifest 均为 `compute_ggml;compute_vulkan`；CUDA static 另含
  `compute_cuda`。三种安装前缀均携带 ggml、volk、Vulkan-Headers license closure，且生产配置
  不取得 glslang。ggml 通用包默认 native OFF；另一次干净 install configure 令
  `HYLE_SIMD_EXACT_TARGET=AVX2`，ggml 日志确认 `/arch:AVX2`，证明 exact HYLE/ggml ISA 同步生效；
  exact AVX2/AVX512 与本机 native 都是显式性能形状。

### preset/toolchain/CI 验证

- `CMakePresets.json` 升到与项目最低 CMake 3.25 一致的 schema 6；Windows/Linux/macOS 均有
  configure/build/test/workflow，`cmake --list-presets=all` 通过。Unix preset 不锁 Ninja；CPU 开发
  contract 显式关闭 TLS/CUDA，避免构建机偶然能力改变日常产物。
- 本机同时安装旧 VS 2022 与新 VS 18 时，第一次 generator-free smoke 真抓出 CMake 默认选择旧
  MSVC 19.38 的问题；最终 Windows 入口明确为受支持 VS 2022 generator，或由
  `scripts/hyle-windows.ps1` 经 vswhere 选择最新 C++ toolchain 后使用 Ninja，不再静默选旧编译器。
- `cmake/HyleCiPackageSmoke.cmake` 从空 scratch 制作 shared/static mem SDK，并运行同一个 pure-C
  error+mem consumer：shared 覆盖 `mem -> platform -> error` loader chain，static 覆盖 mimalloc
  link-only archive，且 `.c` 保持 C17、最终链接使用 toolchain C++ driver。脚本限制 `--parallel 2`；
  CI 在三平台复用该流程，并另跑 core C++/C/header/surface 聚合。
- Linux AArch64 reference toolchain 分离 host program 与 target library/include/package 搜索，并用
  static-library try-compile 禁止目标程序执行。WSL 从临时原生目录用 GCC 14.2 真 configure/build，
  `file` 确认产物为 ARM aarch64 ELF shared object；CI 对同一 preset 设硬门。

### 最终发布矩阵（`61f38ac`）

- Windows MSVC 19.50、WSL GCC 14.2、Apple M4 AppleClang 21 均从空 build tree 生成
  `HYLE_COMPONENTS=all`、shared、TLS/CUDA OFF 的确定性 CPU SDK；14 个组件完整 install。
- 每个平台都从安装前缀独立 configure/build/run 四个消费者：C++ simd+mem、纯 C error、纯 C
  io+HTTP/2、C++ compute，合计各 4/4。Windows 安装 13 个 HYLE DLL，compute consumer 只 staging
  自身闭包的 8 个；M4 识别 NEON，compute 至少枚举 host device。
- 第一次 WSL 真运行暴露 ELF executable RUNPATH 不会传递给间接 shared dependency；没有用
  `LD_LIBRARY_PATH` 掩盖，而是让每个安装的 HYLE shared target 自带 sibling-relative lookup。
  修复后 `readelf` 确认 simd/platform 为 `RUNPATH [$ORIGIN]`，`otool` 确认二者为
  `LC_RPATH @loader_path`；安装 CMake 文件三平台均无 source/build 绝对路径。
- 代表性 source embed 已由 Windows container sample 真运行；core 子集和 package smoke 由三平台
  CI 守门，AArch64 GCC 14 交叉 target 另有不可执行目标程序的硬门。完整 compute/io 矩阵保留为
  release check，不把每次提交 CI 变成全量依赖构建。

### 本轮四项代码量

| 固定口径 | code |
|---|---:|
| 全库（含 docs/in-tree third_party） | 274,902 |
| 自实现（含测试） | 134,163 |
| 自实现（不含测试） | 92,124 |
| build/package 实现 | 5,079 |

最后一项固定为 77 个自有 CMake 文件 + preset/CI + 8 个 build/fetch/setup/QEMU shell/PowerShell
guard；不含 vendored CMake、测试源码和文档。CMake 验收脚本分别守 package version、
production-shape、component contract 与 complete-source 交付，故相对旧 4,005 的增长属于产品构建
模块，不是测试源码误计。

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
