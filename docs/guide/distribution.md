# HYLE 三种交付与用户接入

HYLE 的 proprietary 授权与交付物形状是两回事。权利人可以向特定客户交付完整源码，也可以只交付
静态或动态 SDK；客户能做什么由单独书面授权决定，不由文件扩展名决定。

## 先看你拿到什么

| 用户拿到的交付物 | 推荐入口 | 用户是否编译 HYLE | 典型场景 |
|---|---|---:|---|
| headers + shared libraries | `find_package(Hyle)` | 否 | 多个程序共享 HYLE、缩短最终链接、允许替换同一 ABI 代际的 DLL/so/dylib |
| headers + static libraries | `find_package(Hyle)` | 否 | 单一可执行、部署时不携带 HYLE 动态库、受控静态闭包 |
| complete source | `add_subdirectory()` / FetchContent | 是 | 需要审查或修改源码、统一编译选项/LTO/目标 ISA，或者客户自行制作内部 SDK |

三种形状都只通过 `hyle::<component>` target 消费。不要手写 include path、`.lib`、`.a`、`.so` 或
DLL 列表；CMake target 才携带语言标准、模块闭包、系统库、第三方 link-only archive 和运行时 staging
规则。以下示例选择 `mem` 与 `simd`，依赖的 `error/platform/util_core` 会自动进入闭包；公开
core 头会随 SDK 安装，但未调用 `hyle_core` 符号时不会虚构一个 core binary component 依赖。
在 macOS 上，任何包含 platform 的闭包要求 deployment target ≥14.4；顶层构建未指定时 HYLE
默认该值，源码嵌入方若已有更低的 `CMAKE_OSX_DEPLOYMENT_TARGET` 会在 configure 阶段得到明确错误。

## 用户一：headers + shared libraries

客户工程只需要安装前缀，不需要 HYLE 源码：

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_app LANGUAGES C CXX)

find_package(Hyle 0.1 REQUIRED COMPONENTS mem simd)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE hyle::mem hyle::simd)
hyle_stage_runtime_dependencies(my_app)
```

配置时告诉 CMake SDK 在哪里：

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/hyle-sdk
cmake --build build --config Release
```

也可以只设置 `Hyle_DIR` 指向包含 `HyleConfig.cmake` 的目录，但优先使用可整体迁移的
`CMAKE_PREFIX_PATH`。Windows 没有 rpath，`hyle_stage_runtime_dependencies()` 会在构建后把该
executable 的传递 HYLE DLL 闭包复制到旁边；它在 Unix 和 static SDK 上是 no-op。Unix HYLE 动态库
自身使用 `$ORIGIN`/`@loader_path` 查找同目录传递依赖。
安装前缀可以整体移动：不要保留原构建目录或原 prefix 来“帮助”消费。HYLE 的 package gate 正是先
搬走完整 shared/static SDK，再从新位置编译并运行 focused consumers；如果发行方重排 `include/lib/bin`
以外的相对布局，必须重新生成 package，不能手改导出的绝对路径。

用户不再设置 `HYLE_STATIC=OFF`：shared/static 是交付前缀已经固定的属性，可用只读的
`Hyle_STATIC` 判断当前包形状。不要把两个 linkage、两个 ABI minor 或不同 ISA floor 的 SDK 混进同一
prefix。

## 用户二：headers + static libraries

消费端 CMake 与 shared SDK 相同：

```cmake
find_package(Hyle 0.1 REQUIRED COMPONENTS mem simd)
add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE hyle::mem hyle::simd)
hyle_stage_runtime_dependencies(my_app) # static 包上安全 no-op
```

区别只在交付方提供的是 static prefix，`Hyle_STATIC` 为 true。必须链接 imported target：例如 static
`hyle::mem` 会通过 target 传播 mimalloc archive 和匹配工具链的 C++ link driver；手写
`libhyle_mem.a`/`hyle_mem.lib` 会丢失这些闭包。纯 C 调用仍包含 `hyle/c/*.h` 并链接同一个 target，
无需把 HYLE C++ 实现复制进 C translation unit。

static 不等于跨工具链 ABI 通用。客户应用仍应使用交付合同指定的 compiler runtime、配置和目标 ISA；
需要跨这些边界时，选择 shared C ABI 或 complete source 重新构建。

## 用户三：complete source

先校验随包提供的摘要和 `SOURCE-MANIFEST.txt`，然后选择下面一种集成方式。源码授权不自动允许再分发。

### 直接加入父工程（默认推荐）

把源码解压到例如 `vendor/hyle`：

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_app LANGUAGES C CXX)

set(HYLE_COMPONENTS "mem;simd")
set(HYLE_BUILD_TESTS OFF)
set(HYLE_BUILD_BENCH OFF)
set(HYLE_STATIC ON) # OFF 则在父工程里构建 shared HYLE
add_subdirectory(vendor/hyle)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE hyle::mem hyle::simd)
hyle_stage_runtime_dependencies(my_app)
```

这些变量必须在 `add_subdirectory()` 之前设置。HYLE 只把所选组件及其依赖闭包加入父工程；作为
subproject 时不会创建源码分发 target，也不会接管父工程的安装或发布流程。

### 由 FetchContent 取得授权源码包

```cmake
include(FetchContent)
set(HYLE_COMPONENTS "mem;simd")
set(HYLE_BUILD_TESTS OFF)
set(HYLE_BUILD_BENCH OFF)
set(HYLE_STATIC ON)
FetchContent_Declare(hyle
  URL      https://approved.example/hyle-0.1.0-source.zip
  URL_HASH SHA256=<交付方提供的摘要>
  SOURCE_SUBDIR hyle-0.1.0-source)
FetchContent_MakeAvailable(hyle)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE hyle::mem hyle::simd)
```

私有下载地址、认证和 source mirror 由客户父工程管理；不要把访问凭据写进 HYLE CMake cache 或源码。

### 先从源码制作客户自己的 SDK

不想让每个应用重复编译 HYLE 时，客户可从收到的源码建立内部 prefix：

```bash
cmake -S hyle-source -B build-hyle \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYLE_COMPONENTS="mem;simd" \
  -DHYLE_BUILD_TESTS=OFF \
  -DHYLE_INSTALL=ON \
  -DHYLE_STATIC=ON \
  -DCMAKE_INSTALL_PREFIX=/approved/internal/hyle-sdk
cmake --build build-hyle --config Release
cmake --install build-hyle --config Release
```

之后所有应用按“用户一/二”的 `find_package()` 方式使用。CMake 没有另一套标准的“已安装源码树”消费
语义；完整源码保留工程目录结构直接构建，需要二进制 prefix 时再执行上述 install。

## 交付方如何制作三种交付物

`HYLE_COMPONENTS` 不写时默认构建 `all`；写了就必须至少有一个有效组件名，显式
`-DHYLE_COMPONENTS=` 会报错，而不会因脚本变量丢失偷偷变成全库。逗号、分号都可分隔多个根。
源码嵌入可在 `add_subdirectory(hyle)` 前用普通 CMake 变量设置它，HYLE 会让 cache/GUI 中的值与
实际解析值保持一致。每次 configure 日志都会同时打印 requested 与 resolved 闭包。

### 制作 headers + shared libraries

```bash
cmake -S . -B build-sdk-shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYLE_COMPONENTS="mem;simd" \
  -DHYLE_BUILD_TESTS=OFF \
  -DHYLE_INSTALL=ON \
  -DHYLE_STATIC=OFF \
  -DCMAKE_INSTALL_PREFIX=/path/to/hyle-sdk
cmake --build build-sdk-shared --config Release
cmake --install build-sdk-shared --config Release
```

交付安装前缀。它包含公开头、所选组件 DLL/shared objects、import libraries、CMake package、
feature/component manifest、HYLE notice 和实际编入实现的第三方许可证。Windows consumer 对 executable
调用 `hyle_stage_runtime_dependencies(target)`，即可 staging 传递 HYLE DLL。Unix 安装库用
`$ORIGIN`/`@loader_path` 解析同一 `lib/` 内的 HYLE 传递依赖；整体移动 prefix 不需要重写它们。
0.x 阶段每个 minor 是一个 ABI 代际：`find_package(Hyle 0.1 ...)` 不会误接受 0.2，Unix/Mach-O
动态库也以 `0.1` 作为 SONAME/install-name。ABI 尚未冻结不等于省略版本边界。

### 制作 headers + static libraries

把 shared recipe 的 `HYLE_STATIC` 改成 `ON`：

```bash
cmake -S . -B build-sdk-static \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYLE_COMPONENTS="mem;simd" \
  -DHYLE_BUILD_TESTS=OFF \
  -DHYLE_INSTALL=ON \
  -DHYLE_STATIC=ON \
  -DCMAKE_INSTALL_PREFIX=/path/to/hyle-sdk
cmake --build build-sdk-static --config Release
cmake --install build-sdk-static --config Release
```

不要从安装目录手写 `.lib`/`.a` 文件名。imported target 会传播 C++ linker runtime、系统库和 mimalloc、
nghttp2、ggml、volk 等私有 link-only archive；CUDA compute 包还会传播所需 Toolkit targets。绕开
target 就丢失静态闭包。

### compute SDK 的可复现能力契约

正式 compute 包不要使用环境探测式默认值，显式声明 CUDA 能力；CUDA 包还必须声明实际支持的
GPU architecture：

```bash
cmake -S . -B build-compute-sdk \
  -DCMAKE_BUILD_TYPE=Release -DHYLE_COMPONENTS=compute \
  -DHYLE_BUILD_TESTS=OFF -DHYLE_INSTALL=ON \
  -DHYLE_COMPUTE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='80;86;89' \
  -DCMAKE_INSTALL_PREFIX=/path/to/hyle-compute-sdk
```

CUDA-free 包写 `HYLE_COMPUTE_CUDA=OFF`；真正不含 GPU backend 的 CPU-only 包还要写
`HYLE_COMPUTE_VULKAN=OFF`。shared 包把 ggml/volk 静态实现吸收到
`hyle_compute`，不额外分发供应商 DLL；static 包通过 imported target 传播对应 archive、头与系统
依赖。两种包都安装准确的 ggml/volk/Vulkan-Headers license closure 和 shader helper，用户不需要
供应商源码 checkout。

workflow 对该重型形状只定义一条 Linux job，避免在三平台重复下载：从中央 lock 下载并按
archive SHA-256 缓存 ggml、nghttp2、Vulkan-Headers、volk、glslang、GoogleTest、Tracy，展开后再验证
无 `.git` 完整树。它先只构建并运行 HTTP/2、ggml bridge、Vulkan 三个真实 adapter tests 一次；然后关闭
tests/bench，以 `--parallel 2` 分别 build/install static/shared 产品形状。
源码无关 downstream 对两种形状都检查 feature manifest、许可证、真实 HTTP/2 SETTINGS 输出与 host
`GgmlBackend`；static 必须导出私有闭包 targets/headers，shared 则必须不泄漏它们。glslang 仍只属于
tests/bench，该 production gate 会拒绝其被取得或安装。最后另做小型 obs-only shared/static trace-on SDK，
由 installed consumer 编译并运行 C/C++ Tracy surface，验证 exact source、公开宏、target/header/license、
Threads/`dl` static closure 与运行时 staging；static consumer 是允许的单最终 executable topology，不是
多 DLL/plugin 复制 singleton。这样无需把重型全模块矩阵再复制一遍。
这段是持续验证合同，不是“当前已绿”声明；hosted run 与本地三平台证据必须在
[`pre-release-readiness`](../milestones/pre-release-readiness.md) 单独签署。

可迁移 compute 包默认 `HYLE_COMPUTE_GGML_NATIVE=OFF`；x86 ggml 使用 SSE4.2 安全基线。面向已知
机器且性能优先时，用 `HYLE_SIMD_EXACT_TARGET=AVX2`/`AVX512` 让 HYLE 与 ggml 使用相同 ISA floor，
或仅在同构本机源码构建显式打开 `HYLE_COMPUTE_GGML_NATIVE=ON`。后者产物不要发给更老 CPU。

### 制作 complete source

源码交付必须来自干净、已经审查并提交的 Git tree：

```bash
cmake -S . -B build-source-dist \
  -DHYLE_COMPONENTS=core \
  -DHYLE_BUILD_TESTS=OFF
cmake --build build-source-dist --target hyle_source_package
```

产物位于 `build-source-dist/dist/`：

- `hyle-<version>-source.zip`
- `hyle-<version>-source.zip.sha256`

zip 包含该 commit 的全部 tracked HYLE 源码、头、C ABI、构建文件、测试、benchmark、sample、文档、
脚本和 in-tree third_party。`SOURCE-MANIFEST.txt` 在归档时写入精确 commit。build tree、cache、日志、
本机 `.deps` 和其他 ignored/untracked 文件不会进入客户包；工作树不干净时 target 会直接失败。

客户解压后仍用普通组件入口构建。需要离线或自行审查 fetched dependency 时，把中央 lock 对应的
精确源码树放在解压根的 `deps/<lock-name>/`，再通过包内 preload 完全断网配置。例如只制作启用
HTTP/2 的 io 产品时，放入 `deps/nghttp2/` 后运行：

```bash
cmake -C hyle-<version>-source/hyle-offline-deps.cmake \
  -S hyle-<version>-source -B build \
  -DHYLE_COMPONENTS=io -DHYLE_BUILD_TESTS=OFF -DHYLE_BUILD_BENCH=OFF \
  -DHYLE_IO_TLS=OFF -DHYLE_IO_HTTP2=ON
```

ggml、volk、Vulkan-Headers、glslang、GoogleTest 等依赖使用同一
`deps/<lock-name>/` 形状；preload 验证后才设置标准
`FETCHCONTENT_SOURCE_DIR_<UPPERCASE_NAME>`。手工 direct override 仍会被中央 lock 复验，但它只注入一项
源码，不会自动补齐已选 component/feature 的完整断网闭包，也不会替你打开 disconnected 门：目录须为
精确 commit/tree 且无 tracked、untracked、ignored 项的 Git checkout，或 lock 中官方 archive 的逐目录/文件原始字节/安全相对链接
manifest 完全一致的无 `.git` 展开树。客户因此不必为了 source bundle 携带 Git metadata，同时任何
增删改、空目录或链接变化都会在显式 override 的 `MakeAvailable` 前失败；父工程若利用 FetchContent
“第一次声明胜出”预声明同名源码，wrapper 还会复验实际 population，错误身份不能成功配置/package。
父 CMake/toolchain 本身仍是可执行信任根。依赖源码交付及其许可证仍是独立 closure，不能把本机
下载缓存直接称为 HYLE 原创源码。若一次交付包含多项依赖，用源码树提供的
`hyle-offline-deps.cmake` 扫描 `deps/<lock-name>/` 并生成标准变量
（`cmake -C hyle-offline-deps.cmake ...`）；该文件不重复供应商清单，并默认完全断网。bundle 应同时
保存官方 archive SHA-256（传输证明）与 lock expanded-tree SHA-256（展开后内容证明）。不要要求客户把缓存塞进仓库
`.deps`，也不要复制
含特定编译器生成物/stamp 的 `<build>/_deps`。bundle manifest 应固定每项上游、pin、校验与 license。
若内部补丁使展开树不同，只能在保留独立签名/SBOM 证明时通过 preload 的显式 unsafe escape；该形状必须
是无 symlink 的普通目录树（安全相对链接只属于 digest 完全匹配的官方展开树），普通 direct override 不能
自我豁免。
OpenSSL/CUDA 等系统 SDK 仍由离线 toolchain 或 package prefix 提供，不属于 FetchContent 源码目录。
如何从联网准备机制作三平台工具链/依赖介质并在干净断网机器验收，见
[`setup/offline.md`](../setup/offline.md)。

## 非 CMake 工程

C ABI 不要求应用源码使用 CMake，但 SDK 的传递依赖元数据以 `HyleConfig.cmake` 为准。优先使用
构建系统已有的 CMake-package import；没有该能力时，用一个很薄的 CMake bridge 链接所需
`hyle::<module>`，再把生成的本地 target/最终制品接入主工程。不要从 static SDK 手抄一串
`.a`/`.lib`：这会漏掉 C++ linker runtime、mimalloc/nghttp2/ggml/volk 以及可选 CUDA/OpenSSL
闭包。0.1 暂不提供只对简单 shared leaf 正确、对 static/feature 包不完整的 pkg-config 文件。

## 交付前检查

- static/shared SDK 必须用一个脱离源码树的 consumer configure/build/run；
- 模块的 focused 参考 consumer 位于 `samples/package/<module>/`；它们验证具体 package 合同，
  不单独证明模块已严格收口。例如 `error/c` 是真正从
  `project(... LANGUAGES C)` 开始的 C ABI 工程，`error/cpp` 覆盖 `Status`/`Result<T>`；
- source zip 必须核对 SHA-256、`SOURCE-MANIFEST.txt`、根 `LICENSE` 和 `THIRD_PARTY_NOTICES.md`；
- 合同中明确允许的使用、修改、再分发、披露范围；不要用“给了源码/没给源码”代替授权条款。
