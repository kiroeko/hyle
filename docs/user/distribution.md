# HYLE 交付与集成

HYLE 支持三种 C++ 产品形状。技术形状与商业许可分开：闭源项目也可以向特定客户交付完整源码；
是否允许接收、修改或再分发由合同和根级法律文本决定，不由 CMake option 决定。

| 交付 | 收到什么 | 适合场景 |
|---|---|---|
| 完整源码 | 公开/私有实现、CMake、已授权依赖描述 | 审计、深度定制、全程序优化、特殊平台 |
| headers + static libraries | 公开头、静态库、CMake package metadata | 单一部署 artifact、固定工具链/配置 |
| headers + shared libraries | 公开头、动态库、CMake package metadata | 多个 C++ 程序共享 artifact、独立部署库 |

三种形状都面向 C++23。它们不是 Native FFI，也不承诺不同编译器、标准库、runtime、build mode
之间的 C++ ABI 兼容。预发布阶段应把 HYLE、编译器/stdlib、linkage、ISA/backend 和关键依赖视为
一个版本化构建合同。

## 1. 完整源码

父工程在 `add_subdirectory` 前选择产品根：

```cmake
set(HYLE_COMPONENTS "mem;container" CACHE STRING "" FORCE)
set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/hyle hyle-build)

target_link_libraries(my_app PRIVATE hyle::container)
```

`HYLE_COMPONENTS` 是分号或逗号分隔的根组件；省略或设为 `all` 会构建完整产品，显式空值会报错。
依赖闭包由中央组件图解析，但用户 target 仍只链接实际使用的 `hyle::<module>`。

源码接收者需要按 [开发环境](../developer/setup/README.md)准备工具链和依赖。没有网络的环境见
[离线依赖指南](../developer/setup/offline.md)。不要把本机构建缓存、FetchContent working tree 或
临时 build 目录当作交付物。

### 制作授权源码包

在干净 Git checkout 中配置任意顶层 build，然后：

```bash
cmake --build <build-dir> --target hyle_source_package
```

产物位于 `<build-dir>/dist/hyle-<version>-source.zip`，旁边有 SHA-256。target 只打包 Git tracked
tree，并先审计依赖版本；它不会把 `.git`、build cache 或未提交文件带给客户。制作方仍须按授权
核对第三方许可证与接收者范围。

## 2. 安装 headers + static libraries

```bash
cmake -S . -B build-sdk-static \
  -DHYLE_COMPONENTS="mem;container" \
  -DHYLE_BUILD_TESTS=OFF \
  -DHYLE_INSTALL=ON \
  -DHYLE_STATIC=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/hyle-static
cmake --build build-sdk-static --config Release
cmake --install build-sdk-static --config Release
```

static package 会通过 imported targets 传播必要的库顺序、公开 compile definitions 和可重定位的
依赖信息。消费者不应手工枚举 `.a`/`.lib` 或猜依赖顺序。

## 3. 安装 headers + shared libraries

```bash
cmake -S . -B build-sdk-shared \
  -DHYLE_COMPONENTS="mem;container" \
  -DHYLE_BUILD_TESTS=OFF \
  -DHYLE_INSTALL=ON \
  -DHYLE_STATIC=OFF \
  -DCMAKE_INSTALL_PREFIX=/opt/hyle-shared
cmake --build build-sdk-shared --config Release
cmake --install build-sdk-shared --config Release
```

shared package 仍是 C++ SDK。运行时必须能找到动态库闭包；Linux/macOS 使用正常的 rpath/loader
策略，Windows consumer 可使用 package 提供的 `hyle_stage_runtime_dependencies()` 把 imported DLL
闭包复制到 executable 旁。示例见 [`samples/04_prebuilt_lib`](../../samples/04_prebuilt_lib/README.md)。

## 4. 消费安装包

static 和 shared consumer 的 CMake 写法相同：

```cmake
find_package(Hyle 0.1 REQUIRED COMPONENTS mem container)
target_link_libraries(my_app PRIVATE hyle::container)

if(WIN32)
  hyle_stage_runtime_dependencies(my_app)
endif()
```

配置时用 `CMAKE_PREFIX_PATH` 或 `Hyle_DIR` 指向相应安装 prefix。不要把 static 与 shared SDK 安装
到同一个 prefix 后依赖偶然搜索顺序；也不要从一个 SDK 拿头、从另一个拿库。

`find_package` 会拒绝不存在或未打包的 component。imported target 负责公开头、C++23 floor、
传递组件和必要第三方依赖；若用户必须手工补 link/include 参数，应视为 package bug。

## 5. 配置合同

交付方应随 artifact 记录：

- HYLE commit/version、目标平台与架构；
- compiler、stdlib/runtime、Debug/Release、static/shared；
- 选择的组件与解析闭包；
- baseline/native/exact ISA 和 CUDA/Vulkan/ggml/HTTP/TLS 等可选 backend；
- 依赖 lock/来源、许可证清单与 package hash；
- 已执行的 source/static/shared consumer smoke。

不要依赖 producer build tree 中的绝对路径。安装包必须可搬迁，且离开源码 checkout 后仍能被一个
独立最小 C++ 工程 `find_package`、编译、链接和运行。

## 6. 如何选择

- 需要修改底层 kernel、审计全部实现、做全程序 LTO/PGO 或适配特殊平台：完整源码。
- 需要单一 native deployment、接受每次更新重新链接：static SDK。
- 需要多个同工具链 C++ 程序共享库或独立替换 artifact：shared SDK。
- 需要 Java/.NET/Python/Rust：当前没有产品 FFI；不要把 shared SDK 的 C++ symbols 当成稳定边界。

项目尚未发布。模块在新标准下达到 `CLOSED` 前，artifact 可以用于内部试验，但不应对外宣称该
模块拥有稳定 API/ABI 或完整性能证书。
