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
一个版本化构建合同。早期公开 C schema/bridge 已移除，不能从 shared artifact 的存在推断 C 或
managed 调用入口。

## 1. 完整源码

父工程在 `add_subdirectory` 前选择产品根：

```cmake
set(HYLE_COMPONENTS "mem;container")
set(HYLE_BUILD_TESTS OFF)
set(HYLE_BUILD_BENCH OFF)
add_subdirectory(path/to/hyle hyle-build)

target_link_libraries(my_app PRIVATE hyle::container)
```

`HYLE_COMPONENTS` 是分号或逗号分隔的产品根。HYLE 作为顶层工程时，省略或设为 `all` 会构建
完整产品；通过 `add_subdirectory()` 嵌入时父工程必须显式设置，防止一次依赖意外构建整库。
显式空值会报错。用户只链接实际使用的 `hyle::<product>`。为组织内部依赖而存在的
`_factor_*` detail target 由产品 façade 聚合，不是产品名，也不能传给 `HYLE_COMPONENTS`。
源码、static SDK 和 shared SDK 使用同一套产品名。

HYLE 根工程只要求 C++23 编译器；若所选闭包包含 mem，其内部 mimalloc 实现会按需启用 C 编译器。
这不建立 C 产品 surface，也不要求 core/error 等纯 C++ 闭包安装 C 工具链。

源码接收者需要按 [开发环境](../developer/setup/README.md)准备工具链和依赖。没有网络的环境见
[离线依赖指南](../developer/setup/offline.md)。不要把本机构建缓存、FetchContent working tree 或
临时 build 目录当作交付物。联网机器首次 configure 会自动把锁定源码放进跨 build 的
`HYLE_DEP_SOURCE_CACHE`；依赖 binary/subbuild 仍属于当前 build。需要离线交付时搬运由 HYLE 管理工具
生成并带 marker 的完整 source cache，或使用指南规定的逐项受审源码 bundle，不能复制某个 build 的
`_deps` 充当源码介质。

### 制作授权源码包

在干净 Git checkout 中配置任意顶层 build，然后：

```bash
cmake --build <build-dir> --target hyle_source_package
```

产物位于 `<build-dir>/dist/hyle-<version>-source.zip`，旁边有 SHA-256。target 只打包 Git tracked
tree，并先审计依赖版本；它不会把 `.git`、build cache 或未提交文件带给客户。制作方仍须按授权
核对第三方许可证与接收者范围。

接收者可以在解包目录设置 `HYLE_BUILD_TESTS=ON`，运行所选组件的产品与模块测试。因为归档刻意
不含 `.git`，只服务于维护者 Git revision/LOC ownership 的自检不会注册；这不影响 header、模块、
文档和 closure 合同。若需要运行该项维护自检，应使用真正的 Git checkout，而不是给归档伪造
repository metadata。

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
依赖信息。安装只包含所选组件闭包拥有的公开头；trace foundation header 也只有闭包存在 trace
consumer 时才安装，不会复制整棵 HYLE 公开头树。消费者不应手工枚举 `.a`/`.lib`、猜依赖顺序
或从未安装组件借头。

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
策略，Windows consumer 可使用 package 提供的 `hyle_stage_runtime()` 把 imported DLL
闭包复制到 executable 旁。该 opt-in 在每次构建 consumer 时检查并只更新不同文件，即使 EXE 无需
重链接也会更新 DLL；静态 HYLE 用户链接自己的 shared kernel 时同样适用，空 DLL 闭包也合法。
shared factor 默认只导出显式标注的 C++ API；内部 C、汇编与第三方符号
不会因此成为受支持入口。安装验证会检查实际动态库的导出表，但这既不建立 C ABI，也不承诺跨
工具链 C++ ABI。它与 static SDK 使用同一 selected-closure header 规则。示例见
[`samples/04_prebuilt_lib`](../../samples/04_prebuilt_lib/README.md)。

## 4. 消费安装包

static 和 shared consumer 的 CMake 写法相同：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem container)
target_link_libraries(my_app PRIVATE hyle::container)

if(WIN32)
  hyle_stage_runtime(my_app)
endif()
```

配置时用 `CMAKE_PREFIX_PATH` 或 `Hyle_DIR` 指向相应安装 prefix。不要把 static 与 shared SDK 安装
到同一个 prefix 后依赖偶然搜索顺序；也不要从一个 SDK 拿头、从另一个拿库。

Visual Studio 生成器下，在创建 consumer target 的同一 CMake 目录调用
`hyle_stage_runtime()`；其链接的库可以来自其它目录。

Linux 静态 SDK 若要链入 shared library，生产 SDK 时设置
`-DCMAKE_POSITION_INDEPENDENT_CODE=ON`。普通静态 EXE 与这种 PIC SDK 是明确的 producer 配置；
在 consumer 上设置 PIC 不会重编译已经安装的非 PIC archive。选择源码嵌入且 `HYLE_STATIC=ON`
时，同样在 `add_subdirectory()` 前设置该 CMake 选项。

`find_package` 会拒绝不存在或未打包的产品。`Hyle_AVAILABLE_COMPONENTS` 沿用 CMake 命名，但只列
产品 façade；包内 `hyle::_factor_*` 仅用于 imported target 的传递闭包，不是用户 API。imported target 负责公开头、C++23 floor、
传递组件和必要第三方依赖；若用户必须手工补 link/include 参数，应视为 package bug。

SIMD 有一个显式专家开发层，三种交付写法相同。普通用户仍只链接 `hyle::simd`；需要 HYLE ordinary
API 尚未覆盖的 Highway/xsimd header-only 能力，或要生成固定/多 ISA 自定义 kernel 时，调用安装包自动
加载的 helper：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS simd)
add_library(my_kernel OBJECT my_kernel.cc)
hyle_target_simd(TARGET my_kernel ISA AVX2)
target_link_libraries(my_app PRIVATE hyle::simd)
target_sources(my_app PRIVATE $<TARGET_OBJECTS:my_kernel>)
```

helper 会让该 target 显式消费 `hyle::simd_expert`；SDK 自带锁定 Highway/xsimd 开发头，不在 consumer
configure 时下载依赖。普通 `hyle::simd` target 不传播这些 include/definition。expert 头类型只留在
target-bound TU，跨静态/动态边界仍传 pointer/count/POD；它不是额外 binary ABI，也不承诺需要上游
`.cc` 的 compiled contrib。`hyle_target_simd` 会把 ISA flags 施加到整个 CMake target，所以通常应像示例
一样为 kernel 建立专用 OBJECT target；不要误把一个多源应用整体编成 AVX2。若只需锁定 vendor headers、
不需要 helper 增加 ISA flags，可直接让该 OBJECT target 链接 `hyle::simd_expert`。完整 direct/fixed/fat
选择见 [SIMD 使用指南](simd-usage.md)。

预发布 `0.y.z` 包只接受完整精确版本；`find_package(Hyle 0.y.z EXACT ...)` 中 patch 也属于构建
合同，shared loader identity 同样使用完整版本。1.0 后才按 major compatibility 演进。

当前已验收产品中的 IO TLS、HTTP/2 和 trace 等昂贵能力默认关闭，并且只有 owner 组件/gate
成立时才出现对应 cache 项。

Compute/Graph 使用唯一 `HYLE_COMPUTE_ROUTES` 集合，省略时默认完整 CPU。普通用户仍链接
`hyle::compute/graph`；未选路线在取依赖闭包前排除，不影响 GPU 所需 host 控制和传输。
统一配置正在按 owner 卡执行三交付验收，当前状态见 registry。配置示例：

| 交付能力 | `HYLE_COMPUTE_ROUTES` |
|---|---|
| 完整 CPU | cpu |
| CUDA / Apple MLX-Metal / Vulkan | cuda / apple / vulkan |
| CPU 与一个 GPU 的 SDK | cpu;cuda / cpu;apple / cpu;vulkan |

这些值由源码父工程在 `add_subdirectory()` 前设置，或由 SDK producer 的 CMake 配置选择；安装消费者
读取 `Hyle_AVAILABLE_FEATURES` 中的 `compute_cpu/compute_cuda/compute_onednn`，不能在自己工程里
用宏创建未交付路线。选中 target 传播相应 `HYLE_COMPUTE_*` 存在性宏；它们不改变公共对象布局。
未选 native/provider 的头、空实现 target、SDK 查找与运行库不进入安装包。
路线集合顺序不是 preference；空、未知、AUTO 和不支持的平台组合明确拒绝。
旧六个 CPU/CUDA/Apple/Vulkan/oneDNN/Torch CPU 开关要求迁移和 fresh build，不保留双轨。
presence 宏由 target 传播，不能在 consumer 手写宏创建未交付能力。
mixed SDK 只证明编入多路线，跨路线联合图仍须 H1–H3 另行验收。

安装包提供 `Hyle_AVAILABLE_COMPUTE_ROUTES` 与 `Hyle_COMPUTE_RECIPE_ID`；source 和安装 target
都可通过 `get_target_property(... hyle::compute HYLE_COMPUTE_ROUTES)` / `HYLE_COMPUTE_RECIPE_ID`
查询实际配置。fingerprint 随版本/flags/linkage 等 recipe 变化，用于冷诊断，不是 ABI 或性能证书。
安装 consumer 可在 `find_package` 前设置 `HYLE_COMPUTE_ROUTES` 断言所需子集；缺路线报错，
不会由此修改 SDK、下载 Provider 或重编。源码/安装应用统一调用 `hyle_stage_runtime(app)`。

CPU 路线在支持的 x86-64 目标自动接入锁定 oneDNN；macOS ARM64 使用原 ggml/ATen CPU 程序。
必要 Provider 不提供普通 ON/OFF；缺少合格必要依赖时明确失败。
普通 Direct 使用 `hyle::compute::cpu::create_context()`，普通 Graph 使用
`hyle::graph::cpu::create_context()`；用户不需要排列 Provider。专家仍可显式限定 Provider。
static SDK 传递锁定 oneDNN archive/header 闭包，shared SDK 把实现私有链接进 provider library，
二者安装 Apache-2.0 notice。不选择 CPU 不探测/取得其数学 Provider。
构建 feature 只记录可用实现；默认规则的性能验收由 CPU 路线任务记录。

标准 CPU 配置还自动取得锁定原 LibTorch CPU SDK：
Windows/Linux x86_64为`2.8.0+cpu`，macOS ARM64为`2.8.0`。source/static/shared复用同一锁与本机cache，
安装消费者由依赖shim解析SDK，不要求用户列第三方target。HYLE static仍需要外部shared CPU runtime；
二进制HYLE prefix不包含整个LibTorch开发包。离线迁移需同时带已验证的本机依赖cache，
或显式`HYLE_COMPUTE_TORCH_CPU_ROOT`指向同版本、CPU-only、可搬移的外部SDK。
Mac受管stage修正原CPU dylib的libomp加载边并保留原archive身份，不修改用户外部SDK；
用户声明不匹配版本/目标或非可搬移SDK时configure明确失败。当前Mac资格与未完成项见
[CPU F3证据](../../evidence/correctness/cpu-final-f3/README.md)。

Compute/Graph 的 CUDA 路线由 `HYLE_COMPUTE_ROUTES=cuda` 选择，
普通 `compute-cuda` configure/build preset 选择 CUDA-only Compute+Graph、Release、tests/bench OFF，
build 并行度 2；`"-DHYLE_COMPUTE_ROUTES=cpu;cuda"` 可显式取得 mixed SDK，三交付仍用既有 install/static 参数。
选中时要求用户预装 CUDA Toolkit 13.3+，缺失失败；未选完全不探测。
CUDA Toolkit 是机器 SDK，不进入 FetchContent cache，也不会由 HYLE 下载。启用后的 source/static/shared
都传播 `CUDAToolkit` package requirement；安装包消费者同样必须能被 `find_package(CUDAToolkit 13.3)`
找到。选择公开产品会接入 ggml CUDA、cuBLASLt、SGLang、DeepSeek 与 CCCL 等实现；Direct 使用
`compute::cuda::create_context()`，Graph 使用 `graph::cuda::create_context/compile/prepare_replay()`。
ggml 与 SGLang activation 的许可证/NOTICE 随选中路线安装；原始系统 CUDA SDK 仍由机器提供。
当前三交付/性能验收状态见 module-status。Windows 消费者在链接公开产品后调用
`hyle_stage_runtime(my_app)`，会把所需 HYLE、CUDA Runtime、cuBLAS 与 cuBLASLt
DLL 放在 exe 同目录；源码、静态 SDK 和动态 SDK 使用同一入口。构建时仍需安装 CUDA Toolkit，
运行机器需具备相应 NVIDIA 驱动与 VC runtime。
Apple MLX/Metal 的构建、metallib 邻置与已验收边界见 [Apple 指南](apple-compute.md)。
Vulkan 的 ON/OFF、系统 loader、嵌入 SPIR-V 与当前交付验收见 [Vulkan 指南](vulkan-compute.md)；
开启 Vulkan 不取得 CPU/CUDA/MLX 实现，二进制 SDK 不要求 glslang。各路线的验收状态以 registry
和当前卡为准。HIP/ROCm 当前无限期不纳入产品，不提供占位组件或配置项。
当前 trace 只有 Tracy，不存在 `HYLE_TRACE_BACKEND` 选择项。

## 5. 配置合同

交付方应随 artifact 记录：

- HYLE commit/version、目标平台与架构；
- compiler、stdlib/runtime、Debug/Release、static/shared；
- 选择的组件与解析闭包；
- baseline/native/exact ISA、HTTP/TLS/trace，以及实际存在的 compute backend/provider 等已选择 feature；
- 依赖 lock/来源、许可证清单与 package hash；
- 已执行的 source/static/shared consumer smoke。

不要依赖 producer build tree 中的绝对路径。安装包必须可搬迁，且离开源码 checkout 后仍能被一个
独立最小 C++ 工程 `find_package`、编译、链接和运行。

### 非 CMake 工程与发布归档

HYLE 当前不发布 pkg-config、Bazel 或 Meson 的第二份依赖图。非 CMake 主工程可以用一个很薄的 CMake
bridge 生成/导入 HYLE target，再把最终 native artifact 接入自己的构建；不要手抄 static archive
顺序、CUDA/system library 或 compile definition。出现具名非 CMake 消费者后，才会以该消费者的
static/shared 完整闭包为门增加原生入口，而不是先生成只能链接简单组件的半接口。

`cmake --install` 产生的 relocatable prefix 就是受检 binary SDK 内容。zip/tar、代码签名、SBOM 和
artifact repository 属发行流水线：对这个 prefix 归档并随附上面的配置合同即可，不需要在 HYLE 中
复制一个供应商或操作系统专用打包开关。解包后仍应运行一次独立 `find_package` consumer smoke；只把
若干 `.lib/.a/.dll/.so` 拿走不构成完整 SDK。

## 6. 如何选择

- 需要修改底层 kernel、审计全部实现、做全程序 LTO/PGO 或适配特殊平台：完整源码。
- 需要单一 native deployment、接受每次更新重新链接：static SDK。
- 需要多个同工具链 C++ 程序共享库或独立替换 artifact：shared SDK。
- 需要 Java/.NET/Python/Rust：当前没有产品 FFI；不要把 shared SDK 的 C++ symbols 当成稳定边界。

项目尚未发布。模块在新标准下达到 `CLOSED` 前，artifact 可以用于内部试验，但不应对外宣称该
模块拥有稳定 API/ABI 或完整性能证书。
