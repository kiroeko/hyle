# HYLE 交付形状

HYLE 的 proprietary 授权与交付物形状是两回事。权利人可以向特定客户交付完整源码，也可以只交付
静态或动态 SDK；客户能做什么由单独书面授权决定，不由文件扩展名决定。

## headers + shared libraries

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

## headers + static libraries

把上面的 `HYLE_STATIC` 改成 `ON`，其余消费方式相同：

```cmake
find_package(Hyle 0.1 REQUIRED COMPONENTS mem simd)
target_link_libraries(app PRIVATE hyle::mem hyle::simd)
```

不要从安装目录手写 `.lib`/`.a` 文件名。imported target 会传播 C++ linker runtime、系统库和 mimalloc、
nghttp2、ggml、volk 等私有 link-only archive；CUDA compute 包还会传播所需 Toolkit targets。绕开
target 就丢失静态闭包。

## compute SDK 的可复现能力契约

正式 compute 包不要使用环境探测式默认值，显式声明 CUDA 能力；CUDA 包还必须声明实际支持的
GPU architecture：

```bash
cmake -S . -B build-compute-sdk \
  -DCMAKE_BUILD_TYPE=Release -DHYLE_COMPONENTS=compute \
  -DHYLE_BUILD_TESTS=OFF -DHYLE_INSTALL=ON \
  -DHYLE_COMPUTE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='80;86;89' \
  -DCMAKE_INSTALL_PREFIX=/path/to/hyle-compute-sdk
```

CPU-only 包写 `HYLE_COMPUTE_CUDA=OFF`。shared 包把 ggml/volk 静态实现吸收到
`hyle_compute`，不额外分发供应商 DLL；static 包通过 imported target 传播对应 archive、头与系统
依赖。两种包都安装准确的 ggml/volk/Vulkan-Headers license closure 和 shader helper，用户不需要
供应商源码 checkout。

可迁移 compute 包默认 `HYLE_COMPUTE_GGML_NATIVE=OFF`；x86 ggml 使用 SSE4.2 安全基线。面向已知
机器且性能优先时，用 `HYLE_SIMD_EXACT_TARGET=AVX2`/`AVX512` 让 HYLE 与 ggml 使用相同 ISA floor，
或仅在同构本机源码构建显式打开 `HYLE_COMPUTE_GGML_NATIVE=ON`。后者产物不要发给更老 CPU。

## complete source

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

客户解压后仍用普通组件入口构建。需要离线或自行审查 fetched dependency 时，供应依赖源码目录：

```bash
cmake -S hyle-<version>-source -B build \
  -DHYLE_COMPONENTS=io \
  -DFETCHCONTENT_SOURCE_DIR_NGHTTP2=/approved/vendor/nghttp2
```

ggml、volk、Vulkan-Headers、glslang、GoogleTest 等依赖使用同一
`FETCHCONTENT_SOURCE_DIR_<UPPERCASE_NAME>` 机制。依赖源码交付及其许可证仍是独立 closure，不能把本机
下载缓存直接称为 HYLE 原创源码。

## 非 CMake 工程

C ABI 不要求应用源码使用 CMake，但 SDK 的传递依赖元数据以 `HyleConfig.cmake` 为准。优先使用
构建系统已有的 CMake-package import；没有该能力时，用一个很薄的 CMake bridge 链接所需
`hyle::<module>`，再把生成的本地 target/最终制品接入主工程。不要从 static SDK 手抄一串
`.a`/`.lib`：这会漏掉 C++ linker runtime、mimalloc/nghttp2/ggml/volk 以及可选 CUDA/OpenSSL
闭包。0.1 暂不提供只对简单 shared leaf 正确、对 static/feature 包不完整的 pkg-config 文件。

## 交付前检查

- static/shared SDK 必须用一个脱离源码树的 consumer configure/build/run；
- source zip 必须核对 SHA-256、`SOURCE-MANIFEST.txt`、根 `LICENSE` 和 `THIRD_PARTY_NOTICES.md`；
- 合同中明确允许的使用、修改、再分发、披露范围；不要用“给了源码/没给源码”代替授权条款。
