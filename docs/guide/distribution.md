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
调用 `hyle_stage_runtime_dependencies(target)`，即可 staging 传递 HYLE DLL。

## headers + static libraries

把上面的 `HYLE_STATIC` 改成 `ON`，其余消费方式相同：

```cmake
find_package(Hyle 0.1 REQUIRED COMPONENTS mem simd)
target_link_libraries(app PRIVATE hyle::mem hyle::simd)
```

不要从安装目录手写 `.lib`/`.a` 文件名。imported target 会传播 C++ linker runtime、系统库和 mimalloc、
nghttp2 等私有 link-only archive；绕开 target 就丢失静态闭包。

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

## 交付前检查

- static/shared SDK 必须用一个脱离源码树的 consumer configure/build/run；
- source zip 必须核对 SHA-256、`SOURCE-MANIFEST.txt`、根 `LICENSE` 和 `THIRD_PARTY_NOTICES.md`；
- 合同中明确允许的使用、修改、再分发、披露范围；不要用“给了源码/没给源码”代替授权条款。
