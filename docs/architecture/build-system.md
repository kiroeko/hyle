# 构建与组装架构

## 定位

HYLE 的 CMake target 图是项目组织架构的可执行版本，不只是“把当前仓库编过”的脚本。
源码、公开头、C ABI、直接依赖、可选后端和验证面都必须归属到 target；文档解释怎样选择，
不能用目录顺序、全局变量或维护者记忆补全隐藏依赖。

## 产品组件

每个二进制模块有 build-tree target `hyle_<module>` 和消费别名 `hyle::<module>`。用户通过
`HYLE_COMPONENTS` 给出所需的产品根，CMake 计算依赖闭包，只把闭包内模块加入构建：

```text
roots: core error obs math platform time mem simd container util sched io compute graph

platform -> error
time     -> error
mem      -> error + platform
simd     -> platform
container-> mem + platform
util     -> container
sched    -> error + platform + time + mem + container
io       -> error + mem + container + sched
compute  -> error + platform + mem + simd + sched
graph    -> error + mem + container + sched
```

箭头右侧是 target 依赖闭包，不重复 core/util-core 纯头和 `hyle_trace` 横切 gate。`all` 是完整
产品默认值；逗号与分号都可分隔多个根。未知组件在 configure 阶段直接失败，配置日志打印
requested/resolved 清单，因此一次构建“组装了什么”可审计。

组件选择同时控制依赖发现：不选 `io` 就不探测 OpenSSL、不取得 nghttp2；不选 `compute`
就不探测 CUDA，也不取得 ggml/Vulkan-Headers/volk/glslang。第三方默认写入当前 HYLE build
tree 的 `_deps`；需要跨 build tree 共享下载缓存时显式设置 `HYLE_DEPS_DIR`，生成物不能写入
或依赖源码树。

## 四条消费路径

1. **仓库顶层构建**：默认 `all`，产物集中到 `<build>/bin`，tests 默认 ON、bench 默认 OFF。
2. **源码嵌入**：父工程在 `add_subdirectory(hyle)` 前设置 `HYLE_COMPONENTS`；tests 在 subproject
   默认 OFF，HYLE 不改父工程的 `BUILD_SHARED_LIBS`、configuration types 或全局输出目录。
3. **安装包消费**：目标形状是 `find_package(Hyle COMPONENTS ...)` 后继续链接同名
   `hyle::<module>`。当前 install/export 尚未落地，属于本轮 build/package 收口的下一阶段。
4. **直接预构建文件**：当前仅是诊断/ABI smoke 路径；缺少传递依赖建模，不是推荐发布面，
   将由安装包路径取代。

Windows 源码嵌入时没有 rpath。consumer executable 链接共享 HYLE 后调用
`hyle_stage_runtime_dependencies(target)`，会把其传递 DLL 放到 executable 同目录；这比让 HYLE
篡改父工程输出目录更局部、也更容易迁移。Unix build tree 继续使用生成器的 rpath。

## 静态、共享与 ABI

`HYLE_STATIC=OFF` 默认让每个已选 HYLE 模块成为 SHARED target；ON 时显式成为 STATIC target。
实现不改父工程 `BUILD_SHARED_LIBS`。`hyle_add_module` 同时传播 C++23 floor、公开 include、直接
target 依赖、trace/testing ODR gate 和正确的 `HYLE_<MODULE>_BUILD`/`HYLE_STATIC` 可见性宏。

C surface gate 在共享构建检查动态导出表，在静态构建检查 archive 中定义的全局 C 符号；
二者都还要对齐 reviewed manifest、声明和 pure-C 引用，不能用“静态库没有 exports”跳过 ABI
完整性验证。

## 验证产物归属

`hyle_add_test(module ...)` 和 pure-C helper 只在对应产品 target 存在时创建测试。每个已选模块
有 `hyle_<module>_tests` aggregate，统一 `hyle_tests` 依赖全部已解析组件的 aggregate。因此：

- 构建 `hyle_core_tests` 会同时产出 core C++/C binary，不再留下 CTest 指向旧或缺失 binary；
- 构建 `hyle_tests` 后运行 CTest，登记项与当前组件闭包一致；
- 切换同一 build tree 的 `HYLE_COMPONENTS` 会重新生成 target/test 集，而不是沿用旧模块。

benchmark 使用同一套组织边界：只有已解析组件的 benchmark 会进入 target 图；
`hyle_<module>_benches` 构建一个模块的完整 benchmark 集，`hyle_benches` 构建所有已选组件的集合。
Highway/xsimd checkout 只在 `simd` 已选时检查，因此打开 benchmark 不会让 core、mem、container
等组装意外依赖 SIMD 对手。benchmark executable 的模块归属显式登记在 CMake 中，不能靠文件名
或开发者记忆推断。

install/export、cross compile 和三平台完整迁移矩阵尚未取得最终证书，状态只见 build/package
milestone。

## 维护规则

- 新模块必须同时登记 component、直接依赖和 module target；三处不一致由配置测试拦截。
- 新可选第三方只能在其消费组件已选且 feature 已启用时发现/取得。
- 不在 CMake 中写开发机绝对路径；preset 是可选入口包，不是编译成功的隐藏前置。
- parent-owned 的 generator、toolchain、配置、输出目录和 `BUILD_SHARED_LIBS` 不得被 subproject 改写。
- source tree 只存源码与审查过的 vendored 资产；cache、下载、生成头和制品一律进 build tree。
