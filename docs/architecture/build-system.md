# 构建与组装架构

## 定位

HYLE 的 CMake target 图是项目组织架构的可执行版本，不只是“把当前仓库编过”的脚本。
源码、公开头、C ABI、直接依赖、可选后端和验证面都必须归属到 target；文档解释怎样选择，
不能用目录顺序、全局变量或维护者记忆补全隐藏依赖。

授权方式与交付形状正交。HYLE 当前是 proprietary，不代表客户只能拿二进制；经单独书面授权的
客户可以取得完整源码。构建系统把交付形状明确分为三类：

| 交付形状 | 构建入口 | 内容与用途 |
|---|---|---|
| headers + shared libraries | `HYLE_INSTALL=ON; HYLE_STATIC=OFF` | 动态 SDK；安装 target、运行库、头、manifest 和 license closure |
| headers + static libraries | `HYLE_INSTALL=ON; HYLE_STATIC=ON` | 静态 SDK；额外传播 link-only 私有 archive/系统依赖 |
| complete source | `hyle_source_package` | 干净 Git commit 的完整 tracked tree；客户用同一 component/feature 图自行组装 |

三者改变的是 artifact closure，不改变许可证。源码包不能从工作目录盲拷贝：target 遇到 tracked
修改或未跟踪文件会拒绝，产物带 export-substituted commit manifest 与 SHA-256。

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
tree 的 `_deps`，源码与生成物一起归该 build tree 所有；不同 toolchain/build tree 不共享这个
目录。离线或经审查的依赖源码通过标准 `FETCHCONTENT_SOURCE_DIR_<NAME>` 注入，生成物仍留在本
build tree，不能写入或依赖源码树。

TLS 与 CUDA 使用 `AUTO|ON|OFF` 三态：AUTO 仅供便利探测，ON 是硬能力契约，OFF 不运行依赖探测。
正式 SDK/CI 必须显式选择 ON/OFF，因此构建机偶然安装了某个 SDK 不会改变发布形状。

## 四条消费路径

1. **仓库顶层构建**：默认 `all`，产物集中到 `<build>/bin`，tests 默认 ON、bench 默认 OFF。
2. **源码嵌入**：父工程在 `add_subdirectory(hyle)` 前设置 `HYLE_COMPONENTS`；tests 在 subproject
   默认 OFF，HYLE 不改父工程的 `BUILD_SHARED_LIBS`、configuration types 或全局输出目录。
3. **安装包消费**：生产构建执行 `cmake --install`，下游用
   `find_package(Hyle COMPONENTS ...)` 后链接同名 `hyle::<module>`；config 会拒绝安装中不存在的
   component，并传播 C++23、static/shared、模块依赖和内部静态后端。`Hyle_AVAILABLE_COMPONENTS`
   是本次组装的产品清单，`Hyle_AVAILABLE_FEATURES` 是 trace/TLS/CUDA/Vulkan 等能力清单。
4. **直接预构建文件**：仅是诊断/ABI smoke 路径；缺少传递依赖建模，不是推荐发布面。

Windows 源码嵌入时没有 rpath。consumer executable 链接共享 HYLE 后调用
`hyle_stage_runtime_dependencies(target)`，会把其传递 DLL 放到 executable 同目录；这比让 HYLE
篡改父工程输出目录更局部、也更容易迁移。安装包提供同名 helper；Unix build/install tree
继续使用生成器的 rpath。

install export 直接遍历已解析产品 target，不另造一份依赖表；build-tree alias 与 imported target
都叫 `hyle::<module>`。`HYLE_INSTALL=ON` 只允许 `HYLE_BUILD_TESTS=OFF`，因为测试构建会有意改变
`HYLE_TESTING` 门控布局，不能冒充可发布二进制。公开头整体安装（header-only foundation 有跨二进制
组件引用），可用二进制组件由 `Hyle_AVAILABLE_COMPONENTS` 和 imported targets 决定。安装布局服从
GNUInstallDirs；bin/lib/include/data 子目录必须保持相对路径，迁移根由 `CMAKE_INSTALL_PREFIX` 决定，
避免 export 把第一次安装机器的绝对目录烙进 consumer。

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

io 的 pinned nghttp2 作为私有 PIC archive：shared io 将其吸收，不产生第二个运行时 ABI；static io
通过 `LINK_ONLY` 迁移 archive、生成头和 MIT 条款。compute 的 fetched ggml/volk 打包尚未完成，
`HYLE_INSTALL=ON` 遇到这些组合会在 configure 阶段拒绝，不能产出缺运行库的“半可迁移包”。
cross compile 和三平台完整迁移矩阵也尚未取得最终证书，状态只见 build/package milestone。

## 维护规则

- 新模块必须同时登记 component、直接依赖和 module target；三处不一致由配置测试拦截。
- 新可选第三方只能在其消费组件已选且 feature 已启用时发现/取得。
- 不在 CMake 中写开发机绝对路径；preset 是可选入口包，不是编译成功的隐藏前置。
- parent-owned 的 generator、toolchain、配置、输出目录和 `BUILD_SHARED_LIBS` 不得被 subproject 改写。
- source tree 只存源码与审查过的 vendored 资产；cache、下载、生成头和制品一律进 build tree。
- HYLE 当前 proprietary notice、第三方总清单，以及编入所选组件的 vendored/ported 许可证随
  install graph 安装；许可证 payload 与 binary/archive 一样属于可迁移闭包。完整边界见
  [`licensing.md`](./licensing.md)。
