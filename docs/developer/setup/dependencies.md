# 依赖与缓存

初始化参数只选择产品、后端与明确的可选能力。各算子由哪些 Provider 执行，以及需要接入哪些第三方实现，
由 HYLE 内部装配；普通用户无需枚举 oneDNN、ggml、Torch 等开关。

## 三类依赖

| 类别 | 示例 | 谁准备 |
|---|---|---|
| 基础工具与系统 SDK | Python、CMake、C/C++ 工具链、CUDA Toolkit、Metal 组件、OpenSSL | 用户按系统安装；初始化发现并检查 |
| 所选闭包的锁定源码/SDK 资产 | Highway、xsimd、ggml、oneDNN、LibTorch、GoogleTest、nghttp2、Tracy | 在线初始化按中央 lock 取得，已有合格资产则复用 |
| benchmark 比较对象 | 独立 Highway/xsimd checkout、其他对手库 | 仅专项 benchmark 准备，不进入普通初始化 |

具体版本、来源、hash 和许可证以 [HyleDependencyLock.json](../../../cmake/HyleDependencyLock.json) 为准，
文档不再维护第二份 pin 清单。CUDA Toolkit 和显卡驱动是系统 SDK，不能由一个源码 commit 代替。
Vulkan 缺少宿主 glslang 时，初始化可用同一 lock 准备宿主工具；该工具与目标设备 runtime 分开。
宿主工具的构建保存在共享源码 cache 同级的 `tools-v1/`，按依赖身份、系统、架构与编译器隔离并复用。

裸 CMake 的 tests/bench、TLS、HTTP/2、trace 默认关闭；`init.py --tests` 才准备测试依赖，
`--tls` 自动查找已安装的 OpenSSL 3，`--http2`、`--trace` 分别取得所需锁定源码。
`--backends all` 请求当前平台支持的全部完整后端，不因缺少某个系统 SDK 而静默裁剪能力。

## 源码与二进制状态分开

平台无关的锁定源码保存在 checkout/build 之外的 `HYLE_DEP_SOURCE_CACHE`，
可用 `init.py --cache-dir <path>` 或同名环境变量指定。默认位置为：

- Windows：`%LOCALAPPDATA%/hyle/dependencies/v1`。
- macOS：`$HOME/Library/Caches/hyle/dependencies/v1`。
- 其他 Unix：`$XDG_CACHE_HOME/hyle/dependencies/v1`，未设置时为 `$HOME/.cache/hyle/dependencies/v1`。

每个 build 的 `_deps` 保存自己的二进制和子构建，不跨编译器/架构共用。
锁定 LibTorch 等平台 SDK 资产使用共享 cache 内各自的平台/版本身份；不复制进每个 build。
切换配置可复用源码，但不能把旧对象、CMake cache 或整套 build 当作可搬运依赖包。
`HYLE_DEP_BUILD_DIR` 是维护者的构建内部布局选项，不是普通初始化所需参数。

共享 cache 的状态检查和预取仍使用现有维护工具：

```text
cmake -DHYLE_DEP_ACTION=status -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=sync -P cmake/HyleDependencyCacheTool.cmake
```

无参数 `sync` 预取工具管理的锁定源码集合，不代表已准备每个后端的平台 SDK 资产。
普通初始化只准备所选闭包，避免无关下载。损坏资产会明确失败；具名修复、清理旧身份及
版本升级的方法见该工具参数与中央 lock，不由初始化删除用户目录或更新依赖版本。

## 离线和直接 CMake

`init.py --offline` 使用已取得的依赖和系统 SDK，缺少时明确失败，不退回联网。
完整介质准备、宿主工具和测试示例见[离线构建](offline.md)。

组织已有逐项来源审核的源码包时，可保留原生入口 `hyle-offline-deps.cmake`，
将精确源码放在 `deps/<lock-name>/`。它验证中央 lock、展开内容和许可证后，
设置标准 `FETCHCONTENT_SOURCE_DIR_<NAME>`；Tracy 使用 `HYLE_TRACY_DIR`。
完整源码交付与安装消费见[交付说明](../../user/distribution.md)。
