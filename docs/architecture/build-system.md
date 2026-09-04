# 构建与组装架构

> 状态：当前设计。本文描述仓库中实际生效的 CMake 契约，而不是发布承诺。
> Hyle 当前交付面是 C++：源码组装、静态 SDK、共享 SDK。早期公开 C/跨语言表面已经移除，
> 不属于当前产品设计。
>
> 2026-09-01 起，当前实现采用“内部 factor DAG / 外部产品 façade”双层投影；最近一次独立的
> Build/package 复检记录在
> [`build-system-v9-current-tree-replay`](../implementation/tasks/build/build-system-v9-current-tree-replay.md)。
> 此后 Platform/Mem v3 又登记了 `platform_thread`、`platform_fs` 与 `mem_vm`；对应模块轮只证明其
> 受影响 journey，不替 Build/package 横向开发单元签发新的整体收口结论。

## 设计目标

构建系统必须同时满足四件事：

- 模块边界可见：每个组件拥有明确 target，内部依赖不能藏在目录顺序或手工链接里。
- 组装可裁剪：源码使用者、静态 SDK 和共享 SDK 使用同一组件图。
- 用户入口内聚：内部为解环产生的细粒度 factor 不直接变成用户必须理解的产品菜单。
- 外部依赖可复现：身份、版本、内容和许可证来自一份锁定记录，并投影到在线、离线、安装和 CI 流程。
- 回归可持续：新模块加入后自动进入基本构建测试；更昂贵的产品形态和硬件路径分层验证。

构建系统证明“这些源码能按声明的契约组装和消费”，不代替模块的功能、API 或性能收口。

目标架构由 `HyleTargetPlatform.cmake` 在编译器识别后归一化为
`HYLE_TARGET_PROCESSOR`，源码、ISA 与组件能力选择只读取这一事实。不能直接依赖
`CMAKE_SYSTEM_PROCESSOR`：Visual Studio 多平台生成器在 `-A ARM64` 时仍可能把它保留为宿主
`AMD64`，从而让 ARM64 产物错误选择 x86 源码。
归一化结果同时写入只读 global property，供 `add_subdirectory(hyle)` 返回后的父工程 helper 使用；
父工程不得重新从 `CMAKE_SYSTEM_PROCESSOR` 猜一次目标架构。

## 当前 Target 与所有权

当前 target 分为两层：

- 内部 factor 的构建树名称是 `hyle_<factor>`；安装包为表达传递闭包时导出为
  `hyle::_factor_<factor>` detail target。两者都不是普通用户入口。
- 用户在源码树和安装树都只消费 `hyle::<product>` façade；`HYLE_COMPONENTS` 和
  `find_package(Hyle COMPONENTS ...)` 也只接受产品名。
- 当前 `core`、`math` 与 `timer_queue` factor 是 `INTERFACE`；`container_intrusive`、`util_core` 也是
  `INTERFACE` factor，但分别由公开的 `container`、`util` façade 聚合，不是独立用户产品。
- 其余 factor 通常是编译型 target；product façade 始终是无运行时成本的 `INTERFACE` target。
- `hyle_build_contract`、`hyle_testing`、`hyle_trace` 是横切构建 target，不是产品组件根。
- `hyle_build_contract` 只传播所有 C++ factors 共同需要的最低语言、文本编码与预处理语义；它不选择
  exception mode，也不传播某个产品的 ISA 许可。公开 inline 实现若需要特定 ISA，必须由 owning
  factor 直接以 `PUBLIC` usage requirement 传播。当前 x86 Container 的 header-inline SSSE3 probe
  因而由 `hyle_container` 自己拥有；只选择 Core 不会无故提高 ISA permission/floor。
- `hyle::simd_expert` 是 SIMD 用户显式选择的 `INTERFACE` 开发 target，在源码、installed static 和
  installed shared 三种交付中同名存在。`hyle_target_simd()` / `hyle_add_fat_kernel()` 通过它向
  target-bound consumer TU 提供中央锁定 Highway/xsimd 头和一致的 ISA flags。它不被普通
  `hyle::simd` 传递、不建立 vendor register ABI，也不承诺需要上游 `.cc` 的 compiled contrib。
- `hyle_highway_headers` 是由中央 lock 建立的 build-only 外部实现 target。SIMD 和 Container 的 target-bound
  TU 可以直接链接它；它只投影受锁 include/define，不是 HYLE 产品组件，也不制造 `container -> simd`
  内部依赖。
- `hyle_xsimd_headers` 与 `hyle_simd_provider_headers` 只存在于源码构建树；前者投影锁定 xsimd，后者
  组合两个候选给 HYLE 自身 SIMD target-bound TU。安装包不导出这些内部 target，而由
  `hyle::simd_expert` 直接投影随 SDK 安装的精确锁定开发头；xsimd 也不因共享中央 lock 形成新的 HYLE
  组件边。
- 横切 target 也必须遵守真实消费边：模块只有在源码或公开头读取 trace/testing gate 时才传播
  对应 target，并通过 `TRACE`/`TESTING` 显式 opt-in；配置会把声明与当前 selected sources/公开头
  反向对账。纯编译/布局地基 `core` 不声明这两个 gate，不能因全局开关无故携带 profiler runtime
  或 test-only ABI definition。
- Tracy 实现在模块图声明后才建立；`HYLE_TRACE=ON` 而所选闭包没有 trace consumer 会配置失败，
  不会把 Tracy 悄悄变成第二个产品根。

### 双层投影合同

Stage C 能力账本中的 30 项是**候选 semantic factor**。当前有 25 个 identity 同时进入
`module-status.json` 与中央 manifest：21 个已经拥有实际 target、源码 ownership、测试和 HYLE 直接
依赖，`io_tls/io_http/io_http2/io_transfer` 4 个则只是 `QUEUED` 身份保留项。保留项在自己的
`SCOPED/AUDITED` 轮次前没有 target、依赖、公开头或 product-root 承诺，也不会被 `all` 选择。
其余候选尚未登记。所有 factor 都不是用户产品；中央 manifest 另行投影 14 个 product façade。
双层投影把旧
“component”承担的四种含义拆开：

| layer | owns | may depend on | consumer-visible contract |
|---|---|---|---|
| factor | 源码/头 owner、状态、direct `PUBLIC/PRIVATE` 边、测试与唯一算法裁决权 | 其他 factor 与 build-only mechanism | 默认无 `hyle::<factor>`；安装时只可能以 `_factor_*` detail target 承载传递闭包 |
| product façade | 用户问题域和完整 factor-root 集 | 只以 `INTERFACE` 链接 factor roots | `HYLE_COMPONENTS`、`find_package(Hyle COMPONENTS ...)` 与 `hyle::<product>` 的唯一普通入口 |
| feature | 可选产品能力与相应外部依赖 | owning product/factor | 显式 build option 与 package capability metadata |
| build-only mechanism | vendor、生成器、系统库、trace/testing/build contract | 不参与 HYLE semantic DAG | 不进入产品 catalog |

内部 factor 不能链接 façade；否则 `Time -> Container façade -> Mem -> Platform -> Time` 会把已经消除的
环重新引入。façade 自身没有源码、binary、模块状态、API owner 或 closure verdict，只在图外选择一组
factor roots。公开头仍由 factor 唯一拥有，产品安装的是所选 factor 闭包的头并继续执行 direct-edge
审计。

外部 catalog 按完整问题域保持小而稳定：`core,error,time,platform,mem,obs,simd,math,container,util,
sched,io`；`compute,graph` 待产品重写后仍以问题域加入。比如 `container_intrusive` 与未来
`container_queue` 是内部 factor，用户从 `hyle::container` 获得 Container 产品；`util_core/registry/
symbol` 由 `hyle::util` 聚合。只有用户确实需要独立选择的语义产品或 expert journey 才新增公开 target，
不能因为内部有一个 target 就自动公开。

source/static/shared 必须暴露同名 façade。当前首选物理实现是继续保留精确 factor artifact/target，由
INTERFACE façade 聚合：静态链接器可按需抽取对象，共享包可加载同级 factor DSO。CMake export 为表达
传递闭包而出现的 `hyle::_factor_*` 是不承诺的 package detail，既不列入
`Hyle_AVAILABLE_COMPONENTS`，也不写入用户指南。是否进一步像本地 Folly
`23ebefdf96c116700d3e134f04d95a2b60a509de` 那样把 OBJECT 单元聚合成 monolithic shared artifact，必须
由启动、部署或符号边界证据另立任务；不能借 façade 迁移顺手改变物理 ABI。

本地 peer 对账还包括 oneTBB `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` 的少量语义组件、ggml
`d4716378882593333721eb33f153144b6885caf2` 的 frontend/backend 分层，以及 Abseil/Boost 的细粒度
公开 target 反例。结论不是复制任一库，而是把逻辑 factor、物理 artifact 与用户产品入口作为三项分别
裁决。

各层只拥有一类事实：

| 位置 | 唯一职责 |
|---|---|
| `cmake/HyleComponents.cmake` | 组件全集、append-only 身份顺序、Hyle 内部 `PUBLIC`/`PRIVATE` 依赖；构造顺序由图计算 |
| `src/<module>/CMakeLists.txt` | 本模块源码、公开头、编译属性和外部实现 target |
| `tests/<module>/CMakeLists.txt` | 本模块测试；被选组件缺少该文件会配置失败 |
| `bench/<module>/CMakeLists.txt` | 可选 benchmark，不参与正确性证明 |
| `samples/package/<module>/` | 安装包消费样例，由 package smoke 自动发现 |

模块目录不得重新声明 Hyle 内部依赖。跨模块源码调用与 target 图不一致时，应修图或修调用，不能靠 `add_subdirectory()` 顺序偶然通过。

### 单一内部依赖 manifest

当前 `HyleComponents.cmake` 是 factor 关系与产品 catalog 的唯一 manifest，并产生三种投影：

1. `HYLE_COMPONENTS` 先把产品映射为 factor roots，再解析传递闭包，决定实际加入哪些模块目录；
2. `HYLE_DEV_FACTORS` 让顶层维护/CI 精确选择 factor roots，不能用于嵌入或安装；
3. `hyle_add_module()` 生成真实 factor `PUBLIC|PRIVATE` 边，`hyle_add_product_facades()` 在图外生成
   无源码的用户入口。

公开头通常按 `include/hyle/<factor>/` 自动归属。若细粒度 factor 的头必须留在完整产品命名空间中，
例如 `container_intrusive` 的头位于 `include/hyle/container/`，只在该 factor 的中央声明中用
`SPLIT_PUBLIC_HEADERS` 登记一次；factor 头集合、父 namespace 排除、安装集合和 include-owner 审计都
从该记录派生，不得分别维护路径名单。

配置结束时 `hyle_validate_final_module_graph()` 再核对 target 的 `LINK_LIBRARIES` 和
`INTERFACE_LINK_LIBRARIES`，防止后续手工链接绕过 manifest。中央登记顺序与
`module-status.json` 一样是不可改写的 component identity 历史，不兼任依赖顺序：配置先对完整 manifest
做一次稳定 DFS，拒绝 unknown/QUEUED edge 与 cycle，并以登记顺序作为无依赖关系节点的 tie-breaker
得到拓扑；依赖闭包再按该拓扑单次反向传播，target 也严格按解析结果构造。这样旧 factor 可以自然依赖
后来拆出的低层质因数，而不用重排状态历史或复制实现，整体仍为 O(V+E)。随后只对本次 selected closure
读取公开头和实现源码；all-product CI 仍审完整树，单产品源码消费者不会被另一个未选 factor 的施工中
状态阻断，include audit 成本也从全库 `S_all` 收束到 `S_selected`。

当前图的关键关系是：

```text
core        leaf
math        -> PUBLIC core
container_intrusive -> PUBLIC core
util_core   -> PUBLIC core
error       -> PUBLIC core
obs         -> PUBLIC core error time; PRIVATE mem_reclaim
time        -> PUBLIC core
timer_queue -> PUBLIC core time container_intrusive
platform    -> PUBLIC core error
platform_thread -> PUBLIC core error time; PRIVATE platform
platform_fs -> PUBLIC core error time; PRIVATE platform
mem_reclaim -> PUBLIC core
mem         -> PUBLIC core error platform
mem_vm      -> PUBLIC core error mem platform platform_fs util_core
simd        -> PUBLIC core platform
container   -> PUBLIC core mem mem_reclaim; PRIVATE platform
util        -> PUBLIC core error mem container
sched       -> PUBLIC core container_intrusive error time platform platform_thread mem mem_vm util_core; PRIVATE container timer_queue
io          -> PUBLIC core mem container time timer_queue sched
compute     -> PUBLIC core error mem sched; PRIVATE platform platform_thread platform_fs simd
graph       -> PUBLIC core error mem container sched; PRIVATE util_core
```

`io_tls`、`io_http`、`io_http2` 与 `io_transfer` 已登记为 `QUEUED`，但不属于上述可构建图。现有机器 ID
`io` 在后续拆分中承载 IO core（completion/file/net/IPC）语义，避免为概念改名制造状态身份迁移；
`hyle::io` façade 在各因子分别收口前仍只投影现有 `io` 根。保留项禁止 target metadata 和产品根，
`HYLE_DEV_FACTORS=all` 只选择 `HYLE_IMPLEMENTED_COMPONENTS`，显式选择保留项会以可诊断错误失败。

`PUBLIC` 表示消费者编译或链接本 factor 时也需要该 factor 依赖；`PRIVATE` 表示只服务本 factor 实现。
`BUILD_PRIVATE` 保留给确有真实 `$<BUILD_INTERFACE:...>` target 边、但不扩大安装闭包的 HYLE 内部
header seam，不能用来伪装外部依赖或隐藏跨 factor 调用。当前 Container 的字符串/bitset target TU
直接链接 build-only `hyle_highway_headers`，所以中央内部图没有 `container -> simd` 或其他虚构
内部 header 边；来源与许可证由下节的外部依赖登记投影。这里记录的都是实际 target/link 契约。
`container_intrusive` 证明 header-only 能力也应在公开类型穿过 factor 边界
时拥有真实 target：`timer_queue` 暴露 `TimerList`，Sched cancel record 暴露 safe hlist hook，故不能把
这些边藏成 include-path 偶然可见；完整
`container` 产品 façade 聚合 intrusive 与普通容器，但内部 `container` factor 本身不反向聚合它。
`platform/fs.h` 公开暴露 `time::UnixTimestamp`，所以
`platform_fs -> PUBLIC time` 是真实 SDK 边；dependency-light 的 `platform` host factor 不再因此取得
Time。Platform façade 聚合 `platform/platform_thread/platform_fs`，Mem façade 聚合
`mem/mem_vm/mem_reclaim`；
内部 factor 仍只链接自己实际使用的 factor，不能链接 façade。sched 的 waiter 链因 detached wake-link 复用而保留私有特化，
但 cancel callback 已迁移 hlist 并形成真实 direct edge。io、compute 与 graph 的当前公开面都暴露 sched
类型，因此 sched
是无条件 `PUBLIC` 边；无 fcontext 平台会诚实拒绝这三个产品根。未来若要交付 raw proactor 或
host-only compute，必须先拆成拥有独立公开头与 journey 的真实 factor，不能只从图上删边。

Core 的 source contract 不再是 target-neutral 例外。公开头直接 include `<hyle/core/**>` 或
`<hyle/version.h>` 的组件必须直接声明 `PUBLIC core`，即使 `hyle::core` 是 `INTERFACE`、没有
binary symbol。配置阶段会扫描 owned public headers 与实现源码并同中央 manifest 对账：公开
include 必须有直接 `PUBLIC` 边，实现 include 必须有直接 `PUBLIC/PRIVATE` 边；只通过 forward
declaration 暴露外部类型的边用显式 `PUBLIC_TYPE_DEPS` 注释；不进入产品闭包的实现头借用必须匹配
`PRIVATE_BUILD_DEPS` seam。漏边、多余公开边和未注释 type/build edge 都失败。
`container_intrusive`/`util_core` 这样的 header-only target 同样遵守这条规则；共享源码
include root 或复制整棵 include tree 都不能使未登记依赖合法。

该 include-owner 检查只裁决正在组装的 selected closure，不把未选模块重新耦合进 source journey。
选择某产品时，它的全部传递 factor 都会被检查；CI 的 `HYLE_COMPONENTS=all` 因而仍机械覆盖每个 factor。
factor manifest 自身的 unknown edge、cycle、重复和 PUBLIC/PRIVATE 冲突无论选择什么 root 都会失败，
因为这些属于轻量全局 schema，而不是未选源码内容。登记顺序不再被误当成拓扑约束。

`<hyle/version.h>` 是 build/package 拥有的 checked-in 产品元数据，并由 Core include contract
交付；Core 本身不拥有 linked version query。`<hyle/visibility.h>` 的 export/import policy 也由
build/package 拥有，随各 compiled module 的 static/shared 形状维护，而不是 Core capability。
当前 TU 的 OS/arch/compiler/build/sanitizer facts 由 Core 自己的
`<hyle/core/environment.h>` 交付；它不是 runtime Platform binary，也不再使用旧根级
`<hyle/platform.h>`。source、static 与 shared consumer 必须编译同一份环境合同。

Exception mode 同样是 parent/toolchain 合同，而不是 HYLE 的全局默认。普通 MSVC CMake 工程仍会按
其默认选择 `/EHsc`；显式关闭异常的 Core-only translation unit 不会被 `hyle::core` 重新打开，并由
`HYLE_HAS_EXCEPTIONS == 0` 反映当前 TU。这个事实只签 Core：其他产品是否能在 no-exception 闭包中
组装，必须由各自的正常 consumer journey 单独证明，不能从全局 target 没有 `/EHsc` 外推。

## Host 工具链选择入口

CMake 顶层只负责在 language 初始化后验证 floor：GCC 14+、Clang/AppleClang 18+、MSVC 19.43+；它
不能在已有 build cache 中安全换 compiler。Windows 维护命令由 `scripts/hyle-windows.ps1` 导入已安装
Visual Studio 环境；Linux/WSL/macOS 维护命令统一由 `scripts/hyle-unix.sh` 包住任意 argv：

```bash
scripts/hyle-unix.sh cmake --preset linux
scripts/hyle-unix.sh cmake -DHYLE_SOURCE_DIR="$PWD" \
  -DHYLE_SMOKE_ROOT=/tmp/hyle-time -DHYLE_SMOKE_COMPONENT=time \
  -P cmake/HyleCiSelectedComponentSmoke.cmake
```

Unix wrapper 不安装软件、不改 PATH/preset/cache，也不复制 CMake action。调用者未设置 `CC/CXX` 时，
Linux 从已安装的成对 versioned GCC 14+、Clang 18+ 或合格的 unversioned pair 中选择；macOS 选择系统
clang。调用者可以显式设置任意受支持 pair，但只设一个、混用 family 或低于 floor 会在目标命令启动前
失败。随后脚本以 `exec "$@"` 原样执行命令，因此 configure、`cmake -P` smoke、sanitizer driver 和
诊断工具共用一个选择入口而不形成第二套 build graph。已有 cache 已绑定旧 compiler 时必须使用新的
独占 scratch；wrapper 不删除或重写用户数据。

## 外部依赖：单点登记，多路投影

外部依赖分成两个正交登记层，不能相互替代：

| 登记层 | 文件/接口 | 回答的问题 |
|---|---|---|
| 来源与法律锁 | `cmake/HyleDependencyLock.json` | “究竟是哪份源码，内容和许可证是什么？” |
| 产品使用登记 | `hyle_register_package_dependency()` | “哪个模块使用它，安装包需要导出哪些 target、资产和下游依赖？” |

锁文件由 `HyleDependencyVersions.cmake` 读取，记录仓库或归档身份、revision、归档与展开树哈希、集成源码哈希、许可证表达式及许可证文件哈希。模块依赖脚本可以决定适配器的编译选项，但不得复制版本、URL、哈希或许可证事实。

同一登记事实投影到以下路径：

| 路径 | 契约 |
|---|---|
| 在线配置 | 首次 configure 缓存 miss 时，统一 cache resolver 使用锁文件投影出的 URL/SHA256 自动取得源码；命中后只校验和复用 |
| 离线配置 | 搬运同一 resolver 生成的完整共享缓存，或由 `hyle-offline-deps.cmake` 从同一锁验证 `deps/<name>/` 并设置标准 `FETCHCONTENT_SOURCE_DIR_*`；两者都禁止缺失时联网 |
| 安装导出 | package registry 汇总 exported targets、静态闭包、`find_dependency()`、运行时/开发资产和工具 |
| 许可证 | 配置、打包和 CI 均按锁文件校验许可证文件及哈希；安装 `LICENSE` 与第三方 notices |
| CI/cache | lock 哈希参与共享 source-cache key；普通产品构建和 locked-feature job 使用同一 resolver 验证适配器与安装包，不维护第二套下载器 |
| 源码包 | 打包前审计所有 in-tree/integrated-source 锁记录，但不把 FetchContent 下载物塞进源码包 |

源码资源和生成状态使用两个不同合同：

- `HYLE_DEP_SOURCE_CACHE` 保存 archive 与未经修改的展开源码，跨 Debug/Release、编译器、架构、
  sanitizer、build tree 和同用户 checkout 共用。显式 CMake 变量优先于同名环境变量；默认使用平台用户
  cache：Windows `%LOCALAPPDATA%/hyle/dependencies/v1`，macOS
  `$HOME/Library/Caches/hyle/dependencies/v1`，其他 Unix 使用
  `$XDG_CACHE_HOME/hyle/dependencies/v1` 或 `$HOME/.cache/hyle/dependencies/v1`。
- `HYLE_DEP_BUILD_DIR` 默认是当前 Hyle binary tree 的 `_deps`，只保存依赖的 binary/subbuild 与
  CMake 生成状态，不得跨 build tree 共用。

在线首次 configure 缺少某项 source identity 时，会在 per-identity 文件锁内下载官方 archive，先校验
archive SHA-256，再在 staging 中展开并校验完整源码树与许可证，最后原子落入共享 cache。后续
configure 和所有 build 均不重新下载。显式 `FETCHCONTENT_SOURCE_DIR_<NAME>` 始终优先，并走同一
严格源码 verifier；`FETCHCONTENT_FULLY_DISCONNECTED=ON` 时 cache miss 直接失败。
嵌入父工程时 Hyle 只在自己的目录作用域设置 `FETCHCONTENT_BASE_DIR`，不改写父工程的值；共享源码
override 也只对 Hyle 实际取得的锁定名字生效。

维护入口是：

```text
cmake -DHYLE_DEP_ACTION=status -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=sync -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=sync -DHYLE_DEP_NAMES=xsimd -DHYLE_DEP_REPAIR=YES -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=check-updates -P cmake/HyleDependencyCacheTool.cmake
cmake -DHYLE_DEP_ACTION=prune -DHYLE_DEP_PRUNE_CONFIRM=YES -P cmake/HyleDependencyCacheTool.cmake
```

`sync` 与 configure cache-miss 共用取得、锁、staging 和验证实现；`check-updates` 只报告 upstream
事实，不改 lock；`prune` 只处理带 HYLE marker 且不再由当前 lock 引用的 identity。较重的可选后端
发现仍留在相应模块脚本，例如 IO、compute 和 trace。

## Option 语义

用户可理解的 option 应表达产品选择或验证强度；计算得到的 `HYLE_BUILD_<COMPONENT>` 只是内部状态。

`HYLE_COMPONENTS` 只接受产品 façade 名。精确 factor build 使用仅限顶层维护/CI 的
`HYLE_DEV_FACTORS`，它与 `HYLE_COMPONENTS` 互斥且不能用于安装包或普通源码嵌入。`all` 始终表示
所有产品，而不是公开全部 factor。

| 类别 | 选项 | 语义 |
|---|---|---|
| 产品 | `HYLE_COMPONENTS` | `all` 或逗号/分号产品列表；空值、未知名、`all` 与显式名混用均失败 |
| 内部维护 | `HYLE_DEV_FACTORS` | 仅顶层维护/CI 的精确 factor 根；不得与 `HYLE_COMPONENTS` 或 `HYLE_INSTALL=ON` 混用 |
| 链接形态 | `HYLE_STATIC` | `OFF` 明确构建 shared，`ON` 明确构建 static；不改父工程 `BUILD_SHARED_LIBS` |
| 安装 | `HYLE_INSTALL` | 生成可重定位 SDK；当前要求 `HYLE_BUILD_TESTS=OFF` |
| 验证 | `HYLE_BUILD_TESTS`, `HYLE_TEST_INSTRUMENTATION`, `HYLE_WERROR`, `HYLE_BUILD_BENCH` | 分别控制测试、测试插桩、警告门和 benchmark；普通测试默认 shipping layout，诊断插桩须显式开启 |
| Sanitizer | `HYLE_SANITIZER` | `off|asan|tsan|ubsan`；非法值配置失败 |
| 依赖位置 | `HYLE_DEP_SOURCE_CACHE`, `HYLE_DEP_BUILD_DIR` | 前者是跨 build 的锁定源码资源，后者是当前 build 的 binary/subbuild；不再用一个模糊目录混放两者 |
| SIMD | `HYLE_SIMD_EXACT_TARGET` | 在 fresh cache 中只在 simd 被选时声明；空值或 `SCALAR/SSE4/AVX2/AVX512/NEON/SVE/RVV` |
| 自动特性 | `HYLE_IO_TLS`, `HYLE_COMPUTE_CUDA` | 在 fresh cache 中只由所选 owner 组件声明；类型为 `AUTO|ON|OFF`，当前默认 `OFF`；`AUTO` 探测、`ON` 必须存在、`OFF` 不探测 |
| 布尔特性 | `HYLE_IO_HTTP2`, `HYLE_COMPUTE_GGML`, `HYLE_COMPUTE_VULKAN` | 在 fresh cache 中只由所选 owner 组件声明，昂贵能力默认 `OFF`；`HYLE_COMPUTE_GGML_NATIVE` 仅在 ggml 开启时声明 |
| Trace | `HYLE_TRACE`, `HYLE_TRACY_DIR` | 在 fresh cache 中仅闭包有 trace consumer 时声明，默认 `OFF`；路径只在开启时声明。当前只有 Tracy，已删除无选择意义的 `HYLE_TRACE_BACKEND` |

Sanitizer 是整个所选 target 图的构建形状，不是只给测试可执行文件加一个编译参数。对象库和静态库
携带插桩；最终 executable/shared/module target 负责链接相应 runtime。Windows 的 cl.exe 与
clang-cl 目前只支持 `asan`：HYLE 显式链接编译器自带的 dynamic-CRT runtime 和 thunk、保留
SEH interceptor 与调试信息，并把匹配的 ASan DLL 放到最终产物旁边；请求 `tsan`/`ubsan` 会直接
配置失败并要求转到 GCC/Clang 环境。HYLE 自己取得的 GoogleTest 会使用同一 ASan 形状，嵌入父工程
提供的 `GTest::gtest_main` 则由父工程保证 ABI/插桩一致。

顶层配置省略 `HYLE_COMPONENTS` 时默认 `all`；作为子目录嵌入时，父工程必须在
`add_subdirectory()` 前显式设置产品根，防止一次普通依赖引入意外构建整库。在独占的 fresh build
tree 中，测试插桩、SIMD exact、IO/compute feature 和 trace 路径等 cache 控制只在其 owner/gate
成立时声明。CMake 会保留旧配置曾写入的 cache key；改变组件闭包或产品形状时应使用新的独占
scratch，或明确 `--fresh` 后重新配置，不能把旧 key 仍可见误认为当前闭包消费了该能力。生成器、工具链、配置类型、安装前缀、CUDA architectures 和
`FETCHCONTENT_SOURCE_DIR_*` 使用标准 CMake 变量。正式证据与发布配置应把会改变产物的 `AUTO`
固定为 `ON` 或 `OFF`。安装包消费者通常不需要任何 `HYLE_*` 构建 option。

## 三种 C++ 交付形态

### 源码组装

获得授权源码的使用者在 `add_subdirectory()` 前显式设置 `HYLE_COMPONENTS`，组装所需产品并链接
`hyle::<product>`。只有 Hyle 自身作为顶层工程时，省略该变量才表示 `all`。父工程拥有生成器、
工具链、输出目录和标准 CMake 策略；Hyle option 只描述 Hyle 自身。嵌入模式不创建源码发行 target。

根 `project()` 只声明 `LANGUAGES CXX`。C 是内部实现语言而非产品消费者前置条件；在
`HYLE_BUILD_TESTS=OFF` 的产品/source 形状中，当前只有所选闭包含 mem/mimalloc 时才
`enable_language(C)`。测试构建还会进入 GoogleTest 自己的 CMake project，并可能探测 C frontend；
这属于测试工具链需求，不建立 C 产品 surface。

请求 `core` 时唯一的产品 target 是 `hyle::core` INTERFACE target；构建树仍有无 artifact 的
内部 build/trace/testing foundation targets，但 Core 不链接 trace/testing，也不产生
object/archive/runtime。请求其他组件会通过中央图取得它所需的 Core source contract，不能靠父工程
的 include path 偶然可见。

### 静态 SDK

以 `HYLE_INSTALL=ON`、`HYLE_STATIC=ON` 安装后，消费者使用：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS error mem)
target_link_libraries(app PRIVATE hyle::mem)
```

导出 target 必须传播静态链接所需的私有闭包、系统库和已登记的 `find_dependency()`。
若只安装/消费 `core`，package 只提供 headers 与 imported INTERFACE target，不存在 Core archive。

### 共享 SDK

以 `HYLE_INSTALL=ON`、`HYLE_STATIC=OFF` 安装，消费语法相同。Unix 安装 target 使用相对同级 RPATH；Windows 消费 target 可调用 `hyle_stage_runtime_dependencies(target)` 把所需 DLL 放到运行目录。

安装前缀和 GNUInstallDirs 必须保持相对、可搬迁；绝对安装子目录会失败。SDK 只安装所选产品对应的 factor 闭包
拥有的公开头；只有闭包含 `TRACE` consumer 时才额外安装 trace foundation 的 `obs/trace.h`，不会
复制整棵 `include/hyle`。
公开头 include-owner 审计要求每个跨组件 include 都有对应直接 `PUBLIC` 边，防止缺失 target 被整树
安装偶然遮蔽。`Hyle_AVAILABLE_COMPONENTS` 沿用 CMake 的 component 术语，但内容只列产品 façade；
`hyle::_factor_*` detail target 不在其中。源码授权与静态/共享
产物形态是两个独立维度。

预发布 `0.y.z` 的 package version 和 shared-object loader/SOVERSION identity 都使用完整版本并要求
`ExactVersion`；1.0 后才切换为 conventional major compatibility。匹配版本的 C++ headers、artifact、
工具链和配置始终是同一交付合同。

Core-only shared SDK 同样没有 Core DSO/DLL 和 runtime staging；`HYLE_STATIC` 只影响请求到的编译型
模块。Core helper 已内联进 consumer，header 变化后必须重编，不能靠替换其他 shared artifact
更新。

## 源码包与离线构建

顶层 Git checkout 提供 `hyle_source_package`：它审计全部 in-tree/integrated dependency，拒绝 dirty 或 untracked 工作树，以 `git archive` 打包当前提交，并生成源码清单和 SHA256。产物包含已跟踪源码与法律材料，不包含构建缓存或 FetchContent checkout。

离线依赖包是独立输入。推荐在联网准备机把 `HYLE_DEP_SOURCE_CACHE` 指向一个空目录并执行
`HYLE_DEP_ACTION=sync`，然后完整搬运该带 marker 的 cache；离线机把同一变量指向解包位置并设置
`FETCHCONTENT_FULLY_DISCONNECTED=ON`。configure 会重新验证源码 identity，但不会访问网络。

需要逐项交付受审 Git checkout/展开树的组织，也可以继续准备与锁文件匹配的
`deps/<lock-name>/`，再使用：

```powershell
cmake -S . -B build-offline `
  -C hyle-offline-deps.cmake `
  -DHYLE_COMPONENTS=all
```

离线初始化脚本只接受精确、干净的 Git checkout 或能通过展开树与许可证校验的官方归档；缺少被选组件所需依赖时立即失败。两种离线介质都不携带 dependency binary/subbuild，目标机仍在自己的
`HYLE_DEP_BUILD_DIR` 生成它们。`HYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES` 只用于明确承担风险
的异常恢复，不是发布流程。

## CI 分层与持续回归

CI 不把所有平台、后端和性能门塞进一次任务：

| 层级 | 当前职责 |
|---|---|
| PR / main push | 只跑 Linux all-product shipping-layout CPU 基线、文档和产品/factor/依赖合同；昂贵可选后端关闭 |
| 手工 full dispatch | Linux/Windows/macOS CPU 基线，并在适用主机跑 production shape、package consumer 与源码包 |
| 手工 Windows clang-cl | 显式 test instrumentation 的一组底层组件及多个 exact-SIMD artifact |
| 手工 Linux AArch64 cross | 全产品交叉编译；不宣称目标机运行正确性 |
| 手工 locked feature/package | 锁定归档、可选适配器、static/shared SDK 和 trace smoke |
| 手工/发布证据 | sanitizer、真实 TLS/CUDA/Vulkan 设备、SVE/RVV 运行和性能认证 |

当前仍沿用历史内部变量名 `HYLE_KNOWN_COMPONENTS` 存放全部已登记 factor identity 的
只追加审计顺序；实际 target 构造顺序由中央依赖图稳定拓扑解析，不要为了新依赖边
改写历史 identity 顺序。另由
`HYLE_IMPLEMENTED_COMPONENTS` 保存已有 target 的集合。product `all` 只取 façade roots；maintainer
`HYLE_DEV_FACTORS=all` 只取 implemented 集合，因此 `QUEUED` 身份保留项不会要求虚构源码或测试目录。
某个因子从 `QUEUED` 进入实际施工后，只有在真实 target/ownership/测试合同就位并移除 manifest 的
`QUEUED` 标记后，才进入维护者 all-factor 构建。两者都不是用户 component catalog。PR 与 main push
都故意只保留高信号 Linux 门以控制额度；Windows、macOS、
交叉编译、完整交付和 locked feature 由 `workflow_dispatch` 展开。package smoke 自动发现安装消费
样例。更严格的 production roots 仍是显式列表，只能在对应模块通过收口后扩展。CI 绿灯只覆盖
该 job 明示的形状，不能外推硬件、性能或未运行配置。

### 模块持续回归

模块审查后的每次依赖、公开头或 option 改动，都应至少重跑：

1. 本模块及其传递依赖的 source build/test。
2. 用独占 scratch 跑该产品根的 `HyleCiSelectedComponentSmoke.cmake`：source embedding、shared、static
   及搬迁后 installed consumer；Windows 使用 `hyle-windows.ps1 -Action SelectedComponentSmoke`，
   Unix 用 `hyle-unix.sh cmake ... -P cmake/HyleCiSelectedComponentSmoke.cmake`，避免继承不受支持的
   host 默认 compiler。
   smoke 生成的 Darwin 父工程从中央 platform requirement 取得 deployment target；不得让 source 与
   SDK 两种形状偶然使用不同的 macOS loader floor。
3. 涉及外部依赖时的锁、离线和许可证投影。
4. 涉及 ABI 或运行时装载时的 relocated consumer。
5. 涉及性能时的独立、可复现性能门；普通 CI 不替代性能证据。

并行 agent/session 不得复用 preset 的固定 build tree，也不得在另一会话的 cache 上重配。Windows
wrapper 的 `-BinaryDir <absolute-session-scratch>` 为 Configure/Build/Test 提供独占目录；配置单个
产品时传 `-Component <product>`，精确 factor 维护配置传 `-Factor <factor>`。临时 scratch 应位于
明确的会话目录，完成后由拥有它的会话清理。

## 新模块接入协议

以下协议中的“模块”指 factor；中央 product catalog 另行决定它被哪些 façade 聚合。factor task
不得自行创建、命名或链接 façade。

1. 在 `HyleComponents.cmake` 末尾登记 factor identity 和全部 factor `PUBLIC`/`PRIVATE` 边；不要为了
   依赖顺序移动既有登记，构造拓扑由 manifest 自动解析。公开头若位于
   另一个 factor 的产品 namespace，同时在这一次声明中登记 `SPLIT_PUBLIC_HEADERS`。若仅借用另一
   模块的实现 header 且不形成交付需求，登记受审的 `PRIVATE_BUILD_DEPS` seam，不得伪造成产品边。
2. 用 `hyle_add_module()` 创建 `src/<module>` target，只列本模块源码和外部实现 target。
3. 增加 `tests/<module>/CMakeLists.txt` 及模块聚合测试；需要时再增加 benchmark。
4. 若引入外部源码，先更新 dependency lock，再在模块包登记中声明唯一 owner、feature、导出 target、资产和许可证投影。
5. 增加 C++ 源码消费和 static/shared 安装消费样例；确认 relocation。
6. 运行 all-product、依赖图、离线、源码包和适用的 feature 合同并修到通过；新增合同测试时，
   用隔离的负向 fixture 证明它能拒绝缺边/缺资产，不能故意破坏产品树或共享 CI 来“失败一次”。
7. 更新 architecture、user guide、task/evidence；构建通过不等于模块收口。

## 边界与已知遗留

- 当前公开头、模块 target、安装包 consumer 和样例只定义 C++ 产品面；早期 `include/hyle/c/**`、
  公开 `c_*.cc` bridge、C surface/export fixture 与 managed FFI probe 已移除。没有通过
  [Native FFI 准入](native-ffi-policy.md)前不得恢复第二套镜像 API。
- 根工程只启用 `LANGUAGES CXX`；所选闭包需要 mem/mimalloc 时才按需启用内部 C。内部
  `extern "C"`、C 编译单元或汇编桥接不代表存在 C 产品 API。
- 历史 `hyle/macros.h`/compiler-definition target-neutral 例外已经删除；后续新 header-only 公共
  能力必须有明确 owner target 和真实依赖边。配置阶段对 selected closure 的全部 owned public
  headers 做 include-owner / direct-`PUBLIC` 对账；`obs/trace.h` 是仅随 `TRACE` consumer 闭包交付的
  明确 foundation seam。
- production-shape 根列表仍含已重新打开审查的模块；它是回归集合，不是“已冻结”声明。
- 可选特性都由 owner 模块声明，昂贵能力默认 OFF；TLS/CUDA 保留 `AUTO|ON|OFF` 的显式探测语义，
  其他后端为布尔值。没有第二个 trace backend 时不恢复 backend selector。
- 托管 CI 尚未覆盖 sanitizer、真实加速设备及 SVE/RVV 运行；Windows/macOS 只在手工
  `workflow_dispatch` 运行，不属于 PR/main push 门。
- 安装只复制 selected closure 的 owned public headers；缺少 owner 或 `PUBLIC` 边会在配置阶段失败。
- 源码包有意不携带 FetchContent 依赖；需要全离线构建时必须同时准备可搬运的 HYLE 共享源码 cache，
  或 `hyle-offline-deps.cmake` 使用的独立 `deps/<name>` 目录。
- `compute` 与 `graph` 仍是未完成模块。all-product 绿灯只证明现有构建与测试契约，不证明其功能已经完成。
