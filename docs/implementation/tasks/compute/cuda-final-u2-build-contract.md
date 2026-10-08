# CUDA U2：普通构建、软件身份与三交付

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- 当前协调 owner：compute_route_cuda；baseline `cf34e3bf452c0b795fc73a061113afbfa689420f`，Windows canonical。
  现 registry AUDITED 仅授权旧 F41。本卡先 SCOPED 定界，不从旧卡取得新产品授权。
  backend／Provider 依具体 FIX 再合法绑定，不因读构建源码批量重开 CLOSED。
- trigger：用户要求只选择 CUDA 模式取得默认完整计算路线；当前只有 compute-cpu preset，
  CUDA-only 仍需用户组合多条 route flags。Driver API13040、真实驱动发行版、runtime／toolkit 各不相同，
  当前公开 Environment 已明确 Driver API 语义，本轮须核对精确路由事实，不能误称驱动包版本。
- 正常 consumer：source C++ add_subdirectory 与 installed static/shared SDK 的普通 CUDA Direct／Graph，
  依赖 DLL 邻 exe；正交 consumer：CPU+CUDA 同 SDK，CPU 原能力／公共语义不被 GPU route 配置偷偷裁剪。
- scope：标准 preset／高级明确裁剪、必要依赖／架构／软件身份投影、offline、SDK 搬移、真实 loader、三交付。
- non-goals：不改数学、不变 CUDA Queue／Completion／storage owner；不改公共 staging／统计／selector 核心，
  不加入 CPU fallback、模型读取、全 Provider 排序或用户 timing。
- 直接依赖／消费者：既有 CUDA system SDK、ggml core、cuBLAS／原版 Provider、两 route façade；
  公共 package seam 只按现 selector 选择验证，真实 core／格式资格随 U3 同候选复用。
- tier：P0；本包不签任何 operator 胜负，不跑 P1。
- 调查／纠正预算：一次配置／export／loader 审计，一次软件事实盘点，稳定候选一次三交付批次；
  至多一批具名纠正，不为文档／state 重建，不重复 CPU 完成轮。
- stop：B1–B4 及适用验证全部完成再 CANDIDATE／关闭；子项完成不代签整包。
  新 SDK／driver 事实缺失明确 unknown／拒绝特化，不允许伪版本或后台竞速。

## 有限 FIX

| ID | 实际问题／已有能力 | 允许动作 | 验收 |
|---|---|---|---|
| B1 | compute-cuda 普通 preset 缺席，CUDA-only flags 分散 | 增一个 configure/build preset：Compute+Graph、CUDA ON、CPU／Apple／Vulkan OFF，tests/bench OFF、Release、build j2；高级 mixed 只覆盖 CPU=ON | CMake preset 解析与普通 configure、真实 selected products／Provider；无用户内部宏 |
| B2 | 固定 arch 与 toolkit/runtime/Driver API／真实驱动发行版事实分别核对；默认 source SDK 支持域不能由单 pin 名推测 | 先盘点冷事实／版本来源；若确需新增 backend 冷事实，单独 SCOPED→AUDITED 冻结字段／加载／unknown 合同后实施 | exact 软件/硬件事实、未知/不匹配资格、冷绑定，提交无探测 |
| B3 | HyleCudaSdk runtime leaf 与邻 exe staging 已存在，旧三交付不是本候选证明 | KEEP；统一自然 consumer recipe，检查 CUDART/cuBLAS/cuBLASLt、HYLE DLL、SDK relocation/offline、reduced PATH；不复制系统 driver DLL | Windows source/static/shared、Linux 三交付、actual loader／关闭错误，无弹窗 |
| B4 | CPU-only 与 mixed 的 ggml core 不同但有现存同 core CPU IQ1 port；新增 CUDA 端口不得使 CPU 静默退化 | U3 前固定选定原 core/patch 软件身份与同进程 ABI；只有真实问题才重开 runtime，公开能力投影一致 | CUDA-only／mixed、CPU 已准入格式消费差异、provider/pin/license/export/header |

## 公开用法与寿命冻结

用户选择 `cmake --preset compute-cuda` 或 Windows 既有 wrapper 同名 preset，再 build 同名 preset；
系统 C++／CUDA SDK 由用户安装，缺必需 SDK configure 失败。CPU+CUDA 通过显式 CPU=ON，
不让用户排列内部 Provider。系统目标 arch 可高级覆盖，性能资格按实际编入程序与机器／版本，不冒称通用最优。
普通两入口不变；权重／resource 活至 Completion 终态、独立 invocation 管理 workspace，既有失败语义保留。
应用 source/installed consumer 继续调用 `hyle_stage_runtime_dependencies(app)`；Windows 将所需 DLL 邻 exe。

## 预声明验证与七问

B1 的低风险 preset 编辑先用 CMake CLI 解析；稳定候选后 source configure／实际产品构建／consumer
随 B3／U3 集中执行，不为每一 flag 重建。B2 具体公共头风险冻结后单列；B3/B4 公共 package seam 按 selector。
每轮 docs／closure／diff；registry 变化给 previous-ref、状态表由脚本生成。未运行产品项 UNVERIFIED。

七问：版本／license 用既有唯一 lock；共同目录驱动完整能力；复用 backend／Graph／staging；P0 冷热账单；
无新 FFI；三交付与实际软件身份同源；用户只选路线／正常资源。七门最终须逐项引用实际收据，当前不签 PASS。

## B1 子批 AUDITED：只有普通 preset

SCOPED 卡已独立提交 `27aaef0c`，当前 compute_route_cuda 依法从旧 AUDITED record 重定界至本卡。
只授权 B1 的 preset 与普通用法；B2–B4 仍待具体 owner／接口审计，不能用本子项签整包完成。
旧 F41 入口／数学／资格保持，本轮全目标两入口与软件／交付验收仍须覆盖实际现有能力。

源码已确认：现有普通 `HYLE_COMPONENTS=compute;graph` 与四 route flags 能组装 CUDA-only，
CUDA OFF 跳过探测；cuBLAS、ggml、原 SGLang／DeepSeek／CCCL 由 façade 组装，FlashInfer 只 Linux。
系统 Toolkit13.3+ 必需，既有 backend 拥有默认 arch80＋89-real；不向普通用户加入 Provider 选项、
Toolkit 路径、内部宏、fastmath、全局 CUDA flags 或用户 performance profile。
新 preset 只把已存在选项固定为自然 CUDA-only 组合，mixed 可显式覆盖 CPU=ON。

本子批预声明：CMake list-presets／JSON 实际投影、docs／closure／diff；实际 configure/build 与
source/static/shared loader 随 B3/U3 稳定候选集中运行，未运行保持 UNVERIFIED。此低风险编辑不新增镜像测试。

## B4 架构传播子批 SCOPED

B1 产品候选 `f1937b272c60d29e4027e262feeb6942767b4fcf` 的实际普通 preset configure 通过，
Toolkit13.3.73／MSVC19.50，CPU／Apple／Vulkan／tests／bench 均 OFF，CUDA ON。
但 ggml 日志实际选了 `75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real`，
没有继承 backend 的 `80;89-real`。日志与 receipt 保留在 `.scratch/validation/cuda-final-u2/preset-configure/`。
backend/provider 子目录里的普通变量默认不传播给 ggml 所在兄弟目录，导致同一路线没有统一目标架构，
并扩大原版 CUDA 源文件的编译组合。用户指定架构才避开此问题，是具体构建接缝反例。

compute_backend_cuda／compute_ggml_runtime 由这项公开交付／编译范围变化依法 CLOSED→SCOPED，
绑定本卡**仅 B4 架构传播子批**，不重开 CPU 数学／资格、不借此改 resource overlay 或其它 GPU core。
预算一次源账单、一次 default／explicit／CUDA-OFF 参数化配置检查；运行资格及三交付集中后验。
允许方案待 AUDITED 冻结：选中 CUDA SDK 的配置入口向共同父作用域传播现有 `80;89-real` 默认，
保留用户 `CMAKE_CUDA_ARCHITECTURES`／`CUDAARCHS` 选择；CUDA OFF 不设置、不探测。

### B4 架构传播 AUDITED

SCOPED 提交 `bb55d8b7`；源账单确认 `hyle_configure_cuda_dependency()` 在根目录、全部 src 子目录之前调用。
因此仅在它成功发现 CUDA SDK、且无显式 CMake arch／CUDAARCHS 时，把原 backend 默认
`80;89-real` 传至调用方作用域；两兄弟 native target 与原 ggml 都继承该值。
无需改 ggml 数学／patch、target 依赖／新公开对象、CPU 构建或其它路线。显式值和 CUDA-OFF 早返保留。
有限 FIX 是这一个 SDK 配置接缝；三个局部默认防御保留，不要求普通用户多传架构 flag。
当前只允许 B4 此项产品变更，B2 的真实驱动发行版 facts／B3 三交付与其它 B4 core 能力仍未验收。

B4 同一源码审计的具名纠正：ggml 在 enable_language(CUDA) **之前**设置自己的 arch，
所以不能只在存在 CUDAARCHS 时放弃默认；必须将该显式环境值同样传至共同父作用域。
若 CMake 变量已显式定义仍优先，不改 empty/invalid 值的 CMake 诊断。追加一次环境值／显式变量优先级
配置检查；不是新的能力或性能姿势。

## 当前子批结果

B1 产品 `f1937b27`、B4 default 产品 `76929103`、显式环境优先级产品 `7217b109`。
Windows 的普通配置、生成架构／显式覆盖／OFF 检查已有精确候选收据，
见 [U2 配置证据](../../../../evidence/correctness/cuda-final-u2/README.md)。没有新增构建／运行／交付／性能 PASS。
默认 GGML/native 编译集合得到统一，CPU 数学及闭合证据未改；B2／B3／完整 B4 仍须本包实际验收。
选中的公共 package/头/产品 gates 在稳定接线候选集中执行，不能把 CMake configure 成功当产品完成。
所有 Compute+Graph 实现仍由统一 LOC 脚本计数；本子批仅改构建输入，不增加自研算子。

## B2 冷软件事实子批 SCOPED

baseline `8cbfef5c347eff30bf171cf225bbd585876cfe9f`。公开 CUDA Environment 已正确注明
driver_cuda_version 是 API 号，但内部 RouteFacts 把它投影到 kDriverVersion，缺少真实驱动发行版。
这是用户要求按驱动发行版组合路由的具体缺口；不把它改称 Toolkit 或用户手填 timing/profile。

有限范围：backend 冷快照增加独立原始 driver release 文本；复用 Platform DynLib 动态读取系统 NVML，
创建 Context 时读取、完成后释放原库引用，准备／提交无额外 NVML 查询。系统 NVML 不打包／复制，
不要求普通用户启用新 feature；缺库／符号／权限／初始化／查询失败只缺该可选事实，不禁用 CUDA 正常路径。
kDriverVersion 只发布真实 release 文本，CUDA API 整数改由 kApiVersion 发布，runtime/toolkit 仍独立。
原 private RouteFacts 改为拥有数组＋实际有效 span，缺 release 时省略该 fact，不能用空 string 或伪数字污染 subject。
三个实际消费者为 ggml CUDA、CCCL 与 Linux FlashInfer，只迁移冷事实入口，不改数学／资格阈值／热程序。

正常纵切为普通 CUDA Context/Direct/Graph 的软件适用域绑定，正交为无 NVML 仍能正常匹配 SM/runtime 条件。
寿命：snapshot owned/copyable，调用原 NVML 的初始化引用与 shutdown 平衡，不保留库 handle 或全局可变 cache。
版本查询使用 NVIDIA 官方 nvmlSystemGetDriverVersion/Init_v2 的现有合同，
见 [System Queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlSystemQueries.html) 与本机 Toolkit13.3 nvml.h。

owner：compute_backend_cuda 已 AUDITED 但先重新定界 B2；compute_provider_ggml_cuda 的旧 enum 保护卡不授权该迁移，
本卡先 SCOPED 重新定界。compute_provider_cuda 当前旧 FP4 卡同样先定界冷事实入口；
compute_provider_flashinfer 由 backend private API 变化触发 CLOSED→SCOPED，只冷事实消费，不升级 pin。
不重开 Compute／CPU／Graph 公共机制。直接使用 DynLib 的 backend 在中央图显式列 Platform private dep，
其闭包已由 PlatformThread 取得，不引入新的产品 feature。

预算：一次原 SDK／库入口核对，一批产品／消费者迁移，一批实际 backend/sparse matcher/Provider 两入口
及 source/static/shared/headers 验证，与 U3 稳定接线候选合并重型构建；至多一次具名纠正，P0，无此项 P1。
预声明：原 NVML 差分身份、snapshot copy/move/close、缺 release 的稀疏匹配、API/release 不混淆、
selected headers/backend/provider/普通消费者及三交付；未运行 UNVERIFIED，不以查询探针签产品 PASS。

### B2 AUDITED：独立 release/API 与未知事实

SCOPED 提交 `fa22b275`。本机 Windows 系统 NVML 与 WSL 原 driver shim 的 init/query/shutdown
实际返回0，release 都为 `617.14`；CUDA API13040仍是另一个事实。原 SDK 文档确认查询是线程安全，
init_v2 不初始化 GPU 设备，初始化引用按 shutdown 配对。本探针只证明原 carrier，不签 HYLE 入口。
固定接口是 Environment 的80字节 owned release 文本，空表示未知；不解析成伪 CUDA 编码、不增加全局 cache。
Windows 只加载 System32 的系统 NVML，临时关闭当前线程的 loader 弹窗后恢复；Linux 用系统 soname。
固定有限 FIX：一个 backend 冷读取 helper、snapshot 字段、私有有效 span 投影、三个当前消费者访问迁移、
中央已有 Platform 直接依赖声明与 public guide。NVML不链接为必需 DLL、不复制 driver 文件。
已有数学、SM/runtime/toolkit 资格与错误/资源程序保持；缺 NVML 不形成 CUDA Context 错误。
补具名测试覆盖稀疏缺 release、API/release 区分与实际 snapshot 差分，不写镜像实现的测试。

B2 产品 `ffeb224ed`，namespace 纠正后的批次 `45a1d4a8` 已通过 Windows backend 与原 Provider／普通
两入口接线验证。实际快照与原NVML均为617.14；API整数独立，未知release不阻断硬件/API匹配。
原主批的fixture SKIP已只补缺项，未重跑主批。完整[批次证据](../../../../evidence/correctness/cuda-final-u23/README.md)
保留失败与补验边界；本包三交付／Linux／全部能力仍待验，不提前CLOSED。

### 当前联合交付子格

最终U3产品与test-only纠正后的Windows CUDA-only static SDK已完成实际install／prefix搬移、
offline独立configure／link／运行及邻exe DLL。原完整651／649需求逐名无缺项，
边界及复用见[联合资格](../../../../evidence/correctness/cuda-final-u3-windows-static/README.md)。
source普通links与具名纠正／admission亦通过；shared、CPU+CUDA mixed及Linux shipping仍未完，
B3／完整B4不因本子格提前关闭，不新增性能结论。

shared实际构建检出公共邻置copy的同目录writer冲突，独立
[tool task](../build/runtime-staging-directory-writer.md)固定有限目录锁修复；
不是CUDA算子缺口，不在本模块改公共治理，原对象与通过组保留。
