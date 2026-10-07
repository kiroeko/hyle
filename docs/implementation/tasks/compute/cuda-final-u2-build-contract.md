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
