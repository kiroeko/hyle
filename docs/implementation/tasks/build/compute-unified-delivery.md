# C1–C3 / R1：统一 Compute/Graph 配置与交付

<!-- MODULE REVIEW: SCOPED -->

- baseline：`0a167159e08bf2df021c0b7f7fc8942a4dc87c3e`；Windows canonical，启动工作树干净。
- 用户已明确启动实施；[已定设计](../../../architecture/compute-next-stage.md)与
  [入口审计](compute-entry-design.md)是本轮输入，不恢复已关闭旧任务。
- owner：compute、graph、compute_backend_cpu、compute_backend_cuda、compute_provider_native、
  compute_provider_onednn、compute_ggml_runtime；compute_route_cpu/cuda/apple/vulkan、
  graph_route_cpu/cuda/apple/vulkan。共享 build/package seam 没有独立 registry 身份。
- reopen trigger：用户新增唯一路线集合/完整自动 Provider、统一运行资产、预编译 recipe 和
  后续跨路线支持的公开交付需求；E1/E2/E3/E4 的具名入口反例已在上轮源码审计确认。

## 正常旅程与边界

普通 C++ consumer 只选产品和路线集合，源码/静态/动态取得同一完整能力。
主纵切：目标模型 Direct/Graph 的 CPU/GPU 单路线及 mixed SDK；正交纵切：模型无关 tensor 批处理。
依赖继续使用中央 component DAG、依赖 lock/cache、package registry 与 native adapters。
保留已验数学、原生 Context/Program/Queue/资源/完成 owner；源码引导和 runtime 资产不增第二套实现。

C1 先统一构建输入，C2 统一 SDK/资产，R1 校准 identity/冷诊断，接线结束后 C3 一个稳定候选批验。
H1/H2/H3 联合计算须另卡绑定实际 owner；本卡不删除 same-Device guard，不签 mixed 执行。
增量依赖图继续 DEFER。SIMD、模型 loader/engine、算子扩张、非阻塞组合服务、公共治理/计时核心都不改。

## C1 有限范围（先审计，未授权写产品）

| ID | 源码问题/真实旅程 | 允许 FIX | 验收 |
|---|---|---|---|
| C1-01 | 六公开能力开关重复且可裁掉完整 CPU | 一个 route-set 解析器；平台合法性；ordinary 旧参数明确迁移错误；必要 Provider 正常变量派生 | 默认/单项/组合/空/未知/重复/平台错配/旧 cache/父工程输入 |
| C1-02 | 未选 GPU 仍在静态产品根，靠 OPTIONAL_ABSENT 收尾；CPU 默认独立 | 从中央声明过滤实际所选 route 根并取闭包；维护 exact factor 与 TEST_DEPS 不混入用户产品 | 单路线不 probe/fetch/link 未选 GPU/CPU 数学；语义 metadata projection 不受配置依赖 |
| C1-03 | Apple/Vulkan 无对称普通 preset | 四 preset 同一规则，旧 Provider AUTO/六 route knobs 撤出普通 preset | preset metadata、同 parser，低并行，无普通 lite mode |
| C1-04 | 现行 build fixture/selector 参数仍使用旧公开选项 | 仅迁移消费者配置输入/对应合同断言；内部隔离用既有 DEV_FACTORS；不改仪器/selector 算法 | 必要消费 fixture、路由关闭/缺件/未知诊断与工具合同 |

API/usability freeze：`HYLE_COMPUTE_ROUTES=cpu|cuda|apple|vulkan` 或显式集合，顺序无 preference；
省略 Compute/Graph 产品时不探计算依赖，省略路线时完整 CPU；显式空/未知/AUTO 拒绝，大小写归一并去重。
normal full CPU 的 Torch 必须满足已锁 OS/ABI；oneDNN 仅在已准入 x86 平台取得，ARM 不强开。
旧 CMake 开关拒绝，target 派生的 presence 宏保留；cache/root/archive 仅为获取方式。
硬件 CUDA 编译目标/Toolkit/driver 分开；选择新路线不继承不同软件 recipe 的 exact WIN。

## C2/R1 范围（尚待各自有限审计）

C2：复用 generic/Apple helpers，提供一个应用 staging 入口；现 package registry 投影路线/recipe；
installed 请求只验证实际 SDK 子集，不能重编或伪宏；DLL exe 同路径、metallib/Unix loader 及搬迁。
R1：保留 matcher/profile/all-port 语义、现有 mature defaults；检查实际 recipe 与 native facts/冷诊断，
submit 无新增搜索/设备查询/计时；未知环境保持合法默认，缺真实能力拒绝，不偷偷 CPU fallback。
每子包审计后才能固定其产品 FIX；C1 的 AUDITED 不自动授权 C2/R1 新源码/API。

## 预算、验证和停止

- 审计一遍当前入口/源码和现有 fixture；一个配置边界最多一次定向修正，不翻历史性能队列。
- C1 接线只跑无重型 build 的解析/配置合同、docs/closure/diff；数学未改不跑 P1。
- C1/C2/R1 完成后冻结不可变候选；C3 批量四单路线与三 CPU+GPU SDK 的适用平台 source/static/shared、
  public headers、runtime assets、selected consumers、迁移/缺 SDK/离线/OFF 隔离。
- 现有数学/资源证据仅在 recipe/tree/合同身份相同时复用；变化域才跑直接消费者，不能称新数学 PASS。
- 所选 shared-seam selector superset 保留；非相关模块无产品/边界影响项逐项说明，不扩成全库数学重验。
- build 2 / test 1；重型验证串行；WSL/MacBook Air 精确验证副本，不反向同步；只维护本轮有界产物。
- 完成 C1/C2/R1 和 C3 适用门后 CANDIDATE→CLOSED；预算耗尽或实际条件不足按具体恢复条件 BLOCKED。
  适用未运行保持未验，聚合门 INCONCLUSIVE；不能用文档/空 target 关闭。

## 七问与当前裁决

第三方身份/许可证与算法不升级，沿现 lock/registry；完整性是公开配置/交付的新需求；
复用现有图/依赖/资产 owner；P0 只在冷 CMake/准备路径，既有 hot 成本不变；没有 C/FFI 新入口；
三交付在 C3 批量实测；user/developer 当前用法随实际产品迁移，不提前称可用。
当前为 SCOPED，C1-01–04 待 AUDITED，产品改动/新验收为零。
