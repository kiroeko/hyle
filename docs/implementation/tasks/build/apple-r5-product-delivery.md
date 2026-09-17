# R5 A1/A6：Apple 公开产品与原生资源交付

## Closure card / CANDIDATE

- baseline：`9e41c15aab5a97206e82ebc70a0bee0d916e112f`；产品仍为 `d900a8ce`，后续仅 LOC 工具。
- 当前 owner：`compute`、`graph`、`compute_backend_cpu`、`compute_backend_cuda`、
  `compute_provider_native`、`compute_provider_onednn`、`compute_route_cpu`、`compute_route_cuda`、
  `graph_provider_onednn`、`graph_route_cpu`、`graph_route_cuda`。仅重开公开 roots 与交付闭包；
  两个 Apple owner 的 C++ 继续归 [R5 主卡](../compute/apple-v1-four-route-foundation.md) A1–A6。
- 工具 owner：构建包 registry 的 generated runtime asset 投影与 Apple 专属 staging helper。
  本卡是独立 build 任务，不在 Compute 产品实现中暗改公共工具。
- trigger：普通 `hyle::compute/graph` 尚未获得已经实现的 Apple route；MLX 即使 JIT 仍需要
  构建生成的 `mlx.metallib`。现有 registry 的 runtime assets 只接受锁定许可证，不能拿二进制产物
  冒充 license 或依赖开发机绝对 build 路径。
- 继承：十一 owner 的旧 terminal/candidate/performance 完整保存在
  [baseline](../../../../evidence/correctness/apple-r5-product-delivery/inherited-baseline.json)。
  CPU 例外、CUDA 的限定资格/方向和旧 runtime/算法/布局不变，不重跑它们的 P1。
- 主旅程：Apple ON、CPU/CUDA OFF → 普通 C++ Direct/Graph 的变化输入 Q4 FFN → checked result；
  同一自然代码分别 source、relocated static/shared 消费。正交旅程为 CPU-only、no-backend、
  unrelated core 和 Windows/WSL Apple OFF，保持原 DLL staging 合同。
- 范围：下表 D1–D4、真实资源/头/导出/许可证与普通样例/指南。不给用户新增 Provider 排名，
  不改共同执行器、selector 算法、closure checker 或性能仪器。
- 预算：15 分钟定向源码审计，固定 D1–D4；两轮具名 package seam 纠正，重型串行、parallel 2。
  发现其它基础组件缺陷另行定界，不扩全库重审。
- 验证：两 Mac 真实 source/static/shared Direct/Graph FFN、relocation 时撤去原 build 的 metallib
  fallback、共享 MLX runtime 唯一性、安装头/依赖/许可证、offline source cache；Windows/WSL
  Apple OFF 无 SDK 探测、CPU/无后端/core 裁剪；独立 package-registry 工具回归和 broad selector。
  源码身份相同的证据可依法复用；没运行的 applicable 项保持未证明。
- 性能：本卡 P0，旧 owner C++ 未变与冷交付成本；Apple 完整 P1 仍在主卡的稳定候选上执行。
- stop：有限交付项及适用门成立后 CLOSED；三交付失效、资源路径或公开 roots 改变才重新定界。
  本卡不签 R5 的其余风险/P1，更不签 R6/R7。

## 有限待审项

| ID | 缺口 | 允许动作与验收 |
|---|---|---|
| D1 | 公开产品缺少 Apple 实现 | 两个真实 route roots；OFF 保持 INTERNAL_ONLY_ABSENT，不装 SDK 头 |
| D2 | 生成 metallib 没有真实 package seam | 独立 generated runtime spec，绑定 producer target/build 路径，保留 license 锁校验；Apple helper 负责消费者邻置，shared native 库旁保留资源 |
| D3 | 普通 source/static/shared 消费尚未贯通 | 公开 target 的 Direct/Graph Q4 FFN 样例、变化输入和 oracle；真实 relocation/offline/链接闭包 |
| D4 | 新依赖与用户入口未形成说明 | 四项 license/notice、Apple CLI 构建要求、native 数组/结果/准备成本与 staging 指南 |

SCOPED 只允许审计；D1–D4 固定为 AUDITED FIX 后再改上述产品和 package seam。

## 审计冻结 / D1–D4 FIX

- 两 facade 的现有 root union 正好是上列十一 owner；只有追加两个 Apple roots，旧 C++、规则与
  对象布局不改。CPU filter 和 CUDA OFF 沿现有机制执行。
- MLX `device.cpp` 先从包含 MLX 代码的 binary/dylib 同目录找 `mlx.metallib`，最终才尝试编译时
  build 路径。static 消费者因此需要 EXE 邻置；shared 包需要 native dylib 邻置。资源属于 MLX
  生成制品，不是锁定文本 license，也不引入进程全局 `set_metallib_path`。
- registry 增加独立 `GENERATED_RUNTIME_FILE_SPECS`：producer 必须是该 registry 导出的真实 target，
  source 必须位于 build tree，安装目的地复用已有路径约束；configure 允许尚未生成，install 必须
  真正存在。旧 runtime/license hash 多重集校验完全保留。
- 新 Apple 专属 `hyle_stage_apple_runtime(executable)` 在 source 与 installed SDK 提供同一函数，
  只做显式资源邻置；不改变 Windows DLL helper。package 配置按实际 Apple feature 加载该 helper。
  source 的 shared MLX 旁由 owning factor 放置资源，installed shared 则由中央资产投影放置。
- 按上游安装规则只投影 MLX `.h`（排除上游不发布的 `backend/metal/kernels.h`），Metal-C++ 头与
  四项 license 保留；不安装 vendor `.cpp`/shader 源文件充当 header SDK。公开 framework link 使用
  Apple 原生 framework 名称，不能携带维护机 SDK 绝对路径。
- Direct/Graph 样例均通过公开 facade 使用同一个 affine Q4 FFN 数据与独立 oracle。安装包验证要
  撤去原 build fallback；SDK prefix 整体移动后仍能计算，不能以 configure 或设备枚举代替。

七问：成熟 MLX/Metal-C++ 锁与归属明确；真实计算消费驱动；复用现有图/结果和 package registry；
只有冷交付 P0；纯 C++ 无 FFI；三形态同入口；用户显式 staging 与 native lifetime 文档。上述有限
FIX 已冻结，进入实现；其它 A5/A6 与 R5 最终 P1 留在主卡中执行，不计入本卡完成。

## 已执行检查点（非 terminal）

[Air 三交付与 WSL OFF 证据](../../../../evidence/correctness/apple-r5-product-delivery/README.md)
已归档：真正的 Q4 FFN/RMSNorm、变化输入、完整 source、installed static/shared relocation、
撤去 producer metallib fallback、单 MLX runtime、许可证与导出检查通过。
最终 [Windows broad selector](../../../../evidence/correctness/apple-r5-windows-selector/README.md)
和 [Mini 原生/三交付](../../../../evidence/correctness/apple-r5-final-mini/README.md)现已完成。
D1–D4 不可变交付候选为 `cbedb837791da15392019299f2652d9d3d70ec9d`；其全部 shipping inputs
与 Apple 最终候选 `1636365bbe47779de24e84d7c7d1ed8088fe1af2` 相同，逐路径身份记录随 Mini
证据保存。本卡进入 CANDIDATE 后做七门 terminal 裁决；R5 P1 仍归主卡，不因交付通过提前签字。
