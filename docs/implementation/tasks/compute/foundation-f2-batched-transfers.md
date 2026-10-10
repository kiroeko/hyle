# F2：准备后的原生批量范围搬运

> 状态：READY_FOR_SCOPE；依赖 F1 的 guard/retirement 合同，不复制另一套 lease。
> component / owner：拟 compute、compute_backend_cpu/cuda、
> compute_route_cpu/cuda/apple/vulkan；只有普通 Graph 需要本批 native binding 接缝时再绑定对应 owner。

## 共同执行合同

- 基线：`edb2714384037b12a3d1e69839aa895fea17cec1`；设计见
  [资源与流送设计](../../../architecture/compute-resource-streaming-design.md)。
- 本卡为准备完成的有限施工合同，尚未作为 registry 当前 record；当前 owner 仍 CLOSED，验证均未运行。
  实施 goal 启动时先绑定实际 HEAD/工作树，保留后续/用户修改；按协议登记/SCOPED/AUDITED 后才改产品。
- 主纵切：量化推理的外部状态/权重范围及真实成熟数学消费。
  正交纵切：渲染资产流送和分块矩阵管线；不加入 token/layer/KV policy。
- 复用 Mem/Util/Sched/IO/Compute/native 完成，不增加复制 kernel、数学算子、模型 loader 或第二套 runtime。
- 全部产品接线集中后冻结候选，批量执行适用检查；小修只运行影响门，同身份证据复用。
- 四路线资源/程序/完成独立。Windows 为唯一产品真相；CPU Win/WSL/Air，CUDA Win/WSL，
  Apple Air，Vulkan Win/Air；WSL Vulkan 按实际 GPU 资格，不以旧例外代签。
- 三交付：source/installed static/shared 普通 Direct+Graph 消费、SDK 迁址、ON/OFF 新头、路线缺席诊断、
  Windows DLL 同目录。不让普通用户添加内部 factor/provider/stream 开关。
- 默认 P0：有界资源、热分配/提交/完成计数与成本结构；没有 timed 性能赢家承诺。
  出现需性能裁决的具体反例才在稳定候选上按原统一仪器固定一次同合同有预算 P1，
  未资格/未运行如实记，不能无限换姿势或升级仪器。
- build1/test1、重型串行、启动前确认内存余量；复用源码缓存/SDK 构建树，完成后只清本批临时产物。
- 每轮 selected gates、`check-module-closure.py`、`check-docs.py`；
  registry 变化带 `--previous-ref`。文档/状态阶段不构建/P1。提交推送用 `[skip ci]`。
- 调研纠正预算：至多一个具体反例的一次补读、两次有限修复/验收；新公共语义或边界变化必须重新定界。
  所有 FIX/适用门完成则 terminal；不能把研究/文档/旧 CLOSED 计作本轮产品完成。

## 需求与完整能力

实际来源：vLLM cpu/gpu_worker 的 CopyPlan 143–185、transfer_async 534、
cache_kernels.cu 110–194；SGLang l2_transfer 的 producer/index/layout 依赖。
复制任务不是只接受“所有整 tensor 具有相同尺寸”：
host/device 粒度不同、部分首尾块、多个 backing、有限 index/range 更新和两个方向均需表达。

本批 MUST：CPU host copy，CUDA H2D/D2H/同设备 D2D，Apple 原资源 CPU/GPU 可见性与 blit，
Vulkan staging/device 双向及同设备合法 copy。无跨 GPU/peer/multi-GPU；不为 GPU 动态 IDs 新写 kernel。

## API / usability freeze

- 一次 prepare 固定 resource endpoints、capacity/metadata/groups/native version 与 copy 路线。
  每批绑定 host-known 源/目标范围或固定 catalogue IDs；不动态发现资源/Provider，不做时间比赛。
- source 可被多次读；destination 不能交叠；源写依赖不得在 independent batch 中隐式排序。
  exact self-copy/no-op 与空批处理清楚；相依搬运由显式 phase 与 completion 依赖表达。
- 保持原编码 bytes，description/物理容量/alias 分开校验；typed binding 不能用半编码块冒充 tensor。
  需要 native padding 时必须明确已分配范围，不能越过邻块。
- 一阶段一个真实 Completion，保留 F1 guards 和指针/index metadata；
  CPU 同步完成也遵循完整生命周期，不伪装 GPU 异步。native partial acceptance 不丢 prefix。
- CUDA batch 与 fallback 在冷准备按 SDK/driver/stream 能力固定。
  STREAM 是 producer 依赖默认；ANY 仅在严格已证 host source 条件采用，不按方向名称猜测。
  Metal/Vulkan 各用原 command/queue，原有 Vulkan family/可见性 owner 保留。
- 单组/分组可分别返回 ready；不保证物理重叠或全同 buffer pair 能一条 native command，
  但禁止每 copy 重建 plan/event/Completion。报告真实 native calls/groups/waits/staging。

## 有限 FIX 集

| FIX | 必须实现 | 验收 |
|---|---|---|
| F2-01 | CopyBatch/phase 的冷准备、固定资源域/metadata/byte budget 和范围绑定 | 稳定 replay 改 IDs/ranges，满容量/零批/溢出/OOM，无重新搜索 |
| F2-02 | 全批 alias/跨 range 独立性、native usage/方向/对齐检查，self-copy 消除 | 共享 source 允许；重复/交叠写及 A→B→C 同批拒绝；phase 链正确 |
| F2-03 | CPU/CUDA native 批量与冷资格 fallback | 真 byte equality、原生使用计数、严格 producer ordering；不热 cuGetProcAddress |
| F2-04 | Apple 原生 batch/blit/ready，与共享内存 CPU 可见性 | 不隐式绕过同步/把 host 写当 GPU 已完成，pending 显式 |
| F2-05 | Vulkan pair grouping、command/fence、family/cache 依赖与明确 tail | 一个提交 prefix；legacy/imported storage 拒绝/合法 fallback；不复制 native runtime |
| F2-06 | F1 guard 交接、partial acceptance、取消/丢弃/close、slot 退休与再绑定 | 第一项已接受后第二项失败，旧 bytes/index 不重用；真实未知退役路径 |
| F2-07 | Direct/Graph 数学消费、逐组 ready、P0/三交付/文档 | 至少二组二槽，成熟数学消费复制后的真实量化/BF16数据，原规则不变 |

## 预声明验证与停止

有限矩阵覆盖 R=0/1/多范围、同/不同 buffer pair、单/多 phase、重复 source、坏 ID、overlap、
host/device 原编码、aligned/tail、pending 与 retained result。
每路线对一次 staged upload、一次 D2H/read 与 D2D/host 合法变体作数学消费；故障只装饰真实 native calls。
复用现有 native transfer/math fixture，不建立新 benchmark 军团。

P0：prepare allocation/budget 与 bind 校验 O(R log R) 或更好（列出实际算法），
submit O(R+G) 的 native commands/一次 phase completion；热元数据容量固定。
native batch 缺席 fallback O(R) native calls 是明确成本，不虚报单 driver call或性能赢家。
若需优化/计时，先源码/同完成账单，再一份不可变候选、一项完整旅程、有界门；不无限 GPU A/A。

本卡完成后即可供 F3 使用；不等 F3 才首次验证三交付/四路线。
运行期 GPU-produced索引、repack kernel、模型 cache 规则不在本卡。
