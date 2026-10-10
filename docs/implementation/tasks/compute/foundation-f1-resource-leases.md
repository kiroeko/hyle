# F1：外部资源租约与最后使用后回收

> 状态：SCOPED（2026-10-11）；实施 goal 已建立，尚未写本轮产品代码/建立证书。
> component / owner：拟由 compute 负责共同控制；compute_backend_cpu/cuda 与
> compute_route_cpu/cuda/apple/vulkan 负责实际 native 适配；Graph 及其四路线和 CPU+GPU 适配只改 leased 接缝。
> 具体改到的 owner 才在启动时绑定；不重开全部 Provider/算子模块。

## 当前启动绑定

- 实际启动 HEAD / previous registry ref：`833e6844b8e0160ec3eadd3b1945aed577cdad84`，工作树干净。
  其产品代码与设计基线 edb271438 相同，保留后续文档/证据提交，不回退。
- 当前绑定 owner：`compute`、`compute_backend_cuda`，均由 CLOSED 合法进入 SCOPED。
  trigger 为用户新增的外部 lease 完整旅程与 `poll_slot→Completion::settle/destroy` 的具体源码适配疑点。
- 先做 F1-04 源协议/自然接口 AUDIT 并冻结有限实现；其他 native/Graph owner 当前继续 CLOSED，
  真正需要修改时先绑定本卡、SCOPED/AUDITED，不从其历史卡恢复旧施工。
- 未启动构建、故障注入、数学运行或性能；结果目前只有源码疑点，不写 FAIL/PASS。

## 共同执行合同

- 基线：`edb2714384037b12a3d1e69839aa895fea17cec1`；设计见
  [资源与流送设计](../../../architecture/compute-resource-streaming-design.md)。
- 本卡已作为 compute/compute_backend_cuda 的当前 SCOPED record；其他 owner 仍按 registry。
  本轮产品验证未运行。已绑定实际 HEAD/工作树，保留后续/用户修改；完成 AUDITED 后才改产品。
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

## 需求与已审计边界

实际反例：vLLM scheduler 2758–2805 在请求结束/CoW copy 后按 fence 延迟释放块；
SGLang l2_transfer 103–129 对 layout/index 保活；当前 HYLE BufferLease 在析构时归还，
TensorView 不拥有外部 payload，Graph Results 只保护其原声明 owner。
准入的是“外部范围引用及 native use 归零后回收”，不是 cache/sequence manager。

正常任务：取得块→共享读或独占写→绑定真实 Direct/Graph→提交→释放业务 handle→
完成后回池；旧结果仍持有时不能覆盖；取消/错误仍有正确的实际退役路线。

## API / usability freeze

- 共同 lease 值只表达实际 backing/range/generation/读写权及资源 owner。
  typed native factory 接受 CPU/CUDA Buffer、MLX array、Vulkan Tensor 的受控拥有/导入路径；
  不把 MLX/VkBuffer 塞进共同万能资源或复制其 allocator。
- 读租约可共享，写租约 move-only；发布 ready 读结果显式且以已成功完成为前提。
  非重叠块可并行，重叠 root/别名不得通过不同 wrapper 绕过检查。
- 新 lease-aware Direct/Graph binding owner/wrapper 预分配有限 guard 槽，代理原 Invocation。
  bind 自动保留 guard，submit 自动交接完成；plain borrowed API 的对象布局/常態路径不变。
  不采用“给每个原 Invocation 都塞一张资源表”的实现。
- 外部 owner 根/Context/resource/allocator 寿命明确；受控导入不能证明未跟踪别名不并发写，
  不满足独占合同的导入只读或拒绝，不能用空 deleter 假装拥有。
- 默认容量有限，满时明确 ResourceExhausted/Unavailable；retain/release 成本与统计冷查询明确。
  close 停接纳并退役本 owner，不同步/关闭用户所有 Queue 或 device。
- 首轮原生资格先裁决 CUDA unknown observation 的保活疑点；不靠非 OK Status 推断已完成。

## 有限 FIX 集

| FIX | 必须实现/定向修复 | 验收 |
|---|---|---|
| F1-01 | 冷创建 owner、有限块/范围、generation、大小/对齐/metadata budget、拥有与只读导入 | 零/边界/溢出/OOM；旧 generation 不释放新块；真实 native backing |
| F1-02 | shared read/exclusive write、非重叠并行、别名/权限、成功发布 | 多读阻止写/复用，写发布前不能读；共享根只计一次 |
| F1-03 | 使用 token/guard 经 CompletionRetention 与各 native 退役衔接 | business handle/Completion 丢弃仍保活；index/metadata 一同保护 |
| F1-04 | 对 CUDA poll/wait/destroy 的终态分类作定向资格；有实际缺陷才修该 owner/直接消费者 | native query 观察失败而真实 work 未结束的故障注入；不能提前释放；原错误保留 |
| F1-05 | lease-aware Direct 原生提交，真实成功/partial failure/unknown use 退役 | CPU/CUDA/Apple/Vulkan 成熟 MatMul/FFN 等实际数学，不以 copy/mock 代签 |
| F1-06 | lease-aware Graph binding/result 适配，外部输出/输入 alias 保活 | 原 plan 不重编/换 Provider；旧 result 阻止覆盖；四路线及组合边界 |
| F1-07 | 非阻塞 collect、cancel、stop/close/drain、late completion、失效 slot | 取消不等于 GPU 结束；重复关闭/最后 guard/合法线程销毁；容量恢复 |
| F1-08 | 普通入口、指南、P0、三交付与新头/缺席边界 | 自动 route 闭包、DLL、迁址；无热 metadata 增长/无第二 reaper |

## 预声明验证与停止

扩展现有 compute completion、graph cpu_cuda/apple/vulkan contracts 及 native route 用户样例；
保留原数学 reference，新增实际 lease 风险断言。只装饰 native observation/fault call，
不能将 mock 结果当路线资格。至少两独立在途块、两个读者、一写者、旧结果、取消、OOM、部分失败、
未知观察、自然 Direct/Graph 和完整 acquire/use/free；不全量重跑算子目录/P1。

P0 静态账单须列新增 metadata/引用/锁，及旧普通路径不变的证据；
热路径以计数验无 allocation/provider search，不为纳秒差改布局。

完成标准：F1-01至08及全部适用门真实通过，明确原借用 precondition 和未验硬件；
精确报告所有 applicable 未验，不自动继承 WSL Vulkan 例外；按协议形成 terminal。
F2 才提供 bulk copy，F4 才讨论版本历史/自动 CoW，本卡不扩成泛用事务系统。
