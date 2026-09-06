# 已冻结模块的真实推理消费者回放（2026-09）

<!-- READ-ONLY CROSS-MODULE CONSUMER REPLAY: ACTIVE -->

> 本任务把最新真实推理引擎的消费形状回放到已经 `CLOSED + FROZEN` 的 HYLE 单元。它是发现任务，
> 不是第二套模块状态，也不因“参考库有更多 API”自动授权改产品。只有具名事实满足既有 reopen trigger，
> 对应组件才另开自己的 `REOPENED -> SCOPED -> AUDITED` 产品轮。

## 目标、身份与边界

- HYLE baseline：`840ab249a2f4257ebcff5bafe144e7bc7832bb40`；本任务建立时的 canonical checkout 为
  `C:\public\project-hyle\hyle`；
- owner / start date：Codex / 2026-09-06；
- 目标：逐项确认真实模型加载、prefill、decode、KV 生命周期、continuous batching、取消/关闭和
  serving 观测是否暴露已有冻结单元的能力、自然接口、内部复用、性能 workload 或交付缺口；
- 有新能力时，先通过语义擦除、owner、生命周期和准入门，再尽量在正确模块实现；不能复制模型、tensor、
  token、backend 或服务策略；
- 性能方向由真实热路径和成本结构确定。推理仓库用于确定 workload/provenance，不自动成为同合同 peer；
  计时前必须先冻结接口、语义、工作量、ownership、并发、计费边界和 delivery shape；
- `simd` 已由 v6 对同一四组 consumer 完成逐能力回放，本任务只做 reference drift 差异和直接消费者回放，
  不重跑 422-scope 全矩阵；
- `io` 保持现有 `SCOPED` 且暂停，`compute/graph` 保持 `QUEUED`。三者只作为需求和 owner 反例，不在本
  项目回放中修改；
- Build/package 作为每一波的横向交付合同复查，不伪造一个可链接模块状态。

建立任务时存在的 Container/Mem/Util performance replay、`roadmap.md` 与 `status.md` 未提交改动均为
外部既有工作；它们不属于本任务 baseline，不得被覆盖、吸收或提交。

## 锁定推理证据

2026-09-06 对五个 checkout 执行了 clean-tree 检查和远端 `HEAD` 核对；落后的干净仓库只做
`git pull --ff-only`。本轮发现使用以下完整 identity：

| evidence group | clean commit | role |
|---|---|---|
| standalone ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | CPU/backend 基础机制、model-independent reference |
| llama.cpp | `73a43d1f69345aee8bb186ef4b3172cef892f2e5` | 模型加载、CPU/统一内存推理、采样与单机服务 journey |
| ds4 | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` | 紧凑量化 CPU/统一内存引擎及直接热循环 |
| vLLM | `569adb5a9780f9c02d22a6b29826acf711512356` | batching、KV block/offload、runtime 与 custom-op journey |
| SGLang | `5bebe7a033ab57751fac3c9d8db219da62ca8c72` | serving、MoE/kernel、调度、KV/cache 与观测 journey |

Python、Rust、CUDA、Triton 或 vendor kernel 中的业务对象不会按名字下沉为 C++ 基础 API。它们仍可证明
某个模型无关的 ownership、storage、synchronization、mapping、error 或 observability 机制在真实系统中
重复出现。

## 消费路径抽取合同

每个 finding 必须记录以下字段，缺一项就不能成为产品 trigger：

| field | required evidence |
|---|---|
| exact source | repository commit、文件、symbol/调用链，而不是搜索命中数 |
| task shape | 创建、正常操作、组合、关闭/取消和销毁全过程 |
| contract | ownership、lifetime、failure、concurrency/progress、normal size 和 worst case |
| frequency | 下列 hotness 等级及其证据；不得凭直觉写“热” |
| semantic owner | HYLE factor、standard/mature dependency、Compute/Graph/engine 或明确不做 |
| current route | `COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE` |
| cost/performance | 算法档、allocation/copy/lock/atomic/syscall/dispatch/working set 与自然计费边界 |
| generalization | 去掉模型/token/backend 名称后仍自然，并有第二消费者或成熟 peer 重复 |

热度只使用三档：

- `OBSERVED_HOT`：仓库自带 profile、benchmark、trace 或明确性能注释把它列为关键路径；
- `STRUCTURAL_HOT`：从代码可证明它按 token/layer/block/batch/request 重复，但不伪造耗时百分比；
- `CONTROL_PATH`：启动、配置、模型加载一次性步骤、错误或关闭路径；它仍可因延迟、RSS、可靠性重要，
  但不能用纳秒 primitive 榜替代完整 journey。

四个引擎对同一通用机制的独立实现或绕行优先级高于单个仓库的 feature；代码拓扑可以证明重复频率，
不能替代 profile 证明 materiality。

## 性能裁决与防止苹果对橘子

1. 先做 P0：复杂度、遍历次数、临时存储、allocation/copy、锁/原子、syscall、动态分发和 codegen。
2. 只有 `OBSERVED_HOT/STRUCTURAL_HOT` 且 HYLE route 可能产生 material 差异时才建立 P1；先从真实调用
   抽取版本化输入和自然公开/内部 seam，不围绕内部阈值造 benchmark。
3. P1 每格都必须核对工作量、结果、ownership、failure、thread-safety/progress、冷热状态、线程/亲和、
   ISA、delivery、setup/teardown 和 charged phase。同合同强 peer 最多两个；不等价就写
   `NOT_COMPARABLE`，不得用总平均或 5% margin 把差异抹平。
4. 真实 trace/profile 足以稳定重放完整阶段时才升 P2。P2 不能拿不同模型、量化、backend、batch、KV、
   输出质量或计费边界直接比较 tokens/s。
5. 固定 material loss 只做一次有界机制诊断：能指向算法/实现/API 成本则形成 trigger；只能随地址、顺序、
   code placement 翻转则记录仪器边界，不改公共 API、layout 或常态 metadata。
6. 已冻结 workload 不得因结果难看而改形状。consumer repository 不是 peer 时，只用它锁定 workload；
   peer 必须另行锁定版本和等价合同。

## 分波执行与依赖顺序

每波先完成只读 differential；只有 `PRODUCT_TRIGGER` 才按依赖顺序重开单个组件。上游产品改动完成后，
只回放直接消费者，不批量重开全库。

| wave | factors | inference emphasis | build/package replay |
|---|---|---|---|
| A | core, error, container_intrusive, util_core, time | 热循环编译合同、错误传播、stable-node/queue 链、RNG/resource、clock/deadline | header ownership、source/static/shared 与零/最小依赖 |
| B | timer_queue, platform, platform_thread, platform_fs, mem_reclaim | timeout/cancel、topology/ISA、placement、mapping/read、read-mostly reclamation | factor façade、OS 条件源码、传递系统库 |
| C | mem, mem_vm, obs, math | allocation/pool/quota、large mapping/load-to-ready、structured hot observation、scalar sizing | mimalloc/Tracy/OS/vendor lock、三交付 |
| D | container, util, sched | KV/index/storage shape、registry/runtime utility、continuous batching/wait/wake | 聚合 façade、可选依赖、直接 consumer |
| S | simd delta only | ggml/ds4/vLLM/SGLang 新增 dtype、row/kernel primitive 与 fixed-ISA consumer | Highway/xsimd lock、target flags、无成本路由 |

波内每个 factor 的合法结果只能是：

- `CONFIRMED`：当前 candidate 能完成本轮准入 journey，且性能/接口/交付没有新 trigger；
- `METHOD_GAP`：未发现产品问题，但当前证据或工具不足；产品状态保持不变，另记有限 tool/evidence task；
- `PRODUCT_TRIGGER`：记录精确 consumer、缺陷、owner 和重开条款，随后才更新 registry 并建立独立 closure
  card；不能在本项目页直接施工。

## 预算、交付物与停止条件

- 小 factor differential 默认 45 分钟，中型 90 分钟；一个 finding 最多一次源码机制诊断。超时只允许
  `METHOD_GAP`，不允许无界调参；
- 每波交付：reference identity、consumer-path ledger、能力差异、hot-path map、性能 comparability、
  dependency/reuse/build 复核和逐 factor verdict；
- 每个合法 trigger 单独拥有 FIX 集、candidate、验证、红队、三交付和文档，不共享模糊“全库候选”；
- 项目停止条件：18 个既有 `CLOSED` factor 均取得当前 reference identity 下的一个合法 verdict，所有
  `PRODUCT_TRIGGER` 已转入各自状态机；SIMD drift 已明确为 confirmed 或独立 trigger；
- 全轮仍执行 `python scripts/check-module-closure.py` 与 `python scripts/check-docs.py`。只有实际改变
  `module-status.json` 的子轮才保存 previous registry 并用 `--previous` 校验。

## 执行账本

| wave / factor | exact consumer paths | capability/hot-path result | performance method | verdict / next action |
|---|---|---|---|---|
| references | five clean commits above | remote identities locked | N/A | `COMPLETE` |
| A / core | pending | pending | pending | pending |
| A / error | pending | pending | pending | pending |
| A / container_intrusive | pending | pending | pending | pending |
| A / util_core | pending | pending | pending | pending |
| A / time | pending | pending | pending | pending |
| B | pending | pending | pending | pending |
| C | pending | pending | pending | pending |
| D | pending | pending | pending | pending |
| S / simd drift | pending | pending | pending | pending |

