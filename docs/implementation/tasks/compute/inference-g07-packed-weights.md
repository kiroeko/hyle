# G07 packed 权重消费与 checked selection

状态：**SCOPED，2026-10-16**。本卡承接完整 G01–G10 goal 中的 G07，不验收 G04/G05，
也不重开 G06 的 M/S/L/D。baseline 为 `a638938b783b6fdae57e83f572aa0427d7188e7f`；
产品输入保持完整 `669121418669b31f6849c74f64362aaf8d6eb919`。
当前施工 owner 仅 `compute_provider_torch_cuda`、`compute_route_apple`，从各自 CLOSED
进入 SCOPED。其它 owner 只作只读复用/消费审计；需要改产品时先明确纳入合法阶段。
**本卡尚未 AUDITED，不授权产品实现或 P1。**

## 任务与输入

用户任务是冷加载一次实际物理编码，保留 resident 权重，连续 prefill/decode、选中行或
expert，再取得可拥有的数值结果并正确关闭。要求区别 packed 字节、冷转换、常驻增量、
每步 temporary 与 checked waits；不是新增一个总名为量化的 API。
来源为[原 G07 审计](../project/inference-single-machine-foundation-audit-2026-10.md#权重格式和selected-expert的消费形状)、
[消费形状计划](../project/inference-consumer-adaptation-plan-2026-10.md)、
[完整性终点](../project/inference-single-machine-completeness-and-exit-2026-10.md)和
[Qwen 实物范围](../project/qwen35-2b-command-consumer-scope-2026-10.md)。
初查身份和源码在[输入证据](../../../../evidence/architecture/inference-g07-packed-weight-intake/receipt.json)。

主纵切为本地 Qwen3.5-2B 的 dense FFN/完整词表角色，以及原 D5 同模型 GGUF 画像的实际
物理类型；模型 BF16 原始权重保持只读。量化真实 BF16 片段时明确写为本轮 cold transform，
不能称为下载的 GGUF 原始 payload。正交纵切为原具名 MoE 的两 expert/多 slot 选择、重复/
重排 IDs 与旧输出，及独立非模型数值矩阵。片段资格不冒充完整模型生成或 token/s。

固定 peer 仍为 SGLang `7224fce1b5e488c9916f20ec0556d696e595d512`、vLLM
`98f721116227f0f48690deb05c815c057bac0668`、llama.cpp
`cbe28bb3cb0758f0c566a9a3561766821aeb0748`，读取原归档并验证 SHA，不刷新 checkout。
SGLang/vLLM 的 load-time repack 与 apply 分界、llama 原生 packed matmul/selected matmul
用于反证承载方式；它们不能证明任意 MLX bias、F32 scale、舍入合同可以直接交给 Marlin。
同时审受锁定 MLX、ggml、ATen 的真实算法/执行入口与许可，避免复制已有数学。

## 有限能力族与审计问题

| 族 | 冻结的初查范围 | 当前可定位事实 | AUDITED 前必须裁决 |
|---|---|---|---|
| W1 驻留 packed 线性/行读取 | GGUF Q4_K、Q6_K、Q8_0；Qwen 原 BF16 为控制 | 原 CPU/CUDA/Vulkan ggml 与 Apple ggml-Metal 已有承载；Qwen Q4 画像的 tied vocabulary 实为 Q6_K，不能按文件名当成 Q4 | 逐角色 public descriptor→原生执行→旧结果/关闭，核实际不整库展开；已有完整路线应 KEEP/ROUTED，不能以 API 数量或旧功能测试数代签成本 |
| W2 非 GGUF 原生消费 | MLX affine4/group64 的 linear/grouped/embedding；已有 block128 FP8 dense 的 Apple/CUDA 对照 | Apple affine 已用 native quantized_matmul/gather_qmm；CUDA MLX 每 K group 展开，Apple block_dot 解码权重再 matmul | 按真实 dtype/scale/舍入与设备能力选择原生 packed 或一次冷 repack；常驻与 warm temporary 分开，不能改成静默完整 dense 展开后宣布 packed 完成 |
| W3 checked selection | host-known 和 device-produced I32 IDs；重复/重排/边界/越界 | CUDA MLX `any().item<bool>()` 在 selected 消费前做 GPU→host 控制；Apple/Vulkan 已有 safe IDs 与 device status | 借原受审 checked region/status，保持 OutOfRange、失败输出与已接纳 suffix；不得删除检查、信任任意 IDs 或另写调度/完成 owner |
| W4 冷准备与持续组合 | artifact→ready resident→至少64步→旧结果→drain；Direct/Graph、单/多请求与三联合路线 | 原 F1–F3、WeightSlots、constant/region 和 Queue/Completion 已有生命周期 | 回映每次转换/上传、固定/改变 shape、两个 slots、OOM/partial/close；明确 engine 的模型格式、缓存/expert 政策边界 |

这四族是本轮有限审计集合。MXFP4/MXFP8、NF4/AWQ/CT 等现有 public 入口仍作反向能力/
费用映射：共享实现受改动影响时验证原域；未选编码不能写成新 PASS，也不自动取这些库
全部 API 的并集。真正改变上述选定域或公开架构决定时，先回本卡定界，不边实现边扩。
CPU 没有原生 ISA、native 不支持的精度组合与其它旧拒绝域保留具名限制，不能用 CPU
fallback 冒充 GPU 或把一条新消费链扩大成所有模型/编码资格。

## owner 与正常公开旅程

入口从现有 `TensorDesc`/low_precision 物理编码、`model_ops` 或普通 MatMul/Gather 开始：
资源与权重 ready → `Context::prepare`/Graph compile → invocation bind/submit → checked
completion → owning output → close/drain。引擎拥有文件格式解析、权重角色、expert/cache
政策；route/Provider 拥有通用编码承载、执行、错误和实际退休；Graph 只组合原机制。

- `compute_provider_torch_cuda`：W2/W3 的实际 CUDA adapter；当前 reopen trigger 是具名
  repeated-decode/selected 纵切暴露的逐组临时值和额外 IDs 控制等待。复用原 CUDA Queue、
  ATen stream guard 与已接纳前缀保活，不制造第二个 runtime。
- `compute_route_apple`：W2 的实际 MLX lowering、W4 的 native 消费；trigger 是同一纵切中
  block128 权重重复解码，原 G06 关闭范围不包含该权重任务。已有 affine/GGUF 路线优先反证。
- 原 `compute_provider_ggml`、`compute_provider_ggml_cuda`、`compute_provider_ggml_metal`、
  `compute_provider_ggml_vulkan`/`compute_route_vulkan`、CPU route、WeightSlots、F1/F2/F3、
  Graph 和三个 hybrid 暂为 KEEP/BORROW/资格输入，不因此改动或重开。
- `compute_ops` 与公共头当前合同先只读；若确需新的冷 repack/checked 形状，先给自然 C++
  草图、原子分解、默认值、失败、生命周期和成本，再合法纳入 owner，禁止只在样例造包装。

不考虑旧签名兼容。新 API/布局若有必要，直接迁移真实消费者；不加双轨 cache、备用 planner
或 benchmark-only 入口。常量与可变输入分开，borrowed 输入/allocator 保活到实际退休；
不准按地址/shape 偷缓存可能改变的权重或 IDs。

## 审计、实现和验证边界

先用一轮至多4h的源码/自然调用审计固定 KEEP/FIX/BORROW/REJECT、完整 API 和有限 FIX，
中间最多一个检查点。实现前逐 owner AUDITED；发现需变更系统性决定时停止实现并回审。
之后批量实现，批量运行本轮选定检查，再针对实际失败修复。验证超时沿用户常设授权有限
追加同配置时间，保原失败和累计费用；不靠换目录重置预算。不作新的全库缺口调查。

预声明如下；AUDITED 时把涉及的实际 target、命令、shape 与同合同 peer 固定到本卡，
未执行项保持 UNPROVEN，不能先写 PASS：

1. 正常/边界/独立 oracle：真实 FFN/词表角色与 MoE holdout，prefill M1/17/64、Q1 decode
   至少64步，selected IDs 的重复/重排/负数/上界/改变；empty、dtype/packing/group/overflow。
2. 成本：cold repack 次数/字节和 owner、resident 物理字节、warm temporary/分配/上传/
   host waits；无整库 warm 解码，无通过隐藏完整 dense 常量绕过 packed 目标。
3. 生命周期与失败：预算/allocator OOM、已接纳 prefix、failed status 不发布可用结果、旧
   output、两个独立 slots、共享 backing、重绑、close/drain；复用原 F1–F3 无变输入证据。
4. source/static/shared 与独立公开头：只扩受影响路线、factor 和真实直接消费者；原 CPU/
   Vulkan/GGUF 完整输入匹配可复用。新增 public/package seam 才扩大对应闭包。
5. 适用宿主 ASan/UBSan/TSan 与实际 CUDA/Metal 风险；明确宿主模型、driver/shader/private
   UNKNOWN，不能把 host sanitizers 当设备内存证明。Apple 只用实查 MacBook Air。
6. P0 必须覆盖四族。若 FIX 更换实际数学/常态 native 执行路径，对受影响路线固定最多两项
   同合同完整 warm workload；P1 用现有仪器、5%耗时门、n15/必要n31，先语义/费用账、
   route-local A/A 与双顺序资格，再一次完整候选验证。所有 shape/peer/精度/所有权先在
   AUDITED 冻结；无合格合同或仪器就保 INCONCLUSIVE，不无限扩建工具或搜索 tile。

Windows 是产品真相；普通并行最多4 workers，重型 Windows/WSL/Air 串行，保已有缓存和
构建树。CUDA 原 pool1/split3/Ninja2/SM89 合同不变。构建只选真实影响，不全局 rebuild。
本卡不准恢复任何旧 G04/G05 P1、profile 或旧 CUDA 交付启动脚本。

七门、七问、逐公开族覆盖、独立阶段红队和统一 LOC 在候选后一次聚合；已过证据不因
文档修改重跑。固定命令仍为 selector、`check-module-closure.py --previous-ref <改前提交>`、
生成 status、`check-docs.py`、`git diff --check`。预算/停止条件到达时必须按协议明确裁决；
现阶段的下一步是完成上述有限能力/承载与 API 审计，不能把 SCOPED 当产品授权。
