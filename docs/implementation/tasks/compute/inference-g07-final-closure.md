# G07 驻留 packed 权重与 checked selection 有限终审

状态：**G07 ACCEPTED，2026-10-16**。新 G07–G10 goal 完成 **1/4**，原十项累计 **7/10**。
本记录关闭 `compute_provider_torch_cuda` 和 `compute_route_apple` 的本轮有限范围；
G08 输出数值选择、G09 联合流水、G10 CPU 预算继续，完整 Qwen CLI 与 token/s 在其后。

原产品 baseline 为 `669121418669b31f6849c74f64362aaf8d6eb919`；
CUDA C1 candidate 为 `4274ee6cb6275bf36bda1d8c8c0e223f0fa65db5`，
Apple D2 candidate 为 `91ae8e67faa028061d8f5e8e216310c827e024a3`。
[施工卡](inference-g07-packed-weights.md)保留 scope/audit、有限纠正、全部失败和费用；
[最终输入与证据索引](../../../../evidence/correctness/inference-g07-final-qualification/receipt.json)
绑定当前产品文件、原始记录和以下裁决。候选后仅维护测试、文档、状态与证据。

## 范围与正常消费

用户旅程是显式准备/上传原物理编码 → resident 权重 → 普通 Direct 或 Graph prepare →
bind/submit → checked completion → owning output → close/drain。主纵切为 Qwen3.5-2B
FFN/词表实际角色；正交纵切为不同专家、共享 backing、设备生成 IDs 和非模型数值矩阵。
引擎仍拥有文件格式解析、模型角色、expert/cache 政策；没有新增 loader 或 scheduler。

| 有限能力族 / 原公开入口 | 最终处理 | 语义与实际证据 |
|---|---|---|
| W1：GGUF Q4_K/Q6_K/Q8_0 MatMul/Grouped/Gather，BF16 控制 | KEEP/ROUTED；原 CPU/CUDA/Apple/Vulkan native carrier 不变 | [70 文件/20 原记录回映](../../../../evidence/architecture/inference-g07-packed-weight-audit/w1-reconciliation.json)：真实角色 N/K、原生命周期与三个 joint 资源桥；不是新跑全格式×路线×64 矩阵 |
| W2：MLX affine4/group32/64/128、MXFP4/MXFP8 group32 linear/grouped，affine embedding | CUDA 直接读取原 U32/scale/bias；linear M>1 固定四行复用解码；其余保持单行 | [C1 源码/风险](../../../../evidence/correctness/inference-g07-cuda-row-reuse-c1/receipt.json)：24 Provider、16 Direct/16 Graph 实际模型片段、64 步公开消费、Windows 四项；M1/M5 九精度组合及全部 FP8 码值反例 |
| W2：Apple E4M3FN/F32 block128 dense/grouped、动态激活 | 原 MLX owner 内 tile 解码/F32 SIMDgroup；M≤16 用16×16，大 M 用固定64×32×32 | [D2 源码/风险](../../../../evidence/correctness/inference-g07-apple-blocked-d2/receipt.json)：20 Direct、13 Graph、65 步 CPU+Apple 与公开消费、实际 Metal13项；保原量化/scale 顺序/输出类型 |
| W3：host-known/device-produced I32 IDs、checked completion | BORROW 原 Queue/status/退休；CUDA 全 IDs 先检查，再执行 payload kernel | 重复/重排/负数/上界/变更、失败不发布/旧输出不改、accepted prefix/suffix/OOM/close；本轮去掉 MLX 提交前 `.item()` 等待，未改变其它格式的检查成本 |
| W4：resident→持续消费→旧结果→关闭，Direct/Graph、两个 slots 与联合资源桥 | BORROW 原 F1–F3/WeightSlots/Graph/native owner；禁止按地址暗存可变权重 | [Graph 补证](../../../../evidence/correctness/inference-g07-packed-graph-consumers/receipt.json)：两路线各 source/static/shared、M1/17/64 各65步，改变 codes/scales/A、旧输出与销毁；CUDA另验共享backing、两queue、设备TopK→checked grouped失败 |

普通 N2 fixture 的 activation 实际是 **M1、两次 replay**，完整形状指权重 N/K。
终审已纠正此前 M17/M64 的错误推断；补证用 CUDA K128/N9、Apple K256/N19 的正交形状，
完整角色 N/K 与 M64 大形状另由原模型片段/完整 P1 的独立 oracle 证明，不混写成同一用例。
Graph 两个新消费者已登记为正常 source test target，同时以同源码 installed-only 验两种 SDK。

| 横切语义 | 本轮裁决 |
|---|---|
| 数值/边界 | PASS：原 F16/BF16/F32、mixed affine/非整数 bias、原 MX dot 后 scale 与极端 E8M0、FP8码/尾块、原 TF32 foreign policy；没有放宽原门或改成 half partial |
| failure/OOM/别名 | PASS：保原 allocator/Queue 的拒绝与已接纳前缀退休；公共 bind 校验和 Provider `alias_support=false` 在 kernel 前拒绝 input/output 重叠；设备非法 IDs 不读取/写入 payload |
| 寿命/并发/持续输入 | PASS：可变真实输入、重绑、两个 slots/独立队列、共享只读 backing、旧 owning output、关闭与实际退休；没有权重地址缓存或第二个完成 owner |
| 设备与宿主风险 | PASS：C1 memcheck/racecheck各22项、三hostsan各22项；D2 Metal13项、三hostsan各2项；新增Graph实际CUDA两工具零错误/零hazard、Apple API/GPU validation通过 |
| 三交付与公开头 | PASS：下列四份当前SDK合计12消费者、18独立头编译；真实 install→relocate/旧prefix不存在/installed-only，另补两路线三交付Graph；不声称Windows原生Apple资格 |

## 七门与七问

| Gate | Verdict | 依据 |
|---|---|---|
| scope | PASS | W1–W4 有限集合、主/正交消费者、全原入口反向回映与有限 FIX；没有新模型/格式并集 |
| architecture | PASS | 原 Queue/Completion、native stream/worker/array 和 F1–F3 唯一 owner；无新公共API、地址cache、完整dense bank或公共构建重写 |
| correctness | PASS | 上述数值/失败/寿命/并发与实际风险证据，Graph M 行数误读已补真实运行 |
| delivery | PASS | [CUDA C1 SDK](../../../../evidence/delivery/inference-g07-cuda-packed-c1/receipt.json)、[Apple D2 SDK](../../../../evidence/delivery/inference-g07-apple-packed-d2/receipt.json)及source/Graph补证；未变vendor对象按真实输入复用 |
| usability_docs | PASS | [Compute](../../../user/compute.md)、[Apple](../../../user/apple-compute.md)说明物理编码、原生选路、冷/热费用、checked结果和寿命；本轮docs/closure检查 |
| red_team | PASS | 与实现分离的最终自审，非第二审查人；核输入identity、Graph覆盖缺格、别名、隐藏展开/等待、性能端点、SDK/private限制；证据缺格已闭合 |
| engineering_performance | PASS | [P0](../../../../evidence/architecture/inference-g07-packed-weight-audit/p0.json)、[C1](../../../../evidence/architecture/inference-g07-packed-weight-audit/p0-c1.json)、[D2](../../../../evidence/architecture/inference-g07-packed-weight-audit/p0-d2.json)和以下四项合格P1均过原5%门 |

1. **第三方**：固定 SGLang7224fce1、vLLM98f72111、llama.cpp cbe28bb3 的原归档/SHA，
   以及锁定 MLX/ggml/ATen，见[审计](../../../../evidence/architecture/inference-g07-packed-weight-audit/audit.json)。
   CUDA借用的MLX码/许可/NOTICE独立登记；无依赖升级，不把整数zero-point替代任意affine bias。
2. **完整性**：W1–W4按原公开入口和实际消费映射；selected语义、持续变化与三联合资源桥保留。
   新Graph有限补证纠正原结论，不把小形状、源码PASS或格式存在代签整模型/所有路线。
3. **复用**：原资源/错误/退休与MLX/ggml执行设施不复制；仅私有packed数学及原生分块改动。
   CUDA artifact decoder/Softplus仍走ATen，Apple FP4仍沿原独立路径。
4. **性能**：下面均比较不可变669原HYLE的相同公式、物理输入、输出与完成边界；不是llama.cpp，
   也不是kernel-only或token/s。静态源码计数不冒充实际流量、peak或occupancy。
5. **FFI**：原C++23 typed native桥不变，无新增C ABI或跨语言runtime需求。
6. **交付**：四SDK的真实owner artifact绑定candidate；迁址、运行时依赖和公开头实际通过。
   CUDA146个未变native对象仅按真实输入与Ninja旧命令身份复用，不关闭依赖跟踪强行跳编译。
7. **易用**：普通prepare/Graph消费原物理编码，无Provider专用新façade；原生完成/可变输入/
   borrowed寿命/关闭规则明确。模型文件解析与采样政策仍由引擎负责。

## 性能、费用与停止

| 同合同完整 warm 调用，相对原 HYLE669 | 八组合中位吞吐比范围 | 方向 / 原5%门 |
|---|---|---|
| CUDA affine4/group64，M1/K2048/N6144 | 23.897–24.302 | WIN / PASS |
| CUDA 同上，M64 | 1.757–1.825 | WIN / PASS |
| Apple 动态激活+block128 FP8，M1/K2048/N2048 | 2.038–2.060 | WIN / PASS |
| Apple 同上，M64 | 1.227–1.238 | WIN / PASS |

[CUDA最终P1](../../../../evidence/performance/inference-g07-cuda-packed-attempt-3/receipt.json)与
[Apple最终P1](../../../../evidence/performance/inference-g07-apple-packed-attempt-3/receipt.json)
均QUALIFIED、verify0、n15×8，route-local A/A、双执行/分配/placement顺序与controls通过。
各组合CI保留在原始结果，以上范围不是 pooled CI。测量包含完整submit/native前缀完成/
status/result/public owner释放，前次私有retained result在下一次submit释放并收费；最后一次
retained native result、Program/Invocation/进程cache最终析构明确DEFERRED，不宣称完整teardown。
Apple每次原动态量化计费；没有cold-expanded peer或F16/TF32替代。

CUDA原A/A资格失败、第二轮M64合格INCONCLUSIVE/5%FAIL均保留；最终轮1057.570秒，
累计2201.574秒。Apple前两轮M64合格LOSS/FAIL保留；最终565.708秒，累计1694.059秒。
全部合格方向已进入[全库性能总账](../../../implementation/evidence/performance/full-library-scorecard.json)。
Graph消费者初次编译/源码配置失败及同树恢复成本留在补证，不重置为首次成功。

CUDA没有全局decoded权重bank/逐group ATen temporary；四行复用的register/spill/occupancy
仅凭源码未知。Apple大tile scratch12KiB、小tile4KiB，native分配、量化、必要copy/status/
worker等待与最终owner仍计费。W1原GGUF selected的D2H/同步、BF16 grouped可能整库cast、
driver/private allocator上界、LSan与设备/宿主san模型限制均保留。无新增capture/硬预算资格。

[统一LOC](../../../../evidence/architecture/inference-g07-packed-weight-audit/loc-final.json)使用不可变5f83，
self-check PASS：全库3759879、自实现不含测试180904、两owner唯一实现联合6110行/38文件。
产品字节与本记录candidate一致；这是完整实现总量，不是新增行数或引擎待写代码预测。

本轮停止改动及重复P1/SDK/风险。只有可复现的W1–W4正确性/寿命/实际成本/交付缺陷，
新的具名消费者需求、公开合同/依赖变化或合格同合同material回归，才按新卡合法reopen。
G08–G10、完整文本引擎prefill/decode token/s、多GPU/多机不由本记录代签。
