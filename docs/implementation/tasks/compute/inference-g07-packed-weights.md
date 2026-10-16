# G07 packed 权重消费与 checked selection

状态：**2026-10-16，CUDA CANDIDATE；Apple按合格M64回退退回AUDITED；G07–G10 goal ACTIVE**。G01–G06已告一段落，
本卡不重开它们。原baseline为 `a638938b783b6fdae57e83f572aa0427d7188e7f`；
产品输入保持完整 `669121418669b31f6849c74f64362aaf8d6eb919`。
当前施工 owner 仅 `compute_provider_torch_cuda`、`compute_route_apple`，从各自 CLOSED
进入 SCOPED。其它 owner 只作只读复用/消费审计；需要改产品时先明确纳入合法阶段。
**只授权末节冻结的有限FIX和对应验证；P1仍须产品冻结及普通门通过后启动。**

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

## 新goal续跑检查点

用户在G04/G05收口后明确要求创建新goal并开始剩余四项，现已ACTIVE。续跑文档baseline
为`6d8f4395d09ba0d96e2c48f26c9a212a8c920f3b`，原产品与W1–W4集合不变，原调查费用/
预算及失败不重置。[首个检查点](../../../../evidence/architecture/inference-g07-packed-weight-audit/checkpoint-1.json)
核9个intake源码SHA仍匹配，并补4个实际测试/构建/Apple执行锚点；尚未冻结产品FIX。

当前先裁决CUDA逐group临时值与IDs checked wait的成熟承载。现有MLX affine测试包含
非整数bias/scale关系，不能默认无损改装为整数zero-point的AWQ/Marlin。Apple affine
已有quantized_matmul/gather_qmm，应先证明复用；block128的decode位于建图路径，仍须核
explicit constant和runtime input的实际求值边界，不能只凭源码出现decode就认定每步重复。
不按地址缓存可变输入，不新增测试专用公共入口，不因启动新goal重复旧全局缺口调查。

## AUDITED：原生承载、公开合同与有限FIX

[审计记录与受锁定源码](../../../../evidence/architecture/inference-g07-packed-weight-audit/audit.json)
保存实际归档/三peer文件SHA、承载裁决、有限FIX与未运行门；不是新的runtime PASS。

本轮实际核了锁定MLX `2b2bb49790158ecd008d1685c94e3ef119792a00` 的归档SHA，CPU
原公式与CUDA `qmv.cu` 的直接packed行读取/warp归约；未刷新peer checkout。SGLang
固定awq_linear的load后处理/apply分离和vLLM固定auto_awq的冷repack证明准备边界，
不把其整数zero-point公式当任意MLX affine bias的替代。实际Air为Mac16,12、arm64、
macOS27.0.1；已验MLX提供`fast::metal_kernel`及Safe编译模式，不新建Metal runtime。

| FIX | 冻结实现与归属 | 不改变的合同 |
|---|---|---|
| W-CUDA | `compute_provider_torch_cuda`：借原MLX packed qmv行映射/warp归约及已移植的CPU affine/MX公式，在私有CUDA TU中直接读U32 codes、原scale/bias，F32计算；覆盖现有linear/grouped/affine embedding。去掉逐K组ATen展开、matmul/加法temporary与selected `item()`控制等待 | 保所有既有F16/BF16/F32、mixed affine提升、group32/64/128、原MX先dot后scale及E8M0→输出dtype边界；embedding保其乘/加各次原dtype舍入；不无损假定AWQ转换 |
| W-CHECK | 同owner：复用`cuda::detail::acquire_device_status(Queue&)`及原Completion协议，先检查全部I32 IDs，再让native计算在非零status时不读取/写入payload | 越界仍OutOfRange、输出不发布且原内容不改，accepted prefix/旧结果/suffix/close与实际退休不变；没有新status owner、host-known信任捷径或D2H前置等待 |
| W-APPLE | `compute_route_apple`：block128 FP8 dense/原FP8 grouped消费借MLX原SIMDgroup矩阵乘和Safe custom-kernel编译，tile内解FP8并F32累加；每128元素块严格先乘activation scale、再weight scale、再顺序加到F32输出 | 直接读原strided packed数组，禁止完整FP32权重/dots temporary或每步完整格式repack；原FP8量化器、有效区/输出cast、checked IDs/原native生命周期保持；GroupedFloat4保原独立路径 |
| W-QUAL | 两owner及实际consumer测试/文档：补至少64步自然调用、修改输入与IDs、两slots/旧输出/失败及数值反例；既有W1 GGUF路径优先用实际ROUTED与费用证据验收 | W1 Q4_K/Q6_K/Q8_0、BF16控制、四native/三joint不取消；不重写已CLOSED owner，不新增loader、格式registry、通用GEMM或scheduler |

CUDA数据来自本次绑定的borrowed TensorView，每次读真实内容，不缓存地址；prepared只存
不可变形状/类型，invocation没有展开权重bank。linear一条native数学kernel；grouped/
embedding增加一次全IDs checked kernel，Queue自身reset/report按原协议计费。Mlx路径不再
进入ATen准备/提交，ArtifactWeightDecode和Softplus仍走原ATen carrier并保其未知私有分配。
同一个provider按operation报告runtime/cost：MLX不阻塞host、无repack/copy/临时展开workspace；
不新增external capture或strict-budget资格，不改原准备失败/借用寿命合同。

Apple继续由MLX拥有原stream、pipeline、数组分配和完成。私有tile取16×16输出/32元素
staging，F32 SIMDgroup乘加，128元素一组；不通过FP16窄化中间值获取收益，不搜索tile。
框架编译metadata与JIT为冷费用；真实每步codes/scales/quantize/status/输出分配和native
等待照实收费。API签名/物理编码不变，公开头只纠正选型与费用说明，无兼容转发。

### 冻结验证和性能对照

- source：`hyle_compute_provider_torch_cuda_test`的原失败/9精度组合及新增native packed
  tests；现有`hyle_compute_route_cuda_test`/`hyle_graph_route_cuda_test`中受影响的MLX
  N2/model片段和checked失败；Apple `hyle_compute_route_apple_test`及原FP8/block128、
  affine/FP4反偏置、Graph和CPU+Apple实际直接consumer。未变W1其它路线按精确输入复用。
- 数值：独立解码/高精度oracle，mixed affine、非整数bias、多个K组、F32消去与原TF32
  foreign-policy测试；FP8全部码值/符号/零/subnormal与原scale顺序、极端E8M0、非整tile
  M/N、grouped重复/重排/负/上界IDs、失败后原输出不变；原阈值不临时放宽。
- 设备/生命周期：受影响CUDA测试先memcheck/racecheck，Apple单独Metal shader validation；
  原分配/OOM/accepted prefix/两个slots/旧结果/关闭与适用ASan/UBSan/TSan宿主facet。
  shader/driver和private upper的未知保留；只跑本卡实际适用目标，不制造全局san矩阵。
- 交付：两owner受影响公开头与源码/迁址static/shared consumer，借原SDK/编译缓存与未变
  vendor对象。公共签名与DAG不变，module-local增加CUDA TU不授权改公共构建系统。
- P0：核无逐group全局float权重temporary、无selected前置host控制等待、不整bank冷/热
  展开；显式统计packed/active/scales/输出、必要status和未知native费用，64步覆盖W4。
- P1仅在正确性、风险和产品不可变后运行一次：CUDA affine4/group64的完整checked
  warm linear调用，M1与M64、K2048/N6144；Apple block128 FP8同合同完整warm调用，
  M1与M64、K2048/N2048。候选与旧不可变669原公开路径分别准备，消费相同真实输入/
  物理权重/scale/输出oracle，包含submit、checked完成、结果取得/释放；冷准备单列。
  该peer是同语义旧实现，不是额外BF16驻留或较低partial精度的替代；不推导token/s。
  用原统一仪器5%耗时门、route-local A/A、双执行/分配顺序与原n15/必要n31，至多这两
  形状/路线；不因换目录重置失败，不扫姿势或改门槛，先核完成边界与分辨率再采样。

普通构建最多4 workers、native CUDA pool1/split3/Ninja2/SM89，Windows/WSL/Air重型
任务串行。先批量实现上述集合，再批量验证并针对真实失败修复；新系统性决定或新能力
必须退回审计。有限FIX、候选后的七门/红队/LOC和原费用/停止规则不变。

### 2026-10-16 批量实现与首批验证

产品主体在`218ea6f02`实现，`8775071d4`修正vendored manifest的大小写敏感路径排序；
后续提交只补测试与用法。当前`aead0206b`的产品字节与已验证`746d498b4`一致。
[源码与风险检查点](../../../../evidence/correctness/inference-g07-packed-weight-source/receipt.json)
绑定产品输入、实际命令、原失败/停止、XML与原始归档；它是部分验收，owner继续AUDITED。

- CUDA shipping普通测试15项通过；另以原N2真实全部行/K尺寸分别通过Direct/Graph各16项。
  新测覆盖64步修改packed words/IDs、两个槽、旧输出、全部FP8码值、embedding逐次舍入和
  affine混合dtype/group32/64/128。Torch原allocator限额为零时仍能运行native packed数学。
  非法IDs保持原输出，Completion为OutOfRange；失败槽按原合同重建。
- Apple Direct及holdout20项、Graph13项通过。新增CPU→Apple FP8→CPU消费者65次执行中
  改变64次输入、冷常量源释放后持续消费、两个槽和旧结果/close通过；实际报告为两个CPU
  片段、一个Apple片段、两次必要host copy、零host repack。
- CUDA memcheck/racecheck各13项通过，零errors/hazards；Air实际日志确认Metal API与GPU
  validation均启用，新FP8及相关原消费者13项通过。两端ASan/UBSan/TSan宿主facet均通过；
  原外部driver/private pool、LSan及NVIDIA diagnostic-thread lifetime未知不据此改写。
- 原65份W1/provider/联合桥源码与669产品字节一致；复用绑定已登记，最终仍须回映原
  ROUTED/P0/生命周期证据，不能把未变源码等同于所有GGUF交叉组合的新64步实测。
- 未变GGML CUDA的146对象按原source/command/dependency身份复用；新核构建没有触发
  这些vendor CUDA重编译。私有精简测试入口的GLOB、include、support TU与binary路径
  错误均保留；早先过宽的759项Apple运行已显式停止，后续只运行预声明相关域。

当前仍缺Windows/MSVC源码裁决、受影响source/static/shared迁址consumer及公开头、W1/路线
最终回映、不可变候选后的有限P1、七问/红队/LOC与终审。尚无速度/5%门、token/s或llama.cpp
结论；G08–G10未开始，不以本检查点宣布G07或goal完成。Windows首轮私有parent缺少
CUDAToolkit imported target作用域，已在同一source/build追加find_package后续跑，原费用保留。

### 候选冻结与剩余执行入口

不可变candidate为`b483b8939e9c1d28a98eee3a6ecd13b3fd45fc88`，两owner进入CANDIDATE。
Windows/MSVC现已通过[公共显式Provider源码检查](../../../../evidence/correctness/inference-g07-packed-weight-source/windows-source.json)：
4项原生packed测试通过。原270步默认路线预览在构建前停止；实际只构建29步、原status核与
新packed核共用一个CUDA编译pool。默认Direct/Graph的完整N2资格仍明确属于前述WSL记录，
不把这次Windows单元检查扩称完整默认路线或迁址SDK资格。

两个[纯公开头消费者](../../../../evidence/correctness/inference-g07-packed-weight-source/public-source.json)
在源码构建通过64步、两个槽和旧结果检查；Apple还验证关闭原Context后的保留array。
它们分别是`tests/compute_provider_torch_cuda/packed_resident_consumer.cc`与
`tests/compute_route_apple/packed_resident_consumer.cc`，不依赖GTest、产品私有头或原库故障注入。
后续直接用同一源码验证WSL CUDA及Air Apple的迁址static/shared SDK、受影响公开头和必要
现有直接consumer；沿原producer/cache更新两owner，未变vendor对象必须有身份匹配复用证明。
源码检查已经通过，不能用它代签两种SDK。W1原资格的最终回映、七门/红队/LOC仍保留。

P1继续只允许每路线两种预声明形状，未采过本轮计时。进一步冻结dtype和自然调用，避免比较
时临时改为更弱的数学或更小的计费边界：

| 路线 | 操作与固定输入 | 形状 | peer与适配边界 |
|---|---|---|---|
| CUDA | `mlx_quantized_matmul(kAffine4,64)`；A/scale/bias均F32、原U32 words、F32输出，包含非整数bias；公共Context/prepare/invocation | M1与M64，K2048/N6144 | 原669的同一provider源码及MLX header，允许私有benchmark仅重命名factory以同时实例化两版；用同一公共Context::create/prepare/submit/Completion，不改原数学、临时张量或TF32保护 |
| Apple | `float8_block_linear(kNone,1e-10)`；BF16 activation、原E4M3FN/F32 block128权重、BF16输出；每次调用包括原动态量化与F32点积/scale/最终cast | M1与M64，K2048/N2048 | 原669的同一下降源码，只为私有reference重命名导出符号；原lowering作为公共Context::compile的region，核对与原prepare的逻辑desc、原生执行、错误和结果生命周期一致。不得把解码或quantize搬到冷期 |

双方warm边界为既有Program/Invocation上的完整submit、原生prefix完成、结果/状态观察及本次
公共临时owner释放；所有数学、转换、每次必要分配/copy/wait计入。准备、首次JIT、输入上传、
Program/Invocation最终销毁和进程级driver/cache清理单列为冷/teardown，不能写成已纳入warm。
CUDA旧invocation保留的最后ATen临时结果按原自然重放周期在下次submit/最终销毁释放；
Apple保留原MLX私有缓存/完成owner。这些原生私有释放事实必须在适配源码预审和账单再次核实，
不能凭一条“full drain”签同合同。若适配不能证明同等warm合同，先修适配或写不可比，不采样。

先完成两侧独立数值/有效区验证、逐阶段CHARGED/DEFERRED账单及route-local A/A、双执行/
分配顺序和原统一仪器资格；随后沿原n15/必要n31与5%门执行，不刷新失败/未知费用。基准只是
两种具名正常调用的工程比较，不推导token/s、完整模型或llama.cpp成绩。发现产品问题退回
AUDITED；不得在CANDIDATE下改kernel、公共形状或精度。G08–G10仍未开始。

### 迁址SDK与P1适配源码审计

[四种迁址SDK记录](../../../../evidence/delivery/inference-g07-packed-weights/receipt.json)
现已通过：WSL CUDA及实际Air Apple的static/shared各运行packed、权重迁移、联合执行
三个纯公开消费者，共12次；独立公开头共编译18次。原CUDA shared测试链接失败和Apple
shared缺少已清理旧prefix的setup失败、同配置续跑全部保留；不扩大为Windows SDK资格。

[P1适配源码账单](../../../../evidence/architecture/inference-g07-packed-weight-audit/p1-adapter.json)
已核原669实际ATen/from_blob/NoTF32Guard/逐group数学/输出copy及Apple原prepare→compile、
动态量化、native同步/输出wait。两端旧结果都有跨调用保留：稳态的上次释放归入下一次submit，
最后一份结果及Program/Invocation/native缓存最终销毁明确DEFERRED。独立benchmark只重命名
原导出符号，使用同一公共提交与完成路径；不搬动数学到冷期。此记录只是可比性源码预审，
尚未运行独立数值preflight、A/A或P1，不签速度门。owner仍CANDIDATE。

[W1最终回映](../../../../evidence/architecture/inference-g07-packed-weight-audit/w1-reconciliation.json)
绑定70份与669相同的原生/资源桥源码与20项原记录，继续KEEP/ROUTED原Q4_K/Q6_K/Q8_0
及普通dense BF16控制。它连接原实际角色数值、当前已关闭生命周期合同和本轮受影响SDK，
不冒充新跑的全部GGUF×路线×64矩阵。原GGUF CUDA selected的D2H IDs和stream wait仍
明确计费；原BF16 grouped可能整权重cast的限制保留，不能以新MLX kernel概括消除。

[P0账单](../../../../evidence/architecture/inference-g07-packed-weight-audit/p0.json)
给出原物理权重、activation/scales/输出、CUDA无全局展开、Apple必要scale/quantize和4KiB
threadgroup staging、最终owner销毁及native UNKNOWN；没有把逻辑数组字节当私有allocator
peak。实际两端、两形状独立oracle预检均通过；Apple的原/新实现各8个执行/分配/placement
组合A/A已通过，正式P1仍运行中。CUDA普通数值预检最大绝对误差0；Apple选定精确dyadic
输入的double→BF16 oracle逐bit一致。尚不因此签P1或整项完成。

[独立数值预检](../../../../evidence/correctness/inference-g07-packed-weight-preflight/receipt.json)
保存两端primary/placement、两形状、两实现各三次真实运行和原失败；CUDA最初bench为4aa，
Apple修正后的bench为04c。正式统一资格会在最终bench每个进程重新先验oracle，不能只凭
旧预检给修改后的代码代签。benchmark补include/unused常量及验证副本Git object-store链的
修正没有修改产品。两条native runtime/driver私有池和最终销毁UNKNOWN仍保留。

[统一LOC](../../../../evidence/architecture/inference-g07-packed-weight-audit/loc.json)
以不可变7da快照测量（产品与b483相同）：全库3749352、自实现不含测试180854、两个owner
唯一实现联合6060行/38文件。统一工具的原self-check通过；这些是实现总量，不是本轮新增
或推理引擎待写代码预测。

### Apple首轮P1裁决与有限纠正

[原始合格结果](../../../../evidence/performance/inference-g07-apple-packed-attempt-1/receipt.json)
完整保留：n15×8执行/分配/placement组合、route-local A/A与原controls资格PASS、verify0；
M1吞吐比2.030–2.052，WIN/5%门PASS；M64比0.729–0.732，LOSS/5%门FAIL，用时约1.37倍。
不是地址/顺序翻转或未资格点值。原记录及全库性能总账保留失败，不将M1收益代签M64。

仅`compute_route_apple`退回AUDITED，CUDA产品与其candidate不变。授权一次有限W-APPLE-D1：
原FP8 tile热环为每个码调用通用`ldexp`，同一权重在不同M tile重复解码；先将正常E4M3FN
值转换为等价F32 exponent/mantissa位构造，保零/subnormal/NaN/符号及所有F32矩阵、block128
scale顺序。固定MLX原`fp8.h`也直接用码位转换，不能据其half转换忽略NaN或改变本轮精度。
这是待验证的具体成本机制，尚不声称它解释全部37%回退。不换tile、不扫描形状、不改API、
不移数学到冷期，不引入FP16矩阵或完整float权重shadow。

改后只重验Apple原相关20项、Graph13项、65步CPU+Apple及两个公开消费者/迁址static/shared、
实际Metal validation和受影响宿主facet；CUDA/W1未变输入继续复用。冻结新Apple candidate后
只按同一两形状、原669 peer和相同5%/n15–31合同重新资格；原费用/失败不重置。

CUDA首轮A/A的M1不稳定，未形成合格P1结果；原始资格失败保留，只能修正测量资格，不能
据此改CUDA产品或宣称胜负。原完整数值与SDK/风险证据仍有效。G08–G10尚未开始。

CUDA测量资格仅允许一次harness纠正：每次比较前双方完整自然调用交替warm至少1秒，原16
paired rounds改为现有仪器已有的104轮、4ms epoch；两个固定形状/全部order与placement/
original和selected A/A、n15–31、5%与统计核心不变。首轮A/A全量保留，未成功的首条product
进程因ready=FAIL主动拒绝，不是合格性能数据。新artifact/合同必须重新资格，不能复用失败
A/A，也不改CUDA产品。两轮累计费用保持；不再扫计时姿势。
