# Compute 资源租约、批量搬运与文件管线设计

> 2026-10-11，设计与任务准备；尚未实现，不替代当前公开 API/平台资格。
> 用户要求充分调研成熟推理引擎，确定完整常用范围和有限施工卡，再建立实施 goal。
> 基线 `edb2714384037b12a3d1e69839aa895fea17cec1`；产品状态仍以 registry 为准。

## 1. 范围与三个工作包

本轮确定 F1/F2/F3 的通用机制闭包，不复制模型 cache 或推理引擎。
正常旅程必须包含取得、共享/使用、更新、完成/失败、取消、回收与关闭，不能只有 upload 构造入口。

| 包 | 范围 | 任务卡 |
|---|---|---|
| F1 | 外部资源块/范围租约、读写权限、原生使用保活、最后使用后回收与 Direct/Graph 接缝 | [资源租约](../implementation/tasks/compute/foundation-f1-resource-leases.md) |
| F2 | 冷准备的有限范围批量搬运、独立批次/显式相依阶段、原生完成与低成本 replay | [批量搬运](../implementation/tasks/compute/foundation-f2-batched-transfers.md) |
| F3 | 有界文件读取/资源写入，以及资源回写文件；分组 ready、背压、取消/排空与格式适配样例 | [文件管线](../implementation/tasks/compute/foundation-f3-file-resource-streaming.md) |

F1/F2 先合并审计接缝并批量接线，分别保留验收 owner；F3 复用前两项。
三个包各自完成四路线正常入口、生命周期、适用平台与 source/static/shared，不留到总终审首次接线。
F4 状态分支、F5 paged/ragged 数学 region、完整引擎、通用增量图不混入本次施工。

## 2. 深化调研依据与承载裁决

三个参考仓库沿用刚更新的固定版本，未再次追逐 upstream，也不升级 HYLE 产品锁：
SGLang `234ccba7f625e12639e621044496bcc6197c3758`；
vLLM `276fbcff2717bd934cfa37c8a2e4c391f3e7237b`；
llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d`。
本轮是上次[缺口调研](../implementation/tasks/project/inference-foundation-gap-research-2026-10.md)后的
定向补读，重点核实相反方向、部分失败、元数据寿命与关闭，不重新全库扫描。
14 个 peer 文件/12 个 HYLE 实现与合同、原生规范及本地模型头部身份见
[范围研究收据](../../evidence/architecture/compute-resource-streaming-design/scope-research.json)。

| 来源/实际源码 | 常用需求与反例 | HYLE 裁决 |
|---|---|---|
| vLLM block_pool / single_type_kv_cache_manager / scheduler 2758 行起 | 共享、CoW 两端保活、已结束请求仍须等 fence 才释放；refcount 为零不等于 GPU 完成 | SELF 少量组合 glue；复用 Mem 范围/池和原生完成，不借 prefix hash/LRU/sequence manager |
| vLLM cpu/gpu_worker 的 CopyPlan、transfer_async、get_finished/shutdown | 不同 CPU/device 粒度、fragment mapping、双向 copy、临时指针数组、失败排空；顺序队列也不等于批内复制有序 | BORROW 原生复制；HYLE 管有限计划、校验与生命周期，不移植 PyTorch worker |
| vLLM cache_kernels.cu 110–194 | cuGetProcAddress/API 版本与 legacy stream 限制；旧 driver 用原 async copy fallback；ANY 只适合无并发 producer 的 source | 能力解析移到 HYLE 冷准备，默认保持 producer stream ordering；不每批探测 SDK/driver |
| SGLang l2_transfer 103–198 | producer event 后才能读索引，layout move gate，临时索引 record_stream；逐层 ready 可让前面开始消费 | 库支持有限 group/tag 的逐组 ready 与 index/backing 保活；layer/model 解释留给引擎 |
| SGLang hybrid_cache_controller 1191–1247 / cache_controller 489 | 放弃 prefetch 后 ready 到来才能释放；stop 不等于线程已退出，join 失败不能销毁其引用 | 接纳停止、在途退休、旧租约和重复关闭分开；复用 Sched，不新建私有后台线程 |
| SGLang weight_utils 951、1327 / vLLM weight_utils 1283、1378 | bounded shard window 与 stop；成熟 loader 返回的 view 未必离开 context 后仍有效 | 按 bytes/slots 管硬预算并持有 source owner；格式解析留在成熟适配层 |
| llama-model-loader 1509、1690–1760 / ggml-backend 1808 | 对齐 chunk、等待旧 upload 后复用、mmap backing、双向/同设备 copy 与最后排空；fallback 可能整张量暂存 | 不将整 tensor/shard 隐式搬到 RAM；固定 staging，真实依赖与原生同步，报告 copy/host wait |
| vLLM tiering/fs/io | 范围越界、short read/write、临时文件原子发布属于已有 IO/上层职责 | F3 支持双向 byte range；原子发布复用 IO，持久化格式/文件名策略不下沉 |

这些成熟引擎不是可直接链接的小型 C++ 机制库；将其整体依赖搬进 HYLE 会引入模型策略和第二套 runtime。
承载优先顺序为：已有 HYLE 原子→原 CUDA/Metal/Vulkan/CPU 复制与 IO→必要的有限 C++ 生命周期 glue。
不新增自研复制/数学 kernel；有明确 native 能力缺口才另行定界，不能把此卡当算子扩张授权。

## 3. Owner 与构建方向

- F1：共同资源控制由既有 `compute` 拥有，四路线保留 native backing/retirement。
  Direct lease-aware 提交适配归其 Compute owner，Graph lease-aware binding/result 适配归 Graph owner。
  Mem 不引入 Compute/Event/Queue，原 borrow API 不添加无关常驻控制状态。
- F2：共同范围值/校验/成本报告归 `compute`；CPU 与 CUDA/Apple/Vulkan 各拥有自己的 prepared copy program。
  使用既有 native Queue/command/stream；不建立万能 executor 或逐 kernel vtable。
- F3：拟新增内部 compiled factor `compute_stream`，公开入口仍属于 Compute。
  它依赖 `compute`、`io_core`、`sched`、`mem`，native factory/adapter 按实际 public/private 类型确定边。
  Direct Compute 核不反向依赖这个复合层，不创建第二个 IO loop/Scheduler。
- 普通 Compute/Graph SDK 的所选路线闭包包含这些已实施能力，仍只用 `HYLE_COMPUTE_ROUTES`。
  不新增 lease/batch/stream/provider 开关，不让用户手选内部 factor。
  包含 pipeline 的 SDK 增加 IO/Sched 库依赖是交付成本，不代表创建 Context 就启动这些服务。
  CUDA-only 不拉 CPU 数学 Provider；文件读取/host 准备不算 CPU 数学。
- 这里只固定设计。启动实施时先由中央声明登记新 factor QUEUED，再绑定当前 owner 的新记录，
  按 SCOPED→AUDITED 固定有限 FIX；本轮不改 CMake/registry、不制造空 target。

## 4. F1：租约与最后使用

### 必须覆盖

1. 拥有 backing 或有明确外部 owner 保活的受控导入；有限块/范围和 generation，size/alignment/OOM 有错误。
2. 可复制只读租约；独占写租约；读者/在途使用存在时不能取得写权限或复用对应区间。
   不同互不重叠块允许并发；同 backing 的别名按实际物理范围判断，不能只比较两个 wrapper 地址。
3. 成功写完后显式发布只读 ready 版本；失败/未完成内容不得发布成有效结果。
   不实现版本历史/任意 rollback/自动 CoW；需要复制时调用 F2 后发布新块。
4. 提交前检查 backing/范围/desc 与租约匹配，真实接受后交给原 completion/retirement owner 保活。
   payload、指针/索引 metadata、生产者依赖均活到真实最后使用；没有接受工作时允许即时撤销。
5. Direct 与 Graph 的自然 lease-aware 入口自动持有资源。使用独立 opt-in binding owner/wrapper，
   不改变普通借用 TensorView 的布局，不要求用户在每个 token 手写留存 Completion 的 vector。
   Graph 输出若别名外部输入，结果 lease 必须覆盖相应 backing；保持结果不能释放它。
6. 原生完成且引用/使用归零后才回池；有限 pending slots，非阻塞 collect、停止接纳与显式 drain。
   RAII 释放不隐藏全设备 wait。资源根/Context 等原生 owner 必须有明确的 retained 或 borrowed 合同。
7. 取消只是停止后继/请求协作取消，不能当作 GPU 已停。原错误、unknown observation、device lost 分开。
   无法证明完成的使用不得释放/复用；在原 runtime 保持 pending/quarantine 并如实报告，禁止无限重试。

### 新发现的定向资格疑点

CUDA `cuda_backend.cc::poll_slot/wait_slot/completion_destroy` 将成功/NotReady 之外的观察错误直接返回，
共同 `Completion::settle` 会终结 attachment；Vulkan 则已有 nonterminal observation error 的 drain/保活分支。
这是与 F1“未知观察不能当完成”要求的具体适配疑点，尚未注入复现，不写现有产品 FAIL。
F1 必须先裁决哪些 CUDA 错误可证明终止；不能仅凭 Status 非 OK 释放新 lease。
若证明现有基础缺陷，只修实际 owner 和直接消费者，不扩成全库 completion 重审。
回收不能在禁止 CUDA API 的 driver callback 中调用 cudaFree 等；沿已有 host retirement/progress 执行。

### 成本与并发

容量/metadata 冷分配，热路径不增长；共享租约/在途 use 的引用操作显式计账。
控制 owner 线程兼容；只读租约复制/释放跨线程能力须服从 native 最终销毁合同，不能伪称全部 thread-safe。
不支持未纳入 lease 协议的外部别名并发写；受控导入不能猜测第三方私有 allocator/别名。

## 5. F2：有限批次、范围与依赖

- 冷准备固定资源域、最大 copies/groups/总 bytes/metadata 与 native copy 路线；绑定阶段只更新范围/IDs。
  host-known IDs 本批 MUST；GPU-produced IDs 可由已有成熟 native region 处理，本批不新增读回/索引 kernel。
- 一批独立范围允许重复读 source，拒绝重叠 destination 和任意 source→destination 依赖；exact self-copy
  与空范围可消为 no-op。有依赖的复制用显式多阶段，各阶段有合法排序，不默默当 memcpy/memmove。
- 检查加法/乘法溢出、资源/设备身份、导入 alias、大小、对齐、native usage、host pin/可见性。
  bounds/全批校验在接受任何 native work 前完成；运行期 native failure 仍可能部分接受。
- F1 guards 同时保护 source、destination、descriptor/index buffers 与 producer dependency。
  每个 native 阶段一次完成边界，不逐范围 create Invocation/Completion/event。
- CPU 用受审普通 byte copy；CUDA 冷资格选择 native batch 或同 stream 原 memcpy commands；
  Metal 一次原生 blit pass/command；Vulkan 按 buffer pair 分组 VkBufferCopy，在一次合法 command/submit 中完成。
  VkBuffer 的 usage/queue family/cache flush 与 Metal 原资源可见性不因共用内存消失。
- 新批次默认不改变已有 Provider winner；SDK/driver 能力不支持 native batch 时使用已声明同路线原生 fallback，
  不是失败后换 CPU/Provider。fallback 的 host calls/copy/staging 与完成边界报告实际值。
- raw byte copy 保持精确物理字节，不解量化。导出 typed tensor 前另校验完整编码块/desc。
  native tail 对齐若需 padding，只有已拥有足够 physical capacity 才能扩大；不覆盖邻块。
  受控自有 backing 可冷分配 padding，未满足的外部资源在接受前拒绝，不新增隐藏 kernel。
- partial acceptance 后保留已提交 prefix 和全部可能被引用的控制数据；失败不伪装成功完成，未知退役不复用。
  reset/rebind/replay 有明确 pending/failed slot 限制。

原生依据：[CUDA batch copy](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)
没有批内顺序保证，source access order 不能绕开实际 producer；
[Vulkan copy](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyBuffer.html)
要求来源与目标内存不交叠及完整合法 usage；
[Metal blit copy](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy%28from%3Asourceoffset%3Ato%3Adestinationoffset%3Asize%3A%29)
有 macOS 对齐和 overlap 合同。实施核对实际锁定 SDK；不能拿 HYLE 旧 slot 的保守对齐替代原生规范。

## 6. F3：文件/资源双向管线

### 读入与回写

- 输入是已有 File/Mapping owner、文件范围、目标资源范围与可选有限 group/tag；格式名不是必要参数。
- 支持只读映射源和 positional buffered read。direct IO 是显式选择，满足原 IO alignment/capability，
  不静默 fallback；文件尾/short read/offset 溢出不得吞掉。不得读取整 shard/tensor 作为隐式临时值。
- 读入：read→stage→native copy→group ready→计算消费→最后 use 回收。
  回写：producer 完成/设备排序→resource read lease→D2H 或共享内存正确可见→write_all_at→完成/回收。
  不解释 KV/model 状态，不创建文件名、自动淘汰、替换原模型文件或猜持久化策略。
- 支持 whole-request completion 与逐 group ready，早完成 group 可消费；后面失败不把全部组写 PASS，
  已成功结果按自己的 lease 保持有效。group/tag 无 layer/token 含义。
- 最多 K 个 staging/in-flight slots、有限请求/控制记录、共享 byte budget，旧 ready/use leases 仍占额度。
  默认有界背压：容量满明确 Unavailable 或 await permit；不自动丢请求/淘汰用户数据。
- Pipeline 生命周期：create/prepare→admit→ready/consume→停止接纳→cancel/drain/close；
  重复 close、销毁未消费 ticket、只取消一项、late completion 与部分成功均有明确定义。
- 复用 Sched/IO、F1/F2/native retirement；没有第二套 thread pool/poller/全局 cache。
  epoll/kqueue 文件 blocking fallback 无可靠逐操作 cancel 时，外层取消停止后续接纳并等已开始操作退休，
  不宣称 timeout 能中断 pread/pwrite。取消和原生错误不以 HTTP/status retry 自动恢复。
- 报告 logical/staging/native/metadata/unknown/excluded bytes、实际 copies/repack/host waits、峰值 slots、
  partial byte progress、first error 和真实 group-ready。不是吞吐/SOTA 证书。

### 格式解析边界

GGUF/Safetensors 等由引擎或独立 consumer adapter 借成熟解析器承载。
解析器输出 name→file range、shape/dtype/实际 encoding、source lifetime；HYLE 校验范围/描述/资源限制并搬运。
模型配置、权重角色、tensor 名称标准化、tokenizer、混合量化规则不进入 HYLE 文件管线。
GGUF 可借现有 ggml 格式核；本地原始 Safetensors 样例由官方工具生成 checked range manifest，再交 C++ consumer。
不自写正式格式 parser、不因样例把整个 llama.cpp/vLLM/SGLang 作为产品依赖。

本地 Qwen3.5-2B 的实际 safetensors header 为 76,648 bytes，payload 基址 76,656；
第 0 层 linear_attn norm 为 F32[128]，MLP 三个投影为 BF16，不能把整个文件标成统一 BF16。
验证可消费这些实际范围/有限矩阵片段及确定输入，不必加载 4.55 GB 整模型；
完整 BF16 FFN 三矩阵也仅约 72 MiB。范围/切片依据由 consumer 明确提供，不能冒充整模型推理。

## 7. 准入完整性与明确负空间

三个包对推理及渲染资产/分块数值两个纵切分别覆盖：
空/单/多范围、读共享/独占写、in-flight/旧结果、OOM、跨 group ready、短读/写、partial acceptance、
取消、未知观察错误、有效 terminal failure、close/drain、byte 精确保留和 native 数学消费。
不能以 mock provider、设备枚举或只做 copy 代签完整适配；故障注入只装饰真实 native call。

不实现：模型 loader/tokenizer、prefix/专家 cache policy、自动 CoW/任意快照回滚、batch scheduler、
模型级 auto placement、多 GPU/分布式、HIP、GDS/cuFile、GPU ID compaction kernel、新算子目录。
已存在的 cold plan/shape variants、ResourceBudget、IO 原子发布和 Mem 分配器不重写。

## 8. 实施与验证节奏

- 三张卡固定有限 FIX，先集中完成可独立审阅的产品接线，API/算法稳定后冻结候选再集中验证。
- 共享宿主流程/相同 fixture 批量编译运行；三条 GPU 保持原生资源和执行，不逐边调核。
- 新能力以 P0 成本/生命周期为工程门：计数证明无热分配、每批完成/提交数量、有限拷贝及预算。
  不宣称性能赢家；需要原生路线性能裁决时另固定一个同合同完整旅程和有界 P1，不为本轮空跑算子全集。
- 真实四路线 source/static/shared，SDK 迁址、新头 ON/OFF、依赖/DLL 同目录按 selected seam 一次验收。
  CPU 三平台；CUDA Windows/WSL；Apple Air；Vulkan Windows/Air，WSL native GPU 资格按实际设备记录，
  旧 accepted UNQUALIFIED 不自动代签新能力。无合格 GPU时列未验事实及具体决策，不假 PASS。
- build1/test1，重型运行串行；启动前确认内存余量，沿已有 cgroup/job 限额，禁止为此新增庞大验证管理器。
- 同产品树、合同、工具链/机器、fixture/adapter 身份匹配的原 Provider 数学证据可复用；新 lifetime/transfer
  接缝必须新验。文档/状态变化不构建、不重跑 P1，局部失败只重验影响部分。
- 完成基础机制后停止；F4/F5 和完整推理引擎另定界。

本轮设计与三卡共固定 23 项有限 FIX（F1 8、F2 7、F3 8）；没有产品/API/组件图/依赖锁/registry 修改。
只执行 docs/closure 与 selector，构建和性能均未运行。Compute+Graph 实现与
[前轮统一 LOC 收据](../../evidence/architecture/inference-foundation-gap-research/loc-current.json)
同一产品内容：68,204 行/301 文件；全库非测试实现163,431行/652文件。
这些数字不包含新文档，不表示本轮机制已经实现。
