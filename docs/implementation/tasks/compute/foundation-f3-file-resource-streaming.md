# F3：有界文件与计算资源双向管线

> 状态：READY_FOR_SCOPE；依赖 F1/F2 完成的资源/搬运合同。
> component / owner：拟新内部 factor compute_stream（实施时登记 QUEUED）；
> native factory/adapter 归 compute_route_cpu/cuda/apple/vulkan，有限中央构建修改归 build_package。
> IO/Sched/Mem 只消费既有能力；出现具体基础缺陷才另作定向 owner 修复。

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

## 需求与来源

llama.cpp loader 1509、1690–1760 的 staging/event/尾块与排空，
SGLang weight_utils 951、1327 的 bounded window/stop，l2_transfer 的逐层 ready，
vLLM weight_utils 的成熟 ParallelLoader/source view 寿命，以及 tiering/fs/io 的短读/写与范围检查。
共同机制是显式文件范围↔显式计算资源的有界管线，不是模型加载器。

正常旅程包括只读 source/mapping、分批 ready、读入消费、已有 resource 回写、
部分成功、backpressure、取消和关闭；不能只做一次同步读整 tensor 然后 WeightSlots upload。

## API / usability freeze

- 冷创建 Pipeline：既有 IO/Scheduler、native endpoint、固定 slots/bytes/请求/group/metadata 预算。
  原生 factory 接收真实资源；共同层只编排 IO 生命周期，不执行数学或拥有万能 Queue。
- 请求包含已有 File/Mapping owner、offset/length、resource range/F1 leases、有限 group/tag。
  拥有/借用 source 两档的寿命写清；完整 typed 描述由 consumer/成熟格式 adapter 给出。
- admit/next_ready或可等 ticket/whole-request completion、cancel_request、
  stop_admission/drain/close 提供完整 natural C++ 路线。
  在实现 API freeze 时固定名称和一次结果/错误形状，不增加无需求的重复 callback/poll façade。
- group ready 只表示该组成功数据可用，后继计算按原 native dependency 排序；
  old ready/use lease 占容量，不能提早复用 stage。group 后续失败仍保护此前成功组。
- 读：positional buffered 与 immutable mapping，direct IO 明确 opt-in且受原IO capability限制。
  写：已有 writable File/range、write_all_at；文件发布/命名/淘汰复用已有 IO/引擎规则。
  不写用户原模型文件，不隐式 mmap whole file populate 或下载 HF。
- bytes+slots 双重预算，包括 native padding、全部在途、pointer/control 和旧结果；
  容量满明确拒绝或现有 Semaphore/Channel await，默认不自动丢弃。
- 普通 CPU 数据不全部 pin；仅传输边界有 staging，源大于 staging 仍正常 chunk。
  Apple 共享物理内存仍有 CPU/GPU ordering/cache/ready；不承诺所有平台零拷贝。
- 原IO平台不支持安全中断 in-kernel read时取消停止后续任务，已接受部分仍退役；不把 timeout当停止。
  无私有 poller/threadpool，所有真实 completion/first error/partial progress 可查询。
- 包中默认随 Compute/Graph route SDK闭包交付，不新增细粒度用户开关；
  compute_stream 是内部解依赖因子，不新造必须手选的公开“Stream模式”。

## 有限 FIX 集

| FIX | 必须实现 | 验收 |
|---|---|---|
| F3-01 | 中央 compute_stream 身份/依赖/普通 route 及三交付投影 | 无环，无第二名单/空 target；静态共享source一致，不拉无关 CPU Provider |
| F3-02 | create/prepare/admit，File/Mapping owner、range/desc/group 与有限预算 | 错描述、空请求、EOF/offset溢出、OOM、source大于stage、共享backing去重 |
| F3-03 | positional/mapping读→stage→F2/native-ready；多chunk多group并可提前消费 | 短读与尾块有错误/实际字节，不能整 shard 临时buffer；ready不是只完成host读 |
| F3-04 | producer依赖→resource→stage/host→write_all_at反向旅程 | 未完成GPU写不能读取；short write/部分进度；原子发布交已有owner |
| F3-05 | F1 leases、逐组/整请求结果与旧结果容量、独立在途槽 | N=2组/K=2槽以上；早ready旧租约不复用，无隐藏全设备等待 |
| F3-06 | request cancel、stop/drain/close、abandon/late completion、partial acceptance | 原错误不吞；在途内核/IO不假取消；关闭本pipeline不关闭用户所有资源 |
| F3-07 | 原生四路线与实际math consumer、确定输入与资源成本/路径报告 | BF16/F32/至少一种已有block编码保持；copy后Direct+Graph真实输出对参考 |
| F3-08 | 成熟GGUF/Safetensors消费示例、指南、SDK迁址/新头/包闭包与P0 | 解析器在consumer；本地Qwen混合dtype/ranges；DLL同目录，普通入口可发现 |

## 解析器与真实权重验收

格式解析不进入 pipeline。GGUF 用现有 ggml 格式核；本地 Qwen3.5-2B 原始 Safetensors
用官方格式工具生成 checked range manifest/fixture，独立 C++ consumer 消费它。
不能用文件 Q 标签猜所有 tensor 的 encoding，也不新增 HYLE parser/Torch依赖。
如 consumer 需要 C++ Safetensors loader，应独立选成熟 adapter，不制造未定版的产品依赖。

本地文件位置由用户提供，取头部身份/原始字节范围和实际tensor metadata；
读取数据仅限所声明 FFN/规范化矩阵/小片段，不重复全文件散列/同步4.55GB。
已读真实第0层三投影 BF16，linear_attn norm F32[128]；MTP分支也存在，不表示本卡实现MTP模型。
稀有量化可用既有目录的固定小artifact/明确bytes fixture，数学仍用已接成熟Provider。
完整模型、tokenizer和生成loop未实现，不能称“全模型推理验收”。

## 预声明验证与停止

正常 read→math、resource→file→reload→math、early group ready/late fail，
source/sink寿命、EOF/short transfer、cancel、retained result、quota与未知private bytes，四路线及三交付。
建立一套host workflow/相同fixtures集中运行；复用F1/F2已签同身份Native证据，
此卡只新增file/group/close接缝，不重复算子正确性全目录或性能全集。

P0检查logical/native/staging/metadata/old leases账单、峰值inflight、分块上限、热分配和host waits。
真实overlap若报告，必须用原native copy+成熟math的进展证据；只有异步函数名称/两queue不算通过。
可重叠执行不要求可复现吞吐领先；无需空跑P1争夺赢家。
所有FIX与适用门完成后停止，不追加GDS、持久化cache服务、F4/F5或完整引擎。
