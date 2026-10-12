# F4/F5与四路线消费不对称的定向研究

状态：RESEARCH_COMPLETE / DESIGN_PROPOSED，2026-10-12。用户要求在原
[缺口研究](inference-foundation-gap-research-2026-10.md)基础上，按SGLang、llama.cpp、vLLM
源码研究F4/F5及CPU+CUDA与Apple/Vulkan的不对称，再做设计。本卡是研究记录，不是产品closure card。
HYLE baseline `df796ce4846333b42c8f98c6ee1855dec7d527ae`；工作树初始干净，F1–F3已CLOSED。
本轮不改产品/registry/依赖锁，不重开模块，不启动实施goal、模型推理、大项目构建或P1。

固定为一次21个peer文件索引及相关变更的定向阅读、六个锁定Provider文件补读、25个HYLE
公开/实现合同核对；范围只到F4/F5及其直接Graph消费。完成来源、承载、生命周期、成本和有限设计
后停止，不用新库版本无限追加feature。结论是静态设计，不是功能/平台/性能PASS。

## 身份与新鲜度

三个现有参考checkout保持原HEAD/分支及干净工作树；本轮查远端HEAD并fetch精确对象，
用`git show <SHA>:<path>`写研究副本，没有pull/reset/clean或移动checkout。
完整路径、文件SHA、相关更新与HYLE subjects见
[输入收据](../../../../evidence/architecture/foundation-f45-design/source-manifest.json)。

| 参考 | 原研究SHA | 本轮固定SHA | 定向更新结论 |
|---|---|---|---|
| SGLang | 234ccba7f625e12639e621044496bcc6197c3758 | e6072935bfa1036d4e20e8ed042ce1ff09872f57 | Mamba新增checkpoint transfer成员发现，强化sibling状态必须显式纳入；copy的flushed前提未放宽 |
| vLLM | 276fbcff2717bd934cfa37c8a2e4c391f3e7237b | cc9b0f097d49abfa6fc94d1d773dca1cc2b9619c | CoW主体未改，主要为注释与scheduler局部变更；copy两端保活/真实fence不变 |
| llama.cpp | 23b0202a189c44a54625aadcb37a946dd1d6278d | f2cc7282ce0042641df55e0e563b75e824e2b647 | 新Hadamard input图复用及Vulkan实现改动不把KV消费改成通用paged ABI；bounded recurrent rollback不变 |

HYLE锁定FlashInfer仍为0.6.18 / `69ff11fc4954396d98326656dc85debd2223f637`（Apache-2.0），
MLX为0.32.2 / `1f8e74e3f12f31365464a6867c6579f0e9b29d85`（MIT），见
[依赖锁](../../../../cmake/HyleDependencyLock.json)。在线当前文档版本只辅助发现，判定以锁定源码为准，
不把网页0.7.1/0.32.3功能写成当前已接入能力。

## 关键事实及反例

| 来源 | 本轮核实的事实 | 对HYLE设计的约束 |
|---|---|---|
| [SGLang MambaPool::copy_from](https://github.com/sgl-project/sglang/blob/e6072935bfa1036d4e20e8ed042ce1ff09872f57/python/sglang/srt/mem_cache/memory_pool.py#L1177) | conv/temporal/sibling一起复制；ReplaySSM仅复制已flush的checkpoint，活动ring内容不由temporal复制恢复 | F4版本是consumer明确列出的完整资源集合；模型选择checkpoint边界。库不能从没保存的数据恢复过去 |
| [vLLM _apply_cow](https://github.com/vllm-project/vllm/blob/cc9b0f097d49abfa6fc94d1d773dca1cc2b9619c/vllm/v1/core/single_type_kv_cache_manager.py#L453)及[scheduler deferred-free](https://github.com/vllm-project/vllm/blob/cc9b0f097d49abfa6fc94d1d773dca1cc2b9619c/vllm/v1/core/sched/scheduler.py#L2775) | 切换业务block table不等于copy完成；CoW source与新destination都保留到worker fence处理后 | F4候选、源版本、失败已接受prefix全部必须沿F1实际退休；不能仅靠head/refcount释放 |
| [llama recurrent seq_rm](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/llama-memory-recurrent.cpp#L168) | 回退只覆盖保存过的有限snapshot；多sequence共享cell不能只倒退其中一个 | 版本恢复与重新执行应用语义分开，不引入任意rollback/runtime |
| [llama KV seq_cp](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/llama-kv-cache.cpp#L469) | 同stream只共享cell metadata；跨stream要真实复制且该路径要求完整KV buffer | 逻辑分享、物理clone、迁移是不同语义，不能都叫廉价fork |
| [SGLang fast_prefill_plan](https://github.com/sgl-project/sglang/blob/e6072935bfa1036d4e20e8ed042ce1ff09872f57/python/sglang/srt/layers/attention/flashinfer_backend.py#L180) | 使用host-known indptr/length，更新稳定device buffers，但调用私有cached module plan以避开D2H | F5必须保留每批planning/update成本；不将该私有优化猴补当稳定C++公共接口 |
| [vLLM paged metadata builder](https://github.com/vllm-project/vllm/blob/cc9b0f097d49abfa6fc94d1d773dca1cc2b9619c/vllm/v1/attention/backends/flashinfer.py#L1370) | CPU prefix sum/tail长度、GPU page indices及真实request/token数都逐步变化，capture wrapper有固定容量 | metadata值与physical capacity分开；主机快照与device视图要同一版本，不能只按batch尺寸缓存 |
| [vLLM graph compatibility](https://github.com/vllm-project/vllm/blob/cc9b0f097d49abfa6fc94d1d773dca1cc2b9619c/vllm/v1/worker/gpu/cudagraph_utils.py#L104)及[llama can_reuse](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/llama-graph.cpp#L1463) | token/request上界以外，还有uniform长度、最大query长度、microbatch/LoRA等身份；llama逐input检查 | 不能只凭shape/address复用。模型参数/LoRA身份由engine纳入key，库核对自己的资源/plan generation |
| [FlashInfer paged_kv_t](https://github.com/flashinfer-ai/flashinfer/blob/69ff11fc4954396d98326656dc85debd2223f637/include/flashinfer/page.cuh#L38)及[native scheduler](https://github.com/flashinfer-ai/flashinfer/blob/69ff11fc4954396d98326656dc85debd2223f637/include/flashinfer/attention/scheduler.cuh#L426) | 真实端口是arena、page indices、indptr、tail、head/layout；native plan使用host indptr及多个workspace并创建host临时vectors | 原生region接缝可借原C++机制；planning不是免费或保证无分配，需单独账单/预算 |
| [锁定FlashInfer plan](https://github.com/flashinfer-ai/flashinfer/blob/69ff11fc4954396d98326656dc85debd2223f637/flashinfer/prefill.py#L2120) | plan不能进入CUDA Graph/torch.compile；稳定buffer不保证split-KV launch数不变 | F5显式准备本批再run；capture要求native topology资格，不凭固定地址给绿灯 |
| [锁定MLX fast.h](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/fast.h#L47)与[compile.h](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/compile.h#L10) | 标准SDPA端口为Q/K/V/mask，无上述page表；compile有shapeless选项 | 当前HYLE固定shape wrapper限制不等于MLX永远不能动态；原生general需另证无warm JIT/资源界，不直接删除拒绝 |
| [ggml flash_attn_ext](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/ggml/include/ggml.h#L2477)及原Metal/Vulkan调用 | 操作消费普通ggml tensor/view及mask，读到的标准端口没有FlashInfer paged_kv结构 | 可以研究已声明的gather/materialize参考路线，不能将它叫同成本native paged；不声称整个生态没有其他扩展 |

## HYLE当前不对称复核

F1–F3已经补共享/原native退休、有限范围搬运与双向文件管线；旧M2/M3/M5不能继续当未实施缺口。
M1/M4/M6仍需本轮设计，F1–F3的CLOSED不代表它们已交付。

| 能力 | CPU / CPU+CUDA | native Apple/Vulkan与CPU组合 | 结论 |
|---|---|---|---|
| declared domain / finite exact variants | [共同Executable](../../../../include/hyle/graph/executable.h)在create_invocation时固定signature；CPU+CUDA调用共同compile_admitted | [Apple hybrid](../../../../src/graph_route_cpu_apple/compile.cc)及[Vulkan hybrid](../../../../src/graph_route_cpu_vulkan/compile.cc)在dimension_count非0时拒绝；native Graph也固定 | 需要共享shape/coverage语义与有限exact-native variants；不要求相同机器码或虚假general支持 |
| explicit dependencies/effects | 共同Graph验证及执行ordering；effect footprint不是snapshot，也不向数学实现传数据 | 两hybrid编译器对任何显式control edge/effect拒绝；纯GDN initial/final仍可用 | DAG静态control edge是可补的结构缺口；真正外部mutation必须由native region声明preserves_effects并实际携带权限 |
| Provider规划与资源选择 | 共同region proposal/resource/search/replay报告 | native类型/managed MLX及Vk命令保留自身owner；hybrid走独立lowerer | 共享静态验证/shape/effect模型，native执行/退休独立；不能强改万能Context/Queue来凑签名 |
| output/old result/lease | 普通TensorView/WriteLease外部输出与F1 leased wrappers | 原native arrays/Tensors及ResultLease，输入readonly lease已接；外部mutable输出档位不同 | 统一可验证ownership/cost事实，保留自然native返回；F4优先pure out-of-place，按需新定mutable output seam |
| paged/ragged成熟数学 | 当前FlashInfer只有dense single decode/prefill、exact-only、capture_compatible=false；[native wrapper](../../../../src/compute_provider_flashinfer/native_attention.cu)未接batch paged | 已读标准SDPA没有页索引端口；通用byte分页不能直接绑定普通TensorView | 新typed端口+显式batch planning；不把CompositeBuffer本身写成数学完整路线 |
| 平台资格 | FlashInfer [CMake](../../../../src/compute_provider_flashinfer/CMakeLists.txt)目前仅Linux CUDA | Apple Metal与Vulkan各有自己的Provider域 | CUDA Windows paged未准入，不能将Linux证据平移；软件Vulkan仍不代签GPU |

补齐的是功能合同与可组合旅程；统一内存不取消同步，managed replay不冒充录制命令，
explicit CPU materialization不冒充原生paged，也不能因为某路线没有paged native就偷偷跑CPU。

## 结论与设计回指

- F4采用显式不可变资源集合版本、可写draft、成功发布/旧版本保活；纯initial/final状态优先分享，
  只有修改共享块时显式clone-on-write，复用F1/F2，不新建模型事务/回滚runtime。
- F5采用独立容量contract和每批prepare/update/run，host-authoritative metadata快照，
  原native Provider规划/计算、F1 metadata/payload保活。普通Invocation无warm规划的旧合同保留。
- 三组合路线先共享静态语义及有限exact variants，补control/effect admission；capture/general
  和native memory保持真实差异。paged原生承载先锁定Linux CUDA；其余路线如实区分显式参考与缺席。
- 设计与有限后续ledger见[状态/分页与四路线设计](../../../architecture/compute-state-paged-design.md)。
  尚未实施/未跑真实engine、native paged prototype或性能门；不计作模块状态或功能完成。

文档/状态检查与change-impact selector只运行docs/closure，140项skipped/reused理由保留于
[验证收据](../../../../evidence/architecture/foundation-f45-design/validation-result.json)。
实际结果：61个component合同、Markdown/相对链接及diff检查通过；产品/registry/依赖锁
与三个peer checkout HEAD均未改变。只有研究文档PASS，native prototype/engine correctness/P1均NOT_RUN。
