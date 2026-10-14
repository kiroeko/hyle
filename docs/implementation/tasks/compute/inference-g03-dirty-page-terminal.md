# G03 原页内容保留与 F2 写权限返回有限终审

状态：CLOSED，2026-10-14；compute、compute_batch、compute_route_apple、
compute_route_vulkan。固定candidate `b42832de3ddda21bdc3e98ad4cc87e73622cbba4`，
baseline/FIX/授权/全部失败见[本轮施工卡](inference-g03-producer-paged-consumption.md)。
只签GP01/GP02，完整G03与goal0/10保持未完成；不签pending页/producer→attention链。

## 七问、七门与范围

1. 原F1 unique/pending/use/actual retirement、F2 CopyStreamAccess.observe/take_mutable、
   Arena identity/generation/range与原native batch/queue/Completion复用。SELF只有原ready
   整页→writer的接纳与三个公开F2薄入口；无新controller、scheduler、allocator、dependency
   或compat。MLX/ggml/CUDA原pin、许可证与默认实现不改。
2. [公共形状反证](../../../../evidence/architecture/inference-g03-dirty-page-consumer/README.md)
   是初始化整页后只copy1024/16384字节不能经原严格output回接；成熟SG/vLLM/llama.cpp
   只改slot/row。GP01保原内容/identity/generation，GP02观察成功实际退休再返原writer。
   K/V dirty append与正交数值页range update/reduction都取得自然公开旅程；应用的slot/
   request/cache/F4发布政策保持消费方职责。
3. 共享reader/old frame/native uses阻止取得writer，拒绝不消费ready/旧binding；新page仍
   必须caller初始化所有将读bytes。copy output coverage没扩大；partial update须保留原
   PagedPageWriter，drop成功submission后unbind→page.publish，不把未写内容叫新copy。
4. P0：acquire(ReadLease&) O(1)、无allocation/payload copy/新generation；耗尽arena metadata
   quota仍成功。unbind沿原O(accepted phases) nonblocking poll与withdraw，没新state/field/
   wait；actual busy/failed/unknown/closed/retained submission/readonly/published拒绝。
   成功abandon须推进真实观察，不能只检查旧busy标志。原cold prepare/metadata/host waits、
   driver/private UNKNOWN保；本轮无新数学/算法/P1，不冒充零等待或绝对性能对称。
5. 三路线同名公开unbind，common真实CPU/CUDA。资源/Context/Queue/allocator仍稳定借用，
   externally serialized控制。无新C/FFI或普通用户private API；失败不会把状态改为ready。
6. 三交付与头通过：source CPU64/arena9、实际Air native192与copy9、原F3直接14；
   [六host san](../../../../evidence/correctness/inference-g03-dirty-page-host-sanitizers/README.md)
   190.616s；[四relocated native SDK](../../../../evidence/correctness/inference-g03-dirty-page-sdk-stage/README.md)
   58.886s/16heads；[CUDA source与两SDK](../../../../evidence/correctness/inference-g03-dirty-page-cuda-stage/README.md)
   source51.898s＋SDK148.599s、64each/6heads。全部SDK只链hyle::compute、consumer字节
   不变。W1当前native公开compile/link78.109s，不运行exe/GPU，也不代签原G02 Windows门。
7. [使用页](../../../user/batch-attention.md)给完整调用、旧结果/reader/写范围、借用和费用；
   原new page/clone/F4发布路径保。当前subbatch资格可依赖，pending producer→batch还需
   单独完整旅程，不由已有atom或这个ready更新wrapper反证。

红队与施工分开重读page origin/full-range/slot flags、try_write原拒绝、成功后唯一commit、
Writer RAII、native budget/resource borrowing、copy每accepted phase观察、失败与实际退休
区分、submission users门、producer/tail status、reset/closed与output已publish负空间。
没有扩大权限、原源码/交付差异或未映射新入口。CPU/native/source/六SDK/22headers与
定向W1证据都匹配固定产品SHA；测试新增不改变候选或重启原P1。
四owner本轮scope、architecture、correctness、delivery、usability_docs、red_team、
engineering_performance七门PASS，performance P0。

## 保留事实、LOC 与停止

原Apple abandon公开缺口已修observe-before-take；其FAIL/input保。Vulkan fixture的UMA
host-ready copy、empty logical accepted prefix预期及异常teardown导致SIGSEGV/ips保，
只修test逻辑与scope drain。san上传遗漏CPU文件的FAIL4.016s/累计保，TSan只签原已独立
反证的final-counter/Apple同command-buffer模型；无suppression/产品/cache/util修改。
ASan无LSan/shader/driver/private-pool资格。CUDA只tiny transfer/page/native fence，不签
math/model；原warning不删除，原Windows重型大套件未恢复。

[统一LOC self-check](../../../../evidence/correctness/inference-g03-dirty-page-cuda-stage/loc-owner-union.json)
candidate b428：全库2916272、自实现不含测试176449、四owner src明确并集21937。
并集不是单compute_batch、整个Compute或引擎接线LOC。

本有限范围停止重复验证/日常施工。reopen只因可复现的已验权限/内容/identity/退休/预算/
交付错误，或新真实consumer机制；G03完整producer→attention是原goal未决准入域，下一
owner轮须合法scope/audit固定FIX后再改产品。G04–G10/原G01与G02未决门保，goal保持active。
