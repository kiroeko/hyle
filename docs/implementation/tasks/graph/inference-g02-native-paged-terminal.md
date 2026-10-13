# G02 原生精确页联合消费有限终审

状态：native联合有限域DELIVERED；graph_route_cpu_apple、graph_route_cpu_vulkan、compute_batch
本轮CLOSED，2026-10-14。产品candidate `1cd0893979e1672a0f6fac98b783a6a3bfcabbca`；
baseline、NJ01–06/NJ-Q01、授权与原FAIL见[施工卡](inference-g02-native-paged-composition.md)。
graph_batch新node/owned binding也已取得本native域资格，但其原MQ/CPU共同轮Windows门
尚未裁决，当前模块仍CANDIDATE，不签整个G02。原graph5ce也保持其未决门。

## 七问与七门

1. 借原CPU Context/Queue、MLX/ggml Vulkan原锁及G01 page数学、F2、MaterializedNativeState/
   PagedMetadataPolicy、MQ-LP LeaseSubmission与原ResultSlot/retirement。SELF只图descriptor/
   冷owned node-layout映射、薄traits/exact frame接纳与原fragment输出桥；没有新数学、
   controller、scheduler、queue、FFI或compat转发。
2. 主消费者是真实CPU ABS/Cast→native GQA8/2 page16 attention→CPU Cast/Neg；Apple
   F16/BF16、Vulkan F16、D64/128/256、causal/noncausal、两requests混合与single、乱序/
   重复/尾页/未引用sentinel、固定active输出及真physical capacity。
   [原source](../../../../evidence/correctness/inference-g02-native-page-joint-source-stage/README.md)
   216episodes；[最终64步](../../../../evidence/correctness/inference-g02-native-page-joint-long-stage/README.md)
   Apple768/Vulkan384共1152episodes独立double通过。正交two page nodes/independent
   recipes/shared inputs验证完整集合、版本、所有reader/publisher与错误拒绝。
3. 显式prepare_batch全live node集合；plan拥有recipe/layout，冷span不借到热路径。固定
   输出shape变化拒绝，其它topology cold更新原native Program/table。K/V只外部inputs，
   全部live读者限定qualified page ports；不得发布capacity或交给CPU/dense/query节点。
   普通tensor/ReadLease/weight不绕过frames；拒绝绑定保旧输入，拒绝准备须重新完整准备。
4. P0：每plan一个冷page vector、每Invocation可空owner及仅opt-in slots；不为每ordinary
   value分配页表。table只有I32 max refs，两bank原F2真实上传/wait；ready Q wrapper、
   CPU/native边界copy、cold native/transfer替换和metadata如实收费。LeaseUsePool仍有
   O(guards²)去重，private上界UNKNOWN；G06 delta/capture与G09流水不代签。不启动新P1。
5. 两hybrid公开CompileOptions::paged_nodes、prepare_batch和PagedArenaRead bind；原
   GraphBuilder/custom Operation/placement、submit/wait/result/close沿用。Apple只读Q导入
   经private traits准入原Metal authority，再原binding核实际类型/handle/bounds；原direct/
   reference/Vulkan默认exact resource不变。Vulkan cold page输出替换原transfer后retain/
   seal，原backing不变检查保；零byte页输出不制造非空transfer/download。
6. [四份relocated公开Graph SDK](../../../../evidence/correctness/inference-g02-native-page-joint-sdk-stage/README.md)
   只链hyle::graph，actual math/domain/OOM/budget及十二独立头通过；
   [Windows定向公开链接/三头](../../../../evidence/correctness/inference-g02-native-page-joint-windows-compile/README.md)
   MSVC19.50.35728.0、四jobs、419.171s通过，无exe/GPU run，不代签Windows数学。
7. [使用页](../../../user/batch-attention.md)与同一public consumer说明自然调用、完整layout
   集、exact frames、固定shape、借用寿命、ready Q、cost报告、旧结果/错误/close，以及
   effects/capture/private限制；用户不写raw页表、whole-arena permission或第二提交循环。

[联合风险](../../../../evidence/correctness/inference-g02-native-page-joint-risk-stage/README.md)
检查wrong/foreign node/schema/resource/generation、混代/shared layouts、漏/重复集合、fixed
输出shape、所有readers/publishers、raw bypass、拒绝保旧binding、真实全空域。逐caller
allocation OOM：Applecompile50/create68/prepare1/bind1，Vulkan50/71/4/2，无泄漏/错误
admission；旧正确binding在OOM拒绝后实际执行。backing/host/shared预算拒绝通过。
真实CPU queue失败、actual native accepted prefixes的错误/cancel/abandon/unknown/close、
Apple queued metadata失败：所有referenced页在实际prefix退休前不可写，未引用页独立
可写；退休后uses/explicit metadata归零并回收页，错误无输出、不复用。Apple在原FIFO
mutex下确认joint Job实际入队后判close/cancel，不把无关blocker当自身已接纳证明。

[六组宿主风险](../../../../evidence/correctness/inference-g02-native-page-joint-host-sanitizers/README.md)
最终PASS、455.074s。原Apple无模型TSan FAIL保留；纯C++无HYLE/Metal/Sched两/三refs
复现，LLVM21 fence模型未实现，最终仅验证头副本标注真实counter的final acquire，保持
原release RMW和intermediate drops。Apple另沿原Metal同command buffer同步模型。无
suppression、产品/cache/util正常成本改写，无LSan/shader/driver/private-pool资格。

终审重新核descriptor truthful roots/active输出、cold owners与Context稳定借用、全部K/V
live readers、complete shared layouts、拒绝事务、outer旧result门、原native output/seal/
transfer、取消与unknown实际退休、费用/UNKNOWN、SDK公开target与源SHA。三个本轮closed
owner的scope、architecture、correctness、delivery、usability_docs、red_team、
engineering_performance七门PASS；graph_batch原共同门聚合未定不伪写PASS/CLOSED。

## 保留范围与停止

本native有限域停止日常施工/重复数学、SDK、san/P1。reopen只有可复现的已验数学/shape/
root/reader/版本/生命周期/退休/预算错误，或新的真实consumer机制。
Vulkan BF16、量化KV、producer→batch、delta planning、capture、重叠流水和CPU private预算
分别仍归G03–G10；本轮不靠一个小模型或少数正常episodes代签完整goal。

原26d7/graph5ce/common CPU+CUDA的Windows CUDA test与Vulkan未执行门保原事实；
[原共同卡](inference-g02-multi-queue-batch.md)和raw日志不得覆盖。graph_batch/graph当前
仍CANDIDATE，完整G02未完成；总goal0/10。resource限制不授权删除门或重启重型GPU run。

同candidate原loc-report/cloc self-check：[精确四owner src＋controller头并集](../../../../evidence/correctness/inference-g02-native-page-joint-long-stage/loc-owner-union.json)
4355 LOC，全库2911694、自实现不含测试176426。明确union边界，不当作单graph_batch、
全部Graph或vendor数学LOC；原source/risk/SAN/SDK/Windows FAIL与累计分别保各证据。
