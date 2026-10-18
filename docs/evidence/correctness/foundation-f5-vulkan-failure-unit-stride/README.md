# F5 Vulkan 实际失败门与单KV stride修复

baseline135954d68。Win RTX4070/MSVC19.50 Release/shipping/shared与actual Air M4/MoltenVK/
Clang21 shared各Direct/Graph四个独立Context故障episode通过，共16；另各两个ragged
query-cap=1/kv-cap=1的独立nonuniform 128-value oracle通过，共4个。

故障只装饰原Context的native QueueSubmit/GetFenceStatus/WaitForFences PFN，所有成功分支
转发原函数，无产品测试hook/新runtime/kernel。四类为metadata第一个原submit拒绝、math
第一个原submit拒绝、第二个原submit拒绝（第一个实际PFN调用已成功接纳）、真正math已
接纳后一次非退休fence观察错误。unknown按原Vulkan合同返回Pending，三input native uses
仍在；释放公共OwnedResult后controller drain用原native wait/fence退休，保原观察错误、
guards归零，不发布错误结果、closed slot不可复用、metadata quota0。没有把Pending改写成
generic Completion poll error，也不以单个PFN计数外推未计量的provider私有queue费用。

第一次consumer-resume16在注入前触发原ggml.c:3382 `!ggml_is_transposed(a)` fatal，
exit3221226505，原FAIL保留。合法单query/单KV正是decode基础形状，未把capacity改大。
`ggml_is_contiguous`忽略unit axis stride；transpose后可同时contiguous和transposed，而原
lowering只在!contiguous时cont。合法重开owner为compute_provider_ggml_vulkan：
03ecbafa1 SCOPED→611498817 AUDITED，旧scope/源码hash封存于architecture；没有重开
compute_ggml_runtime、CPU ggml或原数学全集。

UNIT-STRIDE-01在plain right transpose的连续但transposed域借原同形ggml_reshape_4d重建
dense unit stride。原peer验证contiguous/相同element count，创建offset0同view_src的view，
无payload复制；非连续仍走原cont，量化/其他数学与runtime不改。原厂assert保留，API不变。
四个nonuniform V oracle验证两个head各64元素原位正确，避免只用零值掩盖stride错误。
原96矩阵+24精确页/F4、ragged、236 caller OOM位置/profile等实际消费继续通过。

consumer-resume17/18因fixture把unknown误当poll错误而FAIL保留；诊断stdout确认Pending与
guards，再按原合同修正fixture，没有因此改产品观察协议。Win私有parent新增SDK header
include使CMake再生成，并重编译其simd generated prerequisites，实际build日志保留；没有
选择全库/旧算子测试或修改公共构建工具。最终只定向真实owner/消费者与六个source-contract
测试，后者见[source-contract.log](source-contract.log)。

原Win/Air Vulkan root/config/tree与预算累计保留：[receipt.json](receipt.json)，Win
configure136.734/build573.457/consumer18.188，Air17.765/182.133/35.986秒，各consumer<180。
Air运行前/runner内身份仍为Mac16,12/16GiB/macOS27.0.1/Clang21/CMake4.3.4，Mini未使用。
Windows canonical→Air disposable单向hash overlay，机器重型串行，库j4/j2/j1、consumer1。

变化选择/skipped reasons见[validation-plan.json](validation-plan.json)。IREE native device
直接边只借未变NativeDevice，不调用region MatMul lowering（native_device.cc:88/138及其头，
全factor无prepare_region调用），本scope不跑其旧数学catalogue；现普通Vulkan façade仍链接
其未变原闭包。installed static/shared与其他风险不由本shared消费者代签。

保持AUDITED/goal active：其他route producer/partial/abandon/未主动close的未知终态组合、
standalone Direct D256 installed消费、18 installed SDK格、san/P0/guide/红队与最终closure
仍继续原目标。此处不是完整F5或原provider旧范围重审完成证书。
