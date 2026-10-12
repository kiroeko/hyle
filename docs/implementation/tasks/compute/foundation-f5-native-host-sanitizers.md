# F5 native host-control sanitizer验证范围

状态：CANDIDATE验证卡补充，产品/API固定25e4fb398；不新增产品能力或修改公共仪器。
承接[活动候选卡](foundation-f5-candidate-validation.md)预声明的native host-control适用风险。
common与CPU三san证据不代签本页。

## 精确facet与执行

| 平台/facet | 实际验证 | 不由此证明 |
|---|---|---|
| Windows actual RTX Vulkan host wrappers/Graph factory/callback/owner | MSVC ASan，原materialized_native与public_native两真实消费者；正常四域、append/F4、old results、caller OOM、native错误/unknown/close/drain | GPU shader/driver heap未被ASan instrument；不是GPU内存或性能证书 |
| actual Air Apple host wrappers/Graph factory/worker边界 | Clang ASan，原materialized_native与public_native两真实消费者，actual Mac16,12身份先核对 | MLX/Metal私有GPU storage/预编译库未完全instrument |
| Linux actual CUDA FI host adapters/Graph batch coordinator/native callbacks | GCC ASan，原public CUDA与native Direct/Graph风险消费者；原C++宿主 instrument，kernel/driver不冒充 | GPU kernel字节访问须靠已声明实际数学/故障/权限证据，不称ASan能看GPU |

控制API均thread-compatible且externally serialized，不制造同一controller的并发control测试。
immutable layout/页/版本分享并发已由common TSan覆盖；新宿主跨线程退休边界如仍有适用
TSan/UBSan风险，必须逐代码/平台定界并补实际证据，不能把整个类别NA。此卡不先签任何PASS。

每个平台只原F5 owner与真实直接消费者，不跑旧算子catalog/全库/P1。复用身份匹配的源码
cache；sanitizer配置必须独立binary tree，不能把ordinary SDK binary当san。普通Compute/Graph
测试parent仅添加选定消费者并调用原hyle_apply_sanitizer，实际compile flags必须核对HYLE
controller/coordinator以及消费者。原native runtime/library借用边界/退休语义保持。

本native平台配置的F5新gate各configure300/build1800/consumer180（普通gtest120），第一次
记录后同root累计，所有FAIL/TIMEOUT保留。不是重置已用common/CPU/native ordinary预算。
库按实时内存j4/j2/j1，consumer1，跨机器重型串行。因公共工具/平台限制无法给必需verdict
则精确记录独立条件，不能用未instrument运行或CPU替代证明。

Windows Vulkan host ASan已实际PASS：[证据](../../../../evidence/correctness/foundation-f5-native-vulkan-host-asan/README.md)。
configure累计153.844/300、build720.312/1800、两consumer9.422/180，包含全部原FAIL/
中止时间。私有oracle只用dtype.h标量inline，不需要SIMD库，已移除额外完整SIMD产品选择；
没有借此修改closed SIMD或改变实际数学。actual Air Apple资格见下文，Linux CUDA继续。

actual Air Apple宿主ASan已PASS：[证据](../../../../evidence/correctness/foundation-f5-native-apple-host-asan/README.md)，
build155.515/1800、两consumer5.211/180。同机器runtime明确拒绝detect_leaks而首进程
exit-6未进入测试，保原FAIL/time，只按实际平台用detect_leaks=0继续；Apple LSan工具facet
不支持，未冒充PASS或删整个san类别。GPU/MLX私有storage仍不由ASan证明。

Linux CUDA ASan已启动原public D256、native Direct/Graph三消费者；普通CUDA产品闭包
重用锁定source cache、独立tree，以原GCC14/CUDA13.3/SM89身份instrument宿主代码。
实际source/worker callback的剩余UBSan/TSan适用性须按代码及平台继续核定，不写未跑PASS。

Linux native callback明确是适用并发facet：新BatchInvocationData的terminal由原native
完成callback写入，pending release/acquire保护其消费；新Coordinator和原F1 use/retained
completion跨线程退休。不能以控制对象externally serialized删除该风险。
补GCC UBSan/TSan两个独立配置，各固定原F5 configure300/build1800/consumer180累计。
验证parent可用内部精确factor graph_batch/compute_provider_flashinfer/compute_backend_cuda/
compute_state/graph_route_cpu，后者是原projected_rejections的额外CPU reader拒绝用例真实
依赖，不能因主体走CUDA而漏链接。三消费者仍调用同一公开API和原FI真实GPU路径；只减少未被该fixture消费的
默认ggml/torch/cuBLAS库编译，不裁剪用户产品或改Provider选择。普通SKU的三交付已由18
SDK另证。新san root首次登记后不更名重取预算，跨机器重型仍串行。

Linux ASan首build555.387秒编译完但Graph fixture因上述CPU依赖遗漏link FAIL；同tree原
剩余build1244.613秒补真实CPU路线，原失败/时间保留，不修改fixture来跳过拒绝风险。

Linux CUDA地址访问ASan三consumer现已PASS：[证据](../../../../evidence/correctness/foundation-f5-native-cuda-host-asan/README.md)。
首运行先cuInit OOM而未运行数学；无HYLE原生对照复现，按NVIDIA文档设置
protect_shadow_gap=0后driver/数学运行恢复。detect_leaks=1的独立诊断保退出FAIL：public
2320/8、Direct1000/5、Graph8920/35，分配首帧均CUDA/NVML/cuRAND；无HYLE对照精确复现
前两者与Graph driver部分，cuRAND差异保持限定。地址门detect_leaks=0通过，没有任何
suppression/改产品；LSan总体INCONCLUSIVE_NATIVE_VENDOR_REPORTS，不称GPU或整进程泄漏门PASS。

UBSan已在合法top-level maintainer factor入口构建；首次误把该选项用于embedding parent
的配置FAIL保留后，只修私有入口，不改产品/公共工具。TSan仍待同有限facet实际验证。

Linux native UBSan现已三consumer实际PASS：
[证据](../../../../evidence/correctness/foundation-f5-native-cuda-host-ubsan/README.md)，configure
11.498/300、build859.339/1800、运行2.145/180。first parent option FAIL与top-level bin/
launcher未启动测试的ENOENT记录保留，按实际路径运行，不重建或重取预算。TSan单独
配置已启动，同product/source/直接consumer和原budget；尚未运行项不计PASS。

Linux native TSan宿主数据竞争门现已三consumer实际PASS：
[证据](../../../../evidence/correctness/foundation-f5-native-cuda-host-tsan/README.md)，configure
8.486/300、build803.101/1800、consumer累计10.536/180。首次公开消费者实际数学通过，
退出66为libnvdiagclt.so.1诊断线程finished/not joined；完全不链接HYLE的CUDA生命周期
对照精确复现同报告。原FAIL/源码/runtime help保留，仅report_thread_leaks=0分离数据竞争
门，report_bugs仍true、halt_on_error=1，无suppression/改exitcode/产品改动。三个真实消费者
数据竞争门PASS不代签外库线程生命周期：后者INCONCLUSIVE_FOREIGN_NVIDIA_DIAGNOSTIC_THREAD。
全平台适用facet和整进程LSan边界仍须最终逐项裁决，不能称整套sanitizer全部通过。

## 剩余平台facet的有限补验

固定25e只读核对：MaterializedNativeState的版本/bank/control字段由外部串行控制，worker不
直接写这些字段；但Apple原Invocation接受新F5 exact-page uses后在实际worker退休并释放，
common/CPU/CUDA TSan不能代签该组合。实际Air Clang支持UBSan/TSan，补两个独立binary
配置，每个首次configure300/build1800/两consumer累计180，沿原public_native与native
消费者及同source manifest运行。实际flags须覆盖apple_context、paged import、native模板、
Graph factory和消费者；不跑旧catalog、全库/P1，不改产品/公共工具，不重复已过ASan。
各root首次后保全部FAIL/time、不换root取预算，任一工具/外库报告逐栈对照再裁决。

Windows MSVC合同明确UBSan/TSan不可用，缺仪器不是语义NOT_APPLICABLE。Vulkan新增
paged import/reuse gate/control wrapper无新独立worker，但读取原runtime mutex保护的
pending/retiring/observer状态且接受新F5页权限；因此Air Vulkan Clang也补UBSan/TSan，
同固定source与原public_native/native消费者，每个独立配置首次configure300/build1800/
两consumer累计180。实际flags核对Vulkan program/memory/原退休/模板与Graph factory。
四配置跨机器串行，原同名mode root累计不重置；Apple不代签Vulkan，Air不代签Windows
MSVC或NVIDIA/MoltenVK私有代码instrument。此有限补验只补原适用风险，不新增产品范围。

Air Apple UBSan两消费者已实际PASS：
[证据](../../../../evidence/correctness/foundation-f5-native-apple-host-ubsan/README.md)，configure
4.164/300、build165.829/1800、consumer3.863/180。actual flags覆盖新native模板/Graph
factory及原apple_context/paged import。Apple TSan独立配置已启动，Vulkan两facet待跑。

Apple TSan实际公开consumer两次FAIL/-6：第一ragged F16数学通过后，后续cold prepare
报告MLX Data函数指针deleter的GCD读/构造线程写。两个简单no-HYLE对照未复现，第三个
按实际宿主template实例化匹配deleter后、只链接同MLX archive的对照复现同读/写/Data/GCD
报告。实际vendor CMake明示Darwin USE_TSAN不支持。完整
[失败与对照](../../../../evidence/correctness/foundation-f5-native-apple-host-tsan-diagnostic/README.md)
保留，仍INCONCLUSIVE_MLX_METAL_HOST_RACE_REPORT，不认定假阳性、不签PASS/NA，原native
风险consumer尚未运行。当前无产品反证归因，产品仍固定；继续独立Vulkan补验与有限诊断。

有限链接对照：同no-HYLE匹配deleter/attention源码，仅将同版本MLX archive换成已资格化
Apple-only shared SDK的libmlx.dylib，24个真实GPU cold worker均退出0，TSAN_OPTIONS仍
halt_on_error=1，无suppression/忽略module。此差分支持“static模板coalescing暴露未插桩
vendor同步”的解释，但未instrument MLX/Metal整体仍不能由此判零race。
允许在同Apple TSan root/原剩余build及consumer累计预算内只重链接原两消费者：全部
HYLE/fixture原instrumented .o/.a不变，只原vendor MLX使用同候选SDK共享native library。
原binary/静态失败保留，实际dylib/hash/otool与flags分别记录；此门只补HYLE-owned host
并发facet，不替代普通static/shared SDK，不代签未插桩vendor整体。仍不改产品/公共仪器，
不能因新binary名字重取预算。重链接与实际消费者须等当前Vulkan重型门完成后串行运行。

Air Vulkan UBSan两消费者已实际PASS：
[证据](../../../../evidence/correctness/foundation-f5-native-vulkan-host-ubsan/README.md)，configure
14.322/300、build245.787/1800、consumer6.337/180。实际MoltenVK GPU数学与原风险fixture
均执行，flags覆盖Vulkan program/memory/批模板/Graph。Vulkan TSan独立配置随后启动。

Apple原两消费者以同HYLE已插桩objects、同候选native MLX DSO重链接后宿主数据竞争门
已实际PASS：[证据](../../../../evidence/correctness/foundation-f5-native-apple-host-tsan/README.md)。
原consumer/断言不变、TSAN_OPTIONS=halt_on_error=1，无suppression/忽略module/改exitcode。
configure4.098/300、build含重链接146.363/1800、consumer含原两FAIL累计8.639/180。
原静态binary/FAIL与no-HYLE差分完整保留；host facet通过不签MLX/Metal whole-process，
后者仍INCONCLUSIVE且未instrument，不撤销独立诊断或把vendor工具限制写成语义NA。
