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
