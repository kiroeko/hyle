# F5 actual Linux CUDA FI宿主ASAN

固定product25e，actual RTX4070/GCC14.2/CUDA13.3/SM89，独立asan binary tree。
本记录PASS仅地址访问ASan，LSan总体不是PASS。实际ASAN_OPTIONS为
halt_on_error=1:protect_shadow_gap=0:detect_leaks=0；没有suppression或ignore错误退出码。
实际commands核对public D256与原native Direct/Graph、BatchInvocation/coordinator/
leased Graph与FI C++ adapter均有对应sanitizer flags；数学仍原成熟FI真实GPU运行。
三consumer全部PASS：四域D256 Direct/Graph、原short/long split、独立slot/旧结果/metadata
generation/update、actual prepare/run错误与suffix/unknown/pending/native retirement、精确页/
append/F4/正常放弃/open failed slot等。实际episode以完整日志为准，不虚构未跑风险。

ASan首build因原projected reader拒绝fixture调用CPU context而private CUDA-only parent漏
依赖，保原link FAIL/555.387秒；同树预算补其真实CPU路线，没有改fixture跳过该风险。
UBSan/TSan parent用内部精确factor仍真实原API/native FI/CPU拒绝用例，只少编未消费的
普通默认Provider；不改变用户产品或Provider选择，18 SDK独立证明普通三交付。

所有原失败、命令/机器/flags/hash/累计见[receipt](receipt.json)/raw.zip。同root预算继续，
无旧算子catalog或P1。GPU kernel/driver私有内存未被宿主san instrument；不能称其访问
或总字节边界已由此通过。此门不代签其他平台/facet/完整F5 closure。

原默认ASan首先cuInit OOM，no-HYLE control同样cuInit=2；按
[NVIDIA配置文档](https://docs.nvidia.com/rapidsmpf/latest/getting-started/)使用
protect_shadow_gap=0后，原生初始化/分配/free/release全部成功。原FAIL不删除，不修改HYLE。
随后detect_leaks=1独立诊断：public2320 bytes/8、Direct1000/5、Graph8920/35，首个非
ASan分配帧全部在libcuda/libnvidia-ml/libcurand；无HYLE operator-new/allocator分配根。
no-HYLE CUDA/NVML完整init/query/shutdown/dlclose精确复现public2320与Direct1000，
Graph的NVML/driver7600亦复现；Graph另1320是cuRAND ELF init栈，单独dlopen/DT_NEEDED
对照的加载次序不同，不能声称所有8920已逐分配精确复现。所有对照/差异保留，不泛化。

因此保LSan外库诊断INCONCLUSIVE与private UNKNOWN，caller quota/正常retirement不是整
进程LSan扫描；common/CPU既有LSan仍各保精确已签范围。地址ASan真实三消费者通过，
不是将未运行数学或启动失败改名PASS。后续最终risk audit须保此仪器/外库限制。
