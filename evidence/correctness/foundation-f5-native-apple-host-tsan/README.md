# F5 actual Air apple宿主TSAN

固定product25e，actual Mac16,12/M4/16GiB、macOS27.0.1/26A434、Apple Clang21。
独立tsan tree、原public_native与native两个真实消费者，普通Compute/Graph产品，
实际数学走原MLX GPU路径；无CPU数学替代。
实际commands核对消费者、native batch模板/Graph factory、原宿主退休与paged import均有
对应sanitizer flags。两个消费者原实际风险全部PASS，具体episode以完整原日志为准。
控制接口外部串行；原worker/fence接纳的页权限保活、取消/失败/close和实际退休沿原机制。

同root configure累计4.098/300、build146.363/1800、
consumer8.639/180，保全部原attempt；库按实际内存j4/j2/j1，consumer1。
未运行旧算子catalog/全库/P1，未改产品或公共验证工具；没有suppression/改exitcode。
此证据只签实际instrument的宿主facet，不签GPU kernel/driver/private vendor code，
不代签Windows MSVC或其他平台。命令/机器/输入与stage hashes、flags见[receipt](receipt.json)/raw.zip。

本门使用同Apple TSan root和原预算：所有HYLE/consumer的原已插桩.o/.a不变，
仅将vendor MLX archive换成已通过Apple-only shared SDK的同候选libmlx.dylib。
两个实际消费者运行与原风险断言未改，TSAN_OPTIONS仍halt_on_error=1；没有suppression、
关闭race检查或改exitcode。实际重链接命令、binary/dylib hash与otool loader保存在receipt。
数据竞争facet只签HYLE-owned host，普通18 SDK交付另证，不能称整套MLX/Metal已instrument。

原静态消费者两次FAIL/-6保留；不链接HYLE的同deleter模板对照也复现原GCD/MLX Data报告，
同对照换共享MLX后24个GPU cold worker退出0。此差分支持static weak-template coalescing
遇到未插桩vendor同步的解释，但不代签vendor实际无竞争。MLX/Metal whole-process facet
仍INCONCLUSIVE，详见独立诊断证据；原FAIL与elapsed计入上文总量，原binary未覆盖。
