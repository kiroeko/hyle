# F5 actual Air vulkan宿主TSAN

固定product25e，actual Mac16,12/M4/16GiB、macOS27.0.1/26A434、Apple Clang21。
独立tsan tree、原public_native与native两个真实消费者，普通Compute/Graph产品，
实际数学走原MoltenVK/ggml GPU路径；无CPU数学替代。
实际commands核对消费者、native batch模板/Graph factory、原宿主退休与paged import均有
对应sanitizer flags。两个消费者原实际风险全部PASS，具体episode以完整原日志为准。
控制接口外部串行；原worker/fence接纳的页权限保活、取消/失败/close和实际退休沿原机制。

同root configure累计27.794/300、build228.223/1800、
consumer21.517/180，保全部原attempt；库按实际内存j4/j2/j1，consumer1。
未运行旧算子catalog/全库/P1，未改产品或公共验证工具；没有suppression/改exitcode。
此证据只签实际instrument的宿主facet，不签GPU kernel/driver/private vendor code，
不代签Windows MSVC或其他平台。命令/机器/输入与stage hashes、flags见[receipt](receipt.json)/raw.zip。

本轮为固定F5 product25e + 独立IREE配置修复11c348e40的组合；receipt明确两份候选，
其他原F5 manifest inputs保持。原首次runtime未instrument导致operation56/adapter64，
wait count错误退出1；原FAIL及探针在独立layout-failure证据和本raw中保留。
唯一两行product修复将HYLE tsan传到IREE官方IREE_ENABLE_TSAN开关，实际commands核对
HYLE adapter与原C proactor均-fsanitize=thread，而不是改宏伪造layout兼容。
沿同root增量reconfigure/build17.281秒继续预算，原两个消费者与断言完全未改，
公开四域均通过，native错误/unknown/close/页权限/append/F4以原实际日志为准。
普通san=off SDK未因该san-only条件自动重跑；资格由独立条件投影证明，不伪造新SDK run。
