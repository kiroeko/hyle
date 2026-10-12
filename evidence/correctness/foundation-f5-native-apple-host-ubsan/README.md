# F5 actual Air apple宿主UBSAN

固定product25e，actual Mac16,12/M4/16GiB、macOS27.0.1/26A434、Apple Clang21。
独立ubsan tree、原public_native与native两个真实消费者，普通Compute/Graph产品，
实际数学走原MLX GPU路径；无CPU数学替代。
实际commands核对消费者、native batch模板/Graph factory、原宿主退休与paged import均有
对应sanitizer flags。两个消费者原实际风险全部PASS，具体episode以完整原日志为准。
控制接口外部串行；原worker/fence接纳的页权限保活、取消/失败/close和实际退休沿原机制。

同root configure累计4.164/300、build165.829/1800、
consumer3.863/180，保全部原attempt；库按实际内存j4/j2/j1，consumer1。
未运行旧算子catalog/全库/P1，未改产品或公共验证工具；没有suppression/改exitcode。
此证据只签实际instrument的宿主facet，不签GPU kernel/driver/private vendor code，
不代签Windows MSVC或其他平台。命令/机器/输入与stage hashes、flags见[receipt](receipt.json)/raw.zip。
