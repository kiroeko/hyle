# 原native资源公开默认构造实际门

candidate a3fa6be8c，实际Air Mac16,12/M4/AppleClang21；[收据](receipt.json)和[原始证据](raw.zip)
保两source、四实际relocated static/shared、六hostsan、actualchanged TU/test flags，以及
两路线public default构造incomplete MemoryState red compile FAIL。修正只将两空默认
构造放到原完整State的TU，并导出原符号；正常factory/math/ownership/layout不改。
consumer只include对应public memory头，empty/default、vector resize、factory move
assignment、empty moved-from、destroy与Context close实际通过，不includeprivate产品头。
SDK实际staging→relocated，原prefix/tree保留，producer只构建对应route目标≤4jobs。
san覆盖宿主空owner/生命周期，原ASan detect_leaks=0限制，不签GPU math/shader/driver。
P0空构造零分配/无字段增加，P1不适用于compile/default空对象修正；原NR性能证据保留。
活动范围见[MC卡](../../../docs/implementation/tasks/compute/inference-native-memory-default-construction.md)。
