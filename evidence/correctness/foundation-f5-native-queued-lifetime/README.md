# F5 原生排队、放弃与open失败slot复核

baselinef9c4be383，AUDITED APPLE-QUEUE-03/VK-OPEN-03，只有test fixture变化，无产品/API改动。
同actual Air Mac16,12/M4/16GiB/macOS27.0.1/Clang21与原MLX Runtime worker执行，不用Mini。

Apple Direct/Graph各四独立Context，共八episode：原square GPU prefix后test Checkpoint只
挡住原worker或抛具名native异常，CPU执行拒绝，借原lifecycle/copy测试的成熟观察边界，
没有产品hook/新runtime/kernel。metadata prepare异步只由一个线程控制该controller，main
不并发控制它；Gate前prepare确实等待，前序异常后Internal、镜像不发布、slot不可复用。
math先prepare/bind后排在原GPU Gate后，cancel、释放public result与controller、前序异常
三类各保持三个input native uses到实际退休；之后guards归零、失败/取消无输出、open
失败slot不可reuse，最终metadata quota0。80ms只证明Gate阻塞，不作性能结果。

Vulkan Win RTX4070与Air M4/MoltenVK复用原四类PFN真实错误门，额外在没有先close的
known/unknown错误实际wait后释放public result并重试prepare/submit，全部拒绝。unknown
原Pending与三uses保持、原fence wait后错误仍ResourceExhausted，controller drain保首错。
旧“释放Pending result后drain”的独立证据在[上一收据](../foundation-f5-vulkan-failure-unit-stride/README.md)，
当前open失败检查没有把closed拒绝当成原错误poison。两profile原单KV nonuniform oracle、
正常96矩阵+24精确页/F4、ragged与236 caller OOM位置保持通过；Apple同样保持原正常与212
caller OOM位置，不由此代签未试的其他driver/private heap/执行域。

同原三profile日志root/config/tree，configure300/build1800/consumer180历史累计，无预算
重置，具体命令/input hash/机器与stage见[receipt.json](receipt.json)和raw.zip。原Win/Air
FAIL/TIMEOUT历史保留，机器重型串行、库j4/j2/j1、consumer1，canonical Windows只向Air
单向hash overlay。变化选择/skipped reasons见[validation-plan.json](validation-plan.json)，
产品与公头未变复用已匹配证据；没有全库/旧算子/P1，docs/closure/diff另验。

这是有限生命周期证据，goal保持active、owners保持AUDITED。其余明确风险汇总、immutable
候选、18 installed SDK、适用san/P0/LOC/guide/分离红队与最终closure仍按原六FIX完成。
