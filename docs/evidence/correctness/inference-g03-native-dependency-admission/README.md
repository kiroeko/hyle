# G03 原 native 取消依赖接纳验证

固定产品 candidate `9404af694`，按[有限卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g03-native-dependency-admission.md)
仅修三 qualifier 的原 cancel 检查，没有公共签名/layout/数学/queue/allocator 改动。
[source 累计](source-receipt.json)/[原始输入与编排](source-raw.zip)/
[输入逐轮回映](source-input-map.json)，[宿主与 SDK 累计](validation-receipt.json)/
[原始配置、instrumented flags、消费者和日志](validation-raw.zip)，
[独立边界检查](red-team-receipt.json)/[原始结果](red-team-raw.zip)，
[Windows](windows-result.json)/[编译日志](windows-build-1.log)，
[影响选择](validation-plan.json)/[语义回映与限制](validation-result.json)。

Air 实查 Mac16,12 / arm64 / macOS27.0.1 / Apple Clang21。Apple 原 native invocation
与 copy recording、Vulkan 实际 Run/copy fence 的 pending cancel 均拒绝后继资格，
源 controls 保留、仍须实际退休，不发布 mutable success。原 attention 成功对照保
原 queue/Runtime/F1 guards；每平台原公共 65 steps、Apple 8/Vulkan 6 生命周期通过。
这里是 private source receiver fixture，不能把它冒充公开 SDK 的 pending-page API。

宿主 Apple/Vulkan 各 ASan/UBSan/TSan，共六配置通过；实际编译旗标在 raw 中。ASan
无 LSan，TSan 使用[既有明确模型](../inference-g03-copy-update-host-sanitizers/raw.zip)，
[本轮实际模型 SHA](models-identity.json)/[远端观测](models-observed.log)与原字节相同，
没有新模型或 suppression；不签 shader/driver/private pools 内部。四个 relocated
static/shared SDK 各原公共 65 steps 和三个独立头通过，only hyle::compute、原
consumer/installed headers 字节核对及 compdb include 路径通过；SDK 不用 private fixture。

边界检查补充实际“来源确认成功后才 request_cancel”的成功保持。最初错误地把
Vulkan suffix 退休当成 source 已被原 Run 观察成功，source 第二次和首 Vulkan ASan
检查失败，均保原日志；修正测试观察顺序、产品不变后，Vulkan source/三 host 模式
和 Apple 三 host 模式最终通过。Vulkan 在尚未观察来源时请求取消仍沿原语义取消，
不能借 GPU 排序把它改写成已成功。原取消后得到失败的页从未发布 ready。

source 累计 23.552 秒（含该 fixture FAIL），宿主/四 SDK 171.656 秒，边界检查
累计 22.343 秒（含该 fixture FAIL），Windows 实际受影响 Vulkan TU/MSVC DLL 编译
2.516 秒。Windows Apple .mm 语义上不适用；没有 Windows GPU/fullmodel、旧矩阵、
CUDA/nvcc 或已闭数学 P1。所有 heavy 串行、最多四 jobs。

selector 的一跳 Graph 因结构依赖被选中；实际 qualifier 调用只在原 F2 bind_producer/
prepare。Graph 节点/桥、数学和状态合同没有改变；本轮按原 native 生命周期、原真实
Batch 接纳消费者与 SDK compile/link 回映，不把旧未完成 Graph Windows correctness
门写成 PASS。完整 pending frame/late native output、后继与来源的最终状态合并仍
属于活动主卡，不能由这个有限 admission 检查代签。

[LOC](loc-owner-union.json) self-check PASS：固定 940，全库 2947154，自实现不含测试
176785，两 owner src 显式并集 13475。P0 只有 bounded qualifier 检查：Apple 原
done acquire 之外增加一次 done acquire 与原 cancel acquire，Vulkan 复用原 mutex
核 bool，无新增字段、原子写、分配/拷贝/native wait。首次 candidate registry 误带
performance 字段被 checker 拒绝，8950153ac 已纠正，未把失败隐藏为 PASS。
