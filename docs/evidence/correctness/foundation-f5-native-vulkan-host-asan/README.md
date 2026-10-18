# F5 actual Windows Vulkan宿主ASan

固定product25e，actual RTX4070/MSVC19.50/static/ASan。独立san binary tree、复用锁定
依赖源码cache，只原F5 native/public消费者；实际commands核对两consumer、Compute/native
controller、Graph factory/common coordinator及leased Graph有/fsanitize=address。
运行公开四域Direct/Graph独立oracle与原完整native风险fixture全部PASS：
非uniform/F16/BF16/GQA/causal/容量padding、页/同generation/two-bank/two-slot/old result，
caller OOM、actual PFN错误/unknown/取消放弃/close/drain/open失败不复用，append/F4/quota0。
具体episode以完整raw日志为准，未跑旧算子目录或P1。

原首次private parent未继承实际target processor的配置FAIL保留，读取原global property修复。
原parent额外链接完整SIMD产品触发无关vqsort ASan重型编译，查明消费者只用header-only
scalar dtype后中止该自有build，原exit/FAIL/elapsed保留，同tree改为仅Compute/Graph
并在原剩余预算内继续；不是observation timeout重启，也未改SIMD或任何产品源码。
无SIMD库后原public/native二consumer链接运行都实际成功，不以推测代替资格。

MSVC ASan只宿主内存访问，未instrument GPU shader/driver私有heap，MSVC没有LSan。
因此不签GPU内存、私有native总内存上界或所有平台san；caller quota/owner退役仍由实际
风险证据证明。全部命令/实际GPU/hash/原失败/累计预算见[receipt](receipt.json)/raw.zip。
其余native Apple/CUDA host facets、P0/LOC/guide/红队/最终closure按完整goal继续。
