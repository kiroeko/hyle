# G02 联合专属权限、OOM 与实际退休源阶段

[receipt](receipt.json)、[raw](raw.zip)、[Apple最终日志](apple-final.log)、
[Vulkan最终日志](vulkan-final.log)保实际Mac16,12/M4/macOS27.0.1/AppleClang21、28个
canonical输入SHA、全部十一份输入archive/FAIL/阶段日志与累计286.946s。source-only，
SDK、宿主san及Windows定向链接未签。

两route实际CPU ABS/Cast→原native page→CPU Cast/Neg，两个独立页recipe/slots共享同一
K/V输入。拒绝无cold binding、重复/foreign node/schema、capacity发布/普通dense reader、
CPU fallback、冷/热混layout、缺/重复layout、active输出shape变化、错误generation、
foreign resource、空frame/错port与raw lease绕行。拒绝后原正确绑定仍可运行；全空图沿
真实Q/native空输出/CPU空后段运行，不制造KV root或全arena权限。

冷HYLE allocator逐点故障：Apple compile50/create68/prepare1/bind1，Vulkan50/71/4/2；
每次拒绝无metadata泄漏、无错误admission，frame替换OOM保旧binding并实际跑出结果。
固定backing236拒绝、host workspace0拒绝、shared metadata额度用实际control占用后
reserve剩余额度再拒绝。native private上界仍UNKNOWN，不签总GPU硬预算。

实际关闭CPU Queue产生前段失败；两route原native边界取消、两public handles放弃、原生
prefix错误和在途close全部通过。所有referenced K/V页保持不可写，未引用页1可写；
丢外部frame/ready owners与public handles仍不能提前写，原native prefix真实退休后所有
页重新可写、host query active uses归零、显式metadata归零，失败无输出且不复用。

Apple以原MLX GPU checkpoint阻塞worker，并在原Runtime的queue_mutex下确认joint的
真实Job已进入FIFO；无新executor、产品hook或伪完成。一个原close fixture曾在joint
native admission前取消，不能证明已接纳prefix，原FAIL保；补真实FIFO资格后通过。
prepare_batch另实际排在失败native prefix后等待，再报告原Internal，不接纳page math。

Vulkan保原DeviceFunctions边界，先让原QueueSubmit实际接受CPU query upload和page math
prefix。GetFenceStatus一次OOM返回UNKNOWN，第二次真实证明前阻塞：public parent仍
Pending/无result、uses和exact页防写；解除后只用原真实fence观察退休，保最初错误。
native admission拒绝则保真实完成的upload prefix，submits2/accepted1。取消/放弃/close
的实际native提交计数保日志，不把整个coarse fragment当单个kernel或宣称精细抢占。

实际产品修正只有AUDITED CPU+Vulkan页专用零域：不为零byte输出retain非空transfer或
download；原native结果/backing检查和CPU零域后段保，报告不计未发生的空page copy。
普通与非空桥行为保原合同。原预算fixture、typed-zero Wsign compile warning及空transfer
FAIL均保，未覆盖或重置。非模型two-node消费与真实模型片段的数学源证据各有明确边界。
