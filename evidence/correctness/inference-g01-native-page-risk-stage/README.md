# G01 原生分页风险与零域阶段

产品 `520027a57`，实际 MacBook Air Mac16,12/M4/arm64/macOS27.0.1/AppleClang21。
见[活动卡](../../../docs/implementation/tasks/compute/inference-g01-native-paged-read.md)。
source 风险只签本记录所跑范围，不代签三交付、宿主 sanitizer 或不可变 P1。

Apple 原 MLX worker/GPU checkpoint 的 metadata 排队错误、取消、放弃、已接纳 native
prefix 错误通过；放弃后先移除 controller/外部 frames/页 readers，原 query guard
仍为1，引用 K/V 页仍不可写，未引用页可写。释放 native gate/原 close 观察后，query
guard为0、所有引用页重新取得真实 writer，metadata零泄漏。逐cold create52分配点
OOM和新冷prepare OOM均通过，拒绝新代际后仍可提交旧已准备的真实 native recipe。

Vulkan metadata拒绝、一次原生提交的接纳前拒绝、接纳后阻塞fence观察OOM、非阻塞
未知观察均通过。阻塞错误真实观察到submit1/accepted1/wait1/drain1，保首个错误、
无结果发布；未知poll保持query guard1且无结果，随后真实fence证明退休。错误槽不复用，
释放外部/控制器后所有exact K/V页均可写。逐cold create55 OOM点与拒绝新准备保旧代际
也通过。原reference多段提交的计数不能代用：最初fixture猜第二次提交触发错误失败，
以及后续两个不成立的接纳/drain假设FAIL全部保留，没有把这些失败重分类为PASS。

零域发现真实公开消费缺口：原PagedArena非空permission合同拒绝空页集合。原生page
仅在Q rows/page refs同时零时接受两个默认空frames，私有程序只绑定真Q容量并输出
原空view；不虚造K/V root或大dummy buffers，原arena/reference/Graph合同不变。
Apple/Vulkan zero output/旧空result拒绝复用、foreign source resource拒绝、精确237byte
fixed metadata backing门（236拒绝）全部PASS。原empty domain FAIL与fixture Buffer.view
误用指针的compile FAIL保留。新空域后的两route普通分页与reference各build/run也PASS。

receipt.json保每轮SHA输入、机器、命令、退出码、elapsed与累计；raw.zip含所有原日志、
复现脚本、失败/最终输入包及原selector selected/skipped reasons。改动不启动全局数学，
不重复已关闭CPU/ragged P1；LSan、shader、driver和private native bytes仍未签。
