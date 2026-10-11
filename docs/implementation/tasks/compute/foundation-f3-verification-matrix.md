# F3逐能力验证矩阵

本页由[活动卡](foundation-f3-file-resource-streaming.md)纳入。当前六owner为CANDIDATE，
产品固定97858fbc8；原3b候选的suffix环由真实红队反证后有限修复，适用运行门现已齐，旧收据
各自的source身份、范围与原失败继续保留。
这里列出原八FIX的真实覆盖及明确限制，不用测试总数或粗估完成百分比替代裁决。

## 证据入口

- 共同/CPU：[首验](../../../../evidence/correctness/foundation-f3-cpu-subject/receipt.json)、
  [十项风险](../../../../evidence/correctness/foundation-f3-file-risk-acceptance/receipt.json)、
  [跨线程导出结果](../../../../evidence/correctness/foundation-f3-wsl-math-exported-drop/receipt.json)。
- CUDA：[Windows原producer寿命](../../../../evidence/correctness/foundation-f3-producer-lifetime/receipt.json)。
- Apple：[Air真实producer/取消/丢弃/late failure](../../../../evidence/correctness/foundation-f3-file-risk-acceptance/receipt.json)。
- Vulkan：[Windows suffix/unknown/producer](../../../../evidence/correctness/foundation-f3-vulkan-file-faults/receipt.json)、
  [Air同风险](../../../../evidence/correctness/foundation-f3-file-risk-acceptance/receipt.json)。
- 数学：[Windows CPU/CUDA/Vulkan](../../../../evidence/correctness/foundation-f3-math-roundtrip-windows/receipt.json)、
  [Air CPU/Apple/Vulkan](../../../../evidence/correctness/foundation-f3-math-air/receipt.json)、
  [WSL CPU/CUDA](../../../../evidence/correctness/foundation-f3-wsl-math-exported-drop/receipt.json)。
- 共同控制instrumentation：[ASan](../../../../evidence/correctness/foundation-f3-asan/receipt.json)、
  [UBSan](../../../../evidence/correctness/foundation-f3-ubsan/receipt.json)、
  [TSan](../../../../evidence/correctness/foundation-f3-tsan/receipt.json)各十二项实际风险。
- 当前增量：[WSL CUDA六项与Windows CPU十二项](../../../../evidence/correctness/foundation-f3-wsl-cuda-deadline/receipt.json)，
  后者包含过期deadline/重复stop/close及借用用户资源保持可用的直接组合。
- [当前完整风险增量](../../../../evidence/correctness/foundation-f3-current-risks/receipt.json)：Air CPU十三、
  Apple六、Win/Air Vulkan各五、WSL CUDA六与共同ASan/UBSan/TSan各十三项，产品candidate固定。
- [原suffix FAIL与纠正](../../../../evidence/correctness/foundation-f3-suffix-retirement/receipt.json)：
  当前97858fbc8的WSL CUDA七、共同三sanitizer各十三、Air CPU十三/Apple六/Vulkan五、Win Vulkan五；
  [Windows CUDA七项](../../../../evidence/correctness/foundation-f3-windows-suffix-retirement/receipt.json)
  直接复验相同失败后缀组合，actual uses和allocator退休通过。
- [explicit direct IO](../../../../evidence/correctness/foundation-f3-direct-io/receipt.json)：当前产品下
  Windows十三项、WSL三sanitizer各十四项、Air十四项；Win/Linux实际对齐读写，macOS原拒绝。
- [当前完整source](../../../../evidence/correctness/foundation-f3-final-source/receipt.json)及
  [Windows CPU/CUDA source/static](../../../../evidence/correctness/foundation-f3-sdk-windows-cpu-cuda-static-f6db4d5b2/receipt.json)；
  下表installed证据均由活动卡逐格链接，十二格当前产品97858fbc8，十格实际数学、软件Vulkan两格按适用性裁决。

## 语义覆盖

| 能力族 / FIX | 正常与边界 | failure / OOM / overflow | ownership / teardown | cancel / deadline / concurrency | 主/正交消费 | 裁决与限制 |
|---|---|---|---|---|---|---|
| 工厂/预算/接纳，01/02 | 有限K、空请求、stage外大范围、描述/offset/backing拒绝；CPU/GPU正常工厂PASS | 冷OOM、adopt失败不consume、已接纳hot OOM、溢出/alias拒绝PASS | File拥有、描述复制、mapping拥有、实际allocator回收PASS | 接纳/观察锁；跨线程导出drop与TSan PASS | byte-only资产/矩阵及真实权重相同公开range/group路径PASS | PASS，当前Win/Air/WSL共同风险与新头独立编译；vendor私有成本UNKNOWN |
| file/mapping→resource，03 | positional/mapping多chunk、GPU-ready、early组PASS；Win/Linux实际direct IO对齐读写PASS | EOF138/128；native后缀320/256、前组保留；direct尾错接纳前拒绝PASS | source caller drop后保活、原writer到实际退休才publish PASS | precancel、GPU pending cancel、过期deadline零IO/native PASS | 四路线真实Direct+Graph及原BF16/F32/已有block编码PASS | PASS；macOS原无portable direct合同，数据旅程NOT_APPLICABLE、拒绝已实际观察 |
| resource→file，04 | ready及真实pending producer、回写→byte oracle→reload PASS | readonly原错、POSIX正数short-write80/96/0、GPU hard/late failure PASS | admitted原Completion保活；失败未启动suffix在原native实际退休后回收PASS | CUDA/Apple真实阻塞producer不提前写；取消/unknown保活PASS | GGUF/Safetensors格式由consumer解释，产品不解析PASS | PASS；原suffix abort6及一次修复保留，Win/WSL CUDA七项实际复验 |
| ready/旧结果/容量，05 | 两组两槽三replay，old ready保持、release后原有限pool可用PASS | capacity拒绝；后组失败保留前组PASS | ReadyGroup/ReadLease保持request/backing；另线程释放后回收PASS | complete与retirement分开；TSan与实际native观察PASS | byte/tag正交旅程与Qwen同公开路径PASS | PASS，适用source/static/shared相同consumer，无整shard隐含缓存 |
| stop/close/abandon，06 | pending非阻塞close为Unavailable；显式drain、重复stop/close PASS | actual late/unknown首错不吞，无fake ready/reuse PASS | 当前Win/WSL CUDA、Air Apple、Win/Air Vulkan实际use与allocator退休PASS | pre/admitted cancel/abandon；deadline/close组合后用户owner仍可用PASS | 未关闭用户整个runtime，原source File继续可用PASS | PASS；取消只停后继chunk，原in-kernel IO可能晚于deadline，不伪强取消 |
| 数学/格式/指南，07/08 | 四路线Qwen四范围完整往返/独立参考、原编码小fixture PASS | checked原始范围、原文件readonly、新临时sink逐byte；原FAIL保留 | 仅72MiB声明范围；旧记录两次collect后同pool reload PASS | 固定K/stage/metadata，native calls/waits实际报告PASS | ordinary Graph与自然route工厂，Direct+Graph三交付PASS | PASS；精确机器/范围，软件Vulkan不签GPU数学，无完整模型或吞吐赢家外推 |

这里的PASS只对应行内具名证据。原适用风险和交付格现已完成，correctness/delivery支持PASS；
不产生GPU sanitizer、未测硬件或完整模型资格，最终CLOSED由terminal record与registry裁决。

## 平台与交付

| 平台/路线 | source真实数学 | F3原生文件风险 | installed static/shared与新seam |
|---|---|---|---|
| Windows CPU | 当前PASS | 当前十三项PASS，含direct IO/deadline/repeat-close | 当前迁址static/shared、七头、两缺席路线及DLL/imports PASS |
| Windows CUDA | 当前PASS | 当前七项PASS，含未启动suffix producer | 当前迁址static/shared、七头、两缺席路线及DLL/imports PASS |
| Windows Vulkan | 当前PASS | 当前五项PASS | 当前迁址static/shared、六头、三缺席路线及DLL/imports PASS |
| WSL CPU | 当前PASS | ASan/UBSan/TSan各十四项PASS，含direct IO/short-write/跨线程 | 当前迁址static/shared、七头、两缺席路线及ldd实际加载PASS |
| WSL CUDA | 当前PASS | 当前七项PASS，含未启动suffix producer | 当前迁址static/shared、七头、两缺席路线及ldd实际加载PASS |
| Air CPU | 当前PASS | 当前十四项PASS；macOS direct IO原拒绝，数据旅程NA | 当前迁址static/shared、七头、两缺席路线、runtime及actual dyld PASS |
| Air Apple | 当前PASS | 当前六项PASS | 当前迁址static/shared、七头、两缺席路线、Metal assets及actual dyld PASS |
| Air Vulkan | 当前PASS | 当前五项PASS | 当前迁址static/shared、六头、三缺席路线及actual dyld PASS |
| WSL Vulkan | GPU数学NOT_APPLICABLE：本轮实际只枚举llvmpipe | native GPU运行NOT_APPLICABLE | 当前source-linked独立consumer构建与迁址static/shared、新六头、三缺席路线、ldd实际解析PASS；不签GPU运行 |

原F1/F2未变的租约/数学/native机制与旧交付证据只能按身份/合同/工具链/机器匹配复用，
不能代签本轮新file seam。未改模块、成熟算子全目录、全库验证与P1按用户要求不选；
为链接真实route构建所需依赖artifact不使该依赖成为新的测试根。

## 最终门

scope及形状审计已有固定基线/八FIX/六owner；产品实现、普通consumer与指南均已有具名位置。
P0源码与实际账单见[成本核对](foundation-f3-p0-cost-review.md)。candidate已冻结，当前微修和
新fixture的平台/sanitizer增量、全部适用交付及LOC self-check均已完成；最终红队处置与证据完整性
审核见terminal record。原F1/F2 CLOSED证据仅签未变合同，本轮新file seam都有具名新证据。
