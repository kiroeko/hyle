# F3逐能力验证矩阵

本页由[活动卡](foundation-f3-file-resource-streaming.md)纳入。当前六owner仍AUDITED，
最终candidate与全部门尚未签署；旧收据各自的source身份、范围与原失败继续保留。
这里列出原八FIX的真实覆盖与缺证，不用测试总数或粗估完成百分比替代裁决。

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

## 语义覆盖

| 能力族 / FIX | 正常与边界 | failure / OOM / overflow | ownership / teardown | cancel / deadline / concurrency | 主/正交消费 | 当前缺证 |
|---|---|---|---|---|---|---|
| 工厂/预算/接纳，01/02 | 有限K、空请求、stage外大范围、描述/offset/backing拒绝；CPU/GPU正常工厂PASS | 冷OOM、adopt失败不consume、已接纳hot OOM、溢出/alias拒绝PASS | File拥有、描述复制、mapping拥有、实际allocator回收PASS | 接纳/观察锁；跨线程导出drop与TSan PASS | byte-only资产/矩阵及真实权重相同公开range/group路径PASS | Air共同风险直接运行；当前微小格式清理后的正常owner回归 |
| file/mapping→resource，03 | positional多chunk、immutable mapping、GPU-ready实际bytes、early组PASS | EOF实读138/复制128；native后缀read320/复制256、前组仍可用PASS | source caller drop后保活、原writer到实际退休才publish PASS | precancel及GPU pending cancel PASS；deadline fixture已有，尚未运行 | 四路线真实Direct+Graph及原BF16/F32/已有block编码PASS | deadline直接运行；WSL CUDA六项 |
| resource→file，04 | ready及真实pending producer，write→byte oracle→reload PASS | readonly sink原错、POSIX正数short-write80/96/0、GPU hard/late failure PASS | 所有admitted producer原Completion保活，abandon不能释放实际use PASS | CUDA/Apple真实producer被阻塞时不提前写，取消/unknown待实际退休PASS | GGUF/Safetensors consumer解释格式，产品不解析PASS | WSL CUDA新producer接缝；Air POSIX short-write直接运行 |
| ready/旧结果/容量，05 | 两组两槽三replay，old ready保持、release后原有限pool可用PASS | capacity明确拒绝；后组失败保留前组PASS | ReadyGroup/ReadLease保持request/backing；另线程释放后才回收PASS | actual producer/unknown分开complete与retirement；TSan PASS | 无模型字段的byte/tag正交旅程与Qwen同路径PASS | Air共同跨线程风险；各SDK相同consumer运行 |
| stop/close/abandon，06 | 非阻塞close pending明确Unavailable，显式drain PASS | actual original late/unknown首错不吞，无fake ready/reuse PASS | Windows CUDA/Air Apple/Win及Air Vulkan实际use与allocator退休PASS | pre/admitted cancel、abandon PASS；正常repeat close和deadline组合尚未运行 | close仅借用本pipeline所需资源；新fixture观察用户owner仍可用 | 新公开API fixture及WSL CUDA运行 |
| 数学/格式/指南，07/08 | 四路线原始Qwen四范围完整往返、独立参考；小GGUF或Apple原affine Q4 PASS | checked范围输入，原文件readonly，新临时sink逐byte；原失败全留档 | 大文件不whole read；旧read/write记录两次collect后同pool reload PASS | fixed K/stage/metadata，原native calls/waits实际报告PASS | ordinary hyle::graph与自然route factory、Direct+Graph source PASS | installed static/shared、新头、缺席、迁址、runtime加载、最终红队 |

这里的PASS只对应行内具名证据。新fixture、缺失平台或交付格均UNPROVEN，聚合correctness/delivery
目前INCONCLUSIVE；本页不产生CLOSED或GPU sanitizer资格。无新产品API/layout/算法counterexample。

## 平台与交付

| 平台/路线 | source真实数学 | F3原生文件风险 | installed static/shared与新seam |
|---|---|---|---|
| Windows CPU | PASS | 当前十二项PASS，含deadline/repeat-close新fixture | UNPROVEN |
| Windows CUDA | PASS | 六项PASS；共同格式清理增量待验 | UNPROVEN |
| Windows Vulkan | PASS | 五项PASS；共同格式清理增量待验 | UNPROVEN |
| WSL CPU | PASS | ASan/UBSan/TSan各十二项PASS；deadline新fixture待验 | UNPROVEN |
| WSL CUDA | PASS | 六项原生风险PASS；当前微小格式变化增量待验 | UNPROVEN |
| Air CPU | PASS | CPU共同十三项fixture尚未运行 | UNPROVEN |
| Air Apple | PASS | 六项PASS；共同格式清理增量待验 | UNPROVEN |
| Air Vulkan | PASS | 五项PASS；共同格式清理增量待验 | UNPROVEN |
| WSL Vulkan | NOT_APPLICABLE：实际只有软件llvmpipe，无合格原生GPU | native GPU运行NOT_APPLICABLE | source/安装构建、新头、缺席与运行库解析仍适用且UNPROVEN |

原F1/F2未变的租约/数学/native机制与旧交付证据只能按身份/合同/工具链/机器匹配复用，
不能代签本轮新file seam。未改模块、成熟算子全目录、全库验证与P1按用户要求不选；
为链接真实route构建所需依赖artifact不使该依赖成为新的测试根。

## 最终门

scope及形状审计已有固定基线/八FIX/六owner；产品实现、普通consumer与指南均已有具名位置。
P0源码与实际账单见[成本核对](foundation-f3-p0-cost-review.md)。candidate冻结、当前格式微修的
增量owner门、剩余平台/交付、新fixture、最终红队和LOC复算尚未完成，全部完成后才裁决七门。
