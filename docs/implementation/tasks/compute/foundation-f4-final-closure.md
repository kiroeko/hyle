# F4 显式状态版本与四路线交付收口

状态：CLOSED，2026-10-12。只关闭F4-01..06及F4-PKG-01；F5六项与整个活动goal仍继续。
新owner compute_state；另九owner只关闭Compute façade新root/package seam，旧算法及原F1–F3
范围保留在[原scope快照](../../../../evidence/architecture/foundation-f4-package-scope/old-scope.json)。
baseline、能力map、有限预算和机器沿[主卡](foundation-f4-state-versions.md)与
[package卡](foundation-f4-package-seam.md)。无依赖升级、模型策略、万能runtime或新数学/copy kernel。

## 不可变候选与有限FIX

candidate29d3c7a3ebe81f0e627986e2e8c8a393aa7832c8，产品实现等价b1e37d75c。
候选后只补测试/文档/证据，src/include/cmake与候选无diff。最终
[收据](../../../../evidence/correctness/foundation-f4-final/receipt.json)核对18个SDK格、10个实际
source/native fault旅程、三sanitizer及共同24项的原日志hash、产品输入和累计预算。
SDK没有缺格；软件WSL Vulkan两格只签适用包门，不代签GPU数学。

| FIX | 实际验收 | Verdict |
|---|---|---|
| F4-01 | immutable schema、owned opaque entry身份、完整成员预检、bounded目录/backing账、host-owned snapshot、容量/OOM | PASS |
| F4-02 | begin分享/pure final readonly replace，只对dirty完整成员原F2 clone；opaque raw clone/不同资源约束明确拒绝 | PASS |
| F4-03 | 全成员ready/无pending writer、同Store expected-base发布冲突、并发单winner、旧version/reader/native保活 | PASS |
| F4-04 | 原F1 writer/native writes成功证明；accepted DataLoss/late MLX Internal、cancel/abandon/unknown/suffix/close、原native实际退休 | PASS |
| F4-05 | 主GDN recurrent+conv+KV layout五成员两step，正交资产集合；四route Direct/Graph及独立oracle、旧值/冲突/dirty更新 | PASS |
| F4-06 | 自然hyle::compute入口/typed原route，source/static/shared、新头/缺路线/迁址/loader、guide与P0 | PASS |
| F4-PKG-01 | façade增加新root，原九owner源码/原公开头无diff；18 installed格含独立native-only缺CPU资格 | PASS |

## 逐能力与风险矩阵

共同测试：[Windows24项](../../../../evidence/correctness/foundation-f4-core-candidate/README.md)，
[ASan/UBSan/TSan各24项](../../../../evidence/correctness/foundation-f4-sanitizers/README.md)。
三san使用候选29产品/测试，GCC14.2/Ubuntu24.04/shipping，无suppression，不借其代签GPU。

| 能力族 / public journey | 正常与边界 | failure / OOM / overflow | ownership / move / teardown | cancel / concurrency | 主 / 正交消费者 | 平台 / delivery |
|---|---|---|---|---|---|---|
| schema/import/immutable version | PASS完整/重复/遗漏/错schema/store/描述/约束/empty schema | PASS quota/snapshot容量；checked目录大小及generation饱和前拒绝，未声称遍历2^64次 | PASS ID保schema、目录refs/escaped reader/native-use与whole-pool去重 | PASS closed拒新接纳；schema/版本只读分享 | PASS五成员 / 资产集合 | PASS实际source与18SDK |
| draft/share/replace/finalize/head | PASS readonly分享/pure输出/完整ready/expected-base冲突/旧值 | PASS metadata替换失败保旧base、pending/error writer拒finalize、固定版本容量 | PASS move/abandon/旧version与独立native输出 | PASS head并发单winner、close不关callerContext | PASS两step recurrent/conv / asset close后Direct | PASS四route原对象，各自然交付 |
| dirty clone/ticket/writer | PASS只复制目标完整byte成员、原F2 output/drop/reset/try_write | PASS foreign writer/root/generation、pending admitted write、actual producer error；unknown不取writer/rearm | PASS原native源/目标last-use、F4 controls全释放后实际退休、clone pool保活 | PASS logical cancel/no GPU抢占/未启动suffix、explicit rearm、native wait与close | PASS conv/单KV块 / 分段资产更新 | PASS CPU/CUDA/Apple/Vulkan实际native及独立SDK |
| package/API/cost | PASS新三头独立TU/普通入口/缺路线/迁址 | PASS缺CPU及其他route具名拒绝，原失败与累计预算保留 | PASS shared loader新prefix；borrowed endpoint/allocator原合同 | NOT_APPLICABLE新deadline接口/FFI；未新增这些接口或承诺 | PASS同普通consumer，不另造benchmark形状 | PASS18格；software Vk仅包 |

schema/Store/Head/Version/Draft/Identity、CloneRoute/Ticket/Writer全部公共入口分别映射上述
F4-01..04；copy/report/rearm是明确冷对象或显式native观察，无未映射benchmark-only API。
API没有deadline或任意rollback承诺；drain是显式等待自己copy，不等待业务reader。

native故障证据：[CUDA Windows](../../../../evidence/correctness/foundation-f4-native-risk-windows-cpu-cuda/README.md)、
[CUDA WSL](../../../../evidence/correctness/foundation-f4-native-risk-wsl-cpu-cuda/README.md)、
[Apple Air](../../../../evidence/correctness/foundation-f4-native-risk-air-cpu-apple/README.md)、
[Vulkan Windows](../../../../evidence/correctness/foundation-f4-native-risk-windows-cpu-vulkan/README.md)、
[Vulkan Air](../../../../evidence/correctness/foundation-f4-native-risk-air-cpu-vulkan/README.md)。
CUDA/Apple用原native任务gate后接受真实F4 copy，证明pending cancel/abandon/concurrent close；
Apple late Internal不能finalize。Vulkan沿已有private设备fn table注入原fence/drain观察错误，
未证明退休不回收live clone、不冷rearm，恢复原观察后实际退休、旧源保持。无新产品fault hook。

ordinary source：[Win CPU+CUDA](../../../../evidence/correctness/foundation-f4-source-windows-cpu-cuda/README.md)、
[Win CPU+Vulkan](../../../../evidence/correctness/foundation-f4-source-windows-cpu-vulkan/README.md)、
[WSL CPU+CUDA](../../../../evidence/correctness/foundation-f4-source-wsl-cpu-cuda/README.md)、
[Air CPU+Apple](../../../../evidence/correctness/foundation-f4-source-air-cpu-apple/README.md)、
[Air CPU+Vulkan](../../../../evidence/correctness/foundation-f4-source-air-cpu-vulkan/README.md)。
GDN/conv非zero两step与独立double oracle、host layout snapshot、expected-base冲突、旧native值、
实际native readonly endpoint clone、单dirty成员更新、取消/冷rearm与资产Graph完整闭环。
page layout只是F4 state元数据；F5 attention尚未因此实现。

## 实际安装交付

每格真实install→rename迁址→only-prefix find_package Compute/Graph消费者→三个公头独立TU→
适用runtime/loader→未包含route具名拒绝。native-only没有CPU factor或CPU数学回退。

| SKU / 系统 | Static | Shared | 实际资格 |
|---|---|---|---|
| CPU+CUDA Windows | [PASS](../../../../evidence/correctness/foundation-f4-sdk-windows-cpu-cuda-static/README.md) | [PASS](../../../../evidence/correctness/foundation-f4-sdk-windows-cpu-cuda-shared/README.md) | RTX4070；受限PATH实际运行/imports/库hash |
| CPU+CUDA WSL | [PASS](../../../../evidence/correctness/foundation-f4-sdk-wsl-cpu-cuda-static/README.md) | [PASS](../../../../evidence/correctness/foundation-f4-sdk-wsl-cpu-cuda-shared/README.md) | RTX4070；ldd真实新prefix |
| CPU+Vulkan Windows | [PASS](../../../../evidence/correctness/foundation-f4-sdk-windows-cpu-vulkan-static/README.md) | [PASS](../../../../evidence/correctness/foundation-f4-sdk-windows-cpu-vulkan-shared/README.md) | 原native Direct/Graph实际运行 |
| CPU+Apple Air | [PASS](../../../../evidence/correctness/foundation-f4-sdk-air-cpu-apple-static/README.md) | [PASS](../../../../evidence/correctness/foundation-f4-sdk-air-cpu-apple-shared/README.md) | M4原MLX/Metal；实际dyld新prefix |
| CPU+Vulkan Air | [PASS](../../../../evidence/correctness/foundation-f4-sdk-air-cpu-vulkan-static/README.md) | [PASS](../../../../evidence/correctness/foundation-f4-sdk-air-cpu-vulkan-shared/README.md) | M4/MoltenVK actual native |
| native Apple Air | [PASS](../../../../evidence/correctness/foundation-f4-native-sdk-air-apple-static/README.md) | [PASS](../../../../evidence/correctness/foundation-f4-native-sdk-air-apple-shared/README.md) | 无CPU；原native完整state journey |
| native Vulkan Windows | [PASS](../../../../evidence/correctness/foundation-f4-native-sdk-windows-vulkan-static/README.md) | [PASS](../../../../evidence/correctness/foundation-f4-native-sdk-windows-vulkan-shared/README.md) | 无CPU；实际native与缺CPU拒绝 |
| native Vulkan Air | [PASS](../../../../evidence/correctness/foundation-f4-native-sdk-air-vulkan-static/README.md) | [PASS](../../../../evidence/correctness/foundation-f4-native-sdk-air-vulkan-shared/README.md) | 无CPU；实际native与dyld |
| native Vulkan WSL | [PASS包](../../../../evidence/correctness/foundation-f4-native-sdk-wsl-vulkan-static/README.md) | [PASS包](../../../../evidence/correctness/foundation-f4-native-sdk-wsl-vulkan-shared/README.md) | only llvmpipe/software；GPU数学/运行未执行，不写PASS |

Air全部实际身份Mac16,12/M4/16GiB/macOS27.0.1/Clang21，只Air，没有Mini代签。
库按实时可用内存j4≥12GiB/j2≥8GiB/j1≥4GiB、test1，重型跨机器串行。同配置binary树与
锁定source cache复用；只新state、实际直接消费者、package seam，未回放全库算子或P1。

## 七门与独立红队

| Gate | 证明与限制 | Verdict |
|---|---|---|
| scope | 六F4 FIX/一个package FIX、两纵切、baseline/有限预算冻结，F5未混入 | PASS |
| architecture | 新state owner位于F1/F2之上，无Mem/Compute/Graph反向边；借原native，不新建runtime/kernel | PASS |
| correctness | 上述完整能力/风险矩阵、common24/三san、actual native取消/错误/unknown/旧结果 | PASS |
| delivery | 五ordinary source profile、18 installed格/新头/迁址/loader/缺路线，native-only独立证明 | PASS |
| usability_docs | [state指南](../../../user/state-versions.md)与普通public consumer，borrow/readonly/schema/容量/冲突/opaque/cancel/rearm限制 | PASS |
| red_team | root分离的候选后审查，见下表/收据；无未处理产品反例，所有证据缺格已补齐 | PASS |
| engineering_performance | [P0](../../../../evidence/performance/foundation-f4-p0/README.md)与MSVC生成代码/统一LOC，无timed方向/engine吞吐声明 | PASS P0 |

2026-10-12 root（未启子agent）按活动卡一遍分离候选后红队、至多一次有限证据纠正，重新核对
API反向映射、store/schema identity、OOMforecast、borrow、pending writer、last-native-use、
expected-base锁与末引用、cold factory环/恢复、显式wait观察、成本、native-only包与guide。

| Finding | Severity / disposition | 证据 |
|---|---|---|
| CANDIDATE时还有native-only/WSL/SDK风险缺格，不能由混合source代签 | EVIDENCE / RESOLVED：18格和适用native实际journey全部齐全 | final receipt及各原job |
| borrowed QuotaAllocator被fixture在native use回调后、原CUDA reaper末metadata清理前销毁 | TEST_LIFETIME / RESOLVED：allocator先于endpoint构造、释放F4controls后显式有界观察原清理；Win/WSL定向通过 | WSL原exit-6/hash/Git fixture/累计时间保留，产品未改 |
| Apple LeaseUse已退休不证明原owner已丢retired Job，过早余额检查 | TEST_OBSERVATION / RESOLVED：显式Context.close join后查allocator余额，不给F4新增隐式wait | Apple原FAIL与四casePASS |
| 冷ledger含二次有限扫描，末record释放O(E)，错误Status可能分配 | COST_BOUNDARY / ACCEPTED且明示：不称全过程O(1)/无分配/paged hotpath性能 | P0源码/原MSVC对象及guide |

其余逐项未发现需退回AUDITED的产品反例。版本身份owner保活、foreign handle拒绝，head
临时identity不是末引用；成功critical section交换O(1)，previous在外drop。typed route不拥有
Store/Version回指环；opaque raw clone拒绝，原native source endpoint按实际owner声明。
新head/directory并发独立TSan，native pending/close沿实际GPU取证，不让san代签native退休。

第三方许可证/版本沿[固定peer研究](../project/foundation-f45-reference-research-2026-10.md)及原锁；
无新依赖/升级。SGLang/vLLM/llama成熟owner及主/正交旅程证明准入；原F1/F2/数学BORROW。
同工具链C++无新FFI；自然三交付、新头/缺路线、guide与成本解释完成七问。

P0：目录O((V+D)E+B+C)，begin/finalize/末record释放O(E)，版本handle/identity共享O(1)，
payload不复制。冷ledger preflight含O(B²+EB+E²)，默认B128/C4，非任意大paged热路径。
dirty clone O(实际成员bytes)，host metadata深复制，native waits/private UNKNOWN分账。
没有逐Tensor/普通Invocation新增字段、worker/reaper、submit隐式规划或timed WIN/PARITY。
统一候选LOC：全库2,505,378行/6111files，self不含tests171,728/700，完整compute_state
1147行/5files（包含三公开头）；self-check PASS，不手加旧模块LOC或用测试数量算完成率。

原核心configure/compile、ctest零匹配、CUDAconv域、native fixture误入口/自阻塞/rearm断言、
TSan启动前exit66与setarch恢复、WSL allocator -6全部保留。预算configure300/build1800/
test120/consumer180按原gate累计；没有因目录/候选/证据提交重置。源copy只Windows→验证副本，
反向只取日志证据。原旧owner源树和原public heads从bc9基线无diff。

reopen仅由可复现状态身份/权限/寿命/交付缺陷、真实新consumer准入、相关锁升级或合格成本
回归触发；历史TODO不是授权。F4签显式完整schema的机制，engine自行定义模型checkpoint、
RNG/position/page policy。F5 native paged/ragged/BatchPlanning与完整推理引擎性能未由F4代签；
总goal继续F5，并在全部完成后按用户单机高性能单/多请求目标做剩余基础库与代码量评估。
