# F3有界文件与资源管线最终收口

状态：CLOSED，2026-10-12。原[活动卡](foundation-f3-file-resource-streaming.md)固定八FIX，
本记录仅关闭F3；原F1八FIX、F2七FIX保持已签CLOSED，合计完成当前goal的23项FIX。
原四路线R1–R7不恢复，完成后停止，不自动扩F4/F5或完整推理引擎。

## 固定范围与候选

六owner：compute、compute_stream、compute_route_cpu/cuda/apple/vulkan。
实际启动baseline为`a932606ee5796328742632e392d5c6c5595ce3fb`；先登记/SCOPED/AUDITED后写产品。
最终不可变产品/普通consumer提交`97858fbc88a6b5a87f109290b956eeaeee632dd2`，
tree `6398479c76fdc9677206ee692943664814463ebd`；随后只有测试、文档、状态与证据变化。
原3b候选由真实suffix producer寿命反证替代，只有共同owner退回AUDITED后有限修复再冻结。
未改变Graph、IO/Sched/Mem、CUDA backend或Provider产品，不新增数学/copy kernel、parser或runtime。

当前原始收据、逐能力风险和每个平台/交付格见[完整矩阵](foundation-f3-verification-matrix.md)，
最终[红队](foundation-f3-final-review.md)和[P0成本](foundation-f3-p0-cost-review.md)支持本裁决。
[最终收据](../../../../evidence/correctness/foundation-f3-final/receipt.json)保存37份源收据与archive SHA
核对、不可变LOC、当前SDK cell目录和原失败边界。

## 原八FIX逐项裁决

| FIX | 实现与实际验收 | Verdict |
|---|---|---|
| F3-01 | 中央内部compute_stream身份/无环依赖，四自然route factory；普通Graph自动闭包，无用户内部开关；新头与source/static/shared通过 | PASS |
| F3-02 | 固定request/group/range/slot/byte容量，File/Mapping拥有/借用、checked描述/offset/padding/alias/backing去重；空请求、OOM、不consume失败通过 | PASS |
| F3-03 | positional buffered/mapping及Win/Linux opt-in direct IO分块到原F2/native资源；两槽、多组、多chunk、early ready、EOF保留真实bytes及前组；Mac direct原拒绝 | PASS；Mac direct数据旅程NA |
| F3-04 | 原producer→resource→stage→原write_all_at；实际GPU被阻塞时不提前写，ready/produced两档、readonly错误、正数short-write prefix、回写/reload逐byte通过 | PASS |
| F3-05 | 原F1 leases、逐组/整请求结果、旧导出句柄占记录/backing，两组两槽三replay；跨线程释放及实际native退休后才复用 | PASS |
| F3-06 | cancel/deadline/stop/close/drain/abandon/late/unknown保持原首错、实际use与quarantine；真实未启动suffix环已修复并Win/WSL七项复验 | PASS |
| F3-07 | CPU/CUDA/Apple/Vulkan真实Direct+Graph；Qwen三个完整BF16投影与F32 norm约72MiB，原小GGUF Q8或Apple affine Q4，实际数据/独立参考及成本账 | PASS |
| F3-08 | 官方Safetensors与成熟GGUF仅在consumer生成checked ranges；普通指南/样例、全部适用迁址SDK/新头/缺席/运行库、P0与红队 | PASS |

## 平台、交付与实际风险

Windows CPU/CUDA/Vulkan、WSL CPU/CUDA、MacBook Air CPU/Apple/Vulkan的当前source普通数学
与原Qwen范围read→early Direct+Graph→write新临时文件→逐byte检查→同pool reload→数学均通过。
每路线logical payload75,497,984 bytes、stage2,097,152 bytes、峰值两slot、三程成功native copies
226,493,952 bytes；其他模型范围不读，不把稀疏验证副本中的hole签为真实模型内容。

五个合格平台/路线组合的static/shared共十格完成实际同consumer数学、新头独立编译、路线头
与package请求缺席、安装/真实rename且原prefix消失。Windows DLL同目录/minimal PATH和imports，
WSL ldd、Air actual dyld全部确认实际加载；MLX/Metal runtime assets按原交付接线。
WSL Vulkan本轮原loader只枚举llvmpipe type4：GPU数学NA，source普通两个consumer构建及
迁址static/shared、新六头、三缺席路线和ldd适用门实际通过。十二installed格范围不混写。

共同风险最终Windows十三项、WSL ASan/UBSan/TSan各十四项、Air十四项；CPU边界/拥有源与
描述/OOM/readonly/短写/旧结果/跨线程/direct IO/deadline/repeated close均有直接断言。
native CUDA Win/WSL各七项、Apple Air六项、Vulkan Win/Air各五项，原producer、取消/丢弃、
后缀错误/unknown观察/late failure与实际uses/allocator退休分别取证，不靠SDK数学代签故障。
正常source及风险fixture只运行实际修改owner/代表消费者；依赖artifact不是新测试根，没有P1。

## 七门与七问

| Gate | 证据与边界 | Verdict |
|---|---|---|
| scope | 八FIX、主Qwen/正交byte资产与矩阵、实际baseline/六owner；原问题源为准入consumer及原API组合缺口 | PASS |
| architecture | 原IO/Sched/Mem/F1/F2/native复合，统一component owner/单图；无第二runtime、万能Queue、模型策略或新kernel | PASS |
| correctness | 完整逐能力矩阵、current native与三sanitizer、actual partial transfer/OOM/lifetime/unknown/deadline/direct IO、新头独立编译 | PASS |
| delivery | 全部适用source/迁址static/shared、ordinary Graph发现、新头/缺席/DLL/ldd/actual dyld，软件Vulkan限定 | PASS |
| usability_docs | 自然四factory、adopt/borrow、ready/produced、错误/容量/close/deadline指南及独立样例，docs/closure检查 | PASS |
| red_team | 独立审查阶段的具名severity/disposition；真实suffix FAIL后有限修复，证据缺格补齐，原FAIL保留 | PASS |
| engineering_performance | 固定K/目录/backing账、实际bytes/native calls/waits，原Task/IO/lease/stack分配计费；有限O(QGR)纠正、无timed赢家 | PASS P0 |

第三方沿原锁定CPU Provider、CUDA/cuBLAS、MLX/Metal、ggml Vulkan/IREE版本及许可合同，本轮
不升级或增加运行依赖；peer/标准来源在原活动卡/设计研究。主与正交public journey同range/group
API，无模型字段特化。下层owner原机制复用；成本/复杂度见P0，沒有同合同性能比赛或未测赢家。
无新的C/Native FFI需求；detail协议仍是同工具链C++共享边界。三交付/导出与普通C++易用性已
实际通过，parser留在应用consumer。七问都有上述源码/测试/consumer/锁定依赖或文档锚点。

## 原失败、成本限制与停止

原File worker affinity、CUDA提交序列/producer寿命、Apple cooperative IO、consumer编译/容量/
GPU布局、native late/unknown、sanitizer configure/ASan1800 TIMEOUT与GCC诊断、TSan初始exit66、
suffix真实abort6、私有GoogleTest重复configure FAIL全保留；局部修复不删原失败。
用户一次ASan追加1800秒已单独计账并完成；库构建按用户纠正后的内存j4/j2/j1、test1/重型门串行。
旧调研/纠正预算不因日期、候选或日志目录重置，没有第二轮研究/全库回放。

vendor私有metadata/物理驻留继续UNKNOWN，不是0；allocator/active uses归零不是VRAM归零。
CUDA sticky terminal物理释放沿原F2限制重启进程，不reset共享primary context。取消/deadline只停
后继chunk，accepted blocking IO/native实际退休后才回收，返回可能晚于deadline。借用Context/
Queue/IO/allocator及导出结果必须满足公开寿命合同；未测机器/版本不外推。

统一LOC快照a97a59ff5的产品与最终97858fbc8一致：全库2,456,943；自实现不含测试170,170；
六owner中央归属实现union26,706/155文件，self-check PASS。数字不是本轮新增量或完成百分比。
实际temporary验证副本删除被自动审批策略拒绝，目录保留，未绕过拒绝；产品与验收不受影响。

六owner进入CLOSED后停止F3。合法reopen trigger仅为新的可复现file/resource权限、范围、ready、
回收/取消/错误或交付缺陷，真实consumer准入的新常用能力、锁定依赖升级或合格成本回归。
历史pending、旧task命令、未完成F4/F5或“再看看”都不恢复本轮。
