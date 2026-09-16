# R3 CPU 默认 Direct / Graph 候选验收

初始候选：`79e96cebeb176c48d66f6b6a156fa9d910b89d82`；当前产品投影为 `d9712e68`。
本目录是进行中的验收记录；R3 总体验收仍 INCONCLUSIVE，尚未宣告路线完成。当前施工卡见
[CPU 四路线基础任务](../../../docs/implementation/tasks/compute/cpu-v1-four-route-foundation.md)。

## 已完成的本轮验证

| 环境/范围 | 结果 | 证据 |
|---|---|---|
| WSL Ubuntu 24.04 / GCC 14.2 / Native + oneDNN AUTO | source 23/23 PASS；含新 routes、直接消费者、相关公共头 | linux-delivery-logs.tar.gz |
| WSL 动态库 SDK | tests-off 实际重构建，安装后搬移；普通 Direct/Graph/Sched consumer 3/3 PASS | 同上 shared 日志 |
| WSL 静态库 SDK | 实际切换为 static 重构建，安装后搬移；相同 consumer 3/3 PASS | 同上 static 日志 |
| MacBook Air / Apple Clang 21 / Native AUTO | source 19/19、relocated shared 3/3 PASS；static configure FAIL，尚未签静态交付 | air-first-logs.tar.gz |

WSL source 日志确认默认 Graph 实际选择 oneDNN region，1/2 worker 分别执行 12/24 个 worker
task，各有两次 nested bulk。变化输入、独立 oracle、并发 Invocation、关闭回收通过，重放的
metadata allocation 为零。显式 Native 比较路径也执行。没有从 source 结果推定安装消费通过。

该出站源码 archive SHA-256：
`3316ee7f1b820a30512c2ef9e04cabfdc38110600f67e95854966e7c6ed18015`。
Windows 为产品真相，WSL 未修改源码。构建使用 Ninja parallel 2，重型验证串行。SDK 阶段关闭
测试并重构建 HYLE；未变化的 oneDNN 对象按正常 CMake 依赖规则复用。外部 recipe 和初始失败
日志一并保留：初次 tests-on + INSTALL=ON 被正确拒绝，修正 recipe 后才得到以上结果。调整
driver 以复用 SDK 构建目录时保留了首段 source build log，续建没有丢弃既有对象或测试错误。

## 性能资格失败的保留

`direct-formal-invocation-failure.zip` 是实际完成采样但资格失败的原包。调用误加 `--formal`；
合同要求的是工程 P1，profile 没有预注册 Formal 八项工具链身份。四个 workload 的 5% 非劣门
及所有 controls 通过，但 build-contract FAIL，聚合性能 INCONCLUSIVE，包为 DIAGNOSTIC。
**这不是 P1 PASS**。不得修改旧 manifest、补写未知 SDK 身份或以它签 Formal 结论。

`graph-first-p1.zip` 是 Graph 的首次工程 P1：资格及 A/A、delta5、delta10、timer-floor、placement
全部 PASS；12 个 workload 中 11 个非劣门 PASS。串行 17×129×257 对完整 Native 图的两层
非劣门 UNRESOLVED，n=31 达到合同上限，总性能 INCONCLUSIVE。所有分层中 Native 均快约
4%～5%，不是零差异；两 worker 的 natural sequence 和 Native 完整比较则显示 WIN。
原 n=15/n=31 数据都保留；不再对旧候选采样。

Air static 失败来自缺席 oneDNN 的私有依赖仍进入新 route 的静态安装接口，source/shared
通过不能代替它。三个 owner 已转入父卡链接的有限纠正卡，处理静态安装边和连续激活分派。
Direct 的正确工程模式、Graph 修正 P1、Windows 最终三交付、Mac 静态及 Mini、定向 sanitizer、
CPU OFF 新头/闭包隔离仍须实际完成。本目录不签尚未运行的门，也不据此关闭 R3。

## 范围与规模

`implementation-plan.json` 保存 selector 的完整计划；中央 manifest 引发的全库保守选择不等于
本轮实际运行范围。适用范围按施工卡限定为 CPU/Graph owners 及真实交付边界，不声称全库 PASS。

统一 `loc-report.py --self-check`：该候选 Compute + Graph 全部实现 **31,018** 行，不含测试；
自实现不含测试 **126,203** 行，全库代码 **700,200** 行。LOC 包含既有 backend/provider，
不是本轮新增量或完成百分比。原始统计在 loc-79e96ceb.json，包校验在 sha256.json。

## 有限纠正候选

产品与 benchmark 固定在 `d9712e68d2b812449a170f474d71a347a27b0ba1`，三名直接 owner 的
状态由 correction card 管理。`correction-initial-checks.zip` 保留修改后的 Windows ON 定向
7/7（实际 HYLE_WERROR=ON）、OFF static source/installed 各 3/3、selector 与脚本 LOC 自检。
OFF static 初轮 recipe 使用了无效的 HYLE_WARNINGS_AS_ERRORS 名称，因此这组只证明静态交付
与正确性，不作为 WERROR 证据；后续 recipe 已改为实际 HYLE_WERROR。

上述正确性发生在冻结提交前，产品文件与此候选一致；其后只补 Graph benchmark 的实际 workload
A/A，MSVC /W4 /WX 语法检查通过。A/A 和修正候选 P1 尚未裁决，不能据此签性能 PASS。

包内 correction-projection.json 绑定原 `79e96ceb` archive 到新候选的四个产品/测试覆盖文件，
并列出 891 个 include/src/tests/samples/cmake/root CMake 文件的 SHA-256。WSL、Air、Mini
向外更新后逐文件匹配；旧日志单独保留。未改变的外部 oneDNN 编译对象可复用，变更配置的
HYLE 目标仍由 CMake/Ninja 实际重建。包内 Unix recipe 是后续验证入口，不代表它已经执行。
当前统一脚本 Compute+Graph 实现计数 31,023 行（不含测试），不是重构完成率。

`graph-correction-aa-stop.zip`：修正候选的 16 次实际 workload A/A 中，一个双 worker 同实现
比较 ratio=0.9027134979，超过既定 gross 停止线。所有原始输出保留；benchmark 的 cold guard
阻止首个产品进程计时，所以没有修正候选的产品性能采样结果。Graph 两个 owner 已裁决
BLOCKED，R3 总体验收 INCONCLUSIVE。普通交付与 sanitizer 将独立完成，不改写这个结果。

## 修正产品的后续普通验收

下面是 d9712e68 产品投影的实际结果，补充上述初始候选记录。均使用实际 `HYLE_WERROR=ON`、
Ninja parallel 2，跨机器重型验证串行。两 Mac 仅证明正确性/交付，未签性能。

| 环境 | source | relocated shared / static | 额外风险 | 原始包 |
|---|---|---|---|---|
| Windows / MSVC 19.50 / oneDNN AUTO | 23/23 PASS | 各 3/3 PASS | DLL 相邻部署后通过，最初 loader 失败保留 | windows-correction-logs.zip |
| WSL / GCC 14.2 / oneDNN AUTO | 23/23 PASS | 各 3/3 PASS | 新 routes / Sched consumer 的 ASan、UBSan、TSan 各 3/3 PASS | linux-correction-logs.tar.gz |
| Air / Native AUTO | 19/19 PASS | 各 3/3 PASS | 缺席 oneDNN 的静态安装边已实际通过 | air-correction-logs.tar.gz |
| Mini / Native AUTO | 19/19 PASS | 各 3/3 PASS | 同一普通 Direct/Graph/Sched consumer | mini-correction-logs.tar.gz |

WSL 首次 strict GCC 构建发现六处测试 GoogleTest 宏 dangling-else；只覆盖 test-only `6353adf4`
花括号修正后通过。包保存失败、修正 projection receipt 与最终日志。产品仍为 d9712e68；
外部 oneDNN JIT 没有被 sanitizer 插桩，sanitizer 只签 HYLE route/consumer 的声明范围。
所有源码来自 Windows 出站 projection；验证副本未反向提供产品修改。Windows 三交付后，
CPU OFF / CUDA OFF 的 source 与 installed consumer 各 2/2 PASS；新 route 及全部 CPU provider
头/export 均无泄漏，CPU OFF + oneDNN ON 明确拒绝，见 windows-cpu-off-logs.zip。工程 P1 待完成。

## 共核诊断与新实验合同

`parallel-aa-diagnosis.zip` 保存固定 16 进程的 power-saving/balanced × caller 共核/独立核对照。
`parallel-aa-lowlatency.zip` 保存用户要求的 8 进程 low-latency 对照。仅 copied benchmark 的 cold
配置有变化，链接 d9712e68 原静态库，记录全部输出、per-round 数据、源码/库 hash 和 host probe。

low-latency 时 caller 共核的双 worker 片段中位约 70.274ms，独立后约 30.780us；每组仍各有
一个同实现比较超过 10% gross 门，独立组最高 ratio=1.10195。两组都是 DIAGNOSTIC，均非 PASS。
证据支持纠正共核资源配置，尚未证明 Sched 内部某一条等待语句就是全部长延迟的原因。

后续明确使用三个独立 P 核的新 v2 合同，见
[独立 caller 验收卡](../../../docs/implementation/tasks/compute/cpu-graph-external-caller-qualification.md)。
这是有限的新合同资格工作，原 v1 失败保留；新 A/A 未通过就不能计时产品。公共工具补丁另有
独立 task 和 11 项 PASS，不是 Compute/Graph 产品进度。最新统一 LOC 与 self-check 在
loc-external-caller-worktree.json：Compute+Graph 实现仍为 31,023 行，不含测试。

修正后的独立源码复核见 [r3-correction-source-review.md](r3-correction-source-review.md)。Windows
内部单测首次遗漏全部 runtime 路径，第二次仅补 HYLE 所在目录仍缺 GoogleTest DLL；两次 loader
失败不算产品测试。按用户要求将同一 build 的 21 个 runtime DLL 复制到 8 个测试 exe 目录，
加载预检通过后才恢复。随后 23/23 正确性通过；source SDK 和安装消费者始终通过既有
`hyle_stage_runtime_dependencies` 将依赖放在 exe 旁。后续 Windows 验证使用非交互错误模式
及 120 秒测试超时，不再依赖弹窗处理错误。产品源码未为此改变。
