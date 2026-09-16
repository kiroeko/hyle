# R3 CPU 默认 Direct / Graph 候选验收

候选：`79e96cebeb176c48d66f6b6a156fa9d910b89d82`。本目录目前是进行中的验收记录，
R3 仍为 CANDIDATE；尚未宣告路线完成。当前施工卡见
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
