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

Direct 的正确工程模式和 Graph 完整片段 P1 另行取证；Windows 最终三交付、两 Mac、定向
sanitizer、CPU OFF 新头/闭包隔离仍须实际执行。本目录不签尚未运行的门，也不据此关闭 R3。

## 范围与规模

`implementation-plan.json` 保存 selector 的完整计划；中央 manifest 引发的全库保守选择不等于
本轮实际运行范围。适用范围按施工卡限定为 CPU/Graph owners 及真实交付边界，不声称全库 PASS。

统一 `loc-report.py --self-check`：该候选 Compute + Graph 全部实现 **31,018** 行，不含测试；
自实现不含测试 **126,203** 行，全库代码 **700,200** 行。LOC 包含既有 backend/provider，
不是本轮新增量或完成百分比。原始统计在 loc-79e96ceb.json，包校验在 sha256.json。
