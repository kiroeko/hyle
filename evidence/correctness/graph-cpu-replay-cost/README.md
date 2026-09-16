# Graph prepared-region 成本修正验收

本目录绑定 [定向任务卡](../../../docs/implementation/tasks/compute/cpu-graph-replay-cost-audit.md)，
产品候选 `1a3a8d0e40a2a17250119369331c712718a4d880`。后续合同/状态提交没有修改产品投影。
新增回归证明 resource validation 留在冷阶段，变化输入正确；错资源、原地址换 Context Queue、
关闭 Context 仍拒绝。完整 Graph/Executable suite 继续覆盖 async、接纳前缀失败和生命周期。

| 环境 | 定向 source/真实 consumer | 搬迁后 shared Graph/Sched | 搬迁后 static Graph/Sched |
|---|---|---|---|
| Windows / MSVC 19.50 | 7/7 PASS | 2/2 PASS | 2/2 PASS |
| WSL / GCC 14 | 7/7 PASS | 2/2 PASS | 2/2 PASS |
| MacBook Air / AppleClang | 6/6 PASS | 2/2 PASS | 2/2 PASS |
| Mac mini / AppleClang | 6/6 PASS | 2/2 PASS | 2/2 PASS |

所有构建 HYLE_WERROR=ON、Ninja parallel 2，重型验证串行。x64 AUTO 接入真实 oneDNN region，
两 Mac 使用当前 Native 支持组合，不能冒充 Apple MLX/Metal 路线验收。Windows 的实际项目/测试 DLL
均先放到 exe 同目录，禁止错误弹窗并设置超时；source 包保留加载预检和 DLL SHA-256。

windows-source.zip 保留首轮 source 完整构建/测试、DLL 记录、已验三个源文件 hash、CMake cache、
实际 recipe 与 selector；windows-shared.zip 保留安装、搬迁和两个普通消费的实际日志。
mac-air.tar.gz / mac-mini.tar.gz 各自保留 source/shared/static 全部日志、cache 和 recipe；891 个
product/build/test 文件均与 Windows 候选投影逐一匹配，仅日志和收据回传，没有反向同步产品代码。

WSL ASan、UBSan、TSan 均各 5/5 PASS，覆盖 Graph/Executable、oneDNN region、默认 Graph 与真实
Sched consumer；linux.tar.gz 保留三交付及全部 sanitizer 的实际日志、cache 与 recipe。外部 oneDNN
JIT 不宣称被插桩。Windows static 使用同一 P1 冻结产物安装/搬迁，Graph/Sched 消费 2/2 PASS；
windows-static.zip 保留日志、cache 与实际 DLL 相邻部署。全部预声明普通验证到此完成。
[新 Graph P1](../../performance/graph-cpu-route-p1-v3-7fa6dc0a-w1-win/README.md) 的资格 PASS，
11/12 非劣 PASS，整体 INCONCLUSIVE；约 12 微秒 decode 的一个 CI 下界仍越过 5% 门。
当前目录不签整个 R3 PASS，也不把普通验证当成速度证据。原 Direct P1 和未变化模块验收继续复用。
