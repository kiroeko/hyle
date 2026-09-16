# Graph prepared-region 成本修正验收

本目录绑定 [定向任务卡](../../../docs/implementation/tasks/compute/cpu-graph-replay-cost-audit.md)，
产品候选 `1a3a8d0e40a2a17250119369331c712718a4d880`。后续合同/状态提交没有修改产品投影。
新增回归证明 resource validation 留在冷阶段，变化输入正确；错资源、原地址换 Context Queue、
关闭 Context 仍拒绝。完整 Graph/Executable suite 继续覆盖 async、接纳前缀失败和生命周期。

| 环境 | 定向 source/真实 consumer | 搬迁后 shared Graph/Sched | 搬迁后 static Graph/Sched |
|---|---|---|---|
| Windows / MSVC 19.50 | 7/7 PASS | 2/2 PASS | 尚待运行 |
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

WSL sanitizer 与新的 Graph P1 尚待执行完成，Windows static 消费将使用该冻结静态产物。
当前目录不签整个 R3 PASS，也不把普通验证当成速度证据。原 Direct P1 和未变化模块验收继续复用。
