# Graph prepared-region 成本修正验收

本目录绑定 [定向任务卡](../../../docs/implementation/tasks/compute/cpu-graph-replay-cost-audit.md)。
Windows source 七项 PASS：Graph/Executable、oneDNN region、默认 CPU Graph、真实 Sched consumer、
两个普通 C++ source 消费者。新增回归证明 backend resource validation 留在冷阶段，变化输入正确；
错资源、原地址换 Context Queue、关闭 Context 仍拒绝。MSVC / HYLE_WERROR=ON / Ninja parallel 2。

windows-source.zip 保留完整构建/测试、DLL 相邻部署 hash、加载预检、源文件 hash、CMake cache 和
实际 recipe/selector。普通验证的源码来自 Windows canonical worktree；随后产品冻结身份由任务卡记录。
未完成的平台、Graph SDK/sanitizer 和性能仍待实际验证，本目录不签整个路线 PASS。
