# M2-B1 CPU 成熟接入证据

`source-audit.json` 记录精确锁定源与有限接入差异，不是执行/覆盖或性能 PASS。
活动范围和 C1–C6 见[当前任务](../../../docs/implementation/tasks/compute/llm-moe-m2-b1-cpu.md)。
`development-windows.json` 与 `development-windows-raw.zip` 是首个实现中检查点：Windows
MSVC Release shared、CPU-only、oneDNN-OFF，7个定向测试目标通过，含20项GoogleTest用例、
3个头自足检查及实际 executor consumer。原始构建失败/纠正与最终通过日志一并保留。
记录绑定产品/测试文件hash，不冒充不可变候选的完整 selector、三交付、跨平台或性能验收。
新成熟需求格仍为0/256，21行 CPU 完整门未完成；自研算子保持冻结。

`index-windows.json` 与 `index-windows-raw.zip` 保留后续 C1-I 标准库 Scan/TopK 检查点：
同一机器/配置的7个目标、28项GoogleTest用例通过，普通 Direct/Graph 都已实际绑定。含整数
溢出不发布输出、任意轴/奇数字节stride、稳定 NaN/ties、原始位模式保留、两输出与重放。
完整 selector、四环境、三交付及P1尚未运行，不以本记录关闭B1或增加完整覆盖格。
