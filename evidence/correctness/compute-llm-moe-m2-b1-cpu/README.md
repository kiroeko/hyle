# M2-B1 CPU 成熟接入证据

`source-audit.json` 记录精确锁定源与有限接入差异，不是执行/覆盖或性能 PASS。
活动范围和 C1–C6 见[当前任务](../../../docs/implementation/tasks/compute/llm-moe-m2-b1-cpu.md)。
`development-windows.json` 与 `development-windows-raw.zip` 是首个实现中检查点：Windows
MSVC Release shared、CPU-only、oneDNN-OFF，7个定向测试目标通过，含20项GoogleTest用例、
3个头自足检查及实际 executor consumer。原始构建失败/纠正与最终通过日志一并保留。
记录绑定产品/测试文件hash，不冒充不可变候选的完整 selector、三交付、跨平台或性能验收。
新成熟需求格仍为0/256，21行 CPU 完整门未完成；自研算子保持冻结。
