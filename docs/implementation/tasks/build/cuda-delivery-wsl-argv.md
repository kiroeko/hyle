# CUDA 交付入口：WSL 原样参数

<!-- MODULE REVIEW: CLOSED -->

- baseline：`b3691265`；本批工具，产品/registry不变。
- trigger：Linux configure首次FAIL；`--`经默认shell重解析，将
  `-DHYLE_COMPONENTS=compute;graph;simd`拆成三个命令。原失败与日志保留在固定任务记录。
- 有限FIX：WSL launch改`--exec`，不经shell重解析；补真实WSL含分号参数反例。
- 预算/验收：一次修复、轻量行为测试与同ID一次configure纠正；通过项复用、失败预算不重置。
  不重跑Windows、不改数学/P1/公共工具。配置未通过前不继续build。

AUDITED：原日志尾部明确出现`graph: command not found`与`simd: command not found`，
源工程只得到compute选择而consumer仍要求graph/simd。固定唯一launch修复为`--exec`；
Windows配方不变，Linuxfixture复制步骤没有launch argv，因此其PASS可复用。

`--exec`修复后14项轻量测试通过，真实WSL子进程收到完整`compute;graph;simd`并写回
canonical XML。同ID第二次configure实际PASS（14.328秒），原FAIL（13.969秒、rc127）
保留；fixture为REUSE，未重跑Windows。工具任务CLOSED，余下实际交付仍继续待验。
