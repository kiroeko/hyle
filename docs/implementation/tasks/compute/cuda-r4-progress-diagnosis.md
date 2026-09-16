# R4 CUDA 长停顿：有界进度诊断

## Diagnostic-only audit

- baseline：`c2edb1c2`；[R4 当前卡](cuda-v1-four-route-foundation.md) 六个 owner 保持 BLOCKED。
- trigger：[两次资格停顿](../../../../evidence/correctness/cuda-r4-qualification-progress-stop/README.md)
  已超过 120 秒；第二次观察到大量 CPU 时间，但捕获启动晚于进程自行退出。24 次启动未再次定位，
  不足以排除或归因于任何具体产品/SDK/夹具组件。
- 本轮是独立只读诊断审计，不授权修改产品。原实现、默认规则、P1 门槛和停止裁决继续保留。
- 总预算：从本卡冻结起最多 30 分钟；最多一个不变诊断程序、最多 80 次启动或 8 分钟观察，
  二者先到即止。首个超过 10 秒的进程立即捕获并终止本批，不解释计时结果为产品性能。
- scope：同一完整 FFN 的真实已构建库；Windows 诊断程序增加共享内存阶段标记，记录 cold
  Context/packing/preparation、数值检查、warmup/measurement、teardown 所处位置。产品源码不变。
- observer：提前就绪、在另一个 P 核运行；child 整个进程仍为 CPU2 / AboveNormal。观察者先验证
  自己能在目标存活时读取阶段、线程角色/CPU 时间、RIP/栈候选和小型 dump，避免超时后才启动工具。
  独立 observer 不是给被测 CUDA 进程增加计算核。

## 有限诊断动作

| ID | 动作 | 可接受结果 |
|---|---|---|
| D1 | 校验已有原始日志与源成本；补阶段/实际进程终态观察 | 区分进程运行、观察者延迟与具体执行阶段，不以日志暂未更新证明死锁 |
| D2 | 用真实 owned child 验证捕获器，执行一次上述有界诊断批 | 取得停顿阶段/模块/线程现场，或明确“预算内未定位” |
| D3 | 只沿实际现场审查源码/指令 | 固定可反证的根因与有限 FIX；没有现场不能凭怀疑改锁、GPU 规则或公共仪器 |

若取得产品缺陷，先按原恢复条件对精确 owner 走 SCOPED/AUDITED，固定修复和进度回归后施工；
若取得夹具/采集缺陷，独立登记该事实的有限修正。没有取得因果证据则本诊断 BLOCKED，保留资料，
不自动扩启动次数或再次运行完整 P1。普通 docs/closure selector gate 随记录完成，诊断程序不算
新基础架构完成或新交付。
