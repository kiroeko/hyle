# G03 CPU/CUDA 公共 CopyBatchUpdate source 旅程

2026-10-14，按[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
执行，AUDITED WORKTREE；[公共consumer](consumer.cc)、[CPU](cpu-final.log)、[CUDA](cuda-final.log)、
[累计记录](receipt.json)、[源码字节](source-identity.json)、[原始日志/编排](raw.zip)。
WSL Ubuntu24.04 / GCC14.2，actual RTX4070 driver617.14 / CUDA13.3，4jobs串行。

同一公共源码，CPU和CUDA各1次16384bytes初始化+64次1024bytes dirty更新，原slot/
generation/metadata不变，foreign control pair拒绝且不消费，双方handle移空、成功writer
返回、重复writer拒绝、原invocation回收。原F2 readonly回读至host再独立比较完整历史。
所有original use实际退休。此处仍显式wait，不代签pending receiver/attention或风险SDK。

实际只重编compute copy_ranges/control/batch_transfer及CUDA四个host backend文件与fixture；
没有nvcc kernel/model math构建或完整模型运行。CPU build含原parent重新configure约69.4s；
没有重写公共构建系统。累计74.464s，四阶段均PASS；后续门与原未知成本仍保留。
