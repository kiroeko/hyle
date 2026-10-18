# G03 CUDA 在途 update 的原权限和真实 prefix

2026-10-14，按[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g03-copy-update-owner.md)
执行，AUDITED WORKTREE；[consumer](consumer.cc)、[实际日志](final.log)、[累计](receipt.json)、
[源码字节](source-identity.json)、[原始编排](raw.zip)。WSL Ubuntu24.04 / GCC14.2，
actual RTX4070 driver617.14 / CUDA13.3；4jobs串行，1024bytes CUDA memset+copy。

原cudaLaunchHostFunc checkpoint阻塞真正stream，F1 write use挂到原producer；原F2
bind_mutable+bind_producer接pending source，无新增producer API。CopyBatchUpdate移入
原两个controls；typed destination保原geometry/last_write，源-only mutable binding及
坏ID拒绝，public writer view/take_mutable无ready。末尾独立空logical phase inline，
仍取得前面原native copy的pending prefix。释放gate后只显式update.wait，实际ready
1024bytes全部49；回收原cold invocation/reset后original native guards清空。

这只签CUDA source权限与prefix的成功在途案例，不签后继attention/readonly native
receiver、取消/unknown/放弃/关闭风险或三交付。只新增编译fixture，无nvcc/model kernel；
原cached parent重新configure约60.2s，产品byte SHA未变化，没有公共构建系统改动。

实际累计62.046秒，两阶段terminal0。
