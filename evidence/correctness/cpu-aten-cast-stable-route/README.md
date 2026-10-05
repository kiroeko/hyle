# CPU Cast/stable route：有限审计证据

[当前卡](../../../docs/implementation/tasks/compute/cpu-aten-cast-stable-route.md)，baseline `b881d2788e0848404c536342b806d6642f3aefa7`。
只运行原已安装SDK的普通Direct/whole Graph尾部diagnostic，没有性能、自研对比或新产品资格。

实际默认gate/Sigmoid与完整gate→MatMul均绑定ggml；Graph只有一个两节点region。
x=-90/value=1e38两者输出0，稳定F32 activation再乘value参考约-7.37461；value=Inf为NaN而参考-Inf。
x=-100同样失效，x=-80参考匹配。Sigmoid(-90)原返回0，F32参考8.19400869e-40。
6个门控组有4个具名反例；不是覆盖比例。完整输出/源码/命令见tail-*，不删除原日志。

Cast与stable Sigmoid原候选还未资格化。TC1–4只是有限审计对象，未改产品，所有产品门UNPROVEN。
