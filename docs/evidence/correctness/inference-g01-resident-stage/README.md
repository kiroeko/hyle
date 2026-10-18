# G01 resident SDPA 实现阶段证据

本轮仍AUDITED，不是candidate/closure或整个G01完成。活动卡见
[resident SDPA](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-resident-sdpa.md)。
[receipt](receipt.json)与[raw.zip](raw.zip)保存实际输入/binary hash、命令及全部原失败。

旧relocated static SDK普通CPU入口准备BF16 GQA8/2 D256容量cache的有效视图，会选择
ATen；修正后普通CPU入口实际选择ggml，数学通过。CPU provider全部unit/逐头通过；
新增两layout、F32/F16/BF16、capacity padding哨兵、不同Q/KV长度、独立value depth、
源只读、重复绑定、非法/空view拒绝均在原native路径测试，不用metadata存在猜PASS。

Vulkan局部build通过，但真实定向oracle失败，不能签新fused域：锁定原flash shader把
Q/K/V或概率转成half，F32 accumulator不等于全F32计算。F32与near-zero F16结果超过
原测试容差；数学oracle/容差没有放宽。原失败binary与366MB日志完整压缩保留。
原测试另错误地期待独立Tensor阻止slot重绑；当前测试改保原ResultLease，旧Tensor
数学快照单独检查。数值误差聚合成每case一个报告，避免重复输出；该更新测试尚未重跑，
不能用源码修正签PASS。原native严格精度dispatch是下步有限问题。

MSVC旧test夹具的条件observer未用、double→float显式转换与private vendored include
归属修正只属test-only；不放松HYLE产品WERROR。首次测试初始化拼写失败和全部build
FAIL保留。普通本机4 jobs，末轮CPU局部2 jobs；GPU只跑小型定向consumer，没有运行
完整大模型矩阵。三交付/san/P1与多请求驻留消费仍待完成，不称全部模式对称。
