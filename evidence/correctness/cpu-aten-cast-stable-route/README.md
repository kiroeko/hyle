# CPU Cast/stable route：有限审计证据

[当前卡](../../../docs/implementation/tasks/compute/cpu-aten-cast-stable-route.md)，baseline `b881d2788e0848404c536342b806d6642f3aefa7`。
以下tail-*记录为原已安装SDK的普通Direct/whole Graph尾部diagnostic；产品资格在后续单列，无性能/自研对比。

实际默认gate/Sigmoid与完整gate→MatMul均绑定ggml；Graph只有一个两节点region。
x=-90/value=1e38两者输出0，稳定F32 activation再乘value参考约-7.37461；value=Inf为NaN而参考-Inf。
x=-100同样失效，x=-80参考匹配。Sigmoid(-90)原返回0，F32参考8.19400869e-40。
6个门控组有4个具名反例；不是覆盖比例。完整输出/源码/命令见tail-*，不删除原日志。

初始SCOPED时Cast/Sigmoid未资格化；以下原程序、产品和候选记录依次更新该阶段，不以probe代签产品。


## 原版准入程序已固定

已锁LibTorch2.8.0 CPU，原DLL身份与原源码/命令/失败日志见original-*和original-manifest.json。
最终 **1,005,272断言**：25容量、993,220 Cast逐位、12,027三dtype Sigmoid。Cast全25组合，
完整F16/BF16/FP8源bits，F32特殊/random/tie上下邻值；独立数学decoder/nearest搜索/NaN位合同。
原to执行有限值转换，原Tensor clamp/where/bit-view/bitwise承担规范中的finite saturation/NaN mapping，
同dtype保持原bits，不写HYLE payload kernel。Sigmoid借原Double kernel后F32与storage舍入。

原fixture早期三个错误（缺header、Byte数量、低精度oracle少舍入）全保留；Byte容量修正后所有Cast
通过，不把无效测试构造误归因vendor。原程序资格不代签产品普通入口、资源、alias/strides或三交付。
该原程序witness完成后TC1–4进入AUDITED并实现；产品与交付资格仍须独立证明，Native/SIMD比较与其它路线继续冻结。


TC1补充：旧SDK默认ggml mixed Cast的NaN bits实际不匹配公开canonical。cast-nan-prior-*绑定原SDK0763fd3e产品；consumer HEAD为7eda8477（只有审计文档，当前产品草稿未编译）。原转换finite数学/region保留，仅NaN storage boundary列入本批有限FIX，所有产品门仍UNPROVEN。


## 当前产品资格（CANDIDATE）

不可变候选`c2df0b8973a80b7140e69983c8e861668483a97c`，与产品`689e9a1b`仅六个fixture不同。
WindowsProvider124/124与ggml58/58已通过；CPU首次158项114通过/7缺fixture跳过/37旧身份失败，
数值oracle未改。六个直接消费者白名单现仅增加精确ATen身份，保留原XML/失败位置与复用证明。
剩余Windows失败批重跑、Graph完整factor、Linux完整factor和两平台三交付尚未完成，不写PASS/CLOSED。
