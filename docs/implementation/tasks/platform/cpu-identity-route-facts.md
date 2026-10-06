# Platform CPU identity：默认CPU规则需要的缓存事实

<!-- MODULE REVIEW: CANDIDATE -->

- owner platform；baseline `f6470870d3de7f2475e245b6ee2ca76905f82249`；真实consumer为[CPU最终默认规则](../compute/cpu-default-domain-routing-final.md)。
- 缺口事实：CPU Device.name仍为HYLE host CPU，已有cpu_features/cache/topology无法表达已测CPU
  SKU；用户要求硬件/软件条件规则，不能把一台i9的证据用于全部AVX2 CPU。
- 有限I1：cpu.h添加固定value CpuIdentity与cached const accessor；cpu.cc复用baseline cpuid
  helper读取x86 vendor/brand/family/model/stepping，零heap/重复调用零syscall；不可观察时全未知。
- 不包含选路、CPU排名、频率估计/affinity/power更改；已有CpuFeatureSet/其它对象layout不变。
- 验证：Windows/Linux实际vendor/brand及稳定地址/终止字符/并发读取，owner cpu tests/header，
  源码/static/shared CPU默认consumer实际调用；ARM未知分支明确，不宣称Apple型号验收。
- 公共API新增为真实硬件规则需求，不是按benchmark行制造metadata；无新增热成本，P0，不跑P1。
- budget：本次有限实现与一次批量验收，最多定向纠正；不重审platform，不改公共仪器。

I1已审计：cpu.cc已有MSVC/GCC cpuid helper、baseline/IPO-off合同；只追加一次缓存identity，
不复制probe，不改CpuFeatureSet。非x86全空未知，不发系统查询；source/static/shared CPUconsumer将实读。

## 不可变候选

候选 `7090222ac6da73edab35c0f4d8d1bd098748a793`；API/算法/有限规则已固定，适用 P0 与批量 correctness/交付验收继续，
不重跑已结束 ND3 P1，不把新候选称为新计时比较。

## 不可变候选

候选 `393c7545b06f599001a4066876f03595d0bb0b15`；API/算法/有限规则已固定，适用 P0 与批量 correctness/交付验收继续，
不重跑已结束 ND3 P1，不把新候选称为新计时比较。
