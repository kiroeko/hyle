# CPU 剩余数值接缝：重复 ScatterAdd 与 RoPE 数据域

<!-- MODULE REVIEW: AUDITED -->

- owner `compute_provider_native`；baseline `e80cf22ccf285934693f40da9cc558b457283d7b`，Windows canonical。
- 主消费：公开 F16/BF16 重复索引聚合，及已有整数 positions/F32 frequency RoPE 的低精度 data。
  正交：两 index widths、axis、byte stride、prefix/尾部 raw bits、empty、alias、越界无写/恢复、变化
  输入、Completion、close/OOM/并发与三交付。不实现模型加载、GPU、新数学或 SIMD 优化。
- trigger：Native half ScatterAdd 实际把 32 个重复的微小更新逐项舍入为零贡献；旧测试验证
  ordered storage rounding，与当前公开“不得逐更新 storage rounding”合同相反。
  ATen 原 Add 已接并实际给出正确结果。RoPE 原 ATen 程序支持 native typed data，但 guard 只放 F32。
- RoPE 精度审计纠正：RN 初案误把 Double phase oracle 当所有模型的相同计算程序。
  [官方 Qwen3.5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
  的 text rotary 强制 Float32 phase；现有 ggml lowering 与此计算级别一致。RN2/RN3退役方案撤回，
  未提交其产品修改；保留 Native I32/ggml 原 RoPE。不同 phase 仅作差异诊断，不签数值 FAIL 或性能
  胜负。原 I64 整数保真域沿已资格化 ATen Double phase，不能反推所有 Float32 模型都应采用它。
- 架构/资源：复用原 Program、typed Tensor/Byte packing、managed/private unknown、原 Double
  sin/cos/clone/prefix copy 与 supplied allocator；不改公共 API、组装器、原库/SDK/线程或治理工具。
- 预算：已完成一批无计时诊断及一次同合同纠正；两个有限产品FIX，批量产品/测试后一次 Windows/
  Linux owning factor、普通CPU/Graph、header、source/static/shared 与 source identity；j2、重型串行、
  邻置DLL/隐藏启动。最多一次具体 fixture/产品失败纠正，不按dtype各自建工程或重复P1。
- P0计入所有转换、Tensor分配、packing/clone/完整index检查和真实wait。未计时，不签自研WIN。
  此子批完成后继续[父卡](cpu-native-final-domain-verdict.md)剩余合格域的一份有限P1。

## 最终有限FIX

| ID | 产品与范围 | 验收 |
|---|---|---|
| RN1 | Native F16/BF16 ScatterAdd冷拒绝，删除其逐项舍入执行；F32 Add、各storage Assign及SIMD保留 | 两cold stages；旧错误预期迁成熟ATen；重复/alias/越界无写/恢复/empty与普通Direct/Graph |
| RN4 | 原ATen RoPE允许F16/BF16 data，frequency仍F32；原Double phase/clone-tail数学不改，版本programs-6 | 三floating/I32-I64、两layout/axis/odd stride、sNaN tail bits、empty/alias/并发/close/OOM；普通整数RoPE两平台三交付 |

RN2/RN3明确REJECT_NOT_SAME_PHASE_CONTRACT，非延期自研优化。ggml无产品修改，其定向审计与关闭
独立记在[RoPE精度卡](cpu-rope-phase-contract-audit.md)，不借当前Native授权改其产品。

七问：原ATen pin/许可证/SDK闭包复用；已有lowp能力接原库；复用runtime与checked资源；P0含
完整原库成本；无公共API或C/FFI；普通入口及三交付实际验收；guide说明成熟Add和两种phase边界。
适用门仍待运行，当前AUDITED不签完成。旧ND-R1已验收事实不变，CPU整个goal仍活动。

候选e4f4bd44已停止变化后启动Windows批次：首次编译通过，Provider143/144。新增lowp RoPE的
未旋转sNaN tail bits暴露公共numeric入口先转F32的接缝，原件保存为provider-first-failed.xml。
RN4定向保持RoPE的原typed Tensor，再交原Double prefix程序；其它numeric程序仍转F32，不放宽
数学/bit门、不写新payload算法。这是本轮具名一次产品纠正，之后只增量编译及必要批次。

纠正候选`775b2bedace019de423edc5d8a6477772b8ef3ec`的Windows owning factor144/144、header1/1、
直接CPU消费者16通过/2缺少旧Qwen fixture skip、Graph22/22、ordinary source/shared-installed消费
实际通过。初次local driver误选全CPU模型目录，运行272.89秒后有界中止，原件保留，不算整套PASS；
改用`CpuDefaultContext.*:*Rope*:*Scatter*`与相应普通Graph/具名RoPE消费者，不改公共治理工具。
两平台驱动都使用同一有限范围；已过factor/header不因收窄消费者而重跑。Windows静态交付与Linux
仍在验证，不签RN全部完成。统一脚本在固定候选计实现61,386行/249文件，不含tests/vendor。
