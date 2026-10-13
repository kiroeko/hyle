# G01 Vulkan 固定原生记录：实现阶段

本轮 AUDITED RR-01/02/03 的有限证据；[收据](receipt.json)、[Air 原始结果](air-result.json)
及 [全部尝试](raw.zip) 保留初次失败、临时诊断和最终实际通过。Windows 相关目标编译
通过，最多4路，未运行本机GPU；实际GPU数学来自 MacBook Air Mac16,12。

native128次同buffer内容更新/metadata owner切换、显式失效、strict/normal缓存交错、
非法维度和零quota通过；公开72个原数值片段、64步offset/backing重绑、旧输出寿命、
独立slots及65-element行间padding通过，原oracle不放宽。临时诊断字段与API已移除。
固定attention图没有Add/RmsNorm consumer时跳过原backend无条件1024-byte partials
clear，并在返回/异常恢复size；否则Air UMA的host memset会使记录不满足无staging资格。

source认证通过，依赖锁仅变更Vulkan patch/NOTICE两个hash；selector对公共lock采取
保守全局计划，原计划完整保留。实际改动范围为此Vulkan承载及直接消费者，未运行项
没有标为PASS。此证据不签全模块、三交付、sanitizer或性能；原合格P1 FAIL原封保留。
