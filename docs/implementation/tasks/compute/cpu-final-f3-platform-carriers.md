# F3 三平台成熟CPU承载

计划包，尚未绑定owner；总合同[CPU最终收口](cpu-final-closure-plan.md)。以F2固定矩阵为输入，启动取实际baseline。
优先复用ggml原CPU、可移植成熟ATen/BLAS/量化程序及已有借入组合，补Mac ARM64和实际平台缺口。
保留原BF16/F16/FP8/FP4/编码与state精度、selected expert/cold转换成本；CPU执行不调用Metal代签。
每个采用库先锁版本/hash/许可证/ABI，完整source/static/shared runtime交付；同进程core归属按F1冻结。
一次批量接线，两入口/正确性/资源/失败/OOM/并发/生命周期/三交付在本包完成；单承载P0，不造赢家P1。
没有合格成熟承载时列精确端口/反例与有限取舍，请求必要决策，不能自己扩数学或删除固定需求。
写产品前真实owner SCOPED/AUDITED，公共Graph BLOCKED不自动恢复；只改有具体反例的合法owner。
