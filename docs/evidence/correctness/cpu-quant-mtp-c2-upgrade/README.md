# C2 CPU core 升级阶段证据

当前范围/剩余项见 [summary](summary.json)；本目录不签C2整包或整个goal完成。

- 固定CPU-only ggml0.26 ffa4＋原Unsloth IQ1推理扩展；GPU envelope仍用原core。
- Windows shared producer原CPU五组回归共9/9目标通过，包括普通Direct/Graph、pool/workspace、
  allocator/threadpool失败与close。具体GoogleTest fixture跳过信息保留在原始XML/log中；不签未运行fixture。
- 最终定向5/5目标通过；原IQ1_XS/XXS/XXXS的MatMul＋GetRows、重复索引、changed replay及配额退还实际通过。
- Provider软件版本投影后普通CPU绑定检查通过；Native旧WIN不应用到新版peer。
- 原MatMul oracle失败保留：未量化F32对照与原库Q8_K activation不同合同。纠正为原编码后的独立
  decoded-weight对照；8个F32算术舍入单位只覆盖原dot的因子提取，不能覆盖量化语义偏差。

产品候选、测试候选和复用边界均在summary/receipts中。测试/DLL在同目录运行，构建j2、测试j1，
没有新P1、自研竞赛、GPU验收或功耗/UAC操作。

Linux及完整source/static/shared、搬移/offline仍未验；C3/C4公开接线也未完成。
下一批继续复用producer，最终资格不得用本目录的有限测试代签。
