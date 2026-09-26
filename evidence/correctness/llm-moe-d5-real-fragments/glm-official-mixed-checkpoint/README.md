# F40：GLM 官方混合 FP8 片段检查点（未通过）

候选 `5a298870a9c412f7f7edb918d0b9aa1834dbcf00`，Windows canonical、RTX 4070 / SM89。
本批仅消费者与测试接线，没有新增 HYLE 数学。**完整画像仍 140/176；本页不签新的画像通过。**

## 普通入口批次

[收据](receipt.json)：构建通过（9.17 秒）；Direct、Graph 各 6 个 GoogleTest，各通过 5 个，
`D5GlmOfficialMixedRemaining` 均失败。全部测试 41.36 秒，没有跳过。
[完整日志](native-tests.log)和[JUnit](tests.xml)保留原失败。

- 新接的两种官方 dense FFN、DSA FP8 indexer、稀疏注意力和 index carry 未报错。
- Flash `swiglu_limit=10` 已用成熟操作补足；跨阈值激活检查通过。补正后的共享 FFN、完整 MoE
  和已有 GGUF dense FFN／MLA／NextN 回归通过。
- 失败集中于 Flash MLA 与 NextN：两入口合计 86 个 MLA 输出元素、4 个 NextN 注意力输出元素
  超过已登记 `abs 2e-5 + rel 0.025` 的整链门。没有放宽误差、修改输入以避开失败或改记 PASS。

## 有界数值定位

在同一候选上临时增加只读测试观察器，单独执行 Flash MLA 四轮实际 cache 反馈。
该诊断工作树不是不可变验收候选；诊断测试已从产品工作树移除。
[诊断收据](trace-receipt.json)、[失败日志](trace-tests.log)、[分析](trace-analysis.json)、
[原始数组](trace-arrays.zip)及其[逐文件 SHA256](trace-manifest.json)保留。

对捕获的实际输入独立重算每段，区别局部实现误差与前段误差传播：

- KB、VB 两个 BF16 head projection 在四轮都逐元素完全一致。
- 注意力本段最大绝对误差不超过 `2.384185791015625e-7`。
- 最终 FP8 输出投影以同一实际 VB 输入重算，四轮都通过原整链数值门。
- 独立全链参考仍未通过：最终输出相对 RMS 误差约 0.029%～0.217%，少量近零元素越过逐点门；
  cache 和投影中的舍入差异经过反馈传播。RMS 只用于定位，不是新增验收门。

这些证据支持累计低精度舍入是主要方向；尚未定位最早分歧，也未完成 NextN 的同等级定位。
不能据此宣称 Provider 或完整模型数值已全部合格。下一步只定位最早的量化／舍入边界，保留原门，
不重跑已通过的整批。DSA FP8 kv_b 的冷准备、其它路线、三交付、P1 和 D6 仍待完成。

复现分析：把 `trace-arrays.zip` 内容解压至 `.scratch/validation/d5-f40-trace/`，执行
`python evidence/correctness/llm-moe-d5-real-fragments/glm-official-mixed-checkpoint/trace-analyze.py`。
测试观察器构造见 [trace-setup.py](trace-setup.py)，隐藏子进程及低并行驱动见
[base-runner.py](base-runner.py)。原始数组全为小端 F32；没有模型权重下载或新的大型构建树。
