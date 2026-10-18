# D5 CUDA：完整片段 P1 工程验收

**测量资格 PASS；两个完整片段均为 PARITY，5% 非劣门 PASS。**
候选 `15dae3023b94c97b8a538347be623384bc40e930`；产品 SDK 仍为
`9273bc71017a1febb69db240b6ea8970b88dd77c`；Direct reference 为
`848022b64663f4cbbf6da83315e8c6c0b3e38d3a`。
Windows W1 / RTX 4070 SM89，Driver API 13040、Runtime API 13040、Toolkit 13030，
MSVC 19.50；P核调用者、既定native-runtime-pair资源集、执行与分配双顺序。

| 完整片段 | 四种顺序的 Graph 中位耗时差范围 | 最不利顺序的区间允许退化 | 裁决 |
|---|---|---|---|
| Qwen3.5-2B 官方 BF16 FFN，M1/H2048/N6144 | -1.28% ～ -0.49% | 4.33% | PARITY / PASS |
| DeepSeek Flash 官方 FP4/FP8 MoE，H4096/N2048/top6 | +0.87% ～ +1.20% | 4.03% | PARITY / PASS |

耗时差由公共仪器的Graph/Direct吞吐比倒数换算，负值表示Graph耗时更低；准确区间见
[summary](run/summary.json)。MoE小幅中位落后如实保留，未声称数学领先或所有调用零开销。
PARITY是本合同5%等价带内裁决，不表示精确相等或已经证明性能方向。
两入口均使用相同成熟Provider，测量包含提交、必要转换/同步、错误检查和本轮所有权释放；
冷准备、输入上传和验证下载单列，不把完整Graph与裸算子比较。

## 本次恢复与停止

用户结束背景串流并明确授权在仍有普通桌面占用、波动的环境测试。没有改变电源计划、
提升权限或结束用户应用。旧A/A失败及42%–43%背景记录继续保留于
[原预检收据](../graph-cuda-d5-fragments-preflight/README.md)，不能唯一还原旧失败原因。
本次仅按原卡恢复一次相同8进程工作负载A/A，全部通过，才进入正式产品比较。

- 四个执行/分配strata各n15，共60个产品进程；两个完整形状均在每个进程执行。
- 公共控制30个进程，四项控制全部通过；没有触发n31，没有增加样本或重复试到通过。
- 前后输入、产物、主机资格和全部validator通过；复用原产品SDK，没有重跑A5或产品测试。
- `hyle-perf.py verify`对原manifest及引用文件严格重放通过：`QUALIFIED`、
  `paths_verified=true`、当前政策适用、provenance `EXACT`。

[manifest](run/manifest.json)、[控制项](run/controls.json)、
[A/A](run/d5-cuda-workload-aa.json)、[严格重放](run/strict-replay.json)保存原内容。
[索引](index.json)绑定原路径、原SHA256及归档SHA256；日志以gzip保留原字节。
本次首个启动被上次Vulkan中断留下的mutex标记阻止，尚未构建或采样；核实Windows/WSL无测试、
原owner PID已退出且marker身份一致后，保存标记并仅清理该文件，再启动本次唯一实际采样。
没有接触其它用户进程或锁。

本结果只签当前CUDA合同两个完整片段，不外推所有算子、其它GPU或整模型吞吐。
本批到此停止；其它路线残余覆盖和D6仍依当前账本处理。

[统一LOC](loc.json)：Compute+Graph为60,373实现行/241文件，不含测试。
已核对路径和构建身份并[清理](cleanup.json)被替代的旧CUDA失败构建与旧Vulkan中断构建，
释放672,170,862字节（约641MiB）；原日志和输入归档保留。保留本次合格CUDA构建供严格重放，
以及最新Vulkan恢复构建和各路线SDK，不删除模型、依赖缓存或其它用户文件。
