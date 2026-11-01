# CUDA 推理性能

更新：2026-11-01。**当前表对应未提交的设备端接受／连续接力候选，尚未通过完整验收。** 普通 decode 汇总领先约2.9%，MTP约6.1%；请求级 prefill 汇总落后约24%–27%，MTP启动和显存仍有缺口。

本页集中维护最近结果、计时口径和复现入口；更新时替换当前表，不追加实验流水。任务顺序见[推理看板](../inference-progress.md)，统一方法见[性能测量](README.md)。

## 当前候选：普通／MTP × prefill／decode

Windows／RTX 4070、Qwen3.5-2B BF16、纯CUDA、greedy、完整词表、K1 MTP；正常构建，单活动／驻留会话、C2048、prefix off、自动decode接力、默认 `--warmup-shapes lazy`。8个不同prompt（聊天、代码、中英文、摘要等），用户输入长度为105／107／79／113／188／109／98／106，每轮生成64个token。

- Hyle：基于 `3b824a922` 的 `device2b-continuous3-WIP`，源码候选未提交；产物为 `.scratch/build-compute-v2-dev/bin/hyle.exe` 及配套DLL。
- 参考：冻结 llama.cpp `d9e03f1`＋GGML `456172e`＋资源／异常清理补丁的匹配输入consumer；不是最新版标准 `llama-bench`。普通模式使用target-only GGUF，MTP模式使用含MTP权重的GGUF。
- 同一窗口四臂轮换交错，每臂3个独立进程，共12个有效计时进程，均exit0；无profiler或内存observer。各臂内部512个输出ID稳定；跨引擎及MTP模式的自然输出有差异，不是逐token等算量回放。

单位tok/s，格式为**中位数［最小–最大］**；范围不是置信区间。

| MTP | 阶段 | Hyle当前候选 | 冻结参考 | Hyle／参考−1 |
|---|---|---:|---:|---:|
| Off | 请求级prefill | 2836［2808–2852］ | 3876［3796–3964］ | −26.8% |
| Off | 完整decode区间 | 104.59［104.52–104.62］ | 101.64［101.61–101.87］ | +2.9% |
| On | 请求级prefill | 2768［2765–2790］ | 3645［3598–3705］ | −24.1% |
| On | 完整decode区间 | 127.73［127.63–127.78］ | 120.39［120.19–120.54］ | +6.1% |

**口径：** prefill分子为912个target输入token＝905个用户token＋7个上轮pending flush，不加draft行；decode分子为504＝512个输出减去每轮prefill产生的首token。模型加载和ready前准备不计入这两列；请求期间发生的载体初始化、提交、等待及结果发布仍在区间内。默认prefill含新形状首次准备，不能称预热后的纯计算吞吐。decode保留ready后实际发生的录图等冷费用，不能从中直接扣除局部计时。

| MTP | Hyle启动到ready | 参考启动到ready | Hyle完整进程tok/s | 参考完整进程tok/s |
|---|---:|---:|---:|---:|
| Off | 1.614s | 1.701s | 69.93 | 68.94 |
| On | 2.735s | 1.721s | 67.12 | 76.53 |

完整进程速度为512个输出／父进程墙时，包含加载、准备、全部prefill／decode、报告和退出。MTP的decode领先没有使本负载的完整进程领先；完整进程速度仍低约12.3%。

<details>
<summary>8轮逐轮数据：每格为 prefill / decode 中位数（tok/s）</summary>

每轮decode分子为63；prefill输入长度（含pending flush）为105／108／80／114／189／110／99／107。

| 轮 | Hyle Off | 参考 Off | Hyle On | 参考 On |
|---|---:|---:|---:|---:|
| 1 | 3596 / 104.34 | 1120 / 101.65 | 4756 / 121.66 | 1111 / 118.42 |
| 2 | 2773 / 104.98 | 5534 / 101.33 | 2598 / 129.57 | 5167 / 123.59 |
| 3 | 2062 / 104.89 | 4426 / 101.70 | 2052 / 122.36 | 4082 / 109.36 |
| 4 | 2696 / 104.82 | 5918 / 101.96 | 2653 / 125.03 | 5382 / 127.00 |
| 5 | 3967 / 104.42 | 7312 / 101.84 | 3744 / 135.24 | 6428 / 126.94 |
| 6 | 2640 / 104.56 | 5630 / 101.44 | 2524 / 125.22 | 5168 / 113.42 |
| 7 | 2407 / 104.19 | 5304 / 101.78 | 2295 / 128.61 | 4688 / 119.64 |
| 8 | 2585 / 104.24 | 5385 / 101.79 | 2455 / 135.31 | 5046 / 126.54 |

逐轮范围、录图与关联准备时间保留在[本窗原报告](../../../.scratch/inference-mtp-gating-20261101/prefill-decode-matrix2/summary.txt)。

</details>

**验收边界：** 本窗MTP第4轮为125.03对127.00 tok/s，逐轮速度标准尚未通过。同一候选另用独立内存进程测得，相对 `88e0f96b6` 的采样峰增加真实负载108MiB、重复负载104MiB，超过64MiB门槛；这些是独立窗口的Windows PDH dedicated＋shared采样下界，不是上面12个计时进程的内存测量。

本机原始数据：[主表与逐轮范围](../../../.scratch/inference-mtp-gating-20261101/prefill-decode-matrix2/summary.txt)、[逐轮TSV](../../../.scratch/inference-mtp-gating-20261101/prefill-decode-matrix2/turns.tsv)、[双方版本与全部命令](../../../.scratch/inference-mtp-gating-20261101/prefill-decode-matrix2/plan.txt)、[独立内存结果](../../../.scratch/inference-mtp-gating-20261101/device-memory3/analysis.txt)。原始日志仅留本地 `.scratch`，不作为随仓库发布的证据包。

## Prefill冷准备诊断（独立窗口）

同一普通模式二进制、同一8轮负载，默认lazy／现有 `--warmup-shapes planned`／冻结参考三臂交错，每臂3个独立进程，9进程均exit0。planned只提前创建已知形状的执行载体，不额外执行模型，不提前录制prefill图；Hyle两种配置的完整输出ID一致。

| 配置 | 启动到ready中位数 | 8轮prefill总耗时中位数 | 汇总prefill tok/s | 第2–8轮prefill tok/s |
|---|---:|---:|---:|---:|
| Hyle默认lazy | 1.599s | 327.9ms | 2781 | 2706 |
| Hyle planned诊断 | 1.728s | 206.3ms | 4420 | 4552 |
| 冻结参考 | 1.702s | 234.6ms | 3887 | 5717 |

载体准备前移使ready增加约129ms、请求级prefill减少约122ms；Hyle完整进程吞吐为69.77／69.76 tok/s，基本不变，不能称为计算提速。第2–8轮planned仍落后参考约20.4%，且不同形状仍是首次使用，不能标为纯稳态结果。当前默认路径尚未改成planned；本诊断不能替换主表的默认配置数字。

之前4082 tok/s来自154-token单轮输入；旧八轮测试重复154-token prompt。它们与本页不同形状的八轮汇总不是同一负载，不能据跨窗口绝对值判断代码退化。首轮参考的首次使用成本也会影响汇总，不能据汇总宣称通用热prefill领先。

本机原始数据：[对照结果与全部范围](../../../.scratch/inference-mtp-gating-20261101/prefill-cold-ab1/summary.txt)、[版本与命令](../../../.scratch/inference-mtp-gating-20261101/prefill-cold-ab1/plan.txt)。本诊断与主表分属不同窗口，不混合样本。

## 已提交版本的历史对照

<details>
<summary>2026-10-31 单轮／重复两轮历史基线</summary>

**2026-10-31 14:00:28～14:02:46，Windows／RTX 4070，Qwen3.5-2B BF16，正常Release shared GPU产品。** 默认完整词表、greedy、普通与K1 MTP均自动接力；P154、每轮64输出、C1024、单活动会话、prefix off。单轮／两轮分别测量，不是双用户合批。

每工况双方各3个独立计时进程和1个独立内存进程，共32进程均exit0，无profile或补样。表中为Hyle／冻结参考的中位数，速度tok/s、显存GiB。decode按原producer排除prefill首token，保留首次实际capture；综合按完整会话实际输出计数。

| 工况 | prefill | decode | 综合输出速度 | dedicated采样峰 |
|---|---:|---:|---:|---:|
| 普通单轮 | 2174.88／1593.68 | 99.07／101.96 | 90.57／89.06 | 4.040／3.728 |
| 普通两轮 | 2690.41／2372.93 | 101.33／102.08 | 94.24／93.98 | 4.171／3.728 |
| MTP单轮 | 2129.37／1508.05 | 121.47／135.73 | 108.30／113.02 | 4.237／3.896 |
| MTP两轮 | 2472.61／2343.57 | 133.88／140.88 | 120.04／124.76 | 4.372／3.896 |

普通decode落后约2.83%／0.73%，MTP落后约10.51%／4.97%；显存采样峰高约8.38%～12.23%。本窗建立基线，没有证明新的稳定提速。完整秒数、样本范围、原报告与命令在[正常产品对照](../../evidence/performance/inference-default-product-2026-10-31.json)。

</details>

## 最短测量路径

先按[构建指南](../setup/README.md)得到正常Release产物，再按[引擎指南](../../user/inference.md)准备本地输入。同一份含MTP权重的输入可运行两种模式：

```text
<build-dir>/bin/hyle --inputs .scratch/qwen-input --backend cuda --mtp off --steps 64 --capacity 1024
<build-dir>/bin/hyle --inputs .scratch/qwen-input --backend cuda --mtp on --mtp-max-drafts 1 --steps 64 --capacity 1024
```

Windows使用实际`hyle.exe`。这是单臂复测入口，不会自动得到上表，也不会下载模型。正式对照还需固定同一输入／参考版本、交错独立进程，按[四项报告口径](README.md#推理对比固定报告)记录prefill、完整decode、独立内存和连续会话综合速度。

复现10-31历史表的精确输入、双轮schedule、双方命令及独立内存观察器直接取原证据中的对应记录；本地缓存入口为`.scratch/inference-plan-cold-facts-20261030/window.py`。先核其固定输入与产物仍可用，不把历史脚本当当前自动执行指令，不为复现隐式下载权重。计时进程不装profiler；ready采样握手只放在独立内存进程。

## 归因时保留的边界

| 问题 | 已知事实 | 后续比较需区分 |
|---|---|---|
| 连续执行与准备 | 已有capture、设备token接力和正常后继发布；warm提示不是权限 | 首次录制／稳态复用、主机准备／真实GPU等待；不能相加重叠step时长代替墙时 |
| MTP额外计算 | 历史单轮窗有2个被丢弃后继；当前设备端选择候选的本页负载无丢弃，拒绝草稿的验证工作仍计入实际行数 | 接受率和无效计算，与提交／重录／读回间隙分别归因 |
| GPU投影与融合 | 部分投影融合及GEMV已使用；减少episode不等于减少kernel | 实际carrier／算法、M1／M2形状、流量及完整消费者净收益 |
| 内存 | 历史表ready未知；当前另列独立采样峰差值，固定池归还不是物理释放 | 加载、准备、各轮prefill／decode及释放后的同步阶段值 |

`actual_draft_rows`计入原Graph已接纳的活跃draft输入，包括之后丢弃／失败及恢复；恢复行是子类，不能再加一次。它不包含padding、provider内部准备、未返回Completion的部分失败或硬件FLOPs。完整定义见[工作量对齐](README.md#对齐工作量)。

本地TensorRT-LLM `46d9966b54d9`的BF16 low-M专用路径要求100≤SM<110且显式启用，本机SM89不能照搬；其BF16 logits后转F32也不能代替Hyle现有F32 scores合同。借鉴源码不等于已复现其收益。

## 可选实验与历史

[显式32K词表实验](../../evidence/performance/inference-draft-vocabulary-2026-10-30.json)在有限场景出现收益，额外head／map占128MiB／128KiB，但它是显式代码域先验，只支持列明的greedy路径，不代签默认full-V、随机接受或其它模型。有限K2信号没有改变默认K1。

较早长输入、B2、CPU、失败候选、主机／GPU跟踪及逐片A/B保存在[历史调查](../../evidence/history/decode-2026-10-31.md)。它们各自绑定当时版本；现行页面不再按日期追加实验流水。

<a id="三轮共享缓存的四项基线2026-10-21"></a>
历史入链：[2026-10-21三轮共享缓存基线](../../evidence/history/decode-2026-10-31.md#三轮共享缓存的四项基线2026-10-21)。
