# D5-A6：CUDA 真实片段性能批次

<!-- MODULE REVIEW: AUDITED -->

baseline：`6c9d33415e6a6234247476115252c2b81a0a3880`。承接 D5-A6，产品 FIX 为零；
当前只修改本地 benchmark/验收材料，不重开 CLOSED Graph owner。产品使用 A5 已验收的
`9273bc71` 静态 SDK；测试/benchmark candidate 另行固定后才能采样。

## 有限动作与命题

1. 复用已有模型 recipe，建立一个同时消费普通 Direct 和 Graph 的外部 benchmark。
2. primary 为 Qwen3.5-2B 官方 BF16 FFN：真实本地权重、H=2048、FFN=6144、M=1，
   RMSNorm→gate/up→SiLU gate→down，两个变化输入。
3. holdout 为 DeepSeek V4 Flash 官方 MoE：H=4096、FFN=2048、256 路由候选、top-6，
   七个驻留专家、FP4 专家/FP8 shared、原生动态量化边界，三个变化输入。
   保留既有源声明与 host oracle；只发布消费者最终输出，调试中间输出不限制融合。
4. 批量编译一次，先校验两入口所有输入的最终输出和重复运行、记录实际 Provider/region、
   workspace 与准备成本，再对不可变 candidate 启动一次适用 P1。
5. 修复本 owner benchmark 引用已经删除的 Marlin benchmark 源文件的问题；不恢复旧数学。

测量单位是一次已准备、数据驻留 GPU 的完整同步片段调用。输入槽位循环变化；不测单 kernel，
不测整模型吞吐。Graph 对照是使用相同成熟 Provider 的完整 Direct 片段执行器，**不是裸叶函数**。
两入口都保有所有者、检查失败并收回全部 completion；普通 Graph 允许其自然融合/资源复用。
本比较只能判断这个同步消费者的计划执行成本，不声称 HYLE 数学优于 ggml/cuBLAS/DeepSeek。
Direct 的绝对成本、准备时间及资源账单同时报告；不能从 Graph/Direct 比率推导 Direct 对上游裸 API 的优越性。

## 源码预审与完成边界

| 项目 | Direct | Graph | 计费 |
|---|---|---|---|
| 输入/权重、精度、发布输出 | 同一 recipe、原 oracle | 同左，普通默认编译 | SAME；不改容差 |
| 所有权/并发 | fixture 保有 Context/Queue/常量/临时区/Invocation；单调用方 | 同左，Graph 管理内部临时区 | 初始化单列；两侧同样禁止在完成前复用 |
| admission/提交 | `src/compute/operation.cc:1794` 检查 bindings，进入 invocation phase，提交到同一 queue | `src/graph/executable.cc:7666` 提交已准备 replay | CHARGED |
| 执行和有序依赖 | 所有节点先按拓扑顺序 submit；保有每个 Completion | Graph 原生计划，不拆成熟 region | CHARGED；不得给 Direct 强加逐节点 host wait |
| 完成/失败可见 | 逐个 wait 已提交 Completion，失败也先 drain 全部 | aggregate wait 覆盖所有 terminal segments（`executable.cc:115`） | CHARGED |
| 释放 completion 与 in-flight | `src/compute/completion.cc:151,189` 的 wait/settle 先销毁 provider completion，再 terminal/release | 同一 completion settle 加 aggregate 子项 | CHARGED；返回前销毁所有本轮 Completion |
| 后续回调/跨轮清理 | 不允许把本轮完成延后到下一轮 | 同左 | 校验重复运行后均不 in-flight |
| 准备/上传/最终下载 | 冷准备；所有输入常驻独立槽位 | 同左 | 分别报告，不混进热执行；输出在测量前后检查 |

成本结构：Direct 每非 view 节点一个公开提交/完成；Graph 的事件数、region 和 workspace 从
公开报告记录。两侧内部原子操作、validation、Provider 原生同步和量化/转换均不得旁路。
不使用私有 enqueue 接缝让 Direct 获得不属于普通用户的捷径，也不人为关闭 Graph 融合。
源码已确认 ggml rank-2 BF16 MatMul 不支持该普通组合，因此不可假称一个裸 ggml region 与
默认 cuBLAS+ggml 路线是同一实现；也不可沿用退役 Marlin 的旧合同。

上述 SAME_CONTRACT 仅限单调用方、阻塞到最终结果/失败可见的片段服务；不比较可取消异步
Graph 对象与缺少同等保证的裸库调用。若实际实现无法达成上述完成/资源边界，停止 P1 并先
纠正比较合同，不能看到结果后改口径。

## 验证与停止预算

- Windows W1、RTX4070 SM89、CUDA13.3.73；低并行2，重型串行，DLL邻置，隐藏子进程。
- 先 verify-only：两个 profile、全部变化输入、两入口，输出、in-flight 与失败退出有效。
- 之后沿用统一 nanobench `compare()` 与统一性能仪器；A/A 必须复用实际 Graph 或 Direct
  实例拓扑，覆盖执行双顺序和分配双顺序。当前未运行，不能写 PASS。
- Graph/Direct 的固定工程非劣门 5%，初始 n=15，至多一次 n=31；不因失败换 profile、
  放宽容差、隐藏转换或关闭同步。A/A 不合格先停止，查源码/环境，不启动完整样本。
- 先完成当前两形状；不扩成全算子目录性能重测，不为这个批次重写公共仪器。
- 只改 benchmark 时验证其编译/真实执行和 docs/closure；已通过 CUDA A5 的 24 项交付与
  459 项模块用例不重复。产品缺陷必须另按当前 owner 固定有限 FIX。

当前 verdict：**工作负载A/A FAIL；产品P1 VALIDATION_NOT_RUN，聚合性能INCONCLUSIVE**。
本卡和 harness 不能增加152/176正确性账本或签D6。

## 消费者批次结果

外部静态 SDK 消费者已合并编译并通过 verify-only：两个入口、全部变化输入、两轮重复共
20 次完整片段输出和完成状态检查，CTest 1/1、零跳过。Graph 报告 Qwen 5 steps/0 regions、
DeepSeek 40 steps/0 regions；未声称已经融合。详见
[正确性收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-performance-consumer/README.md)。
原 consumer 编译与 region identity 初始化失败保留；仅修 consumer，不改产品和数值合同。
此结果只完成上述动作1–4的消费者准备/正确性，统一仪器合同与 A/A/P1 尚未运行。

上述为消费者准备时快照。后续`82229660`已完成一次8进程A/A，出现同实现超过10%的偏差及
factor median超过5%，因此正式P1被guard阻止，产品样本为0。测试全退出后GPU仍持续42%–43%
占用，当前测量阶段因环境待恢复；见[完整失败/环境收据](../../../../evidence/performance/graph-cuda-d5-fragments-preflight/README.md)。
保留原失败，不记产品性能FAIL，不升级n。等GPU空闲再恢复一次原预算资格化；期间其它D5工作继续。

## 固定采样接缝

使用原 `hyle-perf.py`，source 指向本仓库 `bench/graph_route_cuda/external`；Git 身份仍由
Windows canonical 仓库提供。fresh 只指新建外部 consumer/control 构建，既有静态 SDK 复用，
不是重新编译 CUDA 产品。SDK 393 个相关文件、全部使用的 Qwen 数据、recipe 与 DLL 的
SHA256 固定在该目录 `inputs.json`，validator 前后核对，不用缓存路径名字代替输入身份。
Direct reference 是同一消费者源码的不可变稀疏检出，validator 逐字节比对；它证明同一个
Direct 调用程序，没有冒称另一份独立数学实现。两侧共享锁定成熟 Provider 数学是本实验的设计。

A/A 固定为 Graph/Graph 与 Direct/Direct，各执行双顺序×分配双顺序，共8个进程；复用
已有 GPU validator 的5% factor median门、10%单进程粗不稳定门，失败不启动完整产品采样。
产品 P1 固定四个顺序/分配 strata，两个模型形状均属于预声明批次；使用原公共 A/A、5%/10%
known-delta 和 timer-floor controls。GPU 片段不增加第二份 CPU text-placement binary。
记录实际 caller、parent affinity mask、priority 与完整 CUDA 环境，不变更系统电源方案或提权。
当前观测为SM89、driver API13040、runtime API13040、编译Toolkit13030；原CUDA13.3安装目录
及其三个DLL哈希与A5一致。这三类版本事实分别记录，不能从编译器版本推测runtime版本。
