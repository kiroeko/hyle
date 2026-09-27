# D5-A5：CUDA 三交付与 DLL 修复验收

**CUDA A5 PASS。** 产品候选 `9273bc71017a1febb69db240b6ea8970b88dd77c`，
测试名单纠正 `dbf6c085`。完整画像仍为152/176，CUDA正确性44/44；本批不签A6性能或D6。
汇总见 [summary](summary.json)，文件哈希见 [files](files.json)。

## 实际运行

| 交付 | Direct + Graph | 隔离条件 | 原始收据 |
|---|---|---|---|
| 源码父工程 | 8/8，零跳过 | 不可变源码archive；两个普通公开产品入口 | [source](raw/source-provider-runtime/receipt.json) |
| 静态SDK | 8/8，零跳过 | 搬迁prefix、隐藏producer运行目录、PATH仅Windows/System32 | [static](raw/static-provider-runtime/receipt.json) |
| 动态SDK | 8/8，零跳过 | 同上 | [shared](raw/shared-provider-runtime/receipt.json) |

每个入口分别执行真实Qwen3.5-2B BF16 FFN、causal-conv/GDN状态反馈、DeepSeek官方完整
FP4专家/FP8共享MoE及FP4/FP8量化往返。使用原消费者、原容差、真实内宽及变化输入，
没有添加新数学或以设备枚举/复制替代计算。测试host oracle显式使用SIMD及锁定ggml-base。
源码静态producer只构建一次，静态SDK直接安装同一产物；shared复用canonical构建树。

三个SDK DLL（cudart、cuBLAS、cuBLASLt）由公开`hyle_stage_runtime_dependencies()`邻置。
每种交付的`cuda-dll-origins.json`核对实际文件与Toolkit原文件SHA256，`dll-audit.json`
核对PE加载闭包；没有缺失的非系统导入。各自`compile-link-commands.log`及
[链接来源汇总](raw/link-provenance.json)证明installed消费者没有引用canonical产品或producer路径。
验证机为Windows/RTX4070 SM89、MSVC19.50.35728、CUDA Toolkit13.3.73；未代签其它OS/硬件。

## 受影响门与唯一测试纠正

selector计划及逐项skipped reasons见
[完整选择](raw/selected-cuda-gates/selection-plan.json)。实际factor/header命令按target与label
合并执行，交付门由本卡上述真实CUDA消费者承担；没有生成或声称一份通用selector `run PASS`。

八个测试程序共459个GTest。首轮458通过，唯一失败是默认Provider名单仍期望5个，漏了
`b421cb85`已注册且已资格化的DeepSeek Provider；见
[原完整日志](raw/selected-cuda-gates/native-tests.log)。产品和交付输入未变，仅给旧预期数组补
该具名ID；该用例随后通过，见[定向纠正](raw/selected-cuda-correction/receipt.json)。
其余458项不重复运行，原FAIL不覆盖。Graph175/175、各Provider与backend均通过，无跳过。
三组独立头、13个输入工具测试、docs和closure通过；相应日志保存在纠正收据目录。

## 产品修复、成本与停止边界

实际SDK隔离运行揭示CMake把系统CUDA import library声明为UNKNOWN，且共享Provider导出
丢失私有cuBLAS依赖。backend的同一私有config供源码和installed使用，给现有SDK库补SHARED
runtime文件身份；两个Provider补对应私有SHARED依赖，使标准CMake导出保留其DLL闭包。
原SDK版本、includes、链接对象、驱动安装前提和公共staging入口保持一致。

P0：只变CMake配置/交付metadata，无C++、算法、对象布局、执行期分配、同步或选路变化。
红队复核了导出私有依赖、三个DLL来源、installed路径、未选路线OFF、测试可选Provider头、
fixture依赖及原失败保留。SDK及第三方许可证来源不变，驱动仍由系统提供。
普通用法见[交付指南](../../../../docs/user/distribution.md)。

首次外部编译的可选FlashInfer头、测试helper链接，以及SDK配置/导出失败分别保留在`raw/shared*`
目录；这些均未被写成数值PASS。两个失败SDK的CMake导出快照保留后可清理其临时二进制副本。
归档后已清理两个旧SDK及两个验收完成的独立consumer构建目录，见[清理记录](cleanup.json)。
保留有效SDK与唯一源码/static producer供后续P1复用。

统一脚本统计Compute+Graph实现 **60,373行/241文件**，不含测试与第三方源码，见[LOC](loc.json)。
其余路线A5、全部A6及24个未完成画像格仍待验收；本结果不代表整模型推理或整个goal完成。
