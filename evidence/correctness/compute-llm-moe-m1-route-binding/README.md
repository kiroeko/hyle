# M1-R：冷模型条件与原生路线接缝验收

本证据只签 [M1-R 任务卡](../../../docs/implementation/tasks/compute/llm-moe-m1-route-binding.md)
的 R1–R5/R5a/R5b。完整候选为 `c482ce1b33cefecdb949c8b0ec89f9309c3e86ac`，
最后C++运行时变化为 `7016443b1bf895fc4300958e12099e89385de954`；夹具固定于`d4a4b389`，
R5b只协调Apple构建期同目录资源复制。
基线为 `6933817f`。这是准备时的事实传递、资源保活与交付验收，新增模型覆盖格为 **0**。

## 已验内容

- ModelIdentity 一次冷分配拥有三个独立字符串；空对象不分配，复用 Mem/Util 所有权。
  Direct、delegated、Graph operation/region 及 Apple/Vulkan 原生程序保留同一 workload。
- 固定条件验证完整匹配、缺失/未知、版本/配置/phase/region 不符和合法通用候选；
  重复执行不重入冷提案/编译。条件夹具不作为模型优化或完整 Provider 覆盖证明。
- CUDA 在 Context 创建时保存设备、UUID、SM、driver CUDA API、runtime 和 Toolkit 事实。
  `driver_cuda_version` 是 CUDA API 支持级别，不是驱动安装包/补丁号。
- 纠正既有 Apple shared 状态依赖接缝：使用 MLX 导出的公开 depends，冷检查原生 stream；
  保留同一上游 primitive，没有新增 copy、kernel 或算术。
- 多个消费目标写同目录`mlx.metallib`时使用构建进程锁；执行期不加锁。原真实构建失败保留，
  16组双copy诊断未复现，记录为风险结构修复，不冒称诊断复现成功。

## 运行与交付

| 环境 | 原生/头检查 | 三交付 |
|---|---|---|
| Windows | selector 原选 28 门由原始与有限续跑收据共同覆盖 | CPU Compute/Graph；补充 CUDA+Vulkan，均 source/static/shared |
| WSL | 21 项，CPU+CUDA（包含 oneDNN 消费） | CPU+CUDA+oneDNN，source/static/shared |
| MacBook Air | 22 项，CPU+Apple+Vulkan | CPU+Apple+Vulkan，source/static/shared |
| Mac mini | 18 项，CPU+Apple | CPU+Apple，source/static/shared；不签 Formal 性能 |

安装消费先搬移 prefix、隐藏 producer，再构建普通 Direct/Graph consumer；Windows DLL 邻置。
四环境公共大小实测：ModelIdentity 8 字节、WorkloadContext 40 字节，PreparedOperation 与
Executable 各 8 字节；CUDA 环境快照为 296 字节。每种交付的原始输出都在归档中。

Windows ASan 与 WSL ASan+LSan、UBSan、TSan 通过共同 owner/Graph 用例及新增 Direct 绑定用例。
WSL TSan 首次在测试体之前发生地址映射冲突；同一插桩二进制在进程级 `setarch x86_64 -R`
下通过，未修改系统设置或重编译。原失败和续跑记录均保留，CPU 插桩不代签 GPU driver。

## 成本、身份与限制

性能级别 **P0 PASS**，仅签复杂度、一次冷拷贝/分配及复用所有权；没有计时、SOTA 或算子胜负
结论。Invocation/submit 未增加 workload 参数、模型比较、设备查询、分配或锁。长度溢出保护
只签源码检查；构造越过实际存储的 string_view 不是合法运行测试，该夹具已删除并留痕。

原始 Windows、Air/Mini 收据保留各自 commit/tree，不重签成最终候选：Apple 专有修复不改变
Windows 已验路径，GCC 夹具修正不改变产品。验证父工程部署版本和 Air 子进程 loader 环境的
纠正均留存失败日志，已通过项不重跑。复用与差异见 [验证摘要](validation.json) 和
[身份及逐文件摘要](identity.json)，原始命令、日志、cache 与脚本在 raw.zip。

allocator backing 必须活到最后一个身份保留者释放；改变模型/阶段或有关原生环境须重新准备。
未新增跨模型/跨进程缓存。全部自研算子仍冻结，旧 ggml CPU 与 oneDNN BLOCKED 不因本卡解除。
统一 Compute+Graph 实现统计为 **49,804 行、205 个实现文件**（不含测试/第三方），相对 M1-G
新增 391 行冷元数据与接缝；完整统计见 [LOC](implementation-loc.json)。

本轮完成后的具名产物清理见[清理记录](cleanup.json)；缓存、模型与M1-C源码输入保留。
