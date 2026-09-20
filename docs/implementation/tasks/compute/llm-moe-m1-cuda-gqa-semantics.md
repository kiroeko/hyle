# M1-C7：成熟attention接入的GQA语义与旧实现隔离

## Closure card

- owner：`/root`；component：`compute_ops`与`compute_provider_native`；阶段CLOSED。
- baseline：`fb2be1c2`（完整身份由Git解析）；两个owner从M0的SCOPED移交，旧自研数学冻结不变。
- trigger：M1-C实际接入的Qwen head256、Q/KV heads=8/2被
  `src/compute_ops/operation.cc::infer_attention`的同head检查拒绝；FlashInfer原版入口已经实际通过。
- 主纵切：成熟CUDA Provider接收自然SDPA描述的GQA/MQA；正交：原有同head MHA与错误shape拒绝。
- scope：公共shape语义、旧CPU Provider能力边界与回归；没有新attention API、layout或数学实现。
- 预算：计入[M1-C](llm-moe-m1-cuda-admission.md)余下35分钟，不叠加；P0，无P1。
- 依赖：已有TensorDesc/Operation、旧CPU Provider冷prepare；直接消费者为Compute/Graph与新CUDA Provider。
- stop：下列有限改动和适用回归完成，不扩大为旧算法研究或其它语义目录。

## 审计与有限FIX（冻结）

| ID | 能力与当前事实 | 允许改动 | 验收 |
|---|---|---|---|
| S1 | 公共复合语义PARTIAL：现有SDPA只接受同head | Qhead是正数且为正KVhead整数倍；K/V heads一致；保留batch、sequence、depth与mask检查，输出沿Qhead | BSHD/BHSD的8/2、8/1、8/8及不整除/零head/不同KV head |
| S2 | 冻结CPU实现仅同head域，原先依赖S1同head前置条件 | 在native冷prepare分支添加GQA Unsupported，避免公共域放宽后误派发 | GQA拒绝且不执行；原有MHA仍按既有正确性回归通过 |
| S3 | 普通C++语义与发现 | SDPA旁注释、用户guide与受影响语义/Provider测试 | 四环境公共语义、Windows/WSL CPU风险回归；随M1-C公共闭包完成适用三交付 |

原子拆分：S1只确定形状与含义，实际计算仍由准入成熟Provider承担；S2是缺失能力隔离，
不扩旧算子、不研究其性能或改变数学。其它Provider冷注册未声明SDPA，不需要修改其算法。
ownership、concurrency、completion、allocator、输出与mask生命周期均沿既有合同；无新增热路径字段。
分组含义固定为Q head `h`使用KV head `h / (Qheads / KVheads)`；head数必须为正，空sequence
不因此改为非法。BSHD/BHSD按既有layout确定head轴；mask仍面向Qhead的逻辑score shape。
API/usability没有新增对象或重载，默认描述不变。S2只在`native_provider.cc`的冷分支拒绝
非同head，`attention.cc`与实际kernel保持原blob，不引入运行期退回自研的路由。

七问：无新第三方依赖；GQA需求来自已固定模型与原版接口；复用现有schema/错误；P0仅冷检查；
无C/FFI；公开头/普通消费者三交付随M1-C验；guide明确同一SDPA的分组映射。
新增域不代表旧Provider自动支持，也不把形状推导测试当成计算数值通过。

SCOPED/AUDITED元数据只跑docs/closure/diff；产品候选由selector选择compute_ops、native与
真实直接消费者，公共shape回归在四环境执行。最终收据随M1-C保存；未运行不写PASS。

Mac静态回归首次编译发现既有shared-only测试描述符未随其用例条件编译，AppleClang的
`-Werror`报告两个未用常量。仅给该测试描述符加与原用例相同的`!HYLE_STATIC`条件，保留原
警告级别与全部适用用例；产品/API/数学不变。CUDA-OFF三门已通过，GQA门按修正后fixture续跑。

不可变candidate为`43732241adf81ad2bad9f64594e72d752b900b37`；S1–S3产品固定。
Windows/WSL公共语义/旧CPU隔离及已声明host插桩、M1-C自然attention三交付通过；Air四门通过，
Mini待收齐。本卡进入CANDIDATE，尚未关闭，不研究或比较旧attention数学。

## 最终裁决（2026-09-20，CLOSED）

S1–S3完成，candidate仍为`43732241adf81ad2bad9f64594e72d752b900b37`。本节替代前面的进行中
检查点；Windows/WSL/Air/Mini公共形状及旧CPU隔离回归通过，适用WSL host插桩沿不变产品代码
复用；自然GQA在M1-C的Windows/WSL源码/static/shared普通Direct/Graph消费中实际运行。
原始失败、fixture修正与身份见[M1-C证据](../../../../evidence/correctness/compute-llm-moe-m1-cuda-admission/README.md)。

| Gate | 结论与证据 | Verdict |
|---|---|---|
| scope | 只扩共同GQA/MQA形状语义并隔离旧CPU未支持域；S1–S3完成 | PASS |
| architecture | 复用SDPA/TensorDesc/错误；分组映射固定，无新增运行时/第二API | PASS |
| correctness | 四环境正负形状、旧MHA回归与GQA冷拒绝；成熟CUDA 8/2实际数值与资源旅程，适用host插桩 | PASS |
| delivery | 随M1-C闭包的自然attention源码/static/shared真实消费，隐藏producer与搬移prefix | PASS |
| usability_docs | 公开描述与指南说明Q/KV head关系、输出形状、分组和旧CPU域边界 | PASS |
| red_team | 冷检查不进入旧attention数学；空sequence合同不变；Mac静态fixture条件与共享用例一致，未关闭Werror | PASS |
| engineering_performance | P0只有冷shape/支持域验证，无新增submit成本、数学或存储；P1不适用本语义纠正 | PASS |

无本卡阻塞；未来新增形状/语义、实际消费者不相容或正确性反例才重开。全部旧自研数学继续冻结。
统一Compute+Graph统计随M1-C为51,161实现行/217文件，不含测试/第三方；本卡不签模型覆盖。
