# M1-C7：成熟attention接入的GQA语义与旧实现隔离

## Closure card

- owner：`/root`；component：`compute_ops`与`compute_provider_native`；阶段AUDITED。
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
