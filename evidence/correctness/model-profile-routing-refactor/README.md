# 真实模型画像路由系统重构：终端证据

## 候选与范围

- baseline：`af8a42e4c`
- 产品候选：`c9c11001419059068e3497b6eac10fcb70cb6e87`
- 验收候选：`2ada22a89821d20b312407c9575df8f9a923df89`。该提交只把 oneDNN
  allocator OOM 测试的预期错误文本迁移为新的结构化 Provider 诊断；产品实现与产品候选相同。
- 本轮替换 Compute/Graph 冷路径路由合同并迁移 CPU、CUDA、Apple、Vulkan 的既有调用者；没有增加
  operator、Provider、kernel 或 shader，没有改变 M3 覆盖表。

## 实现结果

1. `ModelArtifactProfile` 以一个 allocator allocation 拥有固定 model/artifact revision、configuration/
   manifest digest、variant 与 format family；`WorkloadContext` 持有 exact profile 和 versioned region
   semantic。
2. `RoutePolicy` 合取平台、原生硬件、driver/toolkit/runtime、Provider 版本、模型/制品、operation
   semantic，以及任意 input/output port 的 dtype、storage format、rank 与有界/整除维度条件。
3. `ProviderImplementationId`、policy id 与 preference 在 prepare/compile 期固定。Direct 收集 hard-eligible
   proposal 后确定性排序；同 preference 的不同目标明确报 ambiguity；只有声明的 capability/resource
   decline 可尝试下一候选，接纳后的错误不降级。
4. Graph 继续拥有 DAG、canonical boundary、effects/state 和有界 plan search；Provider 可从真实完整 DAG
   检查混合权重、融合片段或完整 region。Graph plan 固定 implementation/policy/preference 后重复执行。
5. 四路线保留原生 Context、资源、程序、Queue/stream、完成与失败 owner。公开 Direct/Graph 示例和
   source/static/shared consumer 已迁移到同一合同。

真实画像纵切覆盖 Qwen3.5-2B 官方 BF16 与 Unsloth UD-Q4_K_XL、Qwen3.8-2.4T mixed
IQ4/Q4/Q5/Q6/F32 MoE，以及 Kimi/DeepSeek mixed MXFP4、BF16/Q8 projection、F32 state/router。
这些测试使用容量缩小但语义、端口和格式真实的结构，不宣称加载或运行完整 checkpoint。

## 验证矩阵

| 环境 | 结果 |
|---|---|
| Windows Compute core | 63/63 PASS |
| Windows CPU backend/Direct/runtime | 54/54 PASS |
| Windows Graph core/builder | 31/31 PASS |
| Windows Graph executable/region | 283/283 PASS |
| Windows CPU route + Graph | 20/20 PASS |
| Windows oneDNN Graph | 20/20 PASS |
| Windows RTX 4070, SM89, CUDA toolkit 13.3.73, driver 610.62 | Direct 8/8、Graph 27/27 PASS |
| Windows RTX 4070 Vulkan | Direct 125/125、Graph 37/37 PASS |
| Windows selected CPU/oneDNN factors | 10/10 PASS |
| Windows selected CUDA factors | 9/9 PASS |
| Windows selected Vulkan factors | 4/4 PASS |
| Windows C++ delivery | Compute 与 Graph 的 source/static/shared relocation consumer 全部 PASS |
| WSL Ubuntu 24.04 GCC 14.2 | 公共头、Core、CPU Direct/Graph 组合 9/9 PASS |
| WSL Ubuntu 24.04 Clang 20.1.2 | 同一组 9/9 PASS |
| MacBook Air `Mac16,12` | Apple Direct 44/44、Graph 17/17；source/static/shared Compute+Graph consumer 全部 PASS |
| Mac mini `Mac16,10` | Apple portability 头文件/Direct/Graph 4/4 PASS |

`validation-select.py plan --base af8a42e4c --candidate HEAD` 在 canonical Windows 工作区返回
`READY`。其选中的 headers、CPU/oneDNN、CUDA、Vulkan、Compute/Graph delivery 风险已映射到上表的
具名路线 build 并执行；没有把未运行的 generic runner 写成 PASS。P1 未被 selector 选择，也不适用于
本轮只发生在 prepare/compile 的路由变更。

验证中发现的两处失败均已定界：一处临时 Apple Graph consumer 脚本没有保留
`samples/package/{compute,graph}` 的相对布局，修正 staging 后三交付通过；一处 oneDNN OOM 测试仍期待
旧的裸错误文本，迁移为 Provider 前缀诊断后受影响用例及完整 factor 通过。两者都没有引出产品数学或
执行修复。

## P0 成本账单

- `match_route_selector`/`match_route_policy` 的产品调用只存在于 Provider construction、propose、prepare
  与 Graph compile 冷路径。
- Direct `OperationInvocation::submit` 只读取已绑定的 prepared state/provider callback 并校验 bindings；
  Graph `Invocation::submit` 只执行已编译 plan、prepared replay/dependency route 与 segment callback。
  两条热路径都不解释 selector/policy，不查询 profile/设备，不搜索、计时或重新选择 Provider。
- profile 创建进行一次与身份字符串总长成正比的 allocation、校验和复制；复制 handle 只增加引用计数。
  冷 matcher 的上界为 policy entry 数量乘该 entry 声明的 facts/ports/dimension predicates。
- prepared/report metadata 增加 implementation identity、policy identity 和 preference；这是按 artifact/plan
  支付的冷存储，没有逐 submit 分配、锁、设备查询或路由工作。因此 P0 PASS，P1 不适用。

## 红队与边界

- 产品目录对 `RouteRule`、`ProviderRouteKey`、`profile_key`、旧 `ModelIdentity/ModelDescription` 的扫描无
  命中；旧双轨 API 已删除。
- 所有 matcher 调用点均在冷 construction/propose/prepare/compile 路径；submit 没有回连路由器。
- profile exact/unknown/revision/manifest mismatch，任意端口缺失/重复、dtype/format/rank/shape/mode/group
  mismatch，同 preference ambiguity，capability/resource decline 与 fatal-stop 均有定向测试。
- 本轮不提供模型加载、tokenizer、采样或推理 engine，不新增数学覆盖，不实现模型级 CPU+GPU 联合推理、
  HIP、分布式或跨进程 plan cache。M3 的 operator/provider coverage 仍按其独立任务继续。

## 统一 LOC

固定候选的统一脚本结果见 [terminal-loc.json](terminal-loc.json)：Compute+Graph 实现 **60,422** 行、
**252** 个文件；全库自实现且不含测试 **155,607** 行、**603** 个文件；全库代码 **1,115,613** 行。
统计排除 tests、bench、docs、third_party、CMake/build/package scripts 和临时目录，自检通过。
