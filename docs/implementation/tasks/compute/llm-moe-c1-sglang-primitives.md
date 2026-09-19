# C1-AB：SGLang CUDA 归一化与激活成组接入

## SCOPED / AUDITED

- owner `/root`；baseline `7444ab4346e3b05e66c1839d6c170f3d76751c6f`。
- `compute_provider_cuda` 保持 C1 AUDITED，仅按以下有限 FIX 重新定界。复用已关闭的 CUDA
  Direct/Graph 工厂、backend、Mem 与 capture；不修改它们的产品代码。CPU Q2 BLOCKED 不变。
- 需求：矩阵 F03/F04 的 token/head RMSNorm 与激活；主消费为本地 Qwen3.5-2B 的归一化、
  attention gate 与 FFN；正交消费为批处理行归一化和普通数学激活。不是增加模型 API。
- 来源：锁定 SGLang `06992c92ad4494fe9f9636e75be3e35040512447` 的 `rmsnorm.cuh`、
  `impl/norm.cuh`、`tile.cuh`、`warp.cuh`、`type.cuh`、`activation.cuh` 和 `ops/layernorm/norm.py`。
  另核对锁定 ggml `456172ec733a135778adcd32d00e576a58232e45` 的 CUDA `norm.cu`：其
  LayerNorm 使用 E[x²]−E[x]²，GroupNorm 两遍中心化但仅 F32；不因该接口差异认定无成熟实现。
  本批选择已有 SGLang provider 的同合同域；ggml/cuDNN 其余 CUDA 域仍在 C1 成熟覆盖责任内。

SGLang RMSNorm 支持同类型 F16/BF16 输入和必需 weight；一次 F32 归约/乘法、输出最终 RNE。
上游 warp 宽度为64/128/256，CTA 为256倍数至8192，half-CTA 为512倍数至16384。
本批保留上游 pre-Blackwell 的16字节载入、warp/CTA/double-load 算法与形状选择；不把未经
本机验证的32字节 Blackwell 专用变体签为优选。不窄化 F32 weight 以勉强匹配。
HF 专用 RMSNorm 在乘 weight 前窄化，与当前共同合同不同，不接作普通 RMSNorm。

上游 Unary shell 可承载三种已准入激活，但原公式会丢失大负值尾部、且 -Inf 与普通 Unary
极限值不同。只为新增 Unary 接入必要稳定数学适配；已有 packed gate 的独立 IEEE 公式合同
和既有性能证据保持，不借本批静默改变其特殊值。

| FIX | 固定允许改动 |
|---|---|
| AB-A | 在已登记 `sglang_activation` 固定 pin 下抽取原版 norm/行访存 device body 与 Unary shell，记录原始 blob/hash、具体窄适配和许可；更新同一依赖 tree/NOTICE hash，不升级 pin、不改公共构建工具。 |
| AB-B | 复用现有 norm adapter 接入上述全部同类型半精度/具名宽度；准备时绑定 kernel、occupancy、grid 与16字节要求；普通 route 通过已有 provider 注册取得。原位 input0、错误/保活/空域合同沿共同机制；其他域保留现有候选。 |
| AB-C | 普通 SGLang Unary SiLU/GELU/GELU-tanh，F32/F16/BF16 dense 完整16字节向量、最终一次舍入；用现有稳定 sigmoid、erfc 与 -Inf 极限公式适配数值，保持 NaN 和 signed zero。保持上游向量访存/launch 形状，拒绝不符合域，由已注册 Native 承载。 |
| AB-D | 同合同独立 FP64/位级 oracle、默认 Direct/Graph/原生 capture、两槽/变输入/旧结果/关闭、alias/对齐/配额与 prepare 拒绝；实际本地2B张量片段及 source/static/shared 验证，更新来源和普通用法。 |

## 形状、成本与七问

公开 API 仍为 `RmsNorm`/`Unary` 和同一 Graph 节点；用户选择 CUDA，不排序 Provider。
源码接入为 BORROW；HYLE 只负责独立 C++ 编译、CUDA 基础类型/原语适配和既有资源边界。
无需 TVM、Python runtime、私有 CUDA context、线程、allocator 或模型加载器。
Norm O(N)，每行原版 warp/CTA，持久 kernel/grid 元数据仅在准备时分配；无槽私有 device
storage、caller workspace、转换或额外 kernel。occupancy/设备查询只在准备时，运行不搜索。
Unary O(N)，原版16字节向量、一次 launch、零 workspace；数学变化明确列为适配而非原版。
输入输出的容量、对齐、alias 和 owner/完成检查复用 Compute/backend，不能用省检查取得优势。
普通工厂资格不等于性能赢家；完整 C1 P1 冻结后比较相同合同和原始成熟调用的全成本，决定
固定条件规则和可删除重复域。SIMD 基础库不受影响；不会仅因新入口存在删除现有完整能力。

## 验证、预算与停止

- W1 SM89：受影响 provider、默认 Direct/Graph 测试，static/shared；WSL CUDA 同候选适用
  correctness/portability；公开头、source contract、三种独立 C++ 交付。Apple/Vulkan 产品未变，
  不重复其运行资格。Windows DLL 邻 exe、PATH 隔离、禁止弹窗，重型串行、parallel2。
- 核心形状包含64/128/256/512/768/2048/2304/8192/8704/16384及不匹配尾部、F32权重；
  另用数据驱动遍历所有准入宽度，防止模板分派遗漏。Unary 包含大负尾、±0/Inf/NaN、half RNE。
- 普通 Graph 必須证实成熟 provider identity、可复用 capture 和两个独立状态；不是只跑显式
  provider 探针。本地2B真实 norm/gate 张量通过 source/static/shared，oracle独立、资源归零。
- 依 change-impact selector 选择工程门、保留 skipped reasons；未变公共构建/非 CUDA
  交付只在 tree/依赖/工具链/产物身份可证明一致时复用，适用未运行不写 PASS。
- 源码调查最多60分钟/两个具名来源，到上述差异停止；产品纠正最多两轮，consumer 编译
  纠正最多两轮。不得放宽原数学门，不以静默 fallback 冒充成熟实现；本批不运行 P1。
- 已完成门停止重复；归档后清理本批过期源码/重复 producer，保留活动缓存和必要原始证据。

审计已固定，产品尚未实施、验证尚未运行；不计入完整 C1/矩阵完成。
C1 全部适用验收完成后明确通知用户，在 C2 前等待提问。
